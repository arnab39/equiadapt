from typing import List, Optional, Tuple

import torch
from torch import nn


class MLP(nn.Module):
    """a simple 4-layer MLP"""

    def __init__(self, nin: int, nout: int, nh: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Defines a parent class for all the following GNN layers.
class GCL_basic(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          `hidden_dim`: Number of hidden units.
          `num_nodes`: Maximum number of nodes (for self-attentive pooling).
          `global_agg`: Global aggregation function ('attn' or 'sum').
          `temp`: Softmax temperature.
    """

    def __init__(self) -> None:
        super().__init__()

    def edge_model(
        self, source: torch.Tensor, target: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        pass

    def node_model(
        self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        pass

    def forward(
        self,
        x: torch.Tensor,
        edge_index: List[torch.Tensor],
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on equation (2) in https://arxiv.org/pdf/2102.09844.pdf.
        A matrix of edge features is created and used to update node features (m and h in paper).

        Args:
            `x`: Matrix of node embeddings. Shape: (n_nodes * batch_size) x hidden_dim
            `edge_index`: Length 2 list of tensors, containing indices of adjacent nodes; each shape (n_edges * batch_size).
        """
        row, col = edge_index
        # phi_e in the paper. returns a matrix m of edge features used to update node and feature embeddings.
        # x[row], x[col] are embeddings of adjacent nodes
        # edge_attr = edge attributes/features
        edge_feat = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(
            x, edge_index, edge_feat
        )  # updates node embeddings (phi_h in the paper)
        return x, edge_feat


# Graph convolutional layer
# Based on equation (2) from https://arxiv.org/pdf/2102.09844.pdf
class GCL(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.

    Args:
          `hidden_dim`: Number of hidden units.
          `num_nodes`: Maximum number of nodes (for self-attentive pooling).
          `global_agg`: Global aggregation function ('attn' or 'sum').
          `temp`: Softmax temperature.
    """

    def __init__(
        self,
        input_nf: int,
        output_nf: int,
        hidden_dim: int,
        edges_in_nf: int = 0,
        act_fn: nn.Module = nn.ReLU(),
        bias: bool = True,
        attention: bool = False,
        t_eq: bool = False,
        recurrent: bool = True,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.t_eq = t_eq
        self.recurrent = recurrent
        input_edge_nf = input_nf * 2  # because we concatenate
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_dim, bias=bias),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim, bias=bias),
            act_fn,
        )
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_dim, bias=bias),
                act_fn,
                nn.Linear(hidden_dim, 1, bias=bias),
                nn.Sigmoid(),
            )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + input_nf, hidden_dim, bias=bias),
            act_fn,
            nn.Linear(hidden_dim, output_nf, bias=bias),
        )

        # if recurrent:
        # self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def edge_model(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns matrix m from eqn. (2) in paper, of shape (batch_size * n_edges) x hidden_dim.

        Args:
            `source`: Embeddings of nodes start of edge. Shape: (batch_size * n_edges) x input_nf
            `target`: Embeddings of nodes at end of edge. Shape: (batch_size * n_edges) x input_nf
            `edge_attr`: Attributes of edges. Shape: (batch_size * n_edges) x edge_attr_dim
        """
        edge_in = torch.cat(
            [source, target], dim=1
        )  # (batch_size * n_edges) x (input_edge_nf)
        if edge_attr is not None:
            edge_in = torch.cat(
                [edge_in, edge_attr], dim=1
            )  # (batch_size * n_edges) x (input_edge_nf + edges_in_nf)
        out = self.edge_mlp(
            edge_in
        )  # m from paper, (batch_size * n_edges) x hidden_dim
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        return out  # (batch_size * n_edges) x hidden_dim

    def node_model(
        self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns updated node embeddings, h, from paper. Shape: (n_nodes * batch_size) x output_nf

        Args:
            `h`: current node embeddings. Shape: (n_nodes * batch_size) x input_nf
            `edge_index`: Indices of adjacent nodes. Shape: (n_edges * batch_size) x 2
            `edge_attr`: Attributes of edges. Shape: (batch_size * n_edges) x hidden_dim (this is the output of edge_model)
        """
        row, col = edge_index
        # m_i from paper, where m__i = sum of edge attributes for edges adjacent to i (n_nodes x edge_attr_dim)
        agg = unsorted_segment_sum(
            data=edge_attr, segment_ids=row, num_segments=h.size(0)
        )
        out = torch.cat([h, agg], dim=1)
        out = self.node_mlp(
            out
        )  # phi_h from the paper. Shape: (n_nodes * batch_size) x output_nf
        if self.recurrent:
            out = out + h
            # out = self.gru(out, h)
        return out  # Shape: (n_nodes * batch_size) x output_nf


class GCL_rf(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          `hidden_dim`: Number of hidden units.
          `num_nodes`: Maximum number of nodes (for self-attentive pooling).
          `global_agg`: Global aggregation function ('attn' or 'sum').
          `temp`: Softmax temperature.
    """

    def __init__(
        self,
        nf: int = 64,
        edge_attr_nf: int = 0,
        reg: float = 0,
        act_fn: nn.Module = nn.LeakyReLU(0.2),
        clamp: bool = False,
    ) -> None:
        super().__init__()

        self.clamp = clamp
        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(edge_attr_nf + 1, nf), act_fn, layer)
        self.reg = reg

    def edge_model(
        self, source: torch.Tensor, target: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff**2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        if self.clamp:
            m_ij = torch.clamp(m_ij, min=-100, max=100)
        return m_ij

    def node_model(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        row, col = edge_index
        agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))
        x_out = x + agg - x * self.reg
        return x_out


def unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    result_shape = (num_segments, data.size(1), data.size(2))
    segment_ids = segment_ids.unsqueeze(-1).unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, data.size(1), data.size(2))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
