from torch import nn
import torch
import torch.nn.functional as F
import torch_scatter as ts


class MLP(nn.Module):
    """a simple 4-layer MLP"""

    def __init__(self, nin, nout, nh):
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

    def forward(self, x):
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

    def __init__(self):
        super(GCL_basic, self).__init__()

    def edge_model(self, source, target, edge_attr):
        pass

    def node_model(self, h, edge_index, edge_attr):
        pass

    def forward(self, x, edge_index, edge_attr=None):
        
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
        x = self.node_model(x, edge_index, edge_feat) # updates node embeddings (phi_h in the paper)
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
        input_nf,
        output_nf,
        hidden_dim,
        edges_in_nf=0,
        act_fn=nn.ReLU(),
        bias=True,
        attention=False,
        t_eq=False,
        recurrent=True,
    ):
        super(GCL, self).__init__()
        self.attention = attention
        self.t_eq = t_eq
        self.recurrent = recurrent
        input_edge_nf = input_nf * 2 # because we concatenate 
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_dim, bias=bias),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim, bias=bias),
            act_fn,
        )
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_dim, bias=bias), act_fn, nn.Linear(hidden_dim, 1, bias=bias), nn.Sigmoid()
            )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + input_nf, hidden_dim, bias=bias), act_fn, nn.Linear(hidden_dim, output_nf, bias=bias)
        )

        # if recurrent:
        # self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def edge_model(self, source, target, edge_attr):
        """
        Returns matrix m from eqn. (2) in paper, of shape (batch_size * n_edges) x hidden_dim.

        Args:
            `source`: Embeddings of nodes start of edge. Shape: (batch_size * n_edges) x input_nf
            `target`: Embeddings of nodes at end of edge. Shape: (batch_size * n_edges) x input_nf
            `edge_attr`: Attributes of edges. Shape: (batch_size * n_edges) x edge_attr_dim
        """
        edge_in = torch.cat([source, target], dim=1) # (batch_size * n_edges) x (input_edge_nf)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1) # (batch_size * n_edges) x (input_edge_nf + edges_in_nf)
        out = self.edge_mlp(edge_in) # m from paper, (batch_size * n_edges) x hidden_dim
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        return out #(batch_size * n_edges) x hidden_dim

    def node_model(self, h, edge_index, edge_attr):
        """
        Returns updated node embeddings, h, from paper. Shape: (n_nodes * batch_size) x output_nf

        Args:
            `h`: current node embeddings. Shape: (n_nodes * batch_size) x input_nf
            `edge_index`: Indices of adjacent nodes. Shape: (n_edges * batch_size) x 2
            `edge_attr`: Attributes of edges. Shape: (batch_size * n_edges) x hidden_dim (this is the output of edge_model)
        """
        row, col = edge_index
        # m_i from paper, where m__i = sum of edge attributes for edges adjacent to i (n_nodes x edge_attr_dim)
        agg = unsorted_segment_sum(data=edge_attr, segment_ids=row, num_segments=h.size(0))
        out = torch.cat([h, agg], dim=1) 
        out = self.node_mlp(out) # phi_h from the paper. Shape: (n_nodes * batch_size) x output_nf
        if self.recurrent:
            out = out + h
            # out = self.gru(out, h)
        return out #Shape: (n_nodes * batch_size) x output_nf

class GCL_rf(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          `hidden_dim`: Number of hidden units.
          `num_nodes`: Maximum number of nodes (for self-attentive pooling).
          `global_agg`: Global aggregation function ('attn' or 'sum').
          `temp`: Softmax temperature.
    """

    def __init__(self, nf=64, edge_attr_nf=0, reg=0, act_fn=nn.LeakyReLU(0.2), clamp=False):
        super(GCL_rf, self).__init__()

        self.clamp = clamp
        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(edge_attr_nf + 1, nf), act_fn, layer)
        self.reg = reg

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        if self.clamp:
            m_ij = torch.clamp(m_ij, min=-100, max=100)
        return m_ij

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))
        x_out = x + agg - x * self.reg
        return x_out


# Equivariant graph convolutional layer
# Based on equations (3) - (6) from https://arxiv.org/pdf/2102.09844.pdf 
class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          `hidden_dim`: Number of hidden units.
          `num_nodes`: Maximum number of nodes (for self-attentive pooling).
          `global_agg`: Global aggregation function ('attn' or 'sum').
          `temp`: Softmax temperature.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_dim,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.ReLU(),
        recurrent=True,
        coords_weight=1.0,
        attention=False,
        clamp=False,
        norm_diff=False,
        tanh=False,
        num_vectors_in=1,
        num_vectors_out=1,
        last_layer=False,
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.num_vectors_in = num_vectors_in
        self.num_vectors_out = num_vectors_out
        self.last_layer = last_layer
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + num_vectors_in + edges_in_d, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + input_nf + nodes_att_dim, hidden_dim), act_fn, nn.Linear(hidden_dim, output_nf)
        )

        layer = nn.Linear(hidden_dim, num_vectors_in * num_vectors_out, bias=False) # outputs a scalar
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_dim, hidden_dim))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        # if recurrent:
        #    self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def edge_model(self, source, target, radial, edge_attr):
        """
        Returns matrix m from eqn. (3) from paper, of shape (batch_size * n_edges) x hidden_dim.

        Args:
            `source`: Embeddings of nodes start of edge. Shape: (batch_size * n_edges) x input_nf
            `target`: Embeddings of nodes at end of edge. Shape: (batch_size * n_edges) x input_nf
            `radial`: Squared distances between coordinates of adjacent nodes. Shape: (n_edges * batch_size) x 1
            `edge_attr`: Attributes of edges. Shape: (batch_size * n_edges) x edge_attr_dim
        """
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1) # concatenates inputs to be passed into phi_e
        out = self.edge_mlp(out) # phi_e from eqn. (3). Shape: (n_nodes * batch_size) x hidden_dim
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out #Shape: (n_nodes * batch_size) x hidden_dim

    def node_model(self, h, edge_index, edge_attr, node_attr):
        """
        Returns tuple containing updated node embeddings, h, from eqn. (6). and m_i from eqn. (5).
        Shape: ((n_nodes * batch_size) x output_nf, (n_nodes * batch_size) x (2*hidden_dim))

        Args:
            `h`: Node feature embeddings. Shape: (n_nodes * batch_size) x input_nf
            `edge_index`: Indices of adjacent nodes. Shape: (n_edges * batch_size) x 2
            `edge_attr`: Attributes of edges. Matrix m from eqn. (3). Shape: (n_edges * batch_size) x hidden_dim (this is the output of edge_model)
            `node_attr`: Node coordinate embeddings. Shape: (n_nodes * batch_size) x coord_dim
        """
        row, col = edge_index # Indices of adjacent nodes
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0)) # (n_nodes * batch_size) x hidden_dim. m_i from paper.
        if node_attr is not None:
            agg = torch.cat([h, agg, node_attr], dim=1)
        else:
            agg = torch.cat([h, agg], dim=1) # concatenate inputs for phi_h. (n_nodes * batch_size) x (2*hidden_dim)
        # phi_h from eqn. (6). Updates node feature embeddings. Shape: (n_nodes * batch_size) x output_nf
        out = self.node_mlp(agg) 
        if self.recurrent:
            out = h + out
        return out, agg # Shape: ((n_nodes * batch_size) x output_nf, (n_nodes * batch_size) x (2*hidden_dim))

    def coord_model(self, coord, edge_index, coord_diff, radial, edge_feat):
        """
        Returns updated coordinate embeddings from eqn. (4). Shape: n_nodes * batch_size x 3 x 1

        Args:
            `coord`: Coordinates of nodes. Shape: (batch_size * n_nodes) x coord_dim
            `edge_index`: Indices of adjacent nodes. Shape: (n_edges * batch_size) x 2
            `coord_diff`: Differences between coords of adjacent nodes. Shape: (batch_size * n_edges) x coord_dim
            `radial`: Squared distances of coords of adjacent nodes. Shape: (n_edges * batch_size) x 1
            `edge_feat`: Matrix m from eqn. (3). (n_edges * batch_size) x hidden_dim
        """
        row, col = edge_index # indices of adjacent nodes
        # Eqn. (4) phi_x(m_ij). Shape: (n_edges * batch_size) x num_vectors_in x num_vectors_out 
        coord_matrix = self.coord_mlp(edge_feat).view(-1, self.num_vectors_in, self.num_vectors_out) 
        if coord_diff.dim() == 2:
            coord_diff = coord_diff.unsqueeze(2)
            coord = coord.unsqueeze(2).repeat(1, 1, self.num_vectors_out) # n_nodes * batch_size x coord_dim x num_vectors_out
        # coord_diff = coord_diff / radial.unsqueeze(1)
        # 
        trans = torch.einsum("bij,bci->bcj", coord_matrix, coord_diff)  # (n_edges * batch_size) x coord_dim x 1
        trans = torch.clamp(
            trans, min=-100, max=100
        )  # This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0)) # n_nodes * batch_size x coord_dim x 1. sum from eqn. (4)
        if self.last_layer:
            coord = coord.mean(dim=2, keepdim=True) + agg * self.coords_weight
        else:
            coord += agg * self.coords_weight # Update coordinate embeddings following eqn. (4)
        return coord # 

    def coord2radial(self, edge_index, coord):
        """
        Returns a tuple of differences and squared differences of coordinates adjacent vertices.
        ((n_edges * batch_size) x 1, (batch_size * n_edges) x coord_dim)

        Args:
            `edge_attr`: Attributes of edges. Shape: (batch_size * n_edges) x hidden_dim (this is the output of edge_model)
            `coord`: Coordinates of nodes. Shape: (batch_size * n_nodes) x coord_dim
        """
        row, col = edge_index # indices of adjacent nodes.
        coord_diff = coord[row] - coord[col] # differences between cords of adjacent nodes. Shape: (batch_size * n_edges) x coord_dim
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1) # squared distances. Shape: (n_edges * batch_size) x 1

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / (norm)

        if radial.dim() == 3:
            radial = radial.squeeze(1)

        return radial, coord_diff # returns squared dists and diffs

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        """
        Based on equations (3)-(6) in https://arxiv.org/pdf/2102.09844.pdf.
        Updates node feature and coordinate embeddings.
         
        Args:
            `h`: Node feature embeddings. Shape: (n_nodes * batch_size) x hidden_dim
            `edge_index`: Indices of adjacent nodes. Shape: (n_edges * batch_size) x 2
            `coord`: Node coordinates. Shape: (n_nodes * batch_size) x coord_dim
        """
        row, col = edge_index #indices of adjacent nodes
        # squared dists and diffs. (n_edges * batch_size) x 1, (batch_size * n_edges) x coord_dim
        radial, coord_diff = self.coord2radial(edge_index, coord) 
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr) #Shape: (n_edges * batch_size) x hidden_dim
        coord = self.coord_model(coord, edge_index, coord_diff, radial, edge_feat) # Updated coord embeddings from eqn. 4. (n_nodes * batch_size) x coord_dim x 1
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr

# Based on section 3.2 in https://arxiv.org/pdf/2102.09844.pdf. 
class E_GCL_vel(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_dim,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.ReLU(),
        recurrent=True,
        coords_weight=1.0,
        attention=False,
        norm_diff=False,
        tanh=False,
        num_vectors_in=1,
        num_vectors_out=1,
        last_layer=False,
    ):
        E_GCL.__init__(
            self,
            input_nf,
            output_nf,
            hidden_dim,
            edges_in_d=edges_in_d,
            nodes_att_dim=nodes_att_dim,
            act_fn=act_fn,
            recurrent=recurrent,
            coords_weight=coords_weight,
            attention=attention,
            norm_diff=norm_diff,
            tanh=tanh,
            num_vectors_in=num_vectors_in,
            num_vectors_out=num_vectors_out,
            last_layer=last_layer,
        )
        self.norm_diff = norm_diff
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_dim), act_fn, nn.Linear(hidden_dim, num_vectors_in * num_vectors_out)
        )

    def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
        """
        Based on section 3.2 in https://arxiv.org/pdf/2102.09844.pdf.
        Updates node feature, coordinate, and velocity embeddings.
         
        Args:
            `h`: Node feature embeddings. Shape: (n_nodes * batch_size) x hidden_dim
            `edge_index`: Indices of adjacent nodes. Shape: (n_edges * batch_size) x 2
            `coord`: Node coordinates. Shape: (n_nodes * batch_size) x coord_dim
            `vel`: Node velocities. Shape: (n_nodes * batch_size) x vel_dim
        """
        row, col = edge_index #Indices of adjacent nodes
        # squared dists and diffs. (n_edges * batch_size) x 1, (batch_size * n_edges) x coord_dim
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr) #Shape: (n_edges * batch_size) x hidden_dim
        coord = self.coord_model(coord, edge_index, coord_diff, radial, edge_feat) # Updated coord embeddings from eqn. 4. (n_nodes * batch_size) x coord_dim x 1
        # phi_v from eqn. 7. Shape: (n_nodes * batch_size) x num_vectors_in * num_vectors_out
        coord_vel_matrix = self.coord_mlp_vel(h).view(-1, self.num_vectors_in, self.num_vectors_out) 
        if vel.dim() == 2:
            vel = vel.unsqueeze(2)
        coord += torch.einsum("bij,bci->bcj", coord_vel_matrix, vel) # eqn. (7)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr) # updates node embeddings
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr


class GCL_rf_vel(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, nf=64, edge_attr_nf=0, act_fn=nn.LeakyReLU(0.2), coords_weight=1.0):
        super(GCL_rf_vel, self).__init__()
        self.coords_weight = coords_weight
        self.coord_mlp_vel = nn.Sequential(nn.Linear(1, nf), act_fn, nn.Linear(nf, 1))

        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        # layer.weight.uniform_(-0.1, 0.1)
        self.phi = nn.Sequential(
            nn.Linear(1 + edge_attr_nf, nf), act_fn, layer, nn.Tanh()
        )  # we had to add the tanh to keep this method stable

    def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
        row, col = edge_index
        edge_m = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_m)
        x += vel * self.coord_mlp_vel(vel_norm)
        return x, edge_attr

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        return m_ij

    def node_model(self, x, edge_index, edge_m):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_m, row, num_segments=x.size(0))
        x_out = x + agg * self.coords_weight
        return x_out


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1)) 
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1), data.size(2))
    segment_ids = segment_ids.unsqueeze(-1).unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, data.size(1), data.size(2))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
