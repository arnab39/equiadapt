import math

import torch
import torch.nn as nn

from examples.nbody.networks.gcl import GCL


# This model is the parent of all the following models in this file.
class BaseEuclideangraphModel(nn.Module):
    def __init__(
        self, hyperparams, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.learning_rate = (
            hyperparams.learning_rate if hasattr(hyperparams, "learning_rate") else None
        )
        self.weight_decay = (
            hyperparams.weight_decay if hasattr(hyperparams, "weight_decay") else 0.0
        )
        self.patience = (
            hyperparams.patience if hasattr(hyperparams, "patience") else 100
        )


# Model based on https://arxiv.org/pdf/2102.09844.pdf, equations 3-6.
class GNN(BaseEuclideangraphModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.model = "GNN"
        self.hidden_dim = hyperparams.hidden_dim
        self.input_dim = hyperparams.input_dim
        self.n_layers = hyperparams.num_layers
        self.act_fn = nn.SiLU()
        self.attention = 0
        self.recurrent = True
        ### Encoder
        # self.add_module("gcl_0", GCL(self.hidden_dim, self.hidden_dim, self.hidden_dim, edges_in_nf=1, act_fn=act_fn, attention=attention, recurrent=recurrent))
        for i in range(0, self.n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    input_nf=self.hidden_dim,
                    output_nf=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    edges_in_nf=2,
                    act_fn=self.act_fn,
                    attention=self.attention,
                    recurrent=self.recurrent,
                ),
            )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.act_fn,
            nn.Linear(self.hidden_dim, 3),
        )
        self.embedding = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim))

    def forward(self, nodes, loc, edges, vel, edge_attr, _):
        """
        Returns: Node coordinate embeddings
        Args:
            `nodes`: Norms of velocity vectors. Shape: (n_nodes * batch_size) x 1
            `loc`: Coordinates of nodes. Shape: (n_nodes * batch_size) x coord_dim
            `edges`: Length 2 list of vertices, where edges[0][i] is adjacent to edges[1][i].
            `vel`: Velocities of nodes. Shape: (n_nodes * batch_size) x vel_dim
            `edge_attr`: Products of charges along edges. batch_size x n_edges x 1
        """
        # TODO: loc currently have the wrong shape...
        nodes = torch.cat(
            [loc, vel], dim=1
        )  # (n_nodes * batch_size) x (coord_dim + vel_dim)
        h = self.embedding(nodes)  # (n_nodes * batch_size) x hidden_dim
        # h, _ = self._modules["gcl_0"](h, edges, edge_attr=edge_attr)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
        # h is 500x32 and then passed to decoder to become 500x3
        # return h
        return self.decoder(h)  # (n_nodes * batch_size) x 3


class Transformer(BaseEuclideangraphModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        print(hyperparams)
        self.model = "Transformer"
        self.hidden_dim = hyperparams.hidden_dim  # 32
        self.input_dim = hyperparams.input_dim  # 6
        self.n_layers = hyperparams.num_layers  # 4
        self.ff_hidden = hyperparams.ff_hidden
        self.act_fn = nn.ReLU()
        self.dropout = 0
        self.nhead = hyperparams.nheads

        self.pos_encoder = PositionalEncoding(
            hidden_dim=self.hidden_dim, dropout=self.dropout
        )

        self.charge_embedding = nn.Embedding(2, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=7 * self.hidden_dim,
            nhead=self.nhead,
            dim_feedforward=self.ff_hidden,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.n_layers
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=7 * self.hidden_dim, out_features=7 * self.hidden_dim
            ),
            self.act_fn,
            nn.Linear(in_features=7 * self.hidden_dim, out_features=3),
        )

    def forward(self, nodes, loc, edges, vel, edge_attr, charges):
        """
        Forward pass through Transformer model

        Args:
            `nodes`: Norms of velocity vectors. Shape: (n_nodes*batch_size) x 1
            `loc`: Starting locations of nodes. Shape: (n_nodes*batch_size) x 3
            `edges`: list of length 2, where each element is a 2000 dimensional tensor
            `vel`: Starting velocities of nodes. Shape: (n_nodes*batch_size) x 3
            `edge_attr`: Products of charges and squared relative distances between adjacent nodes (each have their own column). Shape: (n_edges*batch_size) x 2
            `charges`: Charges of nodes . Shape: (n_nodes * batch_size) x 1
        """
        # Positional encodings
        pos_encodings = torch.cat([loc, vel], dim=1).unsqueeze(
            2
        )  # n_nodes*batch x 6 x 1
        pos_encodings = self.pos_encoder(
            pos_encodings
        )  # n_nodes*batch x 6 x hidden_dim
        # Charge embeddings
        charges[charges == -1] = 0  # to work with nn.Embedding
        charges = charges.long()
        charges = self.charge_embedding(charges)  # n_nodes*batch x 1 x hidden_dim
        nodes = torch.cat(
            [pos_encodings, charges], dim=1
        )  # n_nodes * batch_size x 7 x hidden_dim
        nodes = nodes.view(
            -1, 5, nodes.shape[1] * nodes.shape[2]
        )  # batch_size x n_nodes x (7 * hidden_dim)
        h = self.encoder(nodes)  # batch_size x n_nodes x (7 * hidden_dim)
        h = h.view(-1, h.shape[2])
        h = self.decoder(h)
        return h


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = hidden_dim
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        ).view(
            1, 1, int(hidden_dim / 2)
        )  # 1 x 1 x (hidden_dim / 2)
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Returns positional encoding of coordinates and velocities.
        Args:
            `x`: Concatenated velocity and coordinate vectors. Shape: (n_nodes * batch_size x 6 x 1)
        """
        pe = torch.zeros(x.shape[0], x.shape[1], self.hidden_dim).to(
            x.device
        )  # (n_nodes * batch_size) x 6 x 32
        sin_terms = torch.sin(x * self.div_term)
        pe[:, :, 0::2] = sin_terms
        pe[:, :, 1::2] = torch.cos(x * self.div_term)
        return self.dropout(pe)
