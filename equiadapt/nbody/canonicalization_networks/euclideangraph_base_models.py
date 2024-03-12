import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import torch_scatter as ts
import math
from equiadapt.nbody.canonicalization_networks.gcl import E_GCL_vel, GCL
from equiadapt.nbody.canonicalization_networks.vn_layers import VNLeakyReLU, VNSoftplus
from equiadapt.nbody.canonicalization_networks.set_base_models import SequentialMultiple


# This model is the parent of all the following models in this file.
class BaseEuclideangraphModel(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.learning_rate = (
            hyperparams.learning_rate if hasattr(hyperparams, "learning_rate") else None
        )
        self.weight_decay = (
            hyperparams.weight_decay if hasattr(hyperparams, "weight_decay") else 0.0
        )
        self.patience = (
            hyperparams.patience if hasattr(hyperparams, "patience") else 100
        )
        # Each input has 5 particles. This list defines all the edges, since our graph is fully connected.
        # vertex at self.edges[0][i] has an edge connecting to self.edges[1][i]
        self.edges = [
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
            [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3],
        ]

        self.loss = nn.MSELoss()

        self.dummy_nodes = torch.zeros(2, 1, device=self.device, dtype=torch.float)
        self.dummy_loc = torch.zeros(2, 3, device=self.device, dtype=torch.float)
        self.dummy_edges = [
            torch.zeros(40, device=self.device, dtype=torch.long),
            torch.zeros(40, device=self.device, dtype=torch.long),
        ]
        self.dummy_vel = torch.zeros(2, 3, device=self.device, dtype=torch.float)
        self.dummy_edge_attr = torch.zeros(40, 2, device=self.device, dtype=torch.float)

    def training_step(self, batch, batch_idx):
        """
        Performs one training step.

        Args:
            `batch`: a list of tensors [loc, vel, edge_attr, charges, loc_end]
            `loc`: batch_size x n_nodes x 3
            `vel`: batch_size x n_nodes x 3
            `edge_attr`: batch_size x n_edges x 1
            `charges`: batch_size x n_nodes x 1
            `loc_end`: batch_size x n_nodes x 3
            `batch_idx`: index of the batch
        """

        batch_size, n_nodes, _ = batch[0].size()
        batch = [d.view(-1, d.size(2)) for d in batch]  # converts to 2D matrices
        loc, vel, edge_attr, charges, loc_end = batch
        edges = self.get_edges(
            batch_size, n_nodes
        )  # returns a list of two tensors, each of size num_edges * batch_size (where num_edges is always 20, since G = K5)

        nodes = (
            torch.sqrt(torch.sum(vel**2, dim=1)).unsqueeze(1).detach()
        )  # norm of velocity vectors
        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(
            1
        )  # relative distances among locations
        edge_attr = torch.cat(
            [edge_attr, loc_dist], 1
        ).detach()  # concatenate all edge properties

        outputs = self(
            nodes, loc.detach(), edges, vel, edge_attr, charges
        )  # self takes a step.

        # outputs and loc_end are both (5*batch_size)x3
        loss = self.loss(outputs, loc_end)

        metrics = {"train/loss": loss}
        self.log_dict(metrics, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs one validation step.

        Args:
        Args:
            `batch`: a list of tensors [loc, vel, edge_attr, charges, loc_end]
            `loc`: batch_size x n_nodes x 3
            `vel`: batch_size x n_nodes x 3
            `edge_attr`: batch_size x n_edges x 1
            `charges`: batch_size x n_nodes x 1
            `loc_end`: batch_size x n_nodes x 3
            `batch_idx`: index of the batch
        """
        batch_size, n_nodes, _ = batch[0].size()
        batch = [d.view(-1, d.size(2)) for d in batch]
        loc, vel, edge_attr, charges, loc_end = batch
        edges = self.get_edges(batch_size, n_nodes)

        nodes = torch.sqrt(torch.sum(vel**2, dim=1)).unsqueeze(1).detach()
        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(
            1
        )  # relative distances among locations
        edge_attr = torch.cat(
            [edge_attr, loc_dist], 1
        ).detach()  # concatenate all edge properties

        outputs = self(nodes, loc.detach(), edges, vel, edge_attr, charges)

        loss = self.loss(outputs, loc_end)
        if self.global_step == 0:
            wandb.define_metric("valid/loss", summary="min")

        metrics = {"valid/loss": loss}
        self.log_dict(metrics, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-12
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.patience, factor=0.5, min_lr=1e-6, mode="max"
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid/loss",
        }

    def validation_epoch_end(self, validation_step_outputs):
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.optimizer.param_groups[0]["lr"])


    def get_edges(self, batch_size, n_nodes):
        """
        Returns a length 2 list of vertices, where edges[0][i] is adjacent to edges[1][i]

        Args:
            `batch_size`: int, defined in `train_nbody.HYPERPARAMS`
            `n_nodes`: number of nodes in each sample.
        """
        edges = [
            torch.LongTensor(self.edges[0]).to(self.device),
            torch.LongTensor(self.edges[1]).to(self.device),
        ]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            # Adds 5i to the vertices in each sample, allowing us to use rows and cols for indexing our data.
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges


# Based on https://arxiv.org/pdf/2102.09844.pdf equation 7
class EGNN_vel(BaseEuclideangraphModel):
    def __init__(self, hyperparams):
        super(EGNN_vel, self).__init__(hyperparams)
        self.model = "EGNN"
        self.hidden_dim = hyperparams.hidden_dim
        self.in_node_nf = hyperparams.in_node_nf
        self.n_layers = 4
        self.act_fn = nn.SiLU()
        self.coords_weight = 1.0
        self.recurrent = True
        self.norm_diff = False
        self.tanh = False
        self.num_vectors = 1

        # self.reg = reg
        ### Encoder
        # self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_dim, self.hidden_dim, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(hyperparams.in_node_nf, self.hidden_dim)
        self.add_module(
            "gcl_%d" % 0,
            E_GCL_vel(
                input_nf=self.hidden_dim,
                output_nf=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                edges_in_d=hyperparams.in_edge_nf,
                act_fn=self.act_fn,
                coords_weight=self.coords_weight,
                recurrent=self.recurrent,
                norm_diff=self.norm_diff,
                tanh=self.tanh,
                num_vectors_out=self.num_vectors,
            ),
        )
        for i in range(1, self.n_layers - 1):
            self.add_module(
                "gcl_%d" % i,
                E_GCL_vel(
                    input_nf=self.hidden_dim,
                    output_nf=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    edges_in_d=hyperparams.in_edge_nf,
                    act_fn=self.act_fn,
                    coords_weight=self.coords_weight,
                    recurrent=self.recurrent,
                    norm_diff=self.norm_diff,
                    tanh=self.tanh,
                    num_vectors_in=self.num_vectors,
                    num_vectors_out=self.num_vectors,
                ),
            )
        self.add_module(
            "gcl_%d" % (self.n_layers - 1),
            E_GCL_vel(
                input_nf=self.hidden_dim,
                output_nf=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                edges_in_d=hyperparams.in_edge_nf,
                act_fn=self.act_fn,
                coords_weight=self.coords_weight,
                recurrent=self.recurrent,
                norm_diff=self.norm_diff,
                tanh=self.tanh,
                num_vectors_in=self.num_vectors,
                last_layer=True,
            ),
        )

    def forward(self, h, x, edges, vel, edge_attr, _):
        """
        Returns: Node coordinate embeddings
        Args:
            `h`: Norms of velocity vectors. Shape: (n_nodes * batch_size) x 1
            `x`: Coordinates of nodes. Shape: (n_nodes * batch_size) x coord_dim
            `edges`: Length 2 list of vertices, where edges[0][i] is adjacent to edges[1][i].
            `vel`: Velocities of nodes. Shape: (n_nodes * batch_size) x vel_dim
            `edge_attr`: Products of charges along edges. batch_size x n_edges x 1
        """
        h = self.embedding(h)  # Node embeddings. (n_nodes * batch_size) x hidden_dim
        # Applies each layer of EGNN
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, vel, edge_attr=edge_attr)
        return x.squeeze(2)  # Predicted coordinates


# Model based on https://arxiv.org/pdf/2102.09844.pdf, equations 3-6.
class GNN(BaseEuclideangraphModel):
    def __init__(self, hyperparams):
        super(GNN, self).__init__(hyperparams)
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


class VNDeepSets(BaseEuclideangraphModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.prediction_mode = hyperparams.out_dim == 1
        self.model = "vndeepsets"
        self.hidden_dim = hyperparams.hidden_dim
        self.layer_pooling = hyperparams.layer_pooling
        self.final_pooling = hyperparams.final_pooling
        self.num_layers = hyperparams.num_layers
        self.nonlinearity = hyperparams.nonlinearity
        self.canon_feature = hyperparams.canon_feature
        self.canon_translation = hyperparams.canon_translation
        self.angular_feature = hyperparams.angular_feature
        self.dropout = hyperparams.dropout
        self.out_dim = hyperparams.out_dim
        self.in_dim = len(self.canon_feature)
        self.first_set_layer = VNDeepSetLayer(
            self.in_dim,
            self.hidden_dim,
            self.nonlinearity,
            self.layer_pooling,
            False,
            dropout=self.dropout,
        )
        self.set_layers = SequentialMultiple(
            *[
                VNDeepSetLayer(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.nonlinearity,
                    self.layer_pooling,
                    dropout=self.dropout,
                )
                for i in range(self.num_layers - 1)
            ]
        )
        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim)
        self.batch_size = hyperparams.batch_size

        self.dummy_input = torch.zeros(1, device=self.device, dtype=torch.long)
        self.dummy_indices = torch.zeros(1, device=self.device, dtype=torch.long)

    def forward(self, nodes, loc, edges, vel, edge_attr, charges):
        batch_indices = torch.arange(self.batch_size, device=self.device).reshape(-1, 1)
        batch_indices = batch_indices.repeat(1, 5).reshape(-1)
        mean_loc = ts.scatter(loc, batch_indices, 0, reduce=self.layer_pooling)
        mean_loc = mean_loc.repeat(5, 1, 1).transpose(0, 1).reshape(-1, 3)
        canonical_loc = loc - mean_loc
        # p = position
        # v = velocity
        # a = angular velocity (cross product of position and velocity)
        if self.canon_feature == "p":
            features = torch.stack([canonical_loc], dim=2)
        if self.canon_feature == "pv":
            features = torch.stack([canonical_loc, vel], dim=2)
        elif self.canon_feature == "pva":
            angular = torch.linalg.cross(canonical_loc, vel, dim=1)
            features = torch.stack([canonical_loc, vel, angular], dim=2)
        elif self.canon_feature == "pvc":
            features = torch.stack([canonical_loc, vel, canonical_loc * charges], dim=2)
        elif self.canon_feature == "pvac":
            angular = torch.linalg.cross(canonical_loc, vel, dim=1)
            features = torch.stack(
                [canonical_loc, vel, angular, canonical_loc * charges], dim=2
            )

        x, _ = self.first_set_layer(features, edges)
        x, _ = self.set_layers(x, edges)

        if self.prediction_mode:
            output = self.output_layer(x)
            output = output.squeeze()
            return output
        else:
            x = ts.scatter(x, batch_indices, 0, reduce=self.final_pooling)
        output = self.output_layer(x)

        output = output.repeat(5, 1, 1, 1).transpose(0, 1)
        output = output.reshape(-1, 3, 4)

        rotation_vectors = output[:, :, :3]
        translation_vectors = output[:, :, 3:] if self.canon_translation else 0.0
        translation_vectors = translation_vectors + mean_loc[:, :, None]

        return rotation_vectors, translation_vectors.squeeze()


class VNDeepSetLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        nonlinearity,
        pooling="sum",
        residual=True,
        dropout=0.0,
    ):
        super().__init__()
        self.in_dim = in_channels
        self.out_dim = out_channels
        self.pooling = pooling
        self.residual = residual
        self.nonlinearity = nonlinearity
        self.dropout = dropout

        self.identity_linear = nn.Linear(in_channels, out_channels)
        self.pooling_linear = nn.Linear(in_channels, out_channels)

        self.dropout_layer = nn.Dropout(self.dropout)

        if self.nonlinearity == "softplus":
            self.nonlinear_function = VNSoftplus(out_channels, share_nonlinearity=False)
        elif self.nonlinearity == "relu":
            self.nonlinear_function = VNLeakyReLU(
                out_channels, share_nonlinearity=False, negative_slope=0.0
            )
        elif self.nonlinearity == "leakyrelu":
            self.nonlinear_function = VNLeakyReLU(
                out_channels, share_nonlinearity=False
            )

    def forward(self, x, edges):
        # here x is the features, which depends on canon_feature
        # check VNDeepSets.forward
        #
        edges_1 = edges[0]
        edges_2 = edges[1]

        identity = self.identity_linear(x)

        nodes_1 = torch.index_select(x, 0, edges_1)
        pooled_set = ts.scatter(nodes_1, edges_2, 0, reduce=self.pooling)
        pooling = self.pooling_linear(pooled_set)

        output = self.nonlinear_function(
            (identity + pooling).transpose(1, -1)
        ).transpose(1, -1)

        output = self.dropout_layer(output)

        if self.residual:
            output = output + x

        return output, edges


class Transformer(BaseEuclideangraphModel):
    def __init__(self, hyperparams):
        super(Transformer, self).__init__(hyperparams)
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
