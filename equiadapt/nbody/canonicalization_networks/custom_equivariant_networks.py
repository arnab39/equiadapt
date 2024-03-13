import torch
import torch.nn as nn
import torch_scatter as ts

from equiadapt.nbody.canonicalization_networks.custom_group_equivariant_layers import (
    VNLeakyReLU,
    VNSoftplus,
)


class VNDeepSets(nn.Module):
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


class SequentialMultiple(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
