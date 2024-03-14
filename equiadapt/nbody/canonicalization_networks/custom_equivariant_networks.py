from typing import Any, Tuple

import torch
import torch.nn as nn
import torch_scatter as ts

from equiadapt.nbody.canonicalization_networks.custom_group_equivariant_layers import (
    VNLeakyReLU,
    VNSoftplus,
)


class VNDeepSets(nn.Module):
    """
    A class representing the VNDeepSets model.

    Args:
        hyperparams: A dictionary containing hyperparameters for the model.
        device (str): The device to run the model on. Defaults to "cuda" if available, otherwise "cpu".

    Attributes:
        device (str): The device the model is running on.
        learning_rate (float): The learning rate for the model.
        weight_decay (float): The weight decay for the model.
        patience (int): The patience value for early stopping.
        prediction_mode (bool): Whether the model is in prediction mode (output dimension is 1).
        model (str): The name of the model.
        hidden_dim (int): The dimension of the hidden layers.
        layer_pooling (str): The type of pooling to use within each layer.
        final_pooling (str): The type of pooling to use in the final layer.
        num_layers (int): The number of layers in the model.
        nonlinearity (str): The nonlinearity function to use.
        canon_feature (str): The type of canonical feature to use.
        canon_translation (bool): Whether to include canonical translation in the features.
        angular_feature (bool): Whether to include angular features in the features.
        dropout (float): The dropout rate.
        out_dim (int): The output dimension of the model.
        in_dim (int): The input dimension of the model.
        first_set_layer (VNDeepSetLayer): The first layer of the VNDeepSets model.
        set_layers (SequentialMultiple): The set of layers in the VNDeepSets model.
        output_layer (nn.Linear): The output layer of the VNDeepSets model.
        batch_size (int): The batch size for the model.
        dummy_input (torch.Tensor): A dummy input tensor for initialization.
        dummy_indices (torch.Tensor): A dummy indices tensor for initialization.
    """

    def __init__(
        self,
        hyperparams: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.device: str = device
        self.learning_rate: float = (
            hyperparams.learning_rate if hasattr(hyperparams, "learning_rate") else None
        )
        self.weight_decay: float = (
            hyperparams.weight_decay if hasattr(hyperparams, "weight_decay") else 0.0
        )
        self.patience: int = (
            hyperparams.patience if hasattr(hyperparams, "patience") else 100
        )
        self.prediction_mode: bool = hyperparams.out_dim == 1
        self.model: str = "vndeepsets"
        self.hidden_dim: int = hyperparams.hidden_dim
        self.layer_pooling: str = hyperparams.layer_pooling
        self.final_pooling: str = hyperparams.final_pooling
        self.num_layers: int = hyperparams.num_layers
        self.nonlinearity: str = hyperparams.nonlinearity
        self.canon_feature: str = hyperparams.canon_feature
        self.canon_translation: bool = hyperparams.canon_translation
        self.angular_feature: bool = hyperparams.angular_feature
        self.dropout: float = hyperparams.dropout
        self.out_dim: int = hyperparams.out_dim
        self.in_dim: int = len(self.canon_feature)
        self.first_set_layer: VNDeepSetLayer = VNDeepSetLayer(
            self.in_dim,
            self.hidden_dim,
            self.nonlinearity,
            self.layer_pooling,
            False,
            dropout=self.dropout,
        )
        self.set_layers: SequentialMultiple = SequentialMultiple(
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
        self.output_layer: nn.Linear = nn.Linear(self.hidden_dim, self.out_dim)
        self.batch_size: int = hyperparams.batch_size

        self.dummy_input: torch.Tensor = torch.zeros(
            1, device=self.device, dtype=torch.long
        )
        self.dummy_indices: torch.Tensor = torch.zeros(
            1, device=self.device, dtype=torch.long
        )

    def forward(
        self,
        nodes: torch.Tensor,
        loc: torch.Tensor,
        edges: torch.Tensor,
        vel: torch.Tensor,
        edge_attr: torch.Tensor,
        charges: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VNDeepSets model.

        Args:
            nodes (torch.Tensor): The nodes tensor.
            loc (torch.Tensor): The location tensor.
            edges (torch.Tensor): The edges tensor.
            vel (torch.Tensor): The velocity tensor.
            edge_attr (torch.Tensor): The edge attributes tensor.
            charges (torch.Tensor): The charges tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The rotation vectors and translation vectors.
        """
        batch_indices: torch.Tensor = torch.arange(
            self.batch_size, device=self.device
        ).reshape(-1, 1)
        batch_indices = batch_indices.repeat(1, 5).reshape(-1)
        mean_loc: torch.Tensor = ts.scatter(
            loc, batch_indices, 0, reduce=self.layer_pooling
        )
        mean_loc = mean_loc.repeat(5, 1, 1).transpose(0, 1).reshape(-1, 3)
        canonical_loc: torch.Tensor = loc - mean_loc

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
    """
    A class representing a layer in the VNDeepSets model.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        nonlinearity (str): The nonlinearity function to use.
        pooling (str): The type of pooling to use. Defaults to "sum".
        residual (bool): Whether to use residual connections. Defaults to True.
        dropout (float): The dropout rate. Defaults to 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nonlinearity: str,
        pooling: str = "sum",
        residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim: int = in_channels
        self.out_dim: int = out_channels
        self.pooling: str = pooling
        self.residual: bool = residual
        self.nonlinearity: str = nonlinearity
        self.dropout: float = dropout

        self.identity_linear: nn.Linear = nn.Linear(in_channels, out_channels)
        self.pooling_linear: nn.Linear = nn.Linear(in_channels, out_channels)

        self.dropout_layer: nn.Dropout = nn.Dropout(self.dropout)

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

    def forward(
        self, x: torch.Tensor, edges: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VNDeepSetLayer.

        Args:
            x (torch.Tensor): The input tensor.
            edges (torch.Tensor): The edges tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output tensor and edges tensor.
        """
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
    """
    A class representing a sequence of multiple layers.

    Inherits from nn.Sequential.

    Args:
        *args: Variable length list of layers.
    """

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SequentialMultiple.

        Args:
            *inputs: Variable length list of input tensors.

        Returns:
            torch.Tensor: The output tensor.
        """
        for module in self._modules.values():
            if type(inputs) is tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
