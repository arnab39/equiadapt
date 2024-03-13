from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from equiadapt.pointcloud.canonicalization_networks.vector_neuron_layers import (
    VNBatchNorm,
    VNLinearLeakyReLU,
    VNMaxPool,
    mean_pool,
)


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Performs k-nearest neighbors search on a given set of points.

    Args:
        x (torch.Tensor): The input tensor representing a set of points.
            Shape: (batch_size, num_points, num_dimensions).
        k (int): The number of nearest neighbors to find.

    Returns:
        torch.Tensor: The indices of the k nearest neighbors for each point in x.
            Shape: (batch_size, num_points, k).
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature_cross(
    x: torch.Tensor, k: int = 20, idx: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes the graph feature cross for a given input tensor.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
        k (int, optional): The number of nearest neighbors to consider. Defaults to 20.
        idx (torch.Tensor, optional): The indices of the nearest neighbors. Defaults to None.

    Returns:
        torch.Tensor: The computed graph feature cross tensor of shape (batch_size, num_dims*3, num_points, k).

    """
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)

    idx_base = torch.arange(0, batch_size).type_as(idx).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)

    feature = (
        torch.cat((feature - x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    )

    return feature


class VNSmall(torch.nn.Module):
    """
    VNSmall is a small variant of the vector neuron equivariant network used for canonicalization of point clouds.

    Args:
        hyperparams (DictConfig): Hyperparameters for the network.

    Attributes:
        n_knn (int): Number of nearest neighbors to consider.
        pooling (str): Pooling type to use, either "max" or "mean".
        conv_pos (VNLinearLeakyReLU): Convolutional layer for positional encoding.
        conv1 (VNLinearLeakyReLU): First convolutional layer.
        bn1 (VNBatchNorm): Batch normalization layer.
        conv2 (VNLinearLeakyReLU): Second convolutional layer.
        dropout (nn.Dropout): Dropout layer.
        pool (Union[VNMaxPool, mean_pool]): Pooling layer.

    Methods:
        __init__: Initializes the VNSmall network.
        forward: Forward pass of the VNSmall network.

    """

    def __init__(self, hyperparams: DictConfig):
        """
        Initialize the VN Small network.

        Args:
            hyperparams (DictConfig): A dictionary-like object containing hyperparameters.

        Raises:
            ValueError: If the specified pooling type is not supported.
        """
        super().__init__()
        self.n_knn = hyperparams.n_knn
        self.pooling = hyperparams.pooling
        self.conv_pos = VNLinearLeakyReLU(3, 64 // 3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64 // 3, 64 // 3, dim=4, negative_slope=0.0)
        self.bn1 = VNBatchNorm(64 // 3, dim=4)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 12 // 3, dim=4, negative_slope=0.0)
        self.dropout = nn.Dropout(p=0.5)

        if self.pooling == "max":
            self.pool = VNMaxPool(64 // 3)
        elif self.pooling == "mean":
            self.pool = mean_pool  # type: ignore
        else:
            raise ValueError(f"Pooling type {self.pooling} not supported")

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VNSmall network.

        For every pointcloud in the batch, the network outputs three vectors that transform equivariantly with respect to SO3 group.

        Args:
            point_cloud (torch.Tensor): Input point cloud tensor of shape (batch_size, num_points, 3).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3, 3).

        """
        point_cloud = point_cloud.unsqueeze(1)
        feat = get_graph_feature_cross(point_cloud, k=self.n_knn)
        out = self.conv_pos(feat)
        out = self.pool(out)

        out = self.bn1(self.conv1(out))
        out = self.conv2(out)
        out = self.dropout(out)

        return out.mean(dim=-1)[:, :3]
