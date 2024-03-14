"""
Layers for vector neuron networks

Taken from Vector Neurons: A General Framework for SO(3)-Equivariant Networks (https://arxiv.org/abs/2104.12229) paper and
their codebase https://github.com/FlyingGiraffe/vnn
"""

from typing import Tuple

import torch
import torch.nn as nn

EPS = 1e-6


class VNLinear(nn.Module):
    """
    Vector Neuron Linear layer.

    This layer applies a linear transformation to the input tensor.

    Methods:
        __init__: Initializes the VNLinear layer.
        forward: Performs forward pass of the VNLinear layer.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes a VNLinear layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass of the VNLinear layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N_feat, 3, N_samples, ...].

        Returns:
            torch.Tensor: Output tensor of shape [B, N_feat, 3, N_samples, ...].
        """
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNBilinear(nn.Module):
    """
    Vector Neuron Bilinear layer.

    VNBilinear applies a bilinear layer to the input features.

    Methods:
        __init__: Initializes the VNBilinear layer.
        forward: Performs forward pass of the VNBilinear layer.
    """

    def __init__(self, in_channels1: int, in_channels2: int, out_channels: int):
        """
        Initializes the VNBilinear layer.

        Args:
            in_channels1 (int): Number of input channels for the first input.
            in_channels2 (int): Number of input channels for the second input.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.map_to_feat = nn.Bilinear(
            in_channels1, in_channels2, out_channels, bias=False
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VNBilinear layer.

        Args:
            x (torch.Tensor): Input features of shape [B, N_feat, 3, N_samples, ...].
            labels (torch.Tensor): Labels of shape [B, N_feat, N_samples].

        Returns:
            torch.Tensor: Output features after applying the bilinear transformation.
        """
        labels = labels.repeat(1, x.shape[2], 1).float()
        x_out = self.map_to_feat(x.transpose(1, -1), labels).transpose(1, -1)
        return x_out


class VNSoftplus(nn.Module):
    """
    Vector Neuron Softplus layer.

    VNSoftplus applies a softplus activation to the input features.

    Methods:
        __init__: Initializes the VNSoftplus layer.
        forward: Performs forward pass of the VNSoftplus layer.
    """

    def __init__(
        self,
        in_channels: int,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.0,
    ):
        """
        Initializes a VNSoftplus layer.

        Args:
            in_channels (int): Number of input channels.
            share_nonlinearity (bool): Whether to share the nonlinearity across channels.
            negative_slope (float): Negative slope parameter for the LeakyReLU activation.

        """
        super().__init__()
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass of the VNSoftplus layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N_feat, 3, N_samples, ...].

        Returns:
            torch.Tensor: Output tensor of shape [B, N_feat, 3, N_samples, ...].

        """
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        angle_between = torch.acos(
            dotprod
            / (
                torch.norm(x, dim=2, keepdim=True) * torch.norm(d, dim=2, keepdim=True)
                + EPS
            )
        )
        mask = torch.cos(angle_between / 2) ** 2
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNLeakyReLU(nn.Module):
    """
    Vector Neuron Leaky ReLU layer.

    VNLLeakyReLU applies a LeakyReLU activation to the input features.

    Methods:
        __init__: Initializes the VNLeakyReLU layer.
        forward: Performs forward pass of the VNLeakyReLU layer.
    """

    def __init__(
        self,
        in_channels: int,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
    ):
        """
        Vector Neuron Leaky ReLU (VNLeakyReLU) module.

        Args:
            in_channels (int): Number of input channels.
            share_nonlinearity (bool, optional): Whether to share the nonlinearity across channels.
                If True, a single linear layer is used to compute the direction.
                If False, a separate linear layer is used for each channel.
                Defaults to False.
            negative_slope (float, optional): Negative slope of the Leaky ReLU activation.
                Defaults to 0.2.
        """
        super().__init__()
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VNLeakyReLU module.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N_feat, 3, N_samples, ...].

        Returns:
            torch.Tensor: Output tensor after applying VNLeakyReLU activation.
        """
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNLinearLeakyReLU(nn.Module):
    """
    Vector Neuron Linear Leaky ReLU layer.

    VNLinearLeakyReLU applies a linear transformation followed by a LeakyReLU activation to the input features.

    Methods:
        __init__: Initializes the VNLinearLeakyReLU layer.
        forward: Performs forward pass of the VNLinearLeakyReLU layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 5,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
    ):
        """
        Vector Neuron Linear Leaky ReLU layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dim (int, optional): Dimension of the input features. Defaults to 5.
            share_nonlinearity (bool, optional): Whether to share the nonlinearity across channels. Defaults to False.
            negative_slope (float, optional): Negative slope of the LeakyReLU activation. Defaults to 0.2.
        """
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VNLinearLeakyReLU layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N_feat, 3, N_samples, ...]

        Returns:
            torch.Tensor: Output tensor of shape [B, N_feat, 3, N_samples, ...]
        """
        # Linear
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNBatchNorm(nn.Module):
    """
    Vector Neuron Batch Normalization layer.

    VNBatchNorm applies batch normalization to the input features.

    Methods:
        __init__: Initializes the VNBatchNorm layer.
        forward: Performs forward pass of the VNBatchNorm layer.
    """

    def __init__(self, num_features: int, dim: int):
        """
        Vector Neuron Batch Normalization layer.

        Args:
            num_features (int): Number of input features.
            dim (int): Dimensionality of the input tensor.

        """
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn1d = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn2d = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vector Neuron Batch Normalization layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N_feat, 3, N_samples, ...].

        Returns:
            torch.Tensor: Output tensor after applying batch normalization.

        """
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        if self.dim == 3 or self.dim == 4:
            norm_bn = self.bn1d(norm)
        elif self.dim == 5:
            norm_bn = self.bn2d(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn

        return x


class VNMaxPool(nn.Module):
    """
    Vector Neuron Max Pooling layer.

    VNMaxPool applies max pooling to the input features.

    Methods:
        __init__: Initializes the VNMaxPool layer.
        forward: Performs forward pass of the VNMaxPool layer.
    """

    def __init__(self, in_channels: int):
        """
        Initializes a VNMaxPool layer.

        Args:
            in_channels (int): The number of input channels.

        """
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs vector neuron max pooling on the input tensor.

        Args:
            x (torch.Tensor): Point features of shape [B, N_feat, 3, N_samples, ...].

        Returns:
            torch.Tensor: Max pooled tensor of shape [B, N_feat, 3, N_samples, ...].
        """
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    Compute the mean pooling of a tensor along a specified dimension.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int, optional): The dimension along which to compute the mean pooling. Default is -1.
        keepdim (bool, optional): Whether to keep the dimension of the input tensor. Default is False.

    Returns:
        torch.Tensor: The mean pooled tensor.

    """
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    """
    Vector Neuron Standard Feature module.

    This module performs standard feature extraction using Vector Neuron layers.
    It takes point features as input and applies a series of VNLinearLeakyReLU layers
    followed by a linear layer to produce the standard features.

    Attributes:
        dim (int): Dimension of the input features.
        normalize_frame (bool): Whether to normalize the frame.

    Methods:
        __init__: Initializes the VNStdFeature module.
        forward: Performs forward pass of the VNStdFeature module.

    Shape:
        - Input: (B, N_feat, 3, N_samples, ...)
        - Output:
            - x_std: (B, N_feat, dim, N_samples, ...)
            - z0: (B, dim, 3)

    Example:
        >>> model = VNStdFeature(in_channels=64, dim=4, normalize_frame=True)
        >>> input = torch.randn(2, 64, 3, 100)
        >>> output, frame_vectors = model(input)
    """

    def __init__(
        self,
        in_channels: int,
        dim: int = 4,
        normalize_frame: bool = False,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
    ):
        """
        Initializes the VNStdFeature layer.

        Args:
            in_channels (int): Number of input channels.
            dim (int, optional): Dimension of the input feature. Defaults to 4.
            normalize_frame (bool, optional): Whether to normalize the frame. Defaults to False.
            share_nonlinearity (bool, optional): Whether to share the nonlinearity across layers. Defaults to False.
            negative_slope (float, optional): Negative slope of the LeakyReLU activation function. Defaults to 0.2.
        """
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        self.vn1 = VNLinearLeakyReLU(
            in_channels,
            in_channels // 2,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        self.vn2 = VNLinearLeakyReLU(
            in_channels // 2,
            in_channels // 4,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels // 4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VNStdFeature module.

        Args:
            x (torch.Tensor): Input point features of shape (B, N_feat, 3, N_samples, ...).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the standard features and the frame vectors.

        Note:
            - The frame vectors are computed only if `normalize_frame` is set to True.
            - The shape of the standard features depends on the value of `dim`.
        """
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            v1 = z0[:, 0, :]
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True))  # ignore type
            u1 = v1 / (v1_norm + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1  # ignore type
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True))  # ignore type
            u2 = v2 / (v2_norm + EPS)

            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum("bijm,bjkm->bikm", x, z0)
        elif self.dim == 3:
            x_std = torch.einsum("bij,bjk->bik", x, z0)
        elif self.dim == 5:
            x_std = torch.einsum("bijmn,bjkmn->bikmn", x, z0)

        return x_std, z0
