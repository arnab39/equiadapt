import torch
import torch.nn as nn

EPS = 1e-6


class VNSoftplus(nn.Module):
    """Custom module implementing the VN Softplus layer."""

    def __init__(
        self,
        in_channels: int,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.0,
    ) -> None:
        """
        Initialize the VNSoftplus layer.

        Args:
            in_channels (int): Number of input channels.
            share_nonlinearity (bool, optional): Whether to share the nonlinearity across channels. Defaults to False.
            negative_slope (float, optional): Negative slope of the LeakyReLU activation. Defaults to 0.0.
        """
        super().__init__()
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VNSoftplus layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
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
    """Custom module implementing the VN LeakyReLU layer."""

    def __init__(
        self,
        in_channels: int,
        share_nonlinearity: bool = False,
        negative_slope: float = 0.2,
    ) -> None:
        """
        Initialize the VNLeakyReLU layer.

        Args:
            in_channels (int): Number of input channels.
            share_nonlinearity (bool, optional): Whether to share the nonlinearity across channels. Defaults to False.
            negative_slope (float, optional): Negative slope of the LeakyReLU activation. Defaults to 0.2.
        """
        super().__init__()
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VNLeakyReLU layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out
