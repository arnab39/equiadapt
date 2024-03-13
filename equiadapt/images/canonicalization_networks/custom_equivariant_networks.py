from typing import Tuple

import torch
import torch.nn as nn

from .custom_group_equivariant_layers import (
    RotationEquivariantConv,
    RotationEquivariantConvLift,
    RotoReflectionEquivariantConv,
    RotoReflectionEquivariantConvLift,
)


class CustomEquivariantNetwork(nn.Module):
    """
    This class represents a custom equivariant network.

    The network is equivariant to a specified group, which can be either the rotation group or the roto-reflection group. The network consists of a sequence of equivariant convolutional layers, each followed by a ReLU activation function.

    Methods:
        __init__: Initializes the CustomEquivariantNetwork instance.
        forward: Performs a forward pass through the network.
    """

    def __init__(
        self,
        in_shape: Tuple[int, int, int, int],
        out_channels: int,
        kernel_size: int,
        group_type: str = "rotation",
        num_rotations: int = 4,
        num_layers: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initializes the CustomEquivariantNetwork instance.

        Args:
            in_shape (Tuple[int, int, int, int]): The shape of the input data.
            out_channels (int): The number of output channels.
            kernel_size (int): The size of the kernel in the convolutional layers.
            group_type (str, optional): The type of group the network is equivariant to. Defaults to "rotation".
            num_rotations (int, optional): The number of rotations in the group. Defaults to 4.
            num_layers (int, optional): The number of layers in the network. Defaults to 1.
            device (str, optional): The device to run the network on. Defaults to "cuda" if available, otherwise "cpu".
        """
        super().__init__()

        if group_type == "rotation":
            layer_list = [
                RotationEquivariantConvLift(
                    in_shape[0], out_channels, kernel_size, num_rotations, device=device
                )  # type: ignore
            ]
            for i in range(num_layers - 1):
                layer_list.append(nn.ReLU())  # type: ignore
                layer_list.append(
                    RotationEquivariantConv(
                        out_channels, out_channels, 1, num_rotations, device=device
                    )  # type: ignore
                )
            self.eqv_network = nn.Sequential(*layer_list)
        elif group_type == "roto-reflection":
            layer_list = [
                RotoReflectionEquivariantConvLift(
                    in_shape[0], out_channels, kernel_size, num_rotations, device=device
                )  # type: ignore
            ]
            for i in range(num_layers - 1):
                layer_list.append(nn.ReLU())  # type: ignore
                layer_list.append(
                    RotoReflectionEquivariantConv(
                        out_channels, out_channels, 1, num_rotations, device=device
                    )  # type: ignore
                )
            self.eqv_network = nn.Sequential(*layer_list)
        else:
            raise ValueError("group_type must be rotation or roto-reflection for now.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input data. It should have the shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output of the network. It has the shape (batch_size, group_size).
        """
        feature_map = self.eqv_network(x)
        group_activatiobs = torch.mean(feature_map, dim=(1, 3, 4))

        return group_activatiobs
