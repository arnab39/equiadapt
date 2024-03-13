import math

import kornia as K
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotationEquivariantConvLift(nn.Module):
    """
    This class represents a rotation equivariant convolutional layer with lifting.

    The layer is equivariant to a group of rotations. The weights of the layer are initialized using the Kaiming uniform initialization method. The layer supports optional bias.

    Methods:
        __init__: Initializes the RotationEquivariantConvLift instance.
        get_rotated_weights: Returns the weights of the layer after rotation.
        forward: Performs a forward pass through the layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_rotations: int = 4,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device: str = "cuda",
    ):
        """
        Initializes the RotationEquivariantConvLift instance.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The size of the kernel.
            num_rotations (int, optional): The number of rotations in the group. Defaults to 4.
            stride (int, optional): The stride of the convolution. Defaults to 1.
            padding (int, optional): The padding of the convolution. Defaults to 0.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
            device (str, optional): The device to run the layer on. Defaults to "cuda".
        """
        super().__init__()
        self.weights = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size).to(device)
        )
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels).to(device))
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None  # type: ignore
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.num_rotations = num_rotations
        self.kernel_size = kernel_size

    def get_rotated_weights(
        self, weights: torch.Tensor, num_rotations: int = 4
    ) -> torch.Tensor:
        """
        Returns the weights of the layer after rotation.

        Args:
            weights (torch.Tensor): The weights of the layer.
            num_rotations (int, optional): The number of rotations in the group. Defaults to 4.

        Returns:
            torch.Tensor: The weights after rotation.
        """
        device = weights.device
        weights = weights.flatten(0, 1).unsqueeze(0).repeat(num_rotations, 1, 1, 1)
        rotated_weights = K.geometry.rotate(
            weights,
            torch.linspace(0.0, 360.0, steps=num_rotations + 1, dtype=torch.float32)[
                :num_rotations
            ].to(device),
        )
        rotated_weights = rotated_weights.reshape(
            self.num_rotations,
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        ).transpose(0, 1)
        return rotated_weights.flatten(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the layer.

        Args:
            x (torch.Tensor): The input data. It should have the shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output of the layer. It has the shape (batch_size, out_channels, num_rotations, height, width).
        """
        batch_size = x.shape[0]
        rotated_weights = self.get_rotated_weights(self.weights, self.num_rotations)
        # shape (out_channels * num_rotations, in_channels, kernel_size, kernel_size)
        x = F.conv2d(x, rotated_weights, stride=self.stride, padding=self.padding)
        x = x.reshape(
            batch_size, self.out_channels, self.num_rotations, x.shape[2], x.shape[3]
        )
        if self.bias is not None:
            x = x + self.bias[None, :, None, None, None]
        return x


class RotoReflectionEquivariantConvLift(nn.Module):
    """
    This class represents a roto-reflection equivariant convolutional layer with lifting.

    The layer is equivariant to a group of rotations and reflections. The weights of the layer are initialized using the Kaiming uniform initialization method. The layer supports optional bias.

    Methods:
        __init__: Initializes the RotoReflectionEquivariantConvLift instance.
        get_rotoreflected_weights: Returns the weights of the layer after rotation, reflection, and permutation.
        forward: Performs a forward pass through the layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_rotations: int = 4,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device: str = "cuda",
    ):
        """
        Initializes the RotoReflectionEquivariantConvLift instance.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The size of the kernel.
            num_rotations (int, optional): The number of rotations in the group. Defaults to 4.
            stride (int, optional): The stride of the convolution. Defaults to 1.
            padding (int, optional): The padding of the convolution. Defaults to 0.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
            device (str, optional): The device to run the layer on. Defaults to "cuda".
        """
        super().__init__()
        num_group_elements = 2 * num_rotations
        self.weights = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size).to(device)
        )
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels).to(device))
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None  # type: ignore
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.num_rotations = num_rotations
        self.kernel_size = kernel_size
        self.num_group_elements = num_group_elements

    def get_rotoreflected_weights(
        self, weights: torch.Tensor, num_rotations: int = 4
    ) -> torch.Tensor:
        """
        Returns the weights of the layer after rotation and reflection.

        Args:
            weights (torch.Tensor): The weights of the layer.
            num_rotations (int, optional): The number of rotations in the group. Defaults to 4.

        Returns:
            torch.Tensor: The weights after rotation, reflection, and permutation.
        """
        device = weights.device
        weights = weights.flatten(0, 1).unsqueeze(0).repeat(num_rotations, 1, 1, 1)
        rotated_weights = K.geometry.rotate(
            weights,
            torch.linspace(0.0, 360.0, steps=num_rotations + 1, dtype=torch.float32)[
                :num_rotations
            ].to(device),
        )
        reflected_weights = K.geometry.hflip(rotated_weights)
        rotoreflected_weights = torch.cat([rotated_weights, reflected_weights], dim=0)
        rotoreflected_weights = rotoreflected_weights.reshape(
            self.num_group_elements,
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        ).transpose(0, 1)
        return rotoreflected_weights.flatten(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the layer.

        Args:
            x (torch.Tensor): The input data. It should have the shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output of the layer. It has the shape (batch_size, out_channels, num_group_elements, height, width).
        """
        batch_size = x.shape[0]
        rotoreflected_weights = self.get_rotoreflected_weights(
            self.weights, self.num_rotations
        )
        # shape (out_channels * num_group_elements, in_channels, kernel_size, kernel_size)
        x = F.conv2d(x, rotoreflected_weights, stride=self.stride, padding=self.padding)
        x = x.reshape(
            batch_size,
            self.out_channels,
            self.num_group_elements,
            x.shape[2],
            x.shape[3],
        )
        if self.bias is not None:
            x = x + self.bias[None, :, None, None, None]
        return x


class RotationEquivariantConv(nn.Module):
    """
    This class represents a rotation equivariant convolutional layer.

    The layer is equivariant to a group of rotations. The weights of the layer are initialized using the Kaiming uniform initialization method. The layer supports optional bias.

    Methods:
        __init__: Initializes the RotationEquivariantConv instance.
        get_rotated_permuted_weights: Returns the weights of the layer after rotation and permutation.
        forward: Performs a forward pass through the layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_rotations: int = 4,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device: str = "cuda",
    ):
        """
        Initializes the RotationEquivariantConv instance.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The size of the kernel.
            num_rotations (int, optional): The number of rotations in the group. Defaults to 4.
            stride (int, optional): The stride of the convolution. Defaults to 1.
            padding (int, optional): The padding of the convolution. Defaults to 0.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
            device (str, optional): The device to run the layer on. Defaults to "cuda".
        """
        super().__init__()
        self.weights = nn.Parameter(
            torch.empty(
                out_channels, in_channels, num_rotations, kernel_size, kernel_size
            ).to(device)
        )
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels).to(device))
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None  # type: ignore
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.num_rotations = num_rotations
        self.kernel_size = kernel_size
        indices = (
            torch.arange(num_rotations)
            .view((1, 1, num_rotations, 1, 1))
            .repeat(
                num_rotations, out_channels * in_channels, 1, kernel_size, kernel_size
            )
        )
        self.permute_indices_along_group = (
            (indices - torch.arange(num_rotations)[:, None, None, None, None])
            % num_rotations
        ).to(device)
        self.angle_list = torch.linspace(
            0.0, 360.0, steps=num_rotations + 1, dtype=torch.float32
        )[:num_rotations].to(device)

    def get_rotated_permuted_weights(
        self, weights: torch.Tensor, num_rotations: int = 4
    ) -> torch.Tensor:
        """
        Returns the weights of the layer after rotation and permutation.

        Args:
            weights (torch.Tensor): The weights of the layer.
            num_rotations (int, optional): The number of rotations in the group. Defaults to 4.

        Returns:
            torch.Tensor: The weights after rotation and permutation.
        """
        weights = weights.flatten(0, 1).unsqueeze(0).repeat(num_rotations, 1, 1, 1, 1)
        permuted_weights = torch.gather(weights, 2, self.permute_indices_along_group)
        rotated_permuted_weights = K.geometry.rotate(
            permuted_weights.flatten(1, 2),
            self.angle_list,
        )
        rotated_permuted_weights = (
            rotated_permuted_weights.reshape(
                self.num_rotations,
                self.out_channels,
                self.in_channels,
                self.num_rotations,
                self.kernel_size,
                self.kernel_size,
            )
            .transpose(0, 1)
            .reshape(
                self.out_channels * self.num_rotations,
                self.in_channels * self.num_rotations,
                self.kernel_size,
                self.kernel_size,
            )
        )
        return rotated_permuted_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the layer.

        Args:
            x (torch.Tensor): The input data. It should have the shape (batch_size, in_channels, num_rotations, height, width).

        Returns:
            torch.Tensor: The output of the layer. It has the shape (batch_size, out_channels, num_rotations, height, width).
        """
        batch_size = x.shape[0]
        x = x.flatten(1, 2)
        # shape (batch_size, in_channels * num_rotations, height, width)
        rotated_permuted_weights = self.get_rotated_permuted_weights(
            self.weights, self.num_rotations
        )
        # shape (out_channels * num_rotations, in_channels * num_rotations, kernal_size, kernal_size)
        x = F.conv2d(
            x, rotated_permuted_weights, stride=self.stride, padding=self.padding
        )
        x = x.reshape(
            batch_size, self.out_channels, self.num_rotations, x.shape[2], x.shape[3]
        )
        if self.bias is not None:
            x = x + self.bias[None, :, None, None, None]
        return x


class RotoReflectionEquivariantConv(nn.Module):
    """
    This class represents a roto-reflection equivariant convolutional layer.

    The layer is equivariant to a group of rotations and reflections. The weights of the layer are initialized using the Kaiming uniform initialization method. The layer supports optional bias.

    Methods:
        __init__: Initializes the RotoReflectionEquivariantConv instance.
        get_rotoreflected_permuted_weights: Returns the weights of the layer after rotation, reflection, and permutation.
        forward: Performs a forward pass through the layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_rotations: int = 4,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device: str = "cuda",
    ):
        """
        Initializes the RotoReflectionEquivariantConv instance.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The size of the kernel.
            num_rotations (int, optional): The number of rotations in the group. Defaults to 4.
            stride (int, optional): The stride of the convolution. Defaults to 1.
            padding (int, optional): The padding of the convolution. Defaults to 0.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
            device (str, optional): The device to run the layer on. Defaults to "cuda".
        """
        super().__init__()
        num_group_elements: int = 2 * num_rotations
        self.weights = nn.Parameter(
            torch.empty(
                out_channels, in_channels, num_group_elements, kernel_size, kernel_size
            ).to(device)
        )
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels).to(device))
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None  # type: ignore
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.num_rotations = num_rotations
        self.kernel_size = kernel_size
        self.num_group_elements = num_group_elements
        indices = (
            torch.arange(num_rotations)
            .view((1, 1, num_rotations, 1, 1))
            .repeat(
                num_rotations, out_channels * in_channels, 1, kernel_size, kernel_size
            )
        )
        self.permute_indices_along_group = (
            indices - torch.arange(num_rotations)[:, None, None, None, None]
        ) % num_rotations
        self.permute_indices_along_group_inverse = (
            indices + torch.arange(num_rotations)[:, None, None, None, None]
        ) % num_rotations
        self.permute_indices_upper_half = torch.cat(
            [
                self.permute_indices_along_group,
                self.permute_indices_along_group_inverse + num_rotations,
            ],
            dim=2,
        )
        self.permute_indices_lower_half = torch.cat(
            [
                self.permute_indices_along_group_inverse + num_rotations,
                self.permute_indices_along_group,
            ],
            dim=2,
        )
        self.permute_indices = torch.cat(
            [self.permute_indices_upper_half, self.permute_indices_lower_half], dim=0
        ).to(device)
        self.angle_list = torch.cat(
            [
                torch.linspace(
                    0.0, 360.0, steps=num_rotations + 1, dtype=torch.float32
                )[:num_rotations],
                torch.linspace(
                    0.0, 360.0, steps=num_rotations + 1, dtype=torch.float32
                )[:num_rotations],
            ]
        ).to(device)

    def get_rotoreflected_permuted_weights(
        self, weights: torch.Tensor, num_rotations: int = 4
    ) -> torch.Tensor:
        """
        Returns the weights of the layer after rotation, reflection, and permutation.

        Args:
            weights (torch.Tensor): The weights of the layer.
            num_rotations (int, optional): The number of rotations in the group. Defaults to 4.

        Returns:
            torch.Tensor: The weights after rotation, reflection, and permutation.
        """
        weights = (
            weights.flatten(0, 1)
            .unsqueeze(0)
            .repeat(self.num_group_elements, 1, 1, 1, 1)
        )
        # shape (num_group_elements, out_channels * in_channels, num_group_elements, kernel_size, kernel_size)
        permuted_weights = torch.gather(weights, 2, self.permute_indices)
        rotated_permuted_weights = K.geometry.rotate(
            permuted_weights.flatten(1, 2), self.angle_list
        )
        rotoreflected_permuted_weights = torch.cat(
            [
                rotated_permuted_weights[: self.num_rotations],
                K.geometry.hflip(rotated_permuted_weights[self.num_rotations :]),
            ]
        )
        rotoreflected_permuted_weights = (
            rotoreflected_permuted_weights.reshape(
                self.num_group_elements,
                self.out_channels,
                self.in_channels,
                self.num_group_elements,
                self.kernel_size,
                self.kernel_size,
            )
            .transpose(0, 1)
            .reshape(
                self.out_channels * self.num_group_elements,
                self.in_channels * self.num_group_elements,
                self.kernel_size,
                self.kernel_size,
            )
        )
        return rotoreflected_permuted_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the layer.

        Args:
            x (torch.Tensor): The input data. It should have the shape (batch_size, in_channels, num_group_elements, height, width).

        Returns:
            torch.Tensor: The output of the layer. It has the shape (batch_size, out_channels, num_group_elements, height, width).
        """
        batch_size = x.shape[0]
        x = x.flatten(1, 2)
        # shape (batch_size, in_channels * num_group_elements, height, width)
        rotoreflected_permuted_weights = self.get_rotoreflected_permuted_weights(
            self.weights, self.num_rotations
        )
        # shape (out_channels * num_group_elements, in_channels * num_group_elements, kernel_size, kernel_size)
        x = F.conv2d(
            x, rotoreflected_permuted_weights, stride=self.stride, padding=self.padding
        )
        x = x.reshape(
            batch_size,
            self.out_channels,
            self.num_group_elements,
            x.shape[2],
            x.shape[3],
        )
        if self.bias is not None:
            x = x + self.bias[None, :, None, None, None]
        return x
