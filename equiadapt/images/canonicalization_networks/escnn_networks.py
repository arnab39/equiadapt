from typing import Tuple

import e2cnn
import torch
from e2cnn import gspaces


class ESCNNEquivariantNetwork(torch.nn.Module):
    """
    This class represents an Equivariant Convolutional Neural Network (Equivariant CNN).

    The network is equivariant to a group of transformations, which can be either rotations or roto-reflections. The network consists of a sequence of equivariant convolutional layers, each followed by batch normalization, a ReLU activation function, and dropout. The number of output channels of the convolutional layers is the same for all layers.

    Methods:
        __init__: Initializes the ESCNNEquivariantNetwork instance.
        forward: Performs a forward pass through the network.
    """

    def __init__(
        self,
        in_shape: tuple,
        out_channels: int,
        kernel_size: int,
        group_type: str = "rotation",
        num_rotations: int = 4,
        num_layers: int = 1,
    ):
        """
        Initializes the ESCNNEquivariantNetwork instance.

        Args:
            in_shape (tuple): The shape of the input data. It should be a tuple of the form (in_channels, height, width).
            out_channels (int): The number of output channels of the convolutional layers.
            kernel_size (int): The size of the kernel of the convolutional layers.
            group_type (str, optional): The type of the group of transformations. It can be either "rotation" or "roto-reflection". Defaults to "rotation".
            num_rotations (int, optional): The number of rotations in the group. Defaults to 4.
            num_layers (int, optional): The number of convolutional layers. Defaults to 1.
        """
        super().__init__()

        self.in_channels = in_shape[0]
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_type = group_type
        self.num_rotations = num_rotations

        if group_type == "rotation":
            self.gspace = gspaces.Rot2dOnR2(num_rotations)
        elif group_type == "roto-reflection":
            self.gspace = gspaces.FlipRot2dOnR2(num_rotations)
        else:
            raise ValueError("group_type must be rotation or roto-reflection for now.")

        # If the group is roto-reflection, then the number of group elements is twice the number of rotations
        self.num_group_elements = (
            num_rotations if group_type == "rotation" else 2 * num_rotations
        )

        r1 = e2cnn.nn.FieldType(
            self.gspace, [self.gspace.trivial_repr] * self.in_channels
        )
        r2 = e2cnn.nn.FieldType(self.gspace, [self.gspace.regular_repr] * out_channels)

        self.in_type = r1
        self.out_type = r2

        modules = [
            e2cnn.nn.R2Conv(self.in_type, self.out_type, kernel_size),
            e2cnn.nn.InnerBatchNorm(self.out_type, momentum=0.9),
            e2cnn.nn.ReLU(self.out_type, inplace=True),
            e2cnn.nn.PointwiseDropout(self.out_type, p=0.5),
        ]
        for _ in range(num_layers - 2):
            modules.append(
                e2cnn.nn.R2Conv(self.out_type, self.out_type, kernel_size),
            )
            modules.append(
                e2cnn.nn.InnerBatchNorm(self.out_type, momentum=0.9),
            )
            modules.append(
                e2cnn.nn.ReLU(self.out_type, inplace=True),
            )
            modules.append(
                e2cnn.nn.PointwiseDropout(self.out_type, p=0.5),
            )

        modules.append(
            e2cnn.nn.R2Conv(self.out_type, self.out_type, kernel_size),
        )

        self.eqv_network = e2cnn.nn.SequentialModule(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input data. It should have the shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output of the network. It has the shape (batch_size, num_group_elements).
        """
        x = e2cnn.nn.GeometricTensor(x, self.in_type)
        out = self.eqv_network(x)

        feature_map = out.tensor
        feature_map = feature_map.reshape(
            feature_map.shape[0],
            self.out_channels,
            self.num_group_elements,
            feature_map.shape[-2],
            feature_map.shape[-1],
        )

        group_activations = torch.mean(feature_map, dim=(1, 3, 4))

        return group_activations


class ESCNNSteerableNetwork(torch.nn.Module):
    """
    This class represents a Steerable Equivariant Convolutional Neural Network (Equivariant CNN).

    The network is equivariant under all planar rotations. The network consists of a sequence of equivariant convolutional layers, each followed by batch normalization and a FourierELU activation function. The number of output channels of the convolutional layers is the same for all layers.

    Methods:
        __init__: Initializes the ESCNNSteerableNetwork instance.
        forward: Performs a forward pass through the network.
    """

    def __init__(
        self,
        in_shape: tuple,
        out_channels: int,
        kernel_size: int = 9,
        group_type: str = "rotation",
        num_layers: int = 1,
    ):
        """
        Initializes the ESCNNSteerableNetwork instance.

        Args:
            in_shape (tuple): The shape of the input data. It should be a tuple of the form (in_channels, height, width).
            out_channels (int): The number of output channels of the convolutional layers.
            kernel_size (int, optional): The size of the kernel of the convolutional layers. Defaults to 9.
            group_type (str, optional): The type of the group of transformations. It can be either "rotation" or "roto-reflection". Defaults to "rotation".
            num_layers (int, optional): The number of convolutional layers. Defaults to 1.
        """
        super().__init__()

        self.group_type = group_type
        assert group_type == "rotation", "group_type must be rotation for now."
        # TODO: Add support for roto-reflection group

        # The model is equivariant under all planar rotations
        self.gspace = gspaces.Rot2dOnR2(N=-1)

        # The input image is a scalar field, corresponding to the trivial representation
        in_type = e2cnn.nn.FieldType(
            self.gspace, in_shape[0] * [self.gspace.trivial_repr]
        )

        # Store the input type for wrapping the images into a geometric tensor during the forward pass
        self.in_type = in_type

        # Initialize the modules list for the sequential network
        modules = []

        # Dynamically add layers based on num_layers
        for _ in range(num_layers):
            activation = e2cnn.nn.FourierELU(
                self.gspace,
                out_channels,
                irreps=[(f,) for f in range(0, 5)],
                N=16,
                inplace=True,
            )
            modules.append(
                e2cnn.nn.R2Conv(
                    in_type,
                    activation.in_type,
                    kernel_size=kernel_size,
                    padding=0,
                    bias=False,
                )
            )
            modules.append(e2cnn.nn.GNormBatchNorm(activation.in_type))
            modules.append(activation)
            in_type = activation.out_type  # Update in_type for the next layer

        # Define the output layer
        out_type = e2cnn.nn.FieldType(
            self.gspace, [self.gspace.irrep(1), self.gspace.irrep(1)]
        )
        modules.append(
            e2cnn.nn.R2Conv(
                in_type, out_type, kernel_size=kernel_size, padding=0, bias=False
            )
        )

        # Combine all modules into a SequentialModule
        self.block = e2cnn.nn.SequentialModule(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input data. It should have the shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output of the network. It has the shape (batch_size, 2, 2).
        """
        x = e2cnn.nn.GeometricTensor(x, self.in_type)
        out = self.block(x)

        feature_maps = out.tensor  # Extract tensor from geometric tensor
        feature_maps = torch.mean(
            feature_maps, dim=(-1, -2)
        )  # Average over spatial dimensions
        feature_maps = feature_maps.reshape(
            feature_maps.shape[0], 2, 2
        )  # Reshape to get vector/vectors of dimension 2
        return feature_maps


# wide resnet equivariant network and utilities
class ESCNNWideBottleneck(e2cnn.nn.EquivariantModule):
    """
    This class represents a wide bottleneck layer for an Equivariant Convolutional Neural Network (Equivariant CNN).

    The layer consists of a sequence of equivariant convolutional layers, each followed by batch normalization and a ReLU activation function. The number of output channels of the convolutional layers is the same for all layers. The input is added to the output of the layer (residual connection).

    Methods:
        __init__: Initializes the ESCNNWideBottleneck instance.
        forward: Performs a forward pass through the layer.
    """

    def __init__(
        self,
        in_type: e2cnn.nn.FieldType,
        middle_type: e2cnn.nn.FieldType,
        out_type: e2cnn.nn.FieldType,
        kernel_size: int = 3,
    ):
        """
        Initializes the ESCNNWideBottleneck instance.

        Args:
            in_type (e2cnn.nn.FieldType): The type of the input field.
            middle_type (e2cnn.nn.FieldType): The type of the middle field.
            out_type (e2cnn.nn.FieldType): The type of the output field.
            kernel_size (int, optional): The size of the kernel of the convolutional layers. Defaults to 3.
        """
        super().__init__()
        self.in_type = in_type
        self.middle_type = middle_type
        self.out_type = out_type
        self.kernel_size = kernel_size

        self.conv_network = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(self.in_type, self.middle_type, 1),
            e2cnn.nn.InnerBatchNorm(self.middle_type, momentum=0.9),
            e2cnn.nn.ReLU(self.middle_type, inplace=True),
            e2cnn.nn.R2Conv(
                self.middle_type, self.out_type, kernel_size, padding=kernel_size // 2
            ),
            e2cnn.nn.InnerBatchNorm(self.out_type, momentum=0.9),
            e2cnn.nn.ReLU(self.out_type, inplace=True),
            e2cnn.nn.R2Conv(self.out_type, self.in_type, 1),
        )

    def forward(self, x: e2cnn.nn.GeometricTensor) -> e2cnn.nn.GeometricTensor:
        """
        Performs a forward pass through the layer.

        Args:
            x (e2cnn.nn.GeometricTensor): The input data.

        Returns:
            e2cnn.nn.GeometricTensor: The output of the layer. The input is added to the output (residual connection).
        """
        out = self.conv_network(x)
        out += x
        return out

    def evaluate_output_shape(self, input_shape: Tuple[int]) -> Tuple[int]:
        """
        Compute the shape the output tensor which would be generated by this module when a tensor with shape ``input_shape`` is provided as input.

        Args:
            input_shape (tuple): shape of the input tensor

        Returns:
            shape of the output tensor

        """
        return self.forward(input_shape).tensor.shape


class ESCNNWideBasic(e2cnn.nn.EquivariantModule):
    """
    This class represents a wide basic layer for an Equivariant Convolutional Neural Network (Equivariant CNN).

    The layer consists of a sequence of equivariant convolutional layers, each followed by batch normalization and a ReLU activation function. The number of output channels of the convolutional layers is the same for all layers. The input is added to the output of the layer (residual connection).

    Methods:
        __init__: Initializes the ESCNNWideBasic instance.
        forward: Performs a forward pass through the layer.
    """

    def __init__(
        self,
        in_type: e2cnn.nn.FieldType,
        middle_type: e2cnn.nn.FieldType,
        out_type: e2cnn.nn.FieldType,
        kernel_size: int = 3,
    ):
        """
        Initializes the ESCNNWideBasic instance.

        Args:
            in_type (e2cnn.nn.FieldType): The type of the input field.
            middle_type (e2cnn.nn.FieldType): The type of the middle field.
            out_type (e2cnn.nn.FieldType): The type of the output field.
            kernel_size (int, optional): The size of the kernel of the convolutional layers. Defaults to 3.
        """
        super().__init__()
        self.in_type = in_type
        self.middle_type = middle_type
        self.out_type = out_type
        self.kernel_size = kernel_size

        self.conv_network = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(self.in_type, self.middle_type, kernel_size),
            e2cnn.nn.InnerBatchNorm(self.middle_type, momentum=0.9),
            e2cnn.nn.ReLU(self.middle_type, inplace=True),
            e2cnn.nn.R2Conv(self.middle_type, self.out_type, kernel_size),
        )

        self.shortcut = None
        if self.in_type != self.out_type:
            self.shortcut = e2cnn.nn.SequentialModule(
                e2cnn.nn.R2Conv(self.in_type, self.out_type, 2 * kernel_size - 1),
            )

    def forward(self, x: e2cnn.nn.GeometricTensor) -> e2cnn.nn.GeometricTensor:
        """
        Performs a forward pass through the layer.

        Args:
            x (e2cnn.nn.GeometricTensor): The input data.

        Returns:
            e2cnn.nn.GeometricTensor: The output of the layer. The input is added to the output (residual connection).
        """
        out = self.conv_network(x)
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        out += shortcut
        return out

    def evaluate_output_shape(self, input_shape: Tuple[int]) -> Tuple[int]:
        """
        Compute the shape the output tensor which would be generated by this module when a tensor with shape ``input_shape`` is provided as input.

        Args:
            input_shape (tuple): shape of the input tensor

        Returns:
            shape of the output tensor

        """
        return self.forward(input_shape).tensor.shape


class ESCNNWRNEquivariantNetwork(torch.nn.Module):
    """
    This class represents a Wide Residual Network (WRN) that is equivariant under rotations or roto-reflections.

    The network consists of a sequence of equivariant convolutional layers, each followed by batch normalization and a ReLU activation function. The number of output channels of the convolutional layers is the same for all layers. The input is added to the output of the layer (residual connection).

    Methods:
        __init__: Initializes the ESCNNWRNEquivariantNetwork instance.
        forward: Performs a forward pass through the network.
    """

    def __init__(
        self,
        in_shape: tuple,
        out_channels: int = 64,
        kernel_size: int = 9,
        group_type: str = "rotation",
        num_layers: int = 12,
        num_rotations: int = 4,
    ):
        """
        Initializes the ESCNNWRNEquivariantNetwork instance.

        Args:
            in_shape (tuple): The shape of the input data. It should be a tuple of the form (in_channels, height, width).
            out_channels (int, optional): The number of output channels of the convolutional layers. Defaults to 64.
            kernel_size (int, optional): The size of the kernel of the convolutional layers. Defaults to 9.
            group_type (str, optional): The type of the group of transformations. It can be either "rotation" or "roto-reflection". Defaults to "rotation".
            num_layers (int, optional): The number of convolutional layers. Defaults to 12.
            num_rotations (int, optional): The number of discrete rotations. Defaults to 4.
        """
        super().__init__()

        self.group_type = group_type

        # The model is equivariant under discrete rotations
        if group_type == "rotation":
            self.gspace = gspaces.Rot2dOnR2(num_rotations)
        elif group_type == "roto-reflection":
            self.gspace = gspaces.FlipRot2dOnR2(num_rotations)
        else:
            raise ValueError("group_type must be rotation or roto-reflection for now.")

        # other initialization
        widen_factor = 2
        self.kernel_size = kernel_size
        self.group_type = group_type
        self.out_channels = out_channels * widen_factor

        self.num_rotations = num_rotations
        self.num_group_elements = (
            num_rotations if group_type == "rotation" else 2 * num_rotations
        )

        nstages = [
            out_channels // 4,
            out_channels // 4 * widen_factor,
            out_channels // 2 * widen_factor,
            out_channels * widen_factor,
        ]
        r1 = e2cnn.nn.FieldType(self.gspace, [self.gspace.trivial_repr] * in_shape[0])
        r2 = e2cnn.nn.FieldType(self.gspace, [self.gspace.regular_repr] * nstages[0])
        r3 = e2cnn.nn.FieldType(self.gspace, [self.gspace.regular_repr] * nstages[1])
        r4 = e2cnn.nn.FieldType(self.gspace, [self.gspace.regular_repr] * nstages[2])
        r5 = e2cnn.nn.FieldType(self.gspace, [self.gspace.regular_repr] * nstages[3])

        self.in_type = r1
        self.out_type = r5

        modules = [
            e2cnn.nn.R2Conv(r1, r2, kernel_size),
            e2cnn.nn.InnerBatchNorm(r2, momentum=0.9),
            e2cnn.nn.ReLU(r2, inplace=True),
        ]

        rs = (
            [r2] * (num_layers // 3)
            + [r3] * (num_layers // 3)
            + [r4] * (num_layers // 3)
        )
        repetitions = num_layers // 3
        for ridx in range(num_layers - 1):
            if ridx % repetitions == repetitions - 1:
                modules.append(
                    ESCNNWideBasic(rs[ridx], rs[ridx + 1], rs[ridx + 1], kernel_size),
                )
                modules.append(
                    e2cnn.nn.InnerBatchNorm(rs[ridx + 1], momentum=0.9),
                )
                modules.append(
                    e2cnn.nn.ReLU(rs[ridx + 1], inplace=True),
                )
            else:
                modules.append(
                    ESCNNWideBottleneck(
                        rs[ridx], rs[ridx + 1], rs[ridx + 1], kernel_size
                    ),
                )
                modules.append(
                    e2cnn.nn.InnerBatchNorm(rs[ridx + 1], momentum=0.9),
                )
                modules.append(
                    e2cnn.nn.ReLU(rs[ridx + 1], inplace=True),
                )

        modules.append(
            e2cnn.nn.R2Conv(r4, r5, kernel_size),
        )

        self.eqv_network = e2cnn.nn.SequentialModule(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input data. It should have the shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output of the network. It has the shape (batch_size, group_size).
        """
        # x = torch.stack(x)
        x = e2cnn.nn.GeometricTensor(x, self.in_type)
        out = self.eqv_network(x)

        feature_map = out.tensor
        feature_map = feature_map.reshape(
            feature_map.shape[0],
            feature_map.shape[1] // self.num_group_elements,
            self.num_group_elements,
            feature_map.shape[-2],
            feature_map.shape[-1],
        )
        feature_fibres = torch.mean(feature_map, dim=(1, 3, 4))

        return feature_fibres
