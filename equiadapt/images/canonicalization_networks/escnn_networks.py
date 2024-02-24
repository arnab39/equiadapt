import torch
import escnn
from escnn import gspaces

class ESCNNEquivariantNetwork(torch.nn.Module):
    def __init__(self,
                 in_shape,
                 out_channels,
                 kernel_size,
                 group_type='rotation',
                 num_rotations=4,
                 num_layers=1):
        super().__init__()

        self.in_channels = in_shape[0]
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_type = group_type
        self.num_rotations = num_rotations

        if group_type == 'rotation':
            self.gspace = gspaces.rot2dOnR2(num_rotations)
        elif group_type == 'roto-reflection':
            self.gspace = gspaces.flipRot2dOnR2(num_rotations)
        else:
            raise ValueError('group_type must be rotation or roto-reflection for now.')

        # If the group is roto-reflection, then the number of group elements is twice the number of rotations
        self.num_group_elements = num_rotations if group_type == 'rotation' else 2 * num_rotations

        r1 = escnn.nn.FieldType(self.gspace, [self.gspace.trivial_repr] * self.in_channels)
        r2 = escnn.nn.FieldType(self.gspace, [self.gspace.regular_repr] * out_channels)

        self.in_type = r1
        self.out_type = r2

        self.eqv_network = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(self.in_type, self.out_type, kernel_size),
            escnn.nn.InnerBatchNorm(self.out_type, momentum=0.9),
            escnn.nn.ReLU(self.out_type, inplace=True),
            escnn.nn.PointwiseDropout(self.out_type, p=0.5),
        )
        for _ in range(num_layers - 2):
            self.eqv_network.append(escnn.nn.R2Conv(self.out_type, self.out_type, kernel_size),)
            self.eqv_network.append(escnn.nn.InnerBatchNorm(self.out_type, momentum=0.9),)
            self.eqv_network.append(escnn.nn.ReLU(self.out_type, inplace=True),)
            self.eqv_network.append(escnn.nn.PointwiseDropout(self.out_type, p=0.5),)

        self.eqv_network.append(escnn.nn.R2Conv(self.out_type, self.out_type, kernel_size),)

    def forward(self, x):
        """
        The forward takes an image as input and returns the activations of
        each group element as output.

        x shape: (batch_size, in_channels, height, width)
        :return: (batch_size, group_size)
        """
        x = escnn.nn.GeometricTensor(x, self.in_type)
        out = self.eqv_network(x)

        feature_map = out.tensor
        feature_map = feature_map.reshape(
            feature_map.shape[0], self.out_channels, self.num_group_elements,
            feature_map.shape[-2], feature_map.shape[-1]
        )

        group_activations = torch.mean(feature_map, dim=(1, 3, 4))

        return group_activations


class ESCNNSteerableNetwork(torch.nn.Module):
    def __init__(self,
                 in_shape: tuple,
                 out_channels: int,
                 kernel_size: int = 9,
                 group_type: str = 'rotation',
                 num_layers: int = 1):
        super().__init__()

        self.group_type = group_type
        assert group_type == 'rotation', 'group_type must be rotation for now.'
        # TODO: Add support for roto-reflection group

        # The model is equivariant under all planar rotations
        self.gspace = gspaces.rot2dOnR2(N=-1)

        # The input image is a scalar field, corresponding to the trivial representation
        in_type = escnn.nn.FieldType(self.gspace, in_shape[0] * [self.gspace.trivial_repr])

        # Store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # Initialize the modules list for the sequential network
        modules = []

        # Dynamically add layers based on num_layers
        for _ in range(num_layers):
            activation = escnn.nn.FourierELU(self.gspace, out_channels, irreps=[(f,) for f in range(0, 5)], N=16, inplace=True)
            modules.append(escnn.nn.R2Conv(in_type, activation.in_type, kernel_size=kernel_size, padding=0, bias=False))
            modules.append(escnn.nn.IIDBatchNorm2d(activation.in_type))
            modules.append(activation)
            in_type = activation.out_type  # Update in_type for the next layer

        # Define the output layer
        out_type = escnn.nn.FieldType(self.gspace, [self.gspace.irrep(1), self.gspace.irrep(1)])
        modules.append(escnn.nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=0, bias=False))

        # Combine all modules into a SequentialModule
        self.block = escnn.nn.SequentialModule(*modules)

    def forward(self, x : torch.Tensor):
        x = self.input_type(x)  # Wrap input images into a geometric tensor
        x = self.block(x)
        x = x.tensor  # Extract tensor from geometric tensor
        x = torch.mean(x, dim=(-1, -2))  # Average over spatial dimensions
        x = x.reshape(x.shape[0], 2, 2)  # Reshape to get vector/vectors of dimension 2
        return x


# wide resnet equivariant network and utilities
class ESCNNWideBottleneck(torch.nn.Module):
    def __init__(
        self,
        in_type,
        middle_type,
        out_type,
        kernel_size=3,
    ):
        super().__init__()
        self.in_type = in_type
        self.middle_type = middle_type
        self.out_type = out_type
        self.kernel_size = kernel_size

        self.conv_network = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(self.in_type, self.middle_type, 1),

            escnn.nn.InnerBatchNorm(self.middle_type, momentum=0.9),
            escnn.nn.ReLU(self.middle_type, inplace=True),
            escnn.nn.R2Conv(self.middle_type, self.out_type, kernel_size, padding=kernel_size//2),


            escnn.nn.InnerBatchNorm(self.out_type, momentum=0.9),
            escnn.nn.ReLU(self.out_type, inplace=True),
            escnn.nn.R2Conv(self.out_type, self.in_type, 1),
        )

    def forward(self, x):
        out = self.conv_network(x)
        out += x
        return out


class ESCNNWideBasic(torch.nn.Module):
    def __init__(
        self,
        in_type,
        middle_type,
        out_type,
        kernel_size=3,
    ):
        super().__init__()
        self.in_type = in_type
        self.middle_type = middle_type
        self.out_type = out_type
        self.kernel_size = kernel_size

        self.conv_network = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(self.in_type, self.middle_type, kernel_size),
            escnn.nn.InnerBatchNorm(self.middle_type, momentum=0.9),
            escnn.nn.ReLU(self.middle_type, inplace=True),

            escnn.nn.R2Conv(self.middle_type, self.out_type, kernel_size),
        )

        self.shortcut = None
        if self.in_type != self.out_type:
            self.shortcut = escnn.nn.SequentialModule(
                escnn.nn.R2Conv(self.in_type, self.out_type, 2*kernel_size-1),
            )

    def forward(self, x):
        out = self.conv_network(x)
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        out += shortcut
        return out

class ESCNNWRNEquivariantNetwork(torch.nn.Module):
    def __init__(self,
                 in_shape: tuple,
                 out_channels: int = 64,
                 kernel_size: int = 9,
                 group_type: str = 'rotation',
                 num_layers: int = 12,
                 num_rotations: int = 4):
        super().__init__()

        self.group_type = group_type

        # The model is equivariant under discrete rotations
        if group_type == 'rotation':
            self.gspace = gspaces.rot2dOnR2(num_rotations)
        elif group_type == 'roto-reflection':
            self.gspace = gspaces.flipRot2dOnR2(num_rotations)
        else:
            raise ValueError('group_type must be rotation or roto-reflection for now.')

        # The input image is a scalar field, corresponding to the trivial representation
        in_type = escnn.nn.FieldType(self.gspace, in_shape[0] * [self.gspace.trivial_repr])

        # Store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # other initialization
        widen_factor = 2
        self.kernel_size = kernel_size
        self.group_type = group_type
        self.out_channels = out_channels * widen_factor

        self.num_rotations = num_rotations
        self.num_group_elements = num_rotations if group_type == 'rotation' else 2 * num_rotations

        nstages = [out_channels//4, out_channels//4 * widen_factor, out_channels//2 * widen_factor, out_channels * widen_factor]
        r1 = escnn.nn.FieldType(self.gspace, [self.gspace.trivial_repr] * in_shape[0])
        r2 = escnn.nn.FieldType(self.gspace, [self.gspace.regular_repr] * nstages[0])
        r3 = escnn.nn.FieldType(self.gspace, [self.gspace.regular_repr] * nstages[1])
        r4 = escnn.nn.FieldType(self.gspace, [self.gspace.regular_repr] * nstages[2])
        r5 = escnn.nn.FieldType(self.gspace, [self.gspace.regular_repr] * nstages[3])

        self.in_type = r1
        self.out_type = r5

        self.eqv_network = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(r1, r2, kernel_size),
            escnn.nn.InnerBatchNorm(r2, momentum=0.9),
            escnn.nn.ReLU(r2, inplace=True),
        )

        rs = [r2] * (num_layers // 3) + [r3] * (num_layers // 3) + [r4] * (num_layers // 3)
        for ridx in range(num_layers - 1):
            if ridx % 4 == 3:
                self.eqv_network.append(ESCNNWideBasic(rs[ridx], rs[ridx+1], rs[ridx+1], kernel_size),)
                self.eqv_network.append(escnn.nn.InnerBatchNorm(rs[ridx+1], momentum=0.9),)
                self.eqv_network.append(escnn.nn.ReLU(rs[ridx+1], inplace=True),)
            else:
                self.eqv_network.append(ESCNNWideBottleneck(rs[ridx], rs[ridx+1], rs[ridx+1], kernel_size),)
                self.eqv_network.append(escnn.nn.InnerBatchNorm(rs[ridx+1], momentum=0.9),)
                self.eqv_network.append(escnn.nn.ReLU(rs[ridx+1], inplace=True),)

        self.eqv_network.append(escnn.nn.R2Conv(r4, r5, kernel_size),)

    def forward(self, x):
        """
        x shape: (batch_size, in_channels, height, width)
        :return: (batch_size, group_size)
        """
        # x = torch.stack(x)
        x = escnn.nn.GeometricTensor(x, self.in_type)
        out = self.eqv_network(x)

        feature_map = out.tensor
        feature_map = feature_map.reshape(feature_map.shape[0],
                                            feature_map.shape[1] // self.num_group_elements, self.num_group_elements,
                                            feature_map.shape[-2], feature_map.shape[-1])
        feature_fibres = torch.mean(feature_map, dim=(1, 3, 4))

        return feature_fibres
