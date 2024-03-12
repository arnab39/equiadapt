import math

import kornia as K
import torch
import torch.nn as nn
from .custom_group_equivariant_layers import RotationEquivariantConvLift, RotationEquivariantConv, RotoReflectionEquivariantConvLift, RotoReflectionEquivariantConv



class CustomEquivariantNetwork(nn.Module):
    def __init__(self,
                 in_shape,
                 out_channels,
                 kernel_size,
                 group_type='rotation',
                 num_rotations=4,
                 num_layers=1,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        if group_type == 'rotation':
            layer_list = [RotationEquivariantConvLift(in_shape[0], out_channels, kernel_size, num_rotations, device=device)]
            for i in range(num_layers - 1):
                layer_list.append(nn.ReLU())
                layer_list.append(RotationEquivariantConv(out_channels, out_channels, 1, num_rotations, device=device))
            self.eqv_network = nn.Sequential(*layer_list)
        elif group_type == 'roto-reflection':
            layer_list = [RotoReflectionEquivariantConvLift(in_shape[0], out_channels, kernel_size, num_rotations, device=device)]
            for i in range(num_layers - 1):
                layer_list.append(nn.ReLU())
                layer_list.append(RotoReflectionEquivariantConv(out_channels, out_channels, 1, num_rotations, device=device))
            self.eqv_network = nn.Sequential(*layer_list)
        else:
            raise ValueError('group_type must be rotation or roto-reflection for now.')

    def forward(self, x):
        """
        x shape: (batch_size, in_channels, height, width)
        :return: (batch_size, group_size)
        """
        feature_map = self.eqv_network(x)
        group_activatiobs = torch.mean(feature_map, dim=(1, 3, 4))

        return group_activatiobs
