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
        self.in_shape = in_shape
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_type = group_type

        if group_type == 'rotation':
            self.gspace = gspaces.rot2dOnR2(num_rotations)
        elif group_type == 'roto-reflection':
            self.gspace = gspaces.flipRot2dOnR2(num_rotations)
        else:
            raise ValueError('group_type must be rotation or roto-reflection for now.')
        
        # If the group is roto-reflection, then the number of group elements is twice the number of rotations
        self.num_group_elements = num_rotations if group_type == 'rotation' else 2 * num_rotations

        r1 = escnn.nn.FieldType(self.gspace, [self.gspace.trivial_repr] * in_shape[0])
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