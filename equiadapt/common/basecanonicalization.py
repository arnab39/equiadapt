import torch
from abc import ABC, abstractmethod

class BaseCanonicalization(torch.nn.Module):
    def __init__(self, canonicalization_network: torch.nn.Module):
        super().__init__()
        self.canonicalization_network = canonicalization_network
        self.canonicalization_info_dict = {}
       
    def forward(self, x, **kwargs):
        """
        Forward method for the canonicalization which takes the input data and
        returns the canonicalized version of the data
        
        Args:
            x: input data
            **kwargs: additional arguments
        
        Returns:
            canonicalized_x: canonicalized version of the input data
        """
        
        canonicalized_x, canonicalization_info_dict = self.canonicalize(x, **kwargs)
        
        self.canonicalization_info_dict = canonicalization_info_dict
        
        return canonicalized_x
    
    def canonicalize(self, x, **kwargs):
        """
        This method takes an input data and 
        returns its canonicalized version
        """
        raise NotImplementedError()
    

    def invert_canonicalization(self, x, **kwargs):
        """
        This method takes the output of the canonicalized data 
        and returns the output for the original data orientation
        """
        raise NotImplementedError()
    
    def add_prior_regularizer(self, loss: torch.Tensor):
        return loss + self.get_prior_regularization_loss()
    
    def get_prior_regularization_loss(self):
        """
        This method returns the prior regularization loss
        """
        raise NotImplementedError()
    

# Idea for the user interface:

# 1. The user creates a canonicalization network or uses our provided networks
#    and a wrap it using equiadapt wrappers.
# example:  canonicalization_network = ESCNNEquivariantNetwork(in_shape, out_channels, kernel_size, group_type='rotation', num_rotations=4, num_layers=3)
#           canonicalizer = GroupEquivariantImageCanonicalization(canonicalization_network, beta=1.0)
#
# 
# 2. The user uses this wrapper with their code to canonicalize the input data
#    example: model = ResNet18()
#             x_canonized = canonicalizer(x)
#             model_out = model(x_canonized)

# 3. The user can also invert the canonicalization for equivariance or not do anything for invariance
#   example: model_out = canonicalizer.invert_canonicalization(model_out)

# 4. The user creates a loss function and a wrapper for it.
#             loss = criterion(model_out, y)
#             loss = canonicalizer.add_prior_regularizer(loss)
#             loss.backward()
    
    