import torch
from abc import ABC, abstractmethod


# Base skeleton for the canonicalization class
# DiscreteGroupCanonicalization and ContinuousGroupCanonicalization 
# will inherit from this class

class BaseCanonicalization(torch.nn.Module):
    def __init__(self, canonicalization_network: torch.nn.Module):
        super().__init__()
        self.canonicalization_network = canonicalization_network
        self.canonicalization_info_dict = {}
       
    def forward(self, x: torch.Tensor, **kwargs):
        """
        Forward method for the canonicalization which takes the input data and
        returns the canonicalized version of the data
        
        Args:
            x: input data
            **kwargs: additional arguments
        
        Returns:
            canonicalized_x: canonicalized version of the input data
        """
        
        return self.canonicalize(x, **kwargs)

    
    def canonicalize(self, x: torch.Tensor, **kwargs):
        """
        This method takes an input data and 
        returns its canonicalized version and
        a dictionary containing the information
        about the canonicalization
        """
        raise NotImplementedError()
    

    def invert_canonicalization(self, x: torch.Tensor, **kwargs):
        """
        This method takes the output of the canonicalized data 
        and returns the output for the original data orientation
        """
        raise NotImplementedError()
    

class IdentityCanonicalization(BaseCanonicalization):
    def __init__(self, canonicalization_network: torch.nn.Module = torch.nn.Identity()):
        super().__init__(canonicalization_network)
    
    def canonicalize(self, x: torch.Tensor, **kwargs):
        return x
    
    def invert_canonicalization(self, x: torch.Tensor, **kwargs):
        return x
    
    def get_prior_regularization_loss(self):
        return torch.tensor(0.0)
    
    def get_identity_metric(self):
        return torch.tensor(1.0)
 
class DiscreteGroupCanonicalization(BaseCanonicalization):
    def __init__(self, 
                 canonicalization_network: torch.nn.Module, 
                 beta: float = 1.0,
                 gradient_trick: str = 'straight_through'):
        super().__init__(canonicalization_network)
        self.beta = beta
        self.gradient_trick = gradient_trick
    
    def groupactivations_to_groupelementonehot(self, group_activations: torch.Tensor):
        """
        This method takes the activations for each group element as input and
        returns the group element in a differentiable manner
        
        Args:
            group_activations: activations for each group element
            
        Returns:
            group_element_onehot: one hot encoding of the group element
        """
        group_activations_one_hot = torch.nn.functional.one_hot(
            torch.argmax(group_activations, dim=-1), self.num_group).float()
        group_activations_soft = torch.nn.functional.softmax(self.beta * group_activations, dim=-1)
        if self.gradient_trick == 'straight_through':
            if self.training:            
                group_element_onehot = (group_activations_one_hot + group_activations_soft - group_activations_soft.detach()) 
            else:
                group_element_onehot = group_activations_one_hot
        elif self.gradient_trick == 'gumbel_softmax':
            group_element_onehot = torch.nn.functional.gumbel_softmax(group_activations, tau=1, hard=True)
        else:
            raise ValueError(f'Gradient trick {self.gradient_trick} not implemented')  
        
        # return the group element one hot encoding
        return group_element_onehot
    
    def canonicalize(self, x: torch.Tensor, **kwargs):
        """
        This method takes an input data and 
        returns its canonicalized version and
        a dictionary containing the information
        about the canonicalization
        """
        raise NotImplementedError()
    

    def invert_canonicalization(self, x: torch.Tensor, **kwargs):
        """
        This method takes the output of the canonicalized data 
        and returns the output for the original data orientation
        """
        raise NotImplementedError()
    
    
    def get_prior_regularization_loss(self):
        group_activations = self.canonicalization_info_dict['group_activations']
        dataset_prior = torch.zeros((group_activations.shape[0],), dtype=torch.long).to(self.device)
        return torch.nn.CrossEntropyLoss()(group_activations, dataset_prior)
          
    
    def get_identity_metric(self):
        group_activations = self.canonicalization_info_dict['group_activations']
        return (group_activations.argmax(dim=-1) == 0).float().mean()


    
class ContinuousGroupCanonicalization(BaseCanonicalization):
    def __init__(self, 
                 canonicalization_network: torch.nn.Module, 
                 beta: float = 1.0,
                 gradient_trick: str = 'straight_through'):
        super().__init__(canonicalization_network)
        self.beta = beta
        self.gradient_trick = gradient_trick
    
    def canonicalizationnetworkout_to_groupelement(self, group_activations: torch.Tensor):
        """
        This method takes the  as input and
        returns the group element in a differentiable manner
        
        Args:
            group_activations: activations for each group element
            
        Returns:
            group_element: group element
        """
        raise NotImplementedError()
    
    def canonicalize(self, x: torch.Tensor, **kwargs):
        """
        This method takes an input data and 
        returns its canonicalized version and
        a dictionary containing the information
        about the canonicalization
        """
        raise NotImplementedError()
    

    def invert_canonicalization(self, x: torch.Tensor, **kwargs):
        """
        This method takes the output of the canonicalized data 
        and returns the output for the original data orientation
        """
        raise NotImplementedError()
    
    
    def get_prior_regularization_loss(self):
        group_elements_rep = self.canonicalization_info_dict['group_element_matrix_representation']   # shape: (batch_size, group_rep_dim, group_rep_dim)
        # Set the dataset prior to identity matrix of size group_rep_dim and repeat it for batch_size
        dataset_prior = torch.eye(group_elements_rep.shape[-1]).repeat(
            group_elements_rep.shape[0], 1, 1).to(self.device)
        return torch.nn.MSELoss()(group_elements_rep, dataset_prior)
    
    def get_identity_metric(self):
        group_elements_rep = self.canonicalization_info_dict['group_element_matrix_representation']
        identity_element = torch.eye(group_elements_rep.shape[-1]).repeat(
            group_elements_rep.shape[0], 1, 1).to(self.device)
        return 1.0 - torch.nn.functional.mse_loss(group_elements_rep, identity_element).mean()
    
          
    

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
    
    