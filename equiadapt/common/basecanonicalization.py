from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import torch

# Base skeleton for the canonicalization class
# DiscreteGroupCanonicalization and ContinuousGroupCanonicalization 
# will inherit from this class

class BaseCanonicalization(torch.nn.Module):
    def __init__(self, canonicalization_network: torch.nn.Module):
        super().__init__()
        self.canonicalization_network = canonicalization_network
        self.canonicalization_info_dict: Dict[str, torch.Tensor] = {}
       
    def forward(self, x: torch.Tensor, targets: List = None, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Forward method for the canonicalization which takes the input data and
        returns the canonicalized version of the data
        
        Args:
            x: input data
            **kwargs: additional arguments
        
        Returns:
            canonicalized_x: canonicalized version of the input data
        """
        
        return self.canonicalize(x, targets, **kwargs)

    
    def canonicalize(self, x: torch.Tensor, targets: List = None, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        This method takes an input data and 
        returns its canonicalized version and
        a dictionary containing the information
        about the canonicalization
        """
        raise NotImplementedError()
    

    def invert_canonicalization(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        This method takes the output of the canonicalized data 
        and returns the output for the original data orientation
        """
        raise NotImplementedError()
    

class IdentityCanonicalization(BaseCanonicalization):
    def __init__(self, canonicalization_network: torch.nn.Module = torch.nn.Identity()):
        super().__init__(canonicalization_network)
    
    def canonicalize(self, x: torch.Tensor, targets: List = None, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        if targets:
            return x, targets
        return x
    
    def invert_canonicalization(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return x
    
    def get_prior_regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)
    
    def get_identity_metric(self) -> torch.Tensor:
        return torch.tensor(1.0)
 
class DiscreteGroupCanonicalization(BaseCanonicalization):
    def __init__(self, 
                 canonicalization_network: torch.nn.Module, 
                 beta: float = 1.0,
                 gradient_trick: str = 'straight_through'):
        super().__init__(canonicalization_network)
        self.beta = beta
        self.gradient_trick = gradient_trick
    
    def groupactivations_to_groupelementonehot(self, group_activations: torch.Tensor) -> torch.Tensor:
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
    
    def canonicalize(self, x: torch.Tensor, targets: List = None, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        This method takes an input data and 
        returns its canonicalized version and
        a dictionary containing the information
        about the canonicalization
        """
        raise NotImplementedError()
    

    def invert_canonicalization(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        This method takes the output of the canonicalized data 
        and returns the output for the original data orientation
        """
        raise NotImplementedError()
    
    
    def get_prior_regularization_loss(self) -> torch.Tensor:
        group_activations = self.canonicalization_info_dict['group_activations']
        dataset_prior = torch.zeros((group_activations.shape[0],), dtype=torch.long).to(self.device)
        return torch.nn.CrossEntropyLoss()(group_activations, dataset_prior)
          
    
    def get_identity_metric(self) -> torch.Tensor:
        group_activations = self.canonicalization_info_dict['group_activations']
        return (group_activations.argmax(dim=-1) == 0).float().mean()


    
class ContinuousGroupCanonicalization(BaseCanonicalization):
    def __init__(self, 
                 canonicalization_network: torch.nn.Module, 
                 beta: float = 1.0):
        super().__init__(canonicalization_network)
        self.beta = beta
    
    def canonicalizationnetworkout_to_groupelement(self, group_activations: torch.Tensor) -> torch.Tensor:
        """
        This method takes the  as input and
        returns the group element in a differentiable manner
        
        Args:
            group_activations: activations for each group element
            
        Returns:
            group_element: group element
        """
        raise NotImplementedError()
    
    def canonicalize(self, x: torch.Tensor, targets: List = None, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        This method takes an input data and 
        returns its canonicalized version and
        a dictionary containing the information
        about the canonicalization
        """
        raise NotImplementedError()
    

    def invert_canonicalization(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        This method takes the output of the canonicalized data 
        and returns the output for the original data orientation
        """
        raise NotImplementedError()
    
    
    def get_prior_regularization_loss(self) -> torch.Tensor:
        group_elements_rep = self.canonicalization_info_dict['group_element_matrix_representation']   # shape: (batch_size, group_rep_dim, group_rep_dim)
        # Set the dataset prior to identity matrix of size group_rep_dim and repeat it for batch_size
        dataset_prior = torch.eye(group_elements_rep.shape[-1]).repeat(
            group_elements_rep.shape[0], 1, 1).to(self.device)
        return torch.nn.MSELoss()(group_elements_rep, dataset_prior)
    
    def get_identity_metric(self) -> torch.Tensor:
        group_elements_rep = self.canonicalization_info_dict['group_element_matrix_representation']
        identity_element = torch.eye(group_elements_rep.shape[-1]).repeat(
            group_elements_rep.shape[0], 1, 1).to(self.device)
        return 1.0 - torch.nn.functional.mse_loss(group_elements_rep, identity_element).mean()
    
    