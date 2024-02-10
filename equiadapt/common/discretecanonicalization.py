import torch

from common.basecanonicalization import BaseCanonicalization


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
        returns the group element
        
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
    
    
    def get_prior_regularization_loss(self):
        group_activations = self.canonicalization_info_dict['group_activations']
        dataset_prior = torch.zeros((group_activations.shape[0],), dtype=torch.long).to(self.device)
        return torch.nn.CrossEntropyLoss()(group_activations, dataset_prior)
          
    
    def get_identity_metric(self):
        group_activations = self.canonicalization_info_dict['group_activations']
        return (group_activations.argmax(dim=-1) == 0).float().mean()
        
        