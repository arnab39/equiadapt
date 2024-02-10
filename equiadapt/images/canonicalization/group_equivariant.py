import torch
import kornia as K
from equiadapt.common.discretecanonicalization import DiscreteGroupCanonicalization
from equiadapt.images.utils import get_action_on_image_features, roll_by_gather
from torchvision import transforms
import math

class GroupEquivariantImageCanonicalization(DiscreteGroupCanonicalization):
    def __init__(self, 
                 canonicalization_network: torch.nn.Module, 
                 canonicalization_hyperparams: dict,
                 in_shape: tuple
                 ):
        super().__init__(canonicalization_network)
        self.beta = canonicalization_hyperparams.beta
        self.group_type = canonicalization_network.group_type
        self.num_rotations = canonicalization_network.num_rotations
        self.num_group = self.num_rotations if self.group_type == 'rotation' else 2 * self.num_rotations
        
        # pad and crop the input image if it is not rotated MNIST
        is_grayscale = in_shape[0] == 1
        self.pad = torch.nn.Identity() if is_grayscale else transforms.Pad(
            math.ceil(in_shape[-2] * 0.4), padding_mode='edge'
        )
        self.crop = torch.nn.Identity() if is_grayscale else transforms.CenterCrop((in_shape[-2], in_shape[-1]))
        self.crop_canonization = torch.nn.Identity() if is_grayscale else transforms.CenterCrop((
            math.ceil(in_shape[-2] * canonicalization_hyperparams.input_crop_ratio), 
            math.ceil(in_shape[-1] * canonicalization_hyperparams.input_crop_ratio)
        ))
        
    def groupactivations_to_groupelement(self, group_activations):
        """
        This method takes the activations for each group element as input and
        returns the group element
        
        Args:
            group_activations: activations for each group element
            
        Returns:
            group_element: group element
        """
        self.device = group_activations.device
        
        # convert the group activations to one hot encoding of group element
        # this conversion is differentiable and will be used to select the group element
        group_elements_one_hot = self.groupactivations_to_groupelementonehot(group_activations)
        
        angles = torch.linspace(0., 360., self.num_rotations+1)[:self.num_rotations].to(self.device)
        group_elements_rot_comp = torch.cat([angles, angles], dim=0) if self.group_type == 'roto-reflection' else angles
        
        group_element_dict = {}
        
        group_element_rot_comp = torch.sum(group_elements_one_hot * group_elements_rot_comp, dim=-1)
        group_element_dict['rotation'] = group_element_rot_comp

        if self.group_type == 'roto-reflection':
            reflect_identifier_vector = torch.cat([torch.zeros(self.num_rotations), 
                 torch.ones(self.num_rotations)], dim=0).to(self.device)
            group_element_reflect_comp = torch.sum(group_elements_one_hot * reflect_identifier_vector, dim=-1)
            group_element_dict['reflection'] = group_element_reflect_comp
        
        return group_element_dict
    
    def get_group_activations(self, x: torch.Tensor):
        """
        This method takes an image as input and 
        returns the group activations
        """
        x_cropped = self.crop_canonization(x)
        group_activations = self.canonicalization_network(x_cropped)
        return group_activations
    
        
    def canonicalize(self, x: torch.Tensor):
        """
        This method takes an image as input and 
        returns the canonicalized image 
        """
        group_activations = self.get_group_activations(x)
        group_element_dict = self.groupactivations_to_groupelement(group_activations)
        
        if 'reflection' in group_element_dict.keys():
            x_reflected = K.geometry.hflip(x)
            reflect_indicator = group_element_dict['reflection'][:,None,None,None]
            x = (1 - reflect_indicator) * x + reflect_indicator * x_reflected

        x = self.pad(x)
        x = K.geometry.rotate(x, -group_element_dict['rotation'])
        x = self.crop(x)
            
        canonicalization_info_dict = {'group_element': group_element_dict, 
                                      'group_activations': group_activations}
        
        return x, canonicalization_info_dict
    
    def invert_canonicalization(self, x_canonicalized_out, induced_rep_type='regular'):
        """
        This method takes the output of canonicalized image as input and
        returns output of the original image
        """
        return get_action_on_image_features(feature_map = x_canonicalized_out,
                                            group_info_dict = {'num_rotations': self.num_rotations,
                                                               'num_group': self.num_group},
                                            group_element_dict = self.canonicalization_info_dict['group_element'],
                                            induced_rep_type = induced_rep_type)
        
        
    