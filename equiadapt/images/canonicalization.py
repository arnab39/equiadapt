import torch
import kornia as K
from equiadapt.common.basecanonicalization import BaseCanonicalization
from equiadapt.images.utils import roll_by_gather
from torchvision import transforms
import math

class GroupEquivariantImageCanonicalization(BaseCanonicalization):
    def __init__(self, 
                 equivariant_canonicalization_network: torch.nn.Module, 
                 canonicalization_hyperparams: dict,
                 in_shape: tuple
                 ):
        super().__init__(equivariant_canonicalization_network)
        self.beta = canonicalization_hyperparams.beta
        self.group_type = equivariant_canonicalization_network.group_type
        self.num_rotations = equivariant_canonicalization_network.num_rotations
        self.num_group = self.num_rotations if self.group_type == 'rotation' else 2 * self.num_rotations
        # padding and cropping
        self.pad = transforms.Pad(math.ceil(in_shape[-2] * 0.4), padding_mode='edge')
        self.crop = transforms.CenterCrop((in_shape[-2], in_shape[-1]))
        self.crop_canonization = transforms.CenterCrop((
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
        device = group_activations.device
        group_activations_one_hot = torch.nn.functional.one_hot(torch.argmax(group_activations, dim=-1), self.num_group).float()
        group_activations_soft = torch.nn.functional.softmax(self.beta * group_activations, dim=-1)
        angles = torch.linspace(0., 360., self.num_rotations+1)[:self.num_rotations].to(device)
        group_elements = torch.cat([angles, angles], dim=0) if self.group_type == 'roto-reflection' else angles
        if self.training:
            group_element = torch.sum((group_activations_one_hot + group_activations_soft - group_activations_soft.detach()) * group_elements, dim=-1)
        else:
            group_element = torch.sum(group_activations_one_hot * group_elements, dim=-1)

        if self.group_type == 'roto-reflection':
            reflect_one_hot = torch.cat(
                [torch.zeros(self.num_rotations), torch.ones(self.num_rotations)]
                , dim=0).to(device)
            if self.training:
                reflect_indicator = torch.sum((group_activations_one_hot + group_activations_soft - group_activations_soft.detach())
                                              * reflect_one_hot, dim=-1)
            else:
                reflect_indicator = torch.sum(group_activations_one_hot * reflect_one_hot, dim=-1)
            return group_element, reflect_indicator
        else:
            return group_element
        
        
    def canonicalize(self, x):
        """
        This method takes an image as input and 
        returns the canonicalized image 
        """
        
        x_cropped = self.crop_canonization(x)

        group_activations = self.canonization_network(x_cropped)
        
        if self.group_type == 'rotation':
            angles = self.groupactivations_to_groupelement(group_activations)
            
            group = [angles]
            
            x = self.pad(x)
            x = K.geometry.rotate(x, -angles)
            x = self.crop(x)
            
        elif self.group_type == 'roto-reflection':
            angles, reflect_indicator = self.groupactivations_to_groupelement(group_activations)
            group = [angles, reflect_indicator]
            
            x_reflected = K.geometry.hflip(x)
            reflect_indicator = reflect_indicator[:,None,None,None]
            x = (1 - reflect_indicator) * x + reflect_indicator * x_reflected

            x = self.pad(x)
            x = K.geometry.rotate(x, -angles)
            x = self.crop(x)
            
        canonicalization_info_dict = {'group': group, 'group_activations': group_activations}
        
        return x, canonicalization_info_dict
    
    def invert_canonicalization(self, x_canonicalized_out, induced_rep_type='regular'):
        """
        This method takes the output of canonicalized image as input and
        returns output of the original image
        """
        batch_size = x_canonicalized_out.shape[0]
        if induced_rep_type == 'regular':
            assert len(x_canonicalized_out.shape) == 4 and x_canonicalized_out.shape[1] % self.num_group == 0
            angles = self.canonicalization_info_dict['group'][0]
            if self.group_type == 'rotation':
                x_out = K.geometry.rotate(x_canonicalized_out, angles)
                x_canonicalized_out = x_out.reshape(batch_size, x_out.shape[1] // self.num_group, self.num_group, x_out.shape[2], x_out.shape[3])
                shift = angles / 360. * x_out.shape[2]
                x_out = roll_by_gather(x_out, shift)
                x_out = x_out.reshape(batch_size, -1, x_out.shape[3], x_out.shape[4])
                return x_out
            elif self.group_type == 'roto-reflection':             
                # TODO: implement roto-reflection equivariance
                raise NotImplementedError
        elif induced_rep_type == 'scalar':
            assert len(x_canonicalized_out.shape) == 4
            angles = self.canonicalization_info_dict['group'][0]
            if self.group_type == 'rotation':
                x_out = K.geometry.rotate(x_canonicalized_out, angles)
                return x_out
            elif self.group_type == 'roto-reflection':
                reflect_indicator = self.canonicalization_info_dict['group'][1]                
                x_out = K.geometry.rotate(x_canonicalized_out, angles)
                x_out_reflected = K.geometry.hflip(x_out)
                x_out = x_out * reflect_indicator[:,None,None,None,None] + x_out_reflected * (1 - reflect_indicator[:,None,None,None,None])
                return x_out
        elif induced_rep_type == 'vector':
            raise NotImplementedError
        else:
            raise ValueError('induced_rep_type must be regular, scalar or vector')
        
    def get_prior_regularization_loss(self):
        raise NotImplementedError()