import math
import torch
import kornia as K
from equiadapt.common.basecanonicalization import BaseCanonicalization
from equiadapt.images.utils import roll_by_gather
from torchvision import transforms
from torch.nn import functional as F


class ContrastiveOptImageCanonicalization(BaseCanonicalization):
    def __init__(self, 
                 canonicalization_network: torch.nn.Module, 
                 canonicalization_hyperparams: dict,
                 in_shape: tuple
                 ):
        super().__init__(canonicalization_network)
        self.beta = canonicalization_hyperparams.beta
        self.group_type = canonicalization_hyperparams.group_type
        self.num_rotations = canonicalization_hyperparams.num_rotations
        self.num_group = self.num_rotations if self.group_type == 'rotation' else 2 * self.num_rotations
        self.out_vector_size = canonicalization_network.out_vector_size
        
        if in_shape[0] == 1:
            # using identity transformation for grayscale images
            self.pad = torch.nn.Identity()
            self.crop = torch.nn.Identity()
            self.crop_canonization = torch.nn.Identity()
        else:
            # padding and cropping
            self.pad = transforms.Pad(math.ceil(in_shape[-2] * 0.4), padding_mode='edge')
            self.crop = transforms.CenterCrop((in_shape[-2], in_shape[-1]))
            self.crop_canonization = transforms.CenterCrop((
                math.ceil(in_shape[-2] * canonicalization_hyperparams.input_crop_ratio), 
                math.ceil(in_shape[-1] * canonicalization_hyperparams.input_crop_ratio)
            ))
        
        self.reference_vector = torch.nn.Parameter(torch.randn(1, self.out_vector_size), requires_grad=False)
        
    def group_augment(self, x : torch.Tensor):
        x_augmented_list = []
        if self.group_type == 'rotation':
            degrees = torch.linspace(0, 360, self.num_rotations + 1)[:-1]
            for rot, degree in enumerate(degrees):
                x_rot = self.pad(x)
                x_rot = transforms.functional.rotate(x, int(degree))
                x_rot = self.crop(x_rot)
                x_augmented_list.append(x_rot)
                
        elif self.group_type == 'roto-reflection':
            degrees = torch.linspace(0, 360, self.num_rotations + 1)[:-1]
            for rot, degree in enumerate(degrees):
                x_rot = self.pad(x)
                x_rot = transforms.functional.rotate(x_rot, int(degree))
                x_rot = transforms.functional.hflip(x_rot)
                x_rot = self.crop(x_rot)
                x_augmented_list.append(x_rot)
        return torch.cat(x_augmented_list, dim=0)
    
    
    def groupactivations_to_groupelement(self, group_activations : torch.Tensor):
        """
        This method takes the activations for each group element as input and
        returns the group element
        
        Args:
            group_activations: activations for each group element
            
        Returns:
            group_element: group element
        """
        self.device = group_activations.device
        group_activations_one_hot = torch.nn.functional.one_hot(torch.argmax(group_activations, dim=-1), self.num_group).float()
        group_activations_soft = torch.nn.functional.softmax(self.beta * group_activations, dim=-1)
        angles = torch.linspace(0., 360., self.num_rotations+1)[:self.num_rotations].to(self.device)
        group_elements = torch.cat([angles, angles], dim=0) if self.group_type == 'roto-reflection' else angles
        if self.training:
            group_element = torch.sum((group_activations_one_hot + group_activations_soft - group_activations_soft.detach()) * group_elements, dim=-1)
        else:
            group_element = torch.sum(group_activations_one_hot * group_elements, dim=-1)

        if self.group_type == 'roto-reflection':
            reflect_one_hot = torch.cat(
                [torch.zeros(self.num_rotations), torch.ones(self.num_rotations)]
                , dim=0).to(self.device)
            if self.training:
                reflect_indicator = torch.sum((group_activations_one_hot + group_activations_soft - group_activations_soft.detach())
                                              * reflect_one_hot, dim=-1)
            else:
                reflect_indicator = torch.sum(group_activations_one_hot * reflect_one_hot, dim=-1)
            return group_element, reflect_indicator
        else:
            return group_element
        
        
    def canonicalize(self, x : torch.Tensor):
        """
        This method takes an image as input and 
        returns the canonicalized image 
        """
        
        x_cropped = self.crop_canonization(x)
        
        x_augmented = self.group_augment(x_cropped)                               # size (batch_size * group_size, in_channels, height, width)
        vector_out = self.canonicalization_network(x_augmented)           # size (batch_size * group_size, reference_vector_size)
        scalar_out = F.cosine_similarity(
            self.reference_vector.repeat(vector_out.shape[0], 1), 
            vector_out
        )                                                                 # size (batch_size * group_size, 1)
        group_activations = scalar_out.reshape(self.num_group, -1).T      # size (batch_size, group_size)
        
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
            
        canonicalization_info_dict = {
            'group': group, 'group_activations': group_activations,
            'vector_out': vector_out
        }
        
        return x, canonicalization_info_dict
    
    def invert_canonicalization(self, 
                                x_canonicalized_out : torch.Tensor, 
                                induced_rep_type : str = 'regular'):
        """
        This method takes the output of canonicalized image as input and
        returns output of the original image
        """
        assert len(x_canonicalized_out.shape) == 4
        batch_size, C, H, W = x_canonicalized_out.shape
        if induced_rep_type == 'regular':
            assert x_canonicalized_out.shape[1] % self.num_group == 0
            angles = self.canonicalization_info_dict['group'][0]
            
            if self.group_type == 'rotation':
                x_out = K.geometry.rotate(x_canonicalized_out, angles)
                x_out = x_out.reshape(batch_size, C // self.num_group, self.num_group, H, W)
                shift = angles / 360. * x_out.shape[2]
                x_out = roll_by_gather(x_out, shift)
                x_out = x_out.reshape(batch_size, -1, H, W)
                return x_out
            
            elif self.group_type == 'roto-reflection':             
                reflect_indicator = self.canonicalization_info_dict['group'][1]
                x_out = K.geometry.rotate(x_canonicalized_out, angles)
                x_out_reflected = K.geometry.hflip(x_out)
                x_out = x_out * reflect_indicator[:,None,None,None,None] + \
                    x_out_reflected * (1 - reflect_indicator[:,None,None,None,None])
                x_out = x_out.reshape(batch_size, C // self.num_group, self.num_group, H, W)
                shift = angles / 360. * self.num_rotations
                # shift the first half of the num_groups chunk by shift and 
                # the second half by -shift
                x_out = torch.cat([
                    roll_by_gather(x_out[:,:,:self.num_rotations], shift), 
                    roll_by_gather(x_out[:,:,self.num_rotations:], -shift)
                ], dim=2)
                x_out = x_out.reshape(batch_size, -1, H, W)  
                return x_out
            
        elif induced_rep_type == 'scalar':
            angles = self.canonicalization_info_dict['group'][0]
            
            if self.group_type == 'rotation':
                x_out = K.geometry.rotate(x_canonicalized_out, angles)
                return x_out
            
            elif self.group_type == 'roto-reflection':
                reflect_indicator = self.canonicalization_info_dict['group'][1]                
                x_out = K.geometry.rotate(x_canonicalized_out, angles)
                x_out_reflected = K.geometry.hflip(x_out)
                x_out = x_out * reflect_indicator[:,None,None,None,None] +\
                    x_out_reflected * (1 - reflect_indicator[:,None,None,None,None])
                return x_out
        else:
            raise ValueError('induced_rep_type must be regular, scalar or vector')
        
    def get_prior_regularization_loss(self):
        group_activations = self.canonicalization_info_dict['group_activations']
        dataset_prior = torch.zeros((group_activations.shape[0],), dtype=torch.long).to(self.device)
        return torch.nn.CrossEntropyLoss()(group_activations, dataset_prior)
        
    
    def get_identity_metric(self):
        group_activations = self.canonicalization_info_dict['group_activations']
        return (group_activations.argmax(dim=-1) == 0).float().mean()
    
    def get_group_contrast_loss(self):
        vectors = self.canonicalization_info_dict['vector_out']
        vectors = vectors.reshape(self.num_group, -1, self.out_vector_size).permute((1, 0, 2)) # (batch_size, group_size, vector_out_size)
        distances = vectors @ vectors.permute((0, 2, 1))
        mask = 1.0 - torch.eye(self.num_group).to(self.device) # (group_size, group_size)
        return torch.abs(distances * mask).sum()
        