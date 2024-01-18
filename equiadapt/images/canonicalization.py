import torch
import kornia as K
from equiadapt.images.utils import roll_by_gather

class GroupEquivariantImageCanonicalization(torch.nn.Module):
    def __init__(self, equivariant_network, beta=1.0):
        super().__init__()
        self.equivariant_network = equivariant_network
        self.beta = beta
        
    def groupactivations_to_groupelement(self, group_activations):
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
        
        
    def forward(self, x):
        """
        This method takes an image as input and 
        returns the canonicalized image and information dictionary which
        contains the activations of each group element.
        
        x shape: (batch_size, in_channels, height, width)
        :return: (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        x_cropped = self.crop_canonization(x)
        # resize for ImageNet
        if self.num_classes == 1000:
            x_cropped = self.resize(x_cropped)
        fibres_activations = self.canonization_network(x_cropped)
        
        if self.group_type == 'rotation':
            angles = self.fibres_to_group(fibres_activations)
            group = [angles]
            
            x = self.pad(x)
            x = K.geometry.rotate(x, -angles)
            x = self.crop(x)
            
        elif self.group_type == 'roto-reflection':
            angles, reflect_indicator = self.fibres_to_group(fibres_activations)
            group = [angles, reflect_indicator]
            x_reflected = K.geometry.hflip(x)
            reflect_indicator = reflect_indicator[:,None,None,None]
            x = (1 - reflect_indicator) * x + reflect_indicator * x_reflected

            x = self.pad(x)
            x = K.geometry.rotate(x, -angles)
            x = self.crop(x)
            
        return x_canonized, group, fibres_activations
    
    
    
        x_canonized, group, fibres_activations = self.get_canonized_images(x)
        # NOTE: specifically for ImageNet hard-code the training to not pass through prediction network
        if self.num_classes == 1000 and self.training:
            return None, fibres_activations, x_canonized, group
        reps = self.base_encoder(x_canonized)
        if self.mode == 'invariance':
            reps = reps.reshape(batch_size, -1)
            return self.predictor(reps), fibres_activations, x_canonized, group
        elif self.mode == 'equivariance':
            assert len(reps.shape) == 4 and reps.shape[1] % self.num_group == 0
            angles = group[0]
            if self.group_type == 'rotation':
                reps = K.geometry.rotate(reps, angles)
                reps = reps.reshape(batch_size, reps.shape[1] // self.num_group, self.num_group, reps.shape[2], reps.shape[3])
                shift = angles / 360. * reps.shape[2]
                reps = roll_by_gather(reps, shift).reshape(batch_size, -1, reps.shape[3], reps.shape[4])
                return self.predictor(reps.reshape(batch_size, -1))
            elif self.group_type == 'roto-reflection':
                reflect_indicator = group[1]
                reps = reps.reshape(batch_size, reps.shape[1] // self.num_group, self.num_group, reps.shape[2], reps.shape[3])
                reps = reps * reflect_indicator[:,None,None,None,None]
                # TODO: implement roto-reflection equivariance
                raise NotImplementedError