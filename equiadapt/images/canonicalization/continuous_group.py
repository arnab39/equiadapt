import torch
import kornia as K
from equiadapt.common.basecanonicalization import ContinuousGroupCanonicalization
from equiadapt.common.utils import gram_schmidt
from equiadapt.images.utils import get_action_on_image_features
from torchvision import transforms
import math
from torch.nn import functional as F

class ContinuousGroupImageCanonicalization(ContinuousGroupCanonicalization):
    def __init__(self, 
                 canonicalization_network: torch.nn.Module, 
                 canonicalization_hyperparams: dict,
                 in_shape: tuple
                 ):
        super().__init__(canonicalization_network)
        
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
        self.group_info_dict = {}
        
    def get_groupelement(self, x: torch.Tensor):
        """
        This method takes the input image and
        maps it to the group element
        
        Args:
            x: input image
            
        Returns:
            group_element: group element
        """
        raise NotImplementedError('get_groupelement method is not implemented')
    
    
    
    def canonicalize(self, x: torch.Tensor):
        """
        This method takes an image as input and 
        returns the canonicalized image 
        
        Args:
            x: input image
        
        Returns:
            x_canonicalized: canonicalized image
        """
        self.device = x.device
        group_element_dict = self.get_groupelement(x)
        rotation_matrices = group_element_dict['rotation']
        
        if 'reflection' in group_element_dict:
            reflect_indicator = group_element_dict['reflection']

            # Reflect the image conditionally
            x = reflect_indicator * x + (1 - reflect_indicator) * K.geometry.hflip(x)
        
        # Compute affine part for warp affine
        alpha, beta = rotation_matrices[:, 0, 0], rotation_matrices[:, 0, 1]
        cx, cy = x.shape[-2] // 2, x.shape[-1] // 2
        affine_part = torch.stack([(1 - alpha) * cx - beta * cy, beta * cx + (1 - alpha) * cy], dim=1)
        
        # Prepare affine matrices for warp affine, adjusting rotation matrix for Kornia compatibility
        affine_matrices = torch.cat([rotation_matrices, affine_part.unsqueeze(-1)], dim=-1)

        # Apply padding, warp affine, and then crop
        x = self.pad(x)
        x = K.geometry.warp_affine(x, affine_matrices, dsize=(x.shape[-2], x.shape[-1]))
        x = self.crop(x)

        return x

        
    def invert_canonicalization(self, x_canonicalized_out: torch.Tensor, induced_rep_type: str = 'vector'):
        """
        This method takes the output of canonicalized image as input and
        returns output of the original image
        
        """
        return get_action_on_image_features(feature_map = x_canonicalized_out,
                                            group_info_dict = self.group_info_dict,
                                            group_element_dict = self.canonicalization_info_dict['group_element'],
                                            induced_rep_type = induced_rep_type)
        


class SteerableImageCanonicalization(ContinuousGroupImageCanonicalization):
    def __init__(self, 
                 canonicalization_network: torch.nn.Module, 
                 canonicalization_hyperparams: dict,
                 in_shape: tuple
                 ):
        super().__init__(canonicalization_network,
                         canonicalization_hyperparams,
                         in_shape)
        self.group_type = canonicalization_network.group_type
    
    def get_rotation_matrix_from_vector(self, vectors: torch.Tensor):
        '''
        This method takes the input vector and returns the rotation matrix
        
        Args:
            vectors: input vector
        
        Returns:
            rotation_matrices: rotation matrices
        '''
        v1 = vectors / torch.norm(vectors, dim=1, keepdim=True)
        v2 = torch.stack([-v1[:, 1], v1[:, 0]], dim=1)
        rotation_matrices = torch.stack([v1, v2], dim=1)
        return rotation_matrices
    
    def get_groupelement(self, x: torch.Tensor):
        """
        This method takes the input image and
        maps it to the group element
        
        Args:
            x: input image
            
        Returns:
            group_element: group element
        """
        
        group_element_dict = {}
        
        # convert the group activations to one hot encoding of group element
        # this conversion is differentiable and will be used to select the group element
        out_vectors = self.canonicalization_network(x)
        
        # Check whether canonicalization_info_dict is already defined
        if not hasattr(self, 'canonicalization_info_dict'):
            self.canonicalization_info_dict = {}

        if self.group_type == 'roto-reflection':
            # Apply Gram-Schmidt to get the rotation matrices/orthogonal frame from
            # a batch of two 2D vectors
            rotation_matrices = gram_schmidt(out_vectors)         # (batch_size, 2, 2)
            
            # Store the matrix representation of the group element for regularization and identity metric
            self.canonicalization_info_dict['group_element_matrix_representation'] = rotation_matrices
        
            # Calculate the determinant to check for reflection
            determinant = rotation_matrices[:, 0, 0] * rotation_matrices[:, 1, 1] - \
                          rotation_matrices[:, 0, 1] * rotation_matrices[:, 1, 0]
            reflect_indicator = (1 - determinant[:, None, None, None]) / 2
            group_element_dict['reflection'] = reflect_indicator
            # Identify matrices with a reflection (negative determinant)
            reflection_indices = determinant < 0

            # For matrices with a reflection, adjust to remove the reflection component
            # This example assumes flipping the sign of the second column as one way to adjust
            # Note: This method of adjustment is context-dependent and may vary based on your specific requirements
            rotation_matrices[reflection_indices, :, 1] *= -1           
        else:
            # Pass the first vector to get the rotation matrix
            rotation_matrices = self.get_rotation_matrix_from_vector(out_vectors[:, 0])
            # Store the matrix representation of the group element for regularization and identity metric
            self.canonicalization_info_dict['group_element_matrix_representation'] = rotation_matrices
        
        group_element_dict['rotation'] = rotation_matrices

        self.canonicalization_info_dict['group_element'] = group_element_dict
            
        
        return group_element_dict
    

class OptimizedSteerableImageCanonicalization(ContinuousGroupImageCanonicalization):
    def __init__(self, 
                 canonicalization_network: torch.nn.Module, 
                 canonicalization_hyperparams: dict,
                 in_shape: tuple
                 ):
        super().__init__(canonicalization_network,
                         canonicalization_hyperparams,
                         in_shape)
        self.group_type = canonicalization_hyperparams.group_type
    
    def get_rotation_matrix_from_vector(self, vectors: torch.Tensor):
        '''
        This method takes the input vector and returns the rotation matrix
        
        Args:
            vectors: input vector
        
        Returns:
            rotation_matrices: rotation matrices
        '''
        v1 = vectors / torch.norm(vectors, dim=1, keepdim=True)
        v2 = torch.stack([-v1[:, 1], v1[:, 0]], dim=1)
        rotation_matrices = torch.stack([v1, v2], dim=1)
        return rotation_matrices
    
    def get_groupelement(self, x: torch.Tensor):
        """
        This method takes the input image and
        maps it to the group element
        
        Args:
            x: input image
            
        Returns:
            group_element: group element
        """
        
        group_element_dict = {}
        
        batch_size = x.shape[0]
        
        # randomly sample generate some agmentations of the input image using rotation and reflection
        # x_augmented = self.group_augment(x)    # size (batch_size * group_size, in_channels, height, width)
        
        out_vectors = self.canonicalization_network(x)
        out_vectors = out_vectors.reshape(batch_size, -1, 2)
        
        # Check whether canonicalization_info_dict is already defined
        if not hasattr(self, 'canonicalization_info_dict'):
            self.canonicalization_info_dict = {}

        if self.group_type == 'roto-reflection':
            # Apply Gram-Schmidt to get the rotation matrices/orthogonal frame from
            # a batch of two 2D vectors
            rotation_matrices = gram_schmidt(out_vectors)         # (batch_size, 2, 2)
            
            # Store the matrix representation of the group element for regularization and identity metric
            self.canonicalization_info_dict['group_element_matrix_representation'] = rotation_matrices
        
            # Calculate the determinant to check for reflection
            determinant = rotation_matrices[:, 0, 0] * rotation_matrices[:, 1, 1] - \
                          rotation_matrices[:, 0, 1] * rotation_matrices[:, 1, 0]
            reflect_indicator = (1 - determinant[:, None, None, None]) / 2
            group_element_dict['reflection'] = reflect_indicator
            # Identify matrices with a reflection (negative determinant)
            reflection_indices = determinant < 0

            # For matrices with a reflection, adjust to remove the reflection component
            # This example assumes flipping the sign of the second column as one way to adjust
            # Note: This method of adjustment is context-dependent and may vary based on your specific requirements
            rotation_matrices[reflection_indices, :, 1] *= -1           
        else:
            # Pass the first vector to get the rotation matrix
            rotation_matrices = self.get_rotation_matrix_from_vector(out_vectors[:, 0])
            # Store the matrix representation of the group element for regularization and identity metric
            self.canonicalization_info_dict['group_element_matrix_representation'] = rotation_matrices
            
        
        group_element_dict['rotation'] = rotation_matrices
        
        self.canonicalization_info_dict['group_element'] = group_element_dict
        
        return group_element_dict
    
    def get_optimization_specific_loss(self):
        raise NotImplementedError('get_optimization_specific_loss is not implemented')
    