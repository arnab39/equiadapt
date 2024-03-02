import torch
import kornia as K
from equiadapt.common.basecanonicalization import ContinuousGroupCanonicalization
from equiadapt.common.utils import gram_schmidt
from torch.nn.modules import Module
from torchvision import transforms
import math
from torch.nn import functional as F
from equiadapt.common.utils import gram_schmidt

class ContinuousGroupNBody(ContinuousGroupCanonicalization):
    def __init__(self, 
                 canonicalization_network: torch.nn.Module, 
                 canonicalization_hyperparams: dict,
                 ):
        super().__init__(canonicalization_network)

    def get_groupelement(self, nodes, loc, edges, vel, edge_attr, charges):
        group_element_dict = {}
        rotation_vectors, translation_vectors = self.canonicalization_network(nodes, loc, edges, vel, edge_attr, charges)
        rotation_matrix = gram_schmidt(rotation_vectors)

                # Check whether canonicalization_info_dict is already defined
        if not hasattr(self, 'canonicalization_info_dict'):
            self.canonicalization_info_dict = {}

        group_element_dict["rotation_matrix"] = rotation_matrix
        group_element_dict["translation_vectors"] = translation_vectors
        group_element_dict["rotation_matrix_inverse"] = rotation_matrix.transpose(1, 2) # Inverse of a rotation matrix is its transpose.
        
        self.canonicalization_info_dict['group_element'] = group_element_dict
        
        return group_element_dict

    def canonicalize(self, nodes, loc, edges, vel, edge_attr, charges):
        
        self.device = nodes.device

        group_element_dict = self.get_groupelement(nodes, loc, edges, vel, edge_attr, charges)
        rotation_matrix = group_element_dict["rotation_matrix"]
        translation_vectors = group_element_dict["translation_vectors"]
        rotation_matrix_inverse = group_element_dict["rotation_matrix_inverse"]

        # Canonicalizes coordinates by rotating node coordinates and translation vectors by inverse rotation. 
        # Shape: (n_nodes * batch_size) x coord_dim. 
        canonical_loc = (
            torch.bmm(loc[:, None, :], rotation_matrix_inverse).squeeze()
            - torch.bmm(translation_vectors[:, None, :], rotation_matrix_inverse).squeeze()
        )
        # Canonicalizes velocities.
        # Shape: (n_nodes * batch_size) x vel_dim. 
        canonical_vel = torch.bmm(vel[:, None, :], rotation_matrix_inverse).squeeze()

        return canonical_loc, canonical_vel

    def invert_canonicalization(self, position_prediction: torch.Tensor):
        rotation_matrix, translation_vectors, _ = self.canonicalization_info_dict['group_element'].values()
        loc = (
            torch.bmm(position_prediction[:, None, :], rotation_matrix).squeeze() + translation_vectors
        )
        return loc