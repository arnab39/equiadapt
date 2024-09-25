from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from equiadapt.common.basecanonicalization import DiscreteGroupCanonicalization
from omegaconf import DictConfig
from equiadapt.pointcloud.canonicalization.utils import get_rotations
from torch.nn import functional as F

# Lookup table mapping discretization types to the number of rotations
DISCRETIZATION_SOLIDS = {
            'tetrahedron': 12,
            'cube': 24,
            'octahedron': 24,
            'icosahedron': 60,
            'dodecahedron': 60
        }

class DiscreteGroupPointcloudCanonicalization(DiscreteGroupCanonicalization):
    """
    This class represents a discrete group point cloud canonicalization model.

    The model is designed to be equivariant under a discrete group of transformations,
    specifically rotations in SO(3). It transforms input point clouds into a canonical form
    using the group transformations.

    Methods:
        __init__: Initializes the DiscreteGroupPointcloudCanonicalization instance.
        group_augment: Augments the input point clouds by applying group transformations (rotations).
        get_groupelement: Maps the input point cloud to a group element.
        get_group_activations: Gets the group activations for the input point clouds.
        transformations_before_canonicalization_network_forward: Applies transformations to the point clouds before processing.
        canonicalize: Canonicalizes the input point clouds.
        invert_canonicalization: Inverts the canonicalization of the output of the canonicalized point cloud.
    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        canonicalization_hyperparams: DictConfig,
    ):
        """
        Initializes the DiscreteGroupPointcloudCanonicalization instance.

        Args:
            canonicalization_network (torch.nn.Module): The canonicalization network.
            canonicalization_hyperparams (DictConfig): The hyperparameters for the canonicalization process.
        """
        super().__init__(canonicalization_network)
        self.beta = canonicalization_hyperparams.beta
        self.discretization_type = canonicalization_hyperparams.group_discretization_type  # e.g., 'icosahedron'

        # Obtain the rotation matrices to discretize SO(3)
        self.rotations = get_rotations(so3_discretization_type=self.discretization_type)
    
        self.num_group = self.rotations.shape[0]  # Update num_group based on actual number of rotations

        assert self.discretization_type in DISCRETIZATION_SOLIDS.keys()
        
        # Check that the number of discrete rotations matches for the right solid
        assert self.num_group == DISCRETIZATION_SOLIDS.get(self.discretization_type, self.num_group)

        self.group_info_dict = {
            "num_group": self.num_group,
            "rotations": self.rotations
        }

    def group_augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Augment the input point clouds by applying group transformations (rotations).

        Args:
            x (torch.Tensor): The input point clouds of shape (batch_size, num_points, 3).

        Returns:
            torch.Tensor: The augmented point clouds of shape (batch_size * num_group, num_points, 3).
        """
        num_points = x.shape[1]

        # Expand x to shape (batch_size, num_group, num_points, 3)
        x_expanded = x.unsqueeze(1).expand(-1, self.num_group, -1, -1)  # (batch_size, num_group, num_points, 3)

        # Expand rotations to shape (1, num_group, 3, 3)
        rotations_expanded = self.rotations.unsqueeze(0)  # (1, num_group, 3, 3)

        # Transpose rotations for correct matrix multiplication
        rotations_transposed = rotations_expanded.transpose(-1, -2)  # (1, num_group, 3, 3)
        

        # Perform batched matrix multiplicationequiadapt
        # x_expanded: (batch_size, num_group, num_points, 3)
        # rotations_transposed: (1, num_group, 3, 3)
        # Resulting x_rotated: (batch_size, num_group, num_points, 3)
        x_rotated = torch.matmul(x_expanded, rotations_transposed)

        # Reshape to (batch_size * num_group, num_points, 3)
        x_rotated = x_rotated.reshape(-1, num_points, 3)

        return x_rotated


    def groupactivations_to_groupelement(self, group_activations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Converts group activations to group elements using one-hot encoding.

        Args:
            group_activations (torch.Tensor): The group activations of shape (batch_size, num_group).

        Returns:
            Dict[str, torch.Tensor]: The group elements, specifically the one-hot encodings.
        """
        # Convert group activations to one-hot encoding of group element
        group_elements_one_hot = self.groupactivations_to_groupelementonehot(group_activations)
        group_element_dict = {'group_elements_one_hot': group_elements_one_hot}
        
        return group_element_dict
    
    def get_groupelement(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            Dict[str, torch.Tensor]: _description_
        """
        group_activations = self.get_group_activations(x)
        group_element_dict = self.groupactivations_to_groupelement(group_activations)
        
        if not hasattr(self, 'canonicalization_info_dict'):
            self.canonicalization_info_dict = {}
            
        # Store in canonicalization_info_dict
        self.canonicalization_info_dict["group_element"] = group_element_dict
        self.canonicalization_info_dict["group_activations"] = group_activations
        
        return group_element_dict

    def canonicalize(
        self, x: torch.Tensor, targets: Optional[List] = None, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Canonicalizes the input point clouds in a differentiable manner.

        Args:
            x (torch.Tensor): The input point clouds of shape (batch_size, 3, num_points).
            targets (Optional[List], optional): The targets. Defaults to None.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List]]: The canonicalized point clouds, and optionally the targets.
        """
        self.rotations = self.rotations.to(x.device)
        x = x.transpose(1,2)
        group_element_dict = self.get_groupelement(x)
        group_elements_one_hot = group_element_dict['group_elements_one_hot']  # Shape (batch_size, num_group)
        batch_size = x.shape[0]

        # Flatten rotations to shape (num_group, 9)
        rotations_flat = self.rotations.view(self.num_group, -1)  # (num_group, 9)

        # Compute weighted sum of rotations
        Rot_flat = torch.matmul(group_elements_one_hot, rotations_flat)  # (batch_size, 9)

        # Reshape to (batch_size, 3, 3)
        Rot = Rot_flat.view(batch_size, 3, 3)  # (batch_size, 3, 3)

        # Apply the rotation matrices to the point clouds
        x_canonical = torch.matmul(x, Rot.transpose(1, 2))  # (batch_size, num_points, 3)
        
        x_canonical = x_canonical.transpose(1,2) # (batch_size, 3, num_points)

        if targets is not None:
            # Canonicalize targets if necessary
            # Implement target transformations if required
            return x_canonical, targets

        return x_canonical

    def invert_canonicalization(
        self, x_canonicalized_out: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
        Inverts the canonicalization of the output of the canonicalized point cloud.

        Args:
            x_canonicalized_out (torch.Tensor): The output of the canonicalized point cloud, shape (batch_size, num_points, feature_dim).
            **kwargs (Any): Additional keyword arguments.

        Returns:
            torch.Tensor: The output corresponding to the original point cloud orientation.
        """
        group_element_dict = self.canonicalization_info_dict["group_element"]
        group_elements_one_hot = group_element_dict['group_elements_one_hot']  # Shape (batch_size, num_group)
        batch_size = x_canonicalized_out.shape[0]

        # Compute the inverse rotations (transpose for rotation matrices)
        rotations_inv = self.rotations.transpose(1, 2)  # (num_group, 3, 3)
        rotations_inv_flat = rotations_inv.view(self.num_group, -1)  # (num_group, 9)

        # Compute weighted sum of inverse rotations
        Rot_inv_flat = torch.matmul(group_elements_one_hot, rotations_inv_flat)  # (batch_size, 9)
        Rot_inv = Rot_inv_flat.view(batch_size, 3, 3)  # (batch_size, 3, 3)

        # Apply the inverse rotation matrices to the canonicalized outputs
        x_inverted = torch.matmul(x_canonicalized_out, Rot_inv.transpose(1, 2))  # (batch_size, num_points, feature_dim)

        return x_inverted
    
    def get_prior(
        self,
        x: torch.Tensor,
        model: torch.nn.Module,
        targets: torch.Tensor,
        metric_function: torch.nn.Module,
        tau: float = 1.0,
    ) -> torch.Tensor:
        """
        Get the prior for the input images.

        Args:
            x (torch.Tensor): The input images. shape = (batch_size, in_channels, height, width)
            model (torch.nn.Module): The prediction model which decides the prior.
            targets (torch.Tensor): The targets for the task. shape = eg. (batch_size, num_classes)
            metric_function (torch.nn.Module): The function to calculate the unnormalized probability masses for each group element.
            tau (float, optional): The temperature parameter. Defaults to 1.0. Decides the sharpness of the prior distribution.

        Returns:
            torch.Tensor: output prior of the model and x. shape = (batch_size, group_size)
        """
        batch_size = x.shape[0]
        with torch.no_grad():
            x_augmented = self.group_augment(x.transpose(1,2)).transpose(1,2)  # Shape (batch_size * num_group, num_points, 3)
            targets_augmented = targets.repeat(
                    self.num_group, 1
                ).flatten()  # size (group_size * batch_size)

            # Get the output of the model for the augmented images
            model_output = model(
                x_augmented
            )  # size (group_size * batch_size, num_classes)
            

            # Get the unnormalized probability masses for each group element
            unnormalized_prob_masses = (
                    metric_function(model_output, targets_augmented)
                    .reshape(self.num_group, batch_size)
                    .transpose(0, 1)
                )  # size (batch_size, group_size)

            # Get the prior for the input images
            prior = F.softmax(
                unnormalized_prob_masses / tau, dim=-1
            )  # size (batch_size, group_size)

        return prior
        


class OptimizedGroupEquivariantPointcloudCanonicalization(DiscreteGroupPointcloudCanonicalization):
    """
    This class represents an optimized (discrete) group equivariant point cloud canonicalization model.

    The model is designed to be equivariant under a discrete group of transformations, specifically rotations in SO(3).
    It optimizes the canonicalization process by learning a reference vector and minimizing losses related to group symmetries.

    Methods:
        __init__: Initializes the OptimizedGroupEquivariantPointcloudCanonicalization instance.
        get_group_activations: Gets the group activations for the input point clouds.
        get_optimization_specific_loss: Gets the loss specific to the optimization process.
        get_artifact_loss: Gets the loss specific to rotation artifacts.
    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        canonicalization_hyperparams: DictConfig,
    ):
        """
        Initializes the OptimizedGroupEquivariantPointcloudCanonicalization instance.

        Args:
            canonicalization_network (torch.nn.Module): The canonicalization network.
            canonicalization_hyperparams (DictConfig): The hyperparameters for the canonicalization process.
        """
        super().__init__(canonicalization_network, canonicalization_hyperparams)
        self.out_vector_size = canonicalization_network.out_vector_size

        self.reference_vector = torch.nn.Parameter(
            torch.randn(1, self.out_vector_size),
            requires_grad=canonicalization_hyperparams.learn_ref_vec,
        )

    def get_group_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gets the group activations for the input point clouds.

        Args:
            x (torch.Tensor): The input point clouds of shape (batch_size, num_points, 3).

        Returns:
            torch.Tensor: The group activations of shape (batch_size, num_group).
        """
        x_augmented = self.group_augment(x)  # Shape (batch_size * num_group, num_points, 3)

        # Pass the augmented point clouds through the canonicalization network
        vector_out = self.canonicalization_network(x_augmented.transpose(1,2))  # Shape (batch_size * num_group, out_vector_size)
        self.canonicalization_info_dict["vector_out"] = vector_out

        # Compute cosine similarity with the reference vector
        reference_vector = self.reference_vector.repeat(vector_out.shape[0], 1)  # Shape (batch_size * num_group, out_vector_size)
        scalar_out = F.cosine_similarity(reference_vector, vector_out, dim=1)  # Shape (batch_size * num_group,)
        group_activations = scalar_out.view(self.num_group, -1).T  # Shape (batch_size, num_group)
        return group_activations

    def get_optimization_specific_loss(self) -> torch.Tensor:
        """
        Gets the loss specific to the optimization process.

        Returns:
            torch.Tensor: The loss.
        """
        vectors = self.canonicalization_info_dict["vector_out"]  # Shape (batch_size * num_group, out_vector_size)

        # Reshape to (batch_size, num_group, out_vector_size)
        batch_size = vectors.shape[0] // self.num_group
        vectors = vectors.view(self.num_group, batch_size, -1).permute(1, 0, 2)  # (batch_size, num_group, out_vector_size)

        # Normalize the vectors
        normalized_vectors = F.normalize(vectors, p=2, dim=-1)
        # Compute pairwise dot products (cosine similarities)
        distances = torch.matmul(normalized_vectors, normalized_vectors.transpose(1, 2))  # (batch_size, num_group, num_group)
        # Create a mask to exclude diagonal elements
        mask = 1.0 - torch.eye(self.num_group).to(vectors.device)  # (num_group, num_group)
        mask = mask.unsqueeze(0)  # (1, num_group, num_group)
        # Compute the loss as the mean of off-diagonal similarities
        loss = torch.abs(distances * mask).mean()

        return loss 

