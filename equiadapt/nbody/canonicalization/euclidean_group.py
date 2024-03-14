from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from equiadapt.common.basecanonicalization import ContinuousGroupCanonicalization


class EuclideanGroupNBody(ContinuousGroupCanonicalization):
    """
    A class representing the continuous group for N-body canonicalization.

    Args:
        canonicalization_network (torch.nn.Module): The canonicalization network.
        canonicalization_hyperparams (dict): Hyperparameters for the canonicalization.

    Attributes:
        canonicalization_info_dict (dict): A dictionary containing the group element information.

    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
    ) -> None:
        super().__init__(canonicalization_network)

    def forward(
        self, x: torch.Tensor, targets: Optional[List] = None, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Forward pass of the continuous group.

        Args:
            nodes: Node attributes.
            **kwargs: Additional keyword arguments. Includes locs, edges, vel, edge_attr, and charges.

        Returns:
            The result of the canonicalization.

        """
        return self.canonicalize(x, None, **kwargs)

    def get_groupelement(
        self,
        nodes: torch.Tensor,
        loc: torch.Tensor,
        edges: torch.Tensor,
        vel: torch.Tensor,
        edge_attr: torch.Tensor,
        charges: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get the group element information.

        Args:
            nodes: Nodes data.
            loc: Location data.
            edges: Edges data.
            vel: Velocity data.
            edge_attr: Edge attributes data.
            charges: Charges data.

        Returns:
            A dictionary containing the group element information.

        """
        group_element_dict: Dict[str, torch.Tensor] = {}
        rotation_vectors, translation_vectors = self.canonicalization_network(
            nodes, loc, edges, vel, edge_attr, charges
        )
        rotation_matrix = self.modified_gram_schmidt(rotation_vectors)

        # Check whether canonicalization_info_dict is already defined
        if not hasattr(self, "canonicalization_info_dict"):
            self.canonicalization_info_dict = {}

        group_element_dict["rotation_matrix"] = rotation_matrix
        group_element_dict["translation_vectors"] = translation_vectors
        group_element_dict["rotation_matrix_inverse"] = rotation_matrix.transpose(
            1, 2
        )  # Inverse of a rotation matrix is its transpose.

        self.canonicalization_info_dict["group_element"] = group_element_dict

        return group_element_dict

    def canonicalize(
        self, x: torch.Tensor, targets: Optional[List] = None, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Canonicalize the input data.

        Args:
            nodes: Node attributes.
            targets: Target data.
            **kwargs: Additional keyword arguments. Includes locs, edges, vel, edge_attr, and charges.

        Returns:
            The canonicalized location and velocity.

        """
        self.device = x.device

        loc, edges, vel, edge_attr, charges = kwargs.values()

        group_element_dict = self.get_groupelement(
            x, loc, edges, vel, edge_attr, charges
        )
        translation_vectors = group_element_dict["translation_vectors"]
        rotation_matrix_inverse = group_element_dict["rotation_matrix_inverse"]

        # Canonicalizes coordinates by rotating node coordinates and translation vectors by inverse rotation.
        # Shape: (n_nodes * batch_size) x coord_dim.
        canonical_loc = (
            torch.bmm(loc[:, None, :], rotation_matrix_inverse).squeeze()
            - torch.bmm(
                translation_vectors[:, None, :], rotation_matrix_inverse
            ).squeeze()
        )
        # Canonicalizes velocities.
        # Shape: (n_nodes * batch_size) x vel_dim.
        canonical_vel = torch.bmm(vel[:, None, :], rotation_matrix_inverse).squeeze()

        return canonical_loc, canonical_vel

    def invert_canonicalization(
        self, x_canonicalized_out: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """This method takes as input the canonicalized output and returns the original output."""
        rotation_matrix, translation_vectors, _ = self.canonicalization_info_dict[
            "group_element"
        ].values()
        loc = (
            torch.bmm(x_canonicalized_out[:, None, :], rotation_matrix).squeeze()
            + translation_vectors
        )
        return loc

    def modified_gram_schmidt(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Apply the modified Gram-Schmidt process to the input vectors.

        Args:
            vectors: Input vectors.

        Returns:
            The orthonormalized vectors.

        """
        v1 = vectors[:, 0]
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
        v2 = vectors[:, 1] - torch.sum(vectors[:, 1] * v1, dim=1, keepdim=True) * v1
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
        v3 = vectors[:, 2] - torch.sum(vectors[:, 2] * v1, dim=1, keepdim=True) * v1
        v3 = v3 - torch.sum(v3 * v2, dim=1, keepdim=True) * v2
        v3 = v3 / torch.norm(v3, dim=1, keepdim=True)
        return torch.stack([v1, v2, v3], dim=1)
