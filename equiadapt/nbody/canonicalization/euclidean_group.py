import torch

from equiadapt.common.basecanonicalization import ContinuousGroupCanonicalization


class EuclideanGroupNBody(ContinuousGroupCanonicalization):
    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        canonicalization_hyperparams: dict,
    ):
        super().__init__(canonicalization_network)

    def forward(self, nodes, loc, edges, vel, edge_attr, charges):
        """
        This method the forward pass of the Euclidean group canonicalization.

        Args:
            nodes (Tensor): The input nodes.
            loc (Tensor): The location tensor.
            edges (Tensor): The input edges.
            vel (Tensor): The velocity tensor.
            edge_attr (Tensor): The edge attribute tensor.
            charges (Tensor): The charge tensor.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the canonicalized location tensor and the canonicalized velocity tensor.
        """
        return self.canonicalize(nodes, loc, edges, vel, edge_attr, charges)

    def get_groupelement(self, nodes, loc, edges, vel, edge_attr, charges):
        """
        This method calculates the group element for canonicalization.

        Args:
            nodes (list): List of nodes.
            loc (list): List of node locations.
            edges (list): List of edges.
            vel (list): List of velocities.
            edge_attr (list): List of edge attributes.
            charges (list): List of charges.

        Returns:
            dict: A dictionary containing the group element information, including:
                - 'rotation_matrix': The rotation matrix.
                - 'translation_vectors': The translation vectors.
                - 'rotation_matrix_inverse': The inverse of the rotation matrix.
        """
        group_element_dict = {}
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

    def canonicalize(self, nodes, loc, edges, vel, edge_attr, charges):
        """
        This method canonicalizes the given inputs by applying a Euclidean transformation on node coordinates and velocities.

        Args:
            nodes (Tensor): The nodes tensor.
            loc (Tensor): The location tensor.
            edges (Tensor): The edges tensor.
            vel (Tensor): The velocity tensor.
            edge_attr (Tensor): The edge attribute tensor.
            charges (Tensor): The charges tensor.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the canonicalized location tensor and the canonicalized velocity tensor.
        """

        self.device = nodes.device

        group_element_dict = self.get_groupelement(
            nodes, loc, edges, vel, edge_attr, charges
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

    def invert_canonicalization(self, position_prediction: torch.Tensor):
        """
        This method inverts the canonicalization transformation applied to the given position prediction.

        Args:
            position_prediction (torch.Tensor): The predicted positions to be inverted.

        Returns:
            torch.Tensor: The inverted positions after applying the inverse canonicalization transformation.
        """
        rotation_matrix, translation_vectors, _ = self.canonicalization_info_dict[
            "group_element"
        ].values()
        loc = (
            torch.bmm(position_prediction[:, None, :], rotation_matrix).squeeze()
            + translation_vectors
        )
        return loc

    def modified_gram_schmidt(self, vectors):
        """
        This method applies the modified Gram-Schmidt algorithm to orthogonalize a set of vectors.

        Args:
            vectors (torch.Tensor): The input vectors to be orthogonalized. 
                                    Shape: (batch_size, num_vectors, vector_dim)

        Returns:
            torch.Tensor: The orthogonalized vectors. 
                          Shape: (batch_size, num_vectors, vector_dim)
        """
        v1 = vectors[:, 0]
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
        v2 = vectors[:, 1] - torch.sum(vectors[:, 1] * v1, dim=1, keepdim=True) * v1
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
        v3 = vectors[:, 2] - torch.sum(vectors[:, 2] * v1, dim=1, keepdim=True) * v1
        v3 = v3 - torch.sum(v3 * v2, dim=1, keepdim=True) * v2
        v3 = v3 / torch.norm(v3, dim=1, keepdim=True)
        return torch.stack([v1, v2, v3], dim=1)
