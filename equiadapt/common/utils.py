import torch

"""
This module contains utility functions and classes that are used for operations on Lie groups.

The module includes a function for the Gram-Schmidt process, which is used to orthogonalize a set of vectors. This function is implemented in a batch-wise manner, meaning it can process multiple sets of vectors at once.

The module also includes a class for parameterizing Lie groups and their representations.
This class supports several types of Lie groups, including the special orthogonal group (SO(n)),
the special Euclidean group (SE(n)), the orthogonal group (O(n)), and the Euclidean group (E(n)).
The class provides methods for generating the basis of the Lie group, as well as for computing the
group representation given a set of parameters.

Functions:
    gram_schmidt(vectors: torch.Tensor) -> torch.Tensor

Classes:
    LieParameterization
"""


def gram_schmidt(vectors: torch.Tensor) -> torch.Tensor:
    """
    Applies the Gram-Schmidt process to orthogonalize a set of three vectors in a batch-wise manner.

    Args:
        vectors (torch.Tensor): A batch of vectors of shape (batch_size, n_vectors, vector_dim),
                                where n_vectors is the number of vectors to orthogonalize (here 3).

    Returns:
        torch.Tensor: The orthogonalized vectors of the same shape as the input.

    Examples:
        >>> vectors = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        >>> result = gram_schmidt(vectors)
        >>> print(result)
        tensor([[[1.0000, 0.0000, 0.0000],
                 [0.0000, 1.0000, 0.0000],
                 [0.0000, 0.0000, 1.0000]]])
    """
    v1 = vectors[:, 0]
    v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
    v2 = vectors[:, 1] - torch.sum(vectors[:, 1] * v1, dim=1, keepdim=True) * v1
    v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
    v3 = (
        vectors[:, 2]
        - torch.sum(vectors[:, 2] * v1, dim=1, keepdim=True) * v1
        - torch.sum(vectors[:, 2] * v2, dim=1, keepdim=True) * v2
    )
    v3 = v3 / torch.norm(v3, dim=1, keepdim=True)
    return torch.stack([v1, v2, v3], dim=1)


class LieParameterization(torch.nn.Module):
    """
    A class for parameterizing Lie groups and their representations for a single block.

    Args:
        group_type (str): The type of Lie group (e.g., 'SOn', 'SEn', 'On', 'En').
        group_dim (int): The dimension of the Lie group.

    Attributes:
        group_type (str): Type of Lie group.
        group_dim (int): Dimension of the Lie group.
    """

    def __init__(self, group_type: str, group_dim: int):
        super().__init__()
        self.group_type = group_type
        self.group_dim = group_dim

    def get_son_bases(self) -> torch.Tensor:
        """
        Generates the basis of the Lie group of SOn.

        Returns:
            torch.Tensor: The son basis of shape (num_params, group_dim, group_dim).
        """
        num_son_bases = self.group_dim * (self.group_dim - 1) // 2
        son_bases = torch.zeros((num_son_bases, self.group_dim, self.group_dim))
        for counter, (i, j) in enumerate(
            [
                (i, j)
                for i in range(self.group_dim)
                for j in range(i + 1, self.group_dim)
            ]
        ):
            son_bases[counter, i, j] = 1
            son_bases[counter, j, i] = -1
        return son_bases

    def get_son_rep(self, params: torch.Tensor) -> torch.Tensor:
        """
        Computes the representation for SOn group.

        Args:
            params (torch.Tensor): Input parameters of shape (batch_size, param_dim).

        Returns:
            torch.Tensor: The representation of shape (batch_size, rep_dim, rep_dim).
        """
        son_bases = self.get_son_bases().to(params.device)
        A = torch.einsum("bs,sij->bij", params, son_bases)
        return torch.matrix_exp(A)

    def get_on_rep(
        self, params: torch.Tensor, reflect_indicators: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the representation for O(n) group, optionally including reflections.

        Args:
            params (torch.Tensor): Input parameters of shape (batch_size, param_dim).
            reflect_indicators (torch.Tensor): Indicators of whether to reflect, of shape (batch_size, 1).

        Returns:
            torch.Tensor: The representation of shape (batch_size, rep_dim, rep_dim).
        """
        son_rep = self.get_son_rep(params)

        # This is a simplified and conceptual approach; actual reflection handling
        # would need to determine how to reflect (e.g., across which axis or plane)
        # and this might not directly apply as-is.
        identity_matrix = torch.eye(self.group_dim)
        reflection_matrix = torch.diag_embed(
            torch.tensor([1] * (self.group_dim - 1) + [-1])
        )
        on_rep = torch.matmul(
            son_rep,
            reflect_indicators * reflection_matrix
            + (1 - reflect_indicators) * identity_matrix,
        )
        return on_rep

    def get_sen_rep(self, params: torch.Tensor) -> torch.Tensor:
        """
        Computes the representation for SEn group.

        Args:
            params (torch.Tensor): Input parameters of shape (batch_size, param_dim).

        Returns:
            torch.Tensor: The representation of shape (batch_size, rep_dim, rep_dim).
        """
        son_param_dim = self.group_dim * (self.group_dim - 1) // 2
        rho = torch.zeros(
            params.shape[0],
            self.group_dim + 1,
            self.group_dim + 1,
            device=params.device,
        )
        rho[:, : self.group_dim, : self.group_dim] = self.get_son_rep(
            params[:, :son_param_dim].unsqueeze(0)
        ).squeeze(0)
        rho[:, : self.group_dim, self.group_dim] = params[:, son_param_dim:]
        rho[:, self.group_dim, self.group_dim] = 1
        return rho

    def get_en_rep(
        self, params: torch.Tensor, reflect_indicators: torch.Tensor
    ) -> torch.Tensor:
        """Computes the representation for E(n) group, including both rotations and translations.

        Args:
            params (torch.Tensor): Input parameters of shape (batch_size, param_dim),
                                   where the first part corresponds to rotation/reflection parameters
                                   and the last 'n' parameters correspond to translation.

        Returns:
            torch.Tensor: The representation of shape (batch_size, rep_dim, rep_dim).
        """
        # Assuming the first part of params is for rotation/reflection and the last part is for translation
        rotation_param_dim = self.group_dim * (self.group_dim - 1) // 2
        translation_param_dim = self.group_dim

        # Separate rotation/reflection and translation parameters
        rotation_params = params[:, :rotation_param_dim]
        translation_params = params[
            :, rotation_param_dim : rotation_param_dim + translation_param_dim
        ]

        # Compute rotation/reflection representation
        rotoreflection_rep = self.get_on_rep(rotation_params, reflect_indicators)

        # Construct the E(n) representation matrix
        en_rep = torch.zeros(
            params.shape[0],
            self.group_dim + 1,
            self.group_dim + 1,
            device=params.device,
        )
        en_rep[:, : self.group_dim, : self.group_dim] = rotoreflection_rep
        en_rep[:, : self.group_dim, self.group_dim] = translation_params
        en_rep[:, self.group_dim, self.group_dim] = 1

        return en_rep

    def get_group_rep(self, params: torch.Tensor) -> torch.Tensor:
        """
        Computes the representation for the specified Lie group.

        Args:
            params (torch.Tensor): Input parameters of shape (batch_size, param_dim).

        Returns:
            torch.Tensor: The group representation of shape (batch_size, rep_dim, rep_dim).
        """
        if self.group_type == "SOn":
            return self.get_son_rep(params)
        elif self.group_type == "SEn":
            return self.get_sen_rep(params)
        elif self.group_type == "On":
            # TODO: currently assuming no reflections
            return self.get_on_rep(
                params, torch.zeros(params.shape[0], 1, device=params.device)
            )
        elif self.group_type == "En":
            return self.get_en_rep(
                params, torch.zeros(params.shape[0], 1, device=params.device)
            )
        else:
            raise ValueError(f"Unsupported group type: {self.group_type}")
