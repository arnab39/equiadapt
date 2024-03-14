# Note that for now we have only implemented canonicalizatin for rotation in the pointcloud setting.
# This is meant to be a proof of concept and we are happy to receive contribution to extend this to other group actions.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig

from equiadapt.common.basecanonicalization import ContinuousGroupCanonicalization
from equiadapt.common.utils import gram_schmidt


class ContinuousGroupPointcloudCanonicalization(ContinuousGroupCanonicalization):
    """
    This class represents a continuous group point cloud canonicalization.

    Args:
        canonicalization_network (torch.nn.Module): The canonicalization network.
        canonicalization_hyperparams (DictConfig): The hyperparameters for canonicalization.

    Attributes:
        device: The device on which the operations are performed.

    Methods:
        get_groupelement: Maps the input point cloud to the group element.
        canonicalize: Returns the canonicalized point cloud.
    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        canonicalization_hyperparams: DictConfig,
    ):
        super().__init__(canonicalization_network)

    def get_groupelement(self, x: torch.Tensor) -> dict:
        """
        This method takes the input image and maps it to the group element.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            dict: The group element.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("get_groupelement method is not implemented")

    def canonicalize(
        self, x: torch.Tensor, targets: Optional[List] = None, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        This method takes an image as input and returns the canonicalized image.

        Args:
            x (torch.Tensor): The input point cloud.
            targets (Optional[List]): The list of targets (optional).
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List]]: The canonicalized point cloud.

        """
        self.device = x.device

        # get the group element dictionary
        group_element_dict = self.get_groupelement(x)

        rotation_matrices = group_element_dict["rotation"]

        # get the inverse of the rotation matrices
        rotation_matrix_inverse = rotation_matrices.transpose(1, 2)

        # apply the inverse rotation matrices to the input point cloud
        x_canonicalized = torch.bmm(
            x.transpose(1, 2), rotation_matrix_inverse
        ).transpose(1, 2)

        return x_canonicalized


class EquivariantPointcloudCanonicalization(ContinuousGroupPointcloudCanonicalization):
    """
    This class represents the equivariant point cloud canonicalization module.

    It inherits from the ContinuousGroupPointcloudCanonicalization class.

    Args:
        canonicalization_network (torch.nn.Module): The canonicalization network module.
        canonicalization_hyperparams (DictConfig): The hyperparameters for the canonicalization.

    Attributes:
        canonicalization_network (torch.nn.Module): The canonicalization network module.
        canonicalization_hyperparams (DictConfig): The hyperparameters for the canonicalization.
        canonicalization_info_dict (dict): A dictionary to store the canonicalization information.
    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        canonicalization_hyperparams: DictConfig,
    ):
        super().__init__(canonicalization_network, canonicalization_hyperparams)

    def get_groupelement(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        This method takes the input image and maps it to the group element.

        Args:
            x (torch.Tensor): The input point cloud.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the group element.
        """
        group_element_dict = {}

        # convert the group activations to one hot encoding of group element
        # this conversion is differentiable and will be used to select the group element
        out_vectors = self.canonicalization_network(x)

        # Check whether canonicalization_info_dict is already defined
        if not hasattr(self, "canonicalization_info_dict"):
            self.canonicalization_info_dict = {}

        group_element_dict["rotation"] = gram_schmidt(out_vectors)
        self.canonicalization_info_dict["group_element_matrix_representation"] = (
            group_element_dict["rotation"]
        )

        self.canonicalization_info_dict["group_element"] = group_element_dict  # type: ignore

        return group_element_dict
