import math
from typing import Any, Dict, List, Optional, Tuple, Union

import kornia as K
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from torchvision import transforms

from equiadapt.common.basecanonicalization import ContinuousGroupCanonicalization
from equiadapt.common.utils import gram_schmidt
from equiadapt.images.utils import get_action_on_image_features


class ContinuousGroupImageCanonicalization(ContinuousGroupCanonicalization):
    """
    This class represents a continuous group image canonicalization model.

    The model is designed to be equivariant under a continuous group of transformations, which can include rotations and reflections.
    Other specific continuous group image canonicalization classes can be derived from this class.

    Methods:
        __init__: Initializes the ContinuousGroupImageCanonicalization instance.
        get_rotation_matrix_from_vector: This method takes the input vector and returns the rotation matrix.
        get_groupelement: This method maps the input image to the group element.
        transformations_before_canonicalization_network_forward: Applies transformations to the input image before forwarding it through the canonicalization network.
        get_group_from_out_vectors: This method takes the output of the canonicalization network and returns the group element.
        canonicalize: This method takes an image as input and returns the canonicalized image.
        invert_canonicalization: Inverts the canonicalization process on the output of the canonicalized image.
    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        canonicalization_hyperparams: DictConfig,
        in_shape: tuple,
    ):
        """
        Initializes the ContinuousGroupImageCanonicalization instance.

        Args:
            canonicalization_network (torch.nn.Module): The canonicalization network.
            canonicalization_hyperparams (DictConfig): The hyperparameters for the canonicalization process.
            in_shape (tuple): The shape of the input images.
        """
        super().__init__(canonicalization_network)

        assert (
            len(in_shape) == 3
        ), "Input shape should be in the format (channels, height, width)"

        # pad and crop the input image if it is not rotated MNIST
        is_grayscale = in_shape[0] == 1
        self.pad = (
            torch.nn.Identity()
            if is_grayscale
            else transforms.Pad(math.ceil(in_shape[-1] * 0.5), padding_mode="edge")
        )
        self.crop = (
            torch.nn.Identity()
            if is_grayscale
            else transforms.CenterCrop((in_shape[-2], in_shape[-1]))
        )
        self.crop_canonization = (
            torch.nn.Identity()
            if is_grayscale
            else transforms.CenterCrop(
                (
                    math.ceil(
                        in_shape[-2] * canonicalization_hyperparams.input_crop_ratio
                    ),
                    math.ceil(
                        in_shape[-1] * canonicalization_hyperparams.input_crop_ratio
                    ),
                )
            )
        )
        self.resize_canonization = (
            torch.nn.Identity()
            if is_grayscale
            else transforms.Resize(size=canonicalization_hyperparams.resize_shape)
        )
        self.group_info_dict: Dict[str, Any] = {}

    def get_groupelement(self, x: torch.Tensor) -> dict:
        """
        This method takes the input image and maps it to the group element

        Args:
            x (torch.Tensor): input image

        Returns:
            dict: group element
        """
        raise NotImplementedError("get_groupelement method is not implemented")

    def transformations_before_canonicalization_network_forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies transformations to the input image before forwarding it through the canonicalization network.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The transformed image.
        """
        x = self.crop_canonization(x)
        x = self.resize_canonization(x)
        return x

    def get_group_from_out_vectors(
        self, out_vectors: torch.Tensor
    ) -> Tuple[dict, torch.Tensor]:
        """
        This method takes the output of the canonicalization network and returns the group element

        Args:
            out_vectors (torch.Tensor): output of the canonicalization network

        Returns:
            dict: group element
            torch.Tensor: group element representation
        """
        group_element_dict = {}

        if self.group_type == "roto-reflection":
            # Apply Gram-Schmidt to get the rotation matrices/orthogonal frame from
            # a batch of two 2D vectors
            rotoreflection_matrices = gram_schmidt(out_vectors)  # (batch_size, 2, 2)

            # Calculate the determinant to check for reflection
            determinant = (
                rotoreflection_matrices[:, 0, 0] * rotoreflection_matrices[:, 1, 1]
                - rotoreflection_matrices[:, 0, 1] * rotoreflection_matrices[:, 1, 0]
            )

            reflect_indicator = (1 - determinant[:, None, None, None]) / 2
            group_element_dict["reflection"] = reflect_indicator

            # Identify matrices with a reflection (negative determinant)
            reflection_indices = determinant < 0

            # For matrices with a reflection, adjust to remove the reflection component
            # This example assumes flipping the sign of the second column as one way to adjust
            # Note: This method of adjustment is context-dependent and may vary based on your specific requirements
            rotation_matrices = rotoreflection_matrices
            rotation_matrices[reflection_indices, :, 1] *= -1
        else:
            # Pass the first vector to get the rotation matrix
            rotation_matrices = self.get_rotation_matrix_from_vector(out_vectors[:, 0])

        group_element_dict["rotation"] = rotation_matrices

        return group_element_dict, (
            rotoreflection_matrices
            if self.group_type == "roto-reflection"
            else rotation_matrices
        )

    def canonicalize(
        self, x: torch.Tensor, targets: Optional[List] = None, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        This method takes an image as input and returns the canonicalized image

        Args:
            x (torch.Tensor): The input image.
            targets (Optional[List]): The targets, if any.

        Returns:
            torch.Tensor: canonicalized image
        """
        self.device = x.device

        # get the group element dictionary with keys as 'rotation' and 'reflection'
        group_element_dict = self.get_groupelement(x)

        rotation_matrices = group_element_dict["rotation"]

        # To apply inverse rotation for canonicalization
        rotation_matrices[:, [0, 1], [1, 0]] *= -1

        if "reflection" in group_element_dict:
            reflect_indicator = group_element_dict["reflection"]

            # Reflect the image conditionally
            x = (1 - reflect_indicator) * x + reflect_indicator * K.geometry.hflip(x)

        # Apply padding before canonicalization
        x = self.pad(x)

        # Compute affine part for warp affine
        alpha, beta = rotation_matrices[:, 0, 0], rotation_matrices[:, 0, 1]
        cx, cy = x.shape[-2] // 2, x.shape[-1] // 2
        affine_part = torch.stack(
            [(1 - alpha) * cx - beta * cy, beta * cx + (1 - alpha) * cy], dim=1
        )

        # Prepare affine matrices for warp affine, adjusting rotation matrix for Kornia compatibility
        affine_matrices = torch.cat(
            [rotation_matrices, affine_part.unsqueeze(-1)], dim=-1
        )

        # Apply warp affine, and then crop
        x = K.geometry.warp_affine(x, affine_matrices, dsize=(x.shape[-2], x.shape[-1]))
        x = self.crop(x)

        return x

    def invert_canonicalization(
        self, x_canonicalized_out: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
        Inverts the canonicalization process on the output of the canonicalized image.

        Args:
            x_canonicalized_out (torch.Tensor): The output of the canonicalized image.

        Returns:
            torch.Tensor: The output corresponding to the original image.
        """
        induced_rep_type = kwargs.get("induced_rep_type", "vector")
        return get_action_on_image_features(
            feature_map=x_canonicalized_out,
            group_info_dict=self.group_info_dict,
            group_element_dict=self.canonicalization_info_dict["group_element"],  # type: ignore
            induced_rep_type=induced_rep_type,
        )


class SteerableImageCanonicalization(ContinuousGroupImageCanonicalization):
    """
    This class represents a steerable image canonicalization model.

    The model is designed to be equivariant under a continuous group of euclidean transformations - rotations and reflections.

    Methods:
        __init__: Initializes the SteerableImageCanonicalization instance.
        get_rotation_matrix_from_vector: This method takes the input vector and returns the rotation matrix.
        get_groupelement: This method maps the input image to the group element.
    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        canonicalization_hyperparams: DictConfig,
        in_shape: tuple,
    ):
        """
        Initializes the SteerableImageCanonicalization instance.

        Args:
            canonicalization_network (torch.nn.Module): The canonicalization network.
            canonicalization_hyperparams (DictConfig): The hyperparameters for the canonicalization process.
            in_shape (tuple): The shape of the input images.
        """
        super().__init__(
            canonicalization_network, canonicalization_hyperparams, in_shape
        )
        self.group_type = canonicalization_network.group_type

    def get_rotation_matrix_from_vector(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        This method takes the input vector and returns the rotation matrix

        Args:
            vectors (torch.Tensor): input vector

        Returns:
            torch.Tensor: rotation matrices
        """
        v1 = vectors / torch.norm(vectors, dim=1, keepdim=True)
        v2 = torch.stack([-v1[:, 1], v1[:, 0]], dim=1)
        rotation_matrices = torch.stack([v1, v2], dim=1)
        return rotation_matrices

    def get_groupelement(self, x: torch.Tensor) -> dict:
        """
        This method takes the input image and maps it to the group element

        Args:
            x (torch.Tensor): input image

        Returns:
            dict: group element
        """
        group_element_dict: Dict[str, Any] = {}

        x = self.transformations_before_canonicalization_network_forward(x)

        # convert the group activations to one hot encoding of group element
        # this conversion is differentiable and will be used to select the group element
        out_vectors = self.canonicalization_network(x)

        # Check whether canonicalization_info_dict is already defined
        if not hasattr(self, "canonicalization_info_dict"):
            self.canonicalization_info_dict = {}

        (
            group_element_dict,
            group_element_representation,
        ) = self.get_group_from_out_vectors(out_vectors)
        self.canonicalization_info_dict["group_element_matrix_representation"] = (
            group_element_representation
        )

        self.canonicalization_info_dict["group_element"] = group_element_dict  # type: ignore

        return group_element_dict


class OptimizedSteerableImageCanonicalization(ContinuousGroupImageCanonicalization):
    """
    This class represents an optimized steerable image canonicalization model.

    The model is designed to be equivariant under a continuous group of transformations, which can include rotations and reflections.

    Methods:
        __init__: Initializes the OptimizedSteerableImageCanonicalization instance.
        get_rotation_matrix_from_vector: This method takes the input vector and returns the rotation matrix.
        group_augment: This method applies random rotations and reflections to the input images.
        get_groupelement: This method maps the input image to the group element.
        get_optimization_specific_loss: This method returns the optimization specific loss.
    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        canonicalization_hyperparams: DictConfig,
        in_shape: tuple,
    ):
        """
        Initializes the OptimizedSteerableImageCanonicalization instance.

        Args:
            canonicalization_network (torch.nn.Module): The canonicalization network.
            canonicalization_hyperparams (DictConfig): The hyperparameters for the canonicalization process.
            in_shape (tuple): The shape of the input images.
        """
        super().__init__(
            canonicalization_network, canonicalization_hyperparams, in_shape
        )
        self.group_type = canonicalization_hyperparams.group_type

    def get_rotation_matrix_from_vector(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        This method takes the input vector and returns the rotation matrix

        Args:
            vectors (torch.Tensor): input vector

        Returns:
            torch.Tensor: rotation matrices
        """
        v1 = vectors / torch.norm(vectors, dim=1, keepdim=True)
        v2 = torch.stack([-v1[:, 1], v1[:, 0]], dim=1)
        rotation_matrices = torch.stack([v1, v2], dim=1)
        return rotation_matrices

    def group_augment(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Augmentation of the input images by applying random rotations and, if applicable, reflections, with corresponding transformation matrices.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Augmented images.
            torch.Tensor: Corresponding transformation matrices.
        """
        batch_size = x.shape[0]

        # Generate random rotation angles (in radians)
        angles = torch.rand(batch_size, device=self.device) * 2 * torch.pi
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)

        # Create tensors for rotation matrices
        rotation_matrices = torch.zeros(batch_size, 2, 3, device=self.device)
        rotation_matrices[:, :2, :2] = torch.stack(
            (cos_a, -sin_a, sin_a, cos_a)
        ).reshape(-1, 2, 2)

        if self.group_type == "roto-reflection":
            # Generate reflection indicators (horizontal flip) with 50% probability
            reflect = (
                torch.randint(0, 2, (batch_size,), device=self.device).float() * 2 - 1
            )
            # Adjust the rotation matrix for reflections
            rotation_matrices[:, 0, 0] *= reflect

        # No need to create a 3x3 matrix or use torch.matmul since reflection is directly applied to rotation_matrices

        # Apply transformations
        # padding the input image to avoid boundary artifacts
        x = self.pad(x)

        # Note: F.affine_grid expects theta of shape (N, 2, 3) for 2D affine transformations
        grid = F.affine_grid(rotation_matrices, list(x.size()), align_corners=False)

        augmented_images = F.grid_sample(x, grid, align_corners=False)

        # crop the augmented images to the original size
        augmented_images = self.crop(augmented_images)

        # Necessary to get the correct rotation matrix for the augmented images
        # F.grid_sample and K.geometry.warp_affine use different conventions for the transformation matrix
        rotation_matrices[:, [0, 1], [1, 0]] *= -1

        # Return augmented images and the transformation matrices used
        return augmented_images, rotation_matrices[:, :, :2]

    def get_groupelement(self, x: torch.Tensor) -> dict:
        """
        Maps the input image to the group element.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            dict: The group element.
        """
        group_element_dict: Dict[str, Any] = {}

        batch_size = x.shape[0]

        # randomly sample generate some agmentations of the input image using rotation and reflection

        x_augmented, group_element_representations_augmented_gt = self.group_augment(
            x
        )  # size (batch_size * group_size, in_channels, height, width)

        x_all = torch.cat(
            [x, x_augmented], dim=0
        )  # size (batch_size * 2, in_channels, height, width)

        x_all = self.transformations_before_canonicalization_network_forward(x_all)

        out_vectors_all = self.canonicalization_network(
            x_all
        )  # size (batch_size * 2, out_vector_size)

        out_vectors_all = out_vectors_all.reshape(
            2 * batch_size, -1, 2
        )  # size (batch_size * 2, num_vectors, 2)

        out_vectors, out_vectors_augmented = out_vectors_all.chunk(2, dim=0)

        # Check whether canonicalization_info_dict is already defined
        if not hasattr(self, "canonicalization_info_dict"):
            self.canonicalization_info_dict = {}

        (
            group_element_dict,
            group_element_representations,
        ) = self.get_group_from_out_vectors(out_vectors)
        # Store the matrix representation of the group element for regularization and identity metric
        self.canonicalization_info_dict["group_element_matrix_representation"] = (
            group_element_representations
        )
        self.canonicalization_info_dict["group_element"] = group_element_dict  # type: ignore

        _, group_element_representations_augmented = self.get_group_from_out_vectors(
            out_vectors_augmented
        )
        self.canonicalization_info_dict[
            "group_element_matrix_representation_augmented"
        ] = group_element_representations_augmented
        self.canonicalization_info_dict[
            "group_element_matrix_representation_augmented_gt"
        ] = group_element_representations_augmented_gt

        return group_element_dict

    def get_optimization_specific_loss(self) -> torch.Tensor:
        """
        This method returns the optimization specific loss

        Returns:
            torch.Tensor: optimization specific loss
        """
        (
            group_element_representations_augmented,
            group_element_representations_augmented_gt,
        ) = (
            self.canonicalization_info_dict[
                "group_element_matrix_representation_augmented"
            ],
            self.canonicalization_info_dict[
                "group_element_matrix_representation_augmented_gt"
            ],
        )
        return F.mse_loss(
            group_element_representations_augmented,
            group_element_representations_augmented_gt,
        )
