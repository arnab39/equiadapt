import math
from typing import Any, Dict, List, Optional, Tuple, Union

import kornia as K
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from torchvision import transforms

from equiadapt.common.basecanonicalization import DiscreteGroupCanonicalization
from equiadapt.images.utils import (
    flip_boxes,
    flip_masks,
    get_action_on_image_features,
    rotate_boxes,
    rotate_masks,
)


class DiscreteGroupImageCanonicalization(DiscreteGroupCanonicalization):
    """
    This class represents a discrete group image canonicalization model.

    The model is designed to be equivariant under a discrete group of transformations, which can include rotations and reflections.
    Other discrete group canonicalizers can be derived from this class.

    Methods:
        __init__: Initializes the DiscreteGroupImageCanonicalization instance.
        groupactivations_to_groupelement: Takes the activations for each group element as input and returns the group element.
        get_groupelement: Maps the input image to a group element.
        transformations_before_canonicalization_network_forward: Applies transformations to the input images before passing it through the canonicalization network.
        canonicalize: Canonicalizes the input images.
        invert_canonicalization: Inverts the canonicalization of the output of the canonicalized image.
    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        canonicalization_hyperparams: DictConfig,
        in_shape: tuple,
    ):
        """
        Initializes the DiscreteGroupImageCanonicalization instance.

        Args:
            canonicalization_network (torch.nn.Module): The canonicalization network.
            canonicalization_hyperparams (DictConfig): The hyperparameters for the canonicalization process.
            in_shape (tuple): The shape of the input images.
        """
        super().__init__(canonicalization_network)

        self.beta = canonicalization_hyperparams.beta

        assert (
            len(in_shape) == 3
        ), "Input shape should be in the format (channels, height, width)"

        # DEfine all the image transformations here which are used during canonicalization
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

    def groupactivations_to_groupelement(self, group_activations: torch.Tensor) -> dict:
        """
        This method takes the activations for each group element as input and returns the group element

        Args:
            group_activations (torch.Tensor): activations for each group element.

        Returns:
            dict: group element.
        """
        # convert the group activations to one hot encoding of group element
        # this conversion is differentiable and will be used to select the group element
        group_elements_one_hot = self.groupactivations_to_groupelementonehot(
            group_activations
        )

        angles = torch.linspace(0.0, 360.0, self.num_rotations + 1)[
            : self.num_rotations
        ].to(self.device)
        group_elements_rot_comp = (
            torch.cat([angles, angles], dim=0)
            if self.group_type == "roto-reflection"
            else angles
        )

        group_element_dict = {}

        group_element_rot_comp = torch.sum(
            group_elements_one_hot * group_elements_rot_comp, dim=-1
        )
        group_element_dict["rotation"] = group_element_rot_comp

        if self.group_type == "roto-reflection":
            reflect_identifier_vector = torch.cat(
                [torch.zeros(self.num_rotations), torch.ones(self.num_rotations)], dim=0
            ).to(self.device)
            group_element_reflect_comp = torch.sum(
                group_elements_one_hot * reflect_identifier_vector, dim=-1
            )
            group_element_dict["reflection"] = group_element_reflect_comp

        return group_element_dict

    def get_group_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gets the group activations for the input images.

        Args:
            x (torch.Tensor): The input images.

        Returns:
            torch.Tensor: The group activations.
        """
        raise NotImplementedError(
            "get_group_activations is not implemented for"
            "the DiscreteGroupImageCanonicalization class"
        )

    def get_groupelement(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Maps the input image to a group element.

        Args:
            x (torch.Tensor): The input images.

        Returns:
            dict[str, torch.Tensor]: The corresponding group elements.
        """
        group_activations = self.get_group_activations(x)
        group_element_dict = self.groupactivations_to_groupelement(group_activations)

        # Check whether canonicalization_info_dict is already defined
        if not hasattr(self, "canonicalization_info_dict"):
            self.canonicalization_info_dict = {}

        self.canonicalization_info_dict["group_element"] = group_element_dict  # type: ignore
        self.canonicalization_info_dict["group_activations"] = group_activations

        return group_element_dict

    def transformations_before_canonicalization_network_forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies transformations to the input images before passing it through the canonicalization network.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The pre-canonicalized image.
        """
        x = self.crop_canonization(x)
        x = self.resize_canonization(x)
        return x

    def canonicalize(
        self, x: torch.Tensor, targets: Optional[List] = None, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Canonicalizes the input images.

        Args:
            x (torch.Tensor): The input images.
            targets (Optional[List], optional): The targets for instance segmentation. Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List]]: The canonicalized image, and optionally the targets.
        """
        self.device = x.device
        group_element_dict = self.get_groupelement(x)

        x = self.pad(x)

        if "reflection" in group_element_dict.keys():
            reflect_indicator = group_element_dict["reflection"][:, None, None, None]
            x = (1 - reflect_indicator) * x + reflect_indicator * K.geometry.hflip(x)

        x = K.geometry.rotate(x, -group_element_dict["rotation"])

        x = self.crop(x)

        if targets:
            # canonicalize the targets (for instance segmentation, masks and boxes)
            image_width = x.shape[-1]

            if "reflection" in group_element_dict.keys():
                # flip masks and boxes
                for t in range(len(targets)):
                    targets[t]["boxes"] = flip_boxes(targets[t]["boxes"], image_width)
                    targets[t]["masks"] = flip_masks(targets[t]["masks"])

            # rotate masks and boxes
            for t in range(len(targets)):
                targets[t]["boxes"] = rotate_boxes(
                    targets[t]["boxes"], group_element_dict["rotation"][t], image_width
                )
                targets[t]["masks"] = rotate_masks(
                    targets[t]["masks"], -group_element_dict["rotation"][t].item()  # type: ignore
                )

            return x, targets

        return x

    def invert_canonicalization(
        self, x_canonicalized_out: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
        Inverts the canonicalization of the output of the canonicalized image.

        Args:
            x_canonicalized_out (torch.Tensor): The output of the canonicalized image.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            torch.Tensor: The output corresponding to the original image.
        """
        induced_rep_type = kwargs.get("induced_rep_type", "regular")
        return get_action_on_image_features(
            feature_map=x_canonicalized_out,
            group_info_dict=self.group_info_dict,
            group_element_dict=self.canonicalization_info_dict["group_element"],  # type: ignore
            induced_rep_type=induced_rep_type,
        )


class GroupEquivariantImageCanonicalization(DiscreteGroupImageCanonicalization):
    """
    This class represents a discrete group equivariant image canonicalization model.

    The model is designed to be equivariant under a discrete group of transformations, which can include rotations and reflections.

    Methods:
        __init__: Initializes the GroupEquivariantImageCanonicalization instance.
        get_group_activations: Gets the group activations for the input images.
    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        canonicalization_hyperparams: DictConfig,
        in_shape: tuple,
    ):
        """
        Initializes the GroupEquivariantImageCanonicalization instance.

        Args:
            canonicalization_network (torch.nn.Module): The canonicalization network.
            canonicalization_hyperparams (DictConfig): The hyperparameters for the canonicalization process.
            in_shape (tuple): The shape of the input images.
        """
        super().__init__(
            canonicalization_network, canonicalization_hyperparams, in_shape
        )
        self.group_type = canonicalization_network.group_type
        self.num_rotations = canonicalization_network.num_rotations
        self.num_group = (
            self.num_rotations
            if self.group_type == "rotation"
            else 2 * self.num_rotations
        )
        self.group_info_dict = {
            "num_rotations": self.num_rotations,
            "num_group": self.num_group,
        }

    def get_group_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gets the group activations for the input image.

        This method takes an image as input, applies transformations before forwarding it through the canonicalization network,
        and then returns the group activations.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The group activations.
        """
        x = self.transformations_before_canonicalization_network_forward(x)
        group_activations = self.canonicalization_network(x)
        return group_activations


class OptimizedGroupEquivariantImageCanonicalization(
    DiscreteGroupImageCanonicalization
):
    """
    This class represents an optimized (discrete) group equivariant image canonicalization model.

    The model is designed to be equivariant under a discrete group of transformations, which can include rotations and reflections.

    Methods:
        __init__: Initializes the OptimizedGroupEquivariantImageCanonicalization instance.
        rotate_and_maybe_reflect: Rotate and maybe reflect the input images.
        group_augment: Augment the input images by applying group transformations (rotations and reflections).
        get_group_activations: Gets the group activations for the input images.
        get_optimization_specific_loss: Gets the loss specific to the optimization process.
    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
        canonicalization_hyperparams: DictConfig,
        in_shape: tuple,
    ):
        """
        Initializes the OptimizedGroupEquivariantImageCanonicalization instance.

        Args:
            canonicalization_network (torch.nn.Module): The canonicalization network.
            canonicalization_hyperparams (DictConfig): The hyperparameters for the canonicalization process.
            in_shape (tuple): The shape of the input images.
        """
        super().__init__(
            canonicalization_network, canonicalization_hyperparams, in_shape
        )
        self.group_type = canonicalization_hyperparams.group_type
        self.num_rotations = canonicalization_hyperparams.num_rotations
        self.artifact_err_wt = canonicalization_hyperparams.artifact_err_wt
        self.num_group = (
            self.num_rotations
            if self.group_type == "rotation"
            else 2 * self.num_rotations
        )
        self.out_vector_size = canonicalization_network.out_vector_size

        # group optimization specific cropping and padding (required for group_augment())
        group_augment_in_shape = canonicalization_hyperparams.resize_shape
        self.crop_group_augment = (
            torch.nn.Identity()
            if in_shape[0] == 1
            else transforms.CenterCrop(group_augment_in_shape)
        )
        self.pad_group_augment = (
            torch.nn.Identity()
            if in_shape[0] == 1
            else transforms.Pad(
                math.ceil(group_augment_in_shape * 0.5), padding_mode="edge"
            )
        )

        self.reference_vector = torch.nn.Parameter(
            torch.randn(1, self.out_vector_size),
            requires_grad=canonicalization_hyperparams.learn_ref_vec,
        )
        self.group_info_dict = {
            "num_rotations": self.num_rotations,
            "num_group": self.num_group,
        }

    def rotate_and_maybe_reflect(
        self, x: torch.Tensor, degrees: torch.Tensor, reflect: bool = False
    ) -> List[torch.Tensor]:
        """
        Rotate and maybe reflect the input images.

        Args:
            x (torch.Tensor): The input image.
            degrees (torch.Tensor): The degrees of rotation.
            reflect (bool, optional): Whether to reflect the image. Defaults to False.

        Returns:
            List[torch.Tensor]: The list of rotated and maybe reflected images.
        """
        x_augmented_list = []
        for degree in degrees:
            x_rot = self.pad_group_augment(x)
            x_rot = K.geometry.rotate(x_rot, -degree)
            if reflect:
                x_rot = K.geometry.hflip(x_rot)
            x_rot = self.crop_group_augment(x_rot)
            x_augmented_list.append(x_rot)
        return x_augmented_list

    def group_augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Augment the input images by applying group transformations (rotations and reflections).

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The augmented image.
        """
        degrees = torch.linspace(0, 360, self.num_rotations + 1)[:-1].to(self.device)
        x_augmented_list = self.rotate_and_maybe_reflect(x, degrees)

        if self.group_type == "roto-reflection":
            x_augmented_list += self.rotate_and_maybe_reflect(x, degrees, reflect=True)

        return torch.cat(x_augmented_list, dim=0)

    def get_group_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gets the group activations for the input image.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The group activations.
        """
        x = self.transformations_before_canonicalization_network_forward(x)
        x_augmented = self.group_augment(
            x
        )  # size (batch_size * group_size, in_channels, height, width)
        vector_out = self.canonicalization_network(
            x_augmented
        )  # size (batch_size * group_size, reference_vector_size)
        self.canonicalization_info_dict = {"vector_out": vector_out}

        if self.artifact_err_wt:
            # select a random rotation for each image in the batch
            rotation_indices = torch.randint(
                0, self.num_rotations, (x_augmented.shape[0],)
            ).to(self.device)

            # apply the rotation degree to the images
            x_dummy = self.pad_group_augment(x_augmented)
            x_dummy = K.geometry.rotate(
                x_dummy, -rotation_indices * 360 / self.num_rotations
            )
            x_dummy = self.crop_group_augment(x_dummy)

            # invert the image back to the original orientation
            x_dummy = self.pad_group_augment(x_dummy)
            x_dummy = K.geometry.rotate(
                x_dummy, rotation_indices * 360 / self.num_rotations
            )
            x_dummy = self.crop_group_augment(x_dummy)

            vector_out_dummy = self.canonicalization_network(
                x_dummy
            )  # size (batch_size * group_size, reference_vector_size)
            self.canonicalization_info_dict.update(
                {"vector_out_dummy": vector_out_dummy}
            )

        scalar_out = F.cosine_similarity(
            self.reference_vector.repeat(vector_out.shape[0], 1), vector_out
        )  # size (batch_size * group_size, 1)
        group_activations = scalar_out.reshape(
            self.num_group, -1
        ).T  # size (batch_size, group_size)
        return group_activations

    def get_optimization_specific_loss(self) -> torch.Tensor:
        """
        Gets the loss specific to the optimization process.

        Returns:
            torch.Tensor: The loss.
        """
        vectors = self.canonicalization_info_dict["vector_out"]

        # compute error to reduce rotation artifacts
        rotation_artifact_error = 0
        if self.artifact_err_wt:
            vectors_dummy = self.canonicalization_info_dict["vector_out_dummy"]
            rotation_artifact_error = torch.nn.functional.mse_loss(
                vectors_dummy, vectors
            )  # type: ignore

        # error to ensure that the vectors are (as much as possible) orthogonal
        vectors = vectors.reshape(self.num_group, -1, self.out_vector_size).permute(
            (1, 0, 2)
        )  # (batch_size, group_size, vector_out_size)
        distances = vectors @ vectors.permute((0, 2, 1))
        mask = 1.0 - torch.eye(self.num_group).to(
            self.device
        )  # (group_size, group_size)

        return (
            torch.abs(distances * mask).mean()
            + self.artifact_err_wt * rotation_artifact_error
        )
