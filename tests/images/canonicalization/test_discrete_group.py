import pytest
import torch
from omegaconf import DictConfig

from equiadapt.images.canonicalization.discrete_group import (
    GroupEquivariantImageCanonicalization,
)
from equiadapt.images.canonicalization_networks.escnn_networks import (
    ESCNNEquivariantNetwork,
)


@pytest.fixture
def init_args() -> dict:
    """
    Initialize the arguments for the canonicalization function.

    Returns:
        dict: A dictionary containing the initialization arguments.
    """
    # Mock initialization arguments
    canonicalization_hyperparams = DictConfig(
        {
            "input_crop_ratio": 0.9,
            "resize_shape": (32, 32),
            "beta": 0.1,
        }
    )
    return {
        "canonicalization_network": ESCNNEquivariantNetwork(
            in_shape=(3, 64, 64),
            out_channels=32,
            kernel_size=3,
            group_type="rotation",
            num_rotations=4,
            num_layers=2,
        ),
        "canonicalization_hyperparams": canonicalization_hyperparams,
        "in_shape": (3, 64, 64),
    }


# try both types of induced representations (regular and scalar)
@pytest.mark.parametrize("induced_rep, num_channels", [("regular", 12), ("scalar", 3)])
def test_invert_canonicalization_induced_rep(
    induced_rep: str, num_channels: int, init_args: dict
) -> None:
    """
    Test the inversion of the canonicalization-induced representation.

    Args:
        induced_rep (str): The type of induced representation.
        num_channels (int): The number of channels in the sample image.
    """

    # Initialize the canonicalization function
    dgic = GroupEquivariantImageCanonicalization(**init_args)

    # Apply the canonicalization function
    image = torch.randn((1, 3, 64, 64))

    _ = dgic(image)  # to populate the canonicalization_info_dict

    canonicalized_image = torch.randn((1, num_channels, 64, 64))

    # Invert the canonicalization-induced representation
    inverted_image = dgic.invert_canonicalization(
        canonicalized_image, **{"induced_rep_type": induced_rep}
    )
