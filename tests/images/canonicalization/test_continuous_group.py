from typing import Generator
from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import DictConfig

from equiadapt import ContinuousGroupImageCanonicalization


@pytest.fixture
def sample_input() -> torch.Tensor:
    """
    Fixture that returns a sample input tensor.

    Returns:
        torch.Tensor: A batch with one color image of size 64x64.
    """
    return torch.rand((1, 3, 64, 64))


@pytest.fixture
def grayscale_input() -> torch.Tensor:
    """
    Fixture function that returns a grayscale input tensor.

    Returns:
        torch.Tensor: A batch with one grayscale image of size 64x64.
    """
    return torch.rand((1, 1, 64, 64))


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
        }
    )
    return {
        "canonicalization_network": torch.nn.Identity(),  # Placeholder
        "canonicalization_hyperparams": canonicalization_hyperparams,
        "in_shape": (3, 64, 64),
    }


def test_initialization(init_args: dict) -> None:
    """
    Test the initialization of ContinuousGroupImageCanonicalization.

    Args:
        init_args (dict): A dictionary containing the initialization arguments.
    """
    cgic = ContinuousGroupImageCanonicalization(**init_args)
    assert cgic.pad is not None, "Pad should be initialized."
    assert cgic.crop is not None, "Crop should be initialized."


def test_transformation_before_canonicalization_network_forward(
    sample_input: torch.Tensor, init_args: dict
) -> None:
    """
    Test the `transformations_before_canonicalization_network_forward` method of the ContinuousGroupImageCanonicalization class.

    Args:
        sample_input: The input sample to be transformed.
        init_args: The initialization arguments for the ContinuousGroupImageCanonicalization class.

    Returns:
        None

    Raises:
        AssertionError: If the transformed image size is not [1, 3, 32, 32].

    """

    cgic = ContinuousGroupImageCanonicalization(**init_args)
    transformed = cgic.transformations_before_canonicalization_network_forward(
        sample_input
    )
    assert transformed.size() == torch.Size(
        [1, 3, 32, 32]
    ), "The transformed image should be resized to (32, 32)."


@pytest.fixture
def canonicalization_instance() -> (
    Generator[ContinuousGroupImageCanonicalization, None, None]
):
    """
    Generates an instance of ContinuousGroupImageCanonicalization with specified parameters.

    Returns:
        Generator[ContinuousGroupImageCanonicalization, None, None]: A generator that yields the instance.
    """
    instance = ContinuousGroupImageCanonicalization(
        canonicalization_network=Mock(),
        canonicalization_hyperparams={
            "input_crop_ratio": 0.9,
            "resize_shape": (32, 32),
        },
        in_shape=(3, 64, 64),
    )
    # Mocking the get_groupelement method to return a fixed group element
    with patch.object(
        instance,
        "get_groupelement",
        return_value={
            "rotation": torch.eye(2).unsqueeze(0),
            "reflection": torch.tensor([[[[0]]]]),
        },
    ):
        yield instance
