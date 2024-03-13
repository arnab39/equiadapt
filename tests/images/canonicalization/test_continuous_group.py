from unittest.mock import Mock

import pytest
import torch
from omegaconf import DictConfig

from equiadapt import (
    ContinuousGroupImageCanonicalization,
)  # Update with your actual import path


@pytest.fixture
def sample_input():
    # Create a sample input tensor
    return torch.rand((1, 3, 64, 64))  # A batch with one color image of size 64x64


@pytest.fixture
def grayscale_input():
    # Create a grayscale input tensor
    return torch.rand((1, 1, 64, 64))  # A batch with one grayscale image of size 64x64


@pytest.fixture
def init_args():
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


def test_initialization(init_args):
    cgic = ContinuousGroupImageCanonicalization(**init_args)
    assert cgic.pad is not None, "Pad should be initialized."
    assert cgic.crop is not None, "Crop should be initialized."


def test_transformation_before_canonicalization_network_forward(
    sample_input, init_args
):
    cgic = ContinuousGroupImageCanonicalization(**init_args)
    transformed = cgic.transformations_before_canonicalization_network_forward(
        sample_input
    )
    assert transformed.size() == torch.Size(
        [1, 3, 32, 32]
    ), "The transformed image should be resized to (32, 32)."


@pytest.fixture
def canonicalization_instance():
    instance = ContinuousGroupImageCanonicalization(
        canonicalization_network=Mock(),
        canonicalization_hyperparams={
            "input_crop_ratio": 0.9,
            "resize_shape": (32, 32),
        },
        in_shape=(3, 64, 64),
    )
    # Mocking the get_groupelement method to return a fixed group element
    instance.get_groupelement = Mock(
        return_value={
            "rotation": torch.eye(2).unsqueeze(0),
            "reflection": torch.tensor([[[[0]]]]),
        }
    )
    return instance
