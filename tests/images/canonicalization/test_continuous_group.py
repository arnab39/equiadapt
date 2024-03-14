from typing import Any, Dict
from unittest.mock import Mock

import pytest
import torch
from omegaconf import DictConfig
from torch import Tensor

from equiadapt import ContinuousGroupImageCanonicalization

@pytest.fixture
def sample_input() -> Tensor:
    # Create a sample input tensor
    return torch.rand((1, 3, 64, 64))  # A batch with one color image of size 64x64

@pytest.fixture
def grayscale_input() -> Tensor:
    # Create a grayscale input tensor
    return torch.rand((1, 1, 64, 64))  # A batch with one grayscale image of size 64x64

@pytest.fixture
def init_args() -> Dict[str, Any]:
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

def test_initialization(init_args: Dict[str, Any]) -> None:
    cgic = ContinuousGroupImageCanonicalization(**init_args)
    assert cgic.pad is not None, "Pad should be initialized."
    assert cgic.crop is not None, "Crop should be initialized."

def test_transformation_before_canonicalization_network_forward(
    sample_input: Tensor, init_args: Dict[str, Any]
) -> None:
    cgic = ContinuousGroupImageCanonicalization(**init_args)
    transformed = cgic.transformations_before_canonicalization_network_forward(
        sample_input
    )
    assert transformed.size() == torch.Size(
        [1, 3, 32, 32]
    ), "The transformed image should be resized to (32, 32)."
