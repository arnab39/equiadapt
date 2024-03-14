import numpy as np
import torch
from omegaconf import DictConfig

from equiadapt.common.basecanonicalization import IdentityCanonicalization
from equiadapt.pointcloud.canonicalization.continuous_group import (
    EquivariantPointcloudCanonicalization,
)
from equiadapt.pointcloud.canonicalization_networks import VNSmall


def get_canonicalization_network(
    canonicalization_type: str,
    canonicalization_hyperparams: DictConfig,
) -> torch.nn.Module:
    """
    The function returns the canonicalization network based on the canonicalization type

    Args:
        canonicalization_type (str): defines the type of canonicalization network
        options are 1) vector_neuron
    """
    if canonicalization_type == "identity":
        return torch.nn.Identity()

    canonicalization_network_dict = {
        "group_equivariant": {
            "vector_neuron_small": VNSmall,
        },
    }

    if canonicalization_type not in canonicalization_network_dict:
        raise ValueError(f"{canonicalization_type} is not implemented")
    if (
        canonicalization_hyperparams.network_type
        not in canonicalization_network_dict[canonicalization_type]
    ):
        raise ValueError(
            f"{canonicalization_hyperparams.network_type} is not implemented for {canonicalization_type}"
        )

    canonicalization_network = canonicalization_network_dict[canonicalization_type][
        canonicalization_hyperparams.network_type
    ](canonicalization_hyperparams.network_hyperparams)

    return canonicalization_network


def get_canonicalizer(
    canonicalization_type: str,
    canonicalization_network: torch.nn.Module,
    canonicalization_hyperparams: DictConfig,
) -> torch.nn.Module:
    """
    The function returns the canonicalization network based on the canonicalization type

    Args:
        canonicalization_type (str): defines the type of canonicalization network
        options are 1) vector_neuron
    """

    if canonicalization_type == "identity":
        return IdentityCanonicalization(canonicalization_network)

    canonicalizer_dict = {
        "group_equivariant": EquivariantPointcloudCanonicalization,
    }

    if canonicalization_type not in canonicalizer_dict:
        raise ValueError(
            f"{canonicalization_type} needs a canonicalization network implementation."
        )

    canonicalizer = canonicalizer_dict[canonicalization_type](
        canonicalization_network=canonicalization_network,
        canonicalization_hyperparams=canonicalization_hyperparams,
    )

    return canonicalizer


def random_shift_point_cloud(
    batch_data: torch.Tensor, shift_range: float = 0.1
) -> torch.Tensor:
    """Randomly shift point cloud. Shift is per point cloud.
    Input:
      batch_data: BxNx3 array, original batch of point clouds
      shift_range: float, range of random shifts
    Return:
      shifted_batch_data: BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    # generate random shifts in pytorch from -shift_range to shift_range of shape (B, 3)
    shifts = (
        torch.rand((B, 3), device=batch_data.device) * 2 * shift_range - shift_range
    )
    shifted_batch_data = batch_data.clone()
    for batch_index in range(B):
        shifted_batch_data[batch_index, :, :] += shifts[batch_index, :]
    return shifted_batch_data


def random_scale_point_cloud(
    batch_data: torch.Tensor, scale_low: float = 0.8, scale_high: float = 1.2
) -> torch.Tensor:
    """Randomly scale the point cloud. Scale is per point cloud.
    Input:
        batch_data: BxNx3 array, original batch of point clouds
        scale_low: float, lower bound of random scale
        scale_high: float, upper bound of random scale
    Return:
        scaled_batch_data: BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    # generate random scales in pytorch from scale_low to scale_high of shape (B) and put it on the device of batch_data
    scales = (
        torch.rand((B,), device=batch_data.device) * (scale_high - scale_low)
        + scale_low
    )
    scaled_batch_data = batch_data.clone()
    for batch_index in range(B):
        scaled_batch_data[batch_index, :, :] *= scales[batch_index]
    return scaled_batch_data


def random_point_dropout(
    batch_pc: torch.Tensor, max_dropout_ratio: float = 0.9
) -> torch.Tensor:
    """batch_pc: BxNx3"""
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = torch.where(torch.rand(batch_pc.shape[1]) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc.clone()[
                b, 0, :
            ]  # set to the first point
    return batch_pc
