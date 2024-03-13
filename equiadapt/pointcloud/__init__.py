"""This package contains modules for the equiadapt pointcloud canonicalization."""

from equiadapt.pointcloud import canonicalization, canonicalization_networks
from equiadapt.pointcloud.canonicalization import (
    ContinuousGroupPointcloudCanonicalization,
    EquivariantPointcloudCanonicalization,
    continuous_group,
)
from equiadapt.pointcloud.canonicalization_networks import (
    EPS,
    VNBatchNorm,
    VNBilinear,
    VNLeakyReLU,
    VNLinear,
    VNLinearLeakyReLU,
    VNMaxPool,
    VNSmall,
    VNSoftplus,
    VNStdFeature,
    equivariant_networks,
    get_graph_feature_cross,
    knn,
    mean_pool,
    vector_neuron_layers,
)

__all__ = [
    "ContinuousGroupPointcloudCanonicalization",
    "EPS",
    "EquivariantPointcloudCanonicalization",
    "VNBatchNorm",
    "VNBilinear",
    "VNLeakyReLU",
    "VNLinear",
    "VNLinearLeakyReLU",
    "VNMaxPool",
    "VNSmall",
    "VNSoftplus",
    "VNStdFeature",
    "canonicalization",
    "canonicalization_networks",
    "continuous_group",
    "equivariant_networks",
    "get_graph_feature_cross",
    "knn",
    "mean_pool",
    "vector_neuron_layers",
]
