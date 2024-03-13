"""This package contains equivariant modules and networks for the equiadapt pointcloud canonicalization."""

from equiadapt.pointcloud.canonicalization_networks import (
    equivariant_networks,
    vector_neuron_layers,
)
from equiadapt.pointcloud.canonicalization_networks.equivariant_networks import (
    VNSmall,
    get_graph_feature_cross,
    knn,
)
from equiadapt.pointcloud.canonicalization_networks.vector_neuron_layers import (
    EPS,
    VNBatchNorm,
    VNBilinear,
    VNLeakyReLU,
    VNLinear,
    VNLinearLeakyReLU,
    VNMaxPool,
    VNSoftplus,
    VNStdFeature,
    mean_pool,
)

__all__ = [
    "EPS",
    "VNBatchNorm",
    "VNBilinear",
    "VNLeakyReLU",
    "VNLinear",
    "VNLinearLeakyReLU",
    "VNMaxPool",
    "VNSmall",
    "VNSoftplus",
    "VNStdFeature",
    "equivariant_networks",
    "get_graph_feature_cross",
    "knn",
    "vector_neuron_layers",
    "mean_pool",
]
