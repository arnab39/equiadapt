"""This package contains equivariant modules and networks for the equiadapt pointcloud canonicalization."""

from equiadapt.pointcloud.canonicalization_networks import (
    equivariant_networks,
    vector_neuron_layers,
    utils,
    non_equivariant_networks,
)
from equiadapt.pointcloud.canonicalization_networks.equivariant_networks import (
    VNSmall,
)
from equiadapt.pointcloud.canonicalization_networks.utils import (
    get_graph_feature, knn, get_graph_feature_cross
)
from equiadapt.pointcloud.canonicalization_networks.non_equivariant_networks import (
    DGCNN_small,
    PointNet_small,
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
    "non_equivariant_networks",
    "PointNet_small",
    "DGCNN_small",
    "utils",
    "get_graph_feature",
]
