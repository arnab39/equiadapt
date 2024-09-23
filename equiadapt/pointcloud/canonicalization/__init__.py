"""This module contains the pointcloud canonicalization methods."""

from equiadapt.pointcloud.canonicalization.continuous_group import (
    ContinuousGroupPointcloudCanonicalization,
    EquivariantPointcloudCanonicalization,
)

from equiadapt.pointcloud.canonicalization.discrete_group import (
    DiscreteGroupPointcloudCanonicalization,
    OptimizedGroupEquivariantPointcloudCanonicalization
)

__all__ = [
    "ContinuousGroupPointcloudCanonicalization",
    "EquivariantPointcloudCanonicalization",
    "DiscreteGroupPointcloudCanonicalization",
    "OptimizedGroupEquivariantPointcloudCanonicalization"
]
