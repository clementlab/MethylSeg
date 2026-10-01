"""Multisample MethylSeg orchestration and aggregation interfaces."""

from .aggregator import (
    AggregationMode,
    AggregatorConfig,
    MethylSegAggregator,
)
from .launcher import (
    AggregationLauncher,
    ClusterLauncherConfig,
    LauncherConfig,
    LocalLauncherConfig,
    ResolutionMethylSegConfig,
    RunMethod,
)

__all__ = [
    "AggregationLauncher",
    "AggregationMode",
    "AggregatorConfig",
    "ClusterLauncherConfig",
    "LauncherConfig",
    "LocalLauncherConfig",
    "MethylSegAggregator",
    "ResolutionMethylSegConfig",
    "RunMethod",
]
