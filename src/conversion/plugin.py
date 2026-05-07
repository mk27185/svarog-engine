"""
Plug-in protocols for terrain generation modules.

Any class that implements the protocol can be swapped in to
TerrainPipeline without changes to the pipeline code.
"""
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class RoadPlugin(Protocol):
    """Road geometry generation plug-in."""

    def generate(
        self,
        highways: list[dict],
        z_grid: np.ndarray,
        meta: dict,
        *,
        obj_name: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        ...


@runtime_checkable
class BuildingPlugin(Protocol):
    """Building geometry generation plug-in."""

    def generate(
        self,
        buildings: list[dict],
        z_grid: np.ndarray,
        meta: dict,
        *,
        obj_name: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        ...
