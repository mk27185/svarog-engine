"""
ConfigLoader
============
Loads a YAML batch config file and returns a list of TileConfig objects
together with pipeline kwargs.

YAML format
-----------
    output_root: outputs/my_world   # required
    upsample_factor: 10             # optional, overrides world-config
    draco: true                     # optional, overrides world-config
    tiles:
      - name: prague_center
        bbox: [50.07, 14.43, 50.08, 14.44]
      - name: prague_north
        bbox: [50.11, 14.40, 50.13, 14.43]
      # XYZ tile — bbox derived automatically
      - xyz: [15, 17898, 11245]

The bbox is (south, west, north, east) in decimal degrees.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .task_runner import TileConfig
from .tile_splitter import TileSplitter, XYZ


@dataclass
class BatchConfig:
    output_root:     str
    tiles:           list[TileConfig]
    upsample_factor: int  | None = None
    draco:           bool | None = None


class ConfigLoader:

    @staticmethod
    def load(path: str | Path) -> BatchConfig:
        """Load a YAML config file and return a BatchConfig."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Config file must be a YAML mapping: {path}")
        if "output_root" not in raw:
            raise ValueError("Config file must have 'output_root' key.")
        if "tiles" not in raw or not isinstance(raw["tiles"], list):
            raise ValueError("Config file must have a 'tiles' list.")

        tiles = []
        for entry in raw["tiles"]:
            tiles.append(ConfigLoader._parse_tile(entry))

        return BatchConfig(
            output_root=raw["output_root"],
            tiles=tiles,
            upsample_factor=raw.get("upsample_factor"),
            draco=raw.get("draco"),
        )

    @staticmethod
    def _parse_tile(entry: dict) -> TileConfig:
        if not isinstance(entry, dict):
            raise ValueError(f"Each tile entry must be a dict, got: {entry!r}")

        # XYZ-only entry
        if "xyz" in entry and "bbox" not in entry:
            z, x, y = entry["xyz"]
            tile = TileSplitter.from_xyz(int(z), int(x), int(y))
            if "name" in entry:
                tile = TileConfig(
                    name=entry["name"], bbox=tile.bbox, xyz=tile.xyz
                )
            return tile

        if "bbox" not in entry:
            raise ValueError(
                f"Tile entry must have 'bbox' or 'xyz': {entry!r}"
            )

        bbox_raw = entry["bbox"]
        if len(bbox_raw) != 4:
            raise ValueError(f"bbox must have 4 values [s, w, n, e]: {bbox_raw}")

        bbox = tuple(float(v) for v in bbox_raw)

        xyz = None
        if "xyz" in entry:
            z, x, y = entry["xyz"]
            xyz = XYZ(int(z), int(x), int(y))

        return TileConfig(
            name=entry.get("name") or f"tile_{bbox[0]:.4f}_{bbox[1]:.4f}",
            bbox=bbox,
            output_dir=entry.get("output_dir"),
            xyz=xyz,
        )
