"""
TaskRunner
==========
Batch-tile orchestration: runs TerrainPipeline over a list of tile
definitions and records results in a tile-manifest.json.

Tile definition (dict)
----------------------
Required keys:
  name  (str)   – used as output file prefix and tile identifier
  bbox  (tuple) – (south, west, north, east) in decimal degrees

Optional keys:
  output_dir (str) – override the global output_root for this tile

Example
-------
    from src.pipeline.task_runner import TaskRunner, TileConfig

    tiles = [
        TileConfig(name="prague_center", bbox=(50.07, 14.43, 50.08, 14.44)),
        TileConfig(name="prague_north",  bbox=(50.11, 14.40, 50.13, 14.43)),
    ]

    runner = TaskRunner(pipeline=pipeline, output_root="outputs/batch")
    summary = runner.run(tiles, manifest_path="outputs/batch/tile-manifest.json")
    print(summary)
"""
from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tile_manifest import TileManifest, _utc_now


# ── Tile definition ───────────────────────────────────────────────────────────

@dataclass
class TileConfig:
    name:       str
    bbox:       tuple[float, float, float, float]
    output_dir: str | None = None   # overrides TaskRunner.output_root


# ── Runner ────────────────────────────────────────────────────────────────────

class TaskRunner:
    """
    Iterate over TileConfig list, run the pipeline for each tile, write manifest.

    Parameters
    ----------
    pipeline    : configured TerrainPipeline instance
    output_root : base directory; each tile gets its own sub-folder <output_root>/<name>/
    version     : manifest version string (semver)
    zoom        : max zoom level recorded in the manifest
    layers      : map-serving layer descriptors (forwarded to TileManifest)
    stop_on_error : if True, abort the batch on the first failed tile
    """

    def __init__(
        self,
        pipeline,
        output_root:    str = "outputs",
        version:        str = "1.0.0",
        zoom:           int = 14,
        layers:         list[dict] | None = None,
        stop_on_error:  bool = False,
    ):
        self.pipeline      = pipeline
        self.output_root   = output_root
        self.version       = version
        self.zoom          = zoom
        self.layers        = layers or []
        self.stop_on_error = stop_on_error

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        tiles:         list[TileConfig],
        manifest_path: str | Path = "tile-manifest.json",
    ) -> dict[str, Any]:
        """
        Process all tiles and write a tile-manifest.json.

        Returns a summary dict:
          {
            "total":     int,
            "completed": int,
            "failed":    int,
            "manifest":  str,   # resolved path
            "tiles":     [...]  # per-tile result dicts
          }
        """
        manifest = TileManifest(
            version=self.version,
            zoom=self.zoom,
            layers=self.layers,
        )

        completed = 0
        failed    = 0
        results   = []

        print(f"\n{'='*60}")
        print(f"  TaskRunner: {len(tiles)} tile(s)  →  {manifest_path}")
        print(f"{'='*60}")

        for i, tile in enumerate(tiles, 1):
            tile_dir = tile.output_dir or os.path.join(self.output_root, tile.name)
            os.makedirs(tile_dir, exist_ok=True)

            # Swap the pipeline's output dir so every helper writes into tile_dir
            old_output_dir = getattr(self.pipeline.converter, "output_dir", None)
            if old_output_dir is not None:
                self.pipeline.converter.output_dir = tile_dir
            if hasattr(self.pipeline, "road_mesh") and self.pipeline.road_mesh:
                self.pipeline.road_mesh.output_dir = tile_dir
            if hasattr(self.pipeline, "building_extruder") and self.pipeline.building_extruder:
                self.pipeline.building_extruder.output_dir = tile_dir

            print(f"\n[{i}/{len(tiles)}] Tile '{tile.name}'  bbox={tile.bbox}")

            try:
                result = self.pipeline.run_pipeline(tile.bbox, tile.name)
                status  = "completed"
                outputs = {
                    "terrain":     result.get("terrain"),
                    "roads":       result.get("roads"),
                    "buildings":   result.get("buildings"),
                    "sdf_texture": result.get("sdf_texture"),
                }
                manifest.add_tile(tile.name, tile.bbox,
                                  status="completed", outputs=outputs)
                completed += 1
                tile_summary = {"name": tile.name, "status": "completed",
                                "outputs": outputs}

            except Exception as exc:
                msg = traceback.format_exc()
                print(f"  ✗ Tile '{tile.name}' selhal: {exc}")
                manifest.add_tile(tile.name, tile.bbox,
                                  status="failed", error=str(exc))
                failed += 1
                tile_summary = {"name": tile.name, "status": "failed",
                                "error": str(exc)}
                if self.stop_on_error:
                    results.append(tile_summary)
                    break

            finally:
                # Restore original output dir
                if old_output_dir is not None:
                    self.pipeline.converter.output_dir = old_output_dir

            results.append(tile_summary)

        resolved = manifest.write(manifest_path)
        print(f"\n{'='*60}")
        print(f"  Batch hotov — {completed}/{len(tiles)} OK, {failed} selhalo")
        print(f"  Manifest: {resolved}")
        print(f"{'='*60}\n")

        return {
            "total":     len(tiles),
            "completed": completed,
            "failed":    failed,
            "manifest":  str(resolved),
            "tiles":     results,
        }
