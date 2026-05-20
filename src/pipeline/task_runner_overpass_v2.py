"""
TaskRunner + batched Overpass (v2)
==================================
Three Overpass round-trips for the **union** of all tile bboxes (highways,
buildings, landcover), then each tile receives a local clip via
``ClippingOsmClient`` (see ``osm_batch_prefetch``).

Does not modify the legacy ``TaskRunner.run`` implementation — this is an
optional orchestration wrapper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.extraction.osm_batch_prefetch import (
    ClippingOsmClient,
    all_tiles_have_osm_json_cache,
    compute_tile_union_bbox,
    prefetch_highways_and_buildings,
    warn_union_bbox_if_heavy,
)

from .task_runner import TaskRunner, TileConfig


def run_task_batch_overpass_v2(
    task_runner: TaskRunner,
    *,
    tiles: list[TileConfig],
    manifest_path: str | Path,
    output_root: str,
    force_osm: bool,
) -> dict[str, Any]:
    """
    Run ``TaskRunner.run`` with at most **three** Overpass downloads for the
    whole batch (when multiple tiles all need fresh OSM).

    If every tile already has both OSM JSON caches and ``force_osm`` is false,
    skips any network fetch (same as a normal dry re-run).

    When a union fetch runs, ``pipeline.osm_force_download`` is forced **True**
    so per-tile caches are all regenerated from the same snapshot (avoids a
    mix of stale files and fresh union data).
    """
    pipeline = task_runner.pipeline

    if len(tiles) <= 1:
        return task_runner.run(tiles, manifest_path=manifest_path)

    if not force_osm and all_tiles_have_osm_json_cache(tiles, output_root):
        print(
            "\n  OSM batch v2: všechny dlaždice už mají OSM cache — "
            "přeskakuji union Overpass.\n"
        )
        return task_runner.run(tiles, manifest_path=manifest_path)

    union = compute_tile_union_bbox(tiles)
    warn_union_bbox_if_heavy(union)
    print(
        f"\n  OSM batch v2: 3× Overpass pro union bbox "
        f"(S,W,N,E)={union}\n"
    )
    hw, bld, lc = prefetch_highways_and_buildings(union)
    n_lc = sum(len(v) for v in lc.values())
    print(
        f"     staženo {len(hw)} silnic, {len(bld)} budov/částí, "
        f"{n_lc} landcover prvků "
        f"(voda {len(lc.get('water_polygons', []))}, řeky {len(lc.get('waterways', []))}, "
        f"zelen {len(lc.get('green_polygons', []))}, železnice {len(lc.get('railways', []))}) "
        f"→ ořez podle jednotlivých dlaždic\n"
    )
    pipeline.osm_client = ClippingOsmClient(hw, bld, lc)
    pipeline.osm_force_download = True
    return task_runner.run(tiles, manifest_path=manifest_path)
