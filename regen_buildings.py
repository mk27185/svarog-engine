#!/usr/bin/env python3
"""
Regenerate building OBJs and re-export GLBs for all existing tile directories.

Uses cached data only (no network):
  - reads  {tile}/_osm_buildings.json   → building geometry
  - reads  union DEM from .dem_cache/   → terrain z_grid for elevation sampling
  - writes {tile}/_buildings.obj         (new OBJ with fixed Z placement)
  - writes {tile}/.glb                   (re-exported combining existing terrain + new buildings)

Usage:
    python regen_buildings.py outputs/prague-center/
"""

import json
import math
import os
import sys
from pathlib import Path

UPSAMPLE_FACTOR = 10   # must match what was used during original tile generation


def tile_to_latlon(z: int, x: int, y: int) -> tuple[float, float]:
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def find_union_dem(output_root: Path) -> str | None:
    """Locate the union DEM TIF in the .dem_cache folder."""
    cache_dir = output_root / ".dem_cache"
    if not cache_dir.exists():
        return None
    tifs = sorted(cache_dir.glob("*.tif"))
    # Prefer the non-buffered file (no '_buf' suffix) if present
    unbuffered = [t for t in tifs if "_buf" not in t.name]
    return str(unbuffered[0] if unbuffered else (tifs[0] if tifs else None))


def main(output_root: str) -> None:
    root = Path(output_root)
    if not root.exists():
        sys.exit(f"Directory not found: {root}")

    union_dem = find_union_dem(root)
    if not union_dem:
        sys.exit(
            f"No union DEM found in {root}/.dem_cache/  "
            f"– run the full pipeline once first."
        )
    print(f"Union DEM: {union_dem}")

    from src.conversion.terrain_converter import TerrainConverter
    from src.conversion.building_extruder  import BuildingExtruder
    from src.conversion.gltf_exporter      import GltfExporter

    tile_dirs = sorted(root.glob("[0-9]*/*/*"))
    tile_dirs = [d for d in tile_dirs if d.is_dir()]
    total = len(tile_dirs)
    print(f"Found {total} tile directories\n")

    ok = err = skip = 0

    for i, td in enumerate(tile_dirs, 1):
        try:
            y_str = td.name
            x_str = td.parent.name
            z_str = td.parent.parent.name
            z, x, y = int(z_str), int(x_str), int(y_str)
        except ValueError:
            skip += 1
            continue

        bld_json = td / f"{y}_osm_buildings.json"
        glb_path = td / f"{y}.glb"
        terrain_obj = td / f"{y}_terrain.obj"

        if not bld_json.exists() or not terrain_obj.exists():
            skip += 1
            continue

        with open(bld_json) as f:
            buildings = json.load(f)

        n_lat, w_lon = tile_to_latlon(z, x, y)
        s_lat, e_lon = tile_to_latlon(z, x + 1, y + 1)
        bbox = (s_lat, w_lon, n_lat, e_lon)

        try:
            conv = TerrainConverter(output_dir=str(td))
            z_grid, meta = conv.build_grid_seamless(
                bbox_orig=bbox,
                union_dem_path=union_dem,
                upsample_factor=UPSAMPLE_FACTOR,
            )
        except Exception as e:
            print(f"  ✗ {z}/{x}/{y} z_grid: {e}")
            err += 1
            continue

        try:
            extruder = BuildingExtruder(output_dir=str(td))
            bld_obj = extruder.generate(buildings, z_grid, meta, obj_name=str(y))
        except Exception as e:
            print(f"  ✗ {z}/{x}/{y} buildings: {e}")
            err += 1
            continue

        # Determine SDF path
        sdf_path = str(td / f"{y}_roads_sdf.png")
        has_sdf  = os.path.exists(sdf_path)

        import numpy as np
        cx      = meta["total_width_m"] / 2
        cy      = meta["total_height_m"] / 2
        elev_min = float(np.nanmin(z_grid))
        elev_max = float(np.nanmax(z_grid))

        pipeline_result = {
            "terrain":     str(terrain_obj),
            "roads":       None,
            "buildings":   bld_obj,
            "sdf_texture": sdf_path if has_sdf else None,
        }

        try:
            exporter = GltfExporter(output_dir=str(td), use_draco=True)
            export_fn = (
                exporter.export_draco if exporter.use_draco else exporter.export
            )
            export_fn(
                pipeline_result,
                output_name=str(y),
                tile_center_local=(cx, cy),
                elev_min=elev_min,
                elev_max=elev_max,
                has_sdf=has_sdf,
            )
            ok += 1
        except Exception as e:
            print(f"  ✗ {z}/{x}/{y} GLB: {e}")
            err += 1
            continue

        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] ok={ok}  err={err}  skip={skip}")

    print(f"\nDone. ok={ok}  err={err}  skip={skip}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python regen_buildings.py <output_dir>")
    main(sys.argv[1])
