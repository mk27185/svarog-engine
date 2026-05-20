#!/usr/bin/env python3
"""
Regenerate *only* the SDF road PNG textures for existing tile output directories.

Usage:
    python regen_sdf.py outputs/prague-center/

Works entirely from cached data (no network access needed):
  - reads  {tile_dir}/{y}_osm_highways.json   for road geometries
  - writes {tile_dir}/{y}_roads_sdf.png        (overwrites old PNG)

Meta is reconstructed from the tile (z, x, y) coordinates so the SDF
coordinate system exactly matches the terrain OBJ produced by build_grid_seamless.
"""

import json
import math
import os
import sys
from pathlib import Path

from src.conversion.sdf_generator import SDFGenerator


# ── slippy tile helpers ────────────────────────────────────────────────────────

def tile_to_latlon(z: int, x: int, y: int) -> tuple[float, float]:
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def tile_meta(z: int, x: int, y: int) -> dict:
    """Reconstruct the meta dict needed by SDFGenerator from tile xyz."""
    n_lat, w_lon = tile_to_latlon(z, x, y)
    s_lat, e_lon = tile_to_latlon(z, x + 1, y + 1)

    lat_mean_rad = math.radians((n_lat + s_lat) / 2)
    mpd_lat = 111_320.0
    mpd_lon = 111_320.0 * math.cos(lat_mean_rad)

    return {
        "lat_origin":        s_lat,
        "lon_origin":        w_lon,
        "meters_per_deg_lat": mpd_lat,
        "meters_per_deg_lon": mpd_lon,
        "total_width_m":     (e_lon - w_lon) * mpd_lon,
        "total_height_m":    (n_lat - s_lat) * mpd_lat,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main(output_root: str) -> None:
    root = Path(output_root)
    if not root.exists():
        sys.exit(f"Directory not found: {root}")

    gen = SDFGenerator(resolution=1024)

    # Structure: <root>/<z>/<x>/<y>/
    tile_dirs = sorted(root.glob("[0-9]*/*/*"))
    tile_dirs = [d for d in tile_dirs if d.is_dir()]

    total = len(tile_dirs)
    print(f"Found {total} tile directories under {root}")

    ok = err = skip = 0
    for i, td in enumerate(tile_dirs, 1):
        # Path parts: .../<root>/<z>/<x>/<y>
        try:
            y_str = td.name
            x_str = td.parent.name
            z_str = td.parent.parent.name
            z, x, y = int(z_str), int(x_str), int(y_str)
        except ValueError:
            skip += 1
            continue

        hw_path  = td / f"{y}_osm_highways.json"
        sdf_path = td / f"{y}_roads_sdf.png"

        if not hw_path.exists():
            skip += 1
            continue

        with open(hw_path) as f:
            highways = json.load(f)

        meta = tile_meta(z, x, y)
        try:
            gen.generate(highways, meta, str(sdf_path))
            ok += 1
        except Exception as e:
            print(f"  ✗ {z}/{x}/{y}: {e}")
            err += 1

        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] ok={ok} err={err} skip={skip}")

    print(f"\nDone. ok={ok}  err={err}  skip={skip}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python regen_sdf.py <output_dir>")
    main(sys.argv[1])
