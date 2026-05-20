"""
CLI entry with batched Overpass (v2) for multi-tile bbox / YAML batches.

Run from repo root::

    python -m src.cli_overpass_v2 --bbox "49.43,14.11,50.10,15.07" --output outputs/run/

Legacy single-tile / per-tile Overpass remains::

    python -m src ...
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli_overpass_v2",
        description=(
            "Generate terrain GLB tiles — multi-tile batches use 2× Overpass "
            "(union bbox) + local clip per tile. Single-tile runs behave like v1."
        ),
    )

    area = parser.add_mutually_exclusive_group(required=False)
    area.add_argument(
        "--bbox", metavar="S,W,N,E",
        help="Bounding box in decimal degrees: south,west,north,east",
    )
    area.add_argument(
        "--tile", metavar="Z/X/Y",
        help="Single XYZ tile (e.g. 15/17698/11100)",
    )
    area.add_argument(
        "--config", metavar="FILE",
        help="YAML batch config with tiles list",
    )

    parser.add_argument("--output", metavar="DIR", default="outputs",
                        help="Output root directory (default: outputs)")
    parser.add_argument("--tile-size", type=int, default=None, metavar="M",
                        help="Tile edge in metres — overrides world-config.json tile_size_m")
    parser.add_argument("--draco", action="store_true", default=None,
                        help="Produce Draco-compressed GLB (overrides world-config)")
    parser.add_argument("--no-draco", dest="draco", action="store_false",
                        help="Force uncompressed GLB")
    parser.add_argument("--upsample", type=int, default=None, metavar="N",
                        help="DEM upsample factor (overrides world-config)")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort batch on first failed tile")
    parser.add_argument("--force-osm", action="store_true",
                        help="Re-download OSM (union batch) even when per-tile cache exists")

    args = parser.parse_args()

    if args.config is None and args.bbox is None and args.tile is None:
        parser.print_help()
        sys.exit(0)

    from src.pipeline.world_config import WorldConfig
    world = WorldConfig.load()

    tile_size_m = args.tile_size if args.tile_size is not None else world.tile_size_m
    use_draco   = args.draco     if args.draco    is not None else world.draco
    upsample    = args.upsample  if args.upsample is not None else world.upsample_factor

    from src.pipeline.task_runner import TileConfig
    from src.pipeline.tile_splitter import TileSplitter

    tiles: list[TileConfig] = []
    output_root = args.output

    if args.config:
        from src.pipeline.config_loader import ConfigLoader
        batch = ConfigLoader.load(args.config)
        tiles = batch.tiles
        output_root = batch.output_root
        if batch.upsample_factor is not None:
            upsample = batch.upsample_factor
        if batch.draco is not None:
            use_draco = batch.draco

    elif args.tile:
        z, x, y = [int(p) for p in args.tile.split("/")]
        tiles = [TileSplitter.from_xyz(z, x, y)]

    elif args.bbox:
        parts = [float(v) for v in args.bbox.split(",")]
        if len(parts) not in (4,):
            parser.error("--bbox requires 4 comma-separated values: lat1,lon1,lat2,lon2")
        # Accept any two-corner order — normalise to (south, west, north, east)
        lats = sorted([parts[0], parts[2]])
        lons = sorted([parts[1], parts[3]])
        bbox = (lats[0], lons[0], lats[1], lons[1])
        tiles = TileSplitter.xyz_grid(bbox, tile_size_m)
        zoom = TileSplitter.zoom_for_tile_size(tile_size_m,
                                               lat=(bbox[0] + bbox[2]) / 2)
        print(f"bbox → {len(tiles)} tile(s)  "
              f"(tile_size_m={tile_size_m} → zoom {zoom})  "
              f"[S={bbox[0]}, W={bbox[1]}, N={bbox[2]}, E={bbox[3]}]")

    if not tiles:
        sys.exit("No tiles to process.")

    from src.extraction.opentopography_client import OpenTopographyClient
    from src.extraction.osm_client import OsmClient
    from src.conversion.terrain_converter import TerrainConverter
    from src.conversion.building_extruder import BuildingExtruder
    from src.conversion.sdf_generator import SDFGenerator
    from src.conversion.landcover_generator import LandcoverGenerator
    from src.conversion.navmesh_builder import NavmeshBuilder
    from src.conversion.gltf_exporter import GltfExporter
    from src.pipeline.terrain_pipeline import TerrainPipeline
    from src.pipeline.task_runner import TaskRunner
    from src.pipeline.task_runner_overpass_v2 import run_task_batch_overpass_v2

    pipeline = TerrainPipeline(
        client=OpenTopographyClient(),
        converter=TerrainConverter(output_dir=output_root),
        osm_client=OsmClient(),
        sdf_generator=SDFGenerator(),
        landcover_generator=LandcoverGenerator(),
        navmesh_builder=NavmeshBuilder(),
        building_extruder=BuildingExtruder(output_dir=output_root),
        gltf_exporter=GltfExporter(output_dir=output_root, use_draco=use_draco),
        upsample_factor=upsample,
        osm_force_download=args.force_osm,
    )

    tile_zoom = tiles[0].xyz.z if tiles and tiles[0].xyz is not None else zoom
    summary = run_task_batch_overpass_v2(
        TaskRunner(
            pipeline=pipeline,
            output_root=output_root,
            zoom=tile_zoom,
            stop_on_error=args.stop_on_error,
        ),
        tiles=tiles,
        manifest_path=os.path.join(output_root, "tile-manifest.json"),
        output_root=output_root,
        force_osm=args.force_osm,
    )

    print(f"\nDone: {summary['completed']}/{summary['total']} tiles")
    print(f"Manifest: {summary['manifest']}")


if __name__ == "__main__":
    main()
