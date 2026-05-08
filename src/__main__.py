"""
svarog-engine CLI
=================
    python -m svarog_engine --help

Examples
--------
    # Bbox — tile size comes from svarog-contracts/world-config.json
    python -m svarog_engine --bbox "50.07,14.43,50.09,14.46" --output outputs/test/

    # Single XYZ tile (explicit z/x/y)
    python -m svarog_engine --tile 15/17698/11100 --output outputs/test/

    # Batch config YAML
    python -m svarog_engine --config tiles.yaml

    # Override tile size or Draco for this run only
    python -m svarog_engine --bbox "50.07,14.43,50.09,14.46" --tile-size 200 --draco --output outputs/

    # Force re-download of OSM data even when a cache exists
    python -m svarog_engine --tile 15/17698/11100 --force-osm --output outputs/test/

Options override world-config.json from svarog-contracts where applicable.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m svarog_engine",
        description="Generate terrain GLB tiles from OpenTopography + OSM data.",
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
                        help="Re-download OSM data even when a cache exists")

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
        if len(parts) != 4:
            parser.error("--bbox must be S,W,N,E (4 values)")
        tiles = TileSplitter.xyz_grid(tuple(parts), tile_size_m)
        zoom = TileSplitter.zoom_for_tile_size(tile_size_m,
                                               lat=(parts[0] + parts[2]) / 2)
        print(f"bbox → {len(tiles)} tile(s)  "
              f"(tile_size_m={tile_size_m} → zoom {zoom})")

    if not tiles:
        sys.exit("No tiles to process.")

    from src.extraction.opentopography_client import OpenTopographyClient
    from src.extraction.osm_client import OsmClient
    from src.conversion.terrain_converter import TerrainConverter
    from src.conversion.building_extruder import BuildingExtruder
    from src.conversion.sdf_generator import SDFGenerator
    from src.conversion.gltf_exporter import GltfExporter
    from src.pipeline.terrain_pipeline import TerrainPipeline
    from src.pipeline.task_runner import TaskRunner

    pipeline = TerrainPipeline(
        client=OpenTopographyClient(),
        converter=TerrainConverter(output_dir=output_root),
        osm_client=OsmClient(),
        sdf_generator=SDFGenerator(),
        building_extruder=BuildingExtruder(output_dir=output_root),
        gltf_exporter=GltfExporter(output_dir=output_root, use_draco=use_draco),
        upsample_factor=upsample,
        osm_force_download=args.force_osm,
    )

    summary = TaskRunner(
        pipeline=pipeline,
        output_root=output_root,
        stop_on_error=args.stop_on_error,
    ).run(tiles, manifest_path=os.path.join(output_root, "tile-manifest.json"))

    print(f"\nDone: {summary['completed']}/{summary['total']} tiles")
    print(f"Manifest: {summary['manifest']}")


if __name__ == "__main__":
    main()
