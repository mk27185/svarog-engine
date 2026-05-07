"""
svarog-engine CLI
=================
    python -m svarog_engine --help

Examples
--------
    # Single bbox
    python -m svarog_engine --bbox "50.07,14.43,50.08,14.44" --output outputs/test/

    # City name → full XYZ tile grid at zoom from world-config
    python -m svarog_engine --city "Prague" --output outputs/prague/

    # Specific XYZ tile
    python -m svarog_engine --tile 15/17898/11245 --output outputs/prague/

    # Batch config YAML
    python -m svarog_engine --config tiles.yaml

    # Override Draco and zoom
    python -m svarog_engine --city "Prague" --zoom 18 --draco --output outputs/

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

    # ── Area input ────────────────────────────────────────────────────────────
    area = parser.add_mutually_exclusive_group(required=False)
    area.add_argument(
        "--bbox", metavar="S,W,N,E",
        help="Bounding box in decimal degrees: south,west,north,east",
    )
    area.add_argument(
        "--city", metavar="NAME",
        help="Place name resolved via Nominatim geocoding (e.g. 'Prague')",
    )
    area.add_argument(
        "--tile", metavar="Z/X/Y",
        help="Single XYZ slippy-map tile (e.g. 15/17898/11245)",
    )
    area.add_argument(
        "--config", metavar="FILE",
        help="YAML batch config with tiles list",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--output", metavar="DIR", default="outputs",
                        help="Output root directory (default: outputs)")

    # ── Overrides ─────────────────────────────────────────────────────────────
    parser.add_argument("--zoom", type=int, default=None,
                        help="XYZ zoom level (overrides world-config)")
    parser.add_argument("--draco", action="store_true", default=None,
                        help="Produce Draco-compressed GLB (overrides world-config)")
    parser.add_argument("--no-draco", dest="draco", action="store_false",
                        help="Force uncompressed GLB")
    parser.add_argument("--upsample", type=int, default=None, metavar="N",
                        help="DEM upsample factor (overrides world-config)")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort batch on first failed tile")

    args = parser.parse_args()

    if args.config is None and args.bbox is None and args.city is None and args.tile is None:
        parser.print_help()
        sys.exit(0)

    # ── Load world config ─────────────────────────────────────────────────────
    from src.pipeline.world_config import WorldConfig
    world = WorldConfig.load()

    zoom           = args.zoom     if args.zoom     is not None else world.glb_zoom
    use_draco      = args.draco    if args.draco    is not None else world.draco
    upsample       = args.upsample if args.upsample is not None else world.upsample_factor

    # ── Resolve tiles ─────────────────────────────────────────────────────────
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
        bbox = tuple(parts)
        tiles = TileSplitter.xyz_grid(bbox, zoom)
        print(f"Bbox → {len(tiles)} tile(s) at zoom {zoom}")

    elif args.city:
        from src.extraction.geocoder import Geocoder, GeocoderError
        try:
            bbox = Geocoder.city_to_bbox(args.city)
            print(f"'{args.city}' → bbox {bbox}")
        except GeocoderError as e:
            sys.exit(f"Geocoding failed: {e}")
        tiles = TileSplitter.xyz_grid(bbox, zoom)
        print(f"→ {len(tiles)} tile(s) at zoom {zoom}")

    if not tiles:
        sys.exit("No tiles to process.")

    # ── Build pipeline ────────────────────────────────────────────────────────
    from src.extraction.opentopography_client import OpenTopographyClient
    from src.extraction.osm_client import OsmClient
    from src.conversion.terrain_converter import TerrainConverter
    from src.conversion.road_mesh import RoadMesh
    from src.conversion.building_extruder import BuildingExtruder
    from src.conversion.gltf_exporter import GltfExporter
    from src.pipeline.terrain_pipeline import TerrainPipeline
    from src.pipeline.task_runner import TaskRunner

    pipeline = TerrainPipeline(
        client=OpenTopographyClient(),
        converter=TerrainConverter(output_dir=output_root),
        road_mesh=RoadMesh(output_dir=output_root),
        osm_client=OsmClient(),
        building_extruder=BuildingExtruder(output_dir=output_root),
        gltf_exporter=GltfExporter(output_dir=output_root, use_draco=use_draco),
        upsample_factor=upsample,
    )

    manifest_path = os.path.join(output_root, "tile-manifest.json")
    runner = TaskRunner(
        pipeline=pipeline,
        output_root=output_root,
        stop_on_error=args.stop_on_error,
    )
    summary = runner.run(tiles, manifest_path=manifest_path)

    print(f"\nDone: {summary['completed']}/{summary['total']} tiles")
    print(f"Manifest: {summary['manifest']}")


if __name__ == "__main__":
    main()
