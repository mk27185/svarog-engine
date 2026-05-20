# Svarog Engine

Terrain pipeline converting OpenTopography DEM + OpenStreetMap data into 3D OBJ meshes, SDF road textures and GLB tiles for Three.js.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Generating tiles

All commands are run from the repo root with the venv active.

### Bounding box → tile grid

```bash
# Bbox in decimal degrees (S,W,N,E) — splits into XYZ tiles at zoom from world-config
python -m src --bbox "50.07,14.43,50.09,14.46" --output outputs/test/

# Force Draco compression (zoom / tile geometry still follows world-config tile_size_m)
python -m src --bbox "50.07,14.43,50.09,14.46" --draco --output outputs/hd/
```

### Single XYZ tile

```bash
python -m src --tile 15/17698/11100 --output outputs/test/
```

### Batch YAML config

```yaml
# tiles.yaml
output_root: outputs/my_world
upsample_factor: 10  # optional override
draco: true          # optional override
tiles:
  - name: center
    bbox: [50.07, 14.43, 50.08, 14.44]
  - xyz: [15, 17698, 11100]
```

```bash
python -m src --config tiles.yaml
```

### Global config

Tile zoom, Draco and upsample defaults live in `../svarog-contracts/world-config.json`:

| Key | Default | Effect |
|---|---|---|
| `glb_zoom` | 15 | XYZ zoom level (zoom 15 ≈ 785 m/tile, zoom 18 ≈ 98 m/tile) |
| `draco` | true | Draco-compress the GLB |
| `upsample_factor` | 10 | DEM upsample (30 m SRTM → ~3 m grid) |

### Output files per tile

| File | Description |
|---|---|
| `<name>_terrain.obj` | Terrain quad mesh (bilinear upsampled SRTM) |
| `<name>_roads.obj` | 3D road strips with side walls |
| `<name>_buildings.obj` | Extruded OSM building footprints |
| `<name>_roads_sdf.png` | RGBA SDF road texture (R=distance, G=rank, B=width, A=mask) |
| `<name>.glb` | Combined GLB (terrain + roads + buildings), Y-up, Three.js ready |
| `<name>_osm_highways.json` | Cached OSM highway data (re-used on subsequent runs) |
| `<name>_osm_buildings.json` | Cached OSM building + building:part data (re-used on subsequent runs) |

For XYZ tiles the structure is `outputs/{z}/{x}/{y}/`. A `tile-manifest.json` is written to the output root.

### OSM data cache

After the first successful Overpass API download, highway and building data are saved as JSON files next to the tile outputs. Subsequent runs for the same tile load from these files instead of querying Overpass, which avoids rate-limiting and speeds up re-generation (e.g. when tweaking terrain parameters).

```bash
# Normal run — uses cache if present, downloads otherwise
python -m src --tile 15/17698/11100 --output outputs/test/

# Force re-download of OSM data (e.g. after the map has been updated)
python -m src --tile 15/17698/11100 --force-osm --output outputs/test/
```

To clear the cache for a tile manually, delete the two `_osm_*.json` files in its output directory.

### Inspecting output

**GLB in Blender** — File → Import → glTF 2.0. The scene has three named objects: `terrain`, `roads`, `buildings`. Coordinate system is Y-up, renders upright immediately.

**GLB online** — drop the file into [gltf.report](https://gltf.report) or [sandbox.babylonjs.com](https://sandbox.babylonjs.com).

**OBJ in Blender** — File → Import → Wavefront (.obj). All three OBJ files share the same local metric origin (SW corner), so importing them together aligns correctly. The `_roads_sdf.png` can be loaded as a texture on the terrain mesh to visualise the SDF.

---

## Running tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

All tests are fully offline — no network calls, no real TIF files. Shared fixtures live in `tests/conftest.py`.

| Module | Test file | What's covered |
|---|---|---|
| `TerrainConverter` | `test_terrain_converter.py` | grid shape, elevation range, upsample, OBJ vertex count, UV coords |
| `TerrainStamper` | `test_terrain_stamper.py` | stamp shape, road blending, no-road passthrough, half-width lookup |
| `RoadMesh` | `test_road_mesh.py` | OBJ output, vertex/face count, junction Z, cross-section geometry |
| `BuildingExtruder` | `test_building_extruder.py` | extrusion, wall/roof face count, height/min_height tags, CDT roof triangulation, terrain-following base, empty input, roof shapes (pyramid/cone, skillion, gabled/hipped, dome/onion) |
| `SDFGenerator` | `test_sdf_generator.py` | RGBA range, alpha mask, output dtype |
| `TerrainPipeline` | `test_terrain_pipeline.py` | end-to-end with mock data, missing OSM client, DEM failure |
| `GltfExporter` | `test_gltf_exporter.py` | GLB output, Draco output, vertex centering |
| `TileSplitter` | `test_tile_splitter.py` | XYZ ↔ bbox, tile size, center coords |
| `TaskRunner` | `test_task_runner.py` | batch run, manifest, error handling |
| `WorldConfig` | `test_world_config.py` | load defaults, override values |

Useful flags:

```bash
python -m pytest tests/test_terrain_converter.py -v   # single file
python -m pytest tests/ -k "test_stamp_lowers"        # by name
python -m pytest tests/ -x                            # stop on first failure
python -m pytest tests/ -s                            # show print output
```

---

## Project structure

```
src/
  __main__.py               # CLI entry point (python -m src from repo root)
  conversion/
    terrain_converter.py    # GeoTIFF → terrain grid + OBJ
    terrain_stamper.py      # Blend road geometry into elevation
    road_mesh.py            # 3D road strips with side walls
    building_extruder.py    # OSM building footprints → extruded OBJ
    sdf_generator.py        # Multi-channel SDF road texture (RGBA PNG)
    gltf_exporter.py        # Combine OBJs into GLB (plain or Draco)
    cdt_mesh.py             # CDT triangulation helper
    geo_utils.py            # build_meta, to_local, sample_z, subdivide_polyline
    plugin.py               # RoadPlugin / BuildingPlugin Protocol interfaces
  extraction/
    osm_client.py           # OSM highways + buildings (incl. building:part S3DB)
    opentopography_client.py# SRTM DEM (GeoTIFF)
  pipeline/
    terrain_pipeline.py     # Orchestrates the full pipeline
    task_runner.py          # Batch execution + TileConfig
    tile_manifest.py        # Write / validate tile-manifest.json
    tile_splitter.py        # XYZ tile ↔ bbox / mercator conversions
    world_config.py         # Load svarog-contracts/world-config.json
    config_loader.py        # Load YAML batch config
tests/
  conftest.py               # Shared fixtures
  test_*.py                 # One file per module
```
