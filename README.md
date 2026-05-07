# Svarog Engine

Terrain pipeline that converts OpenTopography DEM + OpenStreetMap data into 3D OBJ meshes and SDF road textures.

**Outputs:** `_terrain.obj` · `_roads.obj` · `_buildings.obj` · `_roads_sdf.png` · `.glb` (Three.js ready)

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Generating Output Files

All generation commands require an active venv and must be run from the repo root.

### CLI (recommended)

```bash
source venv/bin/activate

# Single bbox → XYZ grid at zoom from world-config (default zoom 15)
python -m svarog_engine --bbox "50.07,14.43,50.09,14.46" --output outputs/test/

# City name → geocode via Nominatim → XYZ grid
python -m svarog_engine --city "Prague" --output outputs/prague/

# Specific XYZ tile
python -m svarog_engine --tile 15/17698/11100 --output outputs/test/

# Batch config file
python -m svarog_engine --config tiles.yaml

# Override zoom and enable Draco
python -m svarog_engine --city "Prague" --zoom 18 --draco --output outputs/prague_hd/
```

Tile size is controlled globally via `svarog-contracts/world-config.json` (`glb_zoom`):
- zoom 15 ≈ **785 m** per tile in Prague
- zoom 17 ≈ **196 m** per tile in Prague
- zoom 18 ≈ **98 m** per tile in Prague

### Batch YAML config

```yaml
# tiles.yaml
output_root: outputs/my_world
upsample_factor: 10   # optional override
draco: true           # optional override
tiles:
  - name: prague_center
    bbox: [50.07, 14.43, 50.08, 14.44]
  - xyz: [15, 17698, 11100]   # XYZ tile — bbox derived automatically
```

```bash
python -m svarog_engine --config tiles.yaml
```

### Single tile (legacy / manual)

```bash
source venv/bin/activate
python -m src.pipeline.terrain_pipeline
```

Hardcoded Prague test area, outputs to `outputs/test_results/`.

### Batch run (multiple tiles)

Create a script in the repo root, e.g. `run_batch.py`:

```python
from src.extraction.opentopography_client import OpenTopographyClient
from src.extraction.osm_client import OsmClient
from src.conversion.terrain_converter import TerrainConverter
from src.conversion.road_mesh import RoadMesh
from src.conversion.building_extruder import BuildingExtruder
from src.pipeline.terrain_pipeline import TerrainPipeline
from src.pipeline.task_runner import TaskRunner, TileConfig

output_root = "outputs/batch"

from src.conversion.gltf_exporter import GltfExporter

pipeline = TerrainPipeline(
    client=OpenTopographyClient(),
    converter=TerrainConverter(output_dir=output_root),
    road_mesh=RoadMesh(output_dir=output_root),
    osm_client=OsmClient(),
    building_extruder=BuildingExtruder(output_dir=output_root),
    gltf_exporter=GltfExporter(output_dir=output_root),  # omit for OBJ-only output
    upsample_factor=10,
)

tiles = [
    TileConfig(name="prague_center", bbox=(50.07, 14.43, 50.08, 14.44)),
    TileConfig(name="prague_north",  bbox=(50.11, 14.40, 50.13, 14.43)),
]

runner = TaskRunner(pipeline=pipeline, output_root=output_root)
runner.run(tiles, manifest_path=f"{output_root}/tile-manifest.json")
```

Then run:

```bash
source venv/bin/activate
python run_batch.py
```

Each tile gets its own subdirectory under `output_root`. A `tile-manifest.json` is written with the status and output paths of every tile.

### Output files per tile

| File | Description |
|---|---|
| `<name>_terrain.obj` | Terrain mesh (regular quad grid, bilinear upsampled SRTM) |
| `<name>_roads.obj` | 3D road strips with side walls |
| `<name>_buildings.obj` | Extruded OSM building footprints |
| `<name>_roads_sdf.png` | RGBA SDF road texture (R=distance, G=rank, B=width, A=mask) |
| `<name>.glb` | Combined GLB (terrain + roads + buildings), Y-up, Three.js ready |
| `<name>.glb` (Draco) | Same but with `KHR_draco_mesh_compression` — needs `DRACOLoader` in Three.js |

### Generating a GLB quickly (single tile)

The fastest way to get a `.glb` for inspection — create `run_glb.py` in the repo root:

```python
from src.extraction.opentopography_client import OpenTopographyClient
from src.extraction.osm_client import OsmClient
from src.conversion.terrain_converter import TerrainConverter
from src.conversion.road_mesh import RoadMesh
from src.conversion.building_extruder import BuildingExtruder
from src.conversion.gltf_exporter import GltfExporter
from src.pipeline.terrain_pipeline import TerrainPipeline

output_dir = "outputs/glb_test"

pipeline = TerrainPipeline(
    client=OpenTopographyClient(),
    converter=TerrainConverter(output_dir=output_dir),
    road_mesh=RoadMesh(output_dir=output_dir),
    osm_client=OsmClient(),
    building_extruder=BuildingExtruder(output_dir=output_dir),
    gltf_exporter=GltfExporter(output_dir=output_dir),
    upsample_factor=10,
)

result = pipeline.run_pipeline((50.07, 14.43, 50.08, 14.44), "prague")
print("GLB:", result["glb"])
```

```bash
source venv/bin/activate
python run_glb.py
```

Output: `outputs/glb_test/prague.glb`

For a Draco-compressed version, call `gltf_exporter.export_draco(result, output_name="prague_draco")` after `run_pipeline`.

### Inspecting the GLB

**Blender** — File → Import → glTF 2.0 (.glb/.gltf). The scene contains three named objects: `terrain`, `roads`, `buildings`. Coordinate system is Y-up so the scene renders upright immediately.

**Online** — drag the `.glb` into [gltf.report](https://gltf.report) or [sandbox.babylonjs.com](https://sandbox.babylonjs.com) for a quick no-install check.

### Inspecting in Blender (OBJ)

All `.obj` files use the same local metric coordinate system (origin = SW corner of the tile), so they can be imported together and will align correctly.

**File → Import → Wavefront (.obj)** — select all three OBJ files at once.

---

## Running Tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

All tests are fully offline — no network calls, no real TIF files. Shared fixtures (synthetic elevation grid, mock OSM/DEM clients) live in `tests/conftest.py`.

### Test coverage

| Module | Test file | Tests |
|---|---|---|
| `TerrainConverter` | `test_terrain_converter.py` | grid shape, elevation range, upsample, OBJ vertex count, UV coords |
| `TerrainStamper` | `test_terrain_stamper.py` | stamp shape, road blending, no-road passthrough, half-width lookup |
| `RoadMeshGenerator` | `test_road_mesh.py` | OBJ output, vertex/face count, junction Z, cross-section geometry |
| `BuildingExtruder` | `test_building_extruder.py` | extrusion, wall/roof face count, height tags, empty input |
| `SDFGenerator` | `test_sdf_generator.py` | RGBA range, alpha mask, output dtype |
| `TerrainPipeline` | `test_terrain_pipeline.py` | end-to-end with mock data, missing OSM client, DEM failure |

### Useful flags

```bash
# Run a single test file
python -m pytest tests/test_terrain_converter.py -v

# Run a single test by name
python -m pytest tests/ -k "test_stamp_lowers_near_road" -v

# Stop on first failure
python -m pytest tests/ -x

# Show stdout (print statements)
python -m pytest tests/ -s
```

---

## Project Structure

```
src/
  conversion/
    terrain_converter.py   # GeoTIFF → OBJ terrain mesh
    terrain_stamper.py     # Blend road geometry into terrain elevation
    road_mesh.py           # 3D road strips with side walls
    building_extruder.py   # OSM building footprints → extruded OBJ
    sdf_generator.py       # Multi-channel SDF road texture (RGBA PNG)
    cdt_mesh.py            # CDT triangulation helper
    geo_utils.py           # Shared: build_meta, to_local, sample_z, subdivide_polyline
  extraction/
    osm_client.py          # Download OSM highways + buildings
    opentopography_client.py  # Download SRTM DEM (GeoTIFF)
  pipeline/
    terrain_pipeline.py    # Orchestrates the full pipeline end-to-end
tests/
  conftest.py              # Shared fixtures (synthetic grid, mock clients)
  test_*.py                # One file per module
```
