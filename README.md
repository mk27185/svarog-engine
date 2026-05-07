# Svarog Engine

Terrain pipeline that converts OpenTopography DEM + OpenStreetMap data into 3D OBJ meshes and SDF road textures.

**Outputs:** `_terrain.obj` · `_roads.obj` · `_buildings.obj` · `_roads_sdf.png`

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

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
