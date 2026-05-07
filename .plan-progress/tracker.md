# Svarog-Engine Progress Tracker
**Last updated:** 2026-05-07 (11:45)

---

## Phase 1 — Done ✅

- [x] Project structure (`src/conversion/`, `src/extraction/`, `src/pipeline/`, `tests/`)
- [x] `OSMClient` — download & parse OSM data
- [x] `OpenTopographyClient` — download DEM data
- [x] `TiffToObjConverter` — DEM → OBJ terrain mesh
- [x] `TerrainStamper` — stamp roads, water, green areas onto terrain
- [x] `SDFGenerator` — multi-channel RGBA road texture (SDF + rank + width + mask)
- [x] `RoadMeshGenerator` — OBJ road mesh from OSM
- [x] `BuildingExtruder` + `CDTMesher` — building extrusion from OSM polygons
- [x] `TerrainPipeline` — orchestrates full pipeline end-to-end
- [x] Test run with Prague data — full pipeline generates `.obj` models + SDF texture (`outputs/test_sdf_run/`)
- [x] **Refactoring:** `subdivide_polyline()` and `smooth_profile()` extracted to `geo_utils.py`
- [x] Unit tests — **44 tests, all passing** (`TiffToObjConverter`, `TerrainStamper`, `RoadMeshGenerator`, `BuildingExtruder`, `TerrainPipeline`, `SDFGenerator`)
- [x] `pytest.ini` + `conftest.py` with shared fixtures
- [x] Fix `TerrainStamper` bug: node dict keys unpacked as strings instead of values

---

## Phase 2 — Modularization 🔜

### Plug-in Architecture
- [x] Extract Road Generation as a plug-in module — `RoadPlugin` protocol v `plugin.py`, `RoadMesh.generate()` implementuje protokol, `generate_obj` alias pro zpětnou kompatibilitu
- [x] Extract Building Anchoring as a plug-in module — `BuildingPlugin` protocol, `BuildingExtruder.generate()` wrapper nad `extrude_buildings()`
- [x] `TerrainPipeline` přijímá libovolný objekt splňující protokol (type hints `RoadPlugin | None`, `BuildingPlugin | None`), volá `.generate()` na obou

### Batch Processing
- [x] Task Runner — `src/pipeline/task_runner.py` · `TaskRunner(pipeline, output_root)` · `runner.run(tiles, manifest_path)` · `stop_on_error` flag
- [x] Tile manifest — `src/pipeline/tile_manifest.py` · `TileManifest` validuje přes `jsonschema` proti schématu z `svarog-contracts/schemas/tile-manifest.json` · schéma rozšířeno o `tiles[]` se statusem, bbox, outputs a error

### Output Formats
- [ ] Output GLB/GLTF instead of OBJ
- [ ] Draco mesh compression

### Navigation
- [ ] NavMesh Factory — recast-navigation integration

---

## Known Issues
- Road SDF texture channels are correct (R=SDF, G=rank, B=width, A=mask) but very narrow roads (~5 px) produce SDF clustering around 128 (edge), because the road center isn't far enough from the edge to register significant positive SDF.
- `test_sdf_has_range` tests overall range rather than per-road accuracy — acceptable given narrow test data.

---

## How to Run Tests
```bash
cd svarog-engine
source venv/bin/activate
python -m pytest tests/ -v
```
**Principle:** Each converter/extractor is a stateless class. Tests supply synthetic input (coordinates, tags, pixel arrays) and assert on dimensions, value ranges, and geometric sanity.
