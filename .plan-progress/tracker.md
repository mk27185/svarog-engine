# Svarog-Engine Progress Trackerjed
**Last updated:** 2026-05-06

---

## What's Done ✅
- [x] Project structure (`src/conversion/`, `src/extraction/`, `src/pipeline/`, `tests/`)
- [x] `OSMClient` — download & parse OSM data
- [x] `OpenTopographyClient` — download DEM data
- [x] `TiffToObjConverter` — DEM → OBJ terrain mesh
- [x] `TerrainStamper` — stamp roads, water, green areas onto terrain
- [x] `SDFGenerator` — multi-channel RGBA road texture (SDF + rank + width + mask)
- [x] `RoadMeshGenerator` — OBJ road mesh from OSM
- [x] `BuildingExtruder` + `CDTMesher` — building extrusion from OSM polygons
- [x] `TerrainPipeline` — orchestrates full pipeline end-to-end
- [x] Unit tests extraction & SDF generator (4 SDF tests Passing)
- [x] Test run with Prague data — full pipeline generates `.obj` models + SDF texture (`outputs/test_sdf_run/`)

## What's Remaining 🔜
### High Priority
- [x] Run all unit tests → **DONE** (pass all 4)
- [x] .plan-progress tracking → **DONE (this file)**
- [ ] Fix SDF Generator bug: nodes dict → tuple (already fixed in sdf_generator.py line 106)
- [ ] Unit tests for: `TiffToObjConverter`, `TerrainStamper`, `RoadMeshGenerator`, `BuildingExtruder`
- [ ] Unit tests for `TerrainPipeline` (end-to-end with mock data)
- [ ] Add `pytest.ini`/`conftest.py` with shared fixtures

### Phase 2 — Modularization (from analysis doc)
- [ ] Extract Road Generation and Building Anchoring as plug-in modules  
- [ ] Task Runner — batch-tile orchestration
- [ ] Tile manifest (`tile-manifest.json`) with schema validation  
- [ ] Output GLB/GLTF instead of OBJ + Draco compression  
- [ ] NavMesh Factory (recast-navigation integration)

## Known Issues
- Road SDF texture channels are correct (R=SDF, G=rank, B=width, A=mask) but very narrow roads (~5 px) produce SDF that clusters around 128 (edge) because the road center isn't far enough from the edge to register significant positive SDF.
- `test_sdf_has_range` tests overall range rather than per-road accuracy — acceptable given narrow test data.

## How to Run Tests
```bash
cd svarog-engine
source venv/bin/activate
python -m pytest tests/ -v
```
**Principle:** Each converter/extractor is a stateless class. Tests supply synthetic input (coordinates, tags, pixel arrays) and assert on dimensions, value ranges, and geometric sanity.
