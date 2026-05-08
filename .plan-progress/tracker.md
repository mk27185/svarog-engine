# Svarog-Engine Progress Tracker
**Last updated:** 2026-05-07 (13:00)

---

## Phase 1 — Core Pipeline ✅

- [x] Project structure (`src/conversion/`, `src/extraction/`, `src/pipeline/`, `tests/`)
- [x] `OSMClient` — download & parse OSM highways + buildings
- [x] `OpenTopographyClient` — download SRTM DEM (GeoTIFF)
- [x] `TerrainConverter` — DEM → bilinear-upsampled grid + OBJ terrain mesh
- [x] `TerrainStamper` — blend road geometry into elevation grid
- [x] `SDFGenerator` — multi-channel RGBA road SDF texture (distance / rank / width / mask)
- [x] `RoadMesh` — 3D road strips with side walls from OSM highways
- [x] `BuildingExtruder` + CDT triangulation — extruded building footprints from OSM
- [x] `TerrainPipeline` — orchestrates full pipeline end-to-end
- [x] `pytest.ini` + `conftest.py` + 44 unit tests (all modules, fully offline)

---

## Phase 2 — Modularization ✅

### Plug-in Architecture
- [x] `RoadPlugin` / `BuildingPlugin` `@runtime_checkable` Protocol interfaces (`src/conversion/plugin.py`)
- [x] `RoadMesh.generate()` implements `RoadPlugin`; `generate_obj` alias for backward compat
- [x] `BuildingExtruder.generate()` wrapper implements `BuildingPlugin`
- [x] `TerrainPipeline` accepts any conforming object via `RoadPlugin | None`, `BuildingPlugin | None`

### Batch Processing
- [x] `TaskRunner` (`src/pipeline/task_runner.py`) — iterate tiles, run pipeline, `stop_on_error` flag
- [x] `TileManifest` (`src/pipeline/tile_manifest.py`) — write + validate `tile-manifest.json` via `jsonschema` against `svarog-contracts/schemas/tile-manifest.json`

### Output Formats
- [x] `GltfExporter` (`src/conversion/gltf_exporter.py`) — terrain + roads + buildings → single GLB; Z-up→Y-up transform; named nodes; `"glb"` key in pipeline result
- [x] Draco compression — `export_draco()`, `KHR_draco_mesh_compression`, ~70–90 % smaller files

---

## Phase 3 — XYZ Tile Pipeline ✅

Goal: client-compatible GLB tiles at `{z}/{x}/{y}.glb`, origin at tile centre, Y-up, north=-Z.

- [x] `svarog-contracts/schemas/world-config.json` + `svarog-contracts/world-config.json` (zoom 15, draco, upsample)
- [x] `WorldConfig` (`src/pipeline/world_config.py`) — load from contracts repo, dataclass, defaults fallback
- [x] `TileSplitter` (`src/pipeline/tile_splitter.py`) — Web Mercator XYZ↔bbox, `from_xyz()`, `xyz_grid()`, `tile_size_meters()`, `tile_center_latlon()`, `tile_center_mercator()`
- [x] `GltfExporter` vertex centering — `tile_center_local=(cx, cy)` shifts vertices to tile centre
- [x] `TaskRunner` XYZ output structure — `TileConfig.xyz=(z, x, y)` → output dir `{root}/{z}/{x}/{y}/`
- [x] `ConfigLoader` (`src/pipeline/config_loader.py`) — load YAML batch config (`bbox` or `xyz` per tile)
- [x] CLI `src/__main__.py` — `--bbox S,W,N,E`, `--tile Z/X/Y`, `--config FILE`, `--zoom`, `--draco/--no-draco`, `--upsample`, `--stop-on-error`

Total tests: **95, all passing**

---

## Phase 4 — Navigation (pending)

- [ ] NavMesh Factory — recast-navigation integration

---

## Known Issues

- Road SDF channels are correct but very narrow roads (~5 px wide in test data) produce SDF values clustering near 128 (edge). Root cause: road centre is too close to tile edge in synthetic test data to produce significant positive SDF distance.
