# Svarog-Engine Progress Tracker
**Last updated:** 2026-05-07 (12:45)

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
- [x] Output GLB/GLTF instead of OBJ — `GltfExporter` v `src/conversion/gltf_exporter.py` · kombinuje terrain+roads+buildings do jednoho GLB · Z-up→Y-up transform · pojmenované nody (Three.js ready) · `gltf_exporter` parametr v `TerrainPipeline`, result dict má klíč `"glb"`
- [x] Draco mesh compression — `GltfExporter.export_draco()` · KHR_draco_mesh_compression · `pygltflib` + `DracoPy` · ~70-90 % menší soubory · nastavitelný `quantization_bits`

### Navigation
- [ ] NavMesh Factory — recast-navigation integration

---

## Phase 3 — XYZ Tile Pipeline (klient-kompatibilní výstup) 🗺️

### Kontext z klientského kódu (`svarog` repo)

Klient načítá tiles jako **standardní XYZ slippy map**: `{z}/{x}/{y}.glb` ze `COMBINED_GLB_URL`.
- `GLB_ZOOM = 15` → tile ≈ **1222 m** na rovníku, **~720 m v Praze** (zeměpisná šířka 50°)
- Vertices v GLB jsou **relativní k centru dlaždice** (`gltfScene.position = latLonToWorld(centerLat, centerLon)`)
- Světový souřadnicový systém: **EPSG:3857 Web Mercator, Y-up, Z otočené** (sever = -Z)
- DRACOLoader je **již zapojený** v klientovi

**Problém:** Engine teď generuje GLB s originem v SW rohu a Z-up (opraveno pro Y-up), ale:
1. Origin musí být **střed dlaždice**, ne SW roh
2. Z-osa musí být **otočená** (sever = -Z v Three.js)
3. Souřadnice musí být v **EPSG:3857 metrech** (ne lokální metrické z SW rohu)
4. Výstupní cesta musí být `{z}/{x}/{y}.glb`, ne `{name}.glb`

---

### svarog-contracts: globální world config

Chceme mít `glb_zoom`, velikost dlaždice a další parametry na jednom místě — v contracts repo.
Oba projekty (engine + klient) z toho čtou.

**Nový soubor: `svarog-contracts/schemas/world-config.json` (schema)**
**Nový soubor: `svarog-contracts/world-config.json` (instance — aktuální hodnoty)**

```json
{
  "glb_zoom": 15,
  "draco": true,
  "upsample_factor": 10,
  "load_radius_tiles": 1
}
```

`glb_zoom` určuje velikost dlaždice přes standardní Web Mercator formuli:
`tile_size_m = 40_075_016 / 2^zoom * cos(lat)`
→ zoom 15 ≈ 720 m v Praze, zoom 17 ≈ 180 m, zoom 18 ≈ 90 m

Pokud chceš 100×100 m dlaždice → zoom 18 (88 m v Praze). Změna jednoho čísla v contracts, engine i klient se přizpůsobí.

---

### Plán implementace

#### 3a — World config `svarog-contracts/world-config.json`
- [x] Schema `svarog-contracts/schemas/world-config.json`
- [x] Instance `svarog-contracts/world-config.json` (výchozí: zoom 15, draco true)
- [x] `src/pipeline/world_config.py` — načte z contracts repo, vrátí dataclass `WorldConfig`, fallback na defaults

#### 3b — XYZ tile splitter `src/pipeline/tile_splitter.py`
Standard Web Mercator XYZ → bbox a zpět.

```python
# Oblast → grid XYZ tiles
tiles = TileSplitter.xyz_grid(bbox=(49.94, 14.22, 50.18, 14.71), zoom=15)
# → [TileConfig(name="15/17898/11245", bbox=(...), xyz=(15,17898,11245)), ...]

# Jeden tile → bbox
tile = TileSplitter.from_xyz(z=15, x=17898, y=11245)
```

- [x] `TileSplitter.from_xyz(z, x, y)` → `TileConfig` s bbox + atributem `xyz`
- [x] `TileSplitter.xyz_grid(bbox, zoom)` → `list[TileConfig]`
- [x] `tile_size_meters(zoom, lat)`, `tile_center_latlon`, `tile_center_mercator`

#### 3c — GLB koordinátový systém `GltfExporter` update
Aktuální GLB má origin v SW rohu a Z-up→Y-up ale bez otočení Z.
Klient očekává origin v **centru dlaždice** a **sever = -Z**.

- [x] `GltfExporter.export()` + `export_draco()` přijímají `tile_center_local=(cx, cy)` — vertices posunuty na střed dlaždice
- [x] `GltfExporter(use_draco=True/False)` — pipeline volí export metodu automaticky
- [x] Existující Z-up→Y-up transform je správný (sever → -Z ✓)

#### 3d — XYZ výstupní struktura v `TaskRunner`
Pokud `TileConfig` má atribut `xyz=(z, x, y)`, ukládej výstup jako `{output_root}/{z}/{x}/{y}.glb` místo `{output_root}/{name}.glb`.

- [x] `TaskRunner`: detekuje `xyz` na `TileConfig`, výstup jde do `{z}/{x}/{y}/` adresáře
- [x] `TileConfig` má pole `xyz`

#### 3e — Geocoding `src/extraction/geocoder.py`
```python
bbox = Geocoder.city_to_bbox("Prague")  # Nominatim → (49.94, 14.22, 50.18, 14.71)
```
- [x] `Geocoder.city_to_bbox(name)` + `city_to_point(name)` — Nominatim `/search`, no API key

#### 3f — CLI `src/__main__.py`
```bash
# Oblast zadaná bbox + automatický XYZ grid
python -m svarog_engine --bbox "49.94,14.22,50.18,14.71" --output outputs/prague/

# Oblast zadaná názvem
python -m svarog_engine --city "Prague" --output outputs/prague/

# Konkrétní XYZ tile
python -m svarog_engine --tile 15/17898/11245 --output outputs/prague/

# Config soubor
python -m svarog_engine --config tiles.yaml
```
- [x] `src/__main__.py` — argparse CLI (`--bbox`, `--city`, `--tile`, `--config`, `--zoom`, `--draco`, `--upsample`)
- [x] `ConfigLoader.load("tiles.yaml")` → `BatchConfig` s `list[TileConfig]`

---

### Doporučené pořadí implementace
**3a → 3b → 3c → 3d → 3e → 3f**
(World config nejdřív — ostatní ho čtou. Koordinátový systém je kritický blocker pro klientskou integraci.)

---

### Příklad produkčního workflow (po implementaci Phase 3)
```bash
# Centrum Prahy jako XYZ zoom-15 tiles (~720m), Draco komprese
python -m svarog_engine --city "Prague center" --output outputs/prague/
# → outputs/prague/15/17898/11245.glb, 15/17898/11246.glb, ...
# → outputs/prague/tile-manifest.json
```

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
