import json
import os
import pytest

from src.pipeline.tile_manifest import TileManifest, TilesetDescriptor
from src.pipeline.task_runner import TaskRunner, TileConfig
from src.pipeline.terrain_pipeline import TerrainPipeline
from src.conversion.terrain_converter import TerrainConverter
from src.conversion.road_mesh import RoadMesh
from src.conversion.building_extruder import BuildingExtruder


# ── TileManifest ─────────────────────────────────────────────────────────────

class TestTileManifest:

    def test_validate_minimal_valid(self):
        m = TileManifest(version="1.0.0", zoom=14)
        m.validate()   # should not raise

    def test_validate_bad_zoom_raises(self):
        import jsonschema
        m = TileManifest(version="1.0.0", zoom=-1)
        with pytest.raises(jsonschema.ValidationError):
            m.validate()

    def test_add_tile_completed(self):
        m = TileManifest()
        m.add_tile("t1", (50.07, 14.43, 50.08, 14.44),
                   status="completed",
                   outputs={"terrain": "t1_terrain.obj", "roads": None,
                            "buildings": None, "sdf_texture": None})
        assert len(m.tiles) == 1
        assert m.tiles[0]["name"] == "t1"
        assert m.tiles[0]["status"] == "completed"

    def test_add_tile_failed_requires_error(self):
        import jsonschema
        m = TileManifest()
        m.add_tile("t_bad", (0, 0, 1, 1), status="failed")
        with pytest.raises(jsonschema.ValidationError):
            m.validate()

    def test_add_tile_failed_with_error_passes(self):
        m = TileManifest()
        m.add_tile("t_bad", (0, 0, 1, 1), status="failed", error="DEM download failed")
        m.validate()

    def test_write_creates_file(self, tmp_output):
        m = TileManifest(version="1.2.3", zoom=12)
        m.add_tile("tile_a", (50.0, 14.0, 50.1, 14.1), status="completed",
                   outputs={"terrain": "a.obj", "roads": None,
                            "buildings": None, "sdf_texture": None})
        path = m.write(os.path.join(tmp_output, "tile-manifest.json"))
        assert os.path.exists(path)

    def test_write_content_is_valid_json(self, tmp_output):
        m = TileManifest(version="1.0.0", zoom=14)
        path = m.write(os.path.join(tmp_output, "tile-manifest.json"))
        with open(path) as f:
            data = json.load(f)
        assert data["version"] == "1.0.0"
        assert data["zoom"] == 14
        assert "tiles" in data

    def test_load_roundtrip(self, tmp_output):
        m = TileManifest(version="2.0.0", zoom=16)
        m.add_tile("roundtrip", (1.0, 2.0, 3.0, 4.0), status="pending")
        path = m.write(os.path.join(tmp_output, "rt.json"))

        loaded = TileManifest.load(path)
        assert loaded.tiles[0]["name"] == "roundtrip"
        assert loaded.tiles[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]

    def test_load_invalid_file_raises(self, tmp_output):
        import jsonschema
        bad_path = os.path.join(tmp_output, "bad.json")
        with open(bad_path, "w") as f:
            json.dump({"version": "1.0.0"}, f)   # missing zoom + layers
        with pytest.raises(jsonschema.ValidationError):
            TileManifest.load(bad_path)

    def test_schema_loaded_from_contracts(self):
        """Schema should have the 'tiles' property (extended schema from contracts)."""
        assert "tiles" in TileManifest.SCHEMA.get("properties", {})


# ── TaskRunner ────────────────────────────────────────────────────────────────

class TestTaskRunner:

    def _make_pipeline(self, tmp_output, mock_opentopography, mock_osm_client):
        return TerrainPipeline(
            client=mock_opentopography,
            converter=TerrainConverter(output_dir=tmp_output),
            road_mesh=RoadMesh(output_dir=tmp_output),
            osm_client=mock_osm_client,
            building_extruder=BuildingExtruder(output_dir=tmp_output),
            upsample_factor=2,
        )

    def test_run_single_tile_produces_manifest(
            self, mock_opentopography, mock_osm_client, tmp_output):
        pipeline = self._make_pipeline(tmp_output, mock_opentopography,
                                       mock_osm_client)
        runner = TaskRunner(pipeline=pipeline, output_root=tmp_output)
        tiles  = [TileConfig(name="t1", bbox=(50.07, 14.43, 50.08, 14.44))]
        manifest_path = os.path.join(tmp_output, "tile-manifest.json")

        summary = runner.run(tiles, manifest_path=manifest_path)

        assert os.path.exists(manifest_path)
        assert summary["total"] == 1
        assert summary["completed"] == 1
        assert summary["failed"] == 0

    def test_manifest_tile_entry(
            self, mock_opentopography, mock_osm_client, tmp_output):
        pipeline = self._make_pipeline(tmp_output, mock_opentopography,
                                       mock_osm_client)
        runner = TaskRunner(pipeline=pipeline, output_root=tmp_output)
        tiles  = [TileConfig(name="prague", bbox=(50.07, 14.43, 50.08, 14.44))]
        manifest_path = os.path.join(tmp_output, "tile-manifest.json")

        runner.run(tiles, manifest_path=manifest_path)

        manifest = TileManifest.load(manifest_path)
        assert len(manifest.tiles) == 1
        t = manifest.tiles[0]
        assert t["name"] == "prague"
        assert t["status"] == "completed"
        assert t["outputs"]["terrain"] is not None

    def test_run_multiple_tiles(
            self, mock_opentopography, mock_osm_client, tmp_output):
        pipeline = self._make_pipeline(tmp_output, mock_opentopography,
                                       mock_osm_client)
        runner = TaskRunner(pipeline=pipeline, output_root=tmp_output)
        tiles = [
            TileConfig(name="tile_a", bbox=(50.07, 14.43, 50.08, 14.44)),
            TileConfig(name="tile_b", bbox=(50.08, 14.43, 50.09, 14.44)),
        ]
        manifest_path = os.path.join(tmp_output, "tile-manifest.json")
        summary = runner.run(tiles, manifest_path=manifest_path)

        assert summary["total"] == 2
        assert summary["completed"] == 2

        manifest = TileManifest.load(manifest_path)
        names = [t["name"] for t in manifest.tiles]
        assert "tile_a" in names
        assert "tile_b" in names

    def test_failed_tile_recorded_in_manifest(
            self, mock_opentopography, tmp_output):
        from unittest.mock import MagicMock
        bad_client = MagicMock()
        bad_client.get_dem.return_value = "/nonexistent/dem.tif"

        pipeline = TerrainPipeline(
            client=bad_client,
            converter=TerrainConverter(output_dir=tmp_output),
            road_mesh=RoadMesh(output_dir=tmp_output),
            osm_client=None,
            building_extruder=None,
            upsample_factor=2,
        )

        # Patch run_pipeline to raise so we can test error handling
        original = pipeline.run_pipeline
        def raising_run(bbox, name):
            raise RuntimeError("Simulated DEM failure")
        pipeline.run_pipeline = raising_run

        runner = TaskRunner(pipeline=pipeline, output_root=tmp_output)
        tiles  = [TileConfig(name="bad_tile", bbox=(0.0, 0.0, 1.0, 1.0))]
        manifest_path = os.path.join(tmp_output, "tile-manifest.json")

        summary = runner.run(tiles, manifest_path=manifest_path)

        assert summary["failed"] == 1
        assert summary["completed"] == 0

        manifest = TileManifest.load(manifest_path)
        t = manifest.tiles[0]
        assert t["status"] == "failed"
        assert "error" in t

    def test_run_produces_tileset_json(
            self, mock_opentopography, mock_osm_client, tmp_output):
        """TaskRunner must write tileset.json alongside tile-manifest.json."""
        pipeline = self._make_pipeline(tmp_output, mock_opentopography,
                                       mock_osm_client)
        from src.conversion.gltf_exporter import GltfExporter
        pipeline.gltf_exporter = GltfExporter(output_dir=tmp_output)

        runner = TaskRunner(pipeline=pipeline, output_root=tmp_output)
        tiles = [
            TileConfig(name="t_a", bbox=(50.07, 14.43, 50.08, 14.44)),
            TileConfig(name="t_b", bbox=(50.08, 14.43, 50.09, 14.44)),
        ]
        manifest_path = os.path.join(tmp_output, "tile-manifest.json")
        summary = runner.run(tiles, manifest_path=manifest_path)

        tileset_path = os.path.join(tmp_output, "tileset.json")
        assert os.path.exists(tileset_path), "tileset.json was not created"
        assert summary["tileset"] == tileset_path

        with open(tileset_path) as f:
            ts = json.load(f)

        assert ts["tilejson"] == "3.0.0"
        assert ts["minzoom"] == ts["maxzoom"] == runner.zoom
        assert len(ts["bounds"]) == 4
        # Union bounds must span both tiles
        lon_w, lat_s, lon_e, lat_n = ts["bounds"]
        assert lat_s <= 50.07 and lat_n >= 50.09
        assert lon_w <= 14.43 and lon_e >= 14.44

    def test_tileset_extras_contain_elev_and_sdf(
            self, mock_opentopography, mock_osm_client, tmp_output):
        """tileset.json extras must aggregate elev_min/max across all tiles."""
        pipeline = self._make_pipeline(tmp_output, mock_opentopography,
                                       mock_osm_client)
        from src.conversion.gltf_exporter import GltfExporter
        pipeline.gltf_exporter = GltfExporter(output_dir=tmp_output)

        runner = TaskRunner(pipeline=pipeline, output_root=tmp_output)
        tiles = [TileConfig(name="elev_t", bbox=(50.07, 14.43, 50.08, 14.44))]
        manifest_path = os.path.join(tmp_output, "tile-manifest.json")
        runner.run(tiles, manifest_path=manifest_path)

        with open(os.path.join(tmp_output, "tileset.json")) as f:
            ts = json.load(f)

        extras = ts.get("extras", {})
        assert "has_sdf" in extras
        assert isinstance(extras.get("has_sdf"), bool)
        # elev_min/max are written when pipeline returns them
        if extras.get("elev_min") is not None:
            assert extras["elev_min"] <= extras["elev_max"]

    def test_stop_on_error_aborts_remaining(
            self, mock_opentopography, mock_osm_client, tmp_output):
        pipeline = self._make_pipeline(tmp_output, mock_opentopography,
                                       mock_osm_client)

        call_count = 0
        original = pipeline.run_pipeline
        def patched_run(bbox, name):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first tile fails")
            return original(bbox, name)

        pipeline.run_pipeline = patched_run

        runner = TaskRunner(pipeline=pipeline, output_root=tmp_output,
                            stop_on_error=True)
        tiles = [
            TileConfig(name="first",  bbox=(50.07, 14.43, 50.08, 14.44)),
            TileConfig(name="second", bbox=(50.08, 14.43, 50.09, 14.44)),
        ]
        manifest_path = os.path.join(tmp_output, "tile-manifest.json")
        summary = runner.run(tiles, manifest_path=manifest_path)

        assert call_count == 1          # second tile was never attempted
        assert summary["failed"] == 1
        assert summary["completed"] == 0
