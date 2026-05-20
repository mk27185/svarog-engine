import json
import os
import pytest

from src.pipeline.tile_manifest import TilesetDescriptor


class TestTilesetDescriptor:

    def test_write_creates_file(self, tmp_output):
        td = TilesetDescriptor(zoom=14)
        td.add_tile((50.07, 14.43, 50.08, 14.44))
        path = td.write(os.path.join(tmp_output, "tileset.json"))
        assert os.path.exists(path)

    def test_write_returns_resolved_path(self, tmp_output):
        td = TilesetDescriptor(zoom=14)
        td.add_tile((50.07, 14.43, 50.08, 14.44))
        path = td.write(os.path.join(tmp_output, "tileset.json"))
        assert os.path.isabs(str(path))

    def test_write_before_add_tile_raises(self, tmp_output):
        td = TilesetDescriptor(zoom=14)
        with pytest.raises(ValueError, match="no tiles"):
            td.write(os.path.join(tmp_output, "empty.json"))

    def test_tilejson_version(self, tmp_output):
        td = TilesetDescriptor(zoom=14)
        td.add_tile((50.07, 14.43, 50.08, 14.44))
        path = td.write(os.path.join(tmp_output, "tileset.json"))
        with open(path) as f:
            doc = json.load(f)
        assert doc["tilejson"] == "3.0.0"

    def test_zoom_fields(self, tmp_output):
        td = TilesetDescriptor(zoom=15)
        td.add_tile((50.07, 14.43, 50.08, 14.44))
        path = td.write(os.path.join(tmp_output, "ts.json"))
        with open(path) as f:
            doc = json.load(f)
        assert doc["minzoom"] == 15
        assert doc["maxzoom"] == 15

    def test_bounds_single_tile(self, tmp_output):
        """bounds must be [lon_w, lat_s, lon_e, lat_n] (TileJSON convention)."""
        td = TilesetDescriptor(zoom=14)
        td.add_tile((50.07, 14.43, 50.08, 14.44))  # lat_s, lon_w, lat_n, lon_e
        path = td.write(os.path.join(tmp_output, "ts.json"))
        with open(path) as f:
            doc = json.load(f)
        lon_w, lat_s, lon_e, lat_n = doc["bounds"]
        assert abs(lon_w - 14.43) < 1e-5
        assert abs(lat_s - 50.07) < 1e-5
        assert abs(lon_e - 14.44) < 1e-5
        assert abs(lat_n - 50.08) < 1e-5

    def test_bounds_union_multiple_tiles(self, tmp_output):
        """Union bbox across multiple tiles."""
        td = TilesetDescriptor(zoom=14)
        td.add_tile((50.07, 14.43, 50.08, 14.44))
        td.add_tile((50.08, 14.44, 50.09, 14.45))
        path = td.write(os.path.join(tmp_output, "ts_union.json"))
        with open(path) as f:
            doc = json.load(f)
        lon_w, lat_s, lon_e, lat_n = doc["bounds"]
        assert abs(lon_w - 14.43) < 1e-5
        assert abs(lat_s - 50.07) < 1e-5
        assert abs(lon_e - 14.45) < 1e-5
        assert abs(lat_n - 50.09) < 1e-5

    def test_center_is_midpoint(self, tmp_output):
        td = TilesetDescriptor(zoom=14)
        td.add_tile((50.07, 14.43, 50.09, 14.45))
        path = td.write(os.path.join(tmp_output, "ts_center.json"))
        with open(path) as f:
            doc = json.load(f)
        lon_c, lat_c, zoom = doc["center"]
        assert abs(lon_c - 14.44) < 1e-4
        assert abs(lat_c - 50.08) < 1e-4
        assert zoom == 14

    def test_has_sdf_false_by_default(self, tmp_output):
        td = TilesetDescriptor(zoom=14)
        td.add_tile((50.07, 14.43, 50.08, 14.44))
        path = td.write(os.path.join(tmp_output, "ts.json"))
        with open(path) as f:
            doc = json.load(f)
        assert doc["extras"]["has_sdf"] is False

    def test_has_sdf_true_when_any_tile_has_sdf(self, tmp_output):
        td = TilesetDescriptor(zoom=14)
        td.add_tile((50.07, 14.43, 50.08, 14.44), has_sdf=False)
        td.add_tile((50.08, 14.43, 50.09, 14.44), has_sdf=True)
        path = td.write(os.path.join(tmp_output, "ts.json"))
        with open(path) as f:
            doc = json.load(f)
        assert doc["extras"]["has_sdf"] is True

    def test_elev_min_max_aggregated(self, tmp_output):
        td = TilesetDescriptor(zoom=14)
        td.add_tile((50.07, 14.43, 50.08, 14.44), elev_min=241.0, elev_max=270.0)
        td.add_tile((50.08, 14.43, 50.09, 14.44), elev_min=255.0, elev_max=310.0)
        path = td.write(os.path.join(tmp_output, "ts.json"))
        with open(path) as f:
            doc = json.load(f)
        assert abs(doc["extras"]["elev_min"] - 241.0) < 0.01
        assert abs(doc["extras"]["elev_max"] - 310.0) < 0.01

    def test_elev_none_when_not_provided(self, tmp_output):
        td = TilesetDescriptor(zoom=14)
        td.add_tile((50.07, 14.43, 50.08, 14.44))  # no elev
        path = td.write(os.path.join(tmp_output, "ts.json"))
        with open(path) as f:
            doc = json.load(f)
        assert doc["extras"]["elev_min"] is None
        assert doc["extras"]["elev_max"] is None

    def test_version_in_extras(self, tmp_output):
        td = TilesetDescriptor(zoom=14, version="2.3.1")
        td.add_tile((50.07, 14.43, 50.08, 14.44))
        path = td.write(os.path.join(tmp_output, "ts.json"))
        with open(path) as f:
            doc = json.load(f)
        assert doc["extras"]["svarog_version"] == "2.3.1"

    def test_creates_parent_dirs(self, tmp_output):
        td = TilesetDescriptor(zoom=14)
        td.add_tile((50.07, 14.43, 50.08, 14.44))
        nested = os.path.join(tmp_output, "deep", "nested", "tileset.json")
        td.write(nested)
        assert os.path.exists(nested)

    def test_output_is_valid_json(self, tmp_output):
        td = TilesetDescriptor(zoom=14)
        td.add_tile((50.07, 14.43, 50.08, 14.44), elev_min=240.0,
                    elev_max=280.0, has_sdf=True)
        path = td.write(os.path.join(tmp_output, "ts.json"))
        with open(path) as f:
            doc = json.load(f)
        assert isinstance(doc, dict)
        for key in ("tilejson", "tiles", "minzoom", "maxzoom", "bounds",
                    "center", "extras"):
            assert key in doc, f"Missing key: {key}"
