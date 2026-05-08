import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.conversion.terrain_converter import TerrainConverter


class TestTerrainConverter:

    def test_build_grid_returns_correct_shape(self, mock_opentopography, tmp_output):
        tif_path = mock_opentopography.get_dem(None)
        converter = TerrainConverter(output_dir=tmp_output)
        z_grid, meta = converter.build_grid(tif_path, upsample_factor=2)

        assert z_grid.ndim == 2
        # z_grid is (h+1, w+1)
        assert z_grid.shape[0] == meta["h"] + 1
        assert z_grid.shape[1] == meta["w"] + 1
        assert z_grid.dtype == np.float32

    def test_build_grid_elevations_in_range(self, mock_opentopography, tmp_output):
        tif_path = mock_opentopography.get_dem(None)
        converter = TerrainConverter(output_dir=tmp_output)
        z_grid, _ = converter.build_grid(tif_path, upsample_factor=1)

        assert z_grid.min() > 180   # synthetic grid centers around 200
        assert z_grid.max() < 215

    def test_build_grid_nonexistent_file(self, tmp_output):
        converter = TerrainConverter(output_dir=tmp_output)
        with pytest.raises(FileNotFoundError):
            converter.build_grid("/nonexistent/path/file.tif")

    def test_write_obj_creates_file(self, z_grid, meta, tmp_output):
        converter = TerrainConverter(output_dir=tmp_output)
        out_path = converter.write_obj(z_grid, meta, obj_name="test_terrain",
                                       output_dir=tmp_output)

        assert os.path.exists(out_path)
        assert out_path.endswith("_terrain.obj")

        with open(out_path) as f:
            lines = f.readlines()

        vertex_lines = [l for l in lines if l.startswith("v ")]
        face_lines   = [l for l in lines if l.startswith("f ")]
        assert len(vertex_lines) > 0
        assert len(face_lines) > 0

    def test_write_obj_vertex_count(self, z_grid, meta, tmp_output):
        converter = TerrainConverter(output_dir=tmp_output)
        out_path = converter.write_obj(z_grid, meta, obj_name="test_verts",
                                       output_dir=tmp_output)

        with open(out_path) as f:
            vertex_lines = [l for l in f if l.startswith("v ")]

        expected_vertices = (meta["h"] + 1) * (meta["w"] + 1)
        assert len(vertex_lines) == expected_vertices

    def test_write_obj_with_uv(self, z_grid, meta, tmp_output):
        converter = TerrainConverter(output_dir=tmp_output)
        out_path = converter.write_obj_with_uv(z_grid, meta, obj_name="test_uv",
                                               output_dir=tmp_output)

        with open(out_path) as f:
            content = f.read()

        assert "vt " in content, "OBJ should contain UV coordinate lines"
        # UV faces use v/uvt/vt index format: f v/vt v/vt v/vt
        face_lines = [l for l in content.splitlines() if l.startswith("f ")]
        assert len(face_lines) > 0

    def test_upsample_factor(self, mock_opentopography, tmp_output):
        tif_path = mock_opentopography.get_dem(None)
        converter = TerrainConverter(output_dir=tmp_output)

        _, meta1 = converter.build_grid(tif_path, upsample_factor=1)
        _, meta2 = converter.build_grid(tif_path, upsample_factor=5)

        assert meta2["h"] > meta1["h"]
        assert meta2["w"] > meta1["w"]
        assert meta2["cell_width_m"] < meta1["cell_width_m"]

    def test_convert_tif_to_obj_integration(self, mock_opentopography, tmp_output):
        tif_path = mock_opentopography.get_dem(None)
        converter = TerrainConverter(output_dir=tmp_output)
        out_path = converter.convert_tif_to_obj(
            tif_path, obj_name="integration_test",
            output_dir=tmp_output, upsample_factor=2
        )

        assert os.path.exists(out_path)
        assert out_path.endswith(".obj")
        with open(out_path) as f:
            content = f.read()
        assert "v " in content
        assert "f " in content
