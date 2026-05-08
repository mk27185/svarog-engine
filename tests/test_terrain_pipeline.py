import os
from unittest.mock import MagicMock
import pytest

from src.pipeline.terrain_pipeline import TerrainPipeline
from src.conversion.terrain_converter import TerrainConverter
from src.conversion.building_extruder import BuildingExtruder


def _make_pipeline(mock_opentopography, mock_osm_client, tmp_output,
                   sdf_generator=None, building_extruder=None):
    return TerrainPipeline(
        client=mock_opentopography,
        converter=TerrainConverter(output_dir=tmp_output),
        osm_client=mock_osm_client,
        sdf_generator=sdf_generator,
        building_extruder=building_extruder or BuildingExtruder(output_dir=tmp_output),
        upsample_factor=2,
    )


class TestTerrainPipeline:

    def test_run_pipeline_produces_terrain(self, mock_opentopography,
                                          mock_osm_client, tmp_output):
        pipeline = _make_pipeline(mock_opentopography, mock_osm_client, tmp_output)
        result = pipeline.run_pipeline((50.07, 14.43, 50.08, 14.44), "pipe_test")

        assert result["terrain"] is not None
        assert os.path.exists(result["terrain"])
        assert result["terrain"].endswith("_terrain.obj")

    def test_run_pipeline_produces_buildings(self, mock_opentopography,
                                             mock_osm_client, tmp_output):
        pipeline = _make_pipeline(mock_opentopography, mock_osm_client, tmp_output)
        result = pipeline.run_pipeline((50.07, 14.43, 50.08, 14.44), "bld_pipe")

        assert result["buildings"] is not None
        assert os.path.exists(result["buildings"])
        assert result["buildings"].endswith("_buildings.obj")

    def test_run_pipeline_no_osm_client_skips_sdf_buildings(
            self, mock_opentopography, tmp_output):
        pipeline = TerrainPipeline(
            client=mock_opentopography,
            converter=TerrainConverter(output_dir=tmp_output),
            osm_client=None,
            building_extruder=None,
            upsample_factor=2,
        )
        result = pipeline.run_pipeline((50.07, 14.43, 50.08, 14.44), "no_osm")

        assert result["terrain"] is not None
        assert result["sdf_texture"] is None
        assert result["buildings"] is None

    def test_run_pipeline_dem_failure_returns_empty(self, tmp_output):
        bad_client = MagicMock()
        bad_client.get_dem.return_value = "/nonexistent/dem.tif"

        pipeline = TerrainPipeline(
            client=bad_client,
            converter=TerrainConverter(output_dir=tmp_output),
            osm_client=None,
            upsample_factor=2,
        )
        result = pipeline.run_pipeline((50.07, 14.43, 50.08, 14.44), "fail_demo")

        assert result["terrain"] is None

    def test_result_dict_has_expected_keys(self, mock_opentopography,
                                           mock_osm_client, tmp_output):
        pipeline = _make_pipeline(mock_opentopography, mock_osm_client, tmp_output)
        result = pipeline.run_pipeline((50.07, 14.43, 50.08, 14.44), "keys_test")

        assert "terrain"     in result
        assert "sdf_texture" in result
        assert "buildings"   in result
        assert "glb"         in result
        assert "roads"       not in result
