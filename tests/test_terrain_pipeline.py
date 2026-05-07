import os
from unittest.mock import MagicMock, patch
import pytest

from src.pipeline.terrain_pipeline import TerrainPipeline
from src.conversion.terrain_converter import TerrainConverter
from src.conversion.road_mesh import RoadMesh
from src.conversion.building_extruder import BuildingExtruder


class TestTerrainPipeline:

    def test_run_pipeline_produces_terrain(self, mock_opentopography,
                                          mock_osm_client, tmp_output):
        converter = TerrainConverter(output_dir=tmp_output)
        road_mesh = RoadMesh(output_dir=tmp_output)
        building_extruder = BuildingExtruder(output_dir=tmp_output)

        pipeline = TerrainPipeline(
            client=mock_opentopography,
            converter=converter,
            road_mesh=road_mesh,
            osm_client=mock_osm_client,
            building_extruder=building_extruder,
            upsample_factor=2,
        )

        bbox = (50.07, 14.43, 50.08, 14.44)
        result = pipeline.run_pipeline(bbox, "pipe_test")

        assert result["terrain"] is not None
        assert os.path.exists(result["terrain"])
        assert result["terrain"].endswith("_terrain.obj")

    def test_run_pipeline_produces_roads(self, mock_opentopography,
                                        mock_osm_client, tmp_output):
        converter = TerrainConverter(output_dir=tmp_output)
        road_mesh = RoadMesh(output_dir=tmp_output)
        building_extruder = BuildingExtruder(output_dir=tmp_output)

        pipeline = TerrainPipeline(
            client=mock_opentopography,
            converter=converter,
            road_mesh=road_mesh,
            osm_client=mock_osm_client,
            building_extruder=building_extruder,
            upsample_factor=2,
        )

        bbox = (50.07, 14.43, 50.08, 14.44)
        result = pipeline.run_pipeline(bbox, "road_pipe")

        assert result["roads"] is not None
        assert os.path.exists(result["roads"])
        assert result["roads"].endswith("_roads.obj")

    def test_run_pipeline_produces_buildings(self, mock_opentopography,
                                             mock_osm_client, tmp_output):
        converter = TerrainConverter(output_dir=tmp_output)
        road_mesh = RoadMesh(output_dir=tmp_output)
        building_extruder = BuildingExtruder(output_dir=tmp_output)

        pipeline = TerrainPipeline(
            client=mock_opentopography,
            converter=converter,
            road_mesh=road_mesh,
            osm_client=mock_osm_client,
            building_extruder=building_extruder,
            upsample_factor=2,
        )

        bbox = (50.07, 14.43, 50.08, 14.44)
        result = pipeline.run_pipeline(bbox, "bld_pipe")

        assert result["buildings"] is not None
        assert os.path.exists(result["buildings"])
        assert result["buildings"].endswith("_buildings.obj")

    def test_run_pipeline_no_osm_client_skips_roads_buildings(
            self, mock_opentopography, tmp_output):
        converter = TerrainConverter(output_dir=tmp_output)
        road_mesh = RoadMesh(output_dir=tmp_output)

        pipeline = TerrainPipeline(
            client=mock_opentopography,
            converter=converter,
            road_mesh=road_mesh,
            osm_client=None,   # no OSM
            building_extruder=None,
            upsample_factor=2,
        )

        bbox = (14.43, 50.07, 14.44, 50.08)
        result = pipeline.run_pipeline(bbox, "no_osm")

        assert result["terrain"] is not None
        assert result["roads"] is None
        assert result["buildings"] is None

    def test_run_pipeline_dem_failure_returns_empty(self, tmp_output):
        converter = TerrainConverter(output_dir=tmp_output)
        road_mesh = RoadMesh(output_dir=tmp_output)
        bad_client = MagicMock()
        bad_client.get_dem.return_value = "/nonexistent/dem.tif"

        pipeline = TerrainPipeline(
            client=bad_client,
            converter=converter,
            road_mesh=road_mesh,
            osm_client=None,
            building_extruder=None,
            upsample_factor=2,
        )

        bbox = (14.43, 50.07, 14.44, 50.08)
        result = pipeline.run_pipeline(bbox, "fail_demo")

        assert result["terrain"] is None

    def test_result_dict_has_expected_keys(self, mock_opentopography,
                                           mock_osm_client, tmp_output):
        converter = TerrainConverter(output_dir=tmp_output)
        road_mesh = RoadMesh(output_dir=tmp_output)
        building_extruder = BuildingExtruder(output_dir=tmp_output)

        pipeline = TerrainPipeline(
            client=mock_opentopography,
            converter=converter,
            road_mesh=road_mesh,
            osm_client=mock_osm_client,
            building_extruder=building_extruder,
            upsample_factor=2,
        )

        bbox = (50.07, 14.43, 50.08, 14.44)
        result = pipeline.run_pipeline(bbox, "keys_test")

        assert "terrain" in result
        assert "roads" in result
        assert "buildings" in result
        assert "sdf_texture" in result
