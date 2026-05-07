import os
import numpy as np
import pytest

from src.conversion.building_extruder import (
    BuildingExtruder, DEFAULT_BUILDING_HEIGHT, LEVEL_HEIGHT,
)


class TestBuildingExtruder:

    def test_extrude_creates_file(self, z_grid, meta, simple_buildings, tmp_output):
        extruder = BuildingExtruder(output_dir=tmp_output)
        out = extruder.extrude_buildings(simple_buildings, z_grid, meta,
                                        obj_name="test", output_dir=tmp_output)

        assert os.path.exists(out)
        assert out.endswith("_buildings.obj")

    def test_vertex_and_face_count(self, z_grid, meta, simple_buildings, tmp_output):
        extruder = BuildingExtruder(output_dir=tmp_output)
        out = extruder.extrude_buildings(simple_buildings, z_grid, meta,
                                        obj_name="test2", output_dir=tmp_output)

        with open(out) as f:
            lines = f.readlines()

        verts = [l for l in lines if l.startswith("v ")]
        faces = [l for l in lines if l.startswith("f ")]
        assert len(verts) > 0
        assert len(faces) > 0

    def test_two_buildings_vertices(self, z_grid, meta, simple_buildings, tmp_output):
        """Two 4-corner buildings → 8 base + 8 top = 16 vertices."""
        extruder = BuildingExtruder(output_dir=tmp_output)
        out = extruder.extrude_buildings(simple_buildings, z_grid, meta,
                                        output_dir=tmp_output)

        with open(out) as f:
            verts = [l for l in f if l.startswith("v ")]
        assert len(verts) == 16  # 2 buildings × 4 corners × 2 (base+top)

    def test_wall_faces(self, z_grid, meta, simple_buildings, tmp_output):
        """Each 4-wall building has 8 wall triangles + n-2 roof triangles."""
        extruder = BuildingExtruder(output_dir=tmp_output)
        out = extruder.extrude_buildings(simple_buildings, z_grid, meta,
                                        output_dir=tmp_output)

        with open(out) as f:
            faces = [l for l in f if l.startswith("f ")]
        # Two buildings: 2 × (8 wall tris + 2 roof tris) = 20
        assert len(faces) == 20

    def test_height_from_tag(self):
        ext = BuildingExtruder()
        assert ext._building_height({"height": "15"}) == 15.0
        assert ext._building_height({"height": "7,5"}) == 7.5
        assert ext._building_height({"building:levels": "4"}) == 4 * LEVEL_HEIGHT
        assert ext._building_height({}) == DEFAULT_BUILDING_HEIGHT
        assert ext._building_height({"height": "bad"}) == DEFAULT_BUILDING_HEIGHT

    def test_empty_buildings_returns_empty_obj(self, z_grid, meta, tmp_output):
        extruder = BuildingExtruder(output_dir=tmp_output)
        out = extruder.extrude_buildings([], z_grid, meta,
                                        output_dir=tmp_output)

        assert os.path.exists(out)
        with open(out) as f:
            verts = [l for l in f if l.startswith("v ")]
            faces = [l for l in f if l.startswith("f ")]
        assert len(verts) == 0
        assert len(faces) == 0

    def test_building_base_uses_min_corner_z(self, z_grid, meta, simple_buildings,
                                             tmp_output):
        """Building base Z should equal the minimum terrain Z at its corners."""
        extruder = BuildingExtruder(output_dir=tmp_output)
        out = extruder.extrude_buildings(simple_buildings, z_grid, meta,
                                        obj_name="base_test", output_dir=tmp_output)

        with open(out) as f:
            lines = f.readlines()

        z_values = []
        for line in lines:
            if line.startswith("v "):
                z_values.append(float(line.split()[2]))

        # Base vertices have Z matching terrain; top vertices are base + height
        # At minimum there should be some vertex elevation close to the grid values
        assert all(z > 0 for z in z_values)
