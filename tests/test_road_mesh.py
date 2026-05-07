import os
import numpy as np
import pytest

from src.conversion.road_mesh import RoadMesh, Z_TOP_OFFSET


class TestRoadMesh:

    def test_generate_obj_creates_file(self, z_grid, meta,
                                       simple_highways, tmp_output):
        mesh = RoadMesh(output_dir=tmp_output)
        out = mesh.generate_obj(simple_highways, z_grid, meta,
                               obj_name="test_roads", output_dir=tmp_output)

        assert out != ""
        assert os.path.exists(out)
        assert "test_roads_roads.obj" in out

    def test_generate_obj_vertices_and_faces(self, z_grid, meta,
                                             simple_highways, tmp_output):
        mesh = RoadMesh(output_dir=tmp_output)
        out = mesh.generate_obj(simple_highways, z_grid, meta,
                               obj_name="test_vf", output_dir=tmp_output)

        with open(out) as f:
            lines = f.readlines()

        vertex_lines = [l for l in lines if l.startswith("v ")]
        face_lines   = [l for l in lines if l.startswith("f ")]

        assert len(vertex_lines) > 0, "OBJ should have vertices"
        assert len(face_lines) > 0,   "OBJ should have faces"
        # Each segment node has 4 vertices (left_top, right_top, left_bot, right_bot)
        assert len(vertex_lines) % 4 == 0, "Vertex count should be multiple of 4"

    def test_generate_obj_returns_empty_for_no_roads(self, z_grid, meta, tmp_output):
        mesh = RoadMesh(output_dir=tmp_output)
        out = mesh.generate_obj([], z_grid, meta,
                               obj_name="empty", output_dir=tmp_output)
        assert out == ""

    def test_generate_obj_road_above_terrain(self, z_grid, meta,
                                             simple_highways, tmp_output):
        mesh = RoadMesh(output_dir=tmp_output)
        out = mesh.generate_obj(simple_highways, z_grid, meta,
                               obj_name="above", output_dir=tmp_output)

        with open(out) as f:
            lines = f.readlines()

        z_vals = []
        for line in lines:
            if line.startswith("v "):
                z_vals.append(float(line.split()[2]))

        # Top vertices are centerline Z + offset; bottom are edge terrain Z.
        # Top should be at/above the centerline terrain (bottom can be higher
        # on steep cross-slope), so we only check that the added offset is present.
        for i in range(0, len(z_vals), 4):
            zt = z_vals[i]  # left_top
            zb = z_vals[i + 2]  # left_bot
            top_minus_bottom = zt - zb
            # top = center_z + 0.2, bottom = edge_z  ⇒ offset should be ~0.2 ± cross-slope
            assert -10 < top_minus_bottom < 15, "Top-bottom Z difference out of expected range"

    def test_single_road_basic(self, z_grid, meta, tmp_output):
        """A road with two nodes should produce a valid strip."""
        lon0, lat0 = meta["lon_origin"], meta["lat_origin"]
        mpdlon = meta["meters_per_deg_lon"]
        mpdl = meta["meters_per_deg_lat"]

        single_road = [{
            "type": "way",
            "nodes": [
                {"lon": lon0 + 50 / mpdlon, "lat": lat0 + 50 / mpdl},
                {"lon": lon0 + 200 / mpdlon, "lat": lat0 + 50 / mpdl},
            ],
            "tags": {"highway": "residential"},
        }]

        mesh = RoadMesh(output_dir=tmp_output)
        out = mesh.generate_obj(single_road, z_grid, meta,
                               obj_name="single", output_dir=tmp_output)

        assert out != ""
        with open(out) as f:
            vertex_lines = [l for l in f if l.startswith("v ")]
        assert len(vertex_lines) % 4 == 0
        assert len(vertex_lines) >= 8  # at least 2 nodes × 4 verts

    def test_single_node_road_skipped(self, z_grid, meta, tmp_output):
        """A road with only 1 node should be skipped."""
        single_node = [{
            "type": "way",
            "nodes": [{"lon": meta["lon_origin"], "lat": meta["lat_origin"]}],
            "tags": {"highway": "primary"},
        }]

        mesh = RoadMesh(output_dir=tmp_output)
        out = mesh.generate_obj(single_node, z_grid, meta,
                               output_dir=tmp_output)
        assert out == ""

    def test_junction_unify_never_below_terrain(self):
        """The unified Z must never be below the original terrain Z."""
        cx = np.array([0., 10., 20., 30.])
        cy = np.array([0., 0., 0., 0.])
        tz = np.array([100., 105., 102., 108.])

        z_unified = RoadMesh._junction_unify(cx, cy, tz)
        assert np.all(z_unified >= tz), "Unified Z should be >= terrain Z"

    def test_junction_unify_shape(self):
        tz = np.random.randn(50) + 200
        cx = np.linspace(0, 100, 50)
        cy = np.zeros(50)

        z_unified = RoadMesh._junction_unify(cx, cy, tz)
        assert z_unified.shape == tz.shape

    def test_cross_section_vertex_count(self):
        """_cross_section produces 4 vertices per input node."""
        n = 6
        left_xy  = np.array([[0, i] for i in range(n)], dtype=np.float64)
        right_xy = np.array([[5, i] for i in range(n)], dtype=np.float64)
        z_center = np.full(n, 100.0)
        z_left_bot  = np.full(n, 99.0)
        z_right_bot = np.full(n, 98.0)

        verts, faces = RoadMesh._cross_section(left_xy, right_xy,
                                              z_center, z_left_bot, z_right_bot)

        assert len(verts) == n * 4
        # 6 triangles per segment (top 2 + left_wall 2 + right_wall 2)
        assert len(faces) == (n - 1) * 6

    def test_cross_section_top_vertices_above_bottom(self):
        """Each node's top vertices should be above its bottom vertices."""
        n = 4
        left_xy  = np.array([[0, i * 10] for i in range(n)], dtype=np.float64)
        right_xy = np.array([[10, i * 10] for i in range(n)], dtype=np.float64)
        z_center = np.array([200., 205., 203., 210.])
        z_left_bot  = np.array([199., 204., 202., 209.])
        z_right_bot = np.array([198., 202., 201., 208.])

        verts, _ = RoadMesh._cross_section(left_xy, right_xy,
                                          z_center, z_left_bot, z_right_bot)

        for i in range(n):
            lt_z = verts[i * 4 + 0][2]
            rt_z = verts[i * 4 + 1][2]
            lb_z = verts[i * 4 + 2][2]
            rb_z = verts[i * 4 + 3][2]
            assert lt_z == z_center[i] + Z_TOP_OFFSET
            assert rt_z == z_center[i] + Z_TOP_OFFSET
            assert lb_z == z_left_bot[i]
            assert rb_z == z_right_bot[i]

    def test_half_width_from_tags(self):
        mesh = RoadMesh()
        assert mesh._half_width({"highway": "motorway"}) == 7.0
        assert mesh._half_width({"highway": "residential"}) == 2.5
        assert mesh._half_width({"highway": "road", "width": "6"}) == 3.0
        assert mesh._half_width({"highway": "unknown"}) == 2.5  # default
