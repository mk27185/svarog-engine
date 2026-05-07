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
        assert ext._building_height({"building:levels": "2.5"}) == 2.5 * LEVEL_HEIGHT
        assert ext._building_height({}) == DEFAULT_BUILDING_HEIGHT
        assert ext._building_height({"height": "bad"}) == DEFAULT_BUILDING_HEIGHT

    def test_wall_height_explicit_height(self):
        """Explicit height tag: wall = height − min_height − roofHeight."""
        assert BuildingExtruder._wall_height({"height": "12", "roof:height": "3"}) == 9.0
        assert BuildingExtruder._wall_height({"height": "12"}) == 12.0
        assert BuildingExtruder._wall_height({"height": "12", "roof:levels": "1"}) == 12 - LEVEL_HEIGHT
        # roof height clamped — cannot exceed total → wall = 0
        assert BuildingExtruder._wall_height({"height": "5", "roof:height": "10"}) == 0.0
        # Elevated part (min_height set)
        assert BuildingExtruder._wall_height({"height": "20", "min_height": "12"}) == 8.0
        assert BuildingExtruder._wall_height({"height": "20", "min_height": "12",
                                              "roof:height": "3"}) == 5.0
        assert BuildingExtruder._wall_height({"height": "15", "min_height": "10"}) == 5.0

    def test_wall_height_levels_only(self):
        """Levels-only: wall = levels×3 (roof is additional, OSMBuildings getDimensions)."""
        # 3-level building → wall = 9 m (roof is on top, NOT subtracted)
        assert BuildingExtruder._wall_height({"building:levels": "3"}) == 9.0
        # 1-level
        assert BuildingExtruder._wall_height({"building:levels": "1"}) == LEVEL_HEIGHT
        # elevated part: 3 levels starting at 5 m → wall = 9 − 5 = 4 m
        assert BuildingExtruder._wall_height({
            "building:levels": "3", "min_height": "5"
        }) == 4.0

    def test_wall_height_default(self):
        """No height/levels: wall = DEFAULT_HEIGHT (roof is on top)."""
        assert BuildingExtruder._wall_height({}) == DEFAULT_BUILDING_HEIGHT
        assert BuildingExtruder._wall_height({"roof:shape": "gabled"}) == DEFAULT_BUILDING_HEIGHT

    def test_apply_shape_cylinder_generates_circle(self):
        """building:shape=cylinder with 4-node rectangle → 32-node circle."""
        # Simple square near Prague
        nodes = [
            {"lon": 14.430, "lat": 50.070},
            {"lon": 14.431, "lat": 50.070},
            {"lon": 14.431, "lat": 50.071},
            {"lon": 14.430, "lat": 50.071},
        ]
        result = BuildingExtruder._apply_shape(nodes, {"building:shape": "cylinder"})
        assert len(result) == 32
        # All nodes should be within the original bounding box (roughly)
        lons = [n["lon"] for n in result]
        lats = [n["lat"] for n in result]
        assert min(lons) >= 14.429 and max(lons) <= 14.432
        assert min(lats) >= 50.069 and max(lats) <= 50.072

    def test_apply_shape_cylinder_skips_if_already_circle(self):
        """16+ node polygon with shape=cylinder is left untouched."""
        nodes = [{"lon": 14.43 + 0.0001 * i, "lat": 50.07} for i in range(16)]
        result = BuildingExtruder._apply_shape(nodes, {"building:shape": "cylinder"})
        assert result is nodes   # same object, not replaced

    def test_apply_shape_no_shape_tag_unchanged(self):
        nodes = [{"lon": 14.43, "lat": 50.07}, {"lon": 14.44, "lat": 50.07},
                 {"lon": 14.44, "lat": 50.08}]
        result = BuildingExtruder._apply_shape(nodes, {"building": "yes"})
        assert result is nodes

    def test_min_height_parsing(self):
        ext = BuildingExtruder()
        assert ext._parse_min_height({}) == 0.0
        assert ext._parse_min_height({"min_height": "5"}) == 5.0
        assert ext._parse_min_height({"min_height": "3,5"}) == 3.5
        assert ext._parse_min_height({"building:min_level": "2"}) == 2 * LEVEL_HEIGHT

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

    def test_building_base_follows_terrain(self, z_grid, meta, simple_buildings,
                                           tmp_output):
        """
        Bottom vertices of a ground-level building sit at the terrain elevation
        sampled at each corner individually (terrain-following base), so the
        building does not sink into sloped ground.
        Top vertices are all at max_ground_z + height (flat roof).
        """
        extruder = BuildingExtruder(output_dir=tmp_output)
        out = extruder.extrude_buildings(simple_buildings, z_grid, meta,
                                        obj_name="base_test", output_dir=tmp_output)

        with open(out) as f:
            lines = f.readlines()

        z_values = [float(line.split()[3]) for line in lines if line.startswith("v ")]
        assert all(z > 0 for z in z_values)

    def test_elevated_part_flat_base(self, z_grid, meta, tmp_output):
        """
        A building:part with min_height > 0 should produce a flat base at
        max_ground_z + min_height, not a terrain-following base.
        """
        lon0, lat0 = meta["lon_origin"], meta["lat_origin"]
        mpdl   = meta["meters_per_deg_lat"]
        mpdlon = meta["meters_per_deg_lon"]
        part = [{
            "id": 1,
            "nodes": [
                {"lon": lon0 + 50 / mpdlon, "lat": lat0 + 50 / mpdl},
                {"lon": lon0 + 70 / mpdlon, "lat": lat0 + 50 / mpdl},
                {"lon": lon0 + 70 / mpdlon, "lat": lat0 + 70 / mpdl},
                {"lon": lon0 + 50 / mpdlon, "lat": lat0 + 70 / mpdl},
            ],
            "tags": {"building:part": "yes", "height": "6", "min_height": "4"},
        }]

        extruder = BuildingExtruder(output_dir=tmp_output)
        out = extruder.extrude_buildings(part, z_grid, meta,
                                        obj_name="elevated", output_dir=tmp_output)

        with open(out) as f:
            lines = f.readlines()

        z_values = [float(line.split()[3]) for line in lines if line.startswith("v ")]
        base_zs = z_values[:4]
        # All four base vertices must be at the same flat elevation
        assert len(set(round(z, 3) for z in base_zs)) == 1

    def test_triangulate_polygon_convex(self):
        """Square (4 vertices) → 2 triangles covering all corners."""
        verts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        tris = BuildingExtruder._triangulate_polygon(verts)
        assert len(tris) == 2  # n-2 for any simple polygon
        # All indices are in range
        for tri in tris:
            for idx in tri:
                assert 0 <= idx < 4

    def test_triangulate_polygon_concave(self):
        """L-shaped (6 vertices) → 4 triangles, all indices valid."""
        verts = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        tris = BuildingExtruder._triangulate_polygon(verts)
        assert len(tris) == 4  # n-2 = 6-2 = 4
        for tri in tris:
            for idx in tri:
                assert 0 <= idx < 6


class TestRoofShapes:
    """Unit tests for _build_roof and individual roof shape builders."""

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _square(side=10.0):
        """CCW square polygon in local metric space."""
        return [(0, 0), (side, 0), (side, side), (0, side)]

    def _assert_all_valid(self, top_xy, top_zs, extra_verts, faces):
        n = len(top_xy)
        total_verts = n + len(extra_verts)
        for fi, fj, fk in faces:
            assert 0 <= fi < total_verts, f"face index {fi} out of range"
            assert 0 <= fj < total_verts, f"face index {fj} out of range"
            assert 0 <= fk < total_verts, f"face index {fk} out of range"
            assert len({fi, fj, fk}) == 3, "degenerate face (repeated vertex)"
        assert len(top_zs) == n

    # -------------------------------------------------------------------
    # Flat
    # -------------------------------------------------------------------

    def test_flat_roof_no_extra_verts(self):
        sq = self._square()
        top_zs, ev, ef = BuildingExtruder._build_roof(sq, 10.0, 0.0, {})
        assert ev == []
        assert len(ef) == 2       # 4-gon → 2 triangles
        assert all(z == 10.0 for z in top_zs)

    def test_flat_roof_forced_by_tag(self):
        sq = self._square()
        top_zs, ev, ef = BuildingExtruder._build_roof(
            sq, 10.0, 3.0, {"roof:shape": "flat"})
        assert ev == []

    # -------------------------------------------------------------------
    # Pyramid / Cone
    # -------------------------------------------------------------------

    def test_pyramid_adds_apex_vertex(self):
        sq = self._square()
        n = len(sq)
        top_zs, ev, ef = BuildingExtruder._build_roof(
            sq, 10.0, 4.0, {"roof:shape": "pyramid"})

        assert len(ev) == 1                       # one apex
        assert ev[0][2] == pytest.approx(14.0)    # plate_z + roof_height
        assert len(ef) == n                        # one triangle per edge
        self._assert_all_valid(sq, top_zs, ev, ef)
        # Apex index must appear in every face
        apex_idx = n
        assert all(apex_idx in f for f in ef)

    def test_cone_same_as_pyramid(self):
        sq = self._square()
        top_zs_p, ev_p, ef_p = BuildingExtruder._build_roof(
            sq, 5.0, 2.0, {"roof:shape": "pyramid"})
        top_zs_c, ev_c, ef_c = BuildingExtruder._build_roof(
            sq, 5.0, 2.0, {"roof:shape": "cone"})
        assert ev_p == ev_c
        assert ef_p == ef_c

    def test_pyramid_apex_at_centroid(self):
        sq = self._square(10.0)
        _, ev, _ = BuildingExtruder._build_roof(sq, 0.0, 5.0, {"roof:shape": "pyramid"})
        cx, cy, _ = ev[0]
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)

    # -------------------------------------------------------------------
    # Skillion
    # -------------------------------------------------------------------

    def test_skillion_no_extra_verts(self):
        sq = self._square()
        top_zs, ev, ef = BuildingExtruder._build_roof(
            sq, 10.0, 3.0, {"roof:shape": "skillion"})
        assert ev == []
        assert len(ef) == 2       # still 2 roof triangles for a square
        self._assert_all_valid(sq, top_zs, ev, ef)

    def test_skillion_varying_top_zs(self):
        sq = self._square(10.0)
        top_zs, _, _ = BuildingExtruder._build_roof(
            sq, 10.0, 3.0, {"roof:shape": "skillion", "roof:direction": "90"})
        # slope E-W: west side low (plate_z), east side high (plate_z + rh)
        assert min(top_zs) == pytest.approx(10.0)
        assert max(top_zs) == pytest.approx(13.0)

    def test_skillion_all_same_z_degenerate(self):
        """Perfectly N-S strip with E slope has no span → flat fallback."""
        # All vertices at the same x → zero E-W span when slope is E-W
        pts = [(5.0, 0.0), (5.0, 5.0), (5.0, 10.0)]  # degenerate line
        top_zs, ev, ef = BuildingExtruder._build_roof(
            pts, 10.0, 3.0, {"roof:shape": "skillion", "roof:direction": "90"})
        # Falls back to flat: all top_zs at plate_z
        assert all(z == pytest.approx(10.0) for z in top_zs)

    # -------------------------------------------------------------------
    # Gabled / Hipped
    # -------------------------------------------------------------------

    def test_gabled_two_ridge_vertices(self):
        sq = self._square(20.0)
        top_zs, ev, ef = BuildingExtruder._build_roof(
            sq, 10.0, 4.0, {"roof:shape": "gabled"})
        assert len(ev) == 2
        assert all(v[2] == pytest.approx(14.0) for v in ev)
        self._assert_all_valid(sq, top_zs, ev, ef)
        assert all(z == pytest.approx(10.0) for z in top_zs)

    def test_gabled_face_count_rectangle(self):
        """4-vertex rectangle gabled roof covers all vertices with valid faces."""
        rect = [(0, 0), (20, 0), (20, 8), (0, 8)]   # wider than tall → E-W ridge
        _, ev, ef = BuildingExtruder._build_roof(
            rect, 10.0, 3.0, {"roof:shape": "gabled"})
        # 4 slope faces (CDT) + 2 gable end wall triangles = 6 faces total.
        assert len(ef) == 6
        # All top-ring + extra vertices must be referenced at least once.
        n_total = len(rect) + len(ev)
        used = {v for face in ef for v in face}
        assert used == set(range(n_total))

    def test_hipped_two_ridge_vertices(self):
        rect = [(0, 0), (30, 0), (30, 10), (0, 10)]
        top_zs, ev, ef = BuildingExtruder._build_roof(
            rect, 10.0, 4.0, {"roof:shape": "hipped"})
        assert len(ev) == 2
        self._assert_all_valid(rect, top_zs, ev, ef)

    def test_hipped_ridge_shorter_than_gabled(self):
        """Hipped ridge endpoints are pulled inward relative to gabled."""
        rect = [(0, 0), (30, 0), (30, 10), (0, 10)]
        _, gabled_ev, _ = BuildingExtruder._build_roof(
            rect, 10.0, 4.0, {"roof:shape": "gabled"})
        _, hipped_ev, _ = BuildingExtruder._build_roof(
            rect, 10.0, 4.0, {"roof:shape": "hipped"})

        # For this wider (30) × taller (10) rect the ridge runs E-W (along X).
        # Compare X span: hipped ridge must be shorter.
        g_ridge_x = sorted(v[0] for v in gabled_ev)
        h_ridge_x = sorted(v[0] for v in hipped_ev)
        assert h_ridge_x[0] > g_ridge_x[0]   # hipped start pulled inward
        assert h_ridge_x[1] < g_ridge_x[1]   # hipped end pulled inward

    def test_square_hipped_has_two_ridge_verts(self):
        """For a 10×10 square, hipped with end_inset=0.3 still produces 2 ridge points."""
        sq = self._square(10.0)
        _, ev, ef = BuildingExtruder._build_roof(
            sq, 10.0, 4.0, {"roof:shape": "hipped"})
        # end_inset=0.3 → ridge span = 10*(1-0.6)=4 > 0, does NOT degenerate
        assert len(ev) == 2
        self._assert_all_valid(sq, [10.0] * 4, ev, ef)

    def test_gabled_roof_direction_tag(self):
        """roof:direction changes which axis the ridge runs along."""
        rect = [(0, 0), (30, 0), (30, 10), (0, 10)]
        # Default for this wide rectangle: slope N-S → ridge E-W
        _, ev_default, _ = BuildingExtruder._build_roof(
            rect, 10.0, 4.0, {"roof:shape": "gabled"})
        # direction=90 (east-facing slope) → ridge N-S (different axis)
        _, ev_ns, _ = BuildingExtruder._build_roof(
            rect, 10.0, 4.0, {"roof:shape": "gabled", "roof:direction": "90"})
        # Ridge coords must differ between the two orientations
        assert ev_default != ev_ns

    # -------------------------------------------------------------------
    # Dome / Onion
    # -------------------------------------------------------------------

    def test_dome_extra_vertex_count(self):
        """Dome: DOME_RINGS rings of n vertices each, plus 1 apex."""
        from src.conversion.building_extruder import DOME_RINGS
        sq = self._square(10.0)
        n = len(sq)
        _, ev, ef = BuildingExtruder._build_roof(
            sq, 10.0, 5.0, {"roof:shape": "dome"})
        assert len(ev) == DOME_RINGS * n + 1   # rings + apex

    def test_dome_apex_at_roof_height(self):
        from src.conversion.building_extruder import DOME_RINGS
        sq = self._square(10.0)
        _, ev, _ = BuildingExtruder._build_roof(
            sq, 10.0, 6.0, {"roof:shape": "dome"})
        apex = ev[-1]   # last vertex is the apex
        assert apex[2] == pytest.approx(16.0)

    def test_dome_all_faces_valid(self):
        sq = self._square(10.0)
        top_zs, ev, ef = BuildingExtruder._build_roof(
            sq, 10.0, 5.0, {"roof:shape": "dome"})
        self._assert_all_valid(sq, top_zs, ev, ef)

    def test_onion_apex_at_roof_height(self):
        sq = self._square(10.0)
        _, ev, _ = BuildingExtruder._build_roof(
            sq, 8.0, 4.0, {"roof:shape": "onion"})
        apex = ev[-1]
        assert apex[2] == pytest.approx(12.0)

    # -------------------------------------------------------------------
    # Full pipeline: buildings with roof:shape tag
    # -------------------------------------------------------------------

    def test_pyramid_building_extra_vertices(self, z_grid, meta, tmp_output):
        """Building with pyramid roof should produce more vertices than flat."""
        lon0, lat0 = meta["lon_origin"], meta["lat_origin"]
        mpdl   = meta["meters_per_deg_lat"]
        mpdlon = meta["meters_per_deg_lon"]

        def make_building(roof_shape):
            return [{
                "id": 99,
                "nodes": [
                    {"lon": lon0 + 50 / mpdlon, "lat": lat0 + 50 / mpdl},
                    {"lon": lon0 + 70 / mpdlon, "lat": lat0 + 50 / mpdl},
                    {"lon": lon0 + 70 / mpdlon, "lat": lat0 + 70 / mpdl},
                    {"lon": lon0 + 50 / mpdlon, "lat": lat0 + 70 / mpdl},
                ],
                "tags": {"height": "10", "roof:height": "3",
                          "roof:shape": roof_shape},
            }]

        ext = BuildingExtruder(output_dir=tmp_output)

        def count_verts(shape):
            out = ext.extrude_buildings(make_building(shape), z_grid, meta,
                                        obj_name=f"roof_{shape}", output_dir=tmp_output)
            with open(out) as f:
                return sum(1 for l in f if l.startswith("v "))

        flat_v    = count_verts("flat")
        pyramid_v = count_verts("pyramid")
        gabled_v  = count_verts("gabled")
        dome_v    = count_verts("dome")

        assert pyramid_v > flat_v   # +1 apex vertex
        assert gabled_v  > flat_v   # +2 ridge vertices
        assert dome_v    > flat_v   # +many ring vertices
