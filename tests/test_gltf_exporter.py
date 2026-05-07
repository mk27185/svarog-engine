import json
import os
import struct

import numpy as np
import pytest
import trimesh

from src.conversion.gltf_exporter import GltfExporter, _ZUPTOYUP


def _write_minimal_obj(path: str) -> None:
    """Write a tiny valid OBJ quad (x: 0–100, y: 0–100, z: 200–201)."""
    with open(path, "w") as f:
        f.write("v 0 0 200\nv 100 0 200\nv 100 100 201\nv 0 100 201\n")
        f.write("f 1 2 3\nf 1 3 4\n")


def _write_grid_obj(path: str, cols: int = 5, rows: int = 5,
                    cell: float = 10.0) -> None:
    """Write an (cols×rows) terrain-like grid OBJ centred at the origin."""
    half_x = (cols - 1) * cell / 2
    half_y = (rows - 1) * cell / 2
    with open(path, "w") as f:
        for r in range(rows):
            for c in range(cols):
                x = c * cell - half_x
                y = r * cell - half_y
                f.write(f"v {x} {y} 200\n")
        for r in range(rows - 1):
            for c in range(cols - 1):
                a = r * cols + c + 1
                b = a + 1
                d = a + cols
                e = d + 1
                f.write(f"f {a} {b} {d}\nf {b} {e} {d}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Basic export / smoke tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExportSmoke:

    def test_export_terrain_only_creates_glb(self, tmp_output):
        obj = os.path.join(tmp_output, "t_terrain.obj")
        _write_minimal_obj(obj)
        exporter = GltfExporter(output_dir=tmp_output)
        out = exporter.export({"terrain": obj, "roads": None, "buildings": None},
                              output_name="t")
        assert os.path.exists(out)
        assert out.endswith(".glb")

    def test_glb_has_gltf_magic(self, tmp_output):
        obj = os.path.join(tmp_output, "magic_terrain.obj")
        _write_minimal_obj(obj)
        exporter = GltfExporter(output_dir=tmp_output)
        out = exporter.export({"terrain": obj, "roads": None, "buildings": None},
                              output_name="magic_test")
        with open(out, "rb") as f:
            magic = f.read(4)
        assert magic == b"glTF", f"Expected GLB magic, got {magic!r}"

    def test_export_all_layers(self, tmp_output):
        paths = {}
        for layer in ("terrain", "roads", "buildings"):
            p = os.path.join(tmp_output, f"all_{layer}.obj")
            _write_minimal_obj(p)
            paths[layer] = p
        exporter = GltfExporter(output_dir=tmp_output)
        out = exporter.export({**paths, "sdf_texture": None}, output_name="all")
        assert os.path.exists(out)

    def test_missing_layer_skipped(self, tmp_output):
        obj = os.path.join(tmp_output, "skip_terrain.obj")
        _write_minimal_obj(obj)
        exporter = GltfExporter(output_dir=tmp_output)
        out = exporter.export(
            {"terrain": obj, "roads": "/nonexistent/roads.obj", "buildings": None},
            output_name="skip",
        )
        assert os.path.exists(out)

    def test_no_valid_meshes_raises(self, tmp_output):
        exporter = GltfExporter(output_dir=tmp_output)
        with pytest.raises(ValueError, match="no valid meshes"):
            exporter.export({"terrain": None, "roads": None, "buildings": None},
                            output_name="empty")

    def test_output_path_override(self, tmp_output):
        obj      = os.path.join(tmp_output, "op_terrain.obj")
        explicit = os.path.join(tmp_output, "custom_name.glb")
        _write_minimal_obj(obj)
        exporter = GltfExporter(output_dir=tmp_output)
        out = exporter.export({"terrain": obj, "roads": None, "buildings": None},
                              output_path=explicit)
        assert out == explicit and os.path.exists(explicit)

    def test_name_derived_from_terrain_path(self, tmp_output):
        obj = os.path.join(tmp_output, "myrun_terrain.obj")
        _write_minimal_obj(obj)
        exporter = GltfExporter(output_dir=tmp_output)
        out = exporter.export({"terrain": obj, "roads": None, "buildings": None})
        assert "myrun" in os.path.basename(out)

    def test_yup_transform_matrix(self):
        """Z-up → Y-up: local (0,1,0) → GLTF (0,0,-1)."""
        transformed = _ZUPTOYUP @ np.array([0.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(transformed[:3], [0.0, 0.0, -1.0])


# ─────────────────────────────────────────────────────────────────────────────
# UV correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestTerrainUv:

    def test_uv_corners_with_cx_cy(self):
        """With cx/cy, NW corner → (0,0) and SE corner → (1,1)."""
        cx, cy = 50.0, 50.0
        verts = np.array([
            [-cx, -cy, 0],   # SW: u=0, v=1
            [ cx, -cy, 0],   # SE: u=1, v=1
            [ cx,  cy, 0],   # NE: u=1, v=0
            [-cx,  cy, 0],   # NW: u=0, v=0
        ], dtype=np.float32)
        uv = GltfExporter._terrain_uv(verts, cx, cy)
        np.testing.assert_allclose(uv[0], [0.0, 1.0])   # SW
        np.testing.assert_allclose(uv[1], [1.0, 1.0])   # SE
        np.testing.assert_allclose(uv[2], [1.0, 0.0])   # NE
        np.testing.assert_allclose(uv[3], [0.0, 0.0])   # NW

    def test_uv_fallback_to_aabb(self):
        """Without cx/cy the AABB fallback also maps corners to 0/1."""
        verts = np.array([
            [0.0,   0.0,  0],
            [100.0, 0.0,  0],
            [100.0, 100.0, 0],
            [0.0,   100.0, 0],
        ], dtype=np.float32)
        uv = GltfExporter._terrain_uv(verts)
        # west=0, east=1; south=1, north=0
        np.testing.assert_allclose(uv[:, 0].min(), 0.0)
        np.testing.assert_allclose(uv[:, 0].max(), 1.0)
        np.testing.assert_allclose(uv[:, 1].min(), 0.0)
        np.testing.assert_allclose(uv[:, 1].max(), 1.0)

    def test_uv_cx_cy_matches_sdf_formula(self):
        """
        UV must match sdf_generator.py: u = local_x / total_w, v = 1 - local_y / total_h.
        The centre vertex (0,0) after centering → (0.5, 0.5).
        """
        cx, cy = 109.0, 109.0
        verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        uv = GltfExporter._terrain_uv(verts, cx, cy)
        np.testing.assert_allclose(uv[0], [0.5, 0.5], atol=1e-6)

    def test_uv_range_within_0_1(self):
        """All UV values must lie in [0, 1] for any grid centred at origin."""
        cx, cy = 100.0, 80.0
        rng = np.random.default_rng(42)
        xs = rng.uniform(-cx, cx, 200)
        ys = rng.uniform(-cy, cy, 200)
        verts = np.column_stack([xs, ys, np.zeros(200)]).astype(np.float32)
        uv = GltfExporter._terrain_uv(verts, cx, cy)
        assert uv.min() >= 0.0
        assert uv.max() <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Tile stitching — snap boundary vertices
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapBoundary:

    def _make_grid_mesh(self, cx: float, cy: float,
                        cols: int = 10, rows: int = 10) -> trimesh.Trimesh:
        """
        Build a mesh that mimics DEM cell-centre sampling: extreme vertices
        are `half_cell` inside the boundary, NOT at ±cx / ±cy.
        """
        cell_x = (2 * cx) / cols
        cell_y = (2 * cy) / rows
        xs = np.linspace(-cx + cell_x / 2, cx - cell_x / 2, cols)
        ys = np.linspace(-cy + cell_y / 2, cy - cell_y / 2, rows)
        xx, yy = np.meshgrid(xs, ys)
        verts = np.column_stack([xx.ravel(), yy.ravel(),
                                 np.zeros(cols * rows)]).astype(np.float64)
        faces: list[list[int]] = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                a = r * cols + c
                faces += [[a, a + 1, a + cols], [a + 1, a + cols + 1, a + cols]]
        return trimesh.Trimesh(vertices=verts,
                               faces=np.array(faces), process=False)

    def test_snap_moves_extreme_vertices_to_boundary(self):
        cx, cy = 100.0, 80.0
        mesh   = self._make_grid_mesh(cx, cy)

        # Before snap: extreme X is NOT ±cx
        assert abs(mesh.vertices[:, 0].max() - cx) > 1.0

        GltfExporter._snap_boundary_vertices(mesh, cx, cy)

        # After snap: extreme X/Y must be exactly ±cx / ±cy
        np.testing.assert_allclose(mesh.vertices[:, 0].max(),  cx, atol=1e-6)
        np.testing.assert_allclose(mesh.vertices[:, 0].min(), -cx, atol=1e-6)
        np.testing.assert_allclose(mesh.vertices[:, 1].max(),  cy, atol=1e-6)
        np.testing.assert_allclose(mesh.vertices[:, 1].min(), -cy, atol=1e-6)

    def test_snap_does_not_move_interior_vertices(self):
        cx, cy = 100.0, 100.0
        mesh   = self._make_grid_mesh(cx, cy, cols=20, rows=20)
        before = mesh.vertices.copy()

        GltfExporter._snap_boundary_vertices(mesh, cx, cy)

        # Interior vertices (those not at the extreme row/col) must be unchanged
        tol_x = cx - abs(before[:, 0]).max()
        tol_y = cy - abs(before[:, 1]).max()
        interior = (
            (np.abs(before[:, 0]) < cx - tol_x * 2) &
            (np.abs(before[:, 1]) < cy - tol_y * 2)
        )
        np.testing.assert_array_equal(
            mesh.vertices[interior], before[interior],
            err_msg="Interior vertices must not move during boundary snap",
        )

    def test_snap_preserves_elevation(self):
        """Z coordinates must never be modified by _snap_boundary_vertices."""
        cx, cy = 50.0, 50.0
        mesh   = self._make_grid_mesh(cx, cy)
        mesh.vertices[:, 2] = np.arange(len(mesh.vertices), dtype=np.float64)
        z_before = mesh.vertices[:, 2].copy()

        GltfExporter._snap_boundary_vertices(mesh, cx, cy)

        np.testing.assert_array_equal(mesh.vertices[:, 2], z_before)


# ─────────────────────────────────────────────────────────────────────────────
# GLTF extras
# ─────────────────────────────────────────────────────────────────────────────

class TestGltfExtras:

    def _load_gltf_json(self, glb_path: str) -> dict:
        """Extract and parse the JSON chunk from a GLB file."""
        with open(glb_path, "rb") as f:
            f.read(12)                       # GLB header (magic, version, length)
            chunk_length = struct.unpack("<I", f.read(4))[0]
            f.read(4)                        # chunk type (JSON = 0x4E4F534A)
            return json.loads(f.read(chunk_length).rstrip(b"\x00"))

    def test_extras_present_when_tile_center_given(self, tmp_output):
        obj = os.path.join(tmp_output, "ex_terrain.obj")
        _write_minimal_obj(obj)
        cx, cy = 50.0, 50.0
        exporter = GltfExporter(output_dir=tmp_output)
        out = exporter.export(
            {"terrain": obj, "roads": None, "buildings": None},
            output_name="extras_test",
            tile_center_local=(cx, cy),
        )
        gltf_json = self._load_gltf_json(out)
        extras = gltf_json["scenes"][0].get("extras", {})
        svarog = extras.get("svarog", {})
        assert "sdf_uv_width_m"  in svarog
        assert "sdf_uv_height_m" in svarog
        assert abs(svarog["sdf_uv_width_m"]  - 2 * cx) < 1e-3
        assert abs(svarog["sdf_uv_height_m"] - 2 * cy) < 1e-3

    def test_extras_absent_without_tile_center(self, tmp_output):
        obj = os.path.join(tmp_output, "noex_terrain.obj")
        _write_minimal_obj(obj)
        exporter = GltfExporter(output_dir=tmp_output)
        out = exporter.export(
            {"terrain": obj, "roads": None, "buildings": None},
            output_name="no_extras",
        )
        gltf_json = self._load_gltf_json(out)
        extras = gltf_json["scenes"][0].get("extras")
        assert extras is None


# ─────────────────────────────────────────────────────────────────────────────
# TEXCOORD_0 attribute in the terrain mesh accessor
# ─────────────────────────────────────────────────────────────────────────────

class TestTexcoord0:

    def _find_terrain_primitive(self, gltf_json: dict) -> dict | None:
        for mesh in gltf_json.get("meshes", []):
            if mesh.get("name") == "terrain":
                return mesh["primitives"][0]
        return None

    def _load_gltf_json(self, glb_path: str) -> dict:
        with open(glb_path, "rb") as f:
            f.read(12)
            chunk_length = struct.unpack("<I", f.read(4))[0]
            f.read(4)
            return json.loads(f.read(chunk_length).rstrip(b"\x00"))

    def test_terrain_has_texcoord0(self, tmp_output):
        obj = os.path.join(tmp_output, "uv_terrain.obj")
        _write_minimal_obj(obj)
        exporter = GltfExporter(output_dir=tmp_output)
        out = exporter.export(
            {"terrain": obj, "roads": None, "buildings": None},
            output_name="uv_check",
        )
        gltf_json = self._load_gltf_json(out)
        prim = self._find_terrain_primitive(gltf_json)
        assert prim is not None, "No terrain mesh found in GLB"
        assert "TEXCOORD_0" in prim["attributes"], \
            "Terrain mesh must have TEXCOORD_0 attribute"

    def test_buildings_have_no_texcoord0(self, tmp_output):
        for layer in ("terrain", "buildings"):
            p = os.path.join(tmp_output, f"bld_{layer}.obj")
            _write_minimal_obj(p)
        exporter = GltfExporter(output_dir=tmp_output)
        out = exporter.export(
            {"terrain": os.path.join(tmp_output, "bld_terrain.obj"),
             "roads":    None,
             "buildings": os.path.join(tmp_output, "bld_buildings.obj")},
            output_name="bld_check",
        )
        gltf_json = self._load_gltf_json(out)
        for mesh in gltf_json.get("meshes", []):
            if mesh.get("name") == "buildings":
                assert "TEXCOORD_0" not in mesh["primitives"][0]["attributes"], \
                    "Buildings mesh must NOT have TEXCOORD_0"


# ─────────────────────────────────────────────────────────────────────────────
# Draco
# ─────────────────────────────────────────────────────────────────────────────

class TestDraco:

    def test_export_draco_creates_glb(self, tmp_output):
        obj = os.path.join(tmp_output, "d_terrain.obj")
        _write_minimal_obj(obj)
        out = GltfExporter(output_dir=tmp_output).export_draco(
            {"terrain": obj, "roads": None, "buildings": None},
            output_name="draco_test",
        )
        assert os.path.exists(out) and out.endswith(".glb")

    def test_export_draco_has_gltf_magic(self, tmp_output):
        obj = os.path.join(tmp_output, "dm_terrain.obj")
        _write_minimal_obj(obj)
        out = GltfExporter(output_dir=tmp_output).export_draco(
            {"terrain": obj, "roads": None, "buildings": None},
            output_name="draco_magic",
        )
        with open(out, "rb") as f:
            assert f.read(4) == b"glTF"

    def test_draco_smaller_than_uncompressed(self, tmp_output):
        big = os.path.join(tmp_output, "big_terrain.obj")
        with open(big, "w") as f:
            n = 20
            for r in range(n):
                for c in range(n):
                    f.write(f"v {c * 10} {r * 10} {(c + r) * 2}\n")
            for r in range(n - 1):
                for c in range(n - 1):
                    a = r * n + c + 1
                    f.write(f"f {a} {a+1} {a+n}\nf {a+1} {a+n+1} {a+n}\n")

        exporter = GltfExporter(output_dir=tmp_output)
        result   = {"terrain": big, "roads": None, "buildings": None}
        uncomp   = exporter.export(result, output_name="big_uncompressed")
        draco    = exporter.export_draco(result, output_name="big_draco")
        assert os.path.getsize(draco) < os.path.getsize(uncomp)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline integration
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:

    def test_pipeline_integration(self, mock_opentopography, mock_osm_client,
                                  tmp_output):
        from src.pipeline.terrain_pipeline  import TerrainPipeline
        from src.conversion.terrain_converter import TerrainConverter
        from src.conversion.road_mesh         import RoadMesh
        from src.conversion.building_extruder import BuildingExtruder

        exporter = GltfExporter(output_dir=tmp_output)
        pipeline = TerrainPipeline(
            client=mock_opentopography,
            converter=TerrainConverter(output_dir=tmp_output),
            road_mesh=RoadMesh(output_dir=tmp_output),
            osm_client=mock_osm_client,
            building_extruder=BuildingExtruder(output_dir=tmp_output),
            gltf_exporter=exporter,
            upsample_factor=2,
        )
        result = pipeline.run_pipeline((50.07, 14.43, 50.08, 14.44), "glb_test")

        assert result["glb"] is not None
        assert os.path.exists(result["glb"])
        assert result["glb"].endswith(".glb")
        assert result["terrain"] is not None
