import os
import struct
import pytest
import numpy as np

from src.conversion.gltf_exporter import GltfExporter, _ZUPTOYUP


def _write_minimal_obj(path: str, n_verts: int = 4) -> None:
    """Write a tiny valid OBJ quad to path."""
    with open(path, "w") as f:
        f.write("v 0 0 200\nv 100 0 200\nv 100 100 201\nv 0 100 201\n")
        f.write("f 1 2 3\nf 1 3 4\n")


class TestGltfExporter:

    def test_export_terrain_only_creates_glb(self, tmp_output):
        obj = os.path.join(tmp_output, "t_terrain.obj")
        _write_minimal_obj(obj)

        exporter = GltfExporter(output_dir=tmp_output)
        result = {"terrain": obj, "roads": None, "buildings": None}
        out = exporter.export(result, output_name="t")

        assert os.path.exists(out)
        assert out.endswith(".glb")

    def test_glb_has_gltf_magic(self, tmp_output):
        """GLB files start with magic bytes 0x46546C67 ('glTF')."""
        obj = os.path.join(tmp_output, "t_terrain.obj")
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
        # Combined file should be larger than single-layer
        single_obj = os.path.join(tmp_output, "s_terrain.obj")
        _write_minimal_obj(single_obj)
        single_out = exporter.export({"terrain": single_obj, "roads": None,
                                      "buildings": None}, output_name="single")
        assert os.path.getsize(out) > os.path.getsize(single_out)

    def test_missing_layer_skipped(self, tmp_output):
        obj = os.path.join(tmp_output, "skip_terrain.obj")
        _write_minimal_obj(obj)

        exporter = GltfExporter(output_dir=tmp_output)
        # roads path is nonexistent — should not raise
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
        obj = os.path.join(tmp_output, "op_terrain.obj")
        _write_minimal_obj(obj)
        explicit = os.path.join(tmp_output, "custom_name.glb")

        exporter = GltfExporter(output_dir=tmp_output)
        out = exporter.export({"terrain": obj, "roads": None, "buildings": None},
                              output_path=explicit)
        assert out == explicit
        assert os.path.exists(explicit)

    def test_name_derived_from_terrain_path(self, tmp_output):
        obj = os.path.join(tmp_output, "myrun_terrain.obj")
        _write_minimal_obj(obj)

        exporter = GltfExporter(output_dir=tmp_output)
        out = exporter.export({"terrain": obj, "roads": None, "buildings": None})

        # Should derive name "myrun" from "myrun_terrain.obj"
        assert "myrun" in os.path.basename(out)

    def test_yup_transform_matrix(self):
        """Z-up → Y-up transform: local (0,1,0) should map to GLTF (0,0,-1)."""
        v = np.array([0.0, 1.0, 0.0, 1.0])
        transformed = _ZUPTOYUP @ v
        np.testing.assert_allclose(transformed[:3], [0.0, 0.0, -1.0])

    # ── Draco ─────────────────────────────────────────────────────────────────

    def test_export_draco_creates_glb(self, tmp_output):
        obj = os.path.join(tmp_output, "d_terrain.obj")
        _write_minimal_obj(obj)

        exporter = GltfExporter(output_dir=tmp_output)
        out = exporter.export_draco(
            {"terrain": obj, "roads": None, "buildings": None},
            output_name="draco_test",
        )
        assert os.path.exists(out)
        assert out.endswith(".glb")

    def test_export_draco_has_gltf_magic(self, tmp_output):
        obj = os.path.join(tmp_output, "dm_terrain.obj")
        _write_minimal_obj(obj)

        exporter = GltfExporter(output_dir=tmp_output)
        out = exporter.export_draco(
            {"terrain": obj, "roads": None, "buildings": None},
            output_name="draco_magic",
        )
        with open(out, "rb") as f:
            magic = f.read(4)
        assert magic == b"glTF"

    def test_draco_smaller_than_uncompressed(self, tmp_output):
        """Draco GLB should be smaller than uncompressed GLB for a real mesh."""
        # Use a bigger mesh to make the difference meaningful
        big_obj = os.path.join(tmp_output, "big_terrain.obj")
        with open(big_obj, "w") as f:
            n = 20
            for row in range(n):
                for col in range(n):
                    f.write(f"v {col*10} {row*10} {(col+row)*2}\n")
            for row in range(n - 1):
                for col in range(n - 1):
                    a = row * n + col + 1
                    b = a + 1
                    c = a + n
                    d = c + 1
                    f.write(f"f {a} {b} {c}\nf {b} {d} {c}\n")

        exporter = GltfExporter(output_dir=tmp_output)
        result = {"terrain": big_obj, "roads": None, "buildings": None}

        uncompressed = exporter.export(result, output_name="big_uncompressed")
        draco_out    = exporter.export_draco(result, output_name="big_draco")

        assert os.path.getsize(draco_out) < os.path.getsize(uncompressed), \
            "Draco GLB should be smaller than uncompressed GLB"

    def test_draco_all_layers(self, tmp_output):
        paths = {}
        for layer in ("terrain", "roads", "buildings"):
            p = os.path.join(tmp_output, f"da_{layer}.obj")
            _write_minimal_obj(p)
            paths[layer] = p

        exporter = GltfExporter(output_dir=tmp_output)
        out = exporter.export_draco({**paths, "sdf_texture": None},
                                    output_name="draco_all")
        assert os.path.exists(out)

    def test_pipeline_integration(self, mock_opentopography, mock_osm_client,
                                  tmp_output):
        from src.pipeline.terrain_pipeline import TerrainPipeline
        from src.conversion.terrain_converter import TerrainConverter
        from src.conversion.road_mesh import RoadMesh
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
        assert "terrain" in result and result["terrain"] is not None
