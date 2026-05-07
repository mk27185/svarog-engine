"""
GltfExporter
============
Combines terrain, roads and buildings meshes into a single GLB file
suitable for Three.js (GLTFLoader).

Coordinate system
-----------------
Our pipeline uses a Z-up local metric system (X=east, Y=north, Z=elevation).
GLTF / Three.js uses Y-up right-handed (X=right, Y=up, Z=toward viewer).

The exporter applies the standard Z-up → Y-up node transform so the scene
renders correctly without any rotation in Three.js:
    gltf_x =  local_x   (east  → right)
    gltf_y =  local_z   (elev  → up)
    gltf_z = -local_y   (north → -Z / into screen)

Named nodes
-----------
The exported GLB contains three named nodes:
  "terrain"   – terrain quad mesh (tan)
  "roads"     – road strips with side walls (dark grey)
  "buildings" – extruded building footprints (light grey)

Missing layers (roads/buildings not generated) are silently omitted.

SDF texture
-----------
The SDF PNG is NOT embedded in the GLB — it stays as a separate file.
In Three.js, load it with TextureLoader and apply it as a custom material
on the terrain mesh (it has UV coords if the pipeline used write_obj_with_uv).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import trimesh.visual.material as tvm

# ── Colour palette ────────────────────────────────────────────────────────────
# RGBA float [0-1]; used as PBR baseColorFactor in the GLB
_COLOURS: dict[str, list[float]] = {
    "terrain":   [0.60, 0.52, 0.38, 1.0],   # tan / earthy
    "roads":     [0.25, 0.25, 0.25, 1.0],   # dark asphalt grey
    "buildings": [0.80, 0.78, 0.72, 1.0],   # light stone / concrete
}

# Z-up (local metric) → Y-up (GLTF) node transform
# Rotation −90° around X: (x, y, z) → (x, z, −y)
_ZUPTOYUP: np.ndarray = np.array(
    [[1,  0,  0,  0],
     [0,  0,  1,  0],
     [0, -1,  0,  0],
     [0,  0,  0,  1]],
    dtype=np.float64,
)


class GltfExporter:
    """
    Export pipeline OBJ results to a combined GLB file.

    Parameters
    ----------
    output_dir : default output directory (overridden per call by output_path)
    """

    def __init__(self, output_dir: str = "outputs", use_draco: bool = False):
        self.output_dir = output_dir
        self.use_draco  = use_draco

    # ── Public API ────────────────────────────────────────────────────────────

    def export_draco(
        self,
        result:       dict[str, Any],
        output_name:  str | None = None,
        output_path:  str | None = None,
        *,
        layer_colours:     dict[str, list[float]] | None = None,
        quantization_bits: int = 14,
        tile_center_local: tuple[float, float] | None = None,
    ) -> str:
        """
        Same as export() but produces a Draco-compressed GLB
        (GLTF extension KHR_draco_mesh_compression).

        Draco reduces file size by ~70-90 % vs uncompressed GLB.
        Three.js requires DRACOLoader to be configured alongside GLTFLoader.

        Parameters
        ----------
        quantization_bits : Draco position quantization (11-16 recommended).
                            Higher = better precision, larger file.
        """
        import DracoPy
        import pygltflib

        colours = {**_COLOURS, **(layer_colours or {})}
        layers = {
            "terrain":   result.get("terrain"),
            "roads":     result.get("roads"),
            "buildings": result.get("buildings"),
        }

        # ── Load meshes ───────────────────────────────────────────────────────
        meshes: list[tuple[str, trimesh.Trimesh]] = []
        for name, obj_path in layers.items():
            if not obj_path or not os.path.exists(obj_path):
                continue
            mesh = self._load_obj(obj_path)
            if mesh is None or len(mesh.vertices) == 0:
                continue
            if tile_center_local is not None:
                cx, cy = tile_center_local
                mesh.vertices[:, 0] -= cx
                mesh.vertices[:, 1] -= cy
            meshes.append((name, mesh))
            print(f"   GLB(Draco): loaded '{name}' "
                  f"({len(mesh.vertices):,} verts, {len(mesh.faces):,} faces)")

        if not meshes:
            raise ValueError("GltfExporter: no valid meshes found in pipeline result.")

        glb_bytes = self._build_draco_glb(meshes, colours, quantization_bits)

        out = self._resolve_output_path(output_name, output_path, result)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            f.write(glb_bytes)

        size_kb = len(glb_bytes) / 1024
        print(f"   GLB(Draco): {out}  ({size_kb:,.0f} KB, {len(meshes)} layer(s))")
        return out

    def export(
        self,
        result:       dict[str, Any],
        output_name:  str | None = None,
        output_path:  str | None = None,
        *,
        layer_colours:     dict[str, list[float]] | None = None,
        tile_center_local: tuple[float, float] | None = None,
    ) -> str:
        """
        Build a combined GLB from the paths in *result* (from pipeline.run_pipeline).

        Parameters
        ----------
        result        : pipeline result dict with keys "terrain", "roads",
                        "buildings" (values are file paths or None)
        output_name   : base name for the output file (e.g. "prague_center")
        output_path   : explicit output path; overrides output_name + output_dir
        layer_colours : override default RGBA colours per layer name

        Returns
        -------
        str – resolved path to the written .glb file
        """
        colours = {**_COLOURS, **(layer_colours or {})}

        layers = {
            "terrain":   result.get("terrain"),
            "roads":     result.get("roads"),
            "buildings": result.get("buildings"),
        }

        scene = trimesh.Scene()
        loaded = 0

        for name, obj_path in layers.items():
            if not obj_path or not os.path.exists(obj_path):
                continue

            mesh = self._load_obj(obj_path)
            if mesh is None or len(mesh.vertices) == 0:
                continue

            if tile_center_local is not None:
                cx, cy = tile_center_local
                mesh.vertices[:, 0] -= cx
                mesh.vertices[:, 1] -= cy

            mesh.visual = trimesh.visual.TextureVisuals(
                material=tvm.PBRMaterial(
                    baseColorFactor=np.array(colours[name], dtype=np.float32),
                    doubleSided=True,
                )
            )

            scene.add_geometry(mesh, node_name=name, geom_name=name,
                               transform=_ZUPTOYUP)
            loaded += 1
            print(f"   GLB: loaded '{name}' "
                  f"({len(mesh.vertices):,} verts, {len(mesh.faces):,} faces)")

        if loaded == 0:
            raise ValueError("GltfExporter: no valid meshes found in pipeline result.")

        out = self._resolve_output_path(output_name, output_path, result)
        Path(out).parent.mkdir(parents=True, exist_ok=True)

        glb_bytes = scene.export(file_type="glb")
        with open(out, "wb") as f:
            f.write(glb_bytes)

        size_kb = len(glb_bytes) / 1024
        print(f"   GLB: {out}  ({size_kb:,.0f} KB, {loaded} layer(s))")
        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_obj(path: str) -> trimesh.Trimesh | None:
        """Load an OBJ file; always return a single Trimesh (not a Scene)."""
        loaded = trimesh.load(path, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.geometry.values())
            if not meshes:
                return None
            loaded = trimesh.util.concatenate(meshes)
        if not isinstance(loaded, trimesh.Trimesh):
            return None
        return loaded

    @staticmethod
    def _build_draco_glb(
        meshes:            list[tuple[str, "trimesh.Trimesh"]],
        colours:           dict[str, list[float]],
        quantization_bits: int,
    ) -> bytes:
        """Build a KHR_draco_mesh_compression GLB from a list of (name, Trimesh)."""
        import DracoPy
        import pygltflib

        # GLTF node transform: Z-up → Y-up (column-major)
        transform_matrix = [
            1, 0,  0, 0,
            0, 0, -1, 0,
            0, 1,  0, 0,
            0, 0,  0, 1,
        ]

        gltf_nodes      : list[pygltflib.Node]      = []
        gltf_meshes     : list[pygltflib.Mesh]       = []
        gltf_accessors  : list[pygltflib.Accessor]   = []
        gltf_bufferviews: list[pygltflib.BufferView] = []
        gltf_materials  : list[pygltflib.Material]   = []
        binary_blobs    : list[bytes]                = []
        byte_offset     = 0

        for mesh_idx, (name, mesh) in enumerate(meshes):
            verts = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.uint32)

            draco_bytes = bytes(DracoPy.encode(
                verts, faces,
                quantization_bits=quantization_bits,
            ))
            bv_idx = len(gltf_bufferviews)
            gltf_bufferviews.append(pygltflib.BufferView(
                buffer=0,
                byteOffset=byte_offset,
                byteLength=len(draco_bytes),
            ))
            binary_blobs.append(draco_bytes)
            byte_offset += len(draco_bytes)

            pos_acc_idx = len(gltf_accessors)
            gltf_accessors.append(pygltflib.Accessor(
                componentType=pygltflib.FLOAT,
                type=pygltflib.VEC3,
                count=len(verts),
                min=verts.min(axis=0).tolist(),
                max=verts.max(axis=0).tolist(),
            ))
            idx_acc_idx = len(gltf_accessors)
            gltf_accessors.append(pygltflib.Accessor(
                componentType=pygltflib.UNSIGNED_INT,
                type=pygltflib.SCALAR,
                count=int(len(faces) * 3),
            ))

            mat_idx = len(gltf_materials)
            colour = colours.get(name, [0.5, 0.5, 0.5, 1.0])
            gltf_materials.append(pygltflib.Material(
                name=name,
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorFactor=colour,
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                ),
                doubleSided=True,
            ))

            prim = pygltflib.Primitive(
                attributes=pygltflib.Attributes(POSITION=pos_acc_idx),
                indices=idx_acc_idx,
                material=mat_idx,
                extensions={
                    "KHR_draco_mesh_compression": {
                        "bufferView": bv_idx,
                        "attributes": {"POSITION": 0},
                    }
                },
            )
            gltf_meshes.append(pygltflib.Mesh(name=name, primitives=[prim]))
            gltf_nodes.append(pygltflib.Node(
                name=name, mesh=mesh_idx, matrix=transform_matrix
            ))

        binary_blob = b"".join(binary_blobs)
        root_node   = len(gltf_nodes)
        gltf_nodes.append(pygltflib.Node(
            name="root", children=list(range(len(meshes)))
        ))

        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(name="scene", nodes=[root_node])],
            nodes=gltf_nodes,
            meshes=gltf_meshes,
            accessors=gltf_accessors,
            bufferViews=gltf_bufferviews,
            materials=gltf_materials,
            buffers=[pygltflib.Buffer(byteLength=len(binary_blob))],
            extensionsUsed=["KHR_draco_mesh_compression"],
            extensionsRequired=["KHR_draco_mesh_compression"],
        )
        gltf.set_binary_blob(binary_blob)
        return b"".join(gltf.save_to_bytes())

    def _resolve_output_path(
        self,
        output_name: str | None,
        output_path: str | None,
        result:      dict,
    ) -> str:
        if output_path:
            return output_path

        # Use same directory as terrain if available
        terrain = result.get("terrain")
        out_dir = str(Path(terrain).parent) if terrain else self.output_dir

        # XYZ tile name like "15/17898/11245" → file goes in out_dir/15/17898/11245.glb
        if output_name and "/" in output_name:
            parts = output_name.split("/")
            return os.path.join(out_dir, *parts[:-1], f"{parts[-1]}.glb")

        # Derive name from terrain path if not given
        if not output_name:
            base = Path(terrain).stem if terrain else "scene"
            output_name = base.replace("_terrain", "") or "scene"

        return os.path.join(out_dir, f"{output_name}.glb")
