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
The exported GLB contains up to three named nodes:
  "terrain"   – terrain quad mesh with TEXCOORD_0 UV for SDF overlay
  "roads"     – road strips (optional, dark grey)
  "buildings" – extruded building footprints (optional, light grey)

SDF texture / UV
----------------
The terrain mesh carries TEXCOORD_0 UV that exactly matches sdf_generator.py:

    u = local_x / total_width_m         west → 0, east → 1
    v = 1 − local_y / total_height_m    north → 0, south → 1

After centering at (cx, cy) = (total_width_m/2, total_height_m/2):

    u = (centred_x + cx) / (2·cx)
    v = 1 − (centred_y + cy) / (2·cy)

The cx/cy values are passed by the pipeline via tile_center_local and are
derived directly from meta["total_width_m"] / meta["total_height_m"] —
the authoritative GeoTIFF pixel dimensions.  This is the same source that
sdf_generator.py uses, so alignment is exact regardless of location.

Tile stitching
--------------
DEM vertices are sampled at cell centres, so the outermost row/column of
vertices sits ~half_cell inside the tile boundary.  _snap_boundary_vertices()
moves those extreme vertices to the exact boundary coordinates (±cx, ±cy),
closing the ~1–2 m gap that would otherwise appear between adjacent tiles.
Only X/Y are adjusted; elevation (Z) is left as-is.

GLTF extras
-----------
The scene extras carry the authoritative tile dimensions so that any future
renderer or tool can reconstruct UV or tile placement without the manifest:

    scene.extras = {"svarog": {"sdf_uv_width_m": …, "sdf_uv_height_m": …}}

Draco path
----------
export_draco() currently does not embed TEXCOORD_0 (DracoPy encodes geometry
only).  Use export() for the full-featured non-compressed path.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import trimesh.visual.material as tvm

# ── Colour palette ────────────────────────────────────────────────────────────
_COLOURS: dict[str, list[float]] = {
    "terrain":   [0.60, 0.52, 0.38, 1.0],
    "roads":     [0.25, 0.25, 0.25, 1.0],
    "buildings": [0.80, 0.78, 0.72, 1.0],
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

# Column-major form of _ZUPTOYUP for GLTF node.matrix
_ZUPTOYUP_COL: list[float] = [1, 0, 0, 0,  0, 0, -1, 0,  0, 1, 0, 0,  0, 0, 0, 1]


class GltfExporter:
    """Export pipeline OBJ results to a combined GLB."""

    def __init__(self, output_dir: str = "outputs", use_draco: bool = False):
        self.output_dir = output_dir
        self.use_draco  = use_draco

    # ── Public API ────────────────────────────────────────────────────────────

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
        Build a combined GLB from *result* (from TerrainPipeline.run_pipeline).

        result            : dict with keys "terrain", "roads", "buildings" (paths or None)
        output_name       : base name; overridden by output_path
        layer_colours     : override default RGBA colours per layer name
        tile_center_local : (cx_m, cy_m) where cx = total_width_m / 2.
                            Required for exact UV alignment and tile stitching.
                            Falls back to AABB if omitted (less precise).
        """
        colours   = {**_COLOURS, **(layer_colours or {})}
        meshes    = self._load_meshes(result, colours, tile_center_local, label="GLB")
        glb_bytes = self._build_glb(meshes, colours, tile_center_local)

        out = self._resolve_output_path(output_name, output_path, result)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            f.write(glb_bytes)

        print(f"   GLB: {out}  ({len(glb_bytes) / 1024:,.0f} KB, {len(meshes)} layer(s))")
        return out

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
        Same as export() but with KHR_draco_mesh_compression (~70–90 % smaller).
        Three.js requires DRACOLoader alongside GLTFLoader.
        Note: TEXCOORD_0 is not included in the Draco path.
        """
        import DracoPy    # noqa: PLC0415
        import pygltflib  # noqa: PLC0415

        colours = {**_COLOURS, **(layer_colours or {})}
        meshes  = self._load_meshes(result, colours, tile_center_local,
                                    label="GLB(Draco)")
        glb_bytes = self._build_draco_glb(meshes, colours, quantization_bits)

        out = self._resolve_output_path(output_name, output_path, result)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            f.write(glb_bytes)

        print(f"   GLB(Draco): {out}  "
              f"({len(glb_bytes) / 1024:,.0f} KB, {len(meshes)} layer(s))")
        return out

    # ── Mesh loading ──────────────────────────────────────────────────────────

    def _load_meshes(
        self,
        result:            dict[str, Any],
        colours:           dict[str, list[float]],
        tile_center_local: tuple[float, float] | None,
        label:             str = "GLB",
    ) -> list[tuple[str, trimesh.Trimesh]]:
        """Load terrain/roads/buildings OBJs and centre them at (cx, cy)."""
        layers = {
            "terrain":   result.get("terrain"),
            "roads":     result.get("roads"),
            "buildings": result.get("buildings"),
        }
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
            print(f"   {label}: loaded '{name}' "
                  f"({len(mesh.vertices):,} verts, {len(mesh.faces):,} faces)")
        if not meshes:
            raise ValueError("GltfExporter: no valid meshes found in pipeline result.")
        return meshes

    @staticmethod
    def _load_obj(path: str) -> trimesh.Trimesh | None:
        """Load an OBJ; always return a single Trimesh."""
        loaded = trimesh.load(path, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.geometry.values())
            if not meshes:
                return None
            loaded = trimesh.util.concatenate(meshes)
        return loaded if isinstance(loaded, trimesh.Trimesh) else None

    # ── Terrain geometry helpers ──────────────────────────────────────────────

    @staticmethod
    def _terrain_uv(
        verts: np.ndarray,
        cx:    float | None = None,
        cy:    float | None = None,
    ) -> np.ndarray:
        """
        Compute TEXCOORD_0 UV aligned with sdf_generator.py.

        Preferred (cx, cy provided):
            u = (centred_x + cx) / (2·cx)   ← derived from total_width_m
            v = 1 − (centred_y + cy) / (2·cy)

        Fallback (cx/cy unknown):
            u/v from AABB — ~half_cell error per tile edge, acceptable for
            non-tiled or single-tile use.

        Both give u=0 at the west boundary, u=1 at east; v=0 north, v=1 south.
        """
        if cx is not None and cy is not None:
            u = (verts[:, 0] + cx) / (2.0 * cx)
            v = 1.0 - (verts[:, 1] + cy) / (2.0 * cy)
        else:
            vx, vy = verts[:, 0], verts[:, 1]
            w = float(vx.max() - vx.min()) or 1.0
            h = float(vy.max() - vy.min()) or 1.0
            u = (vx - vx.min()) / w
            v = 1.0 - (vy - vy.min()) / h
        return np.column_stack([u, v]).astype(np.float32)

    @staticmethod
    def _snap_boundary_vertices(
        mesh: trimesh.Trimesh,
        cx:   float,
        cy:   float,
    ) -> None:
        """
        Snap the outermost terrain vertices to the exact tile boundary (±cx, ±cy).

        DEM grids are sampled at cell centres, so the extreme vertices sit
        ~half_cell_width inside the geographic tile boundary.  Without snapping,
        adjacent tiles leave a ~1–2 m seam.

        The snap tolerance is 2× the observed half-cell distance so that only
        the outermost row/column is affected; all interior vertices are unchanged.
        Only X and Y are adjusted; elevation (Z) is preserved.
        """
        vx = mesh.vertices[:, 0].copy()
        vy = mesh.vertices[:, 1].copy()

        # half_cell ≈ gap between extreme vertex and tile boundary
        half_cell_x = float(cx - np.abs(vx).max())
        half_cell_y = float(cy - np.abs(vy).max())

        tol_x = max(half_cell_x * 2.0, 0.1)   # safety floor: 0.1 m
        tol_y = max(half_cell_y * 2.0, 0.1)

        mesh.vertices[vx < -cx + tol_x, 0] = -cx   # west edge
        mesh.vertices[vx >  cx - tol_x, 0] =  cx   # east edge
        mesh.vertices[vy < -cy + tol_y, 1] = -cy   # south edge
        mesh.vertices[vy >  cy - tol_y, 1] =  cy   # north edge

    # ── GLB builder (non-Draco) ───────────────────────────────────────────────

    @staticmethod
    def _build_glb(
        meshes:            list[tuple[str, trimesh.Trimesh]],
        colours:           dict[str, list[float]],
        tile_center_local: tuple[float, float] | None = None,
    ) -> bytes:
        """
        Build an uncompressed GLB using pygltflib.

        Terrain gets POSITION + TEXCOORD_0 (UV baked from cx/cy).
        All other layers get POSITION only.
        GLTF scene.extras carries the authoritative tile dimensions.
        """
        import pygltflib  # noqa: PLC0415

        cx: float | None = None
        cy: float | None = None
        if tile_center_local is not None:
            cx, cy = tile_center_local

        gltf_nodes       : list[pygltflib.Node]      = []
        gltf_meshes      : list[pygltflib.Mesh]       = []
        gltf_accessors   : list[pygltflib.Accessor]   = []
        gltf_bufferviews : list[pygltflib.BufferView] = []
        gltf_materials   : list[pygltflib.Material]   = []
        binary_blobs     : list[bytes]                = []
        byte_offset      = 0

        def _append_view(data: bytes) -> int:
            nonlocal byte_offset
            idx = len(gltf_bufferviews)
            gltf_bufferviews.append(
                pygltflib.BufferView(buffer=0, byteOffset=byte_offset,
                                     byteLength=len(data))
            )
            binary_blobs.append(data)
            byte_offset += len(data)
            return idx

        def _acc_vec3(arr: np.ndarray) -> int:
            f32 = arr.astype(np.float32)
            bv  = _append_view(f32.tobytes())
            idx = len(gltf_accessors)
            gltf_accessors.append(pygltflib.Accessor(
                bufferView=bv, componentType=pygltflib.FLOAT, type=pygltflib.VEC3,
                count=len(f32), min=f32.min(axis=0).tolist(),
                max=f32.max(axis=0).tolist(),
            ))
            return idx

        def _acc_vec2(arr: np.ndarray) -> int:
            f32 = arr.astype(np.float32)
            bv  = _append_view(f32.tobytes())
            idx = len(gltf_accessors)
            gltf_accessors.append(pygltflib.Accessor(
                bufferView=bv, componentType=pygltflib.FLOAT, type=pygltflib.VEC2,
                count=len(f32),
            ))
            return idx

        def _acc_indices(faces: np.ndarray) -> int:
            flat = faces.astype(np.uint32).flatten()
            bv   = _append_view(flat.tobytes())
            idx  = len(gltf_accessors)
            gltf_accessors.append(pygltflib.Accessor(
                bufferView=bv, componentType=pygltflib.UNSIGNED_INT,
                type=pygltflib.SCALAR, count=len(flat),
            ))
            return idx

        for mesh_idx, (name, mesh) in enumerate(meshes):
            if name == "terrain" and cx is not None:
                GltfExporter._snap_boundary_vertices(mesh, cx, cy)  # type: ignore[arg-type]

            verts = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.uint32)

            pos_acc = _acc_vec3(verts)
            idx_acc = _acc_indices(faces)

            if name == "terrain":
                uv_acc = _acc_vec2(GltfExporter._terrain_uv(verts, cx, cy))
                attrs  = pygltflib.Attributes(POSITION=pos_acc, TEXCOORD_0=uv_acc)
            else:
                attrs  = pygltflib.Attributes(POSITION=pos_acc)

            colour  = colours.get(name, [0.5, 0.5, 0.5, 1.0])
            mat_idx = len(gltf_materials)
            gltf_materials.append(pygltflib.Material(
                name=name,
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorFactor=colour, metallicFactor=0.0, roughnessFactor=1.0,
                ),
                doubleSided=True,
            ))
            gltf_meshes.append(pygltflib.Mesh(
                name=name,
                primitives=[pygltflib.Primitive(
                    attributes=attrs, indices=idx_acc, material=mat_idx,
                )],
            ))
            gltf_nodes.append(pygltflib.Node(
                name=name, mesh=mesh_idx, matrix=_ZUPTOYUP_COL,
            ))

        root_idx = len(gltf_nodes)
        gltf_nodes.append(pygltflib.Node(
            name="world", children=list(range(len(meshes))),
        ))

        # Self-describing extras: tile dimensions for UV reconstruction / tooling
        extras: dict | None = None
        if cx is not None:
            extras = {"svarog": {"sdf_uv_width_m": round(2.0 * cx, 6),
                                  "sdf_uv_height_m": round(2.0 * cy, 6)}}  # type: ignore[operator]

        binary_blob = b"".join(binary_blobs)
        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(
                name="scene", nodes=[root_idx], extras=extras,
            )],
            nodes=gltf_nodes,
            meshes=gltf_meshes,
            accessors=gltf_accessors,
            bufferViews=gltf_bufferviews,
            materials=gltf_materials,
            buffers=[pygltflib.Buffer(byteLength=len(binary_blob))],
        )
        gltf.set_binary_blob(binary_blob)
        return b"".join(gltf.save_to_bytes())

    # ── GLB builder (Draco) ───────────────────────────────────────────────────

    @staticmethod
    def _build_draco_glb(
        meshes:            list[tuple[str, trimesh.Trimesh]],
        colours:           dict[str, list[float]],
        quantization_bits: int,
    ) -> bytes:
        """Build a KHR_draco_mesh_compression GLB (geometry only, no TEXCOORD_0)."""
        import DracoPy    # noqa: PLC0415
        import pygltflib  # noqa: PLC0415

        gltf_nodes       : list[pygltflib.Node]      = []
        gltf_meshes      : list[pygltflib.Mesh]       = []
        gltf_accessors   : list[pygltflib.Accessor]   = []
        gltf_bufferviews : list[pygltflib.BufferView] = []
        gltf_materials   : list[pygltflib.Material]   = []
        binary_blobs     : list[bytes]                = []
        byte_offset      = 0

        for mesh_idx, (name, mesh) in enumerate(meshes):
            verts = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.uint32)

            draco_bytes = bytes(DracoPy.encode(
                verts, faces, quantization_bits=quantization_bits,
            ))
            bv_idx = len(gltf_bufferviews)
            gltf_bufferviews.append(pygltflib.BufferView(
                buffer=0, byteOffset=byte_offset, byteLength=len(draco_bytes),
            ))
            binary_blobs.append(draco_bytes)
            byte_offset += len(draco_bytes)

            pos_acc = len(gltf_accessors)
            gltf_accessors.append(pygltflib.Accessor(
                componentType=pygltflib.FLOAT, type=pygltflib.VEC3,
                count=len(verts),
                min=verts.min(axis=0).tolist(), max=verts.max(axis=0).tolist(),
            ))
            idx_acc = len(gltf_accessors)
            gltf_accessors.append(pygltflib.Accessor(
                componentType=pygltflib.UNSIGNED_INT, type=pygltflib.SCALAR,
                count=int(len(faces) * 3),
            ))

            colour  = colours.get(name, [0.5, 0.5, 0.5, 1.0])
            mat_idx = len(gltf_materials)
            gltf_materials.append(pygltflib.Material(
                name=name,
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorFactor=colour, metallicFactor=0.0, roughnessFactor=1.0,
                ),
                doubleSided=True,
            ))
            gltf_meshes.append(pygltflib.Mesh(
                name=name,
                primitives=[pygltflib.Primitive(
                    attributes=pygltflib.Attributes(POSITION=pos_acc),
                    indices=idx_acc, material=mat_idx,
                    extensions={"KHR_draco_mesh_compression": {
                        "bufferView": bv_idx, "attributes": {"POSITION": 0},
                    }},
                )],
            ))
            gltf_nodes.append(pygltflib.Node(
                name=name, mesh=mesh_idx, matrix=_ZUPTOYUP_COL,
            ))

        root_idx = len(gltf_nodes)
        gltf_nodes.append(pygltflib.Node(
            name="root", children=list(range(len(meshes))),
        ))

        binary_blob = b"".join(binary_blobs)
        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(name="scene", nodes=[root_idx])],
            nodes=gltf_nodes, meshes=gltf_meshes, accessors=gltf_accessors,
            bufferViews=gltf_bufferviews, materials=gltf_materials,
            buffers=[pygltflib.Buffer(byteLength=len(binary_blob))],
            extensionsUsed=["KHR_draco_mesh_compression"],
            extensionsRequired=["KHR_draco_mesh_compression"],
        )
        gltf.set_binary_blob(binary_blob)
        return b"".join(gltf.save_to_bytes())

    # ── Output path resolution ────────────────────────────────────────────────

    def _resolve_output_path(
        self,
        output_name: str | None,
        output_path: str | None,
        result:      dict,
    ) -> str:
        if output_path:
            return output_path

        terrain = result.get("terrain")
        out_dir = str(Path(terrain).parent) if terrain else self.output_dir

        if output_name and "/" in output_name:
            parts = output_name.split("/")
            return os.path.join(out_dir, *parts[:-1], f"{parts[-1]}.glb")

        if not output_name:
            base = Path(terrain).stem if terrain else "scene"
            output_name = base.replace("_terrain", "") or "scene"

        return os.path.join(out_dir, f"{output_name}.glb")
