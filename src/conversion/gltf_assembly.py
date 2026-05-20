"""
Shared GLB binary assembly helpers for GltfExporter.

Handles buffer views, accessors, embedded PNG textures, and navmesh extension data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GlbAssembly:
    """Accumulates glTF resources and a single binary blob."""

    nodes: list[Any]       = field(default_factory=list)
    meshes: list[Any]      = field(default_factory=list)
    accessors: list[Any]   = field(default_factory=list)
    bufferviews: list[Any] = field(default_factory=list)
    materials: list[Any]   = field(default_factory=list)
    images: list[Any]      = field(default_factory=list)
    textures: list[Any]    = field(default_factory=list)
    samplers: list[Any]    = field(default_factory=list)
    blobs: list[bytes]     = field(default_factory=list)
    byte_offset: int       = 0
    scene_extensions: dict[str, Any] = field(default_factory=dict)
    extensions_used: list[str] = field(default_factory=list)

    def append_bytes(self, data: bytes) -> int:
        import pygltflib  # noqa: PLC0415

        idx = len(self.bufferviews)
        self.bufferviews.append(
            pygltflib.BufferView(
                buffer=0, byteOffset=self.byte_offset, byteLength=len(data),
            )
        )
        self.blobs.append(data)
        self.byte_offset += len(data)
        return idx

    def acc_vec3(self, arr: np.ndarray) -> int:
        import pygltflib  # noqa: PLC0415

        f32 = arr.astype(np.float32)
        bv  = self.append_bytes(f32.tobytes())
        idx = len(self.accessors)
        self.accessors.append(pygltflib.Accessor(
            bufferView=bv, componentType=pygltflib.FLOAT, type=pygltflib.VEC3,
            count=len(f32), min=f32.min(axis=0).tolist(),
            max=f32.max(axis=0).tolist(),
        ))
        return idx

    def acc_vec2(self, arr: np.ndarray, *, byte_stride: int = 8) -> int:
        import pygltflib  # noqa: PLC0415

        f32 = arr.astype(np.float32)
        bv  = self.append_bytes(f32.tobytes())
        self.bufferviews[-1].byteStride = byte_stride
        idx = len(self.accessors)
        self.accessors.append(pygltflib.Accessor(
            bufferView=bv, componentType=pygltflib.FLOAT, type=pygltflib.VEC2,
            count=len(f32),
        ))
        return idx

    def acc_indices(self, faces: np.ndarray) -> int:
        import pygltflib  # noqa: PLC0415

        flat = faces.astype(np.uint32).flatten()
        bv   = self.append_bytes(flat.tobytes())
        idx  = len(self.accessors)
        self.accessors.append(pygltflib.Accessor(
            bufferView=bv, componentType=pygltflib.UNSIGNED_INT,
            type=pygltflib.SCALAR, count=len(flat),
        ))
        return idx

    def embed_png(self, png_bytes: bytes) -> int:
        """Append PNG bytes; return glTF texture index."""
        import pygltflib  # noqa: PLC0415

        if not self.samplers:
            self.samplers.append(pygltflib.Sampler(
                magFilter=pygltflib.LINEAR,
                minFilter=pygltflib.LINEAR,
                wrapS=pygltflib.CLAMP_TO_EDGE,
                wrapT=pygltflib.CLAMP_TO_EDGE,
            ))

        bv_idx = self.append_bytes(png_bytes)
        img_idx = len(self.images)
        self.images.append(pygltflib.Image(
            bufferView=bv_idx, mimeType="image/png",
        ))
        tex_idx = len(self.textures)
        self.textures.append(pygltflib.Texture(
            source=img_idx, sampler=0,
        ))
        return tex_idx

    def attach_navmesh(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        *,
        walkable_area_m2: float | None = None,
    ) -> None:
        """Add navmesh triangle soup accessors and EXT_svarog_navmesh on the scene."""
        if len(vertices) == 0 or len(indices) == 0:
            return

        vert_acc = self.acc_vec3(vertices.astype(np.float32))
        idx_acc  = self.acc_indices(indices.reshape(-1, 3))

        ext: dict[str, Any] = {
            "version": 1,
            "verticesAccessor": vert_acc,
            "indicesAccessor": idx_acc,
        }
        if walkable_area_m2 is not None:
            ext["walkableAreaM2"] = round(walkable_area_m2, 2)

        self.scene_extensions["EXT_svarog_navmesh"] = ext
        if "EXT_svarog_navmesh" not in self.extensions_used:
            self.extensions_used.append("EXT_svarog_navmesh")

    def binary_blob(self) -> bytes:
        return b"".join(self.blobs)

    def finalize_glb(
        self,
        root_node_idx: int,
        *,
        extras: dict | None = None,
        draco_required: bool = False,
    ) -> bytes:
        import pygltflib  # noqa: PLC0415

        binary_blob = self.binary_blob()
        scene_kwargs: dict[str, Any] = {
            "name": "scene", "nodes": [root_node_idx],
        }
        if extras:
            scene_kwargs["extras"] = extras
        if self.scene_extensions:
            scene_kwargs["extensions"] = self.scene_extensions

        gltf_kwargs: dict[str, Any] = dict(
            scene=0,
            scenes=[pygltflib.Scene(**scene_kwargs)],
            nodes=self.nodes,
            meshes=self.meshes,
            accessors=self.accessors,
            bufferViews=self.bufferviews,
            materials=self.materials,
            buffers=[pygltflib.Buffer(byteLength=len(binary_blob))],
        )
        if self.images:
            gltf_kwargs["images"] = self.images
        if self.textures:
            gltf_kwargs["textures"] = self.textures
        if self.samplers:
            gltf_kwargs["samplers"] = self.samplers
        if self.extensions_used:
            gltf_kwargs["extensionsUsed"] = list(self.extensions_used)
        if draco_required:
            gltf_kwargs["extensionsRequired"] = ["KHR_draco_mesh_compression"]

        gltf = pygltflib.GLTF2(**gltf_kwargs)
        gltf.set_binary_blob(binary_blob)
        return b"".join(gltf.save_to_bytes())


def read_png_bytes(source: str | bytes) -> bytes:
    if isinstance(source, bytes):
        return source
    with open(source, "rb") as fh:
        return fh.read()


def embedded_textures_from_result(result: dict) -> dict[str, str | bytes]:
    """Collect texture file paths from a TerrainPipeline result dict."""
    import os

    out: dict[str, str | bytes] = {}
    for key, gltf_key in (
        ("sdf_texture", "roads_sdf"),
        ("landcover_texture", "landcover"),
    ):
        path = result.get(key)
        if path and os.path.exists(path):
            out[gltf_key] = path
    return out
