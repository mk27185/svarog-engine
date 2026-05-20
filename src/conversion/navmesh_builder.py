"""
Walkable navmesh triangle soup for a terrain tile.

Subtracts building footprints from the tile ground rectangle, triangulates
the walkable polygons, and returns vertices/indices for EXT_svarog_navmesh.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid

import triangle as tr


class NavmeshBuilder:
    """Build tile-local walkable mesh (Z-up, centred at origin)."""

    def build(
        self,
        buildings: list[dict],
        meta: dict,
        *,
        z_ground: float = 0.0,
    ) -> dict[str, Any] | None:
        cx = meta["total_width_m"] / 2.0
        cy = meta["total_height_m"] / 2.0

        ground = box(-cx, -cy, cx, cy)
        holes: list[Polygon] = []

        for bld in buildings:
            poly = self._footprint_polygon(bld, meta, cx, cy)
            if poly is not None and poly.is_valid and poly.area > 1.0:
                holes.append(poly)

        walkable = ground
        if holes:
            try:
                blockers = unary_union(holes)
                walkable = make_valid(ground.difference(blockers))
            except Exception:
                walkable = ground

        if walkable.is_empty:
            return None

        geoms = (
            [walkable] if walkable.geom_type == "Polygon"
            else list(walkable.geoms)  # type: ignore[attr-defined]
        )

        all_verts: list[list[float]] = []
        all_faces: list[list[int]] = []
        vert_offset = 0

        for geom in geoms:
            if geom.geom_type != "Polygon" or geom.area < 0.5:
                continue
            exterior = [(float(x), float(y)) for x, y in geom.exterior.coords[:-1]]
            if len(exterior) < 3:
                continue
            faces = self._triangulate_ring(exterior)
            if not faces:
                continue
            for f in faces:
                all_faces.append([
                    vert_offset + f[0], vert_offset + f[1], vert_offset + f[2],
                ])
            for x, y in exterior:
                all_verts.append([float(x), float(y), z_ground])
            vert_offset += len(exterior)

        if not all_verts or not all_faces:
            return None

        vertices = np.array(all_verts, dtype=np.float32)
        indices = np.array(all_faces, dtype=np.uint32)
        area_m2 = float(walkable.area) if hasattr(walkable, "area") else 0.0

        return {
            "vertices": vertices,
            "indices": indices,
            "walkable_area_m2": area_m2,
        }

    @staticmethod
    def _triangulate_ring(exterior: list[tuple[float, float]]) -> list[list[int]]:
        """CDT triangulation of a simple polygon ring (no holes)."""
        verts = np.array(exterior, dtype=np.float64)
        n = len(verts)
        if n < 3:
            return []
        segs = [[i, (i + 1) % n] for i in range(n)]
        try:
            result = tr.triangulate({"vertices": verts, "segments": segs}, "p")
            tris = result.get("triangles", [])
            return [list(map(int, t)) for t in tris]
        except Exception:
            return []

    @staticmethod
    def _footprint_polygon(
        building: dict, meta: dict, cx: float, cy: float,
    ) -> Polygon | None:
        from src.conversion.geo_utils import to_local

        nodes = building.get("nodes", [])
        if len(nodes) < 3:
            return None
        if isinstance(nodes[0], dict):
            pts = [to_local(n["lon"], n["lat"], meta) for n in nodes]
        else:
            pts = [to_local(lon, lat, meta) for lon, lat in nodes]
        centred = [(x - cx, y - cy) for x, y in pts]
        try:
            return Polygon(centred)
        except Exception:
            return None
