"""
3-D Road Mesh
=============
Clean approach: road strip follows terrain exactly, no global Z manipulation.

Z strategy (why this is correct)
----------------------------------
• Top surface Z = terrain_z sampled at the CENTRELINE + Z_TOP_OFFSET (0.20 m).
  – All roads sharing the same OSM junction node have the same (x,y) centreline
    → same terrain_z → same top height → perfect Z match at every intersection.
  – No max-pooling, no smoothing field → roads never get elevated above terrain.

• Side wall bottom Z = terrain_z sampled at the EDGE position.
  – Walls close the gap between road surface and terrain even on cross-slope roads.
  – No visible seam from any viewing angle.

• Z_TOP_OFFSET = 0.20 m prevents Z-fighting against the terrain mesh at all
  normal viewing distances (city-tile scale).

Adaptive subdivision
--------------------
Motorway/primary: 0.5 m step  – fine geometry for wide important roads
Secondary/tertiary: 1.0 m step
Residential/service/other: 2.0 m step   – saves ~60 % of vertices

Vertex layout per node i (4 vertices):
  4i+1  left_top   (lx, ly, z_centre + offset)
  4i+2  right_top  (rx, ry, z_centre + offset)
  4i+3  left_bot   (lx, ly, terrain_z_left_edge)
  4i+4  right_bot  (rx, ry, terrain_z_right_edge)

Faces per segment i→j (6 triangles):
  TOP        lt0, rt0, lt1  |  rt0, rt1, lt1
  LEFT WALL  lt0, lt1, lb0  |  lt1, lb1, lb0
  RIGHT WALL rt0, rb0, rt1  |  rb0, rb1, rt1
"""

import math
import os

import numpy as np

from .geo_utils      import to_local, sample_z
from .terrain_stamper import ROAD_HALF_WIDTHS, DEFAULT_HALF_WIDTH

Z_TOP_OFFSET = 0.20   # m above terrain (prevents Z-fighting at city scale)
MAX_MITER    = 3.0

# Adaptive subdivision step by OSM highway tag
SUBDIV_BY_HW: dict[str, float] = {
    "motorway":       0.5, "motorway_link":   0.5,
    "trunk":          0.5, "trunk_link":       0.5,
    "primary":        0.5, "primary_link":     0.5,
    "secondary":      1.0, "secondary_link":   1.0,
    "tertiary":       1.0, "tertiary_link":    1.0,
    "unclassified":   1.5,
    "residential":    2.0, "living_street":    2.0,
    "service":        2.0, "track":            2.0,
    "cycleway":       2.0, "pedestrian":       2.0,
    "footway":        3.0, "path":             3.0,
    "steps":          3.0,
}
DEFAULT_SUBDIV = 2.0


class RoadMesh:

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate_obj(
        self,
        highways:   list[dict],
        z_grid:     np.ndarray,
        meta:       dict,
        obj_name:   str | None = None,
        output_dir: str | None = None,
    ) -> str:
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        all_verts: list[tuple] = []
        all_faces: list[tuple] = []

        for road in highways:
            nodes = road.get("nodes", [])
            if len(nodes) < 2:
                continue

            tags   = road.get("tags", {})
            hw     = road.get("highway", tags.get("highway", ""))
            half_w = self._half_width(tags)
            step   = SUBDIV_BY_HW.get(hw, DEFAULT_SUBDIV)

            pts_local = [to_local(lon, lat, meta) for lon, lat in nodes]
            pts       = self._subdivide(pts_local, step)
            if len(pts) < 2:
                continue

            pts_arr = np.asarray(pts, dtype=np.float64)
            normals = self._miter_normals(pts_arr, half_w)

            left_xy  = pts_arr + normals
            right_xy = pts_arr - normals

            # Top surface: centreline terrain Z – same for every road at same OSM node
            z_top = np.array(
                [sample_z(x, y, z_grid, meta) for x, y in pts_arr],
                dtype=np.float64,
            ) + Z_TOP_OFFSET

            # Side wall bottoms: terrain at edge positions
            z_lb = np.array(
                [sample_z(x, y, z_grid, meta) for x, y in left_xy],
                dtype=np.float64,
            )
            z_rb = np.array(
                [sample_z(x, y, z_grid, meta) for x, y in right_xy],
                dtype=np.float64,
            )

            verts, faces = self._cross_section(left_xy, right_xy, z_top, z_lb, z_rb)
            base = len(all_verts)
            all_verts.extend(verts)
            all_faces.extend((f[0]+base, f[1]+base, f[2]+base) for f in faces)

        fname = f"{obj_name}_roads.obj" if obj_name else "roads.obj"
        out   = os.path.join(output_dir, fname)

        with open(out, "w") as f:
            f.write("# Road 3D OBJ – centreline Z + side walls, adaptive subdivision\n")
            f.write(
                f"# Origin: lon={meta['lon_origin']:.6f}, "
                f"lat={meta['lat_origin']:.6f}  "
                f"Z_TOP_OFFSET={Z_TOP_OFFSET} m\n"
            )
            for v in all_verts:
                f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            for fc in all_faces:
                f.write(f"f {fc[0]} {fc[1]} {fc[2]}\n")

        print(
            f"   Road OBJ: {out}  "
            f"({len(all_verts):,} vrcholů, {len(all_faces):,} faces)"
        )
        return out

    # ------------------------------------------------------------------ #
    # Cross-section geometry                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cross_section(
        left_xy:  np.ndarray,   # (n, 2)
        right_xy: np.ndarray,   # (n, 2)
        z_top:    np.ndarray,   # (n,) centreline terrain Z + offset
        z_lb:     np.ndarray,   # (n,) terrain Z at left edge
        z_rb:     np.ndarray,   # (n,) terrain Z at right edge
    ) -> tuple[list, list]:
        n     = len(left_xy)
        verts = []
        for i in range(n):
            lx, ly = left_xy[i]
            rx, ry = right_xy[i]
            zt  = float(z_top[i])
            zlb = float(z_lb[i])
            zrb = float(z_rb[i])
            verts.append((lx, ly, zt))    # left_top
            verts.append((rx, ry, zt))    # right_top
            verts.append((lx, ly, zlb))   # left_bot  ← edge terrain
            verts.append((rx, ry, zrb))   # right_bot ← edge terrain

        faces = []
        for i in range(n - 1):
            lt0 = 4*i+1; rt0 = 4*i+2; lb0 = 4*i+3; rb0 = 4*i+4
            lt1 = 4*i+5; rt1 = 4*i+6; lb1 = 4*i+7; rb1 = 4*i+8
            faces += [
                (lt0, rt0, lt1), (rt0, rt1, lt1),   # top
                (lt0, lt1, lb0), (lt1, lb1, lb0),   # left wall
                (rt0, rb0, rt1), (rb0, rb1, rt1),   # right wall
            ]

        return verts, faces

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _subdivide(pts: list[tuple], step: float) -> list[tuple]:
        result: list[tuple] = []
        for i in range(len(pts) - 1):
            x0, y0 = pts[i];  x1, y1 = pts[i + 1]
            L = math.hypot(x1 - x0, y1 - y0)
            if L < 1e-6:
                continue
            n = max(1, math.ceil(L / step))
            for j in range(n):
                t = j / n
                result.append((x0 + t*(x1-x0), y0 + t*(y1-y0)))
        if pts:
            result.append(pts[-1])
        return result

    @staticmethod
    def _miter_normals(pts: np.ndarray, half_w: float) -> np.ndarray:
        n       = len(pts)
        normals = np.zeros((n, 2))

        def perp(d: np.ndarray) -> np.ndarray:
            L = math.hypot(d[0], d[1])
            return np.array([-d[1], d[0]]) / L if L > 1e-9 else np.array([0.0, 1.0])

        for i in range(n):
            if i == 0:
                norm, scale = perp(pts[1] - pts[0]), 1.0
            elif i == n - 1:
                norm, scale = perp(pts[-1] - pts[-2]), 1.0
            else:
                ni = perp(pts[i] - pts[i-1])
                no = perp(pts[i+1] - pts[i])
                m  = ni + no
                ml = math.hypot(m[0], m[1])
                if ml < 1e-9:
                    norm, scale = ni, 1.0
                else:
                    norm  = m / ml
                    cos_a = max(abs(float(np.dot(ni, norm))), 1.0 / MAX_MITER)
                    scale = min(1.0 / cos_a, MAX_MITER)
            normals[i] = norm * half_w * scale

        return normals

    @staticmethod
    def _half_width(tags: dict) -> float:
        try:
            w = float(tags.get("width", 0) or 0)
            if w > 0:
                return w / 2.0
        except (ValueError, TypeError):
            pass
        return ROAD_HALF_WIDTHS.get(tags.get("highway", ""), DEFAULT_HALF_WIDTH)
