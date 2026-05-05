"""
3-D Road Mesh  –  with global junction Z reconciliation
=======================================================

Z pipeline
----------
1. Subdivide ALL roads at once (0.5 m step), collect every centerline point.
2. Sample terrain_z at every centerline point.
3. Paint those Z values onto a fine 2-D raster (0.5 m/px).
4. Apply scipy maximum_filter2d with radius = JUNCTION_R (≈ half-width × 2):
      every pixel gets the MAX Z of all road-centre points in that radius.
   → roads crossing at different heights are both raised to the higher one.
   → the overlapping "roof" becomes one unified surface.
5. Apply uniform_filter2d (gentle Gaussian-like blur) to smooth out the raised
   boundary and blend the junction plateau smoothly back into the road profile.
6. Re-sample the resulting Z field at each centerline point.
7. Safety floor: z_raised ≥ original terrain_z  (never dip below ground).
8. Per-road 1-D profile smooth (window ≈ 5 m) for fine noise reduction.
9. Add Z_TOP_OFFSET (0.20 m) → guaranteed above terrain everywhere.

Side walls
----------
Bottom of each side wall = terrain Z at the actual EDGE position (not centre).
This closes the gap between road and terrain even on cross-slope roads,
without affecting the top surface Z.

Vertex layout per node i  (4 vertices):
  4i+1  left_top   (lx, ly,  z_unified + offset)
  4i+2  right_top  (rx, ry,  z_unified + offset)
  4i+3  left_bot   (lx, ly,  terrain_z_left_edge)
  4i+4  right_bot  (rx, ry,  terrain_z_right_edge)

Faces per segment i→j  (6 triangles):
  TOP:        lt_i, rt_i, lt_j  |  rt_i, rt_j, lt_j
  LEFT WALL:  lt_i, lt_j, lb_i  |  lt_j, lb_j, lb_i
  RIGHT WALL: rt_i, rb_i, rt_j  |  rb_i, rb_j, rt_j
"""

import os
import math
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter, uniform_filter1d

from .geo_utils import to_local, sample_z
from .terrain_stamper import ROAD_HALF_WIDTHS, DEFAULT_HALF_WIDTH

Z_TOP_OFFSET = 0.20    # m – road surface above terrain
SUBDIV_STEP  = 0.5     # m – subdivision density along centerline
JUNCTION_R   = 10.0    # m – radius for Z max-pooling (covers a junction footprint)
SMOOTH_2D_R  = 6.0     # m – radius for the gentle 2-D blending pass
SMOOTH_1D    = 10      # nodes = 5 m – per-road 1-D profile smooth
MAX_MITER    = 3.0


class RoadMesh:

    def __init__(self, output_dir: str = "outputs",
                 subdivision_step: float = SUBDIV_STEP):
        self.output_dir       = output_dir
        self.subdivision_step = subdivision_step

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

        # ── Pass 1: subdivide all roads, collect centerline points ─────────
        segments: list[dict] = []   # per-road bookkeeping
        all_cx: list[float] = []
        all_cy: list[float] = []

        for road in highways:
            nodes = road.get("nodes", [])
            if len(nodes) < 2:
                continue
            tags   = road.get("tags", {})
            half_w = self._half_width(tags)

            pts_local = [to_local(lon, lat, meta) for lon, lat in nodes]
            pts       = self._subdivide(pts_local)
            if len(pts) < 2:
                continue

            pts_arr = np.array(pts, dtype=np.float64)
            normals = self._miter_normals(pts_arr, half_w)

            start = len(all_cx)
            all_cx.extend(pts_arr[:, 0].tolist())
            all_cy.extend(pts_arr[:, 1].tolist())

            segments.append({
                "pts":     pts_arr,
                "normals": normals,
                "start":   start,
                "end":     start + len(pts_arr),
            })

        if not segments:
            return ""

        cx = np.asarray(all_cx)
        cy = np.asarray(all_cy)

        # ── Pass 2: terrain Z at every centerline point ────────────────────
        print(f"   Road Z sampling ({len(cx):,} points)...")
        tz = np.array([sample_z(x, y, z_grid, meta) for x, y in zip(cx, cy)],
                      dtype=np.float64)

        # ── Pass 3: global junction Z reconciliation (2-D raster) ──────────
        z_unified = self._junction_unify(cx, cy, tz)

        # ── Pass 4: build geometry for each road ───────────────────────────
        all_verts: list[tuple] = []
        all_faces: list[tuple] = []

        for seg in segments:
            pts_arr = seg["pts"]
            normals = seg["normals"]
            s, e    = seg["start"], seg["end"]
            n_pts   = e - s

            # Per-road raised Z (with smooth profile along road direction)
            z_road = z_unified[s:e].copy()
            if n_pts >= 3:
                w      = min(SMOOTH_1D, n_pts)
                z_road = uniform_filter1d(z_road, size=w, mode="nearest")
                z_road = np.maximum(z_road, tz[s:e])   # safety floor

            left_xy  = pts_arr + normals
            right_xy = pts_arr - normals

            # Side wall bottoms: terrain at edge positions
            z_left_bot  = np.array(
                [sample_z(x, y, z_grid, meta) for x, y in left_xy],
                dtype=np.float64,
            )
            z_right_bot = np.array(
                [sample_z(x, y, z_grid, meta) for x, y in right_xy],
                dtype=np.float64,
            )

            verts, faces = self._cross_section(
                left_xy, right_xy, z_road, z_left_bot, z_right_bot
            )
            base = len(all_verts)
            all_verts.extend(verts)
            all_faces.extend((f[0]+base, f[1]+base, f[2]+base) for f in faces)

        fname = f"{obj_name}_roads.obj" if obj_name else "roads.obj"
        out   = os.path.join(output_dir, fname)

        with open(out, "w") as f:
            f.write("# Road 3D OBJ – junction-unified Z, top surface + side walls\n")
            f.write(f"# Origin: lon={meta['lon_origin']:.6f}, lat={meta['lat_origin']:.6f}\n")
            f.write(f"# Roads: {len(segments)}, Z_TOP_OFFSET={Z_TOP_OFFSET} m, "
                    f"JUNCTION_R={JUNCTION_R} m\n")
            for v in all_verts:
                f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            for fc in all_faces:
                f.write(f"f {fc[0]} {fc[1]} {fc[2]}\n")

        print(
            f"   Road 3D OBJ: {out}  "
            f"({len(all_verts):,} vrcholů, {len(all_faces):,} faces)"
        )
        return out

    # ------------------------------------------------------------------ #
    # Junction Z reconciliation                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _junction_unify(
        cx: np.ndarray,
        cy: np.ndarray,
        tz: np.ndarray,
    ) -> np.ndarray:
        """
        Paint terrain Z of all road centreline points onto a fine raster,
        apply 2-D maximum_filter (raises any point to the max Z in its
        JUNCTION_R neighbourhood), then a gentle uniform_filter to smooth
        the plateau boundary, and re-sample at each original point.

        Returns: z_unified (same shape as tz), always ≥ tz.
        """
        res   = SUBDIV_STEP                        # raster cell size = 0.5 m
        pad   = JUNCTION_R + SMOOTH_2D_R + 1.0    # border so filters don't clip

        x_min = cx.min() - pad;  x_max = cx.max() + pad
        y_min = cy.min() - pad;  y_max = cy.max() + pad

        nx = max(2, int((x_max - x_min) / res) + 1)
        ny = max(2, int((y_max - y_min) / res) + 1)

        ix = np.clip(((cx - x_min) / res).astype(np.intp), 0, nx - 1)
        iy = np.clip(((cy - y_min) / res).astype(np.intp), 0, ny - 1)

        # Paint max Z at each cell (multiple roads may share a pixel → take max)
        raster = np.zeros((ny, nx), dtype=np.float64)
        np.maximum.at(raster, (iy, ix), tz)

        # 2-D maximum filter: each cell = max Z within JUNCTION_R
        r_pix   = max(1, int(JUNCTION_R  / res))
        s_pix   = max(1, int(SMOOTH_2D_R / res))
        raised  = maximum_filter(raster, size=2 * r_pix + 1, mode="constant", cval=0.0)

        # Gentle blur to smooth the plateau boundary (blend back to normal profile)
        blended = uniform_filter(raised, size=2 * s_pix + 1, mode="constant", cval=0.0)

        # Re-sample: only use blended where it is > 0 (i.e. near a road)
        # For cells far from any road (blended≈0), fall back to tz
        z_uni = blended[iy, ix]
        z_uni = np.maximum(z_uni, tz)   # safety floor: never below terrain

        return z_uni

    # ------------------------------------------------------------------ #
    # Cross-section geometry                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cross_section(
        left_xy:     np.ndarray,   # (n, 2)
        right_xy:    np.ndarray,   # (n, 2)
        z_center:    np.ndarray,   # (n,) unified terrain Z at centreline
        z_left_bot:  np.ndarray,   # (n,) terrain Z at left edge
        z_right_bot: np.ndarray,   # (n,) terrain Z at right edge
    ) -> tuple[list, list]:
        n     = len(left_xy)
        verts = []
        for i in range(n):
            lx, ly = left_xy[i]
            rx, ry = right_xy[i]
            zt  = float(z_center[i]) + Z_TOP_OFFSET
            zlb = float(z_left_bot[i])
            zrb = float(z_right_bot[i])
            verts.append((lx, ly, zt))    # left_top
            verts.append((rx, ry, zt))    # right_top
            verts.append((lx, ly, zlb))   # left_bot  ← edge terrain
            verts.append((rx, ry, zrb))   # right_bot ← edge terrain

        faces = []
        for i in range(n - 1):
            lt0 = 4*i+1;  rt0 = 4*i+2;  lb0 = 4*i+3;  rb0 = 4*i+4
            lt1 = 4*i+5;  rt1 = 4*i+6;  lb1 = 4*i+7;  rb1 = 4*i+8
            faces.append((lt0, rt0, lt1));  faces.append((rt0, rt1, lt1))   # top
            faces.append((lt0, lt1, lb0));  faces.append((lt1, lb1, lb0))   # left wall
            faces.append((rt0, rb0, rt1));  faces.append((rb0, rb1, rt1))   # right wall

        return verts, faces

    # ------------------------------------------------------------------ #
    # Geometry helpers                                                     #
    # ------------------------------------------------------------------ #

    def _subdivide(self, local_nodes: list[tuple]) -> list[tuple]:
        step = self.subdivision_step
        pts: list[tuple] = []
        for i in range(len(local_nodes) - 1):
            x0, y0 = local_nodes[i]
            x1, y1 = local_nodes[i + 1]
            L = math.hypot(x1 - x0, y1 - y0)
            if L < 1e-6:
                continue
            n = max(1, math.ceil(L / step))
            for j in range(n):
                t = j / n
                pts.append((x0 + t*(x1-x0), y0 + t*(y1-y0)))
        if local_nodes:
            pts.append(local_nodes[-1])
        return pts

    @staticmethod
    def _miter_normals(pts: np.ndarray, half_w: float) -> np.ndarray:
        n       = len(pts)
        normals = np.zeros((n, 2))

        def perp(d):
            L = math.hypot(d[0], d[1])
            return np.array([-d[1], d[0]]) / L if L > 1e-9 else np.array([0.0, 1.0])

        for i in range(n):
            if i == 0:
                norm, scale = perp(pts[1] - pts[0]), 1.0
            elif i == n - 1:
                norm, scale = perp(pts[-1] - pts[-2]), 1.0
            else:
                n_in  = perp(pts[i] - pts[i-1])
                n_out = perp(pts[i+1] - pts[i])
                miter = n_in + n_out
                mlen  = math.hypot(miter[0], miter[1])
                if mlen < 1e-9:
                    norm, scale = n_in, 1.0
                else:
                    norm  = miter / mlen
                    cos_a = max(abs(float(np.dot(n_in, norm))), 1.0/MAX_MITER)
                    scale = min(1.0/cos_a, MAX_MITER)
            normals[i] = norm * half_w * scale

        return normals

    @staticmethod
    def _half_width(tags: dict) -> float:
        hw = tags.get("highway", "")
        try:
            w = float(tags.get("width", 0) or 0)
            if w > 0:
                return w / 2.0
        except (ValueError, TypeError):
            pass
        return ROAD_HALF_WIDTHS.get(hw, DEFAULT_HALF_WIDTH)
