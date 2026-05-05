"""
Road Stamper
============
Imprints road geometry directly into the terrain z_grid — no separate road mesh,
no Z-fighting, no floating geometry, no visible gaps.

Algorithm
---------
1. Subdivide ALL road centerlines (0.5 m step), collect (x, y) points.
2. Sample terrain_z at every centerline point.
3. Junction Z unification via 2-D raster max-pooling (same as road_mesh):
      every centerline point is raised to the local max Z within JUNCTION_R.
      → roads crossing at different heights share one unified "roof" Z.
4. Build scipy cKDTree over all centerline points.
5. For every terrain vertex query the nearest road centerline point:
      dist ≤ half_width            → full stamp  (z = road_z)
      half_width < dist ≤ +BLEND_M → smooth S-curve blend (road_z → terrain_z)
      dist > half_width + BLEND_M  → unchanged terrain_z
6. Return modified z_grid (float32, same shape as input).

Result: seamless terrain mesh with roads naturally carved in.
Road visual distinction requires UV / material masks, not separate geometry.
"""

import math
import numpy as np
from scipy.ndimage  import maximum_filter, uniform_filter
from scipy.spatial  import cKDTree

from .geo_utils      import to_local, sample_z
from .terrain_stamper import ROAD_HALF_WIDTHS, DEFAULT_HALF_WIDTH

SUBDIV_STEP = 0.5    # m
JUNCTION_R  = 10.0   # m – max-pooling radius
SMOOTH_2D_R = 6.0    # m – post-max gentle blur radius
BLEND_M     = 2.0    # m – blend zone width at road boundary


class RoadStamper:

    def __init__(self, blend_m: float = BLEND_M):
        self.blend_m = blend_m

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def stamp(
        self,
        z_grid:   np.ndarray,   # (h+1, w+1) float32 terrain elevation
        meta:     dict,
        highways: list[dict],
    ) -> np.ndarray:
        """Return a copy of z_grid with roads imprinted."""

        # ── 1. Collect centerline points ───────────────────────────────────
        all_cx: list[float] = []
        all_cy: list[float] = []
        all_hw: list[float] = []

        for road in highways:
            nodes = road.get("nodes", [])
            if len(nodes) < 2:
                continue
            hw  = self._half_width(road.get("tags", {}))
            pts = [to_local(lon, lat, meta) for lon, lat in nodes]
            pts = self._subdivide(pts)
            if len(pts) < 2:
                continue
            all_cx.extend(p[0] for p in pts)
            all_cy.extend(p[1] for p in pts)
            all_hw.extend([hw] * len(pts))

        if not all_cx:
            return z_grid.copy()

        cx = np.asarray(all_cx, dtype=np.float64)
        cy = np.asarray(all_cy, dtype=np.float64)
        hw = np.asarray(all_hw, dtype=np.float64)

        # ── 2. Terrain Z at every centerline point ─────────────────────────
        print(f"   Stamp: Z-sampling {len(cx):,} bodů osy silnic...")
        tz = np.array(
            [sample_z(x, y, z_grid, meta) for x, y in zip(cx, cy)],
            dtype=np.float64,
        )

        # ── 3. Junction Z unification ──────────────────────────────────────
        print("   Stamp: unifikace Z na křižovatkách...")
        z_road = self._junction_unify(cx, cy, tz)   # always ≥ tz

        # ── 4. Terrain vertex positions ────────────────────────────────────
        h, w       = meta["h"], meta["w"]
        transform  = meta["transform"]
        lon_origin = meta["lon_origin"]
        lat_origin = meta["lat_origin"]
        mpd_lon    = meta["meters_per_deg_lon"]
        mpd_lat    = meta["meters_per_deg_lat"]

        rows, cols = np.mgrid[0 : h + 1, 0 : w + 1]
        lons = transform.c + cols * transform.a + rows * transform.b
        lats = transform.f + cols * transform.d + rows * transform.e
        vx = (lons - lon_origin) * mpd_lon
        vy = (lats - lat_origin) * mpd_lat
        tv = np.column_stack([vx.ravel(), vy.ravel()])   # (N_verts, 2)

        # ── 5. KD-tree: nearest road centerline point per terrain vertex ───
        print(f"   Stamp: KD-tree query pro {tv.shape[0]:,} terénních vrcholů...")
        tree         = cKDTree(np.column_stack([cx, cy]))
        dists, idxs  = tree.query(tv)

        # ── 6. Apply stamp ─────────────────────────────────────────────────
        z_flat   = z_grid.ravel().astype(np.float64)
        z_r      = z_road[idxs]
        hw_near  = hw[idxs]

        # Full stamp inside road footprint
        inside = dists <= hw_near
        z_flat[inside] = z_r[inside]

        # Smooth-step blend at road boundary
        blend = (~inside) & (dists <= hw_near + self.blend_m)
        if blend.any():
            t = np.clip((dists[blend] - hw_near[blend]) / self.blend_m, 0.0, 1.0)
            t_smooth            = 3 * t**2 - 2 * t**3   # smooth-step
            z_flat[blend]       = (z_r[blend] * (1.0 - t_smooth)
                                   + z_flat[blend] * t_smooth)

        n_stamped = int(inside.sum())
        print(f"   Stamp: {n_stamped:,} vrcholů razítkováno jako silnice")
        return z_flat.reshape(z_grid.shape).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Junction Z unification (identical logic to road_mesh._junction_unify)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _junction_unify(
        cx: np.ndarray,
        cy: np.ndarray,
        tz: np.ndarray,
    ) -> np.ndarray:
        res   = SUBDIV_STEP
        pad   = JUNCTION_R + SMOOTH_2D_R + 1.0
        x_min = cx.min() - pad;  x_max = cx.max() + pad
        y_min = cy.min() - pad;  y_max = cy.max() + pad
        nx = max(2, int((x_max - x_min) / res) + 1)
        ny = max(2, int((y_max - y_min) / res) + 1)
        ix = np.clip(((cx - x_min) / res).astype(np.intp), 0, nx - 1)
        iy = np.clip(((cy - y_min) / res).astype(np.intp), 0, ny - 1)
        raster = np.zeros((ny, nx), dtype=np.float64)
        np.maximum.at(raster, (iy, ix), tz)
        r_pix   = max(1, int(JUNCTION_R  / res))
        s_pix   = max(1, int(SMOOTH_2D_R / res))
        raised  = maximum_filter(raster, size=2 * r_pix + 1, mode="constant", cval=0.0)
        blended = uniform_filter(raised,  size=2 * s_pix + 1, mode="constant", cval=0.0)
        return np.maximum(blended[iy, ix], tz)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _subdivide(pts: list[tuple]) -> list[tuple]:
        step   = SUBDIV_STEP
        result: list[tuple] = []
        for i in range(len(pts) - 1):
            x0, y0 = pts[i];  x1, y1 = pts[i + 1]
            L = math.hypot(x1 - x0, y1 - y0)
            if L < 1e-6:
                continue
            n = max(1, math.ceil(L / step))
            for j in range(n):
                t = j / n
                result.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        if pts:
            result.append(pts[-1])
        return result

    @staticmethod
    def _half_width(tags: dict) -> float:
        try:
            w = float(tags.get("width", 0) or 0)
            if w > 0:
                return w / 2.0
        except (ValueError, TypeError):
            pass
        return ROAD_HALF_WIDTHS.get(tags.get("highway", ""), DEFAULT_HALF_WIDTH)
