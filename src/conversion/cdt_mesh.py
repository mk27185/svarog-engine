"""
CDT Mesh Builder
================
Produces a single terrain+roads OBJ via Delaunay triangulation.

Strategy
--------
1. Collect 2D point set:
     a) SRTM grid corners  (coarse, authoritative elevation data)
     b) Road boundary points  (left + right edge of each road strip,
        subdivided every SUBDIVISION_STEP metres)
2. Run 2-D Delaunay triangulation on the combined point set
   (no constraint segments — avoids road-intersection complexity).
   Dense road-boundary points naturally guide the triangulation to
   produce narrow road-aligned triangles.
3. Assign Z to every vertex:
     SRTM points  → z_grid value
     Road points  → bilinear sample from z_grid + Z_ROAD_OFFSET
   Road-edge Z values are smoothed along each road's profile so
   the strip follows the terrain smoothly (no SRTM staircase effect).
4. Tag each triangle as 'terrain' or 'roads':
     build one Shapely polygon per road strip,
     test triangle centroid with STRtree → fast spatial index.
5. Write OBJ with   g terrain / g roads   groups.

Dependencies: triangle (Shewchuk CDT), shapely ≥ 2.0
"""

import os
import math
import numpy as np
from scipy.ndimage import uniform_filter1d

import triangle as tr
from shapely import STRtree, points as shp_points
from shapely.geometry import Polygon, LineString

from .geo_utils import to_local, sample_z
from .terrain_stamper import ROAD_HALF_WIDTHS, DEFAULT_HALF_WIDTH

SUBDIVISION_STEP = 2.0    # metres between road-boundary vertices
SMOOTH_WINDOW    = 20     # road Z smoothing window (~40 m)
Z_ROAD_OFFSET    = 0.10   # metres road verts sit above terrain


class CDTMeshBuilder:
    """
    Single terrain+roads mesh via Delaunay triangulation.
    No Z-fighting, clean road strips, exact road-edge placement.
    """

    def __init__(self, output_dir: str = "outputs",
                 subdivision_step: float = SUBDIVISION_STEP):
        self.output_dir       = output_dir
        self.subdivision_step = subdivision_step

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def build_obj(
        self,
        z_grid:     np.ndarray,
        meta:       dict,
        highways:   list[dict],
        obj_name:   str | None = None,
        output_dir: str | None = None,
    ) -> str:
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 1 ── collect 2-D vertices ────────────────────────────────────
        print("   CDT: sbírám vrcholy …")
        srtm_xy, srtm_z      = self._srtm_points(z_grid, meta)
        road_xy, road_z, polys = self._road_points(z_grid, meta, highways)

        n_srtm = len(srtm_xy)
        all_xy = np.vstack([srtm_xy, road_xy]) if len(road_xy) else srtm_xy
        all_z  = np.concatenate([srtm_z, road_z]) if len(road_z) else srtm_z

        print(
            f"   CDT: {n_srtm:,} SRTM + {len(road_xy):,} road vrcholů "
            f"= {len(all_xy):,} celkem"
        )

        # 2 ── Delaunay triangulation ──────────────────────────────────
        print("   CDT: triangulace …")
        tri_out  = tr.triangulate({"vertices": all_xy})
        triangles = tri_out["triangles"]          # (T, 3) int indices
        print(f"   CDT: {len(triangles):,} trojúhelníků")

        # 3 ── tag faces ───────────────────────────────────────────────
        print("   CDT: tagování faces …")
        road_mask = self._tag_faces(triangles, all_xy, polys)

        # 4 ── write OBJ ───────────────────────────────────────────────
        fname = f"{obj_name}.obj" if obj_name else "terrain_cdt.obj"
        out   = os.path.join(output_dir, fname)
        self._write_obj(all_xy, all_z, triangles, road_mask, meta, out)
        return out

    # ------------------------------------------------------------------ #
    # Step 1 – point collection                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _srtm_points(
        z_grid: np.ndarray, meta: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = meta["h"], meta["w"]
        cw, ch = meta["cell_width_m"], meta["cell_height_m"]

        rows = np.arange(h + 1, dtype=np.float64)
        cols = np.arange(w + 1, dtype=np.float64)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")

        xy = np.stack([cc.ravel() * cw,
                       (h - rr.ravel()) * ch], axis=1)
        z  = z_grid.ravel().astype(np.float64)
        return xy, z

    def _road_points(
        self,
        z_grid:   np.ndarray,
        meta:     dict,
        highways: list[dict],
    ) -> tuple[np.ndarray, np.ndarray, list[Polygon]]:
        """
        Returns (road_xy, road_z, shapely_polygons).
        road_xy : (M, 2)  left + right edge points for all roads
        road_z  : (M,)    elevation at each road-boundary point + offset
        polygons: one Shapely Polygon per road strip for face tagging
        """
        step    = self.subdivision_step
        all_xy: list[tuple[float, float]] = []
        all_z:  list[float]               = []
        polys:  list[Polygon]             = []

        totx_min = 0.0
        totx_max = meta["total_width_m"]
        toty_min = 0.0
        toty_max = meta["total_height_m"]

        for road in highways:
            nodes = road["nodes"]
            if len(nodes) < 2:
                continue

            tags   = road.get("tags", {})
            half_w = self._half_width(tags)

            local  = [to_local(lon, lat, meta) for lon, lat in nodes]
            segs   = self._subdivide(local, step)
            if len(segs) < 2:
                continue

            pts    = np.array(segs, dtype=np.float64)
            normals = self._miter_normals(pts, half_w)   # already scaled

            left_pts  = pts + normals
            right_pts = pts - normals

            # Z: sample terrain, smooth along profile, add offset
            raw_z = np.array([
                sample_z(x, y, z_grid, meta) for x, y in segs
            ], dtype=np.float64)
            smooth_z = self._smooth(raw_z) + Z_ROAD_OFFSET

            # clip to terrain bbox with a small margin
            margin = half_w + 1.0
            for i, ((lx, ly), (rx, ry)) in enumerate(zip(left_pts, right_pts)):
                for px, py in [(lx, ly), (rx, ry)]:
                    if (totx_min - margin <= px <= totx_max + margin and
                            toty_min - margin <= py <= toty_max + margin):
                        all_xy.append((px, py))
                        all_z.append(float(smooth_z[i]))

            # Shapely polygon via LineString.buffer — always valid,
            # handles self-intersecting miter joints correctly.
            try:
                line = LineString(segs)
                poly = line.buffer(
                    half_w,
                    cap_style="flat",
                    join_style="mitre",
                    mitre_limit=3.0,
                )
                if not poly.is_empty:
                    polys.append(poly)
            except Exception:
                pass

        if not all_xy:
            return np.empty((0, 2)), np.empty(0), polys

        return np.array(all_xy), np.array(all_z), polys

    # ------------------------------------------------------------------ #
    # Step 2 – face tagging                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tag_faces(
        triangles: np.ndarray,
        xy:        np.ndarray,
        polys:     list[Polygon],
    ) -> np.ndarray:
        """Returns boolean array (T,) — True = road triangle."""
        if not polys:
            return np.zeros(len(triangles), dtype=bool)

        # Centroid of each triangle
        cx = xy[triangles, 0].mean(axis=1)   # (T,)
        cy = xy[triangles, 1].mean(axis=1)

        # Shapely STRtree bulk spatial query (Shapely ≥ 2.0)
        tree       = STRtree(polys)
        centroid_geoms = shp_points(np.stack([cx, cy], axis=1))  # (T,)
        # query returns (input_geometry_idx, tree_geometry_idx)
        # input = centroid points → input_idx are our face indices
        face_idx, _ = tree.query(centroid_geoms, predicate="within")

        mask = np.zeros(len(triangles), dtype=bool)
        mask[face_idx] = True
        return mask

    # ------------------------------------------------------------------ #
    # Step 3 – OBJ write                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write_obj(
        xy:        np.ndarray,   # (N, 2)
        z:         np.ndarray,   # (N,)
        triangles: np.ndarray,   # (T, 3)
        road_mask: np.ndarray,   # (T,)  bool
        meta:      dict,
        out_path:  str,
    ) -> None:
        n_road    = road_mask.sum()
        n_terrain = len(triangles) - n_road

        with open(out_path, "w") as f:
            f.write("# CDT Terrain+Roads OBJ – local metric coords\n")
            f.write(f"# Origin: lon={meta['lon_origin']:.6f}, "
                    f"lat={meta['lat_origin']:.6f}\n")
            f.write(f"# Vertices: {len(xy):,}  "
                    f"Faces: terrain={n_terrain:,} roads={n_road:,}\n")

            for i in range(len(xy)):
                f.write(f"v {xy[i,0]:.3f} {xy[i,1]:.3f} {z[i]:.3f}\n")

            f.write("g terrain\n")
            for tri in triangles[~road_mask]:
                v = tri + 1      # 1-based
                f.write(f"f {v[0]} {v[1]} {v[2]}\n")

            f.write("g roads\n")
            for tri in triangles[road_mask]:
                v = tri + 1
                f.write(f"f {v[0]} {v[1]} {v[2]}\n")

        print(
            f"   CDT OBJ: {out_path}  "
            f"({len(xy):,} vrcholů, "
            f"{n_terrain:,} terrain + {n_road:,} road faces)"
        )

    # ------------------------------------------------------------------ #
    # Geometry helpers                                                      #
    # ------------------------------------------------------------------ #

    def _subdivide(self, local_nodes: list[tuple], step: float) -> list[tuple]:
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
                pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        if local_nodes:
            pts.append(local_nodes[-1])
        return pts

    @staticmethod
    def _miter_normals(pts: np.ndarray, half_w: float) -> np.ndarray:
        """Per-vertex perpendicular offset vectors (magnitude = half_w with miter scale)."""
        n       = len(pts)
        normals = np.zeros((n, 2))
        MAX_MITER = 3.0

        def perp(d):
            L = math.hypot(d[0], d[1])
            return np.array([-d[1], d[0]]) / L if L > 1e-9 else np.array([0.0, 1.0])

        for i in range(n):
            if i == 0:
                norm  = perp(pts[1] - pts[0])
                scale = 1.0
            elif i == n - 1:
                norm  = perp(pts[-1] - pts[-2])
                scale = 1.0
            else:
                n_in  = perp(pts[i] - pts[i - 1])
                n_out = perp(pts[i + 1] - pts[i])
                miter = n_in + n_out
                mlen  = math.hypot(miter[0], miter[1])
                if mlen < 1e-9:
                    norm, scale = n_in, 1.0
                else:
                    norm  = miter / mlen
                    cos_a = max(abs(float(np.dot(n_in, norm))), 1.0 / MAX_MITER)
                    scale = min(1.0 / cos_a, MAX_MITER)

            normals[i] = norm * half_w * scale

        return normals

    @staticmethod
    def _smooth(zs: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
        if len(zs) < 3:
            return zs
        w = min(window, len(zs))
        return uniform_filter1d(zs, size=w, mode="nearest")

    @staticmethod
    def _half_width(tags: dict) -> float:
        hw_type = tags.get("highway", "")
        try:
            w = float(tags.get("width", 0) or 0)
            if w > 0:
                return w / 2.0
        except (ValueError, TypeError):
            pass
        return ROAD_HALF_WIDTHS.get(hw_type, DEFAULT_HALF_WIDTH)
