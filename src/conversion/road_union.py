"""
Road Union  –  2D polygon union → 3D road mesh
===============================================

Posloupnost:
    1. Každá osová linie silnice → buffer (Shapely) = 2D pruh šíře road_half_width.
    2. unary_union všech pruhů → jeden (nebo více) Polygon.
       Křižovatky jsou automaticky sloučeny – žádné překrytí, žádný Z-fighting.
    3. Zjednodušení hranice (volitelné).
    4. Triangulace každého polygonu (Triangle CDT, 'p' = bez Steiner bodů).
    5. Pro každý vrchol triangulace: sample_z(x, y, z_grid, meta) → Z povrchu silnice.
       Každý vrchol dostane SVŮJ terrain_z – silnice přesně kopíruje terén v celé ploše.
    6. Boční stěny: jen na VNĚJŠÍ hranici union-polygonu (vnitřní hranice = místa
       kde se cesty dotýkají = nemají stěnu). Stěna jde od povrchu silnice dolů na
       terrain_z hrany (+ garantovaná minimální hloubka z config).
    7. Výstup: groups 'g roads_surface' a 'g roads_walls' v jednom OBJ.
"""

import os
import sys
import math
import numpy as np

try:
    from shapely.geometry import LineString, Point, Polygon, MultiPolygon, GeometryCollection
    from shapely.ops      import unary_union
except ImportError:
    sys.exit("Chybí shapely: pip install shapely")

from .geo_utils import to_local, sample_z


class RoadUnion:

    def __init__(self, cfg):
        """
        cfg: modul nebo objekt s atributy z config.py:
             ROAD_Z_OFFSET, ROAD_SIDE_WALL_MIN_DEPTH, ROAD_HALF_WIDTHS,
             ROAD_DEFAULT_HALF_WIDTH, ROAD_EXCLUDED_TYPES,
             ROAD_BOUNDARY_SIMPLIFY
        """
        self.cfg = cfg

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
        cfg = self.cfg
        output_dir = output_dir or "outputs"
        os.makedirs(output_dir, exist_ok=True)

        # ── 1. Build 2D union of all road strips ───────────────────────────
        print("   RoadUnion: buffering silnic a union...")
        road_poly = self._build_union(highways, meta)

        if road_poly is None or road_poly.is_empty:
            print("   RoadUnion: žádná validní geometrie – přeskačeno")
            return ""

        # ── 2. Road surface: earcut triangulation per polygon ─────────────
        #    Boundary is the road-union polygon (smooth, no grid jaggies).
        #    Densified to ROAD_SURFACE_SUBDIV_M for accurate terrain-Z sampling.
        print("   RoadUnion: earcut triangulace silnic...")
        surf_verts: list[tuple] = []
        surf_faces: list[tuple] = []
        for poly in self._iter_polygons(road_poly):
            if not poly.is_valid or poly.is_empty:
                continue
            sv, sf = self._triangulate_earcut(poly, z_grid, meta)
            base = len(surf_verts)
            surf_verts.extend(sv)
            surf_faces.extend((f[0] + base, f[1] + base, f[2] + base) for f in sf)

        # ── 3. Side walls from union polygon exterior ring(s) ─────────────
        wall_verts: list[tuple] = []
        wall_faces: list[tuple] = []

        for poly in self._iter_polygons(road_poly):
            if not poly.is_valid or poly.is_empty:
                continue
            wv, wf = self._side_walls(
                list(poly.exterior.coords)[:-1],
                z_grid, meta,
            )
            wb = len(wall_verts)
            wall_verts.extend(wv)
            wall_faces.extend((f[0]+wb, f[1]+wb, f[2]+wb) for f in wf)

        # ── 7. Write OBJ ───────────────────────────────────────────────────
        fname = f"{obj_name}_roads.obj" if obj_name else "roads.obj"
        out   = os.path.join(output_dir, fname)

        with open(out, "w") as f:
            f.write("# Road OBJ – 2D union → CDT → per-vertex terrain Z\n")
            f.write(
                f"# Origin: lon={meta['lon_origin']:.6f}, "
                f"lat={meta['lat_origin']:.6f}  "
                f"Z_OFFSET={cfg.ROAD_Z_OFFSET} m\n"
            )
            f.write(f"# Surface: {len(surf_verts):,} vrcholů, {len(surf_faces):,} faces\n")
            f.write(f"# Walls:   {len(wall_verts):,} vrcholů, {len(wall_faces):,} faces\n\n")

            # surface group
            f.write("g roads_surface\n")
            for v in surf_verts:
                f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            for fc in surf_faces:
                f.write(f"f {fc[0]} {fc[1]} {fc[2]}\n")

            # wall group (vertex indices continue from surface)
            offset = len(surf_verts)
            f.write("\ng roads_walls\n")
            for v in wall_verts:
                f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            for fc in wall_faces:
                f.write(
                    f"f {fc[0]+offset} {fc[1]+offset} {fc[2]+offset}\n"
                )

        total_v = len(surf_verts) + len(wall_verts)
        total_f = len(surf_faces) + len(wall_faces)
        print(f"   Road OBJ: {out}  ({total_v:,} vrcholů, {total_f:,} faces)")
        return out

    # ------------------------------------------------------------------ #
    # 2-D union                                                            #
    # ------------------------------------------------------------------ #

    def _build_union(self, highways: list[dict], meta: dict):
        cfg = self.cfg
        excluded = getattr(cfg, "ROAD_EXCLUDED_TYPES", set())
        half_widths: dict = getattr(cfg, "ROAD_HALF_WIDTHS", {})
        default_hw: float = getattr(cfg, "ROAD_DEFAULT_HALF_WIDTH", 2.0)

        polys = []
        for road in highways:
            tags   = road.get("tags", {})
            hw_tag = road.get("highway") or tags.get("highway", "")
            if hw_tag in excluded:
                continue

            nodes = road.get("nodes", [])
            if len(nodes) < 2:
                continue

            # half-width: prefer explicit OSM width tag
            half_w = default_hw
            try:
                w = float(tags.get("width", 0) or 0)
                if w > 0:
                    half_w = w / 2.0
                else:
                    half_w = half_widths.get(hw_tag, default_hw)
            except (TypeError, ValueError):
                half_w = half_widths.get(hw_tag, default_hw)

            pts = [to_local(lon, lat, meta) for lon, lat in nodes]

            try:
                line = LineString(pts)
                if line.length < 1e-3:
                    continue
                poly = line.buffer(
                    half_w,
                    cap_style=2,      # flat caps
                    join_style=2,     # mitre joins (fewer vertices than round)
                    mitre_limit=2.0,
                )
                if poly.is_valid and not poly.is_empty:
                    polys.append(poly)
            except Exception:
                continue

        if not polys:
            return None

        print(f"   RoadUnion: {len(polys)} pruhů → unary_union...")
        union = unary_union(polys)
        if not union.is_valid:
            union = union.buffer(0)   # fix self-intersections
        return union

    # ------------------------------------------------------------------ #
    # Earcut-based road surface                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _densify(coords: list[tuple], step: float) -> list[tuple]:
        """Insert intermediate vertices so no segment is longer than `step` m."""
        if len(coords) < 2:
            return coords
        out = []
        for i in range(len(coords) - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            d = math.hypot(x1 - x0, y1 - y0)
            n = max(1, int(math.ceil(d / step)))
            for k in range(n):
                t = k / n
                out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        out.append(coords[-1])
        return out

    def _triangulate_earcut(
        self,
        poly:   Polygon,
        z_grid: np.ndarray,
        meta:   dict,
    ) -> tuple[list[tuple], list[tuple]]:
        """
        Triangulate a Shapely Polygon using mapbox-earcut.

        Steps:
          1. Densify every ring boundary to ≤ ROAD_SURFACE_SUBDIV_M spacing.
             → enough vertices for accurate per-vertex terrain Z sampling.
          2. Earcut triangulation (handles holes natively, no Steiner points,
             smooth polygon boundary → zero jagged grid edges).
          3. For each vertex: Z = sample_z(x, y, z_grid, meta) + Z_OFFSET.

        Returns (verts_3d [(x,y,z), ...], faces [(i,j,k) 1-indexed, ...]).
        """
        import mapbox_earcut as earcut_mod

        step    = getattr(self.cfg, "ROAD_SURFACE_SUBDIV_M", 2.0)
        z_off   = self.cfg.ROAD_Z_OFFSET

        ext_raw = list(poly.exterior.coords)[:-1]  # drop closing dup
        ext_pts = self._densify(ext_raw, step)

        all_pts: list[tuple] = list(ext_pts)
        ring_ends: list[int] = [len(ext_pts)]

        for interior in poly.interiors:
            hole_raw = list(interior.coords)[:-1]
            hole_pts = self._densify(hole_raw, step)
            all_pts.extend(hole_pts)
            ring_ends.append(len(all_pts))

        verts_2d = np.array(all_pts, dtype=np.float64)            # (N, 2)
        ring_arr = np.array(ring_ends, dtype=np.uint32)            # end indices

        idx_flat = earcut_mod.triangulate_float64(verts_2d, ring_arr)
        if len(idx_flat) == 0:
            return [], []

        faces_np = idx_flat.reshape(-1, 3)

        # Sample Z for every vertex
        xs = verts_2d[:, 0]
        ys = verts_2d[:, 1]
        zs = np.array([sample_z(x, y, z_grid, meta) for x, y in zip(xs, ys)],
                      dtype=np.float64) + z_off

        verts_3d = [(float(xs[i]), float(ys[i]), float(zs[i])) for i in range(len(xs))]
        faces    = [(int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1) for f in faces_np]
        return verts_3d, faces

    def _road_surface_from_grid(
        self,
        road_poly,          # Shapely Polygon or MultiPolygon
        z_grid: np.ndarray,
        meta:   dict,
    ) -> tuple[list[tuple], list[tuple]]:
        """
        Extract terrain-grid triangles inside road_poly as the road surface.

        Algorithm:
          1. For every terrain grid vertex compute local-metric (x,y).
          2. Vectorised shapely.contains_xy check on road_poly.
          3. For each quad cell include a triangle if ≥2 of its 3 corners
             are inside the road polygon.
          4. Z = z_grid[row,col] + ROAD_Z_OFFSET — exact terrain following.

        This avoids all CDT issues with narrow corridors and holes.
        Resolution matches terrain upsampling (~2 m cells at 10×).
        """
        from shapely import contains_xy as _contains_xy

        h, w  = meta["h"], meta["w"]
        cw    = meta["cell_width_m"]
        ch    = meta["cell_height_m"]
        z_off = self.cfg.ROAD_Z_OFFSET

        # Local metric coords of every grid vertex
        x_arr = np.arange(w + 1, dtype=np.float64) * cw       # (w+1,)  E→
        y_arr = (h - np.arange(h + 1, dtype=np.float64)) * ch  # (h+1,)  N↑
        XX, YY = np.meshgrid(x_arr, y_arr)                      # (h+1, w+1)

        # Vectorised containment — True where vertex is inside any road
        inside = _contains_xy(road_poly, XX.ravel(), YY.ravel()).reshape(h + 1, w + 1)

        # Per-quad inside counts for each split triangle
        tl = inside[:-1, :-1];  tr = inside[:-1, 1:]
        bl = inside[1:,  :-1];  br = inside[1:,  1:]

        # Upper-left  triangle (TL,TR,BL) and lower-right (TR,BR,BL)
        inc1 = (tl.astype(np.int8) + tr.astype(np.int8) + bl.astype(np.int8)) >= 2
        inc2 = (tr.astype(np.int8) + br.astype(np.int8) + bl.astype(np.int8)) >= 2

        # Mark used vertices
        used = np.zeros((h + 1, w + 1), dtype=bool)
        used[:-1, :-1] |= inc1;  used[:-1, 1:] |= inc1;  used[1:, :-1] |= inc1
        used[:-1, 1:]  |= inc2;  used[1:,  1:] |= inc2;  used[1:, :-1] |= inc2

        # Build vertex list + index map (only used vertices)
        used_rows, used_cols = np.where(used)
        vert_map = np.full((h + 1, w + 1), -1, dtype=np.int32)
        verts_3d: list[tuple] = []

        for idx, (r, c) in enumerate(zip(used_rows.tolist(), used_cols.tolist())):
            vert_map[r, c] = idx
            verts_3d.append((
                float(XX[r, c]),
                float(YY[r, c]),
                float(z_grid[r, c]) + z_off,
            ))

        # Build faces (1-indexed)
        faces: list[tuple] = []
        rows1, cols1 = np.where(inc1)
        for r, c in zip(rows1.tolist(), cols1.tolist()):
            a, b, d_ = vert_map[r, c], vert_map[r, c+1], vert_map[r+1, c]
            if a >= 0 and b >= 0 and d_ >= 0:
                faces.append((a + 1, b + 1, d_ + 1))

        rows2, cols2 = np.where(inc2)
        for r, c in zip(rows2.tolist(), cols2.tolist()):
            a, b, d_ = vert_map[r, c+1], vert_map[r+1, c+1], vert_map[r+1, c]
            if a >= 0 and b >= 0 and d_ >= 0:
                faces.append((a + 1, b + 1, d_ + 1))

        return verts_3d, faces

    # ------------------------------------------------------------------ #
    # Side walls                                                           #
    # ------------------------------------------------------------------ #

    def _side_walls(
        self,
        ring_coords: list[tuple],   # ordered (x,y) of exterior ring (no dup)
        z_grid:      np.ndarray,
        meta:        dict,
    ) -> tuple[list, list]:
        """
        Build side walls for one closed ring.
        Wall top  = terrain_z + Z_OFFSET  (road surface)
        Wall bot  = min(terrain_z,  road_top - SIDE_WALL_MIN_DEPTH)
        """
        cfg       = self.cfg
        z_off     = cfg.ROAD_Z_OFFSET
        min_depth = getattr(cfg, "ROAD_SIDE_WALL_MIN_DEPTH", 0.20)

        verts: list[tuple] = []
        for (x, y) in ring_coords:
            tz   = float(sample_z(x, y, z_grid, meta))
            z_top = tz + z_off
            z_bot = min(tz, z_top - min_depth)   # guarantee min wall depth
            verts.append((float(x), float(y), z_top))   # top
            verts.append((float(x), float(y), z_bot))   # bottom

        n     = len(ring_coords)
        faces = []
        for i in range(n):
            j  = (i + 1) % n
            t0 = 2*i + 1;  b0 = 2*i + 2
            t1 = 2*j + 1;  b1 = 2*j + 2
            # Two triangles per wall quad (CCW → outward normal)
            faces.append((t0, b0, t1))
            faces.append((b0, b1, t1))

        return verts, faces

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _iter_polygons(geom) -> list[Polygon]:
        """Flatten any geometry to a list of Polygons."""
        if isinstance(geom, Polygon):
            return [geom]
        if isinstance(geom, (MultiPolygon, GeometryCollection)):
            result = []
            for g in geom.geoms:
                result.extend(RoadUnion._iter_polygons(g))
            return result
        return []
