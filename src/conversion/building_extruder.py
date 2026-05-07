"""
OSM building footprint extruder.

Geometry pipeline per building:

  1.  _apply_shape()    – replace nodes for building:shape=cylinder etc.
  2.  _wall_height()    – total_height − roof_height − min_height  (OSMBuildings formula)
  3.  terrain sampling  – per-corner z_grid lookup
  4.  base ring         – terrain-following (min_height=0) or flat offset (min_height>0)
  5.  top ring (plate)  – flat at plate_z, or per-vertex for skillion
  6.  _build_roof()     – extra vertices + faces for the roof cap
  7.  OBJ write

Supported roof:shape values (OSM Simple 3D Buildings):
  flat, pyramid, cone, skillion, gabled, hipped, half-hipped,
  gambrel, mansard, round, dome, onion
"""

import math
import os

import triangle as tr

from .geo_utils import to_local, sample_z

DEFAULT_BUILDING_HEIGHT = 10.0
LEVEL_HEIGHT  = 3.0
CYLINDER_SEGS = 32   # polygon segments for building:shape=cylinder
DOME_RINGS    = 5    # latitude rings for dome/onion roof


class BuildingExtruder:

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, buildings, z_grid, meta, *, obj_name=None, output_dir=None):
        return self.extrude_buildings(buildings, z_grid, meta,
                                      obj_name=obj_name, output_dir=output_dir)

    def extrude_buildings(
        self,
        buildings:  list[dict],
        z_grid,
        meta:       dict,
        obj_name:   str | None = None,
        output_dir: str | None = None,
        tif_path:   str | None = None,   # legacy; ignored when z_grid provided
    ) -> str:
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if z_grid is None:
            if tif_path is None:
                raise ValueError("Provide either z_grid+meta or tif_path.")
            from .terrain_converter import TerrainConverter
            z_grid, meta = TerrainConverter().build_grid(tif_path)

        all_vertices: list[tuple] = []
        all_faces:    list[tuple] = []
        skipped = 0

        # Tile bounds for centroid-based deduplication.
        # Overpass returns every way whose bbox overlaps the query bbox, so large
        # buildings appear in the cache of multiple adjacent tiles.  We assign each
        # building to exactly one tile — the one whose bbox contains the centroid.
        # The full (unclipped) polygon is then extruded, which may overhang the
        # tile edge slightly, but avoids broken roof geometry from polygon clipping.
        _tile_bounds = meta.get("bounds")   # rasterio BoundingBox or None

        for building in buildings:
          try:
            nodes = building["nodes"]
            tags  = building.get("tags", {})

            if len(nodes) < 3:
                skipped += 1
                continue

            # 1. Shape override (cylinder → synthetic circle polygon)
            nodes = self._apply_shape(nodes, tags)
            if len(nodes) < 3:
                skipped += 1
                continue

            # 1b. Centroid-based tile assignment.
            # Skip this building if its centroid falls outside this tile.
            # (It will be rendered by the neighbouring tile that owns it.)
            if _tile_bounds is not None:
                c_lon = sum(nd["lon"] for nd in nodes) / len(nodes)
                c_lat = sum(nd["lat"] for nd in nodes) / len(nodes)
                if not (
                    _tile_bounds.left   <= c_lon <= _tile_bounds.right and
                    _tile_bounds.bottom <= c_lat <= _tile_bounds.top
                ):
                    skipped += 1
                    continue

            # 2. Heights
            wall_h = self._wall_height(tags)
            min_h  = self._parse_min_height(tags)
            roof_h = self._roof_height(tags, self._total_height(tags))

            # 3. Local metric coords + terrain elevation
            local_xy = [to_local(nd["lon"], nd["lat"], meta) for nd in nodes]
            corner_zs = [sample_z(x, y, z_grid, meta) for x, y in local_xy]

            # Normalise to CCW so all face normals point outward.
            # OSM ways may be traced in either direction.
            _n = len(local_xy)
            _area2 = sum(
                local_xy[_i][0] * local_xy[(_i + 1) % _n][1]
                - local_xy[(_i + 1) % _n][0] * local_xy[_i][1]
                for _i in range(_n)
            )
            if _area2 < 0:
                local_xy  = local_xy[::-1]
                corner_zs = corner_zs[::-1]

            # Anchor the building height to the terrain at the footprint
            # centroid.  Using the centroid (rather than the max corner) keeps
            # neighbouring building:parts of the same complex at consistent
            # plate heights even when the terrain varies slightly across parts.
            # A safety clamp ensures the top ring never dips below any base
            # corner (prevents inverted walls on steep slopes).
            cx = sum(x for x, y in local_xy) / len(local_xy)
            cy = sum(y for x, y in local_xy) / len(local_xy)
            anchor_z = sample_z(cx, cy, z_grid, meta)
            # On steep slopes the centroid may be lower than some corners.
            # Clamp so walls are at least 10 % of their height on the high side.
            # Only applies when there are actual walls (wall_h > 0); for roof-only
            # sections (wall_h = 0, e.g. canopies) the centroid anchor is used as-is.
            if wall_h > 0:
                anchor_z = max(anchor_z, max(corner_zs) - wall_h * 0.9)

            # 4. Base ring
            if min_h > 0.0:
                # Elevated part: flat base at anchor + min_height
                base_zs = [anchor_z + min_h] * len(nodes)
            else:
                # Ground-level building: base follows terrain (no sinking)
                base_zs = corner_zs

            # 5. Roof geometry
            plate_z = anchor_z + min_h + wall_h
            top_zs, roof_extra_verts, roof_extra_faces = self._build_roof(
                local_xy, plate_z, roof_h, tags
            )
            n_nodes = len(nodes)

            # 6. Emit geometry
            base_start = len(all_vertices)
            for i, (x, y) in enumerate(local_xy):
                all_vertices.append((x, y, base_zs[i]))

            top_start = len(all_vertices)
            for i, (x, y) in enumerate(local_xy):
                all_vertices.append((x, y, top_zs[i]))

            extra_start = len(all_vertices)
            for v in roof_extra_verts:
                all_vertices.append(v)

            # Walls – two triangles per edge
            for i in range(n_nodes):
                j  = (i + 1) % n_nodes
                b0 = base_start + i + 1
                b1 = base_start + j + 1
                t0 = top_start  + i + 1
                t1 = top_start  + j + 1
                all_faces.append((b0, b1, t0))
                all_faces.append((b1, t1, t0))

            # Roof faces (relative indices → global 1-based)
            for fi, fj, fk in roof_extra_faces:
                def _r(idx, ts=top_start, es=extra_start, nn=n_nodes):
                    if idx < nn:
                        return ts + idx + 1
                    return es + (idx - nn) + 1
                all_faces.append((_r(fi), _r(fj), _r(fk)))

          except Exception:
            skipped += 1
            continue

        suffix = f"{obj_name}_buildings" if obj_name else "buildings"
        out    = os.path.join(output_dir, f"{suffix}.obj")

        with open(out, "w") as f:
            f.write("# Building OBJ – local metric coords\n")
            f.write(f"# Origin: lon={meta['lon_origin']:.6f}, lat={meta['lat_origin']:.6f}\n")
            f.write(f"# Buildings: {len(buildings) - skipped} extruded, {skipped} skipped\n")
            for v in all_vertices:
                f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            for fc in all_faces:
                f.write(f"f {fc[0]} {fc[1]} {fc[2]}\n")

        print(f"   Budovy zapsány: {out}  "
              f"({len(all_vertices)} vrcholů, {len(all_faces)} faces)")
        return out

    # ------------------------------------------------------------------
    # Roof dispatcher
    # ------------------------------------------------------------------

    @staticmethod
    def _build_roof(
        top_xy:      list[tuple[float, float]],
        plate_z:     float,
        roof_height: float,
        tags:        dict,
    ) -> tuple[list[float], list[tuple], list[tuple]]:
        """
        Build roof geometry above the wall plate.

        Returns
        -------
        top_zs        list[float]   per-vertex plate height  (varies for skillion)
        extra_verts   list[tuple]   additional (x,y,z) roof vertices
        extra_faces   list[tuple]   (i,j,k) where i < n → top ring, i ≥ n → extra_verts
        """
        n = len(top_xy)
        shape = (tags.get("roof:shape") or "flat").lower()

        # Flat / no roof section
        if roof_height <= 0.0 or shape in ("flat", "no", "none", ""):
            return [plate_z] * n, [], BuildingExtruder._triangulate_polygon(top_xy)

        # Ridge roofs (gabled, hipped, …) need a ridge direction.
        # OSMBuildings requires an explicit roof:direction tag; without one it
        # renders a flat roof.  Our OBBox auto-direction works well for simple
        # rectangles (≤ 5 unique vertices) but produces broken geometry for
        # complex polygons.  We therefore use the ridged algorithm only when:
        #   a) roof:direction is explicitly tagged, OR
        #   b) the cleaned polygon is simple (≤ 5 unique vertices, i.e. ~rectangular)
        # Otherwise we fall back to a pyramid, which always looks clean.
        def _ridge(end_inset: float):
            clean_n = len(BuildingExtruder._clean_polygon(top_xy))
            has_dir = tags.get("roof:direction") is not None
            if has_dir or clean_n <= 5:
                try:
                    return BuildingExtruder._roof_ridged(
                        top_xy, n, plate_z, roof_height, tags, end_inset)
                except Exception:
                    pass
            return BuildingExtruder._roof_pyramid(top_xy, n, plate_z, roof_height)

        dispatch = {
            "pyramid":     lambda: BuildingExtruder._roof_pyramid(top_xy, n, plate_z, roof_height),
            "cone":        lambda: BuildingExtruder._roof_pyramid(top_xy, n, plate_z, roof_height),
            "skillion":    lambda: BuildingExtruder._roof_skillion(top_xy, n, plate_z, roof_height, tags),
            "dome":        lambda: BuildingExtruder._roof_dome(top_xy, n, plate_z, roof_height, "dome"),
            "onion":       lambda: BuildingExtruder._roof_dome(top_xy, n, plate_z, roof_height, "onion"),
            "gabled":      lambda: _ridge(0.00),
            "hipped":      lambda: _ridge(0.30),
            "half-hipped": lambda: _ridge(0.15),
            "gambrel":     lambda: _ridge(0.00),
            "mansard":     lambda: _ridge(0.00),
            "round":       lambda: _ridge(0.00),
        }
        handler = dispatch.get(shape)
        if handler:
            return handler()

        # Unknown shape → flat
        return [plate_z] * n, [], BuildingExtruder._triangulate_polygon(top_xy)

    # ------------------------------------------------------------------
    # Roof shapes
    # ------------------------------------------------------------------

    @staticmethod
    def _roof_pyramid(top_xy, n, plate_z, roof_height):
        """Single apex at the polygon centroid (pyramid, cone)."""
        cx = sum(x for x, y in top_xy) / n
        cy = sum(y for x, y in top_xy) / n
        apex_idx = n   # one extra vertex
        faces = [(i, (i + 1) % n, apex_idx) for i in range(n)]
        return [plate_z] * n, [(cx, cy, plate_z + roof_height)], faces

    @staticmethod
    def _roof_skillion(top_xy, n, plate_z, roof_height, tags):
        """
        One-directional sloped roof.

        roof:direction (compass bearing) = direction water flows (downslope).
        Low side stays at plate_z; high side rises to plate_z + roof_height.
        Falls back to flat when direction is unresolvable.
        """
        sdx, sdy = BuildingExtruder._slope_vector(top_xy, tags)

        dots    = [x * sdx + y * sdy for x, y in top_xy]
        d_min   = min(dots)
        d_max   = max(dots)
        span    = d_max - d_min

        if span < 0.1:
            return [plate_z] * n, [], BuildingExtruder._triangulate_polygon(top_xy)

        top_zs = [plate_z + roof_height * (d - d_min) / span for d in dots]
        faces  = BuildingExtruder._triangulate_polygon(top_xy)
        return top_zs, [], faces

    # ------------------------------------------------------------------
    # Oriented Bounding Box helpers (S3DB spec §3.5)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_obbox(
        poly: list[tuple[float, float]],
    ) -> tuple[float, float, float, float, float, float, float, float] | None:
        """
        Minimum-area oriented bounding box via rotating-calipers over polygon edges.

        Returns (cx, cy, half_len, half_wid, rdx, rdy, pdx, pdy) where:
            half_len >= half_wid  (long and short half-dimensions)
            (rdx, rdy)            unit vector along the LONG axis
            (pdx, pdy)            unit vector along the SHORT axis (perpendicular, CCW)
        Returns None for degenerate polygons.
        """
        best_area = float("inf")
        result: tuple | None = None

        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            ex = poly[j][0] - poly[i][0]
            ey = poly[j][1] - poly[i][1]
            L  = math.sqrt(ex * ex + ey * ey)
            if L < 1e-10:
                continue
            dx, dy = ex / L, ey / L   # unit along edge
            px, py = -dy, dx          # unit perpendicular (CCW 90°)

            dp = [v[0] * dx + v[1] * dy for v in poly]
            pp = [v[0] * px + v[1] * py for v in poly]
            d0, d1 = min(dp), max(dp)
            p0, p1 = min(pp), max(pp)
            area   = (d1 - d0) * (p1 - p0)

            if area < best_area:
                best_area = area
                dc = (d0 + d1) / 2
                pc = (p0 + p1) / 2
                cx = dc * dx + pc * px
                cy = dc * dy + pc * py
                ld, lp = d1 - d0, p1 - p0
                if ld >= lp:                      # edge direction is the LONG axis
                    result = (cx, cy, ld / 2, lp / 2, dx, dy, px, py)
                else:                             # perpendicular is the LONG axis
                    result = (cx, cy, lp / 2, ld / 2, px, py, dx, dy)

        return result

    @staticmethod
    def _ridge_intersections(
        top_xy: list[tuple[float, float]],
        rdx: float, rdy: float,
        lx: float, ly: float,
        r_min: float, r_max: float,
    ) -> list[tuple[int, float, float, float, float]]:
        """
        Find where the infinite ridge line (lx + t*rdx, ly + t*rdy) intersects
        polygon edges strictly inside each edge (0.001 < s < 0.999).
        Only intersections whose absolute-t lies in [r_min, r_max] are returned.
        Result sorted by t (ascending).
        """
        hits: list[tuple[int, float, float, float, float]] = []
        for i in range(len(top_xy)):
            j  = (i + 1) % len(top_xy)
            ax, ay = top_xy[i]
            bx, by = top_xy[j]
            denom  = rdx * (by - ay) - rdy * (bx - ax)
            if abs(denom) < 1e-12:
                continue
            s = (rdx * (ly - ay) - rdy * (lx - ax)) / denom
            if not (0.001 < s < 0.999):
                continue
            ix = ax + s * (bx - ax)
            iy = ay + s * (by - ay)
            t  = ix * rdx + iy * rdy
            if r_min <= t <= r_max:
                hits.append((i, s, ix, iy, t))
        hits.sort(key=lambda h: h[4])
        return hits

    # ------------------------------------------------------------------
    # Ridge-based roof  (gabled / hipped / half-hipped / mansard …)
    # ------------------------------------------------------------------

    @staticmethod
    def _roof_ridged(top_xy, n, plate_z, roof_height, tags, end_inset: float):
        """
        S3DB-compliant ridge roof for any polygon.

        Implements the Oriented Bounding Box (OBBox) approach from the spec:
          1. Compute the minimum-area OBBox of the footprint.
          2. Determine the ridge direction and dimensions from the OBBox.
          3. Per-vertex z ("clip" operation): extend each wall vertex upward to
             the height of the roof surface directly above it.  This closes the
             gap between irregular walls and the OBBox roof.
          4. Place ridge endpoint(s) R1, R2 as extra vertices at full roof height.
          5. Triangulate the roof cap with CDT, respecting the polygon boundary.
        """
        # --- 1. OBBox --------------------------------------------------------
        obbox = BuildingExtruder._compute_obbox(top_xy)
        if obbox is None:
            return BuildingExtruder._roof_pyramid(top_xy, n, plate_z, roof_height)

        cx, cy, half_len, half_wid, rdx, rdy, pdx, pdy = obbox

        # --- 2. Override ridge direction via roof:direction tag --------------
        roof_dir = (tags.get("roof:direction") or "").strip()
        if roof_dir:
            try:
                bearing = float(roof_dir)
            except ValueError:
                _compass = {"N": 0, "NE": 45, "E": 90, "SE": 135,
                            "S": 180, "SW": 225, "W": 270, "NW": 315}
                bearing = _compass.get(roof_dir.upper(), -1.0)
            if bearing >= 0:
                # roof:direction is the DOWNSLOPE direction; ridge ⊥ to it
                slope_rad = math.radians(bearing)
                sx, sy = math.sin(slope_rad), math.cos(slope_rad)  # downslope
                rdx, rdy = -sy, sx   # ridge = 90° CCW of downslope
                pdx, pdy = -rdy, rdx
                # Recompute half-dimensions from the new projection
                dp = [v[0] * rdx + v[1] * rdy for v in top_xy]
                pp = [v[0] * pdx + v[1] * pdy for v in top_xy]
                half_len = (max(dp) - min(dp)) / 2
                half_wid = (max(pp) - min(pp)) / 2
                cx_proj  = (max(dp) + min(dp)) / 2
                cp_proj  = (max(pp) + min(pp)) / 2
                cx = cx_proj * rdx + cp_proj * pdx
                cy = cx_proj * rdy + cp_proj * pdy

        if half_wid < 0.1:
            return BuildingExtruder._roof_pyramid(top_xy, n, plate_z, roof_height)

        # --- 3. Ridge span ---------------------------------------------------
        inset = 2 * half_len * end_inset
        r1_rel = -half_len + inset    # relative to OBBox centre along ridge
        r2_rel =  half_len - inset
        if r1_rel >= r2_rel:
            return BuildingExtruder._roof_pyramid(top_xy, n, plate_z, roof_height)

        # --- 4. Per-vertex z (clip / wall-extend operation) ------------------
        #  t  = projection onto ridge direction (relative to OBBox centre)
        #  d  = perpendicular distance from ridge centre-line
        #  z_side = factor from side slope  (1 at centre, 0 at eave)
        #  z_hip  = factor from hip ends    (only for hipped; 1 inside ridge span)
        def roof_factor(vx: float, vy: float) -> float:
            t  = (vx - cx) * rdx + (vy - cy) * rdy
            d  = abs((vx - cx) * pdx + (vy - cy) * pdy)
            z_side = max(0.0, 1.0 - d / half_wid)
            if end_inset > 0.0:
                z_h1 = (t - (-half_len)) / max(inset, 1e-10)
                z_h2 = ( half_len - t  ) / max(inset, 1e-10)
                return max(0.0, min(z_side, z_h1, z_h2))
            return max(0.0, z_side)

        top_zs = [plate_z + roof_height * roof_factor(x, y) for x, y in top_xy]

        # --- 5. Ridge endpoints in 2-D (absolute coordinates) ---------------
        R1_idx = n
        R2_idx = n + 1

        if end_inset == 0.0:
            # Gabled: endpoints ON the polygon boundary – find via intersection
            r1_abs = cx * rdx + cy * rdy + r1_rel   # absolute projection of R1
            r2_abs = cx * rdx + cy * rdy + r2_rel
            hits = BuildingExtruder._ridge_intersections(
                top_xy, rdx, rdy, cx, cy, r1_abs, r2_abs
            )
            if len(hits) < 2 or hits[0][0] == hits[-1][0]:
                # Ridge doesn't cross polygon cleanly → fall through to interior
                end_inset_eff = 0.01          # treat as very slightly hipped
                inset_eff = 2 * half_len * end_inset_eff
                r1_rel = -half_len + inset_eff
                r2_rel =  half_len - inset_eff
            else:
                inter1, inter2 = hits[0], hits[-1]
                R1_2d = (inter1[2], inter1[3])
                R2_2d = (inter2[2], inter2[3])
                e1, e2 = inter1[0], inter2[0]

                extra_verts = [
                    (R1_2d[0], R1_2d[1], plate_z + roof_height),
                    (R2_2d[0], R2_2d[1], plate_z + roof_height),
                ]

                # Triangulate each slope half via CDT
                extra_faces = BuildingExtruder._ridged_cap_cdt(
                    top_xy, n, R1_2d, R2_2d, R1_idx, R2_idx, e1, e2
                )
                return top_zs, extra_verts, extra_faces

        # Hipped (or gabled fallback): ridge endpoints are INTERIOR points
        r1_abs_x = cx + r1_rel * rdx
        r1_abs_y = cy + r1_rel * rdy
        r2_abs_x = cx + r2_rel * rdx
        r2_abs_y = cy + r2_rel * rdy
        R1_2d = (r1_abs_x, r1_abs_y)
        R2_2d = (r2_abs_x, r2_abs_y)

        extra_verts = [
            (R1_2d[0], R1_2d[1], plate_z + roof_height),
            (R2_2d[0], R2_2d[1], plate_z + roof_height),
        ]

        # Triangulate entire polygon with R1, R2 as interior Steiner points
        extra_faces = BuildingExtruder._ridged_cap_interior(
            top_xy, n, R1_2d, R2_2d, R1_idx, R2_idx
        )
        return top_zs, extra_verts, extra_faces

    @staticmethod
    def _ridged_cap_cdt(
        top_xy, n,
        R1_2d, R2_2d,
        R1_idx, R2_idx,
        e1: int, e2: int,
    ) -> list[tuple[int, int, int]]:
        """
        Triangulate a gabled roof cap by CDT.
        R1 lies on edge e1 (between vertex e1 and e1+1).
        R2 lies on edge e2 (between vertex e2 and e2+1).
        Returns face index triples using the (top_ring + extra_verts) convention.

        Also emits the two GABLE END WALL triangles that fill the vertical
        triangular gap at the gable ends (from wall-plate at plate_z up to the
        ridge point at plate_z + roof_height).  Without these, the gable ends
        of the building are visibly open.
        """
        # Gable end wall fill triangles – one per ridge endpoint.
        # Each is the triangle (v_i @ plate_z, v_i+1 @ plate_z, R @ plate_z+h).
        # Winding: for a CCW polygon, (e, e+1, R) gives an outward-facing normal.
        gable_fills: list[tuple[int, int, int]] = [
            (e1, (e1 + 1) % n, R1_idx),
            (e2, (e2 + 1) % n, R2_idx),
        ]

        # Build PSLG: polygon boundary with R1/R2 inserted at their edges
        all_pts: list[tuple[float, float]] = list(top_xy) + [R1_2d, R2_2d]
        N = n + 2
        segs: list[list[int]] = []
        for i in range(n):
            j = (i + 1) % n
            if i == e1:
                segs += [[i, R1_idx], [R1_idx, j]]
            elif i == e2:
                segs += [[i, R2_idx], [R2_idx, j]]
            else:
                segs.append([i, j])

        # Map: local PSLG index → (top_ring | extra_verts) index
        local_to_global = list(range(n)) + [R1_idx, R2_idx]

        try:
            verts_2d = [[float(x), float(y)] for x, y in all_pts]
            result   = tr.triangulate({"vertices": verts_2d, "segments": segs}, "p")
            out_v    = result.get("vertices", verts_2d)
            if len(out_v) == N:    # no unexpected Steiner points
                slope_faces = [
                    (local_to_global[int(t[0])],
                     local_to_global[int(t[1])],
                     local_to_global[int(t[2])])
                    for t in result.get("triangles", [])
                ]
                return gable_fills + slope_faces
        except Exception:
            pass

        # Ear-clip fallback on extended polygon (CCW normalised)
        ring_order = list(range(N))
        pts_ext    = list(all_pts)
        area2      = sum(
            pts_ext[i][0] * pts_ext[(i + 1) % N][1]
            - pts_ext[(i + 1) % N][0] * pts_ext[i][1]
            for i in range(N)
        )
        if area2 < 0:
            pts_ext   = pts_ext[::-1]
            ring_order = ring_order[::-1]
        tris = BuildingExtruder._ear_clip(pts_ext, ring_order)
        return gable_fills + [
            (local_to_global[a], local_to_global[b], local_to_global[c])
            for a, b, c in tris
        ]

    @staticmethod
    def _ridged_cap_interior(
        top_xy, n,
        R1_2d, R2_2d,
        R1_idx, R2_idx,
    ) -> list[tuple[int, int, int]]:
        """
        Triangulate a hipped roof cap.
        R1 and R2 are INTERIOR points of the polygon (inset ridge endpoints).
        CDT with R1/R2 added as unconstrained interior vertices.
        """
        all_pts = list(top_xy) + [R1_2d, R2_2d]
        N = n + 2
        local_to_global = list(range(n)) + [R1_idx, R2_idx]

        try:
            verts_2d = [[float(x), float(y)] for x, y in all_pts]
            segs     = [[i, (i + 1) % n] for i in range(n)]  # boundary only
            result   = tr.triangulate({"vertices": verts_2d, "segments": segs}, "p")
            out_v    = result.get("vertices", verts_2d)
            if len(out_v) == N:
                return [
                    (local_to_global[int(t[0])],
                     local_to_global[int(t[1])],
                     local_to_global[int(t[2])])
                    for t in result.get("triangles", [])
                ]
        except Exception:
            pass

        # Fallback: classic edge-to-ridge (only if convex, else pyramid)
        if not BuildingExtruder._is_convex(top_xy):
            # Return pyramid-style apex using R1 as the apex
            return [(i, (i + 1) % n, R1_idx) for i in range(n)]

        # Convex polygon classic assignment
        projs     = [x * (R2_2d[0] - R1_2d[0]) + y * (R2_2d[1] - R1_2d[1])
                     for x, y in top_xy]
        mid_p     = (max(projs) + min(projs)) / 2
        faces: list[tuple[int, int, int]] = []
        for i in range(n):
            j = (i + 1) % n
            if projs[i] <= mid_p and projs[j] <= mid_p:
                faces.append((i, j, R1_idx))
            elif projs[i] > mid_p and projs[j] > mid_p:
                faces.append((i, j, R2_idx))
            elif projs[i] <= mid_p:
                faces += [(i, j, R2_idx), (i, R2_idx, R1_idx)]
            else:
                faces += [(i, j, R1_idx), (i, R1_idx, R2_idx)]
        return faces

    @staticmethod
    def _is_convex(poly: list[tuple[float, float]]) -> bool:
        """Return True when the polygon (assumed CCW) is strictly convex."""
        n = len(poly)
        sign = None
        for i in range(n):
            a = poly[(i - 1) % n]
            b = poly[i]
            c = poly[(i + 1) % n]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if abs(cross) < 1e-10:
                continue
            if sign is None:
                sign = cross > 0
            elif (cross > 0) != sign:
                return False
        return True

    @staticmethod
    def _roof_dome(top_xy, n, plate_z, roof_height, profile: str):
        """
        Hemispherical dome or onion dome, built as DOME_RINGS latitude rings
        of n vertices each, tapering to a central apex.

        The ring vertices are evenly spaced circles in local metric space
        centred at the polygon centroid.  For building:shape=cylinder footprints
        (already ~circular) this gives a smooth rounded cap.
        """
        xs    = [p[0] for p in top_xy]
        ys    = [p[1] for p in top_xy]
        cx    = (min(xs) + max(xs)) / 2
        cy    = (min(ys) + max(ys)) / 2
        base_r = max(0.1, min((max(xs) - min(xs)) / 2,
                               (max(ys) - min(ys)) / 2))

        extra_verts: list[tuple] = []
        ring_starts: list[int]   = []   # index in extra_verts where each ring starts

        for ring_i in range(1, DOME_RINGS + 1):
            frac  = ring_i / DOME_RINGS
            angle = math.pi / 2 * frac

            if profile == "onion":
                # Swell out before 40 % height, then taper
                bulge = 1.0 + 0.35 * math.sin(math.pi * min(frac / 0.4, 1.0))
                r     = base_r * bulge * math.cos(angle)
            else:
                r = base_r * math.cos(angle)

            z = plate_z + roof_height * math.sin(angle)

            ring_starts.append(len(extra_verts))
            for seg in range(n):
                theta = 2 * math.pi * seg / n
                extra_verts.append((cx + r * math.cos(theta),
                                    cy + r * math.sin(theta),
                                    z))

        # Apex
        apex_idx = len(extra_verts)
        extra_verts.append((cx, cy, plate_z + roof_height))

        extra_faces: list[tuple] = []

        # Top ring → ring 1
        rs0 = n + ring_starts[0]
        for seg in range(n):
            ns = (seg + 1) % n
            extra_faces.append((seg, ns, rs0 + ns))
            extra_faces.append((seg, rs0 + ns, rs0 + seg))

        # Successive rings
        for k in range(DOME_RINGS - 1):
            ba = n + ring_starts[k]
            bb = n + ring_starts[k + 1]
            for seg in range(n):
                ns = (seg + 1) % n
                extra_faces.append((ba + seg, ba + ns, bb + ns))
                extra_faces.append((ba + seg, bb + ns, bb + seg))

        # Last ring → apex
        last = n + ring_starts[-1]
        ac   = n + apex_idx
        for seg in range(n):
            ns = (seg + 1) % n
            extra_faces.append((last + seg, last + ns, ac))

        return [plate_z] * n, extra_verts, extra_faces

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slope_vector(top_xy, tags) -> tuple[float, float]:
        """
        Unit-ish vector in the downslope direction used by skillion and
        (rotated 90°) by ridged roofs.

        roof:direction = compass bearing of downslope (0 = N, 90 = E).
        Falls back to the polygon's shorter axis.
        """
        direction = tags.get("roof:direction")
        if direction is not None:
            try:
                bearing = float(direction)
                rad = math.radians(bearing)
                return math.sin(rad), math.cos(rad)
            except (ValueError, TypeError):
                pass

        xs = [p[0] for p in top_xy]
        ys = [p[1] for p in top_xy]
        if (max(xs) - min(xs)) >= (max(ys) - min(ys)):
            return 0.0, 1.0   # wider than tall → slope N-S
        return 1.0, 0.0       # taller than wide → slope E-W

    @staticmethod
    def _apply_shape(nodes: list[dict], tags: dict) -> list[dict]:
        """
        Replace the footprint nodes for special building:shape values.

        cylinder – If the polygon has fewer than 16 nodes (not already a
                   traced circle), replace with a 32-point mathematically
                   correct circle derived from the bounding-box centre and
                   radius, corrected for the cos(lat) Mercator scale factor.
        """
        shape = (tags.get("building:shape") or tags.get("shape") or "").lower()
        if shape != "cylinder":
            return nodes
        if len(nodes) >= 16:
            return nodes   # already a good circle approximation

        lons = [nd["lon"] for nd in nodes]
        lats = [nd["lat"] for nd in nodes]
        cx   = (min(lons) + max(lons)) / 2
        cy   = (min(lats) + max(lats)) / 2

        lat_rad = math.radians(cy)
        mpd_lat = 111_320.0
        mpd_lon = 111_320.0 * math.cos(lat_rad)
        r_m = min(
            (max(lons) - min(lons)) / 2 * mpd_lon,
            (max(lats) - min(lats)) / 2 * mpd_lat,
        )
        if r_m < 0.1:
            return nodes

        r_lon = r_m / mpd_lon
        r_lat = r_m / mpd_lat
        return [
            {
                "lon": cx + r_lon * math.cos(2 * math.pi * i / CYLINDER_SEGS),
                "lat": cy + r_lat * math.sin(2 * math.pi * i / CYLINDER_SEGS),
            }
            for i in range(CYLINDER_SEGS)
        ]

    @staticmethod
    def _clean_polygon(
        xy_verts: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """
        Remove degenerate vertices before triangulation:
          - duplicate / nearly-duplicate adjacent points (< 1e-7 deg apart)
          - collinear points (cross-product < 1e-10)
        Returns a cleaned polygon with at least 3 vertices, or [] if invalid.
        """
        pts = list(xy_verts)
        # Remove near-duplicates
        cleaned: list[tuple[float, float]] = []
        for p in pts:
            if not cleaned or (abs(p[0] - cleaned[-1][0]) > 1e-7
                               or abs(p[1] - cleaned[-1][1]) > 1e-7):
                cleaned.append(p)
        # Also check first-last wrap
        if (len(cleaned) > 1
                and abs(cleaned[0][0] - cleaned[-1][0]) < 1e-7
                and abs(cleaned[0][1] - cleaned[-1][1]) < 1e-7):
            cleaned.pop()

        if len(cleaned) < 3:
            return []

        # Remove collinear points (|cross| < threshold relative to edge lengths)
        result: list[tuple[float, float]] = []
        n = len(cleaned)
        for i in range(n):
            a = cleaned[(i - 1) % n]
            b = cleaned[i]
            c = cleaned[(i + 1) % n]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if abs(cross) > 1e-10:
                result.append(b)

        return result if len(result) >= 3 else []

    @staticmethod
    def _ensure_ccw(
        pts: list[tuple[float, float]],
        idx: list[int],
    ) -> tuple[list[tuple[float, float]], list[int]]:
        """
        Return (pts, idx) with CCW orientation guaranteed.
        OSM ways can be traced in either direction; all triangulators require CCW.
        """
        n = len(pts)
        area2 = sum(
            pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
            for i in range(n)
        )
        if area2 < 0:           # CW → reverse
            return pts[::-1], idx[::-1]
        return pts, idx

    @staticmethod
    def _triangulate_polygon(
        xy_verts: list[tuple[float, float]],
    ) -> list[tuple[int, int, int]]:
        """
        Constrained Delaunay triangulation of a flat polygon.

        Steps:
          1. Clean input  (duplicate / collinear vertices removed)
          2. Normalise to CCW  (OSM polygons can be either orientation)
          3. CDT via triangle library
          4. Ear-clipping fallback  (O(n²), correct for any simple polygon)

        Never falls back to fan triangulation (incorrect for concave shapes).
        """
        cleaned = BuildingExtruder._clean_polygon(xy_verts)
        if len(cleaned) < 3:
            return []

        n = len(cleaned)
        # Remap cleaned indices back to original indices.
        orig_idx = []
        j = 0
        for i, p in enumerate(xy_verts):
            if j < len(cleaned) and cleaned[j] == p:
                orig_idx.append(i)
                j += 1
        if len(orig_idx) != n:
            orig_idx = list(range(n))
            cleaned  = list(xy_verts[:n])

        # Normalise to CCW – required by both CDT and ear-clipping.
        cleaned, orig_idx = BuildingExtruder._ensure_ccw(cleaned, orig_idx)

        if n == 3:
            return [(orig_idx[0], orig_idx[1], orig_idx[2])]

        # --- Constrained Delaunay Triangulation (preferred) ---------------
        try:
            verts_2d = [[float(x), float(y)] for x, y in cleaned]
            segs     = [[i, (i + 1) % n] for i in range(n)]
            result   = tr.triangulate({"vertices": verts_2d, "segments": segs}, "p")
            out_verts = result.get("vertices", verts_2d)
            if len(out_verts) == n:   # no Steiner points added
                return [
                    (orig_idx[int(t[0])], orig_idx[int(t[1])], orig_idx[int(t[2])])
                    for t in result.get("triangles", [])
                ]
        except Exception:
            pass

        # --- Ear-clipping fallback (O(n²), correct for any simple polygon) -
        # Input is already CCW (normalised above).
        return BuildingExtruder._ear_clip(cleaned, orig_idx)

    @staticmethod
    def _ear_clip(
        pts: list[tuple[float, float]],
        orig_idx: list[int],
    ) -> list[tuple[int, int, int]]:
        """
        Simple O(n²) ear-clipping triangulation.
        Works correctly for convex AND concave simple polygons.
        """
        n = len(pts)
        if n < 3:
            return []
        if n == 3:
            return [(orig_idx[0], orig_idx[1], orig_idx[2])]

        remaining = list(range(n))   # indices into pts / orig_idx
        tris: list[tuple[int, int, int]] = []

        def _cross(o, a, b):
            return ((a[0] - o[0]) * (b[1] - o[1])
                    - (a[1] - o[1]) * (b[0] - o[0]))

        def _point_in_triangle(p, a, b, c):
            d1 = _cross(a, b, p)
            d2 = _cross(b, c, p)
            d3 = _cross(c, a, p)
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            return not (has_neg and has_pos)

        def _is_ear(rem, i):
            m  = len(rem)
            pi = rem[(i - 1) % m]
            ci = rem[i]
            ni = rem[(i + 1) % m]
            a, b, c = pts[pi], pts[ci], pts[ni]
            if _cross(a, b, c) <= 0:   # reflex vertex
                return False
            for j, r in enumerate(rem):
                if r in (pi, ci, ni):
                    continue
                if _point_in_triangle(pts[r], a, b, c):
                    return False
            return True

        max_iter = n * n + n   # safety limit
        iters    = 0
        while len(remaining) > 3 and iters < max_iter:
            iters += 1
            m = len(remaining)
            ear_found = False
            for i in range(m):
                if _is_ear(remaining, i):
                    pi = remaining[(i - 1) % m]
                    ci = remaining[i]
                    ni = remaining[(i + 1) % m]
                    tris.append((orig_idx[pi], orig_idx[ci], orig_idx[ni]))
                    remaining.pop(i)
                    ear_found = True
                    break
            if not ear_found:
                break   # degenerate polygon, stop

        if len(remaining) == 3:
            tris.append((orig_idx[remaining[0]],
                         orig_idx[remaining[1]],
                         orig_idx[remaining[2]]))
        return tris

    # ------------------------------------------------------------------
    # Height helpers  (public so tests can call them directly)
    # ------------------------------------------------------------------

    @staticmethod
    def _wall_height(tags: dict) -> float:
        """
        Height of the visible wall section.  Matches OSMBuildings getDimensions():

          explicit height tag  →  wall = height − min_height − roofHeight
              (height is the absolute maximum including roof)

          building:levels only →  wall = levels × 3 − min_height
          no height info       →  wall = DEFAULT_HEIGHT − min_height
              (roof sits on TOP of the levels/default height, not deducted from it)
        """
        min_h = BuildingExtruder._parse_min_height(tags)

        if "height" in tags:
            try:
                total = float(tags["height"].replace(",", ".").split()[0])
            except (ValueError, IndexError):
                total = DEFAULT_BUILDING_HEIGHT
            roof = BuildingExtruder._roof_height(tags, total)
            return max(0.0, total - min_h - roof)

        # No explicit height: levels × 3 (or default) gives the WALL height.
        # Roof height is additional on top — it is NOT deducted from wall height.
        if "building:levels" in tags:
            try:
                wall = float(tags["building:levels"]) * LEVEL_HEIGHT
            except ValueError:
                wall = DEFAULT_BUILDING_HEIGHT
        else:
            wall = DEFAULT_BUILDING_HEIGHT

        return max(0.0, wall - min_h)

    @staticmethod
    def _total_height(tags: dict) -> float:
        """Parse total building height from height / building:levels tags."""
        if "height" in tags:
            try:
                return float(tags["height"].replace(",", ".").split()[0])
            except (ValueError, IndexError):
                pass
        if "building:levels" in tags:
            try:
                return float(tags["building:levels"]) * LEVEL_HEIGHT
            except ValueError:
                pass
        return DEFAULT_BUILDING_HEIGHT

    @staticmethod
    def _roof_height(tags: dict, total_height: float) -> float:
        """
        Parse the roof section height; clamped to total_height.

        Priority:
          1. roof:height (metres)
          2. roof:levels × 3.0 m
          3. roof:angle + half OBBox short dimension  (not available here; handled elsewhere)
          4. S3DB default: 3.0 m for all non-flat shapes (spec §3.4 and OSMBuildings).
        """
        if "roof:height" in tags:
            try:
                rh = float(tags["roof:height"].replace(",", ".").split()[0])
                return min(rh, total_height)
            except (ValueError, IndexError):
                pass
        if "roof:levels" in tags:
            try:
                rh = float(tags["roof:levels"]) * LEVEL_HEIGHT
                return min(rh, total_height)
            except ValueError:
                pass

        # S3DB spec default: non-flat shapes use 1 level (3.0 m).
        # Flat or unrecognised shapes use 0.
        shape = (
            tags.get("roof:shape")
            or tags.get("building:roof:shape")
            or "flat"
        ).lower()
        if shape not in ("flat", "no", "none", ""):
            return min(LEVEL_HEIGHT, total_height)
        return 0.0

    @staticmethod
    def _parse_min_height(tags: dict) -> float:
        """Parse S3DB base elevation offset (min_height / building:min_level)."""
        if "min_height" in tags:
            try:
                return float(tags["min_height"].replace(",", ".").split()[0])
            except (ValueError, IndexError):
                pass
        if "building:min_level" in tags:
            try:
                return float(tags["building:min_level"]) * LEVEL_HEIGHT
            except ValueError:
                pass
        return 0.0

    # Legacy alias (kept for backward-compatible tests)
    @staticmethod
    def _building_height(tags: dict) -> float:
        return BuildingExtruder._total_height(tags)
