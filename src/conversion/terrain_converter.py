import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom as ndimage_zoom
import os

from .geo_utils import build_meta


class TerrainConverter:

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir

    def build_grid_seamless(
        self,
        bbox_orig:       tuple,
        union_dem_path:  str,
        upsample_factor: int   = 1,
    ) -> tuple[np.ndarray, dict]:
        """
        Sample terrain at a regular vertex grid derived directly from the tile's
        exact slippy-map boundaries, interpolating from the union DEM cache.

        Because the vertex positions are computed from the tile math rather than
        from the DEM pixel grid, two adjacent tiles always define their shared
        boundary at the EXACT same geographic coordinates.  Sampling the same
        DEM at those coordinates yields identical elevation values → zero seam.

        Vertex grid size is inferred from the DEM resolution times upsample_factor.
        """
        import rasterio  # noqa: PLC0415
        from rasterio.transform import from_bounds as tf_from_bounds  # noqa: PLC0415

        s_deg, w_deg, n_deg, e_deg = bbox_orig

        with rasterio.open(union_dem_path) as src:
            pixel_h_deg = abs(src.transform.e)
            pixel_w_deg = abs(src.transform.a)

            # Number of vertices: aligned to DEM resolution after upsampling.
            # Using round() then +1 ensures all tiles at the same zoom level
            # get the same vertex count (the DEM-pixel fraction is identical for
            # all slippy tiles at a given zoom).
            n_lat = round((n_deg - s_deg) / pixel_h_deg) * upsample_factor + 1
            n_lon = round((e_deg - w_deg) / pixel_w_deg) * upsample_factor + 1

            # Vertex positions: row 0 = north boundary, row n_lat-1 = south.
            lats = np.linspace(n_deg, s_deg, n_lat)   # descending (north first)
            lons = np.linspace(w_deg, e_deg, n_lon)   # ascending (west first)

            # Read a slightly padded window from the union DEM for interpolation.
            pad = 2  # extra DEM pixels on each side for boundary interpolation
            win = rasterio.windows.from_bounds(
                w_deg - pad * pixel_w_deg,
                s_deg - pad * pixel_h_deg,
                e_deg + pad * pixel_w_deg,
                n_deg + pad * pixel_h_deg,
                src.transform,
            )
            col0  = max(0, math.floor(win.col_off))
            row0  = max(0, math.floor(win.row_off))
            col1  = min(src.width,  math.ceil(win.col_off + win.width))
            row1  = min(src.height, math.ceil(win.row_off + win.height))
            read_win = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
            crop = src.read(1, window=read_win).astype(np.float32)
            ct   = rasterio.windows.transform(read_win, src.transform)

        # Pixel-centre coordinates of the crop (ascending lat for interpolator)
        crop_h, crop_w = crop.shape
        crop_lat_centers = ct.f - (np.arange(crop_h) + 0.5) * pixel_h_deg  # descending
        crop_lon_centers = ct.c + (np.arange(crop_w) + 0.5) * pixel_w_deg  # ascending

        # RegularGridInterpolator requires ascending first axis → flip lat
        interp = RegularGridInterpolator(
            (crop_lat_centers[::-1], crop_lon_centers),
            crop[::-1, :],
            method="linear", bounds_error=False, fill_value=None,
        )

        lon_grid, lat_grid = np.meshgrid(lons, lats)   # shape (n_lat, n_lon)
        pts    = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        z_grid = interp(pts).reshape(n_lat, n_lon).astype(np.float32)

        # Build meta from exact tile boundaries
        h_cells = n_lat - 1
        w_cells = n_lon - 1
        tile_transform = tf_from_bounds(w_deg, s_deg, e_deg, n_deg, w_cells, h_cells)

        class _B:  # lightweight bounds object
            def __init__(self, l, r, b, t):
                self.left = l; self.right = r; self.bottom = b; self.top = t

        meta = build_meta(_B(w_deg, e_deg, s_deg, n_deg), tile_transform)

        cw = meta["cell_width_m"]
        ch = meta["cell_height_m"]
        print(
            f"   Terrain grid (seamless): {h_cells}×{w_cells} buněk  "
            f"({cw:.1f}×{ch:.1f} m/buňka)  [upsample {upsample_factor}×, exact bbox]"
        )
        return z_grid, meta

    def build_grid(
        self,
        tif_path:        str,
        upsample_factor: int          = 1,
        bbox_orig:       tuple | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Read GeoTIFF → (z_grid, meta).

        upsample_factor : integer ≥ 1.
            Uses bilinear interpolation (order=1) — no overshoot artifacts.
            SRTM is 30 m; factor=10 → ~2–3 m, factor=20 → ~1–1.5 m.
            Above 20 adds no real elevation information from SRTM.

        bbox_orig : (south, west, north, east) of the *tile* (not the DEM file).
            When the DEM file is larger than the tile (buffered by get_dem's
            buffer_px), supply the original tile bbox so that build_grid can
            trim the z_grid to just the tile's vertices.  Border vertices are
            then computed by true bilinear interpolation from two real DEM pixels
            (not extrapolation), making adjacent tiles agree exactly on the
            shared boundary elevation → seamless terrain.

        z_grid : (h+1, w+1) float32, corner-interpolated elevation
                 row 0 = north edge, row h = south edge
        meta   : shared geometry metadata for the tile (see geo_utils.build_meta)
        """
        if not os.path.exists(tif_path):
            raise FileNotFoundError(f"TIF soubor neexistuje: {tif_path}")

        import rasterio
        from rasterio.transform import from_bounds as tf_from_bounds

        with rasterio.open(tif_path) as src:
            elevation = src.read(1).astype(np.float32)
            bounds    = src.bounds
            transform = src.transform

        if upsample_factor > 1:
            # order=1 = bilinear — guaranteed no overshoot / false valleys
            elevation = ndimage_zoom(
                elevation, upsample_factor, order=1, mode="reflect"
            )
            new_h, new_w = elevation.shape
            transform = tf_from_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top,
                new_w, new_h,
            )

        # ── compute full z_grid (may include buffer pixels) ───────────────────
        meta_full  = build_meta(bounds, transform)
        z_grid_full = self._interpolate_corners(elevation)

        # ── if bbox_orig provided, trim buffer pixels from z_grid ─────────────
        if bbox_orig is not None:
            s_orig, w_orig, n_orig, e_orig = bbox_orig
            pixel_w = abs(transform.a)
            pixel_h = abs(transform.e)   # transform.e is negative

            # How many pixels were added as buffer on each side?
            # The DEM file starts buffer_px pixels west/north of the tile.
            buf_left   = round((bounds.left   - w_orig) / pixel_w)   # west buffer
            buf_top    = round((n_orig - bounds.top  ) / pixel_h)    # north buffer (rows from top)
            buf_right  = round((e_orig - bounds.right) / pixel_w)    # east buffer
            buf_bottom = round((bounds.bottom - s_orig) / pixel_h)   # south buffer

            # For the common 1-pixel-per-side buffer: buf_* ≈ -1 (DEM extends past tile)
            P_left   = max(0, -buf_left)
            P_top    = max(0, -buf_top)
            P_right  = max(0, -buf_right)
            P_bottom = max(0, -buf_bottom)

            h_full, w_full = meta_full["h"], meta_full["w"]
            # Dimensions of the original (un-buffered) tile in pixels
            h_orig_px = h_full - P_top  - P_bottom
            w_orig_px = w_full - P_left - P_right

            if P_top > 0 or P_left > 0:
                # Slice the interior vertices (trim buffered edge rows/cols)
                r0 = P_top;   r1 = P_top  + h_orig_px + 1   # +1 for corner vertex grid
                c0 = P_left;  c1 = P_left + w_orig_px + 1
                z_grid = z_grid_full[r0:r1, c0:c1].copy()

                # Build meta for the *tile* bounds (not the buffered DEM)
                tile_bounds_obj = type("B", (), {
                    "left":   bounds.left   + P_left   * pixel_w,
                    "right":  bounds.right  - P_right  * pixel_w,
                    "bottom": bounds.bottom + P_bottom * pixel_h,
                    "top":    bounds.top    - P_top    * pixel_h,
                })()
                tile_transform = tf_from_bounds(
                    tile_bounds_obj.left,   tile_bounds_obj.bottom,
                    tile_bounds_obj.right,  tile_bounds_obj.top,
                    w_orig_px, h_orig_px,
                )
                meta = build_meta(tile_bounds_obj, tile_transform)
            else:
                z_grid = z_grid_full
                meta   = meta_full
        else:
            z_grid = z_grid_full
            meta   = meta_full

        cw = meta["cell_width_m"]
        ch = meta["cell_height_m"]
        print(
            f"   Terrain grid: {meta['h']}×{meta['w']} buněk  "
            f"({cw:.1f}×{ch:.1f} m/buňka)  "
            f"[upsample {upsample_factor}×]"
        )
        return z_grid, meta

    def write_obj(
        self,
        z_grid:     np.ndarray,
        meta:       dict,
        obj_name:   str | None = None,
        output_dir: str | None = None,
    ) -> str:
        """Write plain terrain OBJ (regular quad grid, no road modification)."""
        return self._write_obj_impl(
            z_grid, meta, obj_name, output_dir, with_uv=False
        )

    def write_obj_with_uv(
        self,
        z_grid:     np.ndarray,
        meta:       dict,
        obj_name:   str | None = None,
        output_dir: str | None = None,
    ) -> str:
        """Write terrain OBJ with UV texture coordinates (for SDF overlay)."""
        return self._write_obj_impl(
            z_grid, meta, obj_name, output_dir, with_uv=True
        )

    def _write_obj_impl(
        self,
        z_grid:     np.ndarray,
        meta:       dict,
        obj_name:   str | None = None,
        output_dir: str | None = None,
        with_uv:    bool = False,
    ) -> str:
        """Shared OBJ writer (optionally emits VT lines + indexed faces)."""
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        h, w       = meta["h"], meta["w"]
        transform  = meta["transform"]
        bounds     = meta.get("bounds")
        lon_origin = meta.get("lon_origin", bounds.left if bounds else 0.0)
        lat_origin = meta.get("lat_origin", bounds.bottom if bounds else 0.0)
        mpd_lon    = meta["meters_per_deg_lon"]
        mpd_lat    = meta["meters_per_deg_lat"]
        total_w    = meta["total_width_m"]
        total_h    = meta["total_height_m"]

        fname       = f"{obj_name}.obj" if obj_name else "terrain.obj"
        output_file = os.path.join(output_dir, fname)

        with open(output_file, "w") as f:
            f.write(
                "# Terrain OBJ - local metric coords (origin = SW corner)\n"
            )
            f.write(
                f"# Origin: lon={lon_origin:.6f}, lat={lat_origin:.6f}  "
                f"grid={h}x{w}  cell={meta['cell_width_m']:.2f}x"
                f"{meta['cell_height_m']:.2f} m\n"
            )

            for r in range(h + 1):
                for c in range(w + 1):
                    lon, lat = transform * (c, r)
                    x = (lon - lon_origin) * mpd_lon
                    y = (lat - lat_origin) * mpd_lat
                    elev = float(z_grid[r, c])
                    f.write(f"v {x:.3f} {y:.3f} {elev:.3f}\n")

                    if with_uv:
                        uv_x = c / w if w > 0 else 0.0
                        uv_y = 1.0 - (r / h if h > 0 else 0.0)
                        f.write(f"vt {uv_x:.4f} {uv_y:.4f}\n")

            for r in range(h):
                for c in range(w):
                    vi = r * (w + 1) + c
                    v1 = vi + 1
                    v2 = v1 + 1
                    v3 = v1 + (w + 1)
                    v4 = v3 + 1

                    if with_uv:
                        f.write(f"f {v1}/{v1} {v2}/{v2} {v3}/{v3}\n")
                        f.write(f"f {v2}/{v2} {v4}/{v4} {v3}/{v3}\n")
                    else:
                        f.write(f"f {v1} {v2} {v3}\n")
                        f.write(f"f {v2} {v4} {v3}\n")

        total = h * w * 2
        uv_tag = " (uv)" if with_uv else ""
        print(
            f"   Terrain OBJ{uv_tag}: {output_file}  "
            f"({(h + 1) * (w + 1):,} vertices, {total:,} faces)"
        )
        return output_file

    def convert_tif_to_obj(
        self,
        tif_path:        str,
        obj_name:        str | None = None,
        output_dir:      str | None = None,
        upsample_factor: int = 1,
    ) -> str:
        z_grid, meta = self.build_grid(tif_path, upsample_factor=upsample_factor)
        return self.write_obj(z_grid, meta, obj_name, output_dir)

    @staticmethod
    def _interpolate_corners(elevation: np.ndarray) -> np.ndarray:
        h, w   = elevation.shape
        src_r  = np.arange(0.5, h, 1.0)
        src_c  = np.arange(0.5, w, 1.0)
        interp = RegularGridInterpolator(
            (src_r, src_c), elevation,
            method="linear", bounds_error=False, fill_value=None,
        )
        rr, cc = np.meshgrid(np.arange(h + 1), np.arange(w + 1), indexing="ij")
        pts    = np.stack([rr.ravel(), cc.ravel()], axis=-1)
        return interp(pts).reshape(h + 1, w + 1).astype(np.float32)
