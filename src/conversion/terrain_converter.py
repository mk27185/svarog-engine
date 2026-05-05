import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom as ndimage_zoom
import os

from .geo_utils import build_meta


class TerrainConverter:

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir

    def build_grid(
        self,
        tif_path: str,
        upsample_factor: int = 1,
    ) -> tuple[np.ndarray, dict]:
        """
        Read GeoTIFF → (z_grid, meta).

        upsample_factor : integer ≥ 1.
            Uses bilinear interpolation (order=1) — no overshoot artifacts.
            SRTM is 30 m; factor=10 → ~2–3 m, factor=20 → ~1–1.5 m.
            Above 20 adds no real elevation information from SRTM.

        z_grid : (h+1, w+1) float32, corner-interpolated elevation
                 row 0 = north edge, row h = south edge
        meta   : shared geometry metadata (see geo_utils.build_meta)
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

        meta   = build_meta(bounds, transform)
        z_grid = self._interpolate_corners(elevation)

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
        output_dir  = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        h, w       = meta["h"], meta["w"]
        transform  = meta["transform"]
        lon_origin = meta["lon_origin"]
        lat_origin = meta["lat_origin"]
        mpd_lon    = meta["meters_per_deg_lon"]
        mpd_lat    = meta["meters_per_deg_lat"]

        fname       = f"{obj_name}.obj" if obj_name else "terrain.obj"
        output_file = os.path.join(output_dir, fname)

        with open(output_file, "w") as f:
            f.write("# Terrain OBJ – local metric coords (origin = SW corner)\n")
            f.write(
                f"# Origin: lon={lon_origin:.6f}, lat={lat_origin:.6f}  "
                f"grid={h}×{w}  cell={meta['cell_width_m']:.2f}×"
                f"{meta['cell_height_m']:.2f} m\n"
            )
            for r in range(h + 1):
                for c in range(w + 1):
                    lon, lat = transform * (c, r)
                    x = (lon - lon_origin) * mpd_lon
                    y = (lat - lat_origin) * mpd_lat
                    f.write(f"v {x:.3f} {y:.3f} {float(z_grid[r,c]):.3f}\n")

            for r in range(h):
                for c in range(w):
                    v1 = r * (w + 1) + c + 1
                    v2 = v1 + 1
                    v3 = v1 + (w + 1)
                    v4 = v3 + 1
                    f.write(f"f {v1} {v2} {v3}\nf {v2} {v4} {v3}\n")

        total = h * w * 2
        print(f"   Terrain OBJ: {output_file}  ({(h+1)*(w+1):,} vrcholů, {total:,} faces)")
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
