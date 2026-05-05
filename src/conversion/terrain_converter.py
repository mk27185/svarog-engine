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

    def apply_road_burn(
        self,
        z_grid:    np.ndarray,
        meta:      dict,
        road_poly,
        depth:     float = 0.08,
    ) -> np.ndarray:
        """
        Sníží Z hodnoty terrain gridu uvnitř road_poly o `depth` metrů.

        Silnice tak "sedí v krajině" — kopíruje každý svah a kopec,
        ale je mírně zapuštěna (typicky 0.05–0.10 m).

        Vrátí nový z_grid (float32, stejný shape).  Vstupní z_grid není změněn.
        """
        from shapely import contains_xy

        h, w  = meta["h"], meta["w"]
        cw    = meta["cell_width_m"]
        ch    = meta["cell_height_m"]

        # Souřadnice každého vrcholu terrain gridu v lokálním metrickém systému
        x_arr = np.arange(w + 1, dtype=np.float64) * cw
        y_arr = (h - np.arange(h + 1, dtype=np.float64)) * ch
        XX, YY = np.meshgrid(x_arr, y_arr)               # (h+1, w+1)

        inside = contains_xy(
            road_poly, XX.ravel(), YY.ravel()
        ).reshape(h + 1, w + 1)

        burned       = z_grid.copy()
        burned[inside] -= np.float32(depth)

        n = int(inside.sum())
        print(f"   Z-burn: {n:,} vrcholů o {depth*100:.0f} cm zahloubeno do silnice")
        return burned

    def write_obj(
        self,
        z_grid:      np.ndarray,
        meta:        dict,
        obj_name:    str | None = None,
        output_dir:  str | None = None,
        texture_file: str | None = None,
    ) -> str:
        """
        Zapíše terrain OBJ.

        Pokud je zadán texture_file (cesta k PNG):
          - přidá UV souřadnice (vt) pro každý vrchol
          - vygeneruje MTL soubor
          - faces mají formát v/vt

        UV mapování:
          U = x / total_width_m   (0 = západ → 1 = východ)
          V = y / total_height_m  (0 = jih  → 1 = sever)
          (OBJ konvence: V=0 dole = jih, V=1 nahoře = sever)
        """
        output_dir  = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        h, w       = meta["h"], meta["w"]
        transform  = meta["transform"]
        lon_origin = meta["lon_origin"]
        lat_origin = meta["lat_origin"]
        mpd_lon    = meta["meters_per_deg_lon"]
        mpd_lat    = meta["meters_per_deg_lat"]
        total_w    = meta["total_width_m"]
        total_h    = meta["total_height_m"]

        use_uv      = texture_file is not None
        fname       = f"{obj_name}.obj" if obj_name else "terrain.obj"
        output_file = os.path.join(output_dir, fname)

        # ── MTL soubor ─────────────────────────────────────────────────────
        if use_uv:
            mtl_name = fname.replace(".obj", ".mtl")
            mtl_path = os.path.join(output_dir, mtl_name)
            tex_basename = os.path.basename(texture_file)
            with open(mtl_path, "w") as m:
                m.write(f"# Terrain material\nnewmtl terrain\n")
                m.write(f"Ka 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n")
                m.write(f"map_Kd {tex_basename}\n")

        with open(output_file, "w") as f:
            f.write("# Terrain OBJ – local metric coords (origin = SW corner)\n")
            f.write(
                f"# Origin: lon={lon_origin:.6f}, lat={lat_origin:.6f}  "
                f"grid={h}×{w}  cell={meta['cell_width_m']:.2f}×"
                f"{meta['cell_height_m']:.2f} m\n"
            )
            if use_uv:
                f.write(f"mtllib {mtl_name}\n")

            # ── Vrcholy (v) ────────────────────────────────────────────────
            for r in range(h + 1):
                for c in range(w + 1):
                    lon, lat = transform * (c, r)
                    x = (lon - lon_origin) * mpd_lon
                    y = (lat - lat_origin) * mpd_lat
                    f.write(f"v {x:.3f} {y:.3f} {float(z_grid[r,c]):.3f}\n")

            # ── UV souřadnice (vt) ─────────────────────────────────────────
            if use_uv:
                for r in range(h + 1):
                    for c in range(w + 1):
                        lon, lat = transform * (c, r)
                        x = (lon - lon_origin) * mpd_lon
                        y = (lat - lat_origin) * mpd_lat
                        u = x / total_w
                        v = y / total_h
                        f.write(f"vt {u:.6f} {v:.6f}\n")
                f.write("usemtl terrain\n")

            # ── Faces ──────────────────────────────────────────────────────
            for r in range(h):
                for c in range(w):
                    v1 = r * (w + 1) + c + 1
                    v2 = v1 + 1
                    v3 = v1 + (w + 1)
                    v4 = v3 + 1
                    if use_uv:
                        f.write(
                            f"f {v1}/{v1} {v2}/{v2} {v3}/{v3}\n"
                            f"f {v2}/{v2} {v4}/{v4} {v3}/{v3}\n"
                        )
                    else:
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
