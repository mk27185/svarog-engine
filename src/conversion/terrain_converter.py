import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os
import math

class TerrainConverter:

    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir

    @staticmethod
    def interpolate_corners(elevation_array):
        h, w = elevation_array.shape
        # body s realnymi hodnotami v centrech pixelu: (0.5, 0.5), ..., (h-0.5, w-0.5)
        src_r = np.arange(0.5, h, 1)
        src_c = np.arange(0.5, w, 1)

        interp = RegularGridInterpolator((src_r, src_c), elevation_array, method='linear', bounds_error=False, fill_value=None)

        r_corners = np.arange(0, h + 1)
        c_corners = np.arange(0, w + 1)
        rr, cc = np.meshgrid(r_corners, c_corners, indexing='ij')
        pts = np.stack([rr.ravel(), cc.ravel()], axis=-1)

        z_corners = interp(pts).reshape(h + 1, w + 1)
        return z_corners

    def convert_tif_to_obj(self, tif_path, obj_name=None, output_dir=None):
        output_dir = output_dir or self.output_dir
        if not os.path.exists(tif_path):
            raise FileNotFoundError(f"TIF soubor neexistuje: {tif_path}")
        os.makedirs(output_dir, exist_ok=True)

        import rasterio
        with rasterio.open(tif_path) as src:
            elevation = src.read(1).astype(np.float32)
            transform = src.transform
            h, w = elevation.shape
            bounds = src.bounds

        # Převod na lokální metrické souřadnice (počátek = jihozápadní roh bbox)
        # 1° zeměpisné šířky ≈ 111 320 m; 1° délky závisí na šířce
        lat_origin = bounds.bottom
        lon_origin = bounds.left
        lat_mean_rad = math.radians((bounds.bottom + bounds.top) / 2)
        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = 111_320.0 * math.cos(lat_mean_rad)

        z_grid = TerrainConverter.interpolate_corners(elevation)

        output_file = (
            os.path.join(output_dir, f"{obj_name}.obj")
            if obj_name
            else os.path.join(output_dir, os.path.splitext(os.path.basename(tif_path))[0] + ".obj")
        )

        with open(output_file, 'w') as f:
            f.write("# OBJ Mesh – local metric coords (origin = SW corner of bbox), units: meters\n")
            f.write(f"# Origin: lon={lon_origin:.6f}, lat={lat_origin:.6f}\n")
            for r in range(h + 1):
                for c in range(w + 1):
                    lon, lat = transform * (c, r)
                    x = (lon - lon_origin) * meters_per_deg_lon
                    y = (lat - lat_origin) * meters_per_deg_lat
                    z = float(z_grid[r, c])
                    f.write(f"v {x:.3f} {y:.3f} {z:.3f}\n")

            for r in range(h):
                for c in range(w):
                    v1 = r * (w + 1) + c + 1
                    v2 = v1 + 1
                    v3 = v1 + (w + 1)
                    v4 = v3 + 1
                    f.write(f"f {v1} {v2} {v3}\n")
                    f.write(f"f {v2} {v4} {v3}\n")

        return output_file