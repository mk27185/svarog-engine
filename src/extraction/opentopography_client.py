from __future__ import annotations

import os
from pathlib import Path

import requests
import rasterio
import rasterio.mask
import rasterio.windows
from shapely.geometry import box
from dotenv import load_dotenv

load_dotenv()


class OpenTopographyClient:
    """
    Downloads SRTM DEM from OpenTopography.

    For batch tile generation, call prefetch_dem(union_bbox) once before the
    tile loop.  Subsequent get_dem() calls with sub-bboxes will be served from
    the cached file (a rasterio window crop) — no extra API requests.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENTOPOGRAPHY_API_KEY")
        if not self.api_key:
            raise ValueError("OpenTopography API key not found in .env")
        self.base_url = "https://portal.opentopography.org"

        self._cache_path: str | None = None
        self._cache_bbox: tuple | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def prefetch_dem(self, bbox: tuple, output_dir: str = "data/downloads") -> str:
        """
        Download DEM for *bbox* and store it as the local cache.
        All subsequent get_dem() calls that fall within *bbox* will crop from
        this file instead of making new API requests.

        Returns the path to the downloaded TIF.
        """
        path = self._download(bbox, output_dir)
        self._cache_path = path
        self._cache_bbox = bbox
        return path

    def get_dem(self, bbox: tuple, output_dir: str = "data/downloads") -> str | None:
        """
        Return a TIF file for *bbox*.

        If a prefetched cache covers *bbox*, crops from it (no network request).
        Otherwise downloads from OpenTopography directly.
        """
        if self._cache_path and self._bbox_covered(bbox):
            try:
                return self._crop_from_cache(bbox, output_dir)
            except Exception as e:
                print(f"  Cache crop failed ({e}), falling back to download")

        return self._download(bbox, output_dir)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _bbox_covered(self, bbox: tuple) -> bool:
        s, w, n, e = bbox
        cs, cw, cn, ce = self._cache_bbox
        return s >= cs and w >= cw and n <= cn and e <= ce

    def _crop_from_cache(self, bbox: tuple, output_dir: str) -> str:
        s, w, n, e = bbox
        fname = f"dem_{s:.6f}_{w:.6f}_{n:.6f}_{e:.6f}.tif"
        out_path = os.path.join(output_dir, fname)
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(out_path):
            return out_path

        with rasterio.open(self._cache_path) as src:
            geom = box(w, s, e, n)
            out_image, out_transform = rasterio.mask.mask(
                src, [geom], crop=True, all_touched=True
            )
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width":  out_image.shape[2],
                "transform": out_transform,
            })
            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(out_image)

        return out_path

    def _download(self, bbox: tuple, output_dir: str) -> str | None:
        s, w, n, e = bbox
        fname = f"dem_{s:.4f}_{w:.4f}_{n:.4f}_{e:.4f}.tif"
        file_path = os.path.join(output_dir, fname)
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(file_path):
            return file_path

        try:
            response = requests.get(
                f"{self.base_url}/API/globaldem",
                params={
                    "demtype":      "SRTMGL1",
                    "south":        s,
                    "north":        n,
                    "west":         w,
                    "east":         e,
                    "outputFormat": "GTiff",
                    "API_Key":      self.api_key,
                },
                stream=True,
                timeout=60,
            )
            response.raise_for_status()

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return file_path

        except requests.exceptions.RequestException as e:
            print(f"Error fetching DEM: {e}")
            return None
