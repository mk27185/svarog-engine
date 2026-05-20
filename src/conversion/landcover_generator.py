"""
Landcover overlay texture (RGBA PNG) for terrain shader.

Channels (tile UV space, same as SDFGenerator):
  R – water bodies (polygons)
  G – rivers / waterways (line mask)
  B – green areas (parks, forest, grass, …)
  A – railways (line mask)
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from src.conversion.sdf_generator import SDFGenerator

DEFAULT_RESOLUTION = 1024

_RAILWAY_EXCLUDE = frozenset({"abandoned", "disused", "razed", "construction"})


class LandcoverGenerator:
    """Rasterise OSM landcover features into a multi-channel PNG."""

    def __init__(self, resolution: int = DEFAULT_RESOLUTION):
        self.resolution = resolution

    def generate(
        self,
        features: dict[str, list],
        meta: dict,
        output_path: str,
    ) -> str:
        """
        features keys: water_polygons, waterways, green_polygons, railways
        """
        res = self.resolution
        ch = {
            "water": Image.new("F", (res, res), 0.0),
            "river": Image.new("F", (res, res), 0.0),
            "green": Image.new("F", (res, res), 0.0),
            "rail":  Image.new("F", (res, res), 0.0),
        }
        draws = {k: ImageDraw.Draw(v) for k, v in ch.items()}

        tw = meta["total_width_m"]
        th = meta["total_height_m"]
        dim = max(tw, th)

        def _draw_poly(draw: ImageDraw.ImageDraw, nodes: list, fill: float = 1.0) -> None:
            pts = self._nodes_to_pixels(nodes, meta, res)
            if len(pts) < 3:
                return
            draw.polygon(pts, fill=fill)

        def _draw_line(draw: ImageDraw.ImageDraw, nodes: list, width_m: float) -> None:
            local = self._nodes_to_local(nodes, meta)
            if len(local) < 2:
                return
            segs = SDFGenerator._clip_polyline_to_bbox(local, 0.0, 0.0, tw, th)
            stroke = max(2, int(width_m / dim * res))
            for seg in segs:
                pix = [
                    (float(u) * res, float(v) * res)
                    for u, v in SDFGenerator._to_pixel_uv(seg, meta)
                ]
                if len(pix) >= 2:
                    draw.line(pix, width=stroke, fill=1.0)

        for poly in features.get("water_polygons", []):
            nodes = poly.get("nodes", poly) if isinstance(poly, dict) else poly
            _draw_poly(draws["water"], nodes)

        for way in features.get("waterways", []):
            nodes = way.get("nodes", []) if isinstance(way, dict) else way
            tags = way.get("tags", {}) if isinstance(way, dict) else {}
            w = 12.0 if tags.get("waterway") in ("river", "canal") else 6.0
            _draw_line(draws["river"], nodes, w)

        for poly in features.get("green_polygons", []):
            nodes = poly.get("nodes", poly) if isinstance(poly, dict) else poly
            _draw_poly(draws["green"], nodes)

        for way in features.get("railways", []):
            nodes = way.get("nodes", []) if isinstance(way, dict) else way
            tags = way.get("tags", {}) if isinstance(way, dict) else {}
            if tags.get("railway") in _RAILWAY_EXCLUDE:
                continue
            _draw_line(draws["rail"], nodes, 4.0)

        out = np.zeros((res, res, 4), dtype=np.float32)
        out[:, :, 0] = np.asarray(ch["water"], dtype=np.float32)
        out[:, :, 1] = np.asarray(ch["river"], dtype=np.float32)
        out[:, :, 2] = np.asarray(ch["green"], dtype=np.float32)
        out[:, :, 3] = np.asarray(ch["rail"], dtype=np.float32)

        SDFGenerator._save_png(out, output_path)
        print(f"   Landcover texture: {output_path}  ({res}x{res} px)")
        return output_path

    @staticmethod
    def _nodes_to_local(nodes: list, meta: dict) -> list[tuple[float, float]]:
        from src.conversion.geo_utils import to_local

        if not nodes:
            return []
        if isinstance(nodes[0], dict):
            return [to_local(n["lon"], n["lat"], meta) for n in nodes]
        return [to_local(lon, lat, meta) for lon, lat in nodes]

    def _nodes_to_pixels(
        self, nodes: list, meta: dict, res: int,
    ) -> list[tuple[float, float]]:
        local = self._nodes_to_local(nodes, meta)
        if len(local) < 3:
            return []
        return [
            (u * res, v * res)
            for u, v in SDFGenerator._to_pixel_uv(local, meta)
        ]
