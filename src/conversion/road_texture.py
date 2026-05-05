"""
Road Texture Generator
======================
Rasterizuje road-union polygon na PNG texturu pro terrain mesh.

Přístup "soft-edge road burn":
  1. Road polygon → binární maska (bílá = silnice, černá = terén)
  2. Gaussian blur na hranici → měkký přechod (žádný aliasing)
  3. Blend: terrain_color ←(blurred mask)→ road_color
  4. Výsledek: RGB PNG která se přiřadí jako materiál k terrain meshi

UV mapování (shodné s write_obj v terrain_converter.py):
  U = x_local / total_width_m   (0 = západ,  1 = východ)
  V = y_local / total_height_m  (0 = jih,    1 = sever)

PIL y-osa: 0 = nahoře (sever), size = dole (jih)  → py = (1 - V) * size
"""

import os
from typing import Sequence

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    raise ImportError("Chybí Pillow: pip install Pillow")

try:
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
except ImportError:
    raise ImportError("Chybí shapely: pip install shapely")


# ── default barvy (RGB) ────────────────────────────────────────────────────
# Lze přepsat přes config: ROAD_COLOR, TERRAIN_COLOR
DEFAULT_ROAD_COLOR    = (72,  72,  72)   # tmavě šedý asfalt
DEFAULT_TERRAIN_COLOR = (160, 148, 124)  # neutrální písčito-hnědá


def _iter_polygons(geom) -> list:
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        out = []
        for g in geom.geoms:
            out.extend(_iter_polygons(g))
        return out
    return []


class RoadTexture:

    def __init__(self, cfg):
        self.cfg = cfg

    def generate(
        self,
        road_poly,
        meta:       dict,
        output_dir: str,
        base_name:  str = "terrain",
    ) -> str:
        """
        Vygeneruje PNG texturu se silnicemi.

        Parametry z config:
            ROAD_TEXTURE_SIZE    – strana čtverce [px], default 512
            ROAD_TEXTURE_BLUR_PX – poloměr Gaussian blur [px], default 4
            ROAD_COLOR           – RGB tuple asfaltu
            TERRAIN_COLOR        – RGB tuple terénu (pouze výplň – v praxi
                                   překryje satellite/height-based textura)

        Vrátí cestu k PNG souboru.
        """
        cfg       = self.cfg
        size      = int(getattr(cfg, "ROAD_TEXTURE_SIZE",    512))
        blur_px   = float(getattr(cfg, "ROAD_TEXTURE_BLUR_PX", 4))
        road_col  = tuple(getattr(cfg, "ROAD_COLOR",    DEFAULT_ROAD_COLOR))
        terr_col  = tuple(getattr(cfg, "TERRAIN_COLOR", DEFAULT_TERRAIN_COLOR))

        W = meta["total_width_m"]
        H = meta["total_height_m"]

        def to_px(x: float, y: float) -> tuple[int, int]:
            """Local metric → pixel coords (PIL convention: y=0 nahoře = sever)."""
            px = int(round(x / W * (size - 1)))
            py = int(round((1.0 - y / H) * (size - 1)))
            return (max(0, min(size - 1, px)),
                    max(0, min(size - 1, py)))

        # ── 1. Binární maska (L=255 = silnice, L=0 = terén) ───────────────
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)

        for poly in _iter_polygons(road_poly):
            if poly.is_empty:
                continue
            # Vnější obrys → vyplnit bíle (= road)
            ext_px = [to_px(x, y) for x, y in poly.exterior.coords]
            if len(ext_px) >= 3:
                draw.polygon(ext_px, fill=255)
            # Díry (city bloky) → přebarvit zpět na černo (= terrain)
            for interior in poly.interiors:
                hole_px = [to_px(x, y) for x, y in interior.coords]
                if len(hole_px) >= 3:
                    draw.polygon(hole_px, fill=0)

        # ── 2. Soft edge: Gaussian blur ────────────────────────────────────
        if blur_px > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_px))

        # ── 3. Blend terrain_color ↔ road_color ────────────────────────────
        terrain_img = Image.new("RGB", (size, size), terr_col)
        road_img    = Image.new("RGB", (size, size), road_col)
        result      = Image.composite(road_img, terrain_img, mask)

        # ── 4. Uložit ──────────────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{base_name}_texture.png")
        result.save(out_path, "PNG", optimize=True)

        kb = os.path.getsize(out_path) / 1024
        print(f"   Road textura: {out_path}  ({size}×{size}px, {kb:.0f} KB)")
        return out_path
