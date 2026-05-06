"""
SDF Road Texture Generator
==========================
Produce a multi-channel RGBA PNG from OSM road polylines:

  R channel - signed-distance-field  (0.0 = far outside, 0.5 = road edge,
                                     1.0 = deep inside)
  G channel - highway classification (1.0 = motorway, 0.0 = footway/path)
  B channel - normalised road width  (0.0 = narrowest, 1.0 = widest)
  A channel - road mask              (1.0 where any road, 0.0 elsewhere)

The texture overlays roads on a plain terrain mesh in Three.js
without extra geometry.
"""

import numpy as np
from PIL import Image, ImageDraw

# Defaults
DEFAULT_RESOLUTION = 1024

# Ranking: higher = more important road (used for G channel)
HIGHWAY_RANK: dict[str, float] = {
    "motorway":        1.00,
    "motorway_link":   0.90,
    "trunk":           0.85,
    "trunk_link":      0.75,
    "primary":         0.70,
    "primary_link":    0.65,
    "secondary":       0.60,
    "secondary_link":  0.55,
    "tertiary":        0.50,
    "tertiary_link":   0.45,
    "unclassified":    0.40,
    "residential":     0.40,
    "service":         0.30,
    "living_street":   0.30,
    "pedestrian":      0.20,
    "track":           0.15,
    "footway":         0.10,
    "path":            0.10,
    "cycleway":        0.20,
    "steps":           0.20,
    "bridleway":       0.20,
}
_DEFAULT_RANK = 0.35

# Half-widths in metres (mirrors terrain_stamper)
_ROAD_HALF_WIDTHS: dict[str, float] = {
    "motorway": 7.0,       "motorway_link": 3.5,
    "trunk": 6.0,          "trunk_link": 3.0,
    "primary": 5.0,        "primary_link": 2.5,
    "secondary": 4.0,      "secondary_link": 2.0,
    "tertiary": 3.0,       "tertiary_link": 1.5,
    "unclassified": 2.5,   "residential": 2.5,
    "service": 1.5,        "living_street": 2.0,
    "pedestrian": 2.0,     "track": 1.5,
    "footway": 0.75,       "path": 0.75,
    "cycleway": 1.0,       "steps": 1.0,
    "bridleway": 1.25,
}
_DEFAULT_HALF_WIDTH = 2.5


class SDFGenerator:
    """Generate multi-channel SDF road texture from OSM highways."""

    def __init__(
        self,
        resolution: int = DEFAULT_RESOLUTION,
        padding: float = 0.15,
    ):
        self.resolution = resolution
        self.padding = padding

    # ---- PUBLIC API ------------------------------------------------

    def generate(
        self,
        highways: list[dict],
        meta: dict,
        output_path: str,
    ) -> str:
        """Generate RGBA SDF PNG.  Returns file path."""

        res = self.resolution

        # ── collect & convert road segments ──────────────────────
        roads: list[dict] = []
        max_hw = 0.0

        for road in highways:
            nodes = road.get("nodes", [])
            if len(nodes) < 2:
                continue

            tags = road.get("tags", {})
            half_w = self._half_width(tags)
            padded_hw = half_w * (1.0 + self.padding)
            rank = HIGHWAY_RANK.get(
                tags.get("highway", ""), _DEFAULT_RANK
            )

            from .geo_utils import to_local

            # Nodes may be dicts {"lon":..., "lat":...} (OSM parser)
            # or tuples (lon, lat) — handle both
            if nodes and isinstance(nodes[0], dict):
                local_pts = [
                    to_local(n["lon"], n["lat"], meta) for n in nodes
                ]
            else:
                local_pts = [to_local(lon, lat, meta) for lon, lat in nodes]
            pixel_pts = self._to_pixel(local_pts, meta)
            if len(pixel_pts) < 2:
                continue

            roads.append({
                "pixels": pixel_pts,
                "half_w": padded_hw,
                "raw_hw": half_w,
                "rank": rank,
            })
            max_hw = max(max_hw, padded_hw)

        if not roads:
            print("   SDFGenerator: zadne silnice - prazdna textura.")
            empty = np.zeros((res, res, 4), dtype=np.uint8)
            Image.fromarray(empty, "RGBA").save(output_path)
            return output_path

        # ── draw into Pillow float images ───────────────────────
        imgs = {
            ch: Image.new("F", (res, res), 0.0)
            for ch in ("mask", "rank", "width")
        }
        draws = {ch: ImageDraw.Draw(imgs[ch]) for ch in imgs}

        tw = meta["total_width_m"]
        th = meta["total_height_m"]
        dim = max(tw, th)

        for rd in roads:
            pts = rd["pixels"]
            hw  = rd["half_w"]
            rk  = rd["rank"]
            raw = rd["raw_hw"]

            # stroke width: scale road width to texture resolution
            stroke = max(2, int(hw / dim * res))

            polyline = [
                (float(p[0]) * res, float(p[1]) * res) for p in pts
            ]

            draws["mask"].line(polyline, width=stroke, fill=1.0)
            draws["rank"].line(polyline, width=stroke, fill=rk)
            draws["width"].line(polyline, width=stroke, fill=raw)

        # ── distance transform (R channel) ──────────────────────
        mask = np.asarray(imgs["mask"], dtype=np.float32)
        sdf = self._make_sdf(mask)

        # ── normalise B channel ─────────────────────────────────
        raw_w = np.asarray(imgs["width"], dtype=np.float32)
        if max_hw > 0:
            norm_w = np.divide(
                raw_w, max_hw,
                out=np.zeros_like(raw_w),
                where=raw_w > 0,
            )
        else:
            norm_w = raw_w

        # ── merge RGBA ──────────────────────────────────────────
        out = np.zeros((res, res, 4), dtype=np.float32)
        out[:, :, 0] = sdf                                 # R
        out[:, :, 1] = np.asarray(imgs["rank"], dtype=np.float32)  # G
        out[:, :, 2] = norm_w                               # B
        out[:, :, 3] = mask                                 # A

        self._save_png(out, output_path)
        print(f"   SDF texture: {output_path}  ({res}x{res} px)")
        return output_path

    # ---- SDF ------------------------------------------------------

    @staticmethod
    def _make_sdf(mask: np.ndarray) -> np.ndarray:
        """Binary mask -> SDF  [0, 1],  0.5 at road edge."""
        from scipy.ndimage import distance_transform_edt

        d_inside = distance_transform_edt(mask > 0.5)
        d_outside = distance_transform_edt(mask <= 0.5)

        signed = d_inside - d_outside
        mx = max(20.0, abs(signed.max()), abs(signed.min()))
        normalised = 0.5 + (signed / mx) * 0.5
        return np.clip(normalised, 0.0, 1.0)

    # ---- helpers --------------------------------------------------

    @staticmethod
    def _to_pixel(
        local_pts: list[tuple[float, float]],
        meta: dict,
    ) -> np.ndarray:
        """(x_m, y_m) -> normalised (u, v) in [0, 1)."""
        pts = np.array(local_pts, dtype=np.float32)
        u = np.clip(pts[:, 0] / meta["total_width_m"], 0.0, 1.0)
        v = np.clip(1.0 - pts[:, 1] / meta["total_height_m"], 0.0, 1.0)
        return np.stack([u, v], axis=1)

    @staticmethod
    def _half_width(tags: dict) -> float:
        """Half-width in metres (repeated from terrain_stamper logic)."""
        hw_raw = tags.get("width")
        if hw_raw:
            try:
                v = float(hw_raw)
                if v > 0:
                    return v / 2.0
            except (ValueError, TypeError):
                pass
        hwy = tags.get("highway", "")
        return _ROAD_HALF_WIDTHS.get(hwy, _DEFAULT_HALF_WIDTH)

    @staticmethod
    def _save_png(arr: np.ndarray, path: str) -> None:
        arr8 = np.clip(arr, 0.0, 1.0) * 255
        Image.fromarray(arr8.astype(np.uint8), "RGBA").save(path)