"""
Batch Overpass fetch (v2) — one union bbox, local clip per tile
===============================================================
Public instances (e.g. ``overpass-api.de``) publish **fair-use** guidance, not a
fixed max bbox: on the order of **~10⁴ requests/day** and **~1 GiB/day** total
download volume for heavy use; very large queries may still fail with timeout,
HTTP 429, or load shedding.

There is **no guaranteed safe geographic size**. In practice, for **dense
European** OSM data:

* **&lt; ~0.25° edge** (≈ 25–30 km) — usually reliable with current timeouts.
* **~0.5°–0.8°** on a side (roughly **0.25–0.65 deg²**) — often OK for mixed
  suburb/countryside (your Sázava-scale rectangle is in this band); watch for
  slow responses in city cores.
* **&gt; ~1 deg²** in urban areas — increasing risk of **timeout** or huge JSON;
  prefer a local extract (Geofabrik + osmium) or your own Overpass instance.

**RAM (rough):** parsed Python objects for highways + buildings are typically
a few multiples of the JSON body size — **tens of MB** for regional batches,
**low hundreds of MB** for aggressive rectangles on crowded data. Peak usage
**Clip cost:** per-tile subsetting uses a **STRtree** built once on the union
geometries so each tile is O(log n + k), not O(n) over all union features.

This module is **additive**; the legacy per-tile ``OsmClient`` is unchanged.
"""
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

from shapely.geometry import LineString, Polygon, box
from shapely.strtree import STRtree

if TYPE_CHECKING:
    from src.extraction.osm_client import OsmClient
    from src.pipeline.task_runner import TileConfig

# Soft limits: warnings only (degrees² of axis-aligned bbox).
_WARN_UNION_AREA_DEG2 = 0.35   # ~0.6°×0.6° — "getting heavy"
_STRONG_WARN_UNION_AREA_DEG2 = 1.0


def union_bbox_deg2(b: tuple[float, float, float, float]) -> float:
    south, west, north, east = b
    return max(0.0, north - south) * max(0.0, east - west)


def warn_union_bbox_if_heavy(union: tuple[float, float, float, float]) -> None:
    """Print guidance to stdout when the union rectangle is large."""
    area = union_bbox_deg2(union)
    edge_n_s = abs(union[2] - union[0])
    edge_w_e = abs(union[3] - union[1])
    km_approx = math.sqrt(area) * 111.0

    if area >= _STRONG_WARN_UNION_AREA_DEG2:
        print(
            f"\n  ⚠ OSM batch v2: union bbox ≈ {edge_n_s:.2f}° N–S × {edge_w_e:.2f}° E–W "
            f"(~{area:.2f} deg² ≈ {km_approx:.0f} km scale). "
            "Veřejný Overpass může vypršet nebo vrátit obří odpověď — zvaž menší výřez "
            "nebo vlastní instanci / Geofabrik PBF.\n"
        )
    elif area >= _WARN_UNION_AREA_DEG2:
        print(
            f"\n  ℹ OSM batch v2: union ~{area:.2f} deg² — dotaz může trvat dlouho; "
            f"RAM pro rozparsovaná data typicky desítky až ~200+ MB u husté zástavby.\n"
        )


def compute_tile_union_bbox(tiles: list[TileConfig]) -> tuple[float, float, float, float]:
    return (
        min(t.bbox[0] for t in tiles),
        min(t.bbox[1] for t in tiles),
        max(t.bbox[2] for t in tiles),
        max(t.bbox[3] for t in tiles),
    )


def _tile_output_dir(tile: TileConfig, output_root: str) -> str:
    if tile.output_dir:
        return tile.output_dir
    if tile.xyz is not None:
        z, x, y = tile.xyz
        return os.path.join(output_root, str(z), str(x), str(y))
    return os.path.join(output_root, tile.name)


def all_tiles_have_osm_json_cache(
    tiles: list[TileConfig],
    output_root: str,
    *,
    require_landcover: bool = True,
) -> bool:
    """True if every tile dir has highway + building (+ landcover) JSON caches."""
    for t in tiles:
        tdir = _tile_output_dir(t, output_root)
        base = str(t.xyz.y) if t.xyz else t.name
        h = os.path.join(tdir, f"{base}_osm_highways.json")
        b = os.path.join(tdir, f"{base}_osm_buildings.json")
        if not (os.path.isfile(h) and os.path.isfile(b)):
            return False
        if require_landcover:
            lc = os.path.join(tdir, f"{base}_osm_landcover.json")
            if not os.path.isfile(lc):
                return False
    return True


def clip_highways_to_bbox(
    features: list[dict],
    bbox: tuple[float, float, float, float],
) -> list[dict]:
    """
    Keep highway features whose centreline intersects the bbox rectangle.

    Semantics follow Overpass ``way[highway](south,west,north,east)`` in spirit:
    geometry crossing the selection wins even when endpoints lie outside.
    """
    south, west, north, east = bbox
    # minx, miny, maxx, maxy in (lon, lat)
    rect = box(west, south, east, north)
    out: list[dict] = []
    for f in features:
        nodes = f.get("nodes") or []
        if len(nodes) < 2:
            continue
        coords = [(float(n["lon"]), float(n["lat"])) for n in nodes]
        try:
            line = LineString(coords)
            if line.is_empty or not line.intersects(rect):
                continue
        except Exception:
            continue
        out.append(f)
    return out


def clip_buildings_to_bbox(
    features: list[dict],
    bbox: tuple[float, float, float, float],
) -> list[dict]:
    """Keep building / building:part features whose footprint intersects bbox."""
    south, west, north, east = bbox
    rect = box(west, south, east, north)
    out: list[dict] = []
    for f in features:
        nodes = f.get("nodes") or []
        if len(nodes) < 3:
            continue
        coords = [(float(n["lon"]), float(n["lat"])) for n in nodes]
        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or not poly.intersects(rect):
                continue
        except Exception:
            continue
        out.append(f)
    return out


def _index_lines(features: list[dict]) -> tuple[list[dict], list[LineString]]:
    out_f: list[dict] = []
    out_g: list[LineString] = []
    for f in features:
        nodes = f.get("nodes") or []
        if len(nodes) < 2:
            continue
        coords = [(float(n["lon"]), float(n["lat"])) for n in nodes]
        try:
            line = LineString(coords)
            if line.is_empty:
                continue
        except Exception:
            continue
        out_f.append(f)
        out_g.append(line)
    return out_f, out_g


def _index_polys(features: list[dict]) -> tuple[list[dict], list[Polygon]]:
    out_f: list[dict] = []
    out_g: list[Polygon] = []
    for f in features:
        nodes = f.get("nodes") or []
        if len(nodes) < 3:
            continue
        coords = [(float(n["lon"]), float(n["lat"])) for n in nodes]
        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
        except Exception:
            continue
        out_f.append(f)
        out_g.append(poly)
    return out_f, out_g


class ClippingOsmClient:
    """
    Drop-in replacement for ``OsmClient`` that serves subsets of pre-fetched
    highway/building/landcover lists (same ``dict`` shape as ``OsmClient`` parsers).

    Uses a **Shapely STRtree** so each tile clips in ~O(log n + k) instead of
    scanning the full union (the naive path was O(tiles × features) and could
    erase all Overpass savings).
    """

    __slots__ = (
        "_hw_features",
        "_hw_lines",
        "_hw_tree",
        "_b_features",
        "_b_polys",
        "_b_tree",
        "_wp_features",
        "_wp_polys",
        "_wp_tree",
        "_ww_features",
        "_ww_lines",
        "_ww_tree",
        "_gp_features",
        "_gp_polys",
        "_gp_tree",
        "_rw_features",
        "_rw_lines",
        "_rw_tree",
    )

    def __init__(
        self,
        highways: list[dict],
        buildings: list[dict],
        landcover: dict[str, list] | None = None,
    ):
        self._hw_features, self._hw_lines = _index_lines(highways)
        self._hw_tree = STRtree(self._hw_lines) if self._hw_lines else STRtree([])

        self._b_features, self._b_polys = _index_polys(buildings)
        self._b_tree = STRtree(self._b_polys) if self._b_polys else STRtree([])

        lc = landcover or {}
        self._wp_features, self._wp_polys = _index_polys(lc.get("water_polygons", []))
        self._wp_tree = STRtree(self._wp_polys) if self._wp_polys else STRtree([])

        self._ww_features, self._ww_lines = _index_lines(lc.get("waterways", []))
        self._ww_tree = STRtree(self._ww_lines) if self._ww_lines else STRtree([])

        self._gp_features, self._gp_polys = _index_polys(lc.get("green_polygons", []))
        self._gp_tree = STRtree(self._gp_polys) if self._gp_polys else STRtree([])

        self._rw_features, self._rw_lines = _index_lines(lc.get("railways", []))
        self._rw_tree = STRtree(self._rw_lines) if self._rw_lines else STRtree([])

        n_lc = (
            len(self._wp_features) + len(self._ww_features)
            + len(self._gp_features) + len(self._rw_features)
        )
        print(
            f"   OSM batch index: {len(self._hw_features)} silnic, "
            f"{len(self._b_features)} budových polygonů, "
            f"{n_lc} landcover prvků (STRtree)"
        )

    def _rect(self, bbox: tuple[float, float, float, float]) -> Polygon:
        south, west, north, east = bbox
        return box(west, south, east, north)

    def _query_rect_indices(self, tree: STRtree, bbox: tuple[float, float, float, float]) -> list[int]:
        rect = self._rect(bbox)
        raw = tree.query(rect, predicate="intersects")
        if hasattr(raw, "tolist"):
            return [int(i) for i in raw.tolist()]
        return [int(i) for i in raw]

    def get_highways(self, bbox: tuple) -> list[dict]:
        bbf = tuple(map(float, bbox))
        if not self._hw_features:
            return []
        idxs = self._query_rect_indices(self._hw_tree, bbf)
        rect = self._rect(bbf)
        out: list[dict] = []
        for i in idxs:
            try:
                if self._hw_lines[i].intersects(rect):
                    out.append(self._hw_features[i])
            except Exception:
                continue
        return out

    def get_buildings(self, bbox: tuple) -> list[dict]:
        clipped = self.get_buildings_with_parts(bbox)
        return [
            f for f in clipped
            if "building" in f.get("tags", {}) and "building:part" not in f.get("tags", {})
        ]

    def get_buildings_with_parts(self, bbox: tuple) -> list[dict]:
        bbf = tuple(map(float, bbox))
        if not self._b_features:
            return []
        idxs = self._query_rect_indices(self._b_tree, bbf)
        rect = self._rect(bbf)
        out: list[dict] = []
        for i in idxs:
            try:
                if self._b_polys[i].intersects(rect):
                    out.append(self._b_features[i])
            except Exception:
                continue
        return out

    def _clip_indexed_lines(
        self,
        bbox: tuple[float, float, float, float],
        features: list[dict],
        lines: list[LineString],
        tree: STRtree,
    ) -> list[dict]:
        if not features:
            return []
        idxs = self._query_rect_indices(tree, bbox)
        rect = self._rect(bbox)
        out: list[dict] = []
        for i in idxs:
            try:
                if lines[i].intersects(rect):
                    out.append(features[i])
            except Exception:
                continue
        return out

    def _clip_indexed_polys(
        self,
        bbox: tuple[float, float, float, float],
        features: list[dict],
        polys: list[Polygon],
        tree: STRtree,
    ) -> list[dict]:
        if not features:
            return []
        idxs = self._query_rect_indices(tree, bbox)
        rect = self._rect(bbox)
        out: list[dict] = []
        for i in idxs:
            try:
                if polys[i].intersects(rect):
                    out.append(features[i])
            except Exception:
                continue
        return out

    def get_landcover(self, bbox: tuple) -> dict[str, list]:
        bbf = tuple(map(float, bbox))
        return {
            "water_polygons": self._clip_indexed_polys(
                bbf, self._wp_features, self._wp_polys, self._wp_tree,
            ),
            "waterways": self._clip_indexed_lines(
                bbf, self._ww_features, self._ww_lines, self._ww_tree,
            ),
            "green_polygons": self._clip_indexed_polys(
                bbf, self._gp_features, self._gp_polys, self._gp_tree,
            ),
            "railways": self._clip_indexed_lines(
                bbf, self._rw_features, self._rw_lines, self._rw_tree,
            ),
        }


def prefetch_highways_and_buildings(
    union: tuple[float, float, float, float],
    client: OsmClient | None = None,
) -> tuple[list[dict], list[dict], dict[str, list]]:
    """
    Three Overpass round-trips for the full union rectangle:
    highways, S3DB buildings, landcover (water / rivers / green / railways).
    """
    from src.extraction.osm_client import OsmClient

    c = client or OsmClient()
    highways = c.get_highways(union)
    buildings = c.get_buildings_with_parts(union)
    landcover = c.get_landcover(union)
    return highways, buildings, landcover
