"""
OSMBuildings tile data client.

Fetches pre-processed building GeoJSON from the OSMBuildings server:
  https://{s}.data.osmbuildings.org/0.2/{key}/tile/{z}/{x}/{y}.json

The server returns GeoJSON features with numeric properties already in metres:
  height, minHeight, roofHeight, roofShape, wallColor, roofColor, shape, levels, minLevel

We convert these back to OSM-tag-style dicts so the existing BuildingExtruder
pipeline works without modification.

API key: register at https://osmbuildings.org/data/
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

log = logging.getLogger(__name__)

# OSMBuildings uses zoom 15 for its main tile layer
_TILE_ZOOM = 15
_SUBDOMAINS = ["a", "b", "c", "d"]
_BASE_URL = "https://{s}.data.osmbuildings.org/0.2/{key}/tile/{z}/{x}/{y}.json"
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "Chrome/124.0.0.0 Safari/537.36"
)
_REQUEST_DELAY = 0.2   # seconds between requests (be polite)


# ---------------------------------------------------------------------------
# Tile math
# ---------------------------------------------------------------------------

def _lat_lon_to_tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    x = int((lon + 180.0) / 360.0 * (1 << z))
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi)
            / 2.0 * (1 << z))
    return x, y


def _tile_bbox(tx: int, ty: int, z: int) -> tuple[float, float, float, float]:
    """Return (south, west, north, east) of tile in degrees."""
    n = 1 << z

    def merc_to_lat(y_merc: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_merc))))

    west  = tx / n * 360.0 - 180.0
    east  = (tx + 1) / n * 360.0 - 180.0
    north = merc_to_lat(ty / n)
    south = merc_to_lat((ty + 1) / n)
    return south, west, north, east


def _tiles_in_bbox(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float, z: int
) -> list[tuple[int, int]]:
    x0, y0 = _lat_lon_to_tile(max_lat, min_lon, z)   # north-west → small y
    x1, y1 = _lat_lon_to_tile(min_lat, max_lon, z)   # south-east → large y
    return [(tx, ty) for tx in range(x0, x1 + 1) for ty in range(y0, y1 + 1)]


# ---------------------------------------------------------------------------
# HTTP fetch with retry
# ---------------------------------------------------------------------------

def _fetch_tile(
    tx: int, ty: int, z: int, key: str, subdomain_idx: int = 0
) -> dict | None:
    url = _BASE_URL.format(
        s=_SUBDOMAINS[subdomain_idx % len(_SUBDOMAINS)],
        key=key,
        z=z,
        x=tx,
        y=ty,
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/json, */*",
            "Referer": "https://osmbuildings.org/",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
            # Check for the registration-required message
            if "requires a registration" in raw:
                log.error("OSMBuildings API key '%s' rejected: %s", key, raw.strip())
                return None
            return json.loads(raw)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        log.warning("OSMBuildings tile %d/%d/%d fetch failed: %s", z, tx, ty, exc)
        return None


# ---------------------------------------------------------------------------
# GeoJSON → internal building list
# ---------------------------------------------------------------------------

def _props_to_tags(props: dict) -> dict:
    """
    Convert OSMBuildings GeoJSON properties to OSM-tag-style dict.

    OSMBuildings numeric properties (already in metres) are converted to
    string tags so BuildingExtruder can consume them unchanged.
    """
    tags: dict[str, str] = {}

    if props.get("height") is not None:
        tags["height"] = str(props["height"])

    if props.get("minHeight") is not None and props["minHeight"] != 0:
        tags["min_height"] = str(props["minHeight"])

    if props.get("levels") is not None:
        # Only add if no explicit height was given, to avoid double-counting
        if "height" not in tags:
            tags["building:levels"] = str(props["levels"])

    if props.get("minLevel") is not None and props["minLevel"] != 0:
        if "min_height" not in tags:
            tags["min_level"] = str(props["minLevel"])

    if props.get("roofShape"):
        tags["roof:shape"] = str(props["roofShape"])

    if props.get("roofHeight") is not None and props["roofHeight"] != 0:
        tags["roof:height"] = str(props["roofHeight"])

    # Building shape (cylinder, sphere …)
    if props.get("shape"):
        tags["building:shape"] = str(props["shape"])

    # Colour / material (informational — extruder ignores them but keep for future)
    for osm_key, prop_key in [
        ("building:colour",  "wallColor"),
        ("roof:colour",      "roofColor"),
        ("building:material","material"),
    ]:
        if props.get(prop_key):
            tags[osm_key] = str(props[prop_key])

    # Mark as a building part so outer-shell logic applies
    tags["building:part"] = "yes"

    return tags


def _geojson_to_buildings(
    feature_collection: dict,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> list[dict]:
    """
    Extract building dicts from an OSMBuildings GeoJSON FeatureCollection.

    Each dict has keys:
      id    – integer (from GeoJSON id or generated)
      nodes – list of {lon, lat} dicts
      tags  – OSM-tag-style property dict
    """
    buildings: list[dict] = []
    features = feature_collection.get("features", [])

    for feat_idx, feat in enumerate(features):
        geom = feat.get("geometry", {})
        props = feat.get("properties", {}) or {}

        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue

        # Skip features whose centroid is outside our bbox
        # (tiles overlap at borders, avoid duplicates)
        tags = _props_to_tags(props)
        bid  = feat.get("id", -(feat_idx + 1))

        rings_list = (
            geom["coordinates"]
            if geom["type"] == "Polygon"
            else geom["coordinates"]     # MultiPolygon handled ring-by-ring below
        )

        if geom["type"] == "Polygon":
            outer_ring = rings_list[0]
            if not outer_ring:
                continue
            nodes = [{"lon": c[0], "lat": c[1]} for c in outer_ring]
            # Centroid check
            c_lon = sum(n["lon"] for n in nodes) / len(nodes)
            c_lat = sum(n["lat"] for n in nodes) / len(nodes)
            if not (min_lat <= c_lat <= max_lat and min_lon <= c_lon <= max_lon):
                continue
            buildings.append({"id": bid, "nodes": nodes, "tags": tags})

        else:  # MultiPolygon
            for poly_idx, polygon in enumerate(rings_list):
                outer_ring = polygon[0] if polygon else []
                if not outer_ring:
                    continue
                nodes = [{"lon": c[0], "lat": c[1]} for c in outer_ring]
                c_lon = sum(n["lon"] for n in nodes) / len(nodes)
                c_lat = sum(n["lat"] for n in nodes) / len(nodes)
                if not (min_lat <= c_lat <= max_lat and min_lon <= c_lon <= max_lon):
                    continue
                buildings.append({
                    "id": bid * 1000 + poly_idx,
                    "nodes": nodes,
                    "tags": dict(tags),
                })

    return buildings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class OSMBuildingsClient:
    """
    Download and cache building data from the OSMBuildings tile server.

    Parameters
    ----------
    api_key : str
        OSMBuildings API key.  Register at https://osmbuildings.org/data/
    cache_dir : Path | None
        Directory for tile cache.  Pass None to disable caching.
    zoom : int
        Tile zoom level (default 15 — same as OSMBuildings client default).
    """

    def __init__(
        self,
        api_key: str,
        cache_dir: Path | None = None,
        zoom: int = _TILE_ZOOM,
    ) -> None:
        self._key = api_key
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._zoom = zoom

    # ------------------------------------------------------------------
    def fetch_buildings(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        force_download: bool = False,
    ) -> list[dict]:
        """
        Return a flat list of building dicts for the given bounding box.

        Tiles are fetched in sequence; results are cached per-tile.
        """
        tiles = _tiles_in_bbox(min_lat, min_lon, max_lat, max_lon, self._zoom)
        log.info(
            "OSMBuildings: fetching %d tile(s) at z=%d for bbox "
            "(%.4f,%.4f)-(%.4f,%.4f)",
            len(tiles), self._zoom, min_lat, min_lon, max_lat, max_lon,
        )

        all_buildings: list[dict] = []
        seen_ids: set[int] = set()

        for sub_idx, (tx, ty) in enumerate(tiles):
            geojson = self._load_tile(tx, ty, force_download, sub_idx)
            if geojson is None:
                continue

            # Clip to bbox so buildings at tile edges aren't duplicated
            tile_south, tile_west, tile_north, tile_east = _tile_bbox(tx, ty, self._zoom)
            clip_s = max(min_lat, tile_south)
            clip_w = max(min_lon, tile_west)
            clip_n = min(max_lat, tile_north)
            clip_e = min(max_lon, tile_east)

            buildings = _geojson_to_buildings(geojson, clip_s, clip_w, clip_n, clip_e)
            for b in buildings:
                if b["id"] not in seen_ids:
                    seen_ids.add(b["id"])
                    all_buildings.append(b)

            time.sleep(_REQUEST_DELAY)

        log.info("OSMBuildings: %d unique buildings in bbox", len(all_buildings))
        return all_buildings

    # ------------------------------------------------------------------
    def _cache_path(self, tx: int, ty: int) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"osmb_{self._zoom}_{tx}_{ty}.json"

    def _load_tile(
        self, tx: int, ty: int, force_download: bool, sub_idx: int
    ) -> dict | None:
        path = self._cache_path(tx, ty)

        if path and path.exists() and not force_download:
            log.debug("OSMBuildings cache hit: %s", path)
            try:
                return json.loads(path.read_text())
            except Exception:
                pass

        log.debug("OSMBuildings fetching tile %d/%d/%d", self._zoom, tx, ty)
        data = _fetch_tile(tx, ty, self._zoom, self._key, sub_idx)
        if data is None:
            return None

        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))

        return data
