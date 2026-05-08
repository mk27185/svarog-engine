"""
GeoJSON buildings client — reads a local overpass-turbo GeoJSON export.

The file is loaded once into memory; subsequent calls to ``get_buildings_with_parts``
just filter by bbox (fast ~O(n) scan).  For 200 k features the load takes ~4 s
and uses ~500 MB RAM; subsequent queries take milliseconds.

Usage
-----
    from src.extraction.geojson_buildings_client import GeoJSONBuildingsClient
    client = GeoJSONBuildingsClient("export.geojson")
    buildings = client.get_buildings_with_parts((min_lat, min_lon, max_lat, max_lon))
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# GeoJSON property keys that are metadata, not OSM tags
_META_KEYS = {"@id", "@type", "@snapshotDate", "@version", "@changeset",
              "@timestamp", "@user", "@uid", "type"}


def _parse_id(osm_id_str: str | None, fallback: int) -> int:
    """'way/12345' or 'relation/12345' → int (ways positive, relations negative)."""
    if not osm_id_str:
        return fallback
    parts = osm_id_str.split("/")
    try:
        num = int(parts[-1])
    except (ValueError, IndexError):
        return fallback
    if len(parts) >= 2 and parts[0] == "relation":
        return -num
    return num


def _ring_to_nodes(ring: list) -> list[dict]:
    return [{"lon": pt[0], "lat": pt[1]} for pt in ring]


def _centroid(nodes: list[dict]) -> tuple[float, float]:
    n = len(nodes)
    if n == 0:
        return 0.0, 0.0
    return (
        sum(nd["lat"] for nd in nodes) / n,
        sum(nd["lon"] for nd in nodes) / n,
    )


def _feature_to_buildings(feat: dict, idx: int) -> list[dict]:
    """
    Convert one GeoJSON Feature to a list of internal building dicts.

    MultiPolygon → one building per outer ring.
    Polygon      → one building.
    Each building:  {id, nodes, tags}
    """
    geom  = feat.get("geometry")
    props = feat.get("properties") or {}
    if not geom:
        return []

    raw_id = props.get("@id")
    base_id = _parse_id(raw_id, idx)

    # Strip metadata keys; keep OSM tags
    tags = {k: str(v) for k, v in props.items()
            if k not in _META_KEYS and v is not None}

    results: list[dict] = []

    if geom["type"] == "Polygon":
        outer = geom["coordinates"][0] if geom["coordinates"] else []
        if len(outer) < 3:
            return []
        nodes = _ring_to_nodes(outer)
        results.append({"id": base_id, "nodes": nodes, "tags": tags})

    elif geom["type"] == "MultiPolygon":
        for pi, polygon in enumerate(geom["coordinates"]):
            outer = polygon[0] if polygon else []
            if len(outer) < 3:
                continue
            nodes = _ring_to_nodes(outer)
            results.append({
                "id": base_id * 1000 + pi if pi > 0 else base_id,
                "nodes": nodes,
                "tags": dict(tags),
            })

    return results


class GeoJSONBuildingsClient:
    """
    Building data source backed by a local overpass-turbo GeoJSON export.

    Parameters
    ----------
    path : str | Path
        Path to the GeoJSON file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._buildings: list[dict] | None = None   # lazy-loaded

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._buildings is not None:
            return
        log.info("GeoJSON: loading %s ...", self._path)
        print(f"  Načítám GeoJSON budovy: {self._path} ({self._path.stat().st_size // 1_048_576} MB)...",
              flush=True)
        with open(self._path, encoding="utf-8") as fh:
            data = json.load(fh)

        features = data.get("features", [])
        buildings: list[dict] = []
        seen_ids: set[int] = set()

        for idx, feat in enumerate(features):
            for b in _feature_to_buildings(feat, idx):
                if b["id"] not in seen_ids:
                    seen_ids.add(b["id"])
                    buildings.append(b)

        self._buildings = buildings
        print(f"  ✓ {len(buildings)} budov načteno z GeoJSON", flush=True)

    # ------------------------------------------------------------------
    def get_buildings_with_parts(
        self,
        bbox: tuple[float, float, float, float],
    ) -> list[dict]:
        """
        Return buildings whose centroid falls inside bbox.

        Follows the OSM S3DB rule: when building:part=* elements are present,
        the outer building=* shell is suppressed (not rendered in 3D) to avoid
        Z-fighting with the parts.

        bbox : (min_lat, min_lon, max_lat, max_lon)
        """
        self._ensure_loaded()
        min_lat, min_lon, max_lat, max_lon = bbox

        in_bbox: list[dict] = []
        for b in self._buildings:  # type: ignore[union-attr]
            lat, lon = _centroid(b["nodes"])
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                in_bbox.append(b)

        parts = [b for b in in_bbox if "building:part" in b["tags"]]
        if not parts:
            return in_bbox

        # Suppress outer building shells that are covered by parts.
        # Re-use the same Shapely-based logic as OsmClient._suppress_outer_shells.
        try:
            from src.extraction.osm_client import OsmClient
            outlines = [b for b in in_bbox if "building" in b["tags"]
                        and "building:part" not in b["tags"]]
            kept_outlines = OsmClient._suppress_outer_shells(outlines, parts)
            return parts + kept_outlines
        except Exception:
            return in_bbox

    # Alias used by terrain_pipeline
    get_buildings = get_buildings_with_parts

    def get_highways(self, bbox: tuple) -> list:  # noqa: ARG002
        """Not supported — GeoJSON source is buildings-only; roads skipped."""
        return []
