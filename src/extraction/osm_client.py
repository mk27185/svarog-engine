import random
import time

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "svarog-engine/0.1", "Accept": "*/*"}


class OsmClient:
    """Downloads OSM features for a bounding box via Overpass API."""

    # Retry policy for 429 / transient network errors.
    # Waits: 10s, 20s, 40s, 80s → total max ~2.5 min before giving up.
    MAX_RETRIES = 4
    BASE_WAIT   = 10   # seconds for the first retry
    MAX_WAIT    = 90   # cap per individual wait

    # ------------------------------------------------------------------
    # Buildings
    # ------------------------------------------------------------------

    def get_buildings(self, bbox: tuple) -> list[dict]:
        """
        Returns building footprints inside bbox (outer shells only, ways only).
        Prefer get_buildings_with_parts() for complete OSM 3D Buildings support.
        """
        elements = self._query(bbox, tag="building", label="budov")
        return self._parse_ways(elements, required_tag="building")

    def get_buildings_with_parts(self, bbox: tuple) -> list[dict]:
        """
        Fetch the full OSM Simple 3D Buildings dataset for a bbox in one query:

          · way[building]          – simple building outlines
          · way[building:part]     – S3DB detail parts (height / min_height)
          · relation[building]     – multipolygon building outlines
          · relation[building:part]– S3DB detail parts as relations

        building:part ways/relations are returned first (higher priority).
        Outer building outlines whose footprint already overlaps with
        building:part polygons are suppressed to prevent Z-fighting.

        Ways that serve as outer rings of building relations are NOT rendered
        as standalone buildings — they are already represented through the
        relation's geometry, preventing duplicate wall geometry.
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        query = (
            f"[out:json][timeout:90];\n"
            f"(\n"
            f"  way[\"building\"]({min_lat},{min_lon},{max_lat},{max_lon});\n"
            f"  way[\"building:part\"]({min_lat},{min_lon},{max_lat},{max_lon});\n"
            f"  relation[\"building\"]({min_lat},{min_lon},{max_lat},{max_lon});\n"
            f"  relation[\"building:part\"]({min_lat},{min_lon},{max_lat},{max_lon});\n"
            f");\n"
            f"out body;>;out skel qt;"
        )
        elements = self._fetch(query, "budov a částí budov")

        _, way_nodes = self._build_indices(elements)

        # IDs of way members that serve as outer rings of building relations.
        # These ways must NOT also be rendered as standalone buildings —
        # the relation's _relations_by_tag processing already covers them.
        relation_outer_ids: set[int] = {
            m["ref"]
            for el in elements
            if el["type"] == "relation" and "building" in el.get("tags", {})
            for m in el.get("members", [])
            if m.get("type") == "way" and m.get("role") in ("outer", "")
        }

        # --- parts (ways then relations) ---------------------------------
        parts: list[dict] = self._ways_by_tag(elements, way_nodes, "building:part")
        seen_part_ids: set[int] = {p["id"] for p in parts}
        for pr in self._relations_by_tag(elements, way_nodes, "building:part"):
            if pr["id"] not in seen_part_ids:
                parts.append(pr)
                seen_part_ids.add(pr["id"])

        # --- outer shells (ways then relations) --------------------------
        # Exclude ways that are already outer rings of building relations.
        buildings: list[dict] = [
            b for b in self._ways_by_tag(elements, way_nodes, "building")
            if b["id"] not in relation_outer_ids
        ]
        seen_bld_ids: set[int] = {b["id"] for b in buildings}
        for br in self._relations_by_tag(elements, way_nodes, "building"):
            if br["id"] not in seen_bld_ids:
                buildings.append(br)
                seen_bld_ids.add(br["id"])

        # Suppress S3DB outer shells already covered by building:part geometry
        buildings = self._suppress_outer_shells(buildings, parts)

        result = parts + buildings
        print(
            f"   Celkem budov a částí: {len(result)} "
            f"({len(parts)} části, {len(buildings)} obrysy)"
        )
        return result

    # ------------------------------------------------------------------
    # Highways
    # ------------------------------------------------------------------

    def get_highways(self, bbox: tuple) -> list[dict]:
        """
        Returns road/path polylines inside bbox.
        bbox: (min_lat, min_lon, max_lat, max_lon)
        """
        elements = self._query(bbox, tag="highway", label="silnic")
        return self._parse_ways(elements, required_tag="highway")

    # ------------------------------------------------------------------
    # Internal – HTTP
    # ------------------------------------------------------------------

    def _query(self, bbox: tuple, tag: str, label: str) -> list[dict]:
        """Build and execute a single-tag way query for the given bbox."""
        min_lat, min_lon, max_lat, max_lon = bbox
        query = (
            f"[out:json][timeout:60];\n"
            f"(way[\"{tag}\"]({min_lat},{min_lon},{max_lat},{max_lon}););\n"
            f"out body;>;out skel qt;"
        )
        return self._fetch(query, label)

    def _fetch(self, query: str, label: str) -> list[dict]:
        """Execute a raw Overpass QL query string with exponential-backoff retry."""
        print(f"   Stahuji OSM {label} z Overpass API...")

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    OVERPASS_URL, data={"data": query},
                    headers=HEADERS, timeout=120,
                )
            except requests.exceptions.RequestException as e:
                if attempt == self.MAX_RETRIES:
                    print(f"   Chyba Overpass API (pokus {attempt + 1}/{self.MAX_RETRIES + 1}): {e}")
                    return []
                wait = self._backoff(attempt)
                print(f"   Chyba sítě ({e}); opakuji za {wait:.0f}s "
                      f"(pokus {attempt + 2}/{self.MAX_RETRIES + 1})...")
                time.sleep(wait)
                continue

            if resp.status_code == 429:
                if attempt == self.MAX_RETRIES:
                    print(f"   Overpass API: příliš mnoho požadavků (429), "
                          f"přeskakuji OSM {label} pro tuto dlaždici.")
                    return []
                wait = self._backoff(attempt)
                print(f"   Overpass rate limit (429); opakuji za {wait:.0f}s "
                      f"(pokus {attempt + 2}/{self.MAX_RETRIES + 1})...")
                time.sleep(wait)
                continue

            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print(f"   Chyba Overpass API: {e}")
                return []

            return resp.json().get("elements", [])

        return []

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with ±20 % jitter."""
        wait = self.BASE_WAIT * (2 ** attempt)
        wait = min(wait, self.MAX_WAIT)
        wait *= 1.0 + random.uniform(-0.2, 0.2)
        return wait

    # ------------------------------------------------------------------
    # Internal – parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_indices(elements: list[dict]) -> tuple[dict, dict]:
        """
        Build two lookup tables from raw Overpass elements:
          node_coords : node_id  → {"lon": float, "lat": float}
          way_nodes   : way_id   → [{"lon": float, "lat": float}, ...]
        """
        node_coords: dict[int, dict] = {
            el["id"]: {"lon": el["lon"], "lat": el["lat"]}
            for el in elements if el["type"] == "node"
        }
        way_nodes: dict[int, list] = {}
        for el in elements:
            if el["type"] != "way":
                continue
            nodes = [
                node_coords[nid]
                for nid in el.get("nodes", [])
                if nid in node_coords
            ]
            if nodes:
                way_nodes[el["id"]] = nodes
        return node_coords, way_nodes

    @staticmethod
    def _ways_by_tag(
        elements: list[dict],
        way_nodes: dict[int, list],
        tag: str,
    ) -> list[dict]:
        """Extract ways that carry ``tag``, using the prebuilt way_nodes index."""
        results = []
        for el in elements:
            if el["type"] != "way":
                continue
            tags = el.get("tags", {})
            if tag not in tags:
                continue
            nodes = list(way_nodes.get(el["id"], []))
            if len(nodes) > 1 and nodes[0] == nodes[-1]:
                nodes = nodes[:-1]
            if len(nodes) < 3:
                continue
            results.append({"id": el["id"], "nodes": nodes, "tags": tags})
        print(f"   Nalezeno {len(results)} prvků (tag: {tag}, ways).")
        return results

    @staticmethod
    def _relations_by_tag(
        elements: list[dict],
        way_nodes: dict[int, list],
        tag: str,
    ) -> list[dict]:
        """
        Extract building outlines from relations that carry ``tag``.

        Each outer-ring member way of a matching relation is returned as a
        separate building dict carrying the **relation's** tags.  This is the
        key step that lets us render multipolygon buildings whose outer outline
        is only tagged on the relation, not on the individual way.

        Relations with multiple outer rings (building complexes) produce one
        dict per outer ring, each with a unique negative pseudo-ID so
        deduplication logic elsewhere does not collapse them.
        """
        results = []
        for el in elements:
            if el["type"] != "relation":
                continue
            rtags = el.get("tags", {})
            if tag not in rtags:
                continue

            outer_refs = [
                m["ref"] for m in el.get("members", [])
                if m.get("type") == "way" and m.get("role") in ("outer", "")
            ]

            for idx, ref in enumerate(outer_refs):
                nodes = list(way_nodes.get(ref, []))
                if len(nodes) > 1 and nodes[0] == nodes[-1]:
                    nodes = nodes[:-1]
                if len(nodes) < 3:
                    continue
                # Negative pseudo-ID: unique per outer ring, never conflicts
                # with positive OSM way/relation IDs.
                pseudo_id = -(el["id"] * 1000 + idx)
                results.append({
                    "id":       pseudo_id,
                    "nodes":    nodes,
                    "tags":     rtags,
                })

        print(f"   Nalezeno {len(results)} prvků (tag: {tag}, relations).")
        return results

    @staticmethod
    def _suppress_outer_shells(
        buildings: list[dict],
        parts: list[dict],
    ) -> list[dict]:
        """
        Remove outer building outlines whose footprint already overlaps with
        building:part polygons (OSM S3DB outer-shell suppression).

        When an OSM building is modelled with Simple 3D Buildings parts,
        the outer ``building=yes`` way is a "shadow" footprint that should
        **not** be extruded alongside the detailed parts.

        Detection: a building outline is suppressed when any building:part
        polygon overlaps it with at least 15 % area intersection relative to
        the part's own area.  This is more robust than centroid containment,
        which fails for parts near polygon edges or with unusual shapes.
        """
        if not parts:
            return buildings

        try:
            from shapely.geometry import Polygon
            from shapely.strtree import STRtree
        except ImportError:
            return buildings

        # Build valid part polygons with a spatial index for fast lookup.
        part_polys: list = []
        for p in parts:
            coords = [(n["lon"], n["lat"]) for n in p["nodes"]]
            if len(coords) < 3:
                continue
            try:
                pp = Polygon(coords)
                if not pp.is_valid:
                    pp = pp.buffer(0)
                if not pp.is_empty:
                    part_polys.append(pp)
            except Exception:
                pass

        if not part_polys:
            return buildings

        part_tree = STRtree(part_polys)

        kept = []
        suppressed = 0
        for b in buildings:
            coords = [(n["lon"], n["lat"]) for n in b["nodes"]]
            if len(coords) < 3:
                kept.append(b)
                continue
            try:
                bpoly = Polygon(coords)
                if not bpoly.is_valid:
                    bpoly = bpoly.buffer(0)
                if bpoly.is_empty:
                    kept.append(b)
                    continue

                # Query only candidate parts whose bounding box overlaps
                candidates = part_tree.query(bpoly)
                should_suppress = False
                for idx in candidates:
                    pp = part_polys[idx]
                    inter_area = bpoly.intersection(pp).area
                    if inter_area > 0.15 * pp.area:
                        should_suppress = True
                        break

                if should_suppress:
                    suppressed += 1
                else:
                    kept.append(b)
            except Exception:
                kept.append(b)

        if suppressed:
            print(f"   Potlačeno {suppressed} S3DB obrysů (nahrazeny částmi).")
        return kept

    @staticmethod
    def _parse_ways(elements: list[dict], required_tag: str) -> list[dict]:
        """Legacy helper used by get_buildings() and get_highways()."""
        node_coords: dict[int, dict] = {
            el["id"]: {"lon": el["lon"], "lat": el["lat"]}
            for el in elements
            if el["type"] == "node"
        }

        results = []
        for el in elements:
            if el["type"] != "way":
                continue
            tags = el.get("tags", {})
            if required_tag not in tags:
                continue

            coords = [
                node_coords[nid]
                for nid in el.get("nodes", [])
                if nid in node_coords
            ]

            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]

            if len(coords) < 2:
                continue

            results.append({"id": el["id"], "nodes": coords, "tags": tags})

        print(f"   Nalezeno {len(results)} prvků (tag: {required_tag}).")
        return results


if __name__ == "__main__":
    client = OsmClient()
    bbox = (50.07, 14.43, 50.08, 14.44)
    roads = client.get_highways(bbox)
    for r in roads[:5]:
        print(r["id"], r["tags"].get("highway"), "nodes:", len(r["nodes"]))
