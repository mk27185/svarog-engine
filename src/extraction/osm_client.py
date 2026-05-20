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
    # Landcover (water, green, railways) — single Overpass query
    # ------------------------------------------------------------------

    _GREEN_LANDUSE = frozenset({
        "grass", "forest", "meadow", "vineyard", "orchard", "recreation_ground",
    })
    _GREEN_LEISURE = frozenset({"park", "garden", "nature_reserve"})
    _GREEN_NATURAL = frozenset({"wood", "scrub", "heath", "grassland"})
    _WATERWAY_TAGS = frozenset({"river", "stream", "canal", "drain", "ditch"})

    def get_landcover(self, bbox: tuple) -> dict[str, list]:
        """
        Fetch landcover features in one Overpass request.

        Returns dict with keys:
          water_polygons, waterways, green_polygons, railways
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        query = (
            f"[out:json][timeout:90];\n"
            f"(\n"
            f'  way["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});\n'
            f'  way["landuse"="reservoir"]({min_lat},{min_lon},{max_lat},{max_lon});\n'
            f'  way["water"]({min_lat},{min_lon},{max_lat},{max_lon});\n'
            f'  way["waterway"]({min_lat},{min_lon},{max_lat},{max_lon});\n'
            f'  way["landuse"~"^(grass|forest|meadow|vineyard|orchard|recreation_ground)$"]'
            f"({min_lat},{min_lon},{max_lat},{max_lon});\n"
            f'  way["leisure"~"^(park|garden|nature_reserve)$"]'
            f"({min_lat},{min_lon},{max_lat},{max_lon});\n"
            f'  way["natural"~"^(wood|scrub|heath|grassland)$"]'
            f"({min_lat},{min_lon},{max_lat},{max_lon});\n"
            f'  way["railway"]({min_lat},{min_lon},{max_lat},{max_lon});\n'
            f");\n"
            f"out body;>;out skel qt;"
        )
        elements = self._fetch(query, "landcover")
        return self._parse_landcover(elements)

    def _parse_landcover(self, elements: list[dict]) -> dict[str, list]:
        _, way_nodes = self._build_indices(elements)

        water_polygons: list[dict] = []
        waterways: list[dict] = []
        green_polygons: list[dict] = []
        railways: list[dict] = []

        for el in elements:
            if el["type"] != "way":
                continue
            tags = el.get("tags", {})
            nodes = list(way_nodes.get(el["id"], []))
            if len(nodes) < 2:
                continue
            if len(nodes) > 1 and nodes[0] == nodes[-1]:
                nodes = nodes[:-1]

            item = {"id": el["id"], "nodes": nodes, "tags": tags}

            if tags.get("natural") == "water" or tags.get("landuse") == "reservoir":
                if len(nodes) >= 3:
                    water_polygons.append(item)
                continue
            if tags.get("water") in ("lake", "pond", "basin"):
                if len(nodes) >= 3:
                    water_polygons.append(item)
                continue

            wwy = tags.get("waterway", "")
            if wwy in self._WATERWAY_TAGS:
                if len(nodes) >= 2:
                    waterways.append(item)
                continue

            lu = tags.get("landuse", "")
            if lu in self._GREEN_LANDUSE:
                if len(nodes) >= 3:
                    green_polygons.append(item)
                continue
            le = tags.get("leisure", "")
            if le in self._GREEN_LEISURE:
                if len(nodes) >= 3:
                    green_polygons.append(item)
                continue
            nat = tags.get("natural", "")
            if nat in self._GREEN_NATURAL:
                if len(nodes) >= 3:
                    green_polygons.append(item)
                continue

            if "railway" in tags:
                if len(nodes) >= 2:
                    railways.append(item)

        result = {
            "water_polygons": water_polygons,
            "waterways":      waterways,
            "green_polygons": green_polygons,
            "railways":       railways,
        }
        total = sum(len(v) for v in result.values())
        print(
            f"   Landcover: {total} prvků "
            f"(voda {len(water_polygons)}, řeky {len(waterways)}, "
            f"zelen {len(green_polygons)}, železnice {len(railways)})"
        )
        return result

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
    def _assemble_ring(outer_refs: list[int], way_nodes: dict[int, list]) -> list[dict] | None:
        """
        Chain a list of outer way references into one closed polygon ring.

        OSM multipolygon outer rings are often split across several way
        members that share endpoints (one way's last node = next way's first
        node).  This method stitches them into a single node list.

        Returns the assembled node list (without closing duplicate), or
        ``None`` if the ways cannot be chained into a valid ring (≥ 3 nodes).
        """
        # Collect node lists for each way; strip GeoJSON-style closing node.
        segments: list[list[dict]] = []
        for ref in outer_refs:
            nodes = list(way_nodes.get(ref, []))
            if not nodes:
                continue
            if len(nodes) > 1 and nodes[0] == nodes[-1]:
                nodes = nodes[:-1]
            if len(nodes) >= 2:
                segments.append(nodes)

        if not segments:
            return None
        if len(segments) == 1:
            return segments[0] if len(segments[0]) >= 3 else None

        def _same(a: dict, b: dict) -> bool:
            return abs(a["lon"] - b["lon"]) < 1e-7 and abs(a["lat"] - b["lat"]) < 1e-7

        # Greedy forward-chain from segment[0]; try both ends of candidates.
        chain: list[dict] = list(segments[0])
        used: set[int] = {0}

        while len(used) < len(segments):
            matched = False
            tail = chain[-1]
            head = chain[0]

            for i, seg in enumerate(segments):
                if i in used:
                    continue
                if _same(tail, seg[0]):
                    chain.extend(seg[1:])
                    used.add(i)
                    matched = True
                    break
                if _same(tail, seg[-1]):
                    chain.extend(reversed(seg[:-1]))
                    used.add(i)
                    matched = True
                    break
                if _same(head, seg[-1]):
                    chain = list(seg[:-1]) + chain
                    used.add(i)
                    matched = True
                    break
                if _same(head, seg[0]):
                    chain = list(reversed(seg[1:])) + chain
                    used.add(i)
                    matched = True
                    break

            if not matched:
                break  # remaining segments cannot be connected — give up

        # Only return the assembled ring if ALL segments were consumed.
        # If some could not be chained, the caller falls back to per-way handling.
        if len(used) < len(segments):
            return None
        return chain if len(chain) >= 3 else None

    @staticmethod
    def _relations_by_tag(
        elements: list[dict],
        way_nodes: dict[int, list],
        tag: str,
    ) -> list[dict]:
        """
        Extract building outlines from relations that carry ``tag``.

        Outer rings composed of multiple way members are stitched into one
        closed polygon via ``_assemble_ring``.  This handles the common OSM
        pattern where a building's outer boundary is split across several ways.

        Relations with multiple *disconnected* outer rings (building complexes)
        produce one dict per ring, each with a unique negative pseudo-ID.
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

            if not outer_refs:
                continue

            # Try to assemble all outer ways into one ring first.
            ring = OsmClient._assemble_ring(outer_refs, way_nodes)
            if ring and len(ring) >= 3:
                pseudo_id = -(el["id"] * 1000)
                results.append({
                    "id":    pseudo_id,
                    "nodes": ring,
                    "tags":  rtags,
                })
            else:
                # Fall back: treat each outer way as a separate polygon.
                # This handles building complexes with multiple disconnected
                # outer rings (e.g. a courtyard block).
                for idx, ref in enumerate(outer_refs):
                    nodes = list(way_nodes.get(ref, []))
                    if len(nodes) > 1 and nodes[0] == nodes[-1]:
                        nodes = nodes[:-1]
                    if len(nodes) < 3:
                        continue
                    pseudo_id = -(el["id"] * 1000 + idx)
                    results.append({
                        "id":    pseudo_id,
                        "nodes": nodes,
                        "tags":  rtags,
                    })

        print(f"   Nalezeno {len(results)} prvků (tag: {tag}, relations).")
        return results

    @staticmethod
    def _suppress_outer_shells(
        buildings: list[dict],
        parts: list[dict],
    ) -> list[dict]:
        """
        Remove outer building outlines whose footprint is already fully covered
        by building:part polygons (OSM S3DB outer-shell suppression).

        When an OSM building is modelled with Simple 3D Buildings parts that
        collectively cover its entire footprint, the outer ``building=yes``
        outline is a "shadow" that should not be extruded on top of the parts.

        Detection: a building outline is suppressed only when the *union* of
        all overlapping building:part polygons covers at least 85 % of the
        outline's own area.  This prevents incorrectly suppressing partially-
        modelled landmarks (e.g. a cathedral where only domes and towers are
        drawn as parts but the main body is not), while still suppressing fully
        decomposed S3DB buildings where every square metre is a named part.
        """
        if not parts:
            return buildings

        try:
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
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
                if bpoly.is_empty or bpoly.area == 0:
                    kept.append(b)
                    continue

                # Collect all parts whose bounding box overlaps this shell.
                candidate_idxs = part_tree.query(bpoly)
                if not len(candidate_idxs):
                    kept.append(b)
                    continue

                # Suppress only when the parts UNION covers ≥85 % of the shell.
                # Small landmarks (e.g. a cathedral where only the domes are
                # tagged as parts) must not lose their main body shell.
                candidate_geoms = [part_polys[idx] for idx in candidate_idxs]
                parts_union = unary_union(candidate_geoms)
                coverage = bpoly.intersection(parts_union).area / bpoly.area

                if coverage >= 0.85:
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
