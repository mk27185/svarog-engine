import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "svarog-engine/0.1", "Accept": "*/*"}


class OsmClient:
    """Downloads OSM features for a bounding box via Overpass API."""

    # ------------------------------------------------------------------
    # Buildings
    # ------------------------------------------------------------------

    def get_buildings(self, bbox: tuple) -> list[dict]:
        """
        Returns building footprints inside bbox.
        bbox: (min_lat, min_lon, max_lat, max_lon)
        Each dict: {'id', 'nodes': [(lon, lat), ...], 'tags'}
        """
        elements = self._query(bbox, tag="building", label="budov")
        return self._parse_ways(elements, required_tag="building")

    # ------------------------------------------------------------------
    # Highways
    # ------------------------------------------------------------------

    def get_highways(self, bbox: tuple) -> list[dict]:
        """
        Returns road/path polylines inside bbox.
        bbox: (min_lat, min_lon, max_lat, max_lon)
        Each dict: {'id', 'nodes': [(lon, lat), ...], 'tags'}
        """
        elements = self._query(bbox, tag="highway", label="silnic")
        return self._parse_ways(elements, required_tag="highway")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _query(self, bbox: tuple, tag: str, label: str) -> list[dict]:
        min_lat, min_lon, max_lat, max_lon = bbox
        query = (
            f"[out:json][timeout:60];\n"
            f"(way[\"{tag}\"]({min_lat},{min_lon},{max_lat},{max_lon}););\n"
            f"out body;>;out skel qt;"
        )
        print(f"   Stahuji OSM {label} z Overpass API pro bbox {bbox}...")
        try:
            resp = requests.post(
                OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=90
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"   Chyba Overpass API: {e}")
            return []
        return resp.json().get("elements", [])

    @staticmethod
    def _parse_ways(elements: list[dict], required_tag: str) -> list[dict]:
        node_coords: dict[int, tuple[float, float]] = {
            el["id"]: (el["lon"], el["lat"])
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

            # Remove closing duplicate for polygons
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
