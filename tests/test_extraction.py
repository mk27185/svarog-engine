import os
import unittest
from unittest.mock import MagicMock, patch, call

import requests

from src.extraction.opentopography_client import OpenTopographyClient
from src.extraction.osm_client import OsmClient


class TestOpenTopographyClient(unittest.TestCase):
    def setUp(self):
        self.client = OpenTopographyClient()

    def test_prague_small_area(self):
        lat_center = 50.0755
        lon_center = 14.4378
        offset = 0.0018
        bbox = (
            lat_center - offset, lon_center - offset,
            lat_center + offset, lon_center + offset,
        )
        print(f"\nTesting Prague area: {bbox}")
        data = self.client.get_dem(bbox)
        self.assertIsNotNone(data, "Failed to fetch DEM data for Prague area")
        print(f"Retrieved {len(data)} bytes")


class TestOsmClientRetry(unittest.TestCase):
    """Tests for OsmClient retry / backoff logic (no real HTTP calls)."""

    BBOX = (50.07, 14.43, 50.08, 14.44)

    def _ok_response(self, elements: list | None = None) -> MagicMock:
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = {"elements": elements or []}
        r.raise_for_status = MagicMock()
        return r

    def _rate_limit_response(self) -> MagicMock:
        r = MagicMock()
        r.status_code = 429
        r.raise_for_status.side_effect = requests.exceptions.HTTPError("429")
        return r

    @patch("src.extraction.osm_client.time.sleep")
    @patch("src.extraction.osm_client.requests.post")
    def test_success_on_first_attempt(self, mock_post, mock_sleep):
        mock_post.return_value = self._ok_response([
            {"type": "node", "id": 1, "lon": 14.43, "lat": 50.07},
            {"type": "way",  "id": 2, "nodes": [1, 1],
             "tags": {"highway": "primary"}},
        ])
        result = OsmClient().get_highways(self.BBOX)
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()

    @patch("src.extraction.osm_client.time.sleep")
    @patch("src.extraction.osm_client.requests.post")
    def test_retries_on_429_then_succeeds(self, mock_post, mock_sleep):
        """Two 429 responses followed by success → result returned, sleep called twice."""
        elements = [
            {"type": "node", "id": 1, "lon": 14.43, "lat": 50.07},
            {"type": "node", "id": 2, "lon": 14.44, "lat": 50.08},
            {"type": "way",  "id": 3, "nodes": [1, 2],
             "tags": {"highway": "residential"}},
        ]
        mock_post.side_effect = [
            self._rate_limit_response(),
            self._rate_limit_response(),
            self._ok_response(elements),
        ]
        result = OsmClient().get_highways(self.BBOX)
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2
        assert len(result) == 1

    @patch("src.extraction.osm_client.time.sleep")
    @patch("src.extraction.osm_client.requests.post")
    def test_gives_up_after_max_retries(self, mock_post, mock_sleep):
        """Persistent 429 → returns empty list after MAX_RETRIES+1 attempts."""
        client = OsmClient()
        mock_post.return_value = self._rate_limit_response()
        result = client.get_highways(self.BBOX)
        assert result == []
        assert mock_post.call_count == client.MAX_RETRIES + 1
        assert mock_sleep.call_count == client.MAX_RETRIES

    @patch("src.extraction.osm_client.time.sleep")
    @patch("src.extraction.osm_client.requests.post")
    def test_network_error_retries_then_gives_up(self, mock_post, mock_sleep):
        """Network-level exceptions also trigger retry."""
        client = OsmClient()
        mock_post.side_effect = requests.exceptions.ConnectionError("timeout")
        result = client.get_highways(self.BBOX)
        assert result == []
        assert mock_post.call_count == client.MAX_RETRIES + 1

    @patch("src.extraction.osm_client.time.sleep")
    @patch("src.extraction.osm_client.requests.post")
    def test_http_error_not_429_returns_empty(self, mock_post, mock_sleep):
        """Non-429 HTTP errors (e.g. 500) return empty immediately without retry."""
        r = MagicMock()
        r.status_code = 500
        r.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        mock_post.return_value = r
        result = OsmClient().get_highways(self.BBOX)
        assert result == []
        assert mock_post.call_count == 1   # no retry for non-429
        mock_sleep.assert_not_called()

    def test_backoff_increases_exponentially(self):
        client = OsmClient()
        waits = [client._backoff(i) / (1.2) for i in range(4)]   # divide out max jitter
        # Each wait should be roughly double the previous (within jitter tolerance)
        for i in range(1, len(waits)):
            assert waits[i] > waits[i - 1], \
                f"Backoff did not increase: {waits}"


class TestOsmClientBuildingParsing(unittest.TestCase):
    """Unit tests for relation / outer-shell parsing helpers."""

    # ------------------------------------------------------------------
    # Shared synthetic elements
    # ------------------------------------------------------------------

    def _node(self, nid, lon, lat):
        return {"type": "node", "id": nid, "lon": lon, "lat": lat}

    def _way(self, wid, node_ids, tags=None):
        return {"type": "way", "id": wid, "nodes": node_ids, "tags": tags or {}}

    def _relation(self, rid, members, tags=None):
        return {"type": "relation", "id": rid, "members": members, "tags": tags or {}}

    def _make_square_nodes(self, base_id, lon0, lat0, size=0.001):
        """Four nodes forming a small square."""
        return [
            self._node(base_id,     lon0,        lat0),
            self._node(base_id + 1, lon0 + size, lat0),
            self._node(base_id + 2, lon0 + size, lat0 + size),
            self._node(base_id + 3, lon0,        lat0 + size),
            self._node(base_id,     lon0,        lat0),  # closing duplicate
        ]

    # ------------------------------------------------------------------
    # _build_indices
    # ------------------------------------------------------------------

    def test_build_indices_nodes(self):
        nodes = [
            self._node(1, 14.43, 50.07),
            self._node(2, 14.44, 50.08),
        ]
        _, way_nodes = OsmClient._build_indices(nodes)
        assert way_nodes == {}

    def test_build_indices_way(self):
        elements = [
            self._node(1, 14.43, 50.07),
            self._node(2, 14.44, 50.07),
            self._node(3, 14.44, 50.08),
            self._way(10, [1, 2, 3], {}),
        ]
        _, way_nodes = OsmClient._build_indices(elements)
        assert 10 in way_nodes
        assert len(way_nodes[10]) == 3

    # ------------------------------------------------------------------
    # _ways_by_tag
    # ------------------------------------------------------------------

    def test_ways_by_tag_building_part(self):
        nodes = self._make_square_nodes(1, 14.430, 50.070)
        way = self._way(100, [1, 2, 3, 4, 1], {"building:part": "yes", "height": "6"})
        elements = nodes + [way]
        _, wn = OsmClient._build_indices(elements)
        result = OsmClient._ways_by_tag(elements, wn, "building:part")
        assert len(result) == 1
        assert result[0]["id"] == 100
        assert result[0]["tags"]["height"] == "6"

    def test_ways_by_tag_closing_node_removed(self):
        """First == last node should be stripped."""
        elements = [
            self._node(1, 14.43, 50.07),
            self._node(2, 14.44, 50.07),
            self._node(3, 14.44, 50.08),
            self._way(10, [1, 2, 3, 1], {"building": "yes"}),
        ]
        _, wn = OsmClient._build_indices(elements)
        result = OsmClient._ways_by_tag(elements, wn, "building")
        assert len(result[0]["nodes"]) == 3  # closing duplicate removed

    # ------------------------------------------------------------------
    # _relations_by_tag
    # ------------------------------------------------------------------

    def test_relations_by_tag_single_outer(self):
        """A relation with one outer way produces one building dict."""
        nodes = [
            self._node(1, 14.43, 50.07),
            self._node(2, 14.44, 50.07),
            self._node(3, 14.44, 50.08),
        ]
        way = self._way(50, [1, 2, 3], {})   # outer way, no building tag itself
        rel = self._relation(
            999,
            [{"type": "way", "ref": 50, "role": "outer"}],
            {"building": "yes", "building:levels": "4"},
        )
        elements = nodes + [way, rel]
        _, wn = OsmClient._build_indices(elements)
        result = OsmClient._relations_by_tag(elements, wn, "building")
        assert len(result) == 1
        assert result[0]["tags"]["building:levels"] == "4"
        # Negative pseudo-ID, unique per outer ring
        assert result[0]["id"] < 0

    def test_relations_by_tag_multiple_outer_rings(self):
        """A relation with two outer ways → two building dicts with distinct IDs."""
        nodes = [
            self._node(1, 14.43, 50.07), self._node(2, 14.44, 50.07),
            self._node(3, 14.44, 50.08),
            self._node(4, 14.45, 50.07), self._node(5, 14.46, 50.07),
            self._node(6, 14.46, 50.08),
        ]
        way_a = self._way(51, [1, 2, 3], {})
        way_b = self._way(52, [4, 5, 6], {})
        rel = self._relation(
            1000,
            [
                {"type": "way", "ref": 51, "role": "outer"},
                {"type": "way", "ref": 52, "role": "outer"},
            ],
            {"building": "yes"},
        )
        elements = nodes + [way_a, way_b, rel]
        _, wn = OsmClient._build_indices(elements)
        result = OsmClient._relations_by_tag(elements, wn, "building")
        assert len(result) == 2
        assert result[0]["id"] != result[1]["id"]   # unique pseudo-IDs

    def test_relations_by_tag_inner_ring_ignored(self):
        """Inner ring members must not be returned as building outlines."""
        nodes = [
            self._node(1, 14.43, 50.07), self._node(2, 14.44, 50.07),
            self._node(3, 14.44, 50.08),
            self._node(4, 14.435, 50.071), self._node(5, 14.436, 50.071),
            self._node(6, 14.436, 50.075),
        ]
        outer_way = self._way(51, [1, 2, 3], {})
        inner_way = self._way(52, [4, 5, 6], {})
        rel = self._relation(
            1001,
            [
                {"type": "way", "ref": 51, "role": "outer"},
                {"type": "way", "ref": 52, "role": "inner"},
            ],
            {"building": "yes"},
        )
        elements = nodes + [outer_way, inner_way, rel]
        _, wn = OsmClient._build_indices(elements)
        result = OsmClient._relations_by_tag(elements, wn, "building")
        assert len(result) == 1   # inner ring not included

    # ------------------------------------------------------------------
    # _suppress_outer_shells
    # ------------------------------------------------------------------

    def _building_dict(self, bld_id, lon0, lat0, size=0.002):
        """Helper: building dict with a square polygon."""
        return {
            "id": bld_id,
            "nodes": [
                {"lon": lon0,        "lat": lat0},
                {"lon": lon0 + size, "lat": lat0},
                {"lon": lon0 + size, "lat": lat0 + size},
                {"lon": lon0,        "lat": lat0 + size},
            ],
            "tags": {"building": "yes"},
        }

    def _part_dict(self, part_id, lon0, lat0, size=0.001):
        """Helper: building:part dict centred inside the building above."""
        offset = size * 0.1
        return {
            "id": part_id,
            "nodes": [
                {"lon": lon0 + offset,        "lat": lat0 + offset},
                {"lon": lon0 + offset + size,  "lat": lat0 + offset},
                {"lon": lon0 + offset + size,  "lat": lat0 + offset + size},
                {"lon": lon0 + offset,         "lat": lat0 + offset + size},
            ],
            "tags": {"building:part": "yes"},
        }

    @staticmethod
    def _covering_part(part_id: int, lon0: float, lat0: float, size: float) -> dict:
        """Part that exactly matches a _building_dict polygon (100 % coverage)."""
        return {
            "id": part_id,
            "nodes": [
                {"lon": lon0,          "lat": lat0},
                {"lon": lon0 + size,   "lat": lat0},
                {"lon": lon0 + size,   "lat": lat0 + size},
                {"lon": lon0,          "lat": lat0 + size},
            ],
            "tags": {"building:part": "yes"},
        }

    def test_suppress_removes_covered_shell(self):
        """Shell fully covered (100 %) by one part → suppressed."""
        building = self._building_dict(1, 14.43, 50.07, size=0.002)
        part = self._covering_part(2, 14.43, 50.07, size=0.002)
        result = OsmClient._suppress_outer_shells([building], [part])
        assert result == []   # outer shell suppressed

    def test_suppress_partial_model_kept(self):
        """Shell with only a small decorative part (≤25 % coverage) is kept.

        Real-world case: landmark where only a dome/tower is a building:part
        but the main body is not.  The outer shell must stay visible.
        """
        building = self._building_dict(1, 14.43, 50.07, size=0.002)
        # Small dome part — offset inside building, covers only ~25 % of area.
        part = self._part_dict(2, 14.43, 50.07, size=0.001)
        result = OsmClient._suppress_outer_shells([building], [part])
        assert len(result) == 1   # outer shell kept

    def test_suppress_keeps_uncovered_building(self):
        building = self._building_dict(1, 14.43, 50.07, size=0.002)
        # Part is far away – no overlap with the building
        part = self._part_dict(2, 14.50, 50.10, size=0.001)
        result = OsmClient._suppress_outer_shells([building], [part])
        assert len(result) == 1   # building kept

    def test_suppress_no_parts_returns_all(self):
        buildings = [self._building_dict(i, 14.43 + i * 0.01, 50.07) for i in range(3)]
        result = OsmClient._suppress_outer_shells(buildings, [])
        assert len(result) == 3

    def test_suppress_mixed(self):
        """Building A fully covered by its part → suppressed; B without part → kept."""
        bld_a = self._building_dict(1, 14.43, 50.07, size=0.002)
        bld_b = self._building_dict(2, 14.50, 50.07, size=0.002)
        # Part exactly covers A (100 % coverage).
        part = self._covering_part(3, 14.43, 50.07, size=0.002)
        result = OsmClient._suppress_outer_shells([bld_a, bld_b], [part])
        ids = [b["id"] for b in result]
        assert 1 not in ids   # A suppressed
        assert 2 in ids       # B kept


if __name__ == "__main__":
    unittest.main()
