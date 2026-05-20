"""Unit tests for OSM union-batch clipping (no HTTP)."""
from __future__ import annotations

import pytest

from src.extraction.osm_batch_prefetch import (
    ClippingOsmClient,
    all_tiles_have_osm_json_cache,
    clip_buildings_to_bbox,
    clip_highways_to_bbox,
    compute_tile_union_bbox,
    union_bbox_deg2,
)
from src.pipeline.task_runner import TileConfig


def test_union_bbox_deg2():
    assert union_bbox_deg2((49.0, 14.0, 50.0, 15.0)) == pytest.approx(1.0)


def test_compute_tile_union_bbox():
    tiles = [
        TileConfig("a", (49.0, 14.0, 49.5, 14.5)),
        TileConfig("b", (49.4, 14.4, 50.0, 15.0)),
    ]
    assert compute_tile_union_bbox(tiles) == (49.0, 14.0, 50.0, 15.0)


def test_clip_highway_intersects_crossing():
    hw = [{
        "id": 1,
        "nodes": [
            {"lon": 14.0, "lat": 49.0},
            {"lon": 15.0, "lat": 50.0},
        ],
        "tags": {"highway": "primary"},
    }]
    clipped = clip_highways_to_bbox(hw, (49.4, 14.4, 49.6, 14.6))
    assert len(clipped) == 1
    clipped_out = clip_highways_to_bbox(hw, (48.0, 13.0, 48.1, 13.1))
    assert len(clipped_out) == 0


def test_clip_building_intersects():
    bld = [{
        "id": 2,
        "nodes": [
            {"lon": 14.45, "lat": 49.45},
            {"lon": 14.46, "lat": 49.45},
            {"lon": 14.46, "lat": 49.46},
            {"lon": 14.45, "lat": 49.46},
        ],
        "tags": {"building": "yes"},
    }]
    inside = clip_buildings_to_bbox(bld, (49.44, 14.44, 49.47, 14.47))
    assert len(inside) == 1
    outside = clip_buildings_to_bbox(bld, (50.0, 15.0, 50.1, 15.1))
    assert len(outside) == 0


def test_clipping_osm_client_matches_linear_scan():
    highways = []
    for i in range(50):
        lon0 = 14.0 + (i % 10) * 0.01
        lat0 = 50.0 + (i // 10) * 0.01
        highways.append({
            "id": i,
            "nodes": [
                {"lon": lon0, "lat": lat0},
                {"lon": lon0 + 0.02, "lat": lat0 + 0.001},
            ],
            "tags": {"highway": "residential"},
        })
    bld = [{
        "id": 99,
        "nodes": [
            {"lon": 14.43, "lat": 50.07},
            {"lon": 14.44, "lat": 50.07},
            {"lon": 14.44, "lat": 50.08},
        ],
        "tags": {"building": "yes"},
    }]
    bbox = (50.07, 14.43, 50.08, 14.44)
    c = ClippingOsmClient(highways, bld)
    assert {x["id"] for x in c.get_highways(bbox)} == {
        x["id"] for x in clip_highways_to_bbox(highways, bbox)
    }
    assert {x["id"] for x in c.get_buildings_with_parts(bbox)} == {
        x["id"] for x in clip_buildings_to_bbox(bld, bbox)
    }


def test_all_tiles_have_osm_json_cache_false_without_files(tmp_path):
    tiles = [TileConfig("x", (50.0, 14.0, 50.01, 14.01))]
    assert not all_tiles_have_osm_json_cache(tiles, str(tmp_path))


def test_clipping_osm_client_landcover():
    landcover = {
        "water_polygons": [{
            "id": 10,
            "nodes": [
                {"lon": 14.45, "lat": 49.45},
                {"lon": 14.46, "lat": 49.45},
                {"lon": 14.46, "lat": 49.46},
                {"lon": 14.45, "lat": 49.46},
            ],
            "tags": {"natural": "water"},
        }],
        "waterways": [{
            "id": 11,
            "nodes": [
                {"lon": 14.44, "lat": 49.44},
                {"lon": 14.47, "lat": 49.47},
            ],
            "tags": {"waterway": "river"},
        }],
        "green_polygons": [],
        "railways": [],
    }
    bbox = (49.44, 14.44, 49.47, 14.47)
    c = ClippingOsmClient([], [], landcover)
    lc = c.get_landcover(bbox)
    assert len(lc["water_polygons"]) == 1
    assert len(lc["waterways"]) == 1
