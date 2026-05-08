import math
import pytest
from src.pipeline.tile_splitter import TileSplitter, XYZ, _latlon_to_mercator


class TestTileSplitter:

    def test_from_xyz_name(self):
        tile = TileSplitter.from_xyz(15, 17898, 11245)
        assert tile.name == "15/17898/11245"
        assert tile.xyz == XYZ(15, 17898, 11245)

    def test_from_xyz_bbox_ordering(self):
        tile = TileSplitter.from_xyz(15, 17898, 11245)
        south, west, north, east = tile.bbox
        assert south < north
        assert west < east

    def test_from_xyz_bbox_values_reasonable(self):
        # Prague is around 50°N, 14°E
        tile = TileSplitter.from_xyz(15, 17698, 11100)
        south, west, north, east = tile.bbox
        assert 49.0 < south < 51.5
        assert 13.0 < west < 16.0

    def test_xyz_grid_covers_bbox(self):
        bbox = (50.07, 14.43, 50.09, 14.46)
        # tile_size_m=720 → zoom 15 at Prague latitude
        tiles = TileSplitter.xyz_grid(bbox, tile_size_m=720)
        assert len(tiles) >= 1
        for t in tiles:
            assert t.xyz is not None

    def test_xyz_grid_all_different(self):
        tiles = TileSplitter.xyz_grid((50.07, 14.43, 50.09, 14.46), tile_size_m=720)
        names = [t.name for t in tiles]
        assert len(names) == len(set(names))

    def test_zoom_for_tile_size_prague(self):
        # 720 m at Prague (lat≈50°) → zoom 15
        zoom = TileSplitter.zoom_for_tile_size(720, lat=50.0)
        assert zoom == 15

    def test_zoom_for_tile_size_small(self):
        # 90 m at Prague → zoom 18 (or close)
        zoom = TileSplitter.zoom_for_tile_size(90, lat=50.0)
        assert 17 <= zoom <= 19

    def test_tile_size_at_equator(self):
        sz = TileSplitter.tile_size_meters(zoom=0, lat=0)
        assert abs(sz - 40_075_016.686) < 1.0

    def test_tile_size_decreases_with_zoom(self):
        sz15 = TileSplitter.tile_size_meters(zoom=15, lat=50)
        sz18 = TileSplitter.tile_size_meters(zoom=18, lat=50)
        assert sz18 < sz15
        assert abs(sz15 / sz18 - 8) < 0.01   # 2^(18-15) = 8

    def test_tile_size_prague_zoom15(self):
        sz = TileSplitter.tile_size_meters(zoom=15, lat=50.07)
        assert 700 < sz < 850

    def test_tile_size_prague_zoom18_approx_100m(self):
        sz = TileSplitter.tile_size_meters(zoom=18, lat=50.07)
        assert 80 < sz < 120

    def test_tile_center_inside_bbox(self):
        tile = TileSplitter.from_xyz(15, 17698, 11100)
        s, w, n, e = tile.bbox
        lat, lon = TileSplitter.tile_center_latlon(15, 17698, 11100)
        assert s < lat < n
        assert w < lon < e

    def test_tile_center_mercator_reasonable(self):
        cx, cy = TileSplitter.tile_center_mercator(15, 17698, 11100)
        # Prague in EPSG:3857: roughly x=1600000, y=6500000
        assert 1_400_000 < cx < 1_900_000
        assert 6_000_000 < cy < 7_000_000

    def test_latlon_to_mercator_equator(self):
        x, y = _latlon_to_mercator(0.0, 0.0)
        assert abs(x) < 1.0
        assert abs(y) < 1.0

    def test_latlon_to_mercator_lon180(self):
        x, _ = _latlon_to_mercator(0.0, 180.0)
        assert abs(x - math.radians(180) * 6_378_137) < 1.0
