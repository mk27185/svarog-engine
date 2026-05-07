"""
TileSplitter
============
Converts between geographic areas and standard XYZ slippy-map tiles
(Web Mercator, same scheme as OpenStreetMap / Three.js client).

Formulas follow the OSM wiki:
  https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

from .task_runner import TileConfig

EARTH_CIRCUMFERENCE_M = 40_075_016.686   # metres at equator


class XYZ(NamedTuple):
    z: int
    x: int
    y: int


# ── Public API ────────────────────────────────────────────────────────────────

class TileSplitter:

    @staticmethod
    def from_xyz(z: int, x: int, y: int) -> TileConfig:
        """
        Convert a single slippy-map tile (z, x, y) to a TileConfig.

        The tile name is "{z}/{x}/{y}" and the bbox is
        (south, west, north, east) in WGS-84 decimal degrees.
        """
        north, west = _tile_to_latlon(z, x, y)
        south, east = _tile_to_latlon(z, x + 1, y + 1)
        return TileConfig(
            name=f"{z}/{x}/{y}",
            bbox=(south, west, north, east),
            output_dir=None,
            xyz=XYZ(z, x, y),
        )

    @staticmethod
    def xyz_grid(
        bbox:   tuple[float, float, float, float],
        zoom:   int,
    ) -> list[TileConfig]:
        """
        Return all XYZ tiles that cover *bbox* at the given *zoom* level.

        Parameters
        ----------
        bbox : (south, west, north, east) in decimal degrees
        zoom : slippy-map zoom level

        Returns
        -------
        list[TileConfig] ordered row-major (west→east, north→south)
        """
        south, west, north, east = bbox

        x_min, y_min = _latlon_to_tile(north, west, zoom)   # NW corner
        x_max, y_max = _latlon_to_tile(south, east, zoom)   # SE corner

        tiles = []
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                tiles.append(TileSplitter.from_xyz(zoom, x, y))
        return tiles

    @staticmethod
    def tile_size_meters(zoom: int, lat: float = 0.0) -> float:
        """
        Width (and height) of one tile in metres at a given latitude.

        At the equator (lat=0) this equals EARTH_CIRCUMFERENCE / 2^zoom.
        At other latitudes the tile is narrower by cos(lat).
        """
        return EARTH_CIRCUMFERENCE_M / (2 ** zoom) * math.cos(math.radians(lat))

    @staticmethod
    def tile_center_latlon(z: int, x: int, y: int) -> tuple[float, float]:
        """Return (lat, lon) of the centre of tile (z, x, y)."""
        n_lat, w_lon = _tile_to_latlon(z, x, y)
        s_lat, e_lon = _tile_to_latlon(z, x + 1, y + 1)
        return (n_lat + s_lat) / 2, (w_lon + e_lon) / 2

    @staticmethod
    def tile_center_mercator(z: int, x: int, y: int) -> tuple[float, float]:
        """
        Return the tile centre in EPSG:3857 Web Mercator metres (x_m, y_m).
        This is what the Three.js client uses as the tile position origin.
        """
        lat, lon = TileSplitter.tile_center_latlon(z, x, y)
        return _latlon_to_mercator(lat, lon)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """(lat, lon, zoom) → (x, y) tile index."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    # Clamp to valid range
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def _tile_to_latlon(z: int, x: int, y: int) -> tuple[float, float]:
    """Top-left (NW) corner of tile (z, x, y) as (lat, lon)."""
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def _latlon_to_mercator(lat: float, lon: float) -> tuple[float, float]:
    """WGS-84 (lat, lon) → EPSG:3857 (x_m, y_m)."""
    R = 6_378_137.0  # WGS-84 equatorial radius in metres
    x_m = math.radians(lon) * R
    y_m = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R
    return x_m, y_m
