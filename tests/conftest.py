import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.conversion.geo_utils import build_meta, to_local


@pytest.fixture
def tmp_output():
    """Temporary directory for output files."""
    with tempfile.TemporaryDirectory() as td:
        yield td


# ── Synthetic terrain grid + meta ────────────────────────────────────────

@pytest.fixture
def z_grid():
    """Small synthetic elevation grid (11×11 vertices = 10×10 cells)."""
    y, x = np.mgrid[0:11, 0:11]
    return np.sin(x * 0.5) * np.cos(y * 0.5) * 10.0 + 200.0


@pytest.fixture
def meta(z_grid):
    """Matching meta dict for an 10×10 cell grid at 30 m resolution."""
    from affine import Affine
    transform = Affine.translation(14.43, 50.07) * Affine.scale(0.00037, -0.00027)
    bounds_mock = MagicMock()
    bounds_mock.left   = 14.43
    bounds_mock.right  = 14.43 + 0.00037 * 10
    bounds_mock.top    = 50.07
    bounds_mock.bottom = 50.07 - 0.00027 * 10
    return build_meta(bounds_mock, transform)


@pytest.fixture
def simple_highways(meta):
    """Two crossing roads near the center of the grid."""
    lon0, lat0 = meta["lon_origin"], meta["lat_origin"]
    mpdl = meta["meters_per_deg_lat"]
    mpdlon = meta["meters_per_deg_lon"]

    return [
        {
            "type": "way",
            "nodes": [
                {"lon": lon0 + 100 / mpdlon, "lat": lat0 + 150 / mpdl},
                {"lon": lon0 + 250 / mpdlon, "lat": lat0 + 150 / mpdl},
            ],
            "tags": {"highway": "primary"},
        },
        {
            "type": "way",
            "nodes": [
                {"lon": lon0 + 175 / mpdlon, "lat": lat0 + 100 / mpdl},
                {"lon": lon0 + 175 / mpdlon, "lat": lat0 + 200 / mpdl},
            ],
            "tags": {"highway": "secondary"},
        },
    ]


@pytest.fixture
def simple_buildings(meta):
    """Two simple rectangular building footprints."""
    lon0, lat0 = meta["lon_origin"], meta["lat_origin"]
    mpdl = meta["meters_per_deg_lat"]
    mpdlon = meta["meters_per_deg_lon"]

    return [
        {
            "type": "way",
            "nodes": [
                {"lon": lon0 + 50 / mpdlon,  "lat": lat0 + 50 / mpdl},
                {"lon": lon0 + 70 / mpdlon,  "lat": lat0 + 50 / mpdl},
                {"lon": lon0 + 70 / mpdlon,  "lat": lat0 + 70 / mpdl},
                {"lon": lon0 + 50 / mpdlon,  "lat": lat0 + 70 / mpdl},
            ],
            "tags": {"building": "yes", "height": "12"},
        },
        {
            "type": "way",
            "nodes": [
                {"lon": lon0 + 200 / mpdlon, "lat": lat0 + 50 / mpdl},
                {"lon": lon0 + 230 / mpdlon, "lat": lat0 + 50 / mpdl},
                {"lon": lon0 + 230 / mpdlon, "lat": lat0 + 80 / mpdl},
                {"lon": lon0 + 200 / mpdlon, "lat": lat0 + 80 / mpdl},
            ],
            "tags": {"building": "yes", "building:levels": "3"},
        },
    ]


# ── Mock clients ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_opentopography(tmp_output):
    """Mock OpenTopographyClient that writes a synthetic TIF."""
    import rasterio
    from rasterio.transform import from_bounds

    tif_path = os.path.join(tmp_output, "dem.tif")

    # Write a minimal valid GeoTIFF
    arr = np.sin(np.arange(30) * 0.5) * np.cos(np.arange(30)[:, None] * 0.5) * 10 + 200
    with rasterio.open(
        tif_path, "w", driver="GTiff",
        height=30, width=30, count=1, dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(14.43, 50.07, 14.44, 50.08, 30, 30),
    ) as dst:
        dst.write(arr.astype(np.float32), 1)

    client = MagicMock()
    client.get_dem.return_value = tif_path
    return client


@pytest.fixture
def mock_osm_client(simple_highways, simple_buildings):
    client = MagicMock()
    client.get_highways.return_value = simple_highways
    client.get_buildings.return_value = simple_buildings
    return client
