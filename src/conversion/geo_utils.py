"""
Shared coordinate helpers used across conversion modules.

Local metric coordinate system:
    origin  = (bounds.left, bounds.bottom) of the terrain TIF
    X       = eastward metres
    Y       = northward metres
    Z       = elevation metres
"""
import math
import numpy as np

# Minimum segment length to subdivide (metres)
_MIN_SEGMENT = 1e-6


def smooth_profile(zs: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Smooth a 1-D elevation profile using a uniform filter.

    Profiles shorter than 3 points are returned unchanged.  The effective
    window is clamped to ``len(zs)`` so edge artefacts are avoided.
    """
    if len(zs) < 3:
        return zs
    w = min(max(1, window), len(zs))
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(zs, size=w, mode="nearest")


def subdivide_polyline(local_nodes: list[tuple], step: float) -> list[tuple]:
    """
    Subdivide a polyline so that consecutive points are at most ``step``
    metres apart.

    Segments shorter than ``_MIN_SEGMENT`` are skipped (the start point is
    omitted; the end point is always emitted via the final append).
    """
    pts: list[tuple] = []
    for i in range(len(local_nodes) - 1):
        x0, y0 = local_nodes[i]
        x1, y1 = local_nodes[i + 1]
        L = math.hypot(x1 - x0, y1 - y0)
        if L < _MIN_SEGMENT:
            continue
        n = max(1, math.ceil(L / step))
        for j in range(n):
            t = j / n
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    if local_nodes:
        pts.append(local_nodes[-1])
    return pts


def build_meta(bounds, transform) -> dict:
    """Build the shared grid metadata dict from rasterio bounds + transform."""
    lat_mean_rad = math.radians((bounds.bottom + bounds.top) / 2)
    mpd_lat = 111_320.0
    mpd_lon = 111_320.0 * math.cos(lat_mean_rad)

    pixel_w_deg = abs(transform.a)   # degrees per pixel (col)
    pixel_h_deg = abs(transform.e)   # degrees per pixel (row)  [transform.e is negative]

    h_cells = round((bounds.top - bounds.bottom) / pixel_h_deg)
    w_cells = round((bounds.right - bounds.left) / pixel_w_deg)

    return {
        "h": h_cells,
        "w": w_cells,
        "bounds": bounds,
        "transform": transform,
        "lat_origin": bounds.bottom,
        "lon_origin": bounds.left,
        "lat_mean_rad": lat_mean_rad,
        "meters_per_deg_lat": mpd_lat,
        "meters_per_deg_lon": mpd_lon,
        "cell_width_m":  pixel_w_deg * mpd_lon,
        "cell_height_m": pixel_h_deg * mpd_lat,
        "total_width_m":  w_cells * pixel_w_deg * mpd_lon,
        "total_height_m": h_cells * pixel_h_deg * mpd_lat,
    }


def to_local(lon: float, lat: float, meta: dict) -> tuple[float, float]:
    """Convert (lon, lat) → local metric (x, y)."""
    x = (lon - meta["lon_origin"]) * meta["meters_per_deg_lon"]
    y = (lat - meta["lat_origin"]) * meta["meters_per_deg_lat"]
    return x, y


def from_local(x: float, y: float, meta: dict) -> tuple[float, float]:
    """Convert local metric (x, y) → (lon, lat)."""
    lon = meta["lon_origin"] + x / meta["meters_per_deg_lon"]
    lat = meta["lat_origin"] + y / meta["meters_per_deg_lat"]
    return lon, lat


def sample_z(x: float, y: float, z_grid: np.ndarray, meta: dict) -> float:
    """
    Bilinear interpolation of elevation at local metric (x, y) from z_grid.
    z_grid is (h+1) × (w+1): row 0 = north edge, row h = south edge.
    """
    h, w = meta["h"], meta["w"]
    c_f = x / meta["cell_width_m"]
    r_f = h - y / meta["cell_height_m"]

    r0 = int(r_f)
    c0 = int(c_f)
    r1 = r0 + 1
    c1 = c0 + 1

    r0 = max(0, min(r0, h))
    r1 = max(0, min(r1, h))
    c0 = max(0, min(c0, w))
    c1 = max(0, min(c1, w))

    dr = max(0.0, min(1.0, r_f - int(r_f)))
    dc = max(0.0, min(1.0, c_f - int(c_f)))

    return float(
        z_grid[r0, c0] * (1 - dr) * (1 - dc)
        + z_grid[r0, c1] * (1 - dr) * dc
        + z_grid[r1, c0] * dr * (1 - dc)
        + z_grid[r1, c1] * dr * dc
    )
