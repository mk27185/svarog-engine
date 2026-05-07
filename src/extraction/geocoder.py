"""
Geocoder
========
Converts place names to bounding boxes via the Nominatim API (OpenStreetMap).
Free, no API key required.

Usage
-----
    bbox = Geocoder.city_to_bbox("Prague")
    # → (49.94, 14.22, 50.18, 14.71)  (south, west, north, east)
"""
from __future__ import annotations

import urllib.parse
import urllib.request
import json

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT    = "svarog-engine/1.0 (terrain tile generator)"


class GeocoderError(Exception):
    pass


class Geocoder:

    @staticmethod
    def city_to_bbox(
        name:       str,
        *,
        timeout:    float = 10.0,
        result_idx: int   = 0,
    ) -> tuple[float, float, float, float]:
        """
        Look up *name* via Nominatim and return
        (south, west, north, east) in decimal degrees.

        Parameters
        ----------
        name       : place name, e.g. "Prague", "Prague center", "Hradčany"
        timeout    : HTTP timeout in seconds
        result_idx : which Nominatim result to use (0 = best match)

        Raises
        ------
        GeocoderError : if the request fails or no result is found
        """
        params = urllib.parse.urlencode({
            "q":       name,
            "format":  "json",
            "limit":   max(1, result_idx + 1),
            "addressdetails": 0,
        })
        url = f"{_NOMINATIM_URL}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            raise GeocoderError(f"Nominatim request failed: {exc}") from exc

        if not data:
            raise GeocoderError(f"No results for '{name}'.")

        if result_idx >= len(data):
            raise GeocoderError(
                f"result_idx={result_idx} out of range "
                f"(got {len(data)} result(s) for '{name}')."
            )

        item    = data[result_idx]
        bb      = item.get("boundingbox")
        if not bb or len(bb) < 4:
            raise GeocoderError(
                f"Nominatim result for '{name}' has no bounding box."
            )

        # Nominatim returns [south, north, west, east] as strings
        south = float(bb[0])
        north = float(bb[1])
        west  = float(bb[2])
        east  = float(bb[3])
        return south, west, north, east

    @staticmethod
    def city_to_point(
        name:    str,
        *,
        timeout: float = 10.0,
    ) -> tuple[float, float]:
        """
        Return (lat, lon) of the best-match centroid for *name*.
        Useful for centering a manual bbox.
        """
        params = urllib.parse.urlencode({
            "q":      name,
            "format": "json",
            "limit":  1,
        })
        url = f"{_NOMINATIM_URL}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            raise GeocoderError(f"Nominatim request failed: {exc}") from exc

        if not data:
            raise GeocoderError(f"No results for '{name}'.")

        return float(data[0]["lat"]), float(data[0]["lon"])
