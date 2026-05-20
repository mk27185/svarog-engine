"""
TileManifest
============
Reads, writes and validates tile-manifest.json files against the shared
schema in svarog-contracts.

Schema location is resolved in this order:
  1. SVAROG_CONTRACTS_DIR environment variable
  2. Sibling directory  ../svarog-contracts  (relative to this repo root)
  3. Fallback: inline minimal schema (so the engine works without the
     contracts repo present, e.g. in CI that only checks out svarog-engine)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema

# ── Schema resolution ─────────────────────────────────────────────────────────

_SCHEMA_REL_PATH = Path("schemas") / "tile-manifest.json"


def _find_schema() -> dict:
    candidates = []

    env = os.environ.get("SVAROG_CONTRACTS_DIR")
    if env:
        candidates.append(Path(env) / _SCHEMA_REL_PATH)

    # Repo root is three levels up from this file: src/pipeline/tile_manifest.py
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root.parent / "svarog-contracts" / _SCHEMA_REL_PATH)

    for path in candidates:
        if path.exists():
            with open(path) as f:
                return json.load(f)

    # Inline fallback — minimal subset so basic validation still works
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["version", "zoom", "layers"],
        "properties": {
            "version": {"type": "string"},
            "zoom":    {"type": "integer", "minimum": 0},
            "layers":  {"type": "array"},
            "tiles":   {"type": "array"},
        },
    }


# ── Public API ────────────────────────────────────────────────────────────────

class TileManifest:
    """
    Manage a tile-manifest.json document.

    Usage
    -----
    Build programmatically:
        manifest = TileManifest(version="1.0.0", zoom=14)
        manifest.add_tile("prague_center", bbox=(50.07, 14.43, 50.08, 14.44),
                          status="completed", outputs={...})
        manifest.write("outputs/tile-manifest.json")

    Load & validate an existing file:
        manifest = TileManifest.load("outputs/tile-manifest.json")
    """

    SCHEMA: dict = _find_schema()

    def __init__(
        self,
        version: str = "1.0.0",
        zoom:    int = 14,
        layers:  list[dict] | None = None,
    ):
        self._data: dict[str, Any] = {
            "version": version,
            "zoom":    zoom,
            "layers":  layers or [],
            "tiles":   [],
        }

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_tile(
        self,
        name:         str,
        bbox:         tuple[float, float, float, float] | list[float],
        status:       str,
        outputs:      dict[str, str | None] | None = None,
        error:        str | None = None,
        generated_at: str | None = None,
    ) -> None:
        """
        Append a tile record.

        Parameters
        ----------
        name    : tile identifier (used as file prefix)
        bbox    : [south, west, north, east] decimal degrees
        status  : "pending" | "running" | "completed" | "failed"
        outputs : dict with keys terrain/roads/buildings/sdf_texture (or None)
        error   : error message (required when status="failed")
        generated_at : ISO-8601 string; defaults to current UTC time
        """
        record: dict[str, Any] = {
            "name":   name,
            "bbox":   list(bbox),
            "status": status,
        }
        if outputs is not None:
            record["outputs"] = outputs
        if error is not None:
            record["error"] = error
        record["generated_at"] = generated_at or _utc_now()

        self._data["tiles"].append(record)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Raise jsonschema.ValidationError if the document is invalid."""
        jsonschema.validate(instance=self._data, schema=self.SCHEMA)

    def write(self, path: str | Path, *, validate: bool = True) -> Path:
        """Write manifest to *path* (creates parent dirs). Returns resolved path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if validate:
            self.validate()
        with open(path, "w") as f:
            json.dump(self._data, f, indent=2)
        return path.resolve()

    @classmethod
    def load(cls, path: str | Path) -> "TileManifest":
        """Load an existing manifest file, validate it and return a TileManifest."""
        with open(path) as f:
            data = json.load(f)
        jsonschema.validate(instance=data, schema=cls.SCHEMA)
        obj = cls(
            version=data.get("version", "1.0.0"),
            zoom=data.get("zoom", 14),
            layers=data.get("layers", []),
        )
        obj._data["tiles"] = data.get("tiles", [])
        return obj

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def tiles(self) -> list[dict]:
        return list(self._data["tiles"])

    def as_dict(self) -> dict:
        return dict(self._data)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── TilesetDescriptor ─────────────────────────────────────────────────────────

class TilesetDescriptor:
    """
    Writes a tiny (~500 B) TileJSON-like ``tileset.json`` for the whole tileset.

    Unlike TileManifest, this file does **not** list individual tiles — its size
    is constant regardless of how many tiles exist.  Clients derive tile URLs
    from the XYZ convention and the bounds/zoom fields.

    Per-tile metadata (elev_min, elev_max, has_sdf) is embedded in each GLB
    file via scene.extras["svarog"] by GltfExporter, so clients read it after
    loading the GLB rather than from a manifest.

    Output format (TileJSON 3.0.0 + svarog extras)::

        {
          "tilejson": "3.0.0",
          "tiles":    ["{z}/{x}/{y}/{y}.glb"],
          "minzoom":  14,
          "maxzoom":  14,
          "bounds":   [lon_w, lat_s, lon_e, lat_n],
          "center":   [lon_center, lat_center, zoom],
          "extras": {
            "svarog_version": "1.0.0",
            "has_sdf":        true,
            "elev_min":       220.0,
            "elev_max":       310.0
          }
        }

    bounds follow the TileJSON convention: [west, south, east, north] in WGS84.
    """

    def __init__(self, zoom: int, version: str = "1.0.0"):
        self.zoom    = zoom
        self.version = version
        # Per-tile accumulators
        self._bounds:     list[list[float]] = []  # each: [lon_w, lat_s, lon_e, lat_n]
        self._elev_mins:  list[float]       = []
        self._elev_maxs:  list[float]       = []
        self._any_sdf:        bool = False
        self._any_landcover: bool = False
        self._any_navmesh:   bool = False

    def add_tile(
        self,
        bbox:     tuple[float, float, float, float],
        elev_min: float | None = None,
        elev_max: float | None = None,
        has_sdf:  bool = False,
        has_landcover: bool = False,
        has_navmesh:   bool = False,
    ) -> None:
        """
        Register one tile.

        Parameters
        ----------
        bbox     : (lat_s, lon_w, lat_n, lon_e) — same order as TileManifest /
                   TileConfig, i.e. south, west, north, east.
        elev_min : minimum terrain elevation in metres for this tile.
        elev_max : maximum terrain elevation in metres for this tile.
        has_sdf  : True when a _roads_sdf.png exists for this tile.
        """
        lat_s, lon_w, lat_n, lon_e = bbox
        self._bounds.append([lon_w, lat_s, lon_e, lat_n])
        if elev_min is not None:
            self._elev_mins.append(elev_min)
        if elev_max is not None:
            self._elev_maxs.append(elev_max)
        if has_sdf:
            self._any_sdf = True
        if has_landcover:
            self._any_landcover = True
        if has_navmesh:
            self._any_navmesh = True

    def write(self, path: str | Path) -> Path:
        """Write tileset.json to *path* (creates parent dirs). Returns resolved path."""
        if not self._bounds:
            raise ValueError("TilesetDescriptor: no tiles were added before write().")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lon_w = min(b[0] for b in self._bounds)
        lat_s = min(b[1] for b in self._bounds)
        lon_e = max(b[2] for b in self._bounds)
        lat_n = max(b[3] for b in self._bounds)
        lon_c = (lon_w + lon_e) / 2
        lat_c = (lat_s + lat_n) / 2

        extras: dict[str, Any] = {
            "svarog_version": self.version,
            "has_sdf":        self._any_sdf,
            "has_landcover":  self._any_landcover,
            "has_navmesh":    self._any_navmesh,
            "elev_min":       round(min(self._elev_mins), 2) if self._elev_mins else None,
            "elev_max":       round(max(self._elev_maxs), 2) if self._elev_maxs else None,
        }

        doc: dict[str, Any] = {
            "tilejson": "3.0.0",
            "tiles":    ["{z}/{x}/{y}/{y}.glb"],
            "minzoom":  self.zoom,
            "maxzoom":  self.zoom,
            "bounds":   [round(lon_w, 7), round(lat_s, 7),
                         round(lon_e, 7), round(lat_n, 7)],
            "center":   [round(lon_c, 7), round(lat_c, 7), self.zoom],
            "extras":   extras,
        }

        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
        return path.resolve()
