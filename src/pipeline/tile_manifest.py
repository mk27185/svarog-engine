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
