"""
WorldConfig
===========
Loads and validates world-config.json from svarog-contracts.

Resolution order (same pattern as TileManifest):
  1. SVAROG_CONTRACTS_DIR env variable
  2. ../svarog-contracts/  sibling of this repo
  3. Inline defaults (so engine works without contracts repo)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import jsonschema

_CONFIG_FILE   = Path("world-config.json")
_SCHEMA_FILE   = Path("schemas") / "world-config.json"
_INLINE_SCHEMA = {
    "type": "object",
    "required": ["glb_zoom", "draco", "upsample_factor", "load_radius_tiles"],
    "properties": {
        "glb_zoom":          {"type": "integer", "minimum": 10, "maximum": 22},
        "draco":             {"type": "boolean"},
        "upsample_factor":   {"type": "integer", "minimum": 1},
        "load_radius_tiles": {"type": "integer", "minimum": 0},
    },
}
_DEFAULTS = {
    "glb_zoom":          15,
    "draco":             True,
    "upsample_factor":   10,
    "load_radius_tiles": 1,
}


@dataclass
class WorldConfig:
    glb_zoom:          int  = 15
    draco:             bool = True
    upsample_factor:   int  = 10
    load_radius_tiles: int  = 1

    @classmethod
    def load(cls) -> "WorldConfig":
        """Load from contracts repo; fall back to defaults if not found."""
        contracts_dir = _find_contracts_dir()

        if contracts_dir is None:
            return cls(**_DEFAULTS)

        config_path = contracts_dir / _CONFIG_FILE
        schema_path = contracts_dir / _SCHEMA_FILE

        if not config_path.exists():
            return cls(**_DEFAULTS)

        with open(config_path) as f:
            data = json.load(f)

        schema = _INLINE_SCHEMA
        if schema_path.exists():
            with open(schema_path) as f:
                schema = json.load(f)

        jsonschema.validate(instance=data, schema=schema)

        return cls(
            glb_zoom=data["glb_zoom"],
            draco=data["draco"],
            upsample_factor=data["upsample_factor"],
            load_radius_tiles=data["load_radius_tiles"],
        )

    def save(self, contracts_dir: str | Path | None = None) -> Path:
        """Write current values back to world-config.json in contracts repo."""
        if contracts_dir is None:
            d = _find_contracts_dir()
            if d is None:
                raise FileNotFoundError(
                    "Cannot find svarog-contracts dir. "
                    "Set SVAROG_CONTRACTS_DIR env variable."
                )
            contracts_dir = d
        path = Path(contracts_dir) / _CONFIG_FILE
        with open(path, "w") as f:
            json.dump({
                "glb_zoom":          self.glb_zoom,
                "draco":             self.draco,
                "upsample_factor":   self.upsample_factor,
                "load_radius_tiles": self.load_radius_tiles,
            }, f, indent=2)
        return path


def _find_contracts_dir() -> Path | None:
    env = os.environ.get("SVAROG_CONTRACTS_DIR")
    if env:
        p = Path(env)
        if p.exists():
            return p

    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root.parent / "svarog-contracts"
    if candidate.exists():
        return candidate

    return None
