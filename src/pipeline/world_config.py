"""
WorldConfig
===========
Loads and validates world-config.json from svarog-contracts.

Resolution order:
  1. SVAROG_CONTRACTS_DIR environment variable
  2. ../svarog-contracts/  (sibling directory of this repo)

Raises FileNotFoundError if the contracts directory or world-config.json
cannot be found. No fallback defaults — a missing config is a hard error.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import jsonschema

_CONFIG_FILE = Path("world-config.json")
_SCHEMA_FILE = Path("schemas") / "world-config.json"


@dataclass
class WorldConfig:
    tile_size_m:       int
    draco:             bool
    upsample_factor:   int
    load_radius_tiles: int

    @classmethod
    def load(cls) -> "WorldConfig":
        """Load from svarog-contracts. Raises if not found or invalid."""
        contracts_dir = _find_contracts_dir()

        config_path = contracts_dir / _CONFIG_FILE
        schema_path = contracts_dir / _SCHEMA_FILE

        if not config_path.exists():
            raise FileNotFoundError(
                f"world-config.json not found in {contracts_dir}\n"
                "Make sure svarog-contracts is present and up to date."
            )
        if not schema_path.exists():
            raise FileNotFoundError(
                f"Schema not found: {schema_path}\n"
                "Make sure svarog-contracts is present and up to date."
            )

        with open(config_path) as f:
            data = json.load(f)
        with open(schema_path) as f:
            schema = json.load(f)

        jsonschema.validate(instance=data, schema=schema)

        return cls(
            tile_size_m=data["tile_size_m"],
            draco=data["draco"],
            upsample_factor=data["upsample_factor"],
            load_radius_tiles=data["load_radius_tiles"],
        )

    def save(self, contracts_dir: str | Path | None = None) -> Path:
        """Write current values back to world-config.json in contracts repo."""
        d = Path(contracts_dir) if contracts_dir else _find_contracts_dir()
        path = d / _CONFIG_FILE
        with open(path, "w") as f:
            json.dump({
                "tile_size_m":       self.tile_size_m,
                "draco":             self.draco,
                "upsample_factor":   self.upsample_factor,
                "load_radius_tiles": self.load_radius_tiles,
            }, f, indent=2)
        return path


def _find_contracts_dir() -> Path:
    env = os.environ.get("SVAROG_CONTRACTS_DIR")
    if env:
        p = Path(env)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"SVAROG_CONTRACTS_DIR is set to '{env}' but the directory does not exist."
        )

    candidate = Path(__file__).resolve().parents[2].parent / "svarog-contracts"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Cannot find svarog-contracts directory.\n"
        "Expected: ../svarog-contracts/ (sibling of this repo)\n"
        "Or set the SVAROG_CONTRACTS_DIR environment variable."
    )
