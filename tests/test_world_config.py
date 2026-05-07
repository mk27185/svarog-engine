import json
import os
import pytest
from src.pipeline.world_config import WorldConfig, _find_contracts_dir


class TestWorldConfig:

    def test_load_from_contracts(self):
        cfg = WorldConfig.load()
        assert isinstance(cfg.tile_size_m, int)
        assert isinstance(cfg.draco, bool)
        assert isinstance(cfg.upsample_factor, int)
        assert isinstance(cfg.load_radius_tiles, int)

    def test_tile_size_in_valid_range(self):
        cfg = WorldConfig.load()
        assert 50 <= cfg.tile_size_m <= 5000

    def test_contracts_dir_found(self):
        d = _find_contracts_dir()
        assert (d / "world-config.json").exists()

    def test_contracts_has_tiles_property(self):
        """world-config.json values match schema constraints."""
        cfg = WorldConfig.load()
        assert cfg.upsample_factor >= 1
        assert cfg.load_radius_tiles >= 0

    def test_missing_contracts_raises(self, monkeypatch, tmp_path):
        """Missing contracts directory must raise FileNotFoundError, not silently use defaults."""
        monkeypatch.setenv("SVAROG_CONTRACTS_DIR", str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            WorldConfig.load()

    def test_save_and_reload(self, tmp_path):
        cfg = WorldConfig(tile_size_m=200, draco=False, upsample_factor=5,
                         load_radius_tiles=2)
        path = cfg.save(contracts_dir=tmp_path)
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert data["tile_size_m"] == 200
        assert data["draco"] is False


class TestConfigLoader:

    def test_load_basic_yaml(self, tmp_path):
        import yaml
        from src.pipeline.config_loader import ConfigLoader

        cfg_path = tmp_path / "tiles.yaml"
        cfg_path.write_text(yaml.dump({
            "output_root": str(tmp_path / "out"),
            "tiles": [
                {"name": "t1", "bbox": [50.07, 14.43, 50.08, 14.44]},
                {"name": "t2", "bbox": [50.08, 14.43, 50.09, 14.44]},
            ],
        }))

        batch = ConfigLoader.load(cfg_path)
        assert batch.output_root == str(tmp_path / "out")
        assert len(batch.tiles) == 2
        assert batch.tiles[0].name == "t1"

    def test_load_xyz_tile(self, tmp_path):
        import yaml
        from src.pipeline.config_loader import ConfigLoader

        cfg_path = tmp_path / "xyz.yaml"
        cfg_path.write_text(yaml.dump({
            "output_root": str(tmp_path),
            "tiles": [{"xyz": [15, 17698, 11100]}],
        }))

        batch = ConfigLoader.load(cfg_path)
        assert len(batch.tiles) == 1
        t = batch.tiles[0]
        assert t.xyz is not None
        assert t.xyz.z == 15

    def test_load_overrides(self, tmp_path):
        import yaml
        from src.pipeline.config_loader import ConfigLoader

        cfg_path = tmp_path / "override.yaml"
        cfg_path.write_text(yaml.dump({
            "output_root": str(tmp_path),
            "upsample_factor": 5,
            "draco": False,
            "tiles": [{"name": "t1", "bbox": [50.07, 14.43, 50.08, 14.44]}],
        }))

        batch = ConfigLoader.load(cfg_path)
        assert batch.upsample_factor == 5
        assert batch.draco is False

    def test_missing_output_root_raises(self, tmp_path):
        import yaml
        from src.pipeline.config_loader import ConfigLoader

        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text(yaml.dump({"tiles": [{"name": "t1", "bbox": [50, 14, 51, 15]}]}))

        with pytest.raises(ValueError, match="output_root"):
            ConfigLoader.load(cfg_path)
