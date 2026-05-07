import numpy as np
import pytest

from src.conversion.terrain_stamper import (
    TerrainStamper, ROAD_HALF_WIDTHS, DEFAULT_HALF_WIDTH,
)


class TestTerrainStamper:

    def test_stamp_returns_same_shape(self, z_grid, meta, simple_highways):
        stamped = TerrainStamper().stamp(z_grid, meta, simple_highways)
        assert stamped.shape == z_grid.shape

    def test_stamp_lowers_near_road(self, z_grid, meta, simple_highways):
        """Terrain near road centerline should move toward road Z."""
        stamped = TerrainStamper().stamp(z_grid, meta, simple_highways)
        # Wherever roads pass, the elevation should have changed
        diff = np.abs(stamped - z_grid)
        assert diff.max() > 0.1, "Stamping should modify terrain near roads"

    def test_stamp_no_roads_unchanged(self, z_grid, meta):
        stamped = TerrainStamper().stamp(z_grid, meta, [])
        np.testing.assert_array_equal(stamped, z_grid)

    def test_stamp_output_is_float(self, z_grid, meta, simple_highways):
        stamped = TerrainStamper().stamp(z_grid, meta, simple_highways)
        assert np.issubdtype(stamped.dtype, np.floating)

    def test_stamp_values_in_reasonable_range(self, z_grid, meta, simple_highways):
        stamped = TerrainStamper().stamp(z_grid, meta, simple_highways)
        # Shouldn't introduce extreme values
        assert stamped.min() >= z_grid.min() - 5.0
        assert stamped.max() <= z_grid.max() + 5.0

    def test_stamp_single_node_road_ignored(self, z_grid, meta):
        single = [{
            "type": "way",
            "nodes": [{"lon": meta["lon_origin"], "lat": meta["lat_origin"]}],
            "tags": {"highway": "primary"},
        }]
        stamped = TerrainStamper().stamp(z_grid, meta, single)
        np.testing.assert_array_equal(stamped, z_grid)

    def test_half_width_lookup(self):
        assert ROAD_HALF_WIDTHS.get("motorway", DEFAULT_HALF_WIDTH) == 7.0
        assert ROAD_HALF_WIDTHS.get("residential", DEFAULT_HALF_WIDTH) == 2.5
        assert ROAD_HALF_WIDTHS.get("unknown_tag", DEFAULT_HALF_WIDTH) == DEFAULT_HALF_WIDTH
