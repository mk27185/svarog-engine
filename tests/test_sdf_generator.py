import os
import tempfile
import unittest
from PIL import Image
import numpy as np

from src.conversion.sdf_generator import SDFGenerator


class TestSDFGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = SDFGenerator()
        # Meta for a 1000x1000 m square area
        self.meta = {
            "total_width_m":  1000.0,
            "total_height_m": 1000.0,
            # Origin at (0,0) for simplicity
            "lon_origin": 14.43,
            "lat_origin": 50.07,
            "meters_per_deg_lon":  7000.0,
            "meters_per_deg_lat":  111_320.0,
        }
        # Two simple roads: horizontal and vertical through the center
        self.highways = [
            {
                "type": "way",
                "nodes": [
                    {"lon": 14.43 - 1000 / 7000, "lat": 50.07 + 500 / 111_320},
                    {"lon": 14.43 + 1000 / 7000, "lat": 50.07 + 500 / 111_320},
                ],
                "tags": {"highway": "primary", "name": "Test Road H"},
            },
            {
                "type": "way",
                "nodes": [
                    {"lon": 14.43 + 500 / 7000, "lat": 50.07},
                    {"lon": 14.43 + 500 / 7000, "lat": 50.07 + 1000 / 111_320},
                ],
                "tags": {"highway": "secondary"},
            },
        ]

    def test_output_type(self):
        """Generated texture should be RGBA uint8 [0..255]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test_sdf.png")
            self.generator.generate(self.highways, self.meta, out)

            img = Image.open(out)
            arr = np.array(img, dtype=np.uint8)

            self.assertEqual(arr.shape[:2], (1024, 1024))
            self.assertEqual(arr.shape[2], 4)

    def test_alpha_mask(self):
        """Only pixels inside/at least 1 px of roads get Alpha == 255."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test_sdf.png")
            self.generator.generate(self.highways, self.meta, out)

            img = Image.open(out)
            arr = np.array(img, dtype=np.uint8)
            alpha = arr[:, :, 3]

            # 2 roads of ~1024 px x ~5 px width each ~= 10_240 px
            masked = np.count_nonzero(alpha == 255)
            self.assertGreater(masked, 400, "Too few masked pixels")
            self.assertLess(masked, 1024 * 1024, "Too many masked pixels")

    def test_sdf_range(self):
        """R channel (distance) must be in [0, 255]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test_sdf.png")
            self.generator.generate(self.highways, self.meta, out)
            img = Image.open(out)
            sdf = np.array(img, dtype=np.uint8)[:, :, 0]

        self.assertTrue(np.all(sdf >= 0), "SDF contains negative values")
        self.assertTrue(np.all(sdf <= 255), "SDF exceeds 255")

    def test_sdf_has_range(self):
        """SDF R channel should have a non-trivial value range (not all 128)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test_sdf.png")
            self.generator.generate(self.highways, self.meta, out)
            img = Image.open(out)
            arr = np.array(img, dtype=np.uint8)

        sdf = arr[:, :, 0]
        # SDF should show some variation from 0..255, not be a flat value
        self.assertGreater(
            int(sdf.max() - sdf.min()), 10,
            "SDF channel should exhibit a non-trivial distance range"
        )


if __name__ == "__main__":
    unittest.main()