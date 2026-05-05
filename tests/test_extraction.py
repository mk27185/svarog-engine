import os
import unittest
from src.extraction.opentopography_client import OpenTopographyClient

class TestOpenTopographyClient(unittest.TestCase):
    def setUp(self):
        self.client = OpenTopographyClient()

    def test_prague_small_area(self):
        # Center of Prague: 50.0755, 14.4378
        # 200m offset is roughly 0.0018 degrees
        lat_center = 50.0755
        lon_center = 14.4378
        offset = 0.0018
        
        bbox = (
            lat_center - offset,
            lon_center - offset,
            lat_center + offset,
            lon_center + offset
        )
        
        print(f"\nTesting Prague area: {bbox}")
        data = self.client.get_dem(bbox)
        
        # We check if it returned data. Note: 200m might be too small for some global DEM datasets, 
        # but the test checks the API interaction.
        self.assertIsNotNone(data, "Failed to fetch DEM data for Prague area")
        print(f"Retrieved {len(data)} bytes")

if __name__ == "__main__":
    unittest.main()
