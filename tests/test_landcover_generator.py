import os

import numpy as np
from PIL import Image

from src.conversion.landcover_generator import LandcoverGenerator


def test_landcover_empty_features(tmp_output):
    meta = {"total_width_m": 100.0, "total_height_m": 100.0}
    path = os.path.join(tmp_output, "lc_empty.png")
    LandcoverGenerator(resolution=64).generate(
        {"water_polygons": [], "waterways": [], "green_polygons": [], "railways": []},
        meta,
        path,
    )
    assert os.path.exists(path)
    arr = np.array(Image.open(path))
    assert arr.shape == (64, 64, 4)
