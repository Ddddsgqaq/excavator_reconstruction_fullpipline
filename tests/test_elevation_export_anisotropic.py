import unittest

import numpy as np

from elevation_export import dem_to_elevation_msg


class ElevationExportAnisotropicTests(unittest.TestCase):
    def test_independent_horizontal_and_vertical_scales(self):
        msg = dem_to_elevation_msg(
            np.array([[1.0, 3.0], [2.0, 4.0]]),
            (0.0, 2.0), (0.0, 4.0),
            horizontal_scale=2.0,
            vertical_scale=0.5,
            height_resolution=0.1,
        )
        metadata = msg["metadata"]
        self.assertEqual(metadata["min_elevation"], 0.5)
        self.assertEqual(metadata["max_elevation"], 2.0)
        self.assertEqual(metadata["x_span_meters"], 4.0)
        self.assertEqual(metadata["z_span_meters"], 8.0)
        self.assertEqual(metadata["horizontal_m_per_vggt_unit"], 2.0)
        self.assertEqual(metadata["vertical_m_per_vggt_unit"], 0.5)


if __name__ == "__main__":
    unittest.main()
