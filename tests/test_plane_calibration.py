import unittest

import numpy as np

from plane_calibration import PlaneCalibrationRequest, calibrate_local_plane


def _synthetic_scene():
    frames, height, width = 3, 40, 40
    points = np.zeros((frames, height, width, 3), dtype=np.float64)
    confidence = np.ones((frames, height, width), dtype=np.float64)
    semantic = np.ones((frames, height, width), dtype=np.int32)
    yy, xx = np.mgrid[:height, :width]
    # Horizontal VGGT scale is 0.5 m/u: the 13 cm box measures 0.26 units.
    points[..., 0] = xx[None] * 0.26 / 15
    points[..., 2] = yy[None] * 0.26 / 15
    semantic[:, 10:26, 10:26] = 2
    # Vertical VGGT scale is also 0.5 m/u: a 3 cm height measures 0.06 units.
    points[:, 10:26, 10:26, 1] = 0.06
    return points, confidence, semantic


class PlaneCalibrationTests(unittest.TestCase):
    def test_known_box_recovers_horizontal_and_vertical_scales(self):
        points, confidence, semantic = _synthetic_scene()
        result = calibrate_local_plane(
            points, confidence, semantic, np.array([0.0, 1.0, 0.0]),
            PlaneCalibrationRequest(
                object_semantic_id=2,
                object_length_m=0.13,
                object_width_m=0.13,
                object_height_m=0.03,
            ),
        )

        self.assertEqual(result["total_valid_frames"], 3)
        self.assertAlmostEqual(result["horizontal_m_per_vggt_unit"], 0.5, places=2)
        self.assertAlmostEqual(result["vertical_m_per_vggt_unit"], 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
