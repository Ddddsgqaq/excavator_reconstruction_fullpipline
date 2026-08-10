import unittest

import numpy as np

from experiments.scale_test_offline_20260804.run_confidence_error_sweep import (
    KNOWN_RULER_M,
    leave_one_out_ruler_errors,
    ruler_measurement_at_confidence,
)


class ConfidenceErrorSweepTests(unittest.TestCase):
    def test_leave_one_out_scale_has_zero_error_for_consistent_lengths(self):
        result = leave_one_out_ruler_errors(np.array([0.3, 0.3, 0.3]))

        self.assertEqual(result["n_anchors"], 3)
        self.assertAlmostEqual(result["scale_m_per_unit_median"],
                               KNOWN_RULER_M / 0.3)
        self.assertAlmostEqual(result["loo_mae_cm"], 0.0)
        self.assertAlmostEqual(result["loo_rmse_cm"], 0.0)

    def test_fixed_full_mask_extent_is_recovered_from_filtered_support(self):
        height, width = 10, 100
        x = np.arange(width, dtype=np.float64)
        points = np.zeros((height, width, 3), dtype=np.float64)
        points[..., 0] = x[None, :]
        mask = np.ones((height, width), dtype=bool)
        confidence = np.ones((height, width), dtype=np.float64)
        confidence[:, 20:80] = 0.0

        result = ruler_measurement_at_confidence(
            points, mask, confidence, threshold=0.5, tail_percent=10.0)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["length_units"], 79.2, places=6)
        self.assertEqual(result["selected_points"], 400)
        self.assertGreater(result["axis_coverage_fraction"], 0.9)

    def test_too_few_selected_points_is_reported_as_missing(self):
        points = np.zeros((4, 20, 3), dtype=np.float64)
        points[..., 0] = np.arange(20)[None, :]
        mask = np.ones((4, 20), dtype=bool)
        confidence = np.zeros((4, 20), dtype=np.float64)
        confidence[:, 0] = 1.0
        confidence[:, -1] = 1.0

        result = ruler_measurement_at_confidence(
            points, mask, confidence, threshold=0.5, tail_percent=10.0)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
