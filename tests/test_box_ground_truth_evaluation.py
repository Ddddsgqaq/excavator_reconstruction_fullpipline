import unittest

from experiments.scale_test_offline_20260804.evaluate_box_ground_truth import (
    evaluate_object,
    signed_error,
)


class BoxGroundTruthEvaluationTests(unittest.TestCase):
    def test_signed_error_preserves_underestimate_direction(self):
        result = signed_error(8.0, 10.0)

        self.assertEqual(result["signed_error"], -2.0)
        self.assertEqual(result["absolute_error"], 2.0)
        self.assertEqual(result["relative_error_percent"], -20.0)

    def test_exact_cuboid_has_zero_dimension_and_volume_error(self):
        summary = lambda value: {
            "median": value, "q25": value, "q75": value,
        }
        result = {
            "n_valid_frames": 3,
            "length_m": summary(.10),
            "width_m": summary(.05),
            "height_m": summary(.02),
            "volume_ml": summary(100.0),
        }
        truth = {"name": "test", "dimensions_cm": (10.0, 5.0, 2.0)}

        evaluated = evaluate_object(9, result, truth)

        self.assertEqual(evaluated["dimension_mape_percent"], 0.0)
        self.assertEqual(
            evaluated["volume"]["elevation_integrated"]["relative_error_percent"],
            0.0,
        )
        self.assertTrue(
            evaluated["volume"]["true_envelope_within_integrated_frame_iqr"])


if __name__ == "__main__":
    unittest.main()
