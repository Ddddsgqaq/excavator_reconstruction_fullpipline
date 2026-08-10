import unittest

import numpy as np

from experiments.scale_test_offline_20260804.make_process_visualizations import (
    SEMANTIC_COLORS_MPL,
    _semantic_point_colors,
    select_shared_reconstruction_sample,
)


class SharedReconstructionSampleTests(unittest.TestCase):
    def test_rgb_geometry_and_semantic_ids_share_exact_indices(self):
        height, width = 10, 10
        metric = np.arange(height * width * 3, dtype=np.float64).reshape(
            1, height, width, 3)
        confidence = np.arange(height * width, dtype=np.float64).reshape(
            1, height, width)
        masks = (np.arange(height * width).reshape(1, height, width) % 6).astype(
            np.uint8)
        images = np.stack([
            np.arange(height * width, dtype=np.float64).reshape(height, width) / 100,
            np.full((height, width), 0.25),
            np.full((height, width), 0.75),
        ])[None, ...]
        pred = {"depth_conf": confidence, "images": images}

        sample = select_shared_reconstruction_sample(pred, masks, metric)
        keep = confidence >= np.percentile(confidence, 78)

        np.testing.assert_array_equal(sample["xyz"], metric[keep])
        np.testing.assert_array_equal(sample["semantic_ids"], masks[keep])
        np.testing.assert_array_equal(
            sample["rgb"], np.transpose(images, (0, 2, 3, 1))[keep])
        self.assertEqual(len(sample["xyz"]), len(sample["semantic_ids"]))

    def test_unknown_semantic_ids_use_unlabeled_color(self):
        semantic_ids = np.array([0, 1, 5, 99], dtype=np.uint8)
        colors = _semantic_point_colors(semantic_ids)

        np.testing.assert_array_equal(colors[0], SEMANTIC_COLORS_MPL[0])
        np.testing.assert_array_equal(colors[1], SEMANTIC_COLORS_MPL[1])
        np.testing.assert_array_equal(colors[2], SEMANTIC_COLORS_MPL[5])
        np.testing.assert_array_equal(colors[3], SEMANTIC_COLORS_MPL[0])


if __name__ == "__main__":
    unittest.main()
