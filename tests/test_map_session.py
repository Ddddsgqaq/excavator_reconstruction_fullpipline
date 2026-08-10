"""CPU-only tests for durable two-stage map state and trust gates."""

import tempfile
import unittest

import numpy as np

from streaming.change_detection import ChangePolicy, assess_change
from streaming.global_dem import FusionConfig, GlobalDem
from streaming.initialization import InitializationPolicy, finalize_initialization, stage_initial_map
from streaming.map_session import MapSession, SessionState
from streaming.pipeline import DemResult


def _result(value=1.0, *, registered=False, rmse=None, n_frames=4, n_points=100):
    elev = np.full((4, 4), value, dtype=np.float64)
    return DemResult(
        elev=elev, has_data=np.ones_like(elev, dtype=bool),
        x_bounds=(-2.0, 2.0), z_bounds=(-2.0, 2.0),
        R_align=np.eye(3), scale_factor=1.0, gravity_source="test",
        n_frames=n_frames, n_points=n_points,
        ground_xyz=np.asarray([[-1.0, value, -1.0], [0.0, value, 0.0], [1.0, value, 1.0]]),
        registered=registered, registration_rmse=rmse, registration_yaw_deg=0.0,
    )


class MapSessionTests(unittest.TestCase):
    def test_initialization_persists_artifacts_and_requires_review(self):
        with tempfile.TemporaryDirectory() as root:
            session = MapSession.create(root)
            session.transition(SessionState.INITIALIZING)
            result = _result(n_frames=4, n_points=100)
            report = stage_initial_map(
                session, result, [np.zeros((8, 8, 3), dtype=np.uint8)] * 4,
                policy=InitializationPolicy(
                    min_frames=4, min_points=50, min_coverage=0.5,
                    max_elevation_span_m=2.0,
                ),
                fusion_config=FusionConfig(
                    world_size_m=6.0, tile_size_m=2.0, tile_res=4,
                ),
            )
            self.assertTrue(report["passed"])
            self.assertEqual(session.state, SessionState.INIT_REVIEW)
            loaded = MapSession.load(root)
            self.assertEqual(loaded.load_anchor().ref_ground_xyz.shape, (3, 3))
            self.assertEqual(loaded.load_dem()[0].shape, (4, 4))
            self.assertGreater(loaded.load_global_dem().status()["observed_cells"], 0)
            summary = finalize_initialization(loaded, approved=True, note="checked")
            self.assertEqual(summary["state"], "READY")
            self.assertEqual(summary["map_version"], 1)

    def test_failed_quality_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as root:
            session = MapSession.create(root)
            session.transition(SessionState.INITIALIZING)
            stage_initial_map(
                session, _result(n_frames=2, n_points=3),
                [np.zeros((4, 4, 3), dtype=np.uint8)] * 2,
                policy=InitializationPolicy(min_frames=4, min_points=50),
                fusion_config=FusionConfig(world_size_m=6, tile_size_m=2, tile_res=4),
            )
            with self.assertRaisesRegex(ValueError, "quality gates"):
                finalize_initialization(session, approved=True)

    def test_global_dem_snapshot_round_trip_preserves_publish_baseline(self):
        cfg = FusionConfig(world_size_m=6, tile_size_m=2, tile_res=4, change_thresh=0.01)
        dem = GlobalDem((0.0, 0.0), cfg)
        dem.integrate(_result().ground_xyz, 10.0)
        first = dem.changed_tiles()
        restored = GlobalDem.from_snapshot(dem.to_snapshot())
        self.assertEqual(restored.status()["observed_cells"], dem.status()["observed_cells"])
        restored.integrate(_result().ground_xyz, 10.0)
        self.assertTrue(first)
        self.assertEqual(restored.changed_tiles(), [])

    def test_global_dem_does_not_mutate_for_unchanged_or_isolated_noise(self):
        cfg = FusionConfig(world_size_m=6, tile_size_m=2, tile_res=4, change_thresh=0.05)
        dem = GlobalDem((0.0, 0.0), cfg)
        xz = np.asarray([(x, z) for z in (-1.75, -1.25, -0.75, -0.25)
                         for x in (-1.75, -1.25, -0.75, -0.25)], dtype=float)
        base = np.column_stack((xz[:, 0], np.ones(16), xz[:, 1]))
        first = dem.integrate(base, 1.0, aggregation="mean", min_change_m=0.05)
        self.assertEqual(first["decision"], "updated")
        h0, w0, t0 = dem.H.copy(), dem.W.copy(), dem.T.copy()

        # A uniform reconstruction offset is localization bias, not terrain change.
        unchanged = base.copy(); unchanged[:, 1] += 0.03
        report = dem.integrate(unchanged, 2.0, aggregation="mean", min_change_m=0.05)
        self.assertEqual(report["decision"], "unchanged")
        np.testing.assert_array_equal(dem.H, h0)
        np.testing.assert_array_equal(dem.W, w0)
        np.testing.assert_array_equal(dem.T, t0)

        # One noisy cell has no spatial support and must also leave state untouched.
        noisy = base.copy(); noisy[0, 1] += 0.20
        report = dem.integrate(noisy, 3.0, aggregation="mean", min_change_m=0.05)
        self.assertEqual(report["decision"], "unchanged")
        np.testing.assert_array_equal(dem.H, h0)

        # A coherent 2x2 patch is a supported local terrain change.
        changed = base.copy(); changed[[0, 1, 4, 5], 1] += 0.20
        report = dem.integrate(changed, 4.0, aggregation="mean", min_change_m=0.05)
        self.assertEqual(report["decision"], "updated")
        self.assertEqual(report["changed_cells"], 4)
        self.assertFalse(np.array_equal(dem.H, h0))

    def test_change_gate_accepts_small_registered_change_and_rejects_bad_pose(self):
        reference = np.ones((4, 4), dtype=np.float64)
        valid = np.ones_like(reference, dtype=bool)
        policy = ChangePolicy(
            max_registration_rmse_m=0.5, min_coverage=0.5,
            max_changed_fraction=1.0, max_abs_height_change_m=1.0,
        )
        accepted = assess_change(
            _result(1.1, registered=True, rmse=0.1), reference, valid, policy
        )
        rejected = assess_change(
            _result(1.1, registered=False, rmse=2.0), reference, valid, policy
        )
        self.assertTrue(accepted["accepted"])
        self.assertTrue(accepted["has_change"])
        self.assertFalse(rejected["accepted"])
        self.assertGreaterEqual(len(rejected["reasons"]), 2)


if __name__ == "__main__":
    unittest.main()
