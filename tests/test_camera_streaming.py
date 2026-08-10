"""CPU-only coverage for live-camera plumbing and capture-only isolation."""

import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from streaming import camera_source, endpoints
from elevation_plane import _extract_points_with_conf
from streaming.camera_source import OpenCvCameraSource, normalize_mjpeg_url, safe_source_label
from streaming.elevation_publisher import NullPublisher
from streaming.frame_source import FrameListSource, VideoFileSource
from streaming.keyframe_buffer import KeyframeBuffer
from streaming.reconstruct_loop import LoopConfig, ReconstructLoop
from streaming.source_factory import build_frame_source
from streaming.pipeline import dem_result_to_msg


class _FakeCapture:
    def __init__(self, frame):
        self.frame = frame
        self.released = False

    def isOpened(self):
        return not self.released

    def read(self):
        return (not self.released), self.frame.copy()

    def set(self, *_args):
        return True

    def get(self, prop):
        return 30.0 if prop == camera_source.cv2.CAP_PROP_FPS else 0.0

    def release(self):
        self.released = True


class CameraStreamingTests(unittest.TestCase):
    def test_camera_source_emits_rgb_preview_and_releases(self):
        fake = _FakeCapture(np.full((24, 32, 3), [255, 0, 0], dtype=np.uint8))
        with mock.patch.object(camera_source.cv2, "VideoCapture", side_effect=lambda *_args: fake):
            source = OpenCvCameraSource(0, target_fps=1000, backend="auto", reconnect=False)
            iterator = source.frames()
            frame = next(iterator)
            self.assertEqual(frame.shape, (24, 32, 3))
            self.assertEqual(frame[0, 0].tolist(), [0, 0, 255])
            self.assertTrue(np.array_equal(source.latest_frame(), frame))
            self.assertTrue(source.status()["connected"])
            self.assertGreaterEqual(source.status()["decoded"], 1)
            source.close()
            iterator.close()
            self.assertTrue(fake.released)
            self.assertFalse(source.status()["connected"])

    def test_camera_status_redacts_url_credentials(self):
        label = safe_source_label("rtsp://user:secret@10.0.0.4:8554/site?profile=main")
        self.assertNotIn("user", label)
        self.assertNotIn("secret", label)
        self.assertEqual(label, "rtsp://10.0.0.4:8554/site?profile=main")

    def test_ip_webcam_root_resolves_to_mjpeg_video_endpoint(self):
        self.assertEqual(
            normalize_mjpeg_url("http://192.168.31.132:8080/"),
            "http://192.168.31.132:8080/video",
        )
        self.assertEqual(
            normalize_mjpeg_url("http://camera.local:8080/custom.mjpeg?x=1"),
            "http://camera.local:8080/custom.mjpeg?x=1",
        )

    def test_source_factory_keeps_legacy_video_path_contract(self):
        source = build_frame_source(
            source_type="video", source_uri=None, video_path="demo.mp4", target_fps=3.0
        )
        self.assertIsInstance(source, VideoFileSource)
        self.assertEqual(source.path, "demo.mp4")

    def test_interval_mode_skips_orb_and_keeps_sampled_frames(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        buffer = KeyframeBuffer(capacity=3, use_orb=False)
        for _ in range(5):
            self.assertTrue(buffer.offer(frame))
        stats = buffer.stats
        self.assertEqual(stats.selection_mode, "interval")
        self.assertEqual(stats.kept, 5)
        self.assertEqual(stats.skipped, 0)
        self.assertEqual(stats.window, 3)
        self.assertEqual(stats.orb_calls, 0)
        self.assertEqual(stats.orb_total_seconds, 0.0)

    def test_orb_mode_reports_processing_time(self):
        rng = np.random.default_rng(8)
        buffer = KeyframeBuffer(capacity=3, use_orb=True)
        buffer.offer(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        stats = buffer.stats
        self.assertEqual(stats.selection_mode, "orb")
        self.assertEqual(stats.orb_calls, 1)
        self.assertGreater(stats.orb_total_seconds, 0.0)
        self.assertGreater(stats.orb_last_ms, 0.0)

    def test_interval_request_controls_source_sampling_rate(self):
        req = endpoints.StreamStartRequest(
            source_type="video", video_path="demo.mp4", run_reconstruction=False,
            target_fps=9.0, use_orb=False, frame_sample_interval=2.0,
        )
        loop = endpoints._build_loop(req, publisher_override=NullPublisher())
        self.assertFalse(loop.cfg.use_orb)
        self.assertEqual(loop.cfg.frame_sample_interval, 2.0)
        self.assertEqual(loop.source.target_fps, 0.5)

    def test_stream_confidence_gate_matches_viewer_percentile(self):
        pts = np.arange(24, dtype=np.float32).reshape(1, 2, 4, 3)
        conf = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
        selected, selected_conf, _, keep = _extract_points_with_conf(
            {"world_points_from_depth": pts, "depth_conf": conf},
            50.0, "Depthmap and Camera Branch", return_keep_mask=True,
        )
        self.assertEqual(selected.shape[0], 4)
        self.assertEqual(selected_conf.tolist(), [4.0, 5.0, 6.0, 7.0])
        self.assertEqual(np.flatnonzero(keep).tolist(), [4, 5, 6, 7])

    def test_vertical_datum_freezes_first_main_plane(self):
        loop = ReconstructLoop(
            FrameListSource([], target_fps=1.0), NullPublisher(), LoopConfig(capture_only=True)
        )
        first = loop._ensure_vertical_datum(
            np.asarray([[8.0, 10.0], [10.0, 12.0]]), np.ones((2, 2), dtype=bool)
        )
        later = loop._ensure_vertical_datum(
            np.asarray([[100.0, 200.0]]), np.ones((1, 2), dtype=bool)
        )
        self.assertEqual(first, 10.0)
        self.assertEqual(later, first)

    def test_pass_diagnostics_exposes_full_rgb_cloud_not_only_ground(self):
        loop = ReconstructLoop(
            FrameListSource([], target_fps=1.0), NullPublisher(), LoopConfig(capture_only=True)
        )
        points = np.asarray([[0, 0, 0], [1, 2, 1], [2, 3, 2], [3, 4, 3]], dtype=np.float32)
        colors = np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]], dtype=np.uint8)
        result = SimpleNamespace(
            points_aligned=points, point_colors=colors, ground_xyz=points[:2],
            elev=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            has_data=np.asarray([[1, 1], [1, 0]], dtype=bool), scale_factor=1.0,
            n_frames=1, n_points=4, x_bounds=(0.0, 3.0), z_bounds=(0.0, 3.0),
            gravity_source="trajectory", registered=False,
        )
        loop._store_pass_diag(1, [np.zeros((4, 4, 3), dtype=np.uint8)], result, [1.0, 4.0])
        diag = loop.pass_diagnostics()
        self.assertEqual(diag["n_points_shown"], 4)
        self.assertEqual(diag["ground_count"], 2)
        self.assertEqual(diag["dem_min_elevation"], 1.0)
        self.assertEqual(diag["colors"][2], [0, 0, 255])
        self.assertEqual(diag["dem"]["grid_res"], 2)
        self.assertEqual(diag["dem"]["has_data"], [[1, 1], [1, 0]])
        self.assertFalse(diag["fusion_enabled"])
        self.assertIsNone(diag["fusion_dem"])

    def test_unity_payload_uses_viewer_filled_dem_without_nodata(self):
        result = SimpleNamespace(
            elev=np.asarray([[0.0, 0.0, 0.0], [1.0, 9.0, 3.0], [2.0, 4.0, 6.0]]),
            has_data=np.asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
            x_bounds=(0.0, 3.0), z_bounds=(0.0, 3.0), scale_factor=1.0,
        )
        msg = dem_result_to_msg(result, height_resolution=0.01, tile_size_meters=3.0)
        self.assertNotIn(-32768, msg["data"])
        self.assertEqual(msg["metadata"]["nodata_count"], 0)
        self.assertEqual(msg["metadata"]["source_nodata_count"], 1)
        self.assertEqual(msg["metadata"]["dem_preprocessing"], "elevation_viewer_fill_20")
        self.assertEqual(msg["source_valid"], [1, 1, 1, 1, 0, 1, 1, 1, 1])
        expected_center = round(((0 + 0 + 0 + 1 + 3 + 2 + 4 + 6) / 8) / 0.01)
        self.assertEqual(msg["data"][4], expected_center)

    def test_capture_only_loop_never_starts_reconstruction(self):
        rng = np.random.default_rng(7)
        frames = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
        loop = ReconstructLoop(
            FrameListSource(frames, target_fps=200.0),
            NullPublisher(),
            LoopConfig(capture_only=True, interval=0.01, min_frames=2, capacity=4),
        )
        loop.start()
        time.sleep(0.1)
        status = loop.status()
        loop.stop()
        self.assertEqual(status.offered, 4)
        self.assertEqual(status.passes, 0)
        self.assertIsNone(loop._recon_thread)


if __name__ == "__main__":
    unittest.main()
