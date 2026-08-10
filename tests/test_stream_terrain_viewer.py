"""CPU-only tests for the read-only Unity terrain stream viewer."""

import glob
import json
import os
import tempfile
import unittest

from streaming import endpoints
from streaming.elevation_publisher import ElevationPublisher, NullPublisher


def _message(tile_x=0, tile_y=0, values=None):
    values = values or [0, 10, -32768, 30]
    return {
        "timestamp": 123.0,
        "metadata": {
            "width": 2,
            "height": 2,
            "height_resolution": 0.01,
            "tile_x": tile_x,
            "tile_y": tile_y,
            "tile_size_meters": 50.0,
            "nodata_value": -32768,
        },
        "data_type": "int16",
        "data": values,
        "data_order": "row_major",
    }


class TerrainSnapshotTests(unittest.TestCase):
    def test_file_payload_and_viewer_snapshot_are_identical(self):
        with tempfile.TemporaryDirectory() as root:
            publisher = ElevationPublisher(file_out=root)
            msg = _message()
            publisher.publish(msg)
            snapshot = publisher.snapshot()
            with open(os.path.join(root, "elevation_tile_0_0.json")) as handle:
                on_disk = json.load(handle)

            self.assertEqual(snapshot["sequence"], 1)
            self.assertEqual(snapshot["tile_count"], 1)
            self.assertEqual(snapshot["tiles"][0], on_disk)
            msg["data"][0] = 999
            self.assertEqual(publisher.snapshot()["tiles"][0]["data"][0], 0)

    def test_new_stream_archives_stale_tile_files(self):
        with tempfile.TemporaryDirectory() as root:
            stale = os.path.join(root, "elevation_tile_4_-2.json")
            with open(stale, "w", encoding="utf-8") as handle:
                json.dump(_message(4, -2), handle)

            ElevationPublisher(file_out=root, archive_existing_tiles=True)

            self.assertFalse(os.path.exists(stale))
            archived = glob.glob(os.path.join(root, ".previous_stream", "*", os.path.basename(stale)))
            self.assertEqual(len(archived), 1)

    def test_latest_payload_is_kept_per_tile(self):
        with tempfile.TemporaryDirectory() as root:
            publisher = ElevationPublisher(file_out=root)
            publisher.publish(_message(1, 0))
            publisher.publish(_message(-1, 0))
            publisher.publish(_message(1, 0, [1, 2, 3, 4]))
            snapshot = publisher.snapshot()

            self.assertEqual(snapshot["sequence"], 3)
            self.assertEqual(snapshot["tile_count"], 2)
            self.assertEqual(
                [(t["metadata"]["tile_x"], t["metadata"]["tile_y"]) for t in snapshot["tiles"]],
                [(-1, 0), (1, 0)],
            )
            self.assertEqual(snapshot["tiles"][1]["data"], [1, 2, 3, 4])

    def test_null_publisher_exposes_empty_snapshot(self):
        self.assertEqual(NullPublisher().snapshot()["tiles"], [])


class _FakePublisher:
    channels = ["file"]

    def snapshot(self):
        return {
            "sequence": 7,
            "published_at": 456.0,
            "tile_count": 1,
            "tiles": [_message()],
        }


class _FakeLoop:
    def __init__(self):
        self.publisher = _FakePublisher()
        self.stopped = False

    def stop(self):
        self.stopped = True


class TerrainEndpointTests(unittest.TestCase):
    def setUp(self):
        self.previous_loop = endpoints._loop
        self.previous_snapshot = endpoints._last_terrain_snapshot

    def tearDown(self):
        endpoints._loop = self.previous_loop
        endpoints._last_terrain_snapshot = self.previous_snapshot

    def test_stop_preserves_last_terrain_for_postmortem(self):
        loop = _FakeLoop()
        endpoints._loop = loop
        endpoints._last_terrain_snapshot = NullPublisher().snapshot()

        self.assertEqual(endpoints.stream_stop(), {"status": "stopped"})
        terrain = endpoints.stream_terrain()
        self.assertTrue(loop.stopped)
        self.assertTrue(terrain["available"])
        self.assertFalse(terrain["running"])
        self.assertEqual(terrain["sequence"], 7)

    def test_viewer_contains_offline_start_and_orb_controls(self):
        page = os.path.join(os.path.dirname(os.path.dirname(__file__)), "stream_terrain_viewer.html")
        with open(page, encoding="utf-8") as handle:
            html = handle.read()
        self.assertIn("离线视频流测试控制", html)
        self.assertIn('id="ctl-orb"', html)
        self.assertIn('id="ctl-fusion"', html)
        self.assertIn('id="unity-output"', html)
        self.assertIn("/mnt/d/tuanjie/exea1/excavator-app-unity-main/live_elevation", html)
        self.assertIn("启动离线 VGGT 流测试", html)
        self.assertIn("frame_sample_interval", html)
        self.assertIn("use_orb", html)
        self.assertIn("Fill Holes and Show DEM Mesh", html)
        self.assertIn("MeshBasicMaterial", html)
        self.assertIn("demGridLines", html)
        self.assertIn('id="fusion-terrain"', html)
        self.assertIn("fusion_dem", html)
        self.assertIn("vertical_datum_m", html)
        self.assertIn("buildElevationDEM(app.passData.dem)", html)
        self.assertNotIn("unityTileToDEM", html)
        self.assertNotIn("id=\"ctl-tile-size\"", html)
        self.assertNotIn("id=\"cp\"", html)
        self.assertNotIn("id=\"cb\"", html)

    def test_viewer_route_targets_static_page(self):
        response = endpoints.stream_viewer()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store, max-age=0")


if __name__ == "__main__":
    unittest.main()
