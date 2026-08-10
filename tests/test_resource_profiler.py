import json
import tempfile
import time
import unittest
from pathlib import Path

from resource_profiler import ResourceProfiler


class ResourceProfilerTests(unittest.TestCase):
    def test_cpu_profile_is_written_incrementally(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler = ResourceProfiler("unit_test", Path(tmp), enabled=True)
            with profiler.stage("small_step", metadata={"items": 3}):
                time.sleep(0.001)

            partial = json.loads(profiler.path.read_text(encoding="utf-8"))
            self.assertEqual(partial["status"], "running")
            self.assertEqual(partial["stages"][0]["name"], "small_step")
            self.assertGreater(partial["stages"][0]["wall_time_s"], 0)
            self.assertEqual(partial["stages"][0]["metadata"]["items"], 3)
            self.assertFalse(partial["stages"][0]["start"]["cuda_available"])

            path = profiler.finish(metadata={"result_items": 4})
            completed = json.loads(profiler.path.read_text(encoding="utf-8"))
            self.assertEqual(path, str(profiler.path))
            self.assertEqual(completed["status"], "ok")
            self.assertEqual(completed["total"]["stage_count"], 1)
            self.assertEqual(completed["metadata"]["result_items"], 4)

    def test_failed_stage_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler = ResourceProfiler("unit_test_error", Path(tmp), enabled=True)
            with self.assertRaises(ValueError):
                with profiler.stage("bad_step"):
                    raise ValueError("expected")

            data = json.loads(profiler.path.read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "error")
            self.assertEqual(data["stages"][0]["status"], "error")
            self.assertIn("ValueError", data["stages"][0]["error"])


if __name__ == "__main__":
    unittest.main()
