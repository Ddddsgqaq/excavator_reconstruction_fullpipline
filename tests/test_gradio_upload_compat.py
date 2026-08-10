"""Regression tests for Gradio 4/5 upload and frontend configuration compatibility."""

import re
import unittest
from urllib.parse import urlsplit

import orchestrator


class _FileDataLike:
    def __init__(self, path):
        self.path = path


class GradioUploadCompatTests(unittest.TestCase):
    def test_video_dict_uses_video_key(self):
        value = {"video": "/tmp/example.mp4", "subtitles": None}
        self.assertEqual(orchestrator._uploaded_path(value, "video"), "/tmp/example.mp4")

    def test_file_dict_uses_path_key(self):
        self.assertEqual(orchestrator._uploaded_path({"path": "/tmp/a.png"}), "/tmp/a.png")

    def test_filedata_like_object_uses_path_attribute(self):
        self.assertEqual(orchestrator._uploaded_path(_FileDataLike("/tmp/b.png")), "/tmp/b.png")

    def test_live_controls_are_opt_in_and_help_urls_have_valid_ports(self):
        config = orchestrator.demo.get_config_file()
        components = config["components"]
        live_toggle = next(
            c for c in components
            if c["type"] == "checkbox" and c["props"].get("label") == "我要接入实时视频流"
        )
        self.assertFalse(live_toggle["props"]["value"])

        live_buttons = {
            c["props"].get("value"): c["props"].get("interactive")
            for c in components if c["type"] == "button"
        }
        self.assertFalse(live_buttons["① 连接并预览（不运行 VGGT）"])
        self.assertFalse(live_buttons["② 启动相机 → VGGT 滑窗重建"])

        for component in components:
            info = component.get("props", {}).get("info") or ""
            for candidate in re.findall(r"https?://[^；\s]+", info):
                # Gradio autolinks bare URLs in component help text. A symbolic
                # port such as http://IP:端口/ crashes Blocks.svelte during mount.
                urlsplit(candidate).port

    def test_page_has_no_automatic_load_event(self):
        config = orchestrator.demo.get_config_file()
        load_events = [
            dependency for dependency in config["dependencies"]
            if any(target[1] == "load" for target in dependency.get("targets", []))
        ]
        self.assertEqual(load_events, [])


if __name__ == "__main__":
    unittest.main()
