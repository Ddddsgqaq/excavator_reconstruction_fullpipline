"""Prepare scale_test3.mp4 with the repository's offline upload rule."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from resource_profiler import ResourceProfiler


WS = Path(__file__).resolve().parent
VIDEO = ROOT / "scale_test3.mp4"
IMAGES = WS / "images"
FRAME_INTERVAL_S = 0.5


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    profiler = ResourceProfiler(
        "offline_input_preparation", WS,
        metadata={"video": str(VIDEO), "frame_interval_s": FRAME_INTERVAL_S},
    )
    with profiler.stage("decode_video_and_write_frames"):
        capture = cv2.VideoCapture(str(VIDEO))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {VIDEO}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        interval = max(1, int(fps * FRAME_INTERVAL_S))
        count = output_index = 0
        frame_records = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            count += 1
            if count % interval != 0:
                continue
            path = IMAGES / f"{output_index:06d}.png"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"Could not write {path}")
            frame_records.append({
                "output_index": output_index,
                "source_frame_count_1based": count,
                "time_s": count / fps,
                "path": str(path.resolve()),
            })
            output_index += 1
        capture.release()

    with profiler.stage("render_contact_sheet"):
        paths = sorted(IMAGES.glob("*.png"))
        thumb_size = (320, 180)
        label_height = 28
        cols = 4
        rows = (len(paths) + cols - 1) // cols
        sheet = Image.new(
            "RGB", (cols * thumb_size[0], rows * (thumb_size[1] + label_height)),
            (235, 235, 235),
        )
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(paths):
            image = Image.open(path).convert("RGB")
            image.thumbnail(thumb_size)
            x = (index % cols) * thumb_size[0]
            y = (index // cols) * (thumb_size[1] + label_height)
            sheet.paste(image, (x, y + label_height))
            draw.text((x + 8, y + 6), f"frame {index:02d}", fill=(20, 20, 20))
        contact_sheet = WS / "contact_sheet.png"
        sheet.save(contact_sheet)

    profile_path = profiler.finish(metadata={"output_frames": len(frame_records)})
    metadata = {
        "video": str(VIDEO.resolve()),
        "fps": fps,
        "source_frames": source_frames,
        "duration_s": source_frames / fps,
        "source_resolution": [width, height],
        "frame_interval_s": FRAME_INTERVAL_S,
        "interval_frames_floor": interval,
        "output_frames": len(frame_records),
        "frames": frame_records,
        "contact_sheet": str(contact_sheet.resolve()),
        "resource_profile": profile_path,
    }
    with (WS / "input_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
