#!/usr/bin/env python
"""send_test_tile.py — Milestone 0 smoke test for the VGGT → Unity elevation link.

Pushes an ElevationMsg (see TERRAIN_ELEVATION_FORMAT.md) into the excavator-app-unity
terrain importer via two interchangeable channels:

  * file  — write JSON to a directory the Unity `ElevationFileLoader` reads (no broker).
  * mqtt  — publish JSON to topic `01/map/elevation` (needs a broker + paho-mqtt).

The payload is either a synthetic DEM (a mound + a pit on a flat plane) or an existing
elevation_tile JSON. In --animate mode the pit walks in a circle and deepens each frame,
so you can *see* the Unity terrain update live — the whole point of the real-time link.

This exercises the real export path (`elevation_export.dem_to_elevation_msg`), so a green
run here means the format contract between the two projects holds.

Examples
--------
# One synthetic tile written to the Unity file-loader directory (zero deps):
python tools/send_test_tile.py --file-out /mnt/d/tuanjie/exea1/excavator-app-unity-main/Assets/Terrain/live

# Animated pit over MQTT at 1 Hz for 60 frames:
python tools/send_test_tile.py --mqtt --broker 127.0.0.1 --animate --repeat 60 --interval 1.0

# Replay an existing tile JSON over both channels:
python tools/send_test_tile.py --input workspaces/<id>/elevation_tile_0_0.json --mqtt --file-out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# Import the project's real exporter so this test shares the production format path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from elevation_export import dem_to_elevation_msg, write_elevation_json  # noqa: E402

TOPIC = "01/map/elevation"


def synthetic_dem(res: int, tile_m: float, frame: int = 0, animate: bool = False):
    """A flat plane with one Gaussian mound and one Gaussian pit, in metres.

    Returns (elev, x_bounds, z_bounds). Row index = Z, col index = X, matching
    dem_to_elevation_msg's row-major expectation.
    """
    xs = np.linspace(-tile_m / 2, tile_m / 2, res)
    zs = np.linspace(-tile_m / 2, tile_m / 2, res)
    gx, gz = np.meshgrid(xs, zs)  # (res, res); gz varies over rows

    elev = np.zeros((res, res), dtype=np.float64)

    # A static mound to give a fixed reference the eye can lock onto.
    mound_sigma = tile_m * 0.12
    elev += 1.2 * np.exp(-((gx - tile_m * 0.2) ** 2 + (gz - tile_m * 0.2) ** 2)
                         / (2 * mound_sigma ** 2))

    # A pit that (in animate mode) circles the origin and deepens each frame,
    # emulating an excavator digging progressively.
    ang = frame * 0.35 if animate else 0.0
    depth = (0.4 + 0.06 * frame) if animate else 1.5
    px = tile_m * 0.18 * np.cos(ang)
    pz = tile_m * 0.18 * np.sin(ang)
    pit_sigma = tile_m * 0.09
    elev -= depth * np.exp(-((gx - px) ** 2 + (gz - pz) ** 2) / (2 * pit_sigma ** 2))

    return elev, (float(xs[0]), float(xs[-1])), (float(zs[0]), float(zs[-1]))


def build_msg(args, frame: int) -> dict:
    if args.input:
        with open(args.input) as f:
            return json.load(f)

    elev, xb, zb = synthetic_dem(args.res, args.tile_size, frame, args.animate)
    return dem_to_elevation_msg(
        elev, xb, zb,
        height_resolution=args.height_resolution,
        tile_x=args.tile_x, tile_y=args.tile_y,
        tile_size_meters=args.tile_size,
        timestamp=time.time(),
    )


def make_mqtt_client(broker: str, port: int):
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        sys.exit("--mqtt requested but paho-mqtt is not installed "
                 "(`pip install paho-mqtt`). Use --file-out for a broker-free test.")
    # paho-mqtt 2.x requires an explicit callback API version; fall back for 1.x.
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()
    client.connect(broker, port, keepalive=60)
    client.loop_start()
    return client


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("payload")
    src.add_argument("--input", help="Publish this existing ElevationMsg JSON instead of a synthetic one.")
    src.add_argument("--res", type=int, default=128, help="Synthetic grid size (width==height). Default 128.")
    src.add_argument("--tile-size", type=float, default=50.0, help="tile_size_meters. Default 50 (Unity default).")
    src.add_argument("--height-resolution", type=float, default=0.01, help="int16 quantisation step (m). Default 0.01.")
    src.add_argument("--tile-x", type=int, default=0)
    src.add_argument("--tile-y", type=int, default=0)

    chan = ap.add_argument_group("channels (pick at least one)")
    chan.add_argument("--file-out", help="Directory to write elevation_tile_<x>_<y>.json into (atomic write).")
    chan.add_argument("--mqtt", action="store_true", help="Publish over MQTT to topic 01/map/elevation.")
    chan.add_argument("--broker", default="127.0.0.1")
    chan.add_argument("--port", type=int, default=1883)
    chan.add_argument("--topic", default=TOPIC)

    loop = ap.add_argument_group("loop")
    loop.add_argument("--repeat", type=int, default=1, help="Number of frames to send. Default 1.")
    loop.add_argument("--interval", type=float, default=1.0, help="Seconds between frames. Default 1.0.")
    loop.add_argument("--animate", action="store_true", help="Move+deepen the synthetic pit each frame.")

    args = ap.parse_args()

    if not args.file_out and not args.mqtt:
        ap.error("pick at least one channel: --file-out DIR and/or --mqtt")

    client = make_mqtt_client(args.broker, args.port) if args.mqtt else None
    if args.file_out:
        os.makedirs(args.file_out, exist_ok=True)

    try:
        for frame in range(args.repeat):
            msg = build_msg(args, frame)
            payload = json.dumps(msg)
            m = msg["metadata"]

            if args.file_out:
                fname = f"elevation_tile_{m['tile_x']}_{m['tile_y']}.json"
                dst = os.path.join(args.file_out, fname)
                tmp = dst + ".tmp"
                write_elevation_json(tmp, msg)
                os.replace(tmp, dst)  # atomic: Unity never reads a half-written file

            if client is not None:
                client.publish(args.topic, payload, qos=1, retain=True)

            note = (f"frame {frame + 1}/{args.repeat}  "
                    f"{m['width']}x{m['height']}  "
                    f"elev[{m['min_elevation']:.2f},{m['max_elevation']:.2f}]m  "
                    f"tile=({m['tile_x']},{m['tile_y']})  {len(payload)} bytes")
            channels = []
            if args.file_out:
                channels.append("file")
            if client is not None:
                channels.append("mqtt")
            print(f"[send] {note}  -> {'+'.join(channels)}")

            if frame < args.repeat - 1:
                time.sleep(args.interval)
    finally:
        if client is not None:
            client.loop_stop()
            client.disconnect()


if __name__ == "__main__":
    main()
