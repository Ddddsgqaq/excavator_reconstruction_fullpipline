"""elevation_publisher.py — push ElevationMsg to Unity over file and/or MQTT (M3).

Two interchangeable channels, either or both:
  * file  — atomically write elevation_tile_<x>_<y>.json into a directory the Unity
            `ElevationFileLoader` auto-polls. Zero deps, fully local.
  * mqtt  — publish JSON to topic `01/map/elevation` for the Unity `MqttManager`.
            Requires paho-mqtt + a broker; degrades gracefully if unavailable.

Atomic file write (tmp + os.replace) guarantees Unity's poller never reads a half-written
file — matches tools/send_test_tile.py and the ElevationFileLoader "*.json only" contract.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import threading
import time

TOPIC = "01/map/elevation"


class NullPublisher:
    """No-op channel for validating camera capture without reconstruction output."""

    @property
    def channels(self) -> list[str]:
        return []

    def publish(self, _msg: dict) -> None:
        return None

    def snapshot(self) -> dict:
        return {
            "sequence": 0,
            "published_at": None,
            "tile_count": 0,
            "tiles": [],
        }

    def close(self) -> None:
        return None


class ElevationPublisher:
    def __init__(
        self,
        *,
        file_out: str | None = None,
        mqtt: bool = False,
        broker: str = "127.0.0.1",
        port: int = 1883,
        topic: str = TOPIC,
        retain: bool = True,
        archive_existing_tiles: bool = False,
    ):
        if not file_out and not mqtt:
            raise ValueError("ElevationPublisher needs at least one channel: file_out and/or mqtt")
        self.file_out = file_out
        self.topic = topic
        self.retain = retain
        self._client = None
        self._channels: list[str] = []
        self._snapshot_lock = threading.Lock()
        self._latest_tiles: dict[tuple[int, int], dict] = {}
        self._sequence = 0
        self._published_at: float | None = None

        if file_out:
            os.makedirs(file_out, exist_ok=True)
            if archive_existing_tiles:
                stale = sorted(glob.glob(os.path.join(file_out, "elevation_tile_*.json")))
                if stale:
                    archive_dir = os.path.join(file_out, ".previous_stream", str(time.time_ns()))
                    os.makedirs(archive_dir, exist_ok=True)
                    for src in stale:
                        os.replace(src, os.path.join(archive_dir, os.path.basename(src)))
            self._channels.append("file")

        if mqtt:
            self._client = self._make_mqtt_client(broker, port)
            self._channels.append("mqtt")

    @staticmethod
    def _make_mqtt_client(broker: str, port: int):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as e:
            raise RuntimeError(
                "mqtt channel requested but paho-mqtt is not installed "
                "(`pip install paho-mqtt`). Use file_out for a broker-free run."
            ) from e
        # paho 2.x needs an explicit callback API version; fall back for 1.x.
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            client = mqtt.Client()
        client.connect(broker, port, keepalive=60)
        client.loop_start()
        return client

    @property
    def channels(self) -> list[str]:
        return list(self._channels)

    def publish(self, msg: dict) -> None:
        """Send one ElevationMsg over all configured channels."""
        payload = json.dumps(msg)
        m = msg.get("metadata", {})
        tile_x = int(m.get("tile_x", 0))
        tile_y = int(m.get("tile_y", 0))

        if self.file_out:
            fname = f"elevation_tile_{tile_x}_{tile_y}.json"
            dst = os.path.join(self.file_out, fname)
            tmp = dst + ".tmp"
            with open(tmp, "w") as f:
                f.write(payload)
            os.replace(tmp, dst)  # atomic: poller never sees a partial file

        if self._client is not None:
            self._client.publish(self.topic, payload, qos=1, retain=self.retain)

        # Keep an in-memory copy of the exact message accepted by the output channels.
        # The stream viewer reads only this cache, so it cannot alter reconstruction,
        # fusion, file publishing, or the payload Unity consumes.
        with self._snapshot_lock:
            self._latest_tiles[(tile_x, tile_y)] = copy.deepcopy(msg)
            self._sequence += 1
            self._published_at = time.time()

    def snapshot(self) -> dict:
        """Return the latest published payload per tile for read-only diagnostics."""
        with self._snapshot_lock:
            tiles = [
                copy.deepcopy(self._latest_tiles[key])
                for key in sorted(self._latest_tiles)
            ]
            return {
                "sequence": self._sequence,
                "published_at": self._published_at,
                "tile_count": len(tiles),
                "tiles": tiles,
            }

    def close(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
