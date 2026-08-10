#!/usr/bin/env bash
# start_stream.sh — start the VGGT→Unity real-time elevation link (file channel).
#
# Prereqs (see REALTIME_LINK_PLAN.md "运行指引"):
#   1. vggt_service must already be running with the model resident:
#        /home/maomaoyu/miniconda3/envs/vggt50/bin/python vggt_service.py
#      (wait for "VGGT model ready on cuda")
#   2. Unity: ElevationFileLoader with Auto Poll on, Poll Directory = the Windows view
#      of $FILE_OUT below (D:\tuanjie\exea1\excavator-app-unity-main\live_elevation).
#
# Usage:
#   tools/start_stream.sh [video_path] [interval_seconds]
# Defaults: the dji fly clip (has real camera motion), interval 6s.

set -euo pipefail

VIDEO="${1:-/home/maomaoyu/WS/vggt_yoloe/dji_fly_20260511_161858_0_1778489356002_video.mp4}"
INTERVAL="${2:-6}"
# WSL-side path to the shared dir Unity polls (Windows: D:\...\live_elevation).
FILE_OUT="/mnt/d/tuanjie/exea1/excavator-app-unity-main/live_elevation"
HOST="http://localhost:8002"

if [ ! -f "$VIDEO" ]; then
  echo "ERROR: video not found: $VIDEO" >&2
  exit 1
fi

# Fail early with a clear message if the service isn't up yet.
if ! curl -sf "$HOST/stream/status" >/dev/null 2>&1; then
  echo "ERROR: vggt_service not reachable at $HOST." >&2
  echo "       Start it first:  python vggt_service.py   (wait for 'VGGT model ready')" >&2
  exit 1
fi

echo "starting stream:"
echo "  video   = $VIDEO"
echo "  file_out= $FILE_OUT"
echo "  interval= ${INTERVAL}s"
echo

curl -sS -X POST "$HOST/stream/start" \
  -H "Content-Type: application/json" \
  -d "{
        \"video_path\": \"$VIDEO\",
        \"file_out\": \"$FILE_OUT\",
        \"interval\": $INTERVAL,
        \"loop_video\": true
      }"
echo
echo
echo "watch progress:  curl $HOST/stream/status"
echo "terrain viewer:  $HOST/stream/viewer"
echo "stop:            tools/stop_stream.sh"
