#!/usr/bin/env bash
# stop_stream.sh — stop the running VGGT→Unity streaming session.
set -euo pipefail
HOST="http://localhost:8002"
curl -sS -X POST "$HOST/stream/stop"
echo
