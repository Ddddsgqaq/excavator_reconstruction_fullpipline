#!/usr/bin/env bash
# Start all three processes.  Run from any directory.
# Usage:  bash /home/maomaoyu/WS/vggt_yoloe/start_all.sh

set -e

WORKSPACE="/home/maomaoyu/WS/vggt_yoloe"
YOLOE_DIR="/home/maomaoyu/WS/yoloe"
VGGT_DIR="/home/maomaoyu/WS/vggt"

echo "=== Starting YOLOe service (port 8001) ==="
cd "$YOLOE_DIR"
# Replace 'yoloe' with your actual conda env name for YOLOe
conda run -n yoloe --no-capture-output \
    python "$WORKSPACE/yoloe_service.py" --port 8001 &
YOLOE_PID=$!
echo "YOLOe PID: $YOLOE_PID"

echo "=== Starting VGGT service (port 8002) ==="
cd "$VGGT_DIR"
# Replace 'vggt' with your actual conda env name for VGGT
conda run -n vggt50 --no-capture-output \
    python "$WORKSPACE/vggt_service.py" --port 8002 &
VGGT_PID=$!
echo "VGGT PID: $VGGT_PID"

echo "=== Waiting for services to start (10s) ==="
sleep 10

echo "=== Starting Orchestrator (port 7860) ==="
# Orchestrator only needs gradio + requests — run in either env
conda run -n vggt50 --no-capture-output \
    python "$WORKSPACE/orchestrator.py" \
        --yoloe-url http://localhost:8001 \
        --vggt-url  http://localhost:8002 \
        --port 7860 &
ORCH_PID=$!
echo "Orchestrator PID: $ORCH_PID"

echo ""
echo "All services started."
echo "  YOLOe service : http://localhost:8001/docs"
echo "  VGGT service  : http://localhost:8002/docs"
echo "  Orchestrator  : http://localhost:7860"
echo ""
echo "To stop all: kill $YOLOE_PID $VGGT_PID $ORCH_PID"

wait
