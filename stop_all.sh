#!/usr/bin/env bash
# Stop all three services started by start_all.sh.
# Works even if the original shell session is gone.

stop_port() {
    local port=$1
    local name=$2
    local pids
    pids=$(lsof -ti tcp:"$port" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Stopping $name (port $port, PID $pids)..."
        kill "$pids" 2>/dev/null
        # Wait up to 5s for graceful shutdown, then force-kill
        for i in $(seq 1 5); do
            sleep 1
            if ! lsof -ti tcp:"$port" &>/dev/null; then
                echo "  $name stopped."
                return
            fi
        done
        echo "  $name did not stop gracefully, force-killing..."
        kill -9 "$pids" 2>/dev/null
    else
        echo "$name (port $port): not running."
    fi
}

stop_port 8001 "YOLOe service"
stop_port 8002 "VGGT service"
stop_port 7860 "Orchestrator"

echo "Done."
