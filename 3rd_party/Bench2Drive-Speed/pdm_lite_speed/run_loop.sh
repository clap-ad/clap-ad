#!/bin/bash
# run_forever.sh
# Usage:
#   ./run_forever.sh <START_SCRIPT> <PORT>

set -u

START_SCRIPT=$1
PORT=$2

if [ -z "$START_SCRIPT" ] || [ -z "$PORT" ]; then
    echo "Usage: $0 <START_SCRIPT> <PORT>"
    exit 1
fi

if [ ! -f "$START_SCRIPT" ]; then
    echo "Start script not found: $START_SCRIPT"
    exit 1
fi

KILL_SCRIPT=../tools/kill_carla_port.sh
KILL_ROUNDS=5
KILL_SLEEP=3
RESTART_DELAY=10
MAX_RESTARTS=12

restart_count=0

kill_carla() {
    echo "[Supervisor] Killing CARLA on port $PORT"
    for i in $(seq 1 $KILL_ROUNDS); do
        echo "[Supervisor] Kill round $i"
        $KILL_SCRIPT $PORT || true
        sleep $KILL_SLEEP
    done

    if lsof -i :$PORT; then
        echo "[Supervisor] WARNING: port $PORT still in use"
    else
        echo "[Supervisor] Port $PORT is free"
    fi
}

while true; do
    restart_count=$((restart_count + 1))

    if [ $restart_count -gt $MAX_RESTARTS ]; then
        echo "[Supervisor] Max restarts reached ($MAX_RESTARTS), exiting."
        exit 1
    fi

    echo "=========================================="
    echo "[Supervisor] Run #$restart_count"
    echo "[Supervisor] Start script: $START_SCRIPT"
    echo "[Supervisor] Port: $PORT"
    echo "=========================================="

    # Safety: ensure port is free before start
    if lsof -i :$PORT; then
        echo "[Supervisor] Port $PORT already in use, cleaning up"
        kill_carla
        sleep 5
    fi

    # Log file per run
    LOG_DIR=logs/run_$(date +%Y%m%d_%H%M%S)
    mkdir -p "$LOG_DIR"

    # Run start script in background, capture PID
    stdbuf -oL -eL bash "$START_SCRIPT" 2>&1 | tee "$LOG_DIR/stdout.log" &
    SCRIPT_PID=$!

    echo "[Supervisor] Started $START_SCRIPT with PID $SCRIPT_PID"

    # Monitor the log in background for crash patterns
    ( 
        tail -f "$LOG_DIR/stdout.log" | while read -r line; do
            if [[ "$line" == *"Signal 11 caught."* ]] || [[ "$line" == *"Segmentation fault"* ]] || [[ "$line" = *"time-out of"* ]]; then
                echo "[Supervisor] Detected CARLA crash in logs! (in '$line')"
                kill_carla
                # kill the start script if still running
                if kill -0 $SCRIPT_PID 2>/dev/null; then
                    kill -9 $SCRIPT_PID
                fi
                break
            fi
        done
    ) &

    # Wait for the start script to exit normally or be killed
    wait $SCRIPT_PID
    EXIT_CODE=$?

    echo "[Supervisor] Start script exited with code $EXIT_CODE"

    echo "[Supervisor] Cleaning up before restart"
    kill_carla

    echo "[Supervisor] Sleeping $RESTART_DELAY seconds before restart"
    sleep $RESTART_DELAY
done
