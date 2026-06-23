#!/bin/bash
# kill_carla_port.sh
# Usage: ./kill_carla_port.sh <PORT>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <PORT>"
    exit 1
fi

PORT=$1

echo "Killing all CARLA-related processes using port $PORT..."

PIDS=$(pgrep -f "\-carla-rpc-port=${PORT}|\-\-port=${PORT}|\$PORT" || true)

if [ -z "$PIDS" ]; then
    echo "No matching processes found for port $PORT."
    exit 0
fi

for PID in $PIDS; do
    CMD=$(ps -p $PID -o cmd=)
    if [[ "$CMD" == *CarlaUE4* || "$CMD" == *leaderboard* || "$CMD" == *scenario_runner* ]]; then
        echo "Killing process $PID ($CMD)..."
        # if current user cannot kill, try sudo
        if ! kill -9 $PID 2>/dev/null; then
            echo "Trying sudo to kill process $PID..."
            sudo kill -9 $PID
        fi
    else
        echo "Skipping process $PID ($CMD), not CARLA/leaderboard/scenario_runner."
    fi
done

echo "Done."
