#!/bin/bash
#
# Evaluate a CLAP-AD checkpoint on Bench2Drive-Speed (48 routes, desired-speed
# conditioned), reporting Speed-Adherence (SA) alongside Driving Score. This
# is the script that produces the `--bench2drive-speed` numbers in the paper
# (Table 2).
#
# Test-time speed control: with `--bench2drive-speed`, each route's XML
# encodes a requested speed profile; CLAP-AD's controller follows it, capped
# by whichever speed its scoring heads currently consider safe (see
# lead/inference/closed_loop_inference.py::execute_route_and_best_speed_target_speed
# and the `score_threshold` / `acceleration_aware_speed_selection` settings in
# lead/inference/config_closed_loop.py).
#
# Usage:
#   CHECKPOINT=outputs/checkpoints/clapad_tfv6 scripts_clapad/eval_clapad_b2dspeed.sh
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/clapad_tfv6}"
ROUTES="${ROUTES:-data/benchmark_routes/b2dspd_eval48.xml}"

python -u lead/leaderboard_wrapper.py \
    --checkpoint "${CHECKPOINT}" \
    --routes "${ROUTES}" \
    --bench2drive-speed \
    --port "${CARLA_RPC_PORT:-2000}" \
    --traffic-manager-port "${CARLA_TM_PORT:-8000}"
