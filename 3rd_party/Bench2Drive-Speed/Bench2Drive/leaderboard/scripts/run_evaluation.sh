#!/bin/bash
# Must set CARLA_ROOT
export CARLA_ROOT=YOUR_CARLA_PATH
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=leaderboard:$PYTHONPATH
export PYTHONPATH=leaderboard/team_code:$PYTHONPATH
export PYTHONPATH=scenario_runner:$PYTHONPATH
export PYTHONPATH=../TCP:$PYTHONPATH
export SCENARIO_RUNNER_ROOT=scenario_runner

echo "PYTHONPATH=$PYTHONPATH"
export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=$1
export TM_PORT=$2
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True
export IS_BENCH2DRIVE=$3
export PLANNER_TYPE=$9
export GPU_RANK=${10}

# TCP evaluation
export ROUTES=$4
export TEAM_AGENT=$5
export TEAM_CONFIG=$6
export CHECKPOINT_ENDPOINT=$7
export SAVE_PATH=$8

echo "Carla root = ${CARLA_ROOT}"

CUDA_VISIBLE_DEVICES=${GPU_RANK} python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--traffic-manager-port=${TM_PORT} \
--gpu-rank=${GPU_RANK} \
