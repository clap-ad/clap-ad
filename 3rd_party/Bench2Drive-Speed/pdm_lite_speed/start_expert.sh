#!/bin/bash

# This script starts PDM-Lite and the CARLA simulator on a local machine

# Make sure any previously started Carla simulator instance is stopped
# Sometimes calling pkill Carla only once is not enough.
pkill Carla
pkill Carla
pkill Carla

term() {
  echo "Terminated Carla"
  pkill Carla
  pkill Carla
  pkill Carla
  exit 1
}
trap term SIGINT

export CARLA_ROOT=${YOUR_CARLA_PATH}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg

export WORK_DIR=/PATH/TO/PDMLITESPEED
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/scenario_runner
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/leaderboard
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard

export REPETITIONS=1
export DEBUG_CHALLENGE=0

export PTH_ROUTE=${WORK_DIR}/leaderboard/data/test

# Function to handle errors
handle_error() {
  pkill Carla
  exit 1
}

# Set up trap to call handle_error on ERR signal
trap 'handle_error' ERR

# Start the carla server
export PORT=23333
sh ${CARLA_SERVER} -RenderOffScreen -nosound -carla-streaming-port=0 -carla-rpc-port=${PORT} &
sleep 20 # on a fast computer this can be reduced to sth. like 6 seconds

echo 'Port' $PORT

export TEAM_AGENT=${WORK_DIR}/team_code/data_agent.py # use autopilot.py here to only run the expert without data generation
export CHALLENGE_TRACK_CODENAME=MAP
export ROUTES=${PTH_ROUTE}.xml
export TM_PORT=$((PORT + 3))

export CHECKPOINT_ENDPOINT=${PTH_ROUTE}.json
export TEAM_CONFIG=${PTH_ROUTE}.xml
export PTH_LOG='logs'
export RESUME=1
export DATAGEN=1
export SAVE_PATH='output/raw/pdm_lite_speed'
export TM_SEED=0

export P_START=0.0
export P_END=999999.0
export P_EXIT=999999.0
export P_MOVE=3.0
export NO_DESTROY=0 # do not despawn vehicles
export RECOVER=0
export DO_FAIL=0
export DO_OOD=0
export DO_ACC=0
export DO_DEC=0

export RANDOMIZE_TRAFFIC=1
export SPD_MAX_RAND_RATIO=0.2
export SPD_MIN_RAND_RATIO=0.3
export POS_RAND_DISTANCE=25.0
export VEHICLE_RAND_POS_RATIO=2.0
export VEHICLE_RAND_NUM_RATIO=3.0
export VEHICLE_DIST_MIN=5.0
export SHUFFLE_WEATHER=1
export VQA_GEN=1
export VQA_LITE=1 # generate small amount of vqas
export STRICT_MODE=1
export EXPERT_NAME='pdm_lite_speed'

# Start the actual evaluation / data generation
python leaderboard/leaderboard/leaderboard_evaluator_local.py --port=${PORT} --traffic-manager-port=${TM_PORT} --routes=${ROUTES} --repetitions=${REPETITIONS} --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} --agent-config=${TEAM_CONFIG} --debug=0 --resume=${RESUME} --timeout=2000 --traffic-manager-seed=${TM_SEED}

# Kill the Carla server afterwards
pkill Carla
