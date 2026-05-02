#!/usr/bin/env bash
set -euo pipefail

#BASE_RESULTS_DIR="$HOME/results/noziti"
HOST_URL="http://130.233.195.234:5009"
DEVICE_ID="$(hostname)"

#mkdir -p "$BASE_RESULTS_DIR"

run_test() {
  USERS="$1"
  SPAWN_RATE="$2"
  RUN_TIME="$3"
  COOLDOWN="${4:-60}"

  RUN_ID="$(date +%Y%m%d_%H%M%S)_$(hostname)_u${USERS}_sr${SPAWN_RATE}_rt${RUN_TIME}"
  RESULTS_DIR="$HOME/results/$RUN_ID"
  DEVICE_ID="$(hostname)"

  mkdir -p "$RESULTS_DIR"

  echo "Starting RUN_ID=$RUN_ID"

  locust -f load_test_para_icsoc.py \
    --ds-path ../image/ \
    --device-id "$DEVICE_ID" \
    --run-id "$RUN_ID" \
    --results-dir "$RESULTS_DIR" \
    --host http://130.233.195.234:5009 \
    --headless \
    --users "$USERS" \
    --spawn-rate "$SPAWN_RATE" \
    --run-time "$RUN_TIME" \
    --stop-timeout 30 \
    --csv "$RESULTS_DIR/${RUN_ID}" \
    --csv-full-history \
    --html "$RESULTS_DIR/${RUN_ID}.html" \
    --logfile "$RESULTS_DIR/${RUN_ID}.log" |
    tee "$RESULTS_DIR/${RUN_ID}.out"

  echo "Finished RUN_ID=$RUN_ID"
  echo "Cooling down for ${COOLDOWN}s..."
  sleep "$COOLDOWN"
}

# run_test 10 1 1m
# run_test 20 5 5m
# run_test 40 10 5m
# keep the run-up approx around 5tx/s
# wait_time = between(0.1, 0.1) each request send wait 0.1: 1 / (response_time + 0.1s)
run_test 1 1 5m 60
run_test 5 1 5m 60
run_test 10 2 5m 60
run_test 15 3 5m 60
run_test 20 5 5m 60
run_test 30 8 5m 60
run_test 40 10 5m 60
run_test 60 15 5m 90
run_test 80 20 5m 90

# Keep this one for througput test
#run_test 10 2 5m 60    # roughly 10 req/s
#run_test 25 5 5m 60    # roughly 25 req/s
#run_test 50 10 5m 60   # roughly 50 req/s
#run_test 75 15 5m 90   # roughly 75 req/s
#run_test 100 20 5m 120 # roughly 100 req/s
