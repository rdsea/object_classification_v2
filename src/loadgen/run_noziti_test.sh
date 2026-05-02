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

  RUN_ID="$(date +%Y%m%d_%H%M%S)_$(hostname)_u${USERS}_sr${SPAWN_RATE}_rt${RUN_TIME}"
  RESULTS_DIR="$HOME/results/$RUN_ID"
  DEVICE_ID="$(hostname)"

  mkdir -p "$RESULTS_DIR"

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
    --csv "$RESULTS_DIR/${RUN_ID}" \
    --csv-full-history \
    --html "$RESULTS_DIR/${RUN_ID}.html" \
    --logfile "$RESULTS_DIR/${RUN_ID}.log" |
    tee "$RESULTS_DIR/${RUN_ID}.out"
}

# run_test 10 1 1m
# run_test 20 5 5m
# run_test 40 10 5m

run_test 1 1 5m
run_test 5 1 5m
run_test 10 2 5m
run_test 20 5 5m
run_test 40 10 5m
run_test 80 20 5m
