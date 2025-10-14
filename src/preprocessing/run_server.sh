#!/bin/bash

export PORT=5010
export LOG_LEVEL=${LOG_LEVEL:-INFO}
LOG_LEVEL_LOWER=$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')

CMD="uvicorn --host 0.0.0.0 --port $PORT --log-level $LOG_LEVEL_LOWER preprocessing:app"

for value in "$@"; do
  if [[ "$value" == "--debug" ]]; then
    CMD="fastapi dev --host 0.0.0.0 --port $PORT preprocessing.py"
    break
  fi
done

$CMD
