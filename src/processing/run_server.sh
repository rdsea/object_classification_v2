#!/bin/bash

export PORT=5010

CMD="uvicorn --host 0.0.0.0 --port $PORT processing:app"

for value in "$@"; do
  if [[ "$value" == "--debug" ]]; then
    CMD="fastapi dev --host 0.0.0.0 --port $PORT processing.py"
    break
  fi
done

$CMD
