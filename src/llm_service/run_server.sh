#!/bin/bash

export PORT=5012

CMD="uvicorn --host 0.0.0.0 --port $PORT llm_service:app"

for value in "$@"; do
  if [[ "$value" == "--debug" ]]; then
    CMD="fastapi dev --host 0.0.0.0 --port $PORT llm_service.py"
    break
  fi
done

$CMD
