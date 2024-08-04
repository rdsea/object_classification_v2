#!/bin/bash

export PORT=5011

for value in "$@"; do
  if [[ "--debug" = "$value" ]]; then
    fastapi dev --host 0.0.0.0 --port $PORT ensemble.py
  else
    uvicorn --host 0.0.0.0 --port $PORT ensemble:app
  fi
done
