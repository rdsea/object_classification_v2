#!/bin/bash

if [[ "$1" == "--debug" ]]; then
  debug=true
else
  debug=false
fi

export PORT=8010
export CHOSEN_MODEL=MobileNet

if [[ "$debug" == true ]]; then
  fastapi dev --host 0.0.0.0 --port $PORT inference.py
else
  uvicorn --host 0.0.0.0 --port $PORT inference:app
fi
