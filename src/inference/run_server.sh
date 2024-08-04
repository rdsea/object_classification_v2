#!/bin/bash

# vgg_0
# vgg_2_7
# vgg_2_12
# vgg_3_6
# vgg_6
# vgg_6_7
# vgg_7
# vgg_7_6

model=""

while [[ $# -gt 0 ]]; do
  case $1 in
  --model)
    model="$2"
    shift
    ;;
  --debug)
    debug=true
    ;;
  *)
    echo "Unknown option: $1"
    exit 1
    ;;
  esac
  shift
done

if [[ -z "$model" ]]; then
  echo "Error: --model parameter is required."
  exit 1
fi

export CHOSEN_MODEL=$model
case $model in
vgg_0)
  export PORT=8001
  ;;
vgg_2_7)
  export PORT=8002
  ;;
vgg_2_12)
  export PORT=8003
  ;;
vgg_3_6)
  export PORT=8004
  ;;
vgg_6)
  export PORT=8005
  ;;
vgg_6_7)
  export PORT=8006
  ;;
vgg_7)
  export PORT=8007
  ;;
vgg_7_6)
  export PORT=8008
  ;;
*)
  echo "Error: Unknown model specified: $model"
  exit 1
  ;;
esac
if [[ "$debug" == true ]]; then
  fastapi dev --host 0.0.0.0 --port $PORT inference.py
else
  uvicorn --host 0.0.0.0 --port $PORT inference:app
fi
