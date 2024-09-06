#!/bin/bash

# Default values
debug=false
PORT=8010
CHOSEN_MODEL="MobileNet"
# "efficientnet-edgetpu-L_quant",
# "efficientnet-edgetpu-M_quant",
# "efficientnet-edgetpu-S_quant",
# "inception_v1_224_quant",
# "inception_v2_224_quant",
# "inception_v4_299_quant",
# "tf2_mobilenet_v1_1.0_224_ptq",
# "tf2_mobilenet_v2_1.0_224_ptq",
# "tf2_mobilenet_v3_edgetpu_1.0_224_ptq"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --debug) debug=true ;;
  --port)
    PORT="$2"
    shift
    ;;
  --model)
    CHOSEN_MODEL="$2"
    shift
    ;;
  *)
    echo "Unknown parameter passed: $1"
    exit 1
    ;;
  esac
  shift
done

export PORT
export CHOSEN_MODEL

if [[ "$debug" == true ]]; then
  fastapi dev --host 0.0.0.0 --port "$PORT" inference.py
else
  uvicorn --host 0.0.0.0 --port "$PORT" inference:app
fi
