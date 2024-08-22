#!/bin/bash
# NOTE: Need to be run in the parent directory of the 'inference' directory

if [ "$1" == "cpu" ] || [ "$1" == "cuda" ]; then
  TAG=$1
else
  echo "Usage: $0 {cpu|gpu}"
  exit 1
fi

cd ..

echo ""
docker build --platform linux/arm64 -t rdsea/onnx_inference:"$TAG" -f ./inference/Dockerfile."$TAG" .
