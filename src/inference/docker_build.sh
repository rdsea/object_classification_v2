#!/bin/bash
# NOTE: Need to be run in the parent directory of the 'inference' directory
# If qemu is not available, host system may not be able to build the docker image for arm64 platform

if [ "$1" == "cpu" ] || [ "$1" == "cuda" ]; then
  TAG=$1
else
  echo "Usage: $0 {cpu|gpu}"
  exit 1
fi

cd ..

echo ""
docker build -t rdsea/onnx_inference:"$TAG" -f ./inference/Dockerfile."$TAG" .
