#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
apply_hpa=false
for arg in "$@"; do
  if [ "$arg" == "--hpa" ]; then
    apply_hpa=true
  fi
done

kubectl apply -f "$SCRIPT_DIR/EfficientNetB0.yaml"
kubectl apply -f "$SCRIPT_DIR/MobileNetV2.yaml"
kubectl apply -f "$SCRIPT_DIR/ensemble.yaml"
kubectl apply -f "$SCRIPT_DIR/preprocessing.yaml"

if [ "$apply_hpa" = true ]; then
  kubectl apply -f hpa.yaml
fi
