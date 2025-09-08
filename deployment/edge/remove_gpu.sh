#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
apply_hpa=false
for arg in "$@"; do
  if [ "$arg" == "--hpa" ]; then
    apply_hpa=true
  fi
done
kubectl delete -f "$SCRIPT_DIR/EfficientNetB0_gpu.yaml"
kubectl delete -f "$SCRIPT_DIR/MobileNetV2_gpu.yaml"
kubectl delete -f "$SCRIPT_DIR/ensemble.yaml"
kubectl delete -f "$SCRIPT_DIR/preprocessing.yaml"

if [ "$apply_hpa" = true ]; then
  kubectl delete -f hpa.yaml
fi
