#!/bin/bash

apply_hpa=false
for arg in "$@"; do
  if [ "$arg" == "--hpa" ]; then
    apply_hpa=true
  fi
done
kubectl delete -f EfficientNetB0.yaml
kubectl delete -f MobileNetV2.yaml
kubectl delete -f ensemble.yaml
kubectl delete -f preprocessing.yaml

if [ "$apply_hpa" = true ]; then
  kubectl delete -f hpa.yaml
fi
