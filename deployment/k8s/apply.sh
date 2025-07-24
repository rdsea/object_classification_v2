#!/bin/bash

apply_hpa=false
for arg in "$@"; do
  if [ "$arg" == "--hpa" ]; then
    apply_hpa=true
  fi
done

kubectl apply -f EfficientNetB0.yaml
kubectl apply -f MobileNetV2.yaml
kubectl apply -f ensemble.yaml
kubectl apply -f preprocessing.yaml

if [ "$apply_hpa" = true ]; then
  kubectl apply -f hpa.yaml
fi
