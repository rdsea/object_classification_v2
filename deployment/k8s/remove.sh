#!/bin/bash
kubectl delete -f EfficientNetB0.yaml
kubectl delete -f MobileNetV2.yaml
kubectl delete -f ensemble.yaml
kubectl delete -f preprocessing.yaml
