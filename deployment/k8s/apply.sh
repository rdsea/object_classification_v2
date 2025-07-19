#!/bin/bash
kubectl apply -f EfficientNetB0.yaml
kubectl apply -f MobileNetV2.yaml
kubectl apply -f ensemble.yaml
kubectl apply -f preprocessing.yaml
