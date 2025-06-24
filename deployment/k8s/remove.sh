#!/bin/bash
kubectl delete -f EfficientNetB0.yml
kubectl delete -f MobileNetV2.yml
kubectl delete -f ensemble.yml
kubectl delete -f preprocessing.yml
