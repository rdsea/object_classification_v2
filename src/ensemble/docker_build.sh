#!/bin/bash
sudo docker build --platform linux/arm64 -t rdsea/object_classification_ensemble:v1 -f ./Dockerfile .
