#!/bin/bash

cd ..
docker build -t rdsea/object_classification_ensemble:v1 -f ./ensemble/Dockerfile .
