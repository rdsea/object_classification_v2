#!/bin/bash

cd ..
docker build -t rdsea/preprocessing:v1 -f ./preprocessing/Dockerfile .
