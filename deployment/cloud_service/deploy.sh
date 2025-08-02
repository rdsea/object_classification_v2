#!/bin/bash

# RabbitMQ operator
kubectl apply -f https://github.com/rabbitmq/cluster-operator/releases/latest/download/cluster-operator.yml

# Mongodb operator
helm repo add mongodb https://mongodb.github.io/helm-charts
helm repo update
helm install community-operator mongodb/community-operator --namespace mongodb --create-namespace

kubectl create secret generic mongodb-admin-password --from-literal=password=adminadminadmin --namespace mongodb
