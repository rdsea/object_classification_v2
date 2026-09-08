#!/bin/bash

# RabbitMQ operator
kubectl apply -f https://github.com/rabbitmq/cluster-operator/releases/latest/download/cluster-operator.yml

# Scylla
kubectl exec -i scylla-0 -- cqlsh <<EOF
CREATE KEYSPACE IF NOT EXISTS object_detection
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

CREATE TABLE object_detection.results (
    id UUID PRIMARY KEY,
    timestamp timestamp,
    prediction text,
    confidence double
);
EOF
