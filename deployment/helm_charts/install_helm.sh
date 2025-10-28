#!/bin/bash

set -e

helm repo add cilium https://helm.cilium.io/
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add longhorn https://charts.longhorn.io
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
helm repo update

# Cilium
helm install cilium cilium/cilium --version 1.18.3 \
  --namespace kube-system --values values.yaml
kubectl wait --namespace kube-system --for=condition=Available deploy --all --timeout=300s

# Prometheus
kubectl create namespace dashboard || true
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n observe --values values.yaml --create-namespace --version 75.12.0

# Longhorn
helm install longhorn longhorn/longhorn \
  --namespace longhorn-system --create-namespace --version 1.9.0 --values values.yaml
kubectl wait --namespace longhorn-system --for=condition=Available deploy --all --timeout=300s
kubectl wait --namespace longhorn-system --for=condition=Ready pod --all --timeout=300s

# Jaeger
helm install jaeger jaegertracing/jaeger \
  -n observe --create-namespace -f values.yaml --version 3.4.1
kubectl wait --namespace observe --for=condition=Available deploy --all --timeout=300s

# Otel collector
helm install my-opentelemetry-collector open-telemetry/opentelemetry-collector \
  -f values.yaml -n observe --version 0.129.0
kubectl wait --namespace observe --for=condition=Available deploy --all --timeout=300s
