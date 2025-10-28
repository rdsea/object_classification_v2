#!/bin/bash

# Deploy cert-manager
helm repo add jetstack https://charts.jetstack.io
helm repo update
helm install cert-manager jetstack/cert-manager --set crds.enabled=true --namespace cert-manager --create-namespace

# Deploy the CRD
kubectl kustomize "https://github.com/redpanda-data/redpanda-operator//operator/config/crd?ref=v2.4.2" |
  kubectl apply --server-side -f -

# Deploy the operator
helm repo add redpanda https://charts.redpanda.com
helm repo update redpanda
helm upgrade --install redpanda-controller redpanda/operator \
  --namespace redpanda \
  --create-namespace \
  --version v2.4.2
