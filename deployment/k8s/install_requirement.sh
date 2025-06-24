#!/bin/bash

# Metallb for getting external ip
kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.15.2/config/manifests/metallb-native.yaml

# Install gateway-api
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.3.0/standard-install.yaml

# CILIUM_CLI_VERSION=$(curl -s https://raw.githubusercontent.com/cilium/cilium-cli/main/stable.txt)
# CLI_ARCH=amd64
# if [ "$(uname -m)" = "aarch64" ]; then CLI_ARCH=arm64; fi
# curl -L --fail --remote-name-all https://github.com/cilium/cilium-cli/releases/download/${CILIUM_CLI_VERSION}/cilium-linux-${CLI_ARCH}.tar.gz{,.sha256sum}
# sha256sum --check cilium-linux-${CLI_ARCH}.tar.gz.sha256sum
# sudo tar xzvfC cilium-linux-${CLI_ARCH}.tar.gz /usr/local/bin
# rm cilium-linux-${CLI_ARCH}.tar.gz{,.sha256sum}

cilium install --version 1.17.4 \
  --set kubeProxyReplacement=true \
  --set gatewayAPI.enabled=true \
  --set prometheus.enabled=true \
  --set operator.prometheus.enabled=true \
  --set hubble.enabled=true \
  --set hubble.metrics.enableOpenMetrics=true \
  --set hubble.metrics.enabled="{dns,drop,tcp,flow,port-distribution,icmp,httpV2:exemplars=true;labelsContext=source_ip\,source_namespace\,source_workload\,destination_ip\,destination_namespace\,destination_workload\,traffic_direction}"

# Metric server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
