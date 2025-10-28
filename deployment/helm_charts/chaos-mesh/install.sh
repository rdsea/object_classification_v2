#!/bin/bash
helm repo add chaos-mesh https://charts.chaos-mesh.org
# K3s
helm install chaos-mesh chaos-mesh/chaos-mesh -n=chaos-mesh --set chaosDaemon.runtime=containerd --set chaosDaemon.socketPath=/run/k3s/containerd/containerd.sock --version 2.7.2 --create-namespace
