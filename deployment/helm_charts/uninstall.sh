#!/bin/bash

wait_for_namespace_cleanup() {
  ns=$1
  echo "Waiting for all resources to be deleted in namespace: $ns..."
  while kubectl get all -n "$ns" &>/dev/null &&
    [ "$(kubectl get all -n "$ns" --no-headers 2>/dev/null | wc -l)" -ne 0 ]; do
    sleep 2
  done
  echo "All resources deleted in namespace: $ns"
}

# Observe
helm uninstall -n observe prometheus
helm uninstall -n observe tempo
helm uninstall -n observe my-opentelemetry-collector
wait_for_namespace_cleanup observe

# Redpanda
helm uninstall -n redpanda redpanda
wait_for_namespace_cleanup redpanda

# Chaos-mesh
helm uninstall -n chaos-mesh chaos-mesh
wait_for_namespace_cleanup chaos-mesh

# Cert-manager
helm uninstall -n cert-manager cert-manager
wait_for_namespace_cleanup cert-manager

# Longhorn
kubectl -n longhorn-system patch -p '{"value": "true"}' --type=merge lhs deleting-confirmation-flag
kubectl delete pvc -A --all
helm uninstall -n longhorn-system longhorn
wait_for_namespace_cleanup longhorn-system

# Cilium
helm uninstall -n kube-system cilium
# Only wait for cilium-related resources, not full kube-system cleanup
echo "Waiting for Cilium resources to be deleted in kube-system..."
while kubectl get pods -n kube-system -l k8s-app=cilium 2>/dev/null | grep -q cilium; do
  sleep 2
done
echo "Cilium resources deleted in kube-system"

# Metallb
kubectl delete -f https://raw.githubusercontent.com/metallb/metallb/v0.15.2/config/manifests/metallb-native.yaml
kubectl delete -n metallb-system ipaddresspools.metallb.io default-ipaddresspool
kubectl delete -n metallb-system l2advertisements.metallb.io default-advertisement
wait_for_namespace_cleanup metallb-system

# Gateway API
kubectl delete -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.3.0/standard-install.yaml

# Metric server
kubectl delete -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
# Wait for metric-server to be deleted
echo "Waiting for metrics-server resources to be deleted..."
while kubectl get deployment -A | grep -q metrics-server; do
  sleep 2
done
echo "Metrics-server deleted"
