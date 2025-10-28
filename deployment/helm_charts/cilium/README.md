# Deploy Cilium cluster mesh

- [link](https://docs.cilium.io/en/stable/network/clustermesh/clustermesh/)

## Prerequisites

- Remember that each cluster should have unique podCIDR: following setting in cilium helm value, for each cluster, use the CIDR that match the cluster id. For example, cluster id 1 => 11.0.0.0/8

## Running

- Add dns name in cloud cluster to be connected to gateway in edge cluster by adding this to coredns configmap

```bash
kubectl edit configmap coredns -n kube-system
# Or update this in k3s
sudo nano /var/lib/rancher/k3s/server/manifests/coredns.yaml

# Add
    edge-1.svc:53 {
        hosts {
            <ip> gateway.edge-1.svc
            fallthrough
        }
    }
```
