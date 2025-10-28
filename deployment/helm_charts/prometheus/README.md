#### Install the Prometheus PV:

- Change the `local.path` field in PersistentVolume definition of `prometheus-pv.yml` to specify the directory on the host node's filesystem where the actual data for the PersistentVolume will be stored.
- Change the `nodeAffinity` to ensure the PV is bound to Pods scheduled on the node, e.g., `yinfangchen-1` node.
- Apply the Prometheus PV:

```shell
kubectl apply -f prometheus-pv.yml -n observe
```

#### Install Prometheus:

```shell
cd prometheus/
helm install prometheus prometheus/ -n observe
```

#### Uninstall Prometheus:

```shell
helm uninstall prometheus -n observe
```

## Popular metrics

### Node

- CPU usage over time:

```promql
sum by (instance) (1 - sum without (mode) (rate(node_cpu_seconds_total{job="node-exporter",mode=~"idle|iowait|steal"}[1m])))
```

- Load average:

```promql
node_load1
node_load5
node_load15
```

- Memory usage:

```promql
100 - (
  node_memory_MemAvailable_bytes{job="node-exporter"} /
  node_memory_MemTotal_bytes{job="node-exporter"} * 100
)
```

- Disk usage

```promql
rate(node_disk_read_bytes_total{job="node-exporter", instance="130.233.195.211:9100", device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)"}[1m])

rate(node_disk_written_bytes_total{job="node-exporter", instance="130.233.195.211:9100", device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)"}[1m])

rate(node_disk_io_time_seconds_total{job="node-exporter", instance="130.233.195.211:9100", device=~"(/dev/)?(mmcblk.p.+|nvme.+|rbd.+|sd.+|vd.+|xvd.+|dm-.+|md.+|dasd.+)"}[1m])
```

- Network

```promql
rate(node_network_receive_bytes_total{job="node-exporter", instance="130.233.195.224:9100", device!="lo"}[1m]) * 8
rate(node_network_transmit_bytes_total{job="node-exporter", instance="130.233.195.211:9100", device!="lo"}[1m]) * 8
```

### Pod

- CPU usage:

```promql
sum(node_namespace_pod_container:container_cpu_usage_seconds_total:sum_rate5m{namespace="default", pod!="", container!=""}) by (container)
```

- Memory usage:

```promql
sum(container_memory_working_set_bytes{job="kubelet", metrics_path="/metrics/cadvisor", namespace="default", container!="", image!=""}) by (container)
```

### Workload/Service

- CPU usage:

```promql
sum(
    node_namespace_pod_container:container_cpu_usage_seconds_total:sum_rate5m{namespace="default"}
  * on(namespace,pod)
    group_left(workload, workload_type) namespace_workload_pod:kube_pod_owner:relabel{namespace="default"}
) by (workload)
```

- Memory usage:

```promql
sum(
    container_memory_working_set_bytes{namespace="default", container!="", image!=""}
  * on(namespace,pod)
    group_left(workload, workload_type) namespace_workload_pod:kube_pod_owner:relabel{ namespace="default"}
) by (workload)
```

- Network:

```promql
(sum(rate(container_network_receive_bytes_total{job="kubelet", metrics_path="/metrics/cadvisor",namespace="default"}[1m])
* on (namespace,pod)
group_left(workload,workload_type) namespace_workload_pod:kube_pod_owner:relabel{ namespace="default"}) by (workload))

(sum(rate(container_network_transmit_bytes_total{job="kubelet", metrics_path="/metrics/cadvisor",namespace="default"}[1m])
* on (namespace,pod)
group_left(workload,workload_type) namespace_workload_pod:kube_pod_owner:relabel{ namespace="default"}) by (workload))
```

- IO:

```promql
(sum(rate(container_blkio_device_usage_total{job="kubelet", metrics_path="/metrics/cadvisor",namespace="default"}[1m])
* on (namespace,pod)
group_left(workload,workload_type) namespace_workload_pod:kube_pod_owner:relabel{ namespace="default"}) by (workload)) > 0
```

- Latency percentile:

```promql
histogram_quantile(0.95, sum(rate(duration_milliseconds_bucket{ span_kind =~ "SPAN_KIND_SERVER"}[10m])) by (service_name,span_name,le))
histogram_quantile(0.75, sum(rate(duration_milliseconds_bucket{ span_kind =~ "SPAN_KIND_SERVER"}[10m])) by (service_name,span_name,le))
histogram_quantile(0.50, sum(rate(duration_milliseconds_bucket{ span_kind =~ "SPAN_KIND_SERVER"}[10m])) by (service_name,span_name,le))
```

- Request rate per second:

```promql
sum(rate(calls_total{span_kind =~ "SPAN_KIND_SERVER"}[10m])) by (service_name)
```

- Error rate:

```promql
sum(rate(calls_total{status_code = "STATUS_CODE_ERROR", span_kind =~ "SPAN_KIND_SERVER"}[10m])) by (service_name,span_name) / sum(rate(calls_total{ span_kind =~ "SPAN_KIND_SERVER"}[10m])) by (service_name,span_name)
```
