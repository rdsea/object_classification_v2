# Helm Charts

This directory contains Helm charts for deploying the telemetry stack and other components of the RCA@Edge framework.

## Endpoints

Once the charts are deployed, the following endpoints will be available:

### Observability

*   **Prometheus:** `http://<my-gateway>/prometheus/`
*   **Grafana:** `http://<my-gateway>/grafana`
*   **Jaeger:** `http://<my-gateway>/jaeger`
*   **Hubble:** `http://<my-gateway>/hubble/`

### Applications

*   **Longhorn:** `http://<longhorn-gateway>:8000`
*   **Redpanda:** `http://<redpanda-gateway>:8080`