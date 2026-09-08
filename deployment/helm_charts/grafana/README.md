# Grafana chart

## Get admin password

- Get the password from the secret

```bash
kubectl get secret --namespace observe grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
```

## Adding loki as data source

- Set url to `http://loki-gateway.observe.svc.cluster.local`
- Set loki to use single tenant
- Set `X-Scope-OrgID` to `empty`
- Following [https://stackoverflow.com/questions/73205562/unable-to-add-grafana-loki-datasource-in-kubernetes/76754963#76754963]

## Dashboard to use:

- Loki log app: https://grafana.com/grafana/dashboards/18042-logging-dashboard-via-loki-v2/
- Hubble metrics: https://grafana.com/grafana/dashboards/16613-hubble/
- Longhorn: https://grafana.com/grafana/dashboards/17626-longhorn-example-v1-4-0/
