# Centralized Grafana + Prometheus Guide

BiasGuard now uses shared observability services on latitude:

- Grafana: `http://latitude:3000`
- Prometheus: `http://latitude:9090`

No Grafana or Prometheus container is started by BiasGuard `docker-compose.yml`.

## Dashboards in this repo

Dashboard JSON files are stored in:

- `monitoring/grafana_dashboards/LLM_Operations.json`
- `monitoring/grafana_dashboards/Observability.json`

## Import dashboards to centralized Grafana

Use:

```bash
GRAFANA_USER=<admin-user> \
GRAFANA_PASSWORD='<admin-password>' \
bash scripts/import_dashboards_to_latitude_grafana.sh
```

The script imports all JSON files from `monitoring/grafana_dashboards/` into folder `BiasGuard` and overwrites existing dashboards with the same UID.
If folder `BiasGuard` does not exist, it is created automatically.

Optional override:

```bash
GRAFANA_FOLDER='BiasGuard' \
GRAFANA_USER=<admin-user> \
GRAFANA_PASSWORD='<admin-password>' \
bash scripts/import_dashboards_to_latitude_grafana.sh
```

## Quick validation

1. Open Grafana at `http://latitude:3000`.
2. Open one imported BiasGuard dashboard.
3. In Explore, run `up{job="biasguard-api"}` against datasource `Prometheus`.
4. Ensure series are returned.

## Troubleshooting

- **Unauthorized during import**
  - Verify `GRAFANA_USER` and `GRAFANA_PASSWORD` for latitude Grafana.
- **Dashboard imports but panels show no data**
  - Verify datasource `Prometheus` exists in latitude Grafana.
  - Verify Prometheus health: `http://latitude:9090/-/healthy`.
  - Verify target status: `http://latitude:9090/api/v1/targets`.
