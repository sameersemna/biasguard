# Grafana + Prometheus Guide

This project can auto-provision a Prometheus datasource in Grafana on container startup.

## What is provisioned

- Datasource name: `Prometheus`
- Datasource UID: `prometheus`
- URL: `http://latitude:9090`
- File: `docker/grafana/datasources/prometheus.yml`

## Compose wiring

Grafana mounts provisioning folders from the repo:

- `./docker/grafana/datasources` → `/etc/grafana/provisioning/datasources`
- `./docker/grafana/dashboards` → `/etc/grafana/provisioning/dashboards`

## First startup (fresh)

Run:

```bash
docker compose up --build -d
```

Then open Grafana at `http://localhost:3000`.

Default login in this repo:

- User: `admin`
- Password: `biasguard`

Go to **Connections → Data sources** and verify that `Prometheus` exists.

Prometheus is expected to run outside this Docker Compose stack at `http://latitude:9090`.

## Existing Grafana volume behavior

Provisioning runs at Grafana startup. If you already had a populated `grafana_data` volume, UI state may persist from older runs.

To force a clean Grafana state:

```bash
docker compose down -v
docker compose up --build -d
```

## Quick validation

1. Grafana: **Explore**
2. Datasource: `Prometheus`
3. Query: `up`
4. Run query

If `up` returns series, Grafana ↔ Prometheus is connected.

## Troubleshooting

- **Datasource missing**
  - Check container mount paths in `docker-compose.yml`.
  - Check provisioning file syntax in `docker/grafana/datasources/prometheus.yml`.
- **Datasource exists but query fails**
  - Confirm external Prometheus health: `http://latitude:9090/-/healthy`.
- **Changed provisioning but nothing updates**
  - Restart Grafana container: `docker compose restart grafana`.
  - If still stale, reset volume with `docker compose down -v`.
