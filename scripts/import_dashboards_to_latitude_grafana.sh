#!/usr/bin/env bash
set -euo pipefail

GRAFANA_URL="${GRAFANA_URL:-http://latitude:3000}"
GRAFANA_USER="${GRAFANA_USER:-}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-}"
DASHBOARD_DIR="${DASHBOARD_DIR:-monitoring/grafana_dashboards}"

if [[ -z "${GRAFANA_USER}" || -z "${GRAFANA_PASSWORD}" ]]; then
  echo "ERROR: Set GRAFANA_USER and GRAFANA_PASSWORD before running."
  echo "Example: GRAFANA_USER=admin GRAFANA_PASSWORD='***' $0"
  exit 1
fi

for f in "${DASHBOARD_DIR}"/*.json; do
  [[ -f "$f" ]] || continue
  payload=$(jq -cn --slurpfile d "$f" '{dashboard: $d[0], folderId: 0, overwrite: true}')
  response=$(curl -sS -w "\n%{http_code}" -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
    -H "Content-Type: application/json" \
    -X POST "${GRAFANA_URL}/api/dashboards/db" \
    -d "${payload}")
  code=$(echo "${response}" | tail -n1)
  body=$(echo "${response}" | sed '$d')

  if [[ "${code}" != "200" ]]; then
    echo "ERROR importing $(basename "$f"): HTTP ${code}"
    echo "${body}"
    exit 1
  fi

  echo "Imported $(basename "$f")"
done

echo "Dashboard import complete."
