#!/usr/bin/env bash
set -euo pipefail

GRAFANA_URL="${GRAFANA_URL:-http://latitude:3000}"
GRAFANA_USER="${GRAFANA_USER:-}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-}"
DASHBOARD_DIR="${DASHBOARD_DIR:-monitoring/grafana_dashboards}"
GRAFANA_FOLDER="${GRAFANA_FOLDER:-BiasGuard}"

if [[ -z "${GRAFANA_USER}" || -z "${GRAFANA_PASSWORD}" ]]; then
  echo "ERROR: Set GRAFANA_USER and GRAFANA_PASSWORD before running."
  echo "Example: GRAFANA_USER=admin GRAFANA_PASSWORD='***' $0"
  exit 1
fi

# Resolve or create destination folder in Grafana.
folder_search=$(curl -sS -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
  "${GRAFANA_URL}/api/folders?limit=1000")
folder_uid=$(echo "${folder_search}" | jq -r --arg title "${GRAFANA_FOLDER}" '.[] | select(.title == $title) | .uid' | head -n1)

if [[ -z "${folder_uid}" || "${folder_uid}" == "null" ]]; then
  create_payload=$(jq -cn --arg title "${GRAFANA_FOLDER}" '{title: $title}')
  create_resp=$(curl -sS -w "\n%{http_code}" -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
    -H "Content-Type: application/json" \
    -X POST "${GRAFANA_URL}/api/folders" \
    -d "${create_payload}")
  create_code=$(echo "${create_resp}" | tail -n1)
  create_body=$(echo "${create_resp}" | sed '$d')

  if [[ "${create_code}" != "200" ]]; then
    echo "ERROR creating folder ${GRAFANA_FOLDER}: HTTP ${create_code}"
    echo "${create_body}"
    exit 1
  fi

  folder_uid=$(echo "${create_body}" | jq -r '.uid')
fi

if [[ -z "${folder_uid}" || "${folder_uid}" == "null" ]]; then
  echo "ERROR: Could not resolve folder UID for ${GRAFANA_FOLDER}."
  exit 1
fi

for f in "${DASHBOARD_DIR}"/*.json; do
  [[ -f "$f" ]] || continue
  payload=$(jq -cn --slurpfile d "$f" --arg folderUid "${folder_uid}" '{dashboard: $d[0], folderUid: $folderUid, overwrite: true}')
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

  echo "Imported $(basename "$f") into folder ${GRAFANA_FOLDER}"
done

echo "Dashboard import complete."
