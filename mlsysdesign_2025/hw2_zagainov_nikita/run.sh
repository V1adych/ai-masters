#! /bin/bash
set -euo pipefail

cd "$(dirname "$0")"

docker compose build
docker compose up -d

docker compose exec -T airflow-scheduler bash -lc "airflow dags unpause movielens_features || true; airflow dags trigger movielens_features || true"

echo "Waiting for data to appear in Redis..."
for i in {1..180}; do
  count=$(docker compose exec -T redis redis-cli DBSIZE | grep -o '[0-9]*' || echo "0")
  if [ "${count:-0}" -gt 0 ]; then
    echo "Data loaded to Redis (${count} keys)"
    exit 0
  fi
  sleep 5
done

echo "Timeout waiting for DAG to finish" >&2
exit 1
