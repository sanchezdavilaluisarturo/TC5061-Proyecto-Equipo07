#!/usr/bin/env bash
# Guarda CPU y memoria del contenedor cada ~2 s mientras corre la prueba (Ctrl+C para parar).
# Uso: locust/docker_stats.sh [contenedor] [salida.csv]
set -euo pipefail
CONT="${1:-tc5061-equipo07-api-local}"; SALIDA="${2:-reports/carga/p2/docker_stats.csv}"
mkdir -p "$(dirname "$SALIDA")"
echo "timestamp,cpu_pct,mem_uso,mem_pct" > "$SALIDA"
while true; do
  docker stats --no-stream --format '{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}' "$CONT" \
    | sed "s/%//g; s/^/$(date +%s),/" >> "$SALIDA"
done