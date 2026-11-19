#!/usr/bin/env bash
# Exporta de Dozzle (EC2) la CPU y memoria del contenedor de la API, una muestra por segundo, en el
# mismo formato que docker_stats.sh para que graficas_p1.py lo lea. Sin SSH: Dozzle guarda las
# últimas muestras en memoria, así que se corre JUSTO al terminar Locust (como comparar_agrupado.sh).
# Uso: locust/exportar_dozzle.sh salida.csv [segundos]
set -euo pipefail
SALIDA="${1:?uso: $0 salida.csv [segundos]}"; SEG="${2:-330}"
DOZZLE="${DOZZLE:-https://stats.equipo7mlops.com}"
mkdir -p "$(dirname "$SALIDA")"
FIN=$(date +%s)
# El flujo de eventos no termina solo: curl se corta a los 5 s, por eso el || true
{ curl -s -N -m 5 "$DOZZLE/api/events/stream" || true; } | python3 -I -c '
import csv, json, sys
salida, seg, fin = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
vistos = set()
for bloque in sys.stdin.read().split("\n\n"):
    if "containers-changed" not in bloque:
        continue
    for c in json.loads(bloque.split("data: ", 1)[1]):
        vistos.add(c["name"])
        if c["name"].lstrip("/") == "tc5061-equipo07-api":
            cpu, mem = c["stats"]["cpu"][-seg:], c["stats"]["memory"][-seg:]
            with open(salida, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "cpu_pct", "mem_uso", "mem_pct"])
                # La última muestra es la de este momento; las anteriores, un segundo antes cada una
                for i, (c_, m_) in enumerate(zip(cpu, mem)):
                    w.writerow([fin - (len(cpu) - 1 - i), c_, "", m_])
            print(f"{len(cpu)} muestras en {salida}")
            sys.exit(0)
sys.exit(f"No se encontró tc5061-equipo07-api en Dozzle; contenedores vistos: {sorted(vistos)}")
' "$SALIDA" "$SEG" "$FIN"
