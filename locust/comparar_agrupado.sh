#!/usr/bin/env bash
# Corre los tres modos de segmentar la red completa, uno tras otro, con los mismos usuarios, y
# guarda la CPU del contenedor (Dozzle) al terminar cada uno. Uso: comparar_agrupado.sh URL [usuarios] [segundos]
set -euo pipefail
HOST="${1:?uso: $0 URL [usuarios] [segundos]}"; USUARIOS="${2:-5}"; SEG="${3:-150}"
DOZZLE="${DOZZLE:-https://stats.equipo7mlops.com}"
SALIDA=reports/carga/agrupado; mkdir -p "$SALIDA"
for clase in UsuarioIndividual UsuarioLotes UsuarioRedCompleta; do
  echo "== $clase ($USUARIOS usuarios, ${SEG}s)"
  locust -f locust/locustfile_p3_agrupado.py "$clase" --headless --host "$HOST" \
    -u "$USUARIOS" -r 1 -t "${SEG}s" --csv "$SALIDA/$clase" > "$SALIDA/$clase.txt" 2>&1
  # Las últimas ~SEG muestras de CPU y memoria del contenedor de la API
  curl -s -N -m 5 "$DOZZLE/api/events/stream" | python3 -I -c "
import sys, json
for b in sys.stdin.read().split('\n\n'):
    if 'containers-changed' in b:
        for c in json.loads(b.split('data: ', 1)[1]):
            if 'api' in c['name']:
                n = $SEG
                json.dump({'cpu': c['stats']['cpu'][-n:], 'memoria': c['stats']['memory'][-n:]}, open('$SALIDA/$clase.dozzle.json', 'w'))
" || true
  sleep 20  # que la CPU baje antes del siguiente modo
done
