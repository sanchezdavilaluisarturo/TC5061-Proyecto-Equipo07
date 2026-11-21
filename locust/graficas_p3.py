"""Gráficas del reporte de carga de P3 a partir de los CSV que escribe Locust (--csv).

Uso: python locust/graficas_p3.py reports/carga/p3 [reports/carga/docker_stats_p3.csv]
Genera reports/carga/p3_latencia.png, p3_rps_errores.png y p3_tamano.png.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

prefijo = Path(sys.argv[1])
historia = pd.read_csv(f"{prefijo}_stats_history.csv")
# Fila "Aggregated" = todas las peticiones; el resto es por nombre de petición
total = historia[historia["Name"] == "Aggregated"].copy()
total["t"] = total["Timestamp"] - total["Timestamp"].iloc[0]

fig, ax = plt.subplots(figsize=(9, 4))
for col, etiqueta in [("50%", "p50"), ("95%", "p95"), ("99%", "p99")]:
    ax.plot(total["t"], total[col], label=etiqueta)
ax2 = ax.twinx()
ax2.step(total["t"], total["User Count"], color="gray", alpha=0.4, where="post")
ax2.set_ylabel("usuarios")
ax.set(xlabel="segundos", ylabel="latencia (ms)", title="P3: latencia vs usuarios")
ax.legend()
fig.tight_layout()
fig.savefig(f"{prefijo}_latencia.png", dpi=150)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(total["t"], total["Requests/s"], label="peticiones/s")
ax.plot(total["t"], total["Failures/s"], label="errores/s", color="red")
ax.set(xlabel="segundos", ylabel="por segundo", title="P3: throughput y errores")
ax.legend()
fig.tight_layout()
fig.savefig(f"{prefijo}_rps_errores.png", dpi=150)

resumen = pd.read_csv(f"{prefijo}_stats.csv")
resumen = resumen[resumen["Name"] != "Aggregated"]
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(resumen["Name"], resumen["95%"])
ax.set(xlabel="p95 (ms)", title="P3: p95 por tamaño de petición")
fig.tight_layout()
fig.savefig(f"{prefijo}_tamano.png", dpi=150)
print("Listo:", *sorted(prefijo.parent.glob(prefijo.name + "_*.png")), sep="\n  ")
