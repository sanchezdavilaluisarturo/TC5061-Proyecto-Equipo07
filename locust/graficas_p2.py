"""Uso: python locust/graficas_p2.py reports/carga/p2/p2 reports/carga/p2/docker_stats.csv"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

prefijo, recursos = Path(sys.argv[1]), Path(sys.argv[2])
historia = pd.read_csv(f"{prefijo}_stats_history.csv")
total = historia[historia["Name"] == "Aggregated"].copy()
total = total[total["User Count"] > 0]
t0 = total["Timestamp"].iloc[0]
total["t"] = total["Timestamp"] - t0

fig, ax = plt.subplots(figsize=(9, 4))
for col in ["50%", "95%"]:
    ax.plot(total["t"], pd.to_numeric(total[col], errors="coerce"), label=f"p{col[:-1]}")
ax2 = ax.twinx()
ax2.step(total["t"], total["User Count"], color="gray", alpha=0.4, where="post")
ax2.set_ylabel("usuarios")
ax.set_yscale("log")
ax.set(xlabel="segundos", ylabel="latencia (ms)", title="P2: latencia vs usuarios")
ax.legend()
fig.tight_layout()
fig.savefig(f"{prefijo}_latencia.png", dpi=150)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(total["t"], total["Requests/s"], label="peticiones/s")
ax.plot(total["t"], total["Failures/s"], label="errores/s", color="red")
por_nombre = historia.pivot_table(index="Timestamp", columns="Name", values="Requests/s")
pred = por_nombre["POST /p2/predecir [1 estacion]"] + 472 * por_nombre["POST /p2/predecir [472 estaciones]"]
pred = pred.loc[pred.index >= t0]
ax2 = ax.twinx()
ax2.plot(pred.index - t0, pred.values, color="green", alpha=0.6, label="predicciones/s")
ax2.set_ylabel("predicciones/s")
ax2.legend(loc="upper right")
ax.set(xlabel="segundos", ylabel="por segundo", title="P2: throughput y errores")
ax.legend()
fig.tight_layout()
fig.savefig(f"{prefijo}_rps_errores.png", dpi=150)

rec = pd.read_csv(recursos)
rec = rec[(rec["timestamp"] >= t0) & (rec["timestamp"] <= total["Timestamp"].iloc[-1])]
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(rec["timestamp"] - t0, rec["cpu_pct"], label="CPU % (200 = 2 vCPU)")
ax.plot(rec["timestamp"] - t0, rec["mem_pct"], label="memoria % (de 4 GB)")
ax.set(xlabel="segundos", ylabel="%", title="P2: recursos del contenedor")
ax.legend()
fig.tight_layout()
fig.savefig(f"{prefijo}_recursos.png", dpi=150)