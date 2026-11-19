"""Uso: python locust/graficas_p2.py reports/carga/p2/p2 reports/carga/p2/docker_stats.csv
Genera <prefijo>_latencia.png, _rps_errores.png, _recursos.png y _escalones.csv (tabla por
escalón con el costo por predicción).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from escalones_p2 import EscalonesP2
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


# Tabla por escalón con el costo por predicción
PRECIO_HORA = 0.0416  # EC2 t3.medium, Linux on-demand
PISO_CPU = 40  # % de 200: piso de 20 % por vCPU del t3
RECARGO_VCPU_HORA = 0.05  # USD por vCPU-hora sobre el piso (modo unlimited)
DESCARTE = 15  # s al inicio de cada escalón, mientras entran los usuarios
CONSULTA = "POST /p2/predecir [1 estacion]"
BARRIDO = "POST /p2/predecir [472 estaciones]"

por_nombre = historia.pivot_table(
    index="Timestamp", columns="Name", values=["Requests/s", "50%", "95%"]
)
filas, inicio = [], 0
for fin, usuarios, _ in EscalonesP2.escalones:
    ventana = total[(total["t"] >= inicio + DESCARTE) & (total["t"] < fin)]
    marcas = ventana["Timestamp"]
    rps = por_nombre["Requests/s"].loc[marcas]
    t_rec = rec["timestamp"] - t0
    cpu = rec.loc[(t_rec >= inicio + DESCARTE) & (t_rec < fin), ["cpu_pct", "mem_pct"]]
    predicciones_s = (rps[CONSULTA] + 472 * rps[BARRIDO]).mean()
    peticiones_s = ventana["Requests/s"].mean()
    exceso = max(cpu["cpu_pct"].mean() - PISO_CPU, 0) / 100
    usd_hora_recargo = PRECIO_HORA + exceso * RECARGO_VCPU_HORA
    fila = {"usuarios": usuarios, "peticiones_s": peticiones_s, "predicciones_s": predicciones_s}
    # Percentiles de ventana móvil de 10 s (--csv-full-history): mediana dentro del escalón
    for clave, nombre in [("consulta", CONSULTA), ("barrido", BARRIDO)]:
        for p in ["50%", "95%"]:
            valores = pd.to_numeric(por_nombre[p][nombre].loc[marcas], errors="coerce")
            fila[f"{clave}_p{p[:-1]}_ms"] = valores.median()
    fila |= {
        "cpu_media_pct": cpu["cpu_pct"].mean(),
        "cpu_max_pct": cpu["cpu_pct"].max(),
        "mem_max_pct": cpu["mem_pct"].max(),
        "fallos_acumulados": int(ventana["Total Failure Count"].iloc[-1]),
        "usd_millon_predicciones": PRECIO_HORA / (predicciones_s * 3600) * 1e6,
        "usd_millon_predicciones_recargo": usd_hora_recargo / (predicciones_s * 3600) * 1e6,
        "usd_millon_peticiones_recargo": usd_hora_recargo / (peticiones_s * 3600) * 1e6,
    }
    filas.append(fila)
    inicio = fin
pd.DataFrame(filas).round(4).to_csv(f"{prefijo}_escalones.csv", index=False)