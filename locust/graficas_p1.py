"""Gráficas, tabla por escalón y costo por inferencia de la prueba de carga de P1.

Lee el CSV por petición que escribe locustfile_p1.py (--archivo-peticiones) y el de
docker_stats.sh, y calcula todo por escalón: los percentiles de stats_history de Locust son
acumulados desde el inicio y no sirven para comparar escalones.

Uso: python locust/graficas_p1.py reports/carga/p1 [--precio-hora 0.0416]
Lee p1_peticiones.csv y docker_stats.csv de esa carpeta y escribe en ella p1_escalones.csv,
p1_latencia.png, p1_rps_errores.png y p1_recursos.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from escalones_p1 import EscalonesP1
from locustfile_p1 import NOMBRE_BATCH, NOMBRE_CONSULTA, REGISTROS
import matplotlib.pyplot as plt
import pandas as pd

# EC2 t3.medium, Linux on-demand en us-east-2 (USD por hora, verificado el 18-nov-2026)
PRECIO_HORA = 0.0416
# t3.medium: 2 vCPU con piso de 20 % cada una; el exceso sostenido cuesta USD 0.05 por
# vCPU-hora en modo unlimited (página de instancias T3 de AWS)
VCPU = 2
PISO_VCPU = 0.4
RECARGO_VCPU_HORA = 0.05

# Paleta Okabe-Ito (segura para daltonismo)
AZUL = "#0072B2"
NARANJA = "#E69F00"
BERMELLON = "#D55E00"
VERDE = "#009E73"
GRIS = "#999999"
ETIQUETAS = {NOMBRE_CONSULTA: "consulta (1 estación)", NOMBRE_BATCH: "batch (red completa)"}
COLORES = {NOMBRE_CONSULTA: AZUL, NOMBRE_BATCH: NARANJA}


def asignar_escalon(peticiones: pd.DataFrame) -> pd.Series:
    """Escalón de cada petición por los usuarios activos, no por el reloj.

    Locust no cambia de escalón exacto en 60, 120... s (en la corrida local fue cada ~55 s).
    Los escalones solo suben, así que se toma el máximo acumulado: al apagar, el conteo baja.
    """
    pasos = sorted(u for _, u, _ in EscalonesP1.escalones)
    maximo = peticiones.sort_values("t")["usuarios"].cummax()
    return maximo.map(lambda u: next(p for p in pasos if p >= u))


def escalon_de_stats(stats: pd.DataFrame, peticiones: pd.DataFrame) -> pd.Series:
    """Cada muestra de CPU va al escalón cuyas peticiones cubren ese segundo."""
    limites = peticiones.groupby("escalon")["t"].agg(["min", "max"])
    intervalos = pd.IntervalIndex.from_arrays(limites["min"], limites["max"], closed="both")
    posicion = intervalos.get_indexer(stats["t"])
    return pd.Series(limites.index.to_numpy()[posicion], index=stats.index).where(posicion >= 0)


def resumir_escalones(
    peticiones: pd.DataFrame, stats: pd.DataFrame, precio: float
) -> pd.DataFrame:
    """Por escalón: peticiones, latencia, errores, CPU y USD por millón de predicciones."""
    filas = []
    for usuarios, grupo in peticiones.groupby("escalon"):
        duracion = grupo["t"].max() - grupo["t"].min()
        exitosas = grupo[grupo["exito"] == 1]
        predicciones_s = exitosas["nombre"].map(REGISTROS).sum() / duracion
        cpu = stats.loc[stats["escalon"] == usuarios, "cpu_pct"]
        exceso_vcpu = max(cpu.mean() / 100 - PISO_VCPU, 0)
        fila = {
            "usuarios": usuarios,
            "peticiones": len(grupo),
            "peticiones_s": len(grupo) / duracion,
            "predicciones_s": predicciones_s,
            "error_pct": 100 * (1 - grupo["exito"].mean()),
            "cpu_media_pct": cpu.mean(),
            "cpu_max_pct": cpu.max(),
            "mem_max_pct": stats.loc[stats["escalon"] == usuarios, "mem_pct"].max(),
        }
        for nombre, clave in [(NOMBRE_CONSULTA, "consulta"), (NOMBRE_BATCH, "batch")]:
            latencia = grupo.loc[grupo["nombre"] == nombre, "latencia_ms"]
            fila[f"{clave}_p50_ms"] = latencia.quantile(0.50)
            fila[f"{clave}_p95_ms"] = latencia.quantile(0.95)
        # USD por hora entre predicciones por hora, por millón; el recargo solo si se pasa del piso
        fila["usd_millon"] = precio / (predicciones_s * 3600) * 1e6
        fila["usd_millon_con_recargo"] = (
            (precio + exceso_vcpu * RECARGO_VCPU_HORA) / (predicciones_s * 3600) * 1e6
        )
        filas.append(fila)
    return pd.DataFrame(filas).round(4)


def graficar(peticiones, stats, tabla, carpeta: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for nombre, clave in [(NOMBRE_CONSULTA, "consulta"), (NOMBRE_BATCH, "batch")]:
        color = COLORES[nombre]
        ax.plot(
            tabla["usuarios"],
            tabla[f"{clave}_p50_ms"],
            "o-",
            color=color,
            label=f"{ETIQUETAS[nombre]} p50",
        )
        ax.plot(
            tabla["usuarios"],
            tabla[f"{clave}_p95_ms"],
            "s--",
            color=color,
            label=f"{ETIQUETAS[nombre]} p95",
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="usuarios concurrentes",
        ylabel="latencia (ms)",
        title="P1: latencia por escalón",
    )
    ax.minorticks_off()
    ax.set_xticks(tabla["usuarios"], labels=tabla["usuarios"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "p1_latencia.png", dpi=150)

    por_segundo = peticiones.assign(s=peticiones["t"].astype(int)).groupby("s")
    rps = por_segundo.size().rolling(5, min_periods=1).mean()
    errores = (1 - por_segundo["exito"].mean()).mul(100).rolling(5, min_periods=1).mean()
    usuarios = por_segundo["usuarios"].max()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rps.index, rps, color=AZUL, label="peticiones/s (media de 5 s)")
    ax.set(xlabel="segundos", ylabel="peticiones/s", title="P1: throughput y errores")
    ax2 = ax.twinx()
    ax2.plot(errores.index, errores, color=BERMELLON, label="% errores")
    ax2.step(usuarios.index, usuarios, color=GRIS, alpha=0.5, where="post", label="usuarios")
    ax2.set_ylabel("% errores / usuarios")
    fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.92))
    fig.tight_layout()
    fig.savefig(carpeta / "p1_rps_errores.png", dpi=150)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(stats["t"], stats["cpu_pct"], color=VERDE, label="CPU (% de 1 vCPU)")
    ax.plot(stats["t"], stats["mem_pct"], color=NARANJA, label="memoria (%)")
    ax.axhline(100 * VCPU, color=GRIS, linestyle=":", label=f"CPU total ({VCPU} vCPU)")
    ax.axhline(
        100 * PISO_VCPU, color=BERMELLON, linestyle="--", label="piso de créditos t3 (2 x 20 %)"
    )
    ax.set(
        xlabel="segundos",
        ylabel="%",
        title="P1: CPU y memoria del contenedor (docker stats o Dozzle)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "p1_recursos.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("carpeta", type=Path)
    parser.add_argument("--precio-hora", type=float, default=PRECIO_HORA)
    args = parser.parse_args()

    peticiones = pd.read_csv(args.carpeta / "p1_peticiones.csv")
    stats = pd.read_csv(args.carpeta / "docker_stats.csv")
    # docker_stats.sh y Locust corren en la misma máquina: el reloj es el mismo
    inicio = peticiones["timestamp"].min()
    peticiones["t"] = peticiones["timestamp"] - inicio
    stats["t"] = stats["timestamp"] - inicio
    stats = stats[stats["t"].between(0, peticiones["t"].max())].copy()
    peticiones["escalon"] = asignar_escalon(peticiones)
    stats["escalon"] = escalon_de_stats(stats, peticiones)

    tabla = resumir_escalones(peticiones, stats, args.precio_hora)
    tabla.to_csv(args.carpeta / "p1_escalones.csv", index=False)
    graficar(peticiones, stats, tabla, args.carpeta)
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
