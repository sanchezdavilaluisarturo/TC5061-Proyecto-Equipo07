"""Monitoreo de P1 (demanda por estación-hora): conjuntos, drift, versión alternativa y A/B testing.

Usa la base común de monitoreo.py. Los conjuntos salen de dos tablas:
    referencia, valid y actual   la tabla de entrenamiento (p1_demanda_estacion_hora.parquet),
                                 la misma con la que se entrenó y evaluó el champion
    evaluacion                   el mes nuevo, calculado con el mismo pipeline de features sobre
                                 la historia extendida, porque sus rezagos necesitan los días previos

Uso: python -m tc5061_proyecto_equipo07.modeling.monitoreo_p1 tabla
"""

from loguru import logger
import pandas as pd
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import features, monitoreo

app = typer.Typer(add_completion=False)

# Columnas que se comparan entre la tabla de entrenamiento y la extendida
P1_COLUMNAS = [*cfg.P1_VARIABLES, "hora_inicio", "viajes"]
P1_LLAVE = ["origen_id", "hora_inicio"]


def p1_comparar_tablas(entrenamiento: pd.DataFrame, extendida: pd.DataFrame) -> dict[str, int]:
    """Comprueba que la tabla extendida reproduce, fila por fila, la tabla de entrenamiento.

    Cada estación-hora de entrenamiento debe estar en la extendida con los mismos valores: si no,
    el mes nuevo se estaría calculando distinto que los datos del champion (train-serving skew).
    La extendida puede traer filas de más antes del mes nuevo: las horas posteriores al último viaje
    de una estación en el rango original, que con el mes nuevo ya caen dentro de su operación.
    """
    fin = pd.Timestamp(cfg.FRONTERA_MONITOREO)
    previa = extendida[extendida["hora_inicio"] < fin]
    unidas = entrenamiento[P1_COLUMNAS].merge(
        previa[P1_COLUMNAS], on=P1_LLAVE, how="left", suffixes=("", "_ext"), indicator=True
    )
    faltantes = int((unidas["_merge"] == "left_only").sum())
    if faltantes:
        raise ValueError(f"{faltantes} estación-horas de entrenamiento no están en la extendida")

    distintas = pd.Series(False, index=unidas.index)
    for columna in P1_COLUMNAS:
        if columna in P1_LLAVE:
            continue
        original, nueva = unidas[columna], unidas[f"{columna}_ext"]
        if isinstance(original.dtype, pd.CategoricalDtype):
            original, nueva = original.astype(str), nueva.astype(str)
        iguales = (original == nueva) | (original.isna() & nueva.isna())
        if not iguales.all():
            logger.warning(f"{columna}: {int((~iguales).sum())} valores distintos")
        distintas |= ~iguales
    if distintas.any():
        raise ValueError(f"{int(distintas.sum())} estación-horas cambian en la tabla extendida")
    return {"filas_comparadas": len(unidas), "filas_extra": len(previa) - len(unidas)}


def p1_evaluacion(extendida: pd.DataFrame) -> pd.DataFrame:
    """Estación-horas del mes de evaluación, etiquetadas con su periodo."""
    evaluacion = extendida[extendida["hora_inicio"] >= pd.Timestamp(cfg.FRONTERA_MONITOREO)]
    evaluacion = evaluacion.drop(columns="particion").reset_index(drop=True)
    evaluacion["periodo"] = monitoreo.asignar_periodo(evaluacion["hora_inicio"])
    if (evaluacion["periodo"] != "evaluacion").any():
        raise ValueError("La tabla de evaluación trae horas fuera del mes de monitoreo")
    return evaluacion


def p1_conjuntos(
    ruta_entrenamiento=cfg.RUTA_P1_DEMANDA, ruta_evaluacion=cfg.P1_RUTA_EVALUACION
) -> dict[str, pd.DataFrame]:
    """Regresa referencia, valid, actual y evaluacion; las horas en `brecha` se descartan.

    Se etiqueta con `monitoreo.asignar_periodo`, no con la partición de entrenamiento: así las
    24 h previas al mes nuevo también quedan en `brecha`, como en las demás fronteras.
    """
    for ruta, comando in [
        (ruta_entrenamiento, "make features-p1"),
        (ruta_evaluacion, "make monitoreo-p1-tabla"),
    ]:
        if not ruta.exists():
            raise FileNotFoundError(f"Falta {ruta.name}; ejecutar `{comando}`")
    tabla = pd.read_parquet(ruta_entrenamiento, columns=P1_COLUMNAS)
    tabla["periodo"] = monitoreo.asignar_periodo(tabla["hora_inicio"])
    evaluacion = pd.read_parquet(ruta_evaluacion, columns=[*P1_COLUMNAS, "periodo"])
    conjuntos = {
        periodo: tabla[tabla["periodo"] == periodo].reset_index(drop=True)
        for periodo in ["referencia", "valid", "actual"]
    }
    conjuntos["evaluacion"] = evaluacion
    return conjuntos


@app.callback()
def p1_ayuda_monitoreo():
    """Monitoreo de P1: tabla del mes nuevo, drift, versión alternativa y A/B testing."""


@app.command()
def tabla():
    """Calcula las estación-horas del mes nuevo y las verifica contra la tabla de entrenamiento."""
    if not cfg.RUTA_VIAJES_LIMPIOS_EXTENDIDOS.exists():
        raise FileNotFoundError(
            f"Falta {cfg.RUTA_VIAJES_LIMPIOS_EXTENDIDOS.name}; ejecutar `make monitoreo-datos`"
        )
    if not cfg.RUTA_P1_DEMANDA.exists():
        raise FileNotFoundError(f"Falta {cfg.RUTA_P1_DEMANDA.name}; ejecutar `make features-p1`")
    limpios = pd.read_parquet(
        cfg.RUTA_VIAJES_LIMPIOS_EXTENDIDOS,
        columns=["inicio_del_viaje", "origen_id", "periodo_archivo"],
    )
    extendida = features.p1_tabla_modelado(
        limpios, features.p1_cargar_estaciones(), cfg.MES_INICIO, cfg.MES_MONITOREO
    )
    del limpios

    comparacion = p1_comparar_tablas(pd.read_parquet(cfg.RUTA_P1_DEMANDA), extendida)
    logger.info(
        f"Tabla de entrenamiento reproducida: {comparacion['filas_comparadas']} estación-horas "
        f"idénticas; {comparacion['filas_extra']} filas de más antes de {cfg.MES_MONITOREO}"
    )
    evaluacion = p1_evaluacion(extendida)
    evaluacion.to_parquet(cfg.P1_RUTA_EVALUACION, index=False)
    logger.success(
        f"{cfg.P1_RUTA_EVALUACION.name}: {len(evaluacion)} estación-horas, "
        f"{int(evaluacion['viajes'].sum())} viajes, "
        f"{evaluacion['origen_id'].nunique()} estaciones"
    )


if __name__ == "__main__":
    app()
