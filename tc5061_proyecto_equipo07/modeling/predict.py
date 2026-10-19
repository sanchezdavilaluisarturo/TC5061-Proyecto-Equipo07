"""Inferencia con modelos ya ajustados.

P1 (demanda por estación-hora): predecir los viajes del día siguiente por estación-hora y sumarlos
por zona (municipio), que es como lo usa el operador para planear la flota.

P2 (riesgo de estación): convertir las probabilidades del modelo en una clase de riesgo con los
umbrales elegidos en validación.

P3 (clustering de estaciones): asignar grupos con un modelo ya ajustado. Se usa para medir la
estabilidad de los grupos en otro periodo (el año de validación) y, más adelante, para asignar
grupo a las estaciones nuevas, que no tienen historia en entrenamiento.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from tc5061_proyecto_equipo07 import config as cfg


def p1_predecir(modelo, datos: pd.DataFrame) -> np.ndarray:
    """Viajes esperados por estación-hora; `datos` trae al menos las columnas de P1_VARIABLES."""
    return modelo.predict(datos[cfg.P1_VARIABLES])


def p1_predecir_por_zona(datos: pd.DataFrame, prediccion: np.ndarray) -> pd.DataFrame:
    """Suma por día y zona de los viajes predichos (y de los reales, si `datos` los trae).

    La pregunta pide estación o zona: el modelo predice por estación y la zona se obtiene sumando,
    así las dos vistas son consistentes entre sí.
    """
    tabla = pd.DataFrame(
        {
            "dia": datos["hora_inicio"].dt.normalize(),
            "zona": datos["zona"],
            "prediccion": prediccion,
        }
    )
    if "viajes" in datos:
        tabla["real"] = datos["viajes"]
    return tabla.groupby(["dia", "zona"], observed=True).sum().reset_index()


def p3_reasignar_grupos(
    modelo, caracteristicas: pd.DataFrame, columnas_modelo: list[str]
) -> np.ndarray:
    """Regresa el grupo más cercano de cada estación según el modelo ajustado en entrenamiento.

    `caracteristicas` trae las variables de otro periodo (por ejemplo, validación); el modelo
    ya incluye su escalador, así que no hay que estandarizarlas a mano.
    """
    return modelo.predict(caracteristicas[columnas_modelo])


def p3_estabilidad_ari(grupo_original, grupo_reasignado) -> float:
    """Índice de Rand ajustado entre dos asignaciones (1 = idénticas, 0 = azar)."""
    return float(adjusted_rand_score(grupo_original, grupo_reasignado))


def p2_decidir_clase(p: np.ndarray, umbrales: dict[int, float]) -> np.ndarray:
    """Clase predicha: el riesgo cuyo score relativo a su umbral sea mayor y al menos 1."""
    score_bicis, score_anclajes = p[:, 1] / umbrales[1], p[:, 2] / umbrales[2]
    pred = np.zeros(len(p), dtype=int)
    pred[(score_bicis >= 1) & (score_bicis >= score_anclajes)] = 1
    pred[(score_anclajes >= 1) & (score_anclajes > score_bicis)] = 2
    return pred
