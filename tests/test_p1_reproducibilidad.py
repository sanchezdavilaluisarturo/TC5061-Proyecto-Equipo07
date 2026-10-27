"""Pruebas de reproducibilidad de P1: misma semilla, mismas predicciones, métricas y búsqueda."""

import mlflow
import numpy as np
import optuna
import pytest
from test_tuning_p1 import conjuntos_sinteticos

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.modeling import tuning_p1
from tc5061_proyecto_equipo07.modeling.train import p1_construir_modelo, p1_evaluar

# subsample y colsample_bytree muestrean filas y columnas: el resultado depende de la semilla
HIPERPARAMETROS_CON_AZAR = {"subsample": 0.7, "colsample_bytree": 0.7}

MODELOS = [
    ("baseline", None),
    ("hgb", None),
    ("xgb", None),
    ("xgb_optuna", HIPERPARAMETROS_CON_AZAR),
]


def entrenar_y_evaluar(clave, hiperparametros, conjuntos, semilla=cfg.SEMILLA):
    """Entrena con train y regresa las métricas y predicciones en valid."""
    modelo = p1_construir_modelo(clave, semilla=semilla, hiperparametros=hiperparametros)
    train = conjuntos["train"]
    modelo.fit(train[cfg.P1_VARIABLES], train["viajes"])
    return p1_evaluar(modelo, conjuntos["valid"], "valid")


@pytest.mark.parametrize(("clave", "hiperparametros"), MODELOS)
def test_p1_misma_semilla_da_las_mismas_predicciones_y_metricas(clave, hiperparametros):
    conjuntos = conjuntos_sinteticos()
    metricas_1, prediccion_1 = entrenar_y_evaluar(clave, hiperparametros, conjuntos)
    metricas_2, prediccion_2 = entrenar_y_evaluar(clave, hiperparametros, conjuntos)
    np.testing.assert_array_equal(prediccion_1, prediccion_2)
    assert metricas_1 == metricas_2


def test_p1_otra_semilla_cambia_el_modelo_con_muestreo():
    """Contraste: si la semilla no llegara al modelo, la prueba anterior pasaría sin probar nada."""
    conjuntos = conjuntos_sinteticos()
    _, prediccion_42 = entrenar_y_evaluar("xgb_optuna", HIPERPARAMETROS_CON_AZAR, conjuntos)
    _, prediccion_7 = entrenar_y_evaluar(
        "xgb_optuna", HIPERPARAMETROS_CON_AZAR, conjuntos, semilla=7
    )
    assert not np.array_equal(prediccion_42, prediccion_7)
