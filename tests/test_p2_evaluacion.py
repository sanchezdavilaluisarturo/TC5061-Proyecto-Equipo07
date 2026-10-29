"""Pruebas de las métricas de evaluación de P2: AP, F1, log-loss, precision y recall.
Incluye la elección de umbrales de clasificación"""

import numpy as np
import pytest

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.modeling import train
from tc5061_proyecto_equipo07.modeling.predict import p2_decidir_clase
from test_p2_modelado import p2_conjuntos_sinteticos


def test_p2_decidir_clase_compara_cada_riesgo_con_su_umbral():
    umbrales = {1: 0.2, 2: 0.4}
    p = np.array(
        [
            [0.9, 0.1, 0.0],  # ningún riesgo llega a su umbral
            [0.6, 0.3, 0.1],  # sin bicis: 1.5 veces su umbral
            [0.5, 0.1, 0.4],  # sin anclajes justo en su umbral
            [0.2, 0.4, 0.4],  # ambos pasan; sin bicis está más lejos de su umbral (2 contra 1)
            [0.4, 0.2, 0.4],  # empate exacto (1 y 1): gana sin bicis
        ]
    )
    assert p2_decidir_clase(p, umbrales).tolist() == [0, 1, 2, 1, 1]


def test_p2_umbral_maximiza_el_f1():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    # F1 por umbral: 0.1 -> 0.67, 0.35 -> 0.8, 0.4 -> 0.5, 0.8 -> 0.67
    assert train.p2_elegir_umbral(y, p) == pytest.approx(0.35)


def test_p2_umbral_con_separacion_perfecta():
    assert train.p2_elegir_umbral(np.array([0, 0, 1]), np.array([0.1, 0.2, 0.9])) == 0.9


def test_p2_metricas_con_probabilidades_perfectas():
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.eye(3)[y] * 0.97 + 0.01  # 0.98 a la clase real y 0.01 a cada otra
    metricas = train.p2_calcular_metricas(y, p)
    assert set(metricas) == {"ap_sin_bicis", "ap_sin_anclajes", "ap_riesgo", "log_loss"}
    assert metricas["ap_sin_bicis"] == metricas["ap_sin_anclajes"] == metricas["ap_riesgo"] == 1
    assert metricas["log_loss"] == pytest.approx(-np.log(0.98))


def test_p2_log_loss_acepta_un_conjunto_sin_alguna_clase():
    """Un periodo corto puede no tener sin anclajes; log_loss no debe fallar por eso."""
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.7, 0.2, 0.1], [0.3, 0.6, 0.1]])
    with pytest.warns(UserWarning):  # average_precision avisa que no hay positivos
        metricas = train.p2_calcular_metricas(y, p)
    assert np.isfinite(metricas["log_loss"])


def test_p2_evaluacion_elige_umbrales_en_validacion_y_reporta_test():
    conjuntos = p2_conjuntos_sinteticos()
    modelo = train.P2BaselineHistorico().fit(conjuntos["fit"])
    variables = cfg.P2_FEATURES_CON_HISTORICO
    resultado = train.p2_evaluar_modelo(modelo, conjuntos["val"], conjuntos["test"], variables)

    n_test = len(conjuntos["test"])
    assert resultado["proba_test"].shape == (n_test, 3)
    assert resultado["pred"].shape == (n_test,)
    assert set(resultado["pred"]) <= {0, 1, 2}
    # Los umbrales son los de validación, no los de test
    p_val = modelo.predict_proba(conjuntos["val"][variables])
    y_val = conjuntos["val"]["clase"].to_numpy()
    for k in (1, 2):
        assert resultado["umbrales"][k] == train.p2_elegir_umbral(y_val == k, p_val[:, k])
    assert {"f1_sin_bicis", "recall_sin_anclajes", "ap_riesgo"} <= set(resultado["metricas"])
    assert set(resultado["metricas_valid"]) == {
        "ap_sin_bicis",
        "ap_sin_anclajes",
        "ap_riesgo",
        "log_loss",
    }
