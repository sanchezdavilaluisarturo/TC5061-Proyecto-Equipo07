"""Pruebas de los modelos de P2: baseline, construcción, umbrales, métricas y modelos."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.modeling import train


def test_p2_baseline_usa_la_historia_y_suma_uno():
    x = pd.DataFrame({"hist_sin_bicis": [0.1, 0.6, 0.0], "hist_sin_anclajes": [0.2, 0.1, 0.0]})
    modelo = train.P2BaselineHistorico().fit(x)
    p = modelo.predict_proba(x)
    np.testing.assert_allclose(p, [[0.7, 0.1, 0.2], [0.3, 0.6, 0.1], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert modelo.predict(x).tolist() == [0, 1, 0]
    assert modelo.classes_.tolist() == [0, 1, 2]


def test_p2_construir_modelo_aplica_config_semilla_y_min_samples_leaf():
    hgb = train.p2_construir_modelo("hgb", min_samples_leaf=50, semilla=7)
    assert isinstance(hgb, HistGradientBoostingClassifier)
    assert (hgb.min_samples_leaf, hgb.random_state) == (50, 7)
    assert hgb.max_iter == cfg.P2_PARAMETROS_HGB["max_iter"]

    xgb = train.p2_construir_modelo("xgb", semilla=7)
    assert isinstance(xgb, XGBClassifier)
    assert xgb.get_params()["random_state"] == 7
    assert xgb.get_params()["n_estimators"] == cfg.P2_PARAMETROS_XGB["n_estimators"]

    assert isinstance(train.p2_construir_modelo("baseline"), train.P2BaselineHistorico)


def test_p2_hiperparametros_de_la_busqueda_reemplazan_los_de_config():
    xgb = train.p2_construir_modelo("xgb_optuna", hiperparametros={"max_leaves": 7})
    assert xgb.get_params()["max_leaves"] == 7
    # Lo que la búsqueda no toca se queda como en config.py
    assert xgb.get_params()["learning_rate"] == cfg.P2_PARAMETROS_XGB["learning_rate"]
    hgb = train.p2_construir_modelo("hgb", min_samples_leaf=50, hiperparametros={"max_iter": 5})
    assert (hgb.max_iter, hgb.min_samples_leaf) == (5, 50)


def test_p2_candidato_de_optuna_exige_hiperparametros():
    with pytest.raises(ValueError, match="hiperparámetros"):
        train.p2_construir_modelo("xgb_optuna")


def test_p2_modelo_desconocido_falla():
    with pytest.raises(ValueError, match="desconocido"):
        train.p2_construir_modelo("rf")