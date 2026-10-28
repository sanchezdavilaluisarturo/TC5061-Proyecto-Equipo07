"""Pruebas de la búsqueda de P1 con Optuna: espacios, runs padre e hijos y candidato final."""

import optuna

from tc5061_proyecto_equipo07.modeling import train, tuning_p2


def test_p2_espacio_xgb_construye_un_modelo_valido():
    trial = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0)).ask()
    hiperparametros = tuning_p2.ESPACIOS["xgb"](trial)
    assert 30 <= hiperparametros["max_leaves"] <= 500
    assert 0.6 <= hiperparametros["subsample"] <= 1.0
    modelo = train.p2_construir_modelo("xgb_optuna", hiperparametros=hiperparametros)
    assert modelo.get_params()["max_leaves"] == hiperparametros["max_leaves"]


def test_p2_opciones_vacias_de_windows_conservan_config():
    assert tuning_p2.p2_leer_opcion("", int, 15) == 15
    assert tuning_p2.p2_leer_opcion("''", int, 15) == 15
    assert tuning_p2.p2_leer_opcion('""', float, 0.1) == 0.1
    assert tuning_p2.p2_leer_opcion(" '30' ", int, 15) == 30