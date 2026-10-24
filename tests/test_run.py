"""Pruebas del despachador del MLflow Project (modeling/run.py), sin entrenar ni red."""

import copy

import pytest
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.modeling import run


@pytest.fixture
def config_intacta():
    """Restaura los dicts de config.py que el despachador modifica."""
    respaldo = [
        copy.deepcopy(d)
        for d in (cfg.P1_PARAMETROS_HGB, cfg.P1_PARAMETROS_XGB, cfg.P2_PARAMETROS_HGB)
    ]
    yield
    for dicto, original in zip(
        (cfg.P1_PARAMETROS_HGB, cfg.P1_PARAMETROS_XGB, cfg.P2_PARAMETROS_HGB), respaldo
    ):
        dicto.clear()
        dicto.update(original)


def test_parametros_vacios_conservan_config():
    assert run.leer_hiperparametros({"n_estimators": "", "max_depth": " "}) == {}


def test_hiperparametros_se_convierten_a_su_tipo():
    leidos = run.leer_hiperparametros({"n_estimators": "300", "learning_rate": "0.05"})
    assert leidos == {"n_estimators": 300, "learning_rate": 0.05}


def test_xgb_recibe_n_estimators(config_intacta):
    run.aplicar_hiperparametros("p1", "xgb", {"n_estimators": 50})
    assert cfg.P1_PARAMETROS_XGB["n_estimators"] == 50


def test_hgb_traduce_n_estimators_a_max_iter(config_intacta):
    run.aplicar_hiperparametros("p1", "hgb", {"n_estimators": 80})
    assert cfg.P1_PARAMETROS_HGB["max_iter"] == 80


def test_p2_min_samples_leaf_va_por_el_cli(config_intacta):
    extras = run.aplicar_hiperparametros("p2", "hgb", {"min_samples_leaf": 200})
    assert extras == ["--min-samples-leaf", "200"]
    assert "min_samples_leaf" not in cfg.P2_PARAMETROS_HGB


def test_hiperparametro_que_no_aplica_se_rechaza(config_intacta):
    with pytest.raises(typer.BadParameter):
        run.aplicar_hiperparametros("p1", "xgb", {"min_samples_leaf": 5})
    with pytest.raises(typer.BadParameter):
        run.aplicar_hiperparametros("p3", "", {"n_estimators": 5})


def test_comandos_por_pregunta():
    assert run.comandos_de_train("p1", "hgb", []) == [
        ["p1-entrenar", "--registrar", "--modelos", "hgb"]
    ]
    assert run.comandos_de_train("p1", "", [])[0] == ["p1-entrenar", "--registrar"]
    assert [c[0] for c in run.comandos_de_train("p3", "", [])] == ["p3-baseline", "p3-alternativo"]


def test_modelos_validos_salen_de_config():
    assert run.modelos_validos("p1") == list(cfg.P1_MODELOS)
    assert run.modelos_validos("p2") == list(cfg.P2_MODELOS)
    

def test_vacio_de_mlflow_run_en_windows():
    assert run.leer_hiperparametros({"n_estimators": '""', "max_depth": "8"}) == {"max_depth": 8}
