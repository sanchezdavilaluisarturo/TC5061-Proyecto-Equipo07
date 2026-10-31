"""Pruebas del despachador del MLflow Project (modeling/run.py)"""

import copy

import matplotlib.pyplot as plt
import pytest
from test_p2_modelado import p2_cambios_sinteticos
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.modeling import run, train


@pytest.fixture
def config_intacta():
    """Restaura los dicts de config.py que el despachador modifica."""
    respaldo = [
        copy.deepcopy(d)
        for d in (cfg.P1_PARAMETROS_HGB, cfg.P1_PARAMETROS_XGB, cfg.P2_PARAMETROS_HGB,
                  cfg.P2_PARAMETROS_XGB)
    ]
    yield
    for dicto, original in zip(
        (cfg.P1_PARAMETROS_HGB, cfg.P1_PARAMETROS_XGB, cfg.P2_PARAMETROS_HGB,
         cfg.P2_PARAMETROS_XGB), respaldo
    ):
        dicto.clear()
        dicto.update(original)


@pytest.fixture
def cambios_en_disco(tmp_path, monkeypatch):
    """p2-entrenar lee la tabla sintética desde un parquet temporal y deja figuras en tmp_path."""
    plt.switch_backend("Agg")
    monkeypatch.setattr(cfg, "FIGURES_DIR", tmp_path)
    ruta = tmp_path / "p2_cambios_estacion_bin.parquet"
    p2_cambios_sinteticos().to_parquet(ruta, index=False)
    # El valor por defecto de `ruta` se fija al importar: se cambia la función, no cfg
    leer = train.p2_cargar_cambios
    monkeypatch.setattr(train, "p2_cargar_cambios", lambda: leer(ruta))
    return tmp_path


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
    extras = ["--min-samples-leaf", "200"]
    assert run.comandos_de_train("p2", "hgb", extras) == [
        ["p2-entrenar", "--registrar", "--modelos", "hgb", "--min-samples-leaf", "200"]
    ]
    assert run.comandos_de_train("p2", "", []) == [["p2-entrenar", "--registrar"]]


def test_modelos_validos_salen_de_config():
    assert run.modelos_validos("p1") == list(cfg.P1_MODELOS)
    assert run.modelos_validos("p2") == list(cfg.P2_MODELOS)


def test_vacio_de_mlflow_run_en_windows():
    assert run.leer_hiperparametros({"n_estimators": '""', "max_depth": "8"}) == {"max_depth": 8}


def test_p2_xgb_recibe_n_estimators(config_intacta):
    extras = run.aplicar_hiperparametros("p2", "xgb", {"n_estimators": 50})
    assert cfg.P2_PARAMETROS_XGB["n_estimators"] == 50
    assert extras == []


def test_p2_mlflow_run_llega_a_p2_entrenar_con_min_samples_leaf(cambios_en_disco, monkeypatch):
    llamadas = []

    def ajustar_falso(clave, fit, val, test, comunes, registrar, min_samples_leaf, **kwargs):
        llamadas.append((clave, registrar, min_samples_leaf))
        return {"valid_ap_riesgo": 0.5}

    monkeypatch.setattr(train, "p2_ajustar_y_evaluar", ajustar_falso)
    run.entrenar(
        question="p2",
        model="hgb",
        n_estimators="",
        learning_rate="",
        max_depth="",
        min_samples_leaf="20",
    )
    # `mlflow run` siempre registra; min_samples_leaf viaja como opción del CLI
    assert llamadas == [("hgb", True, 20)]
