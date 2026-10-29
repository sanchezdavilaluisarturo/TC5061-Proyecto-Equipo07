"""Pruebas de la búsqueda de P2 con Optuna: espacio, muestra, runs padre e hijos y candidato."""

import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd
import pytest
from test_p2_modelado import p2_conjuntos_sinteticos
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.modeling import train, tuning_p2


@pytest.fixture
def conjuntos_falsos(monkeypatch):
    """La búsqueda lee fit, val y test sintéticos en lugar de la tabla de data/processed/."""
    conjuntos = p2_conjuntos_sinteticos()
    monkeypatch.setattr(tuning_p2, "p2_cargar_cambios", lambda: None)
    monkeypatch.setattr(
        tuning_p2,
        "p2_preparar_conjuntos",
        lambda cambios, peso: (conjuntos["fit"], conjuntos["val"], conjuntos["test"]),
    )
    return conjuntos


@pytest.fixture
def mlflow_temporal(tmp_path, monkeypatch, conjuntos_falsos):
    """MLflow en SQLite temporal y figuras en tmp_path; regresa el cliente de MLflow."""
    plt.switch_backend("Agg")
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(cfg, "MLFLOW_RUN_NAME", None)
    monkeypatch.setattr(cfg, "FIGURES_DIR", tmp_path)
    anterior = mlflow.get_tracking_uri()
    yield mlflow.MlflowClient(uri)
    mlflow.set_tracking_uri(anterior)


def corridas_por_nombre(cliente: mlflow.MlflowClient) -> tuple[dict, list]:
    """Runs del experimento del proyecto, por nombre y en lista."""
    experimento = cliente.get_experiment_by_name(cfg.EXPERIMENTO_MLFLOW)
    corridas = cliente.search_runs([experimento.experiment_id])
    return {c.data.tags["mlflow.runName"]: c for c in corridas}, corridas


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


def test_p2_etiquetas_de_la_busqueda(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    etiquetas = tuning_p2.p2_etiquetas_busqueda("xgb", busqueda="extra")
    assert etiquetas["question"] == "p2"
    assert etiquetas["stage_of_work"] == "search"
    assert etiquetas["model_family"] == cfg.P2_MODELOS["xgb"][2]
    assert etiquetas["github_run_id"] == "123"
    assert etiquetas["busqueda"] == "extra"


def test_p2_preparar_busqueda_solo_muestrea_fit_con_la_semilla(conjuntos_falsos):
    datos = tuning_p2.p2_preparar_busqueda(0.5)
    fit = conjuntos_falsos["fit"]
    assert len(datos["muestra"]) == len(fit) // 2
    assert set(datos["muestra"].index) <= set(fit.index)
    # Valid y test quedan completos: con valid se elige y test solo lo usa el candidato
    assert datos["val"] is conjuntos_falsos["val"] and datos["test"] is conjuntos_falsos["test"]
    # Misma semilla, misma muestra
    assert datos["muestra"].index.equals(tuning_p2.p2_preparar_busqueda(0.5)["muestra"].index)
    assert tuning_p2.p2_preparar_busqueda(1.0)["muestra"] is fit


@pytest.mark.parametrize(
    ("model", "fraccion_train"),
    [("rf", ""), ("xgb", "0"), ("xgb", "1.5")],
)
def test_p2_modelo_o_fraccion_invalidos_se_rechazan(model, fraccion_train):
    with pytest.raises(typer.BadParameter):
        tuning_p2.buscar(
            model=model,
            n_trials="",
            fraccion_train=fraccion_train,
            timeout_min="",
            sin_candidato=True,
        )


def test_p2_busqueda_exige_la_url_de_mlflow(monkeypatch):
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", None)
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        tuning_p2.buscar(
            model="xgb", n_trials="1", fraccion_train="", timeout_min="", sin_candidato=True
        )


def test_p2_busqueda_deja_padre_hijos_y_candidato(mlflow_temporal, tmp_path):
    tuning_p2.buscar(
        model="xgb", n_trials="2", fraccion_train="0.5", timeout_min="", sin_candidato=False
    )
    por_nombre, corridas = corridas_por_nombre(mlflow_temporal)

    padre = por_nombre["p2-xgb-optuna-busqueda"]
    assert padre.data.tags["stage_of_work"] == "search"
    assert padre.data.params["objetivo"] == "valid_ap_riesgo"
    hijos = [c for c in corridas if c.data.tags.get("mlflow.parentRunId") == padre.info.run_id]
    assert len(hijos) == 2
    assert padre.data.params["best_child_run_id"] in {h.info.run_id for h in hijos}
    assert padre.data.metrics["n_trials_terminados"] == 2
    # Test no interviene en la búsqueda: los hijos solo tienen métricas de valid
    assert not any(nombre.startswith("test_") for h in hijos for nombre in h.data.metrics)

    candidato = por_nombre["p2-xgb-optuna"]
    assert candidato.data.tags["stage_of_work"] == "candidate"
    assert candidato.data.tags["busqueda_run_id"] == padre.info.run_id
    # El candidato usa los hiperparámetros del mejor trial, no los de config.py
    assert candidato.data.params["modelo_max_leaves"] == padre.data.params["mejor_max_leaves"]
    # Y se agrega a la comparación de modelos de P2
    comparacion = pd.read_csv(tmp_path / "p2_comparacion_modelos.csv", index_col=0)
    assert "xgb-optuna" in comparacion.index


def test_p2_busqueda_sin_candidato_no_reentrena(mlflow_temporal, tmp_path):
    tuning_p2.buscar(
        model="xgb", n_trials="1", fraccion_train="0.5", timeout_min="", sin_candidato=True
    )
    por_nombre, corridas = corridas_por_nombre(mlflow_temporal)
    assert "p2-xgb-optuna" not in por_nombre
    assert len(corridas) == 2  # el padre y su único trial
    assert not (tmp_path / "p2_comparacion_modelos.csv").exists()
