"""Pruebas de la búsqueda de P1 con Optuna: espacios, runs padre e hijos y candidato final."""

import numpy as np
import optuna
import pandas as pd
import pytest
from test_p1_modelado import p1_tabla_sintetica

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.modeling import tuning_p1
from tc5061_proyecto_equipo07.modeling.train import p1_construir_modelo


def conjuntos_sinteticos(n: int = 1500) -> dict[str, pd.DataFrame]:
    tabla = p1_tabla_sintetica(n)
    tabla["hora_inicio"] = pd.Timestamp("2025-01-01 05:00") + pd.to_timedelta(np.arange(n), "h")
    corte_valid, corte_test = int(n * 0.6), int(n * 0.8)
    particion = np.where(
        np.arange(n) < corte_valid, "train", np.where(np.arange(n) < corte_test, "valid", "test")
    )
    return {
        nombre: tabla[particion == nombre].reset_index(drop=True)
        for nombre in ["train", "valid", "test"]
    }


@pytest.mark.parametrize("clave", ["xgb", "hgb"])
def test_espacio_devuelve_hiperparametros_validos(clave):
    trial = optuna.create_study().ask()
    hiperparametros = tuning_p1.ESPACIOS[clave](trial)
    # Deben poder construir el modelo sin errores
    p1_construir_modelo(clave, hiperparametros=hiperparametros)
    if clave == "hgb":
        assert hiperparametros["l2_regularization"] >= 0.5
        assert hiperparametros["min_samples_leaf"] >= 50


def test_candidato_de_optuna_exige_hiperparametros():
    with pytest.raises(ValueError, match="hiperparámetros"):
        p1_construir_modelo("xgb_optuna")
    modelo = p1_construir_modelo("xgb_optuna", hiperparametros={"max_depth": 3})
    assert modelo.named_steps["modelo"].max_depth == 3


def test_opciones_vacias_conservan_config():
    assert tuning_p1.p1_leer_opcion("", int, 20) == 20
    assert tuning_p1.p1_leer_opcion(" 5", int, 20) == 5


def test_busqueda_deja_padre_hijos_y_candidato(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(cfg, "FIGURES_DIR", tmp_path / "figuras")
    monkeypatch.setattr(tuning_p1, "p1_cargar_conjuntos", conjuntos_sinteticos)

    tuning_p1.buscar(
        model="xgb", n_trials="2", fraccion_train="0.5", timeout_min="", sin_candidato=False
    )

    cliente = mlflow.MlflowClient(uri)
    experimento = cliente.get_experiment_by_name(cfg.EXPERIMENTO_MLFLOW)
    corridas = cliente.search_runs([experimento.experiment_id])
    por_nombre = {c.data.tags["mlflow.runName"]: c for c in corridas}

    padre = por_nombre["p1-xgb-optuna-busqueda"]
    assert padre.data.tags["stage_of_work"] == "search"
    hijos = [c for c in corridas if c.data.tags.get("mlflow.parentRunId") == padre.info.run_id]
    assert len(hijos) == 2
    assert all(h.data.tags["stage_of_work"] == "search" for h in hijos)
    assert padre.data.params["best_child_run_id"] in {h.info.run_id for h in hijos}

    candidato = por_nombre["p1-xgb-optuna"]
    assert candidato.data.tags["stage_of_work"] == "candidate"
    assert candidato.data.tags["busqueda_run_id"] == padre.info.run_id
    # Los hiperparámetros del mejor trial son los del candidato (no los de config.py)
    assert candidato.data.params["modelo_max_depth"] == padre.data.params["mejor_max_depth"]
    # Test no interviene en la búsqueda: los hijos solo tienen métricas de valid
    assert not any(nombre.startswith("test_") for h in hijos for nombre in h.data.metrics)


def test_modelo_invalido_se_rechaza():
    import typer

    with pytest.raises(typer.BadParameter):
        tuning_p1.buscar(
            model="random_forest", n_trials="", fraccion_train="", timeout_min="", sin_candidato=True
        )
