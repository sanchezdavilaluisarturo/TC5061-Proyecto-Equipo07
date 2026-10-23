"""Pruebas de la búsqueda de P3 con Optuna: espacio, restricciones, runs padre e hijos y candidato."""

import numpy as np
import optuna
import pandas as pd
import pytest
from test_modeling import p3_limpios_sinteticos

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset
from tc5061_proyecto_equipo07.modeling import train, tuning_p3


def base_sintetica() -> train.P3Baseline:
    return train.p3_ajustar_baseline(
        p3_limpios_sinteticos(),
        42,
        k_minimo=2,
        k_maximo=4,
        min_salidas_train=50,
        min_salidas_valid=20,
    )


def estaciones_sinteticas() -> pd.DataFrame:
    ids = np.arange(1, 31)
    return pd.DataFrame(
        {
            "id": ids,
            "name": [f"estacion {i}" for i in ids],
            "dpcapacity": 20,
            "latitude": 20.65 + ids * 0.001,
            "longitude": -103.35 + ids * 0.001,
        }
    )


def preparar_experimento_local(tmp_path):
    """Como `p3_preparar_experimento`, pero con artefactos en disco: el proxy `mlflow-artifacts:/`
    del servidor del equipo exige una URL http y no existe con SQLite."""

    def preparar(experimento: str = cfg.P3_EXPERIMENTO, uri: str | None = None) -> str:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        if mlflow.get_experiment_by_name(experimento) is None:
            mlflow.create_experiment(experimento, artifact_location=str(tmp_path / "artefactos"))
        mlflow.set_experiment(experimento)
        return "http://servidor/#/experiments/1/runs/"

    return preparar


def test_espacio_respeta_los_limites_y_construye_el_modelo():
    limites = {"componentes": (2, 20), "k": (3, 8)}
    trial = optuna.create_study().ask()
    configuracion = tuning_p3.p3_espacio_alternativo(trial, limites, max_componentes=5)
    assert 2 <= configuracion["componentes"] <= 5  # el tope de variables manda sobre el de config
    assert 3 <= configuracion["k"] <= 8
    columnas = [f"c{i}" for i in range(10)]
    train.p3_construir_modelo(
        columnas,
        configuracion["componentes"],
        configuracion["ponderar"],
        configuracion["k"],
        42,
    )


def test_trials_llevan_las_restricciones_de_aceptacion(tmp_path, monkeypatch):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("prueba")  # otros tests dejan un experimento activo que ya no existe
    base = base_sintetica()
    with mlflow.start_run():
        estudio, semillas = tuning_p3.p3_buscar(
            base, "alternativo", n_trials=3, timeout_min=0, n_semillas=2, tam_min_aceptable=5
        )
    assert len(estudio.trials) == 3
    for trial in estudio.trials:
        assert set(trial.constraints) == {"ari_valid", "tam_min"}
        propios = semillas[
            (semillas["ponderar"] == trial.params["ponderar"])
            & (semillas["componentes"] == trial.params["componentes"])
            & (semillas["k"] == trial.params["k"])
        ]
        violacion_ari = base.metricas["ari_valid"] - propios["ari_valid"].mean()
        assert trial.constraints["ari_valid"] == pytest.approx(violacion_ari)
        assert trial.constraints["tam_min"] == 5 - propios["tam_min"].min()
    # Una fila por semilla de cada trial
    assert len(semillas) == 3 * 2


def test_etiquetas_de_la_busqueda_usan_la_rubrica():
    etiquetas = tuning_p3.p3_etiquetas_busqueda("nota")
    assert etiquetas["stage_of_work"] == "search"
    assert etiquetas["question"] == "p3"
    assert etiquetas["model_family"] == "pca_kmeans"
    assert etiquetas["fase"] == cfg.FASE_ENTREGA
    assert etiquetas["mlflow.note.content"] == "nota"
    assert "pregunta" not in etiquetas and "etapa" not in etiquetas


def test_busqueda_deja_padre_hijos_y_candidato(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(cfg, "FIGURES_DIR", tmp_path / "figuras")
    monkeypatch.setattr(cfg, "P3_SEMILLAS_BUSQUEDA", 2)
    monkeypatch.setattr(cfg, "P3_TAM_MIN_ACEPTABLE", 5)
    monkeypatch.setattr(tuning_p3, "p3_cargar_base", base_sintetica)
    monkeypatch.setattr(tuning_p3, "p3_preparar_experimento", preparar_experimento_local(tmp_path))
    monkeypatch.setattr(train, "p3_preparar_experimento", preparar_experimento_local(tmp_path))
    monkeypatch.setattr(dataset, "p3_cargar_estaciones", estaciones_sinteticas)

    tuning_p3.buscar(
        model="alternativo", n_trials="4", fraccion_train="", timeout_min="", sin_candidato=False
    )

    cliente = mlflow.MlflowClient(uri)
    experimento = cliente.get_experiment_by_name(cfg.P3_EXPERIMENTO)
    corridas = cliente.search_runs([experimento.experiment_id])
    por_nombre = {c.data.tags["mlflow.runName"]: c for c in corridas}

    padre = por_nombre["pca_kmeans_optuna-busqueda"]
    assert padre.data.tags["stage_of_work"] == "search"
    hijos = [c for c in corridas if c.data.tags.get("mlflow.parentRunId") == padre.info.run_id]
    assert len(hijos) == 4
    assert all(h.data.tags["stage_of_work"] == "search" for h in hijos)
    assert all("train_silueta" in h.data.metrics and "valid_ari" in h.data.metrics for h in hijos)
    assert padre.data.params["best_child_run_id"] in {h.info.run_id for h in hijos}

    candidato = por_nombre["pca_kmeans_optuna"]
    assert candidato.data.tags["stage_of_work"] == "candidate"
    assert candidato.data.tags["busqueda_run_id"] == padre.info.run_id
    assert candidato.data.tags["referencia"] == "baseline"
    # Los parámetros del candidato son los del mejor trial de la búsqueda
    assert candidato.data.params["k"] == padre.data.params["mejor_k"]
    assert candidato.data.params["componentes"] == padre.data.params["mejor_componentes"]
    # Cada métrica dice en qué conjunto se calculó (como valid_mae y test_mae en P1)
    assert {"train_silueta", "valid_ari", "train_tam_grupo_min"} <= set(candidato.data.metrics)
    assert all(n.startswith(("train_", "valid_")) for n in candidato.data.metrics)
    assert padre.data.metrics["mejor_train_silueta"] > 0
    # Punto 5: entrada registrada, firma del modelo y artefactos en subcarpetas
    assert candidato.inputs.dataset_inputs[0].dataset.name == "p3_train"
    archivos = {a.path for a in cliente.list_artifacts(candidato.info.run_id)}
    assert {"plots", "tables", "data", "config"} <= archivos
    # El modelo se registra con firma (entradas y salidas) y con el nombre del run
    mlmodel = next(tmp_path.glob("artefactos/**/MLmodel")).read_text()
    assert "signature:" in mlmodel and "inputs:" in mlmodel and "outputs:" in mlmodel


def test_fraccion_train_y_modelo_invalido_se_rechazan():
    import typer

    with pytest.raises(typer.BadParameter, match="fraccion_train"):
        tuning_p3.buscar(
            model="alternativo", n_trials="", fraccion_train="0.5", timeout_min="", sin_candidato=True
        )
    with pytest.raises(typer.BadParameter):
        tuning_p3.buscar(
            model="dbscan", n_trials="", fraccion_train="", timeout_min="", sin_candidato=True
        )
