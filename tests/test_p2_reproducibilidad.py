"""Pruebas de reproducibilidad de P2: misma semilla, mismas probabilidades, umbrales y búsqueda."""

import mlflow
import numpy as np
import optuna
import pytest
from test_p2_modelado import p2_conjuntos_sinteticos

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.modeling import train, tuning_p2

# subsample muestrea filas: con él, el resultado de XGBoost depende de la semilla
HIPERPARAMETROS_CON_AZAR = {"subsample": 0.7, "colsample_bytree": 0.7}
OTRA_SEMILLA = 7

MODELOS = [
    ("hgb", None),
    ("xgb", None),
    ("xgb_optuna", HIPERPARAMETROS_CON_AZAR),
]


@pytest.fixture
def mlflow_temporal(tmp_path, monkeypatch):
    """MLflow en una base SQLite temporal; al terminar regresa la URI que había."""
    anterior = mlflow.get_tracking_uri()
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        mlflow.set_experiment("p2-reproducibilidad")
        yield
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(anterior)


@pytest.fixture
def busqueda_sintetica(monkeypatch):
    """La búsqueda lee fit, val y test sintéticos en lugar de la tabla de data/processed/."""
    conjuntos = p2_conjuntos_sinteticos()
    monkeypatch.setattr(tuning_p2, "p2_cargar_cambios", lambda: None)
    monkeypatch.setattr(
        tuning_p2,
        "p2_preparar_conjuntos",
        lambda cambios, peso: (conjuntos["fit"], conjuntos["val"], conjuntos["test"]),
    )


def entrenar_y_evaluar(clave, hiperparametros, conjuntos, semilla=cfg.SEMILLA) -> dict:
    """Entrena con fit; los umbrales salen de valid y las métricas de test."""
    variables = cfg.P2_FEATURES_CON_HISTORICO
    modelo = train.p2_construir_modelo(clave, semilla=semilla, hiperparametros=hiperparametros)
    modelo.fit(conjuntos["fit"][variables], conjuntos["fit"]["clase"])
    return train.p2_evaluar_modelo(modelo, conjuntos["val"], conjuntos["test"], variables)


@pytest.mark.parametrize(("clave", "hiperparametros"), MODELOS)
def test_p2_misma_semilla_da_las_mismas_probabilidades_umbrales_y_metricas(clave, hiperparametros):
    conjuntos = p2_conjuntos_sinteticos()
    a = entrenar_y_evaluar(clave, hiperparametros, conjuntos)
    b = entrenar_y_evaluar(clave, hiperparametros, conjuntos)
    np.testing.assert_array_equal(a["proba_test"], b["proba_test"])
    np.testing.assert_array_equal(a["pred"], b["pred"])
    assert a["umbrales"] == b["umbrales"]
    assert a["metricas"] == b["metricas"]
    assert a["metricas_valid"] == b["metricas_valid"]


def test_p2_otra_semilla_cambia_el_modelo_con_muestreo():
    """Contraste: si la semilla no llegara al modelo, la anterior pasaría sin probar nada."""
    conjuntos = p2_conjuntos_sinteticos()
    a = entrenar_y_evaluar("xgb_optuna", HIPERPARAMETROS_CON_AZAR, conjuntos)
    b = entrenar_y_evaluar("xgb_optuna", HIPERPARAMETROS_CON_AZAR, conjuntos, OTRA_SEMILLA)
    assert not np.array_equal(a["proba_test"], b["proba_test"])


def test_p2_muestra_de_la_busqueda_cambia_con_otra_semilla(busqueda_sintetica, monkeypatch):
    muestra = tuning_p2.p2_preparar_busqueda(0.5)["muestra"]
    monkeypatch.setattr(cfg, "SEMILLA", OTRA_SEMILLA)  # la lee fit.sample en cada llamada
    otra = tuning_p2.p2_preparar_busqueda(0.5)["muestra"]
    assert len(otra) == len(muestra)
    assert not otra.index.equals(muestra.index)


def buscar(conjuntos, n_trials: int = 3) -> optuna.Study:
    """Corre la búsqueda de P2 dentro de un run padre, como lo hace `tuning_p2.buscar`."""
    # Muestra fija: así el contraste con otra semilla solo cambia el sampler de Optuna
    muestra = conjuntos["fit"].sample(frac=0.5, random_state=0)
    with mlflow.start_run(run_name="p2-xgb-optuna-busqueda"):
        return tuning_p2.p2_buscar("xgb", muestra, conjuntos["val"], n_trials, 0)


def test_p2_busqueda_con_optuna_es_reproducible(mlflow_temporal):
    conjuntos = p2_conjuntos_sinteticos()
    a, b = buscar(conjuntos), buscar(conjuntos)
    # Mismos hiperparámetros propuestos en cada trial y mismo valid_ap_riesgo
    assert [t.params for t in a.trials] == [t.params for t in b.trials]
    assert [t.value for t in a.trials] == [t.value for t in b.trials]
    assert a.best_trial.number == b.best_trial.number
    assert a.best_params == b.best_params


def test_p2_busqueda_con_otra_semilla_prueba_otras_configuraciones(mlflow_temporal, monkeypatch):
    conjuntos = p2_conjuntos_sinteticos()
    a = buscar(conjuntos)
    monkeypatch.setattr(cfg, "SEMILLA", OTRA_SEMILLA)  # la lee el sampler al crear el estudio
    b = buscar(conjuntos)
    assert [t.params for t in a.trials] != [t.params for t in b.trials]
