"""Pruebas del registro de P1: versión, linaje, alias y carga por alias."""

import mlflow
import pytest
from test_p1_modelado import p1_tabla_sintetica

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.modeling import registro_p1, train
from tc5061_proyecto_equipo07.modeling.train import p1_construir_modelo


@pytest.fixture
def servidor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    return uri


def crear_candidato(etiquetas_extra=None) -> str:
    tabla = p1_tabla_sintetica()
    modelo = p1_construir_modelo("baseline").fit(tabla[cfg.P1_VARIABLES], tabla["viajes"])
    ejemplo = tabla[cfg.P1_VARIABLES].head(5).astype({"zona": str})
    return train.p1_registrar_run(
        "baseline",
        {"modelo": "baseline", "data_version": "abc123"},
        {"valid_mae": 1.0, "test_mae": 1.1},
        [],
        modelo,
        ejemplo,
        tabla,
        etiquetas_extra=etiquetas_extra,
    )


def test_registrar_deja_version_linaje_y_alias(servidor):
    id_run = crear_candidato({"busqueda_run_id": "padre123"})

    version = registro_p1.p1_registrar(id_run, "champion")

    cliente = mlflow.MlflowClient(servidor)
    nombre = cfg.P1_MODELO_REGISTRADO
    assert int(cliente.get_model_version_by_alias(nombre, "champion").version) == version
    registrada = cliente.get_model_version(nombre, version)
    assert registrada.tags["source_run_id"] == id_run
    assert registrada.tags["data_version"] == "abc123"
    assert registrada.tags["busqueda_run_id"] == "padre123"
    assert registrada.tags["fase"] == cfg.FASE_ENTREGA
    assert "valid_mae=1.0000" in registrada.description
    assert cliente.get_registered_model(nombre).description


def test_el_alias_se_mueve_a_la_version_nueva(servidor):
    primera = registro_p1.p1_registrar(crear_candidato(), "champion")
    segunda = registro_p1.p1_registrar(crear_candidato(), "champion")

    cliente = mlflow.MlflowClient(servidor)
    alias = cliente.get_model_version_by_alias(cfg.P1_MODELO_REGISTRADO, "champion")
    assert (primera, segunda) == (1, 2)
    assert int(alias.version) == 2


def test_verificar_carga_por_alias_y_predice(servidor):
    registro_p1.p1_registrar(crear_candidato(), "champion")

    assert registro_p1.p1_verificar("champion") == 5


def test_no_registra_un_run_sin_modelo(servidor):
    mlflow.set_experiment(cfg.EXPERIMENTO_MLFLOW)
    with mlflow.start_run() as corrida:
        mlflow.log_metric("valid_mae", 1.0)

    with pytest.raises(ValueError, match="no tiene un modelo"):
        registro_p1.p1_registrar(corrida.info.run_id, "champion")
