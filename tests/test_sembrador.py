"""Pruebas del sembrador: copia de origen a destino, linaje, idempotencia y origen caído.

El origen y el destino son dos MLflow en SQLite temporales; no se necesita red ni el servidor
del equipo.
"""

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from tc5061_proyecto_equipo07 import sembrador

NOMBRE = "mibici-prueba-sembrador"
UMBRALES = {"umbral_sin_bicis": "0.31", "umbral_sin_anclajes": "0.27"}


@pytest.fixture
def servidores(tmp_path, monkeypatch):
    """URIs de origen y destino; al terminar regresa las URIs globales que había."""
    monkeypatch.chdir(tmp_path)  # los artefactos de ambos quedan bajo tmp_path/mlruns
    # Un origen HTTP inalcanzable debe fallar al instante, no tras 7 reintentos
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    anterior = (mlflow.get_tracking_uri(), mlflow.get_registry_uri())
    yield f"sqlite:///{tmp_path / 'origen.db'}", f"sqlite:///{tmp_path / 'destino.db'}"
    mlflow.set_tracking_uri(anterior[0])
    mlflow.set_registry_uri(anterior[1])


def registrar_en_origen(origen_uri: str, alias: str = "champion") -> str:
    """Registra en el origen una versión con parámetros y linaje y le pone el alias."""
    mlflow.set_tracking_uri(origen_uri)
    mlflow.set_registry_uri(origen_uri)
    mlflow.set_experiment("origen")
    modelo = DummyClassifier(strategy="prior").fit(np.zeros((6, 2)), [0, 1, 2, 0, 1, 2])
    with mlflow.start_run() as run:
        mlflow.log_params(UMBRALES)
        info = mlflow.sklearn.log_model(modelo, name="model")
    version = mlflow.register_model(info.model_uri, NOMBRE)
    cliente = MlflowClient(tracking_uri=origen_uri, registry_uri=origen_uri)
    cliente.set_model_version_tag(NOMBRE, version.version, "source_run_id", run.info.run_id)
    cliente.set_model_version_tag(NOMBRE, version.version, "commit", "abc123")
    cliente.set_registered_model_alias(NOMBRE, alias, version.version)
    return run.info.run_id


def test_copia_el_modelo_con_su_alias_linaje_y_parametros(servidores):
    origen_uri, destino_uri = servidores
    id_run_origen = registrar_en_origen(origen_uri)

    assert sembrador.sembrar_modelo(origen_uri, destino_uri, NOMBRE, "champion") == "copiado"

    destino = MlflowClient(tracking_uri=destino_uri, registry_uri=destino_uri)
    version = destino.get_model_version_by_alias(NOMBRE, "champion")
    # Conserva las etiquetas originales y agrega el linaje del origen
    assert version.tags["commit"] == "abc123"
    assert version.tags["origen_version"] == "1"
    assert version.tags["origen_uri"] == origen_uri
    assert version.tags["origen_run_id"] == id_run_origen
    # source_run_id apunta al run local, que es de donde la API lee los umbrales de P2
    assert version.tags["source_run_id"] != id_run_origen
    assert destino.get_run(version.tags["source_run_id"]).data.params == UMBRALES


def test_el_modelo_copiado_carga_y_predice(servidores):
    origen_uri, destino_uri = servidores
    registrar_en_origen(origen_uri)
    sembrador.sembrar_modelo(origen_uri, destino_uri, NOMBRE, "champion")

    mlflow.set_tracking_uri(destino_uri)
    mlflow.set_registry_uri(destino_uri)
    modelo = mlflow.pyfunc.load_model(f"models:/{NOMBRE}@champion")
    assert len(modelo.predict(np.zeros((3, 2)))) == 3


def test_segunda_corrida_no_duplica(servidores):
    origen_uri, destino_uri = servidores
    registrar_en_origen(origen_uri)
    sembrador.sembrar_modelo(origen_uri, destino_uri, NOMBRE, "champion")

    assert sembrador.sembrar_modelo(origen_uri, destino_uri, NOMBRE, "champion") == "omitido"
    destino = MlflowClient(tracking_uri=destino_uri, registry_uri=destino_uri)
    assert len(destino.search_model_versions(f"name='{NOMBRE}'")) == 1


def test_si_el_alias_se_mueve_en_el_origen_copia_la_version_nueva(servidores):
    origen_uri, destino_uri = servidores
    registrar_en_origen(origen_uri)
    sembrador.sembrar_modelo(origen_uri, destino_uri, NOMBRE, "champion")
    registrar_en_origen(origen_uri)  # versión 2 en el origen, con el alias champion

    assert sembrador.sembrar_modelo(origen_uri, destino_uri, NOMBRE, "champion") == "copiado"
    destino = MlflowClient(tracking_uri=destino_uri, registry_uri=destino_uri)
    version = destino.get_model_version_by_alias(NOMBRE, "champion")
    assert (str(version.version), version.tags["origen_version"]) == ("2", "2")


def test_origen_caido_con_copia_local_la_conserva(servidores):
    origen_uri, destino_uri = servidores
    registrar_en_origen(origen_uri)
    sembrador.sembrar_modelo(origen_uri, destino_uri, NOMBRE, "champion")

    caido = "http://127.0.0.1:9"  # puerto sin servicio
    assert sembrador.sembrar_modelo(caido, destino_uri, NOMBRE, "champion") == "conservado"
    destino = MlflowClient(tracking_uri=destino_uri, registry_uri=destino_uri)
    assert str(destino.get_model_version_by_alias(NOMBRE, "champion").version) == "1"


def test_origen_caido_sin_copia_local_falla(servidores):
    _, destino_uri = servidores
    with pytest.raises(MlflowException):
        sembrador.sembrar_modelo("http://127.0.0.1:9", destino_uri, NOMBRE, "champion")


def test_main_sale_con_0_si_conserva_y_con_1_si_no_hay_que_servir(servidores, monkeypatch):
    origen_uri, destino_uri = servidores
    monkeypatch.setattr(sembrador, "MODELOS_REGISTRADOS", {"p2": NOMBRE})
    monkeypatch.setenv("MLFLOW_TRACKING_URI", destino_uri)
    monkeypatch.setenv("MLFLOW_ORIGEN_URI", "http://127.0.0.1:9")
    assert sembrador.main() == 1  # destino vacío: la API no tendría qué servir

    registrar_en_origen(origen_uri)
    sembrador.sembrar_modelo(origen_uri, destino_uri, NOMBRE, "champion")
    assert sembrador.main() == 0  # origen caído, pero el destino ya está sembrado


def test_main_rechaza_origen_igual_al_destino(servidores, monkeypatch):
    _, destino_uri = servidores
    monkeypatch.setenv("MLFLOW_TRACKING_URI", destino_uri)
    monkeypatch.setenv("MLFLOW_ORIGEN_URI", destino_uri + "/")
    assert sembrador.main() == 1
