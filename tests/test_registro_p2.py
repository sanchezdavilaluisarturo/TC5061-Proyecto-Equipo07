"""Pruebas del registro de P2: versión, linaje, alias, validaciones del run y carga por alias."""

from types import SimpleNamespace

import mlflow
import pytest
from test_p2_modelado import p2_tabla_sintetica

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.modeling import registro_p2, train


@pytest.fixture
def mlflow_temporal(tmp_path, monkeypatch):
    """MLflow en SQLite temporal; al terminar regresa la URI que había."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(cfg, "MLFLOW_RUN_NAME", None)
    anterior = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri(anterior)


def crear_candidato(etiquetas_extra=None, clave="baseline") -> str:
    """Registra un run de P2 con el baseline entrenado en la tabla sintética; regresa su id."""
    tabla = p2_tabla_sintetica()
    variables = cfg.P2_FEATURES_CON_HISTORICO
    modelo = train.p2_construir_modelo(clave).fit(tabla[variables], tabla["clase"])
    return train.p2_registrar_run(
        clave,
        {"modelo": clave, "data_version": "abc123"},
        {"valid_ap_riesgo": 0.8, "test_ap_riesgo": 0.75, "valid_log_loss": 0.4},
        {},
        modelo,
        tabla[variables].head(5),
        tabla,
        etiquetas_extra=etiquetas_extra,
    )


def run_falso(metricas=None, etiquetas=None, parametros=None) -> SimpleNamespace:
    """Objeto con la forma de un run de MLflow, para probar las funciones sin servidor."""
    return SimpleNamespace(
        info=SimpleNamespace(run_id="run123", run_name="p2-xgb"),
        data=SimpleNamespace(
            metrics={"valid_ap_riesgo": 0.81234, **(metricas or {})},
            tags=etiquetas or {},
            params=parametros or {},
        ),
    )


def test_p2_etiquetas_de_version_sin_datos_del_run_usan_respaldos():
    etiquetas = registro_p2.p2_etiquetas_version(run_falso())
    assert etiquetas == {
        "question": "p2",
        "fase": cfg.FASE_ENTREGA,
        "commit": "desconocido",
        "data_version": "desconocido",
        "source_run_id": "run123",
        "valid_ap_riesgo": "0.8123",
    }


def test_p2_etiquetas_de_version_llevan_la_busqueda_si_existe():
    run = run_falso(etiquetas={"busqueda_run_id": "padre1", "commit": "abc"})
    etiquetas = registro_p2.p2_etiquetas_version(run)
    assert etiquetas["busqueda_run_id"] == "padre1"
    assert etiquetas["commit"] == "abc"


def test_p2_descripcion_solo_incluye_las_metricas_del_run():
    run = run_falso(metricas={"test_ap_riesgo": 0.7}, etiquetas={"mlflow.note.content": "XGB."})
    descripcion = registro_p2.p2_descripcion_version(run)
    assert descripcion.startswith("p2-xgb: XGB.")
    assert "valid_ap_riesgo=0.8123, test_ap_riesgo=0.7000" in descripcion
    assert "valid_log_loss" not in descripcion  # el run no la tiene
    assert descripcion.endswith("Run de origen: run123.")


def test_p2_registrar_deja_version_linaje_y_alias(mlflow_temporal):
    id_run = crear_candidato({"busqueda_run_id": "padre123"})

    version = registro_p2.p2_registrar(id_run, "champion")

    cliente = mlflow.MlflowClient(mlflow_temporal)
    nombre = cfg.P2_MODELO_REGISTRADO
    assert int(cliente.get_model_version_by_alias(nombre, "champion").version) == version
    registrada = cliente.get_model_version(nombre, version)
    assert registrada.tags["source_run_id"] == id_run
    assert registrada.tags["question"] == "p2"
    assert registrada.tags["data_version"] == "abc123"
    assert registrada.tags["busqueda_run_id"] == "padre123"
    assert registrada.tags["valid_ap_riesgo"] == "0.8000"
    assert "valid_ap_riesgo=0.8000" in registrada.description
    assert cliente.get_registered_model(nombre).description


def test_p2_challenger_no_reemplaza_al_champion(mlflow_temporal):
    registro_p2.p2_registrar(crear_candidato(), "champion")
    registro_p2.p2_registrar(crear_candidato(), "challenger")

    cliente = mlflow.MlflowClient(mlflow_temporal)
    nombre = cfg.P2_MODELO_REGISTRADO
    assert int(cliente.get_model_version_by_alias(nombre, "champion").version) == 1
    assert int(cliente.get_model_version_by_alias(nombre, "challenger").version) == 2


@pytest.mark.parametrize("clave", ["baseline", "xgb"])
def test_p2_verificar_carga_por_alias_y_predice_las_tres_clases(mlflow_temporal, clave):
    registro_p2.p2_registrar(crear_candidato(clave=clave), "champion")

    # El input_example del run tiene 5 filas
    assert registro_p2.p2_verificar("champion") == 5


def test_p2_no_registra_un_run_sin_modelo(mlflow_temporal):
    mlflow.set_experiment(cfg.EXPERIMENTO_MLFLOW)
    with mlflow.start_run() as corrida:
        mlflow.set_tag("question", "p2")

    with pytest.raises(ValueError, match="no tiene un modelo"):
        registro_p2.p2_registrar(corrida.info.run_id, "champion")


def test_p2_no_registra_un_run_de_otra_pregunta(mlflow_temporal):
    mlflow.set_experiment(cfg.EXPERIMENTO_MLFLOW)
    with mlflow.start_run() as corrida:
        mlflow.set_tag("question", "p1")

    with pytest.raises(ValueError, match="no es de P2"):
        registro_p2.p2_registrar(corrida.info.run_id, "champion")


def test_p2_no_registra_un_run_que_no_termino(mlflow_temporal):
    mlflow.set_experiment(cfg.EXPERIMENTO_MLFLOW)
    with mlflow.start_run() as corrida:
        mlflow.set_tag("question", "p2")
        # Dentro del bloque el run sigue en RUNNING
        with pytest.raises(ValueError, match="no terminó"):
            registro_p2.p2_registrar(corrida.info.run_id, "champion")


def test_p2_registrar_exige_la_url_de_mlflow(monkeypatch):
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", None)
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        registro_p2.registrar(run_id="run123", alias="champion")
