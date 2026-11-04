"""Pruebas de la base de la API: carga desde el Registry, /salud, 503 y dependencias de la imagen."""

from pathlib import Path
import re
from types import SimpleNamespace

from fastapi import HTTPException
from fastapi.testclient import TestClient
import mlflow
from mlflow.exceptions import MlflowException
import pytest
from test_registro_p1 import crear_candidato
from test_registro_p2 import crear_candidato as p2_crear_candidato

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.api import modelos
from tc5061_proyecto_equipo07.api.main import app
from tc5061_proyecto_equipo07.modeling import registro_p1, registro_p2

RAIZ = Path(cfg.PROJ_ROOT)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Registry local con solo P1 registrado como champion (P2 y P3 no existen)."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("MODELO_ALIAS", raising=False)
    mlflow.set_tracking_uri(uri)
    registro_p1.p1_registrar(crear_candidato(), "champion")
    return uri


@pytest.fixture
def registry_p2(tmp_path, monkeypatch):
    """Registry local con solo P2 registrado como champion."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(cfg, "MLFLOW_RUN_NAME", None)
    monkeypatch.delenv("MODELO_ALIAS", raising=False)
    mlflow.set_tracking_uri(uri)
    registro_p2.p2_registrar(p2_crear_candidato(), "champion")


def test_salud_reporta_version_cargada_y_modelos_faltantes(registry):
    with TestClient(app) as cliente:
        respuesta = cliente.get("/salud")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "degradado"
    assert cuerpo["modelos"]["p1"] == {
        "nombre": cfg.P1_MODELO_REGISTRADO,
        "alias": "champion",
        "version": 1,
        "listo": True,
        "error": None,
    }
    assert cuerpo["modelos"]["p2"]["listo"] is False
    assert cuerpo["modelos"]["p2"]["error"]


def test_modelo_cargado_predice_y_guarda_linaje(registry):
    servido = modelos.cargar_modelo("p1", cfg.P1_MODELO_REGISTRADO, "champion")

    assert servido.listo
    assert len(servido.modelo.predict(servido.modelo.input_example)) == 5
    # el linaje de la versión queda disponible para los endpoints (por ejemplo, el run de origen)
    assert "source_run_id" in servido.etiquetas


def test_modelo_cargado_guarda_los_parametros_de_su_run(registry_p2):
    servido = modelos.cargar_modelo("p2", cfg.P2_MODELO_REGISTRADO, "champion")

    # P2 lee de aquí sus umbrales de decisión
    assert servido.parametros["data_version"] == "abc123"


def test_sin_parametros_del_run_el_modelo_se_sirve_igual(registry_p2, monkeypatch):
    def get_run_falla(self, id_run):
        raise MlflowException(f"run {id_run} no encontrado")

    monkeypatch.setattr(modelos.MlflowClient, "get_run", get_run_falla)
    servido = modelos.cargar_modelo("p2", cfg.P2_MODELO_REGISTRADO, "champion")

    # No leer los parámetros no debe provocar una pérdida del modelo
    assert servido.listo
    assert servido.version == 1
    assert servido.error is None
    assert servido.parametros == {}


def test_version_sin_run_de_origen_no_tiene_parametros():
    assert modelos.cargar_parametros("p2", None) == {}


def test_alias_se_toma_de_la_variable_de_entorno(registry, monkeypatch):
    monkeypatch.setenv("MODELO_ALIAS", "challenger")
    cargados = modelos.cargar_modelos()

    assert cargados["p1"].alias == "challenger"
    assert not cargados["p1"].listo  # P1 no tiene challenger en este registry


def test_modelo_no_disponible_responde_503():
    servido = modelos.ModeloServido("p2", "modelo", "champion", error="no existe")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(modelos={"p2": servido})))

    with pytest.raises(HTTPException) as error:
        modelos.obtener_modelo(request, "p2")
    assert error.value.status_code == 503


def test_sin_uri_de_mlflow_la_api_no_arranca(monkeypatch):
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", None)
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        modelos.cargar_modelos()


def nombres_de(archivo: Path) -> set[str]:
    nombres = set()
    for linea in archivo.read_text(encoding="utf-8").splitlines():
        linea = linea.split("#")[0].strip()
        if linea and not linea.startswith("-"):
            nombres.add(re.split(r"[=<>!~;\[ ]", linea)[0].lower().replace("_", "-"))
    return nombres


def test_dependencias_de_la_api_estan_fijadas_en_requirements():
    # La imagen instala requirements-api.txt con requirements.txt como restricciones: un paquete
    # que no esté en requirements.txt entraría sin versión fija
    sin_version = nombres_de(RAIZ / "requirements-api.txt") - nombres_de(RAIZ / "requirements.txt")
    assert sin_version == set()
