"""Pruebas del endpoint de P1: predicción, tipos de la signature y validación de entrada."""

from fastapi.testclient import TestClient
import mlflow
import pytest
from test_p1_modelado import p1_tabla_sintetica

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.api.main import app
from tc5061_proyecto_equipo07.api.p1 import (
    P1_EJEMPLO,
    P1_MAX_REGISTROS,
    P1Registro,
    p1_a_dataframe,
)
from tc5061_proyecto_equipo07.modeling import registro_p1, train
from tc5061_proyecto_equipo07.modeling.train import p1_construir_modelo

# Tipos con los que features.py construye la tabla de P1 (los del champion)
TIPOS_P1 = {
    "origen_id": "int32",
    "hora": "int8",
    "dia_semana": "int8",
    "es_fin_de_semana": "int8",
    "mes": "int8",
    "es_festivo": "int8",
    "capacidad": "int16",
    "latitud": "float32",
    "longitud": "float32",
    "zona": str,
    "dias_desde_apertura": "int16",
}


def usar_registry_local(monkeypatch, carpeta) -> None:
    monkeypatch.chdir(carpeta)
    uri = f"sqlite:///{carpeta / 'mlflow.db'}"
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("MODELO_ALIAS", raising=False)
    mlflow.set_tracking_uri(uri)


@pytest.fixture(scope="module")
def cliente(tmp_path_factory):
    """API con un modelo de P1 registrado como champion, con los mismos tipos que el real.

    Se registra una sola vez por archivo: registrar y cargar el modelo es lo más lento.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        usar_registry_local(monkeypatch, tmp_path_factory.mktemp("registry_p1"))
        tabla = p1_tabla_sintetica()
        tabla["latitud"], tabla["longitud"] = P1_EJEMPLO["latitud"], P1_EJEMPLO["longitud"]
        variables = tabla[cfg.P1_VARIABLES].astype(TIPOS_P1)
        modelo = p1_construir_modelo("baseline").fit(variables, tabla["viajes"])
        id_run = train.p1_registrar_run(
            "baseline", {"data_version": "abc123"}, {"valid_mae": 1.0}, [], modelo,
            variables.head(5), tabla,
        )  # fmt: skip
        registro_p1.p1_registrar(id_run, "champion")
        with TestClient(app) as cliente:
            yield cliente


def registro(**cambios) -> dict:
    return {**P1_EJEMPLO, **cambios}


def test_predice_cada_registro_en_orden_y_suma_por_zona(cliente):
    registros = [
        registro(),
        registro(origen_id=55, hora=8, dia_semana=0, es_fin_de_semana=0),
        registro(origen_id=40, zona="ZPN", latitud=20.72, longitud=-103.40),
    ]
    respuesta = cliente.post("/p1/predecir", json={"registros": registros})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert (cuerpo["modelo"], cuerpo["version"]) == (cfg.P1_MODELO_REGISTRADO, 1)
    predicciones = cuerpo["predicciones"]
    assert [(p["origen_id"], p["hora"]) for p in predicciones] == [(168, 23), (55, 8), (40, 23)]
    assert all(p["viajes_esperados"] >= 0 for p in predicciones)
    assert all(round(p["viajes_esperados"], 4) == p["viajes_esperados"] for p in predicciones)
    gdl = predicciones[0]["viajes_esperados"] + predicciones[1]["viajes_esperados"]
    assert cuerpo["por_zona"]["GDL"] == pytest.approx(gdl, abs=1e-3)
    assert cuerpo["por_zona"]["ZPN"] == pytest.approx(predicciones[2]["viajes_esperados"])


def test_estacion_sin_historia_acepta_rezagos_nulos(cliente):
    sin_historia = registro(
        viajes_rezago_1d=None, viajes_rezago_7d=None, viajes_promedio_4sem=None
    )
    respuesta = cliente.post("/p1/predecir", json={"registros": [sin_historia]})

    assert respuesta.status_code == 200
    assert respuesta.json()["predicciones"][0]["viajes_esperados"] >= 0


def test_dataframe_toma_los_tipos_de_la_signature(cliente):
    modelo = app.state.modelos["p1"].modelo
    datos = p1_a_dataframe([P1Registro(**registro(viajes_rezago_1d=None))], modelo)

    esperados = {c.name: c.type.to_pandas() for c in modelo.metadata.get_input_schema().inputs}
    assert list(datos.columns) == cfg.P1_VARIABLES
    assert {columna: datos[columna].dtype for columna in esperados} == esperados
    assert datos["viajes_rezago_1d"].isna().all()


@pytest.mark.parametrize(
    "cambios, campo",
    [
        ({"hora": 3}, "hora"),  # el sistema no opera de 1:00 a 4:59
        ({"hora": 24}, "hora"),
        ({"dia_semana": 7}, "dia_semana"),
        ({"mes": 13}, "mes"),
        ({"es_festivo": 2}, "es_festivo"),
        ({"zona": "CDMX"}, "zona"),
        ({"latitud": 19.43, "longitud": -99.13}, "latitud"),  # Ciudad de México
        ({"capacidad": 0}, "capacidad"),
        ({"viajes_rezago_1d": -1}, "viajes_rezago_1d"),
        ({"viajes_rezago_7d": 5000}, "viajes_rezago_7d"),
        ({"dias_desde_apertura": -5}, "dias_desde_apertura"),
        ({"hora": "8"}, "hora"),  # strict: el texto no se convierte en silencio
        ({"origen_id": 12.5}, "origen_id"),
    ],
)
def test_valor_fuera_de_rango_o_de_tipo_responde_422(cliente, cambios, campo):
    respuesta = cliente.post("/p1/predecir", json={"registros": [registro(**cambios)]})

    assert respuesta.status_code == 422
    assert campo in {str(p) for e in respuesta.json()["detail"] for p in e["loc"]}


def test_fin_de_semana_incoherente_con_el_dia_responde_422(cliente):
    # Lunes (0) marcado como fin de semana
    incoherente = registro(dia_semana=0, es_fin_de_semana=1)
    respuesta = cliente.post("/p1/predecir", json={"registros": [incoherente]})

    assert respuesta.status_code == 422
    assert "es_fin_de_semana" in respuesta.json()["detail"][0]["msg"]


def test_campo_extra_o_faltante_responde_422(cliente):
    con_extra = registro(viajes=3)  # el target no es una entrada del modelo
    sin_zona = {k: v for k, v in P1_EJEMPLO.items() if k != "zona"}

    for cuerpo in ({"registros": [con_extra]}, {"registros": [sin_zona]}):
        assert cliente.post("/p1/predecir", json=cuerpo).status_code == 422


def test_peticion_vacia_o_mayor_al_tope_responde_422(cliente):
    vacia = {"registros": []}
    excedida = {"registros": [P1_EJEMPLO] * (P1_MAX_REGISTROS + 1)}

    assert cliente.post("/p1/predecir", json=vacia).status_code == 422
    assert cliente.post("/p1/predecir", json=excedida).status_code == 422


def test_sin_modelo_de_p1_responde_503(tmp_path, monkeypatch):
    usar_registry_local(monkeypatch, tmp_path)
    with TestClient(app) as cliente:  # registry vacío: P1 no carga
        respuesta = cliente.post("/p1/predecir", json={"registros": [P1_EJEMPLO]})

    assert respuesta.status_code == 503
    assert "p1" in respuesta.json()["detail"]
