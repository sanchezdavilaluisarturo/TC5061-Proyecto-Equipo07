"""Pruebas del endpoint de P2: predicción, clase según los umbrales del run y validación de entrada."""

import math

from fastapi.testclient import TestClient
import numpy as np
from pydantic import ValidationError
import pytest
from test_api_p1 import usar_registry_local
from test_p2_modelado import p2_tabla_sintetica

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.api.main import app
from tc5061_proyecto_equipo07.api.p2 import (
    P2_EJEMPLO,
    P2_MAX_REGISTROS,
    P2Registro,
    p2_a_dataframe,
)
from tc5061_proyecto_equipo07.modeling import registro_p2, train
from tc5061_proyecto_equipo07.modeling.predict import p2_decidir_clase

# Tipos con los que features.py construye la tabla de P2
TIPOS_P2 = {
    **{columna: "float32" for columna in cfg.P2_FEATURES_SIN_HISTORICO},
    **{f"{c}_{k}": "int16" for c in ("salidas", "entradas") for k in (2, 4)},
    "dpcapacity": "int8",
    "es_fin_semana": "int8",
}
UMBRALES = {"umbral_sin_bicis": 0.2, "umbral_sin_anclajes": 0.1}


def registro(**cambios) -> dict:
    return {**P2_EJEMPLO, **cambios}


SIN_ACTIVIDAD = registro(
    **{f"neto_{k}_relativo": 0.0 for k in (1, 2, 4, 8)},
    **{f"{c}_{k}": 0 for c in ("salidas", "entradas") for k in (2, 4)},
    cumsum_dia_relativo=0.0,
    hora_sin=0.965926,
    hora_cos=0.258819,
    hist_sin_bicis=0.011922,
    hist_sin_anclajes=0.011789,
)


@pytest.fixture(scope="module")
def cliente(tmp_path_factory):
    """API con un XGBoost de P2 registrado como champion, con los tipos y umbrales del real.

    Se registra una sola vez por archivo: registrar y cargar el modelo es lo más lento.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        usar_registry_local(monkeypatch, tmp_path_factory.mktemp("registry_p2"))
        monkeypatch.setattr(cfg, "MLFLOW_RUN_NAME", None)
        tabla = p2_tabla_sintetica()
        variables = tabla[cfg.P2_FEATURES_CON_HISTORICO].astype(TIPOS_P2)
        modelo = train.p2_construir_modelo("xgb").fit(variables, tabla["clase"])
        id_run = train.p2_registrar_run(
            "xgb", {"data_version": "abc123", **UMBRALES}, {"valid_ap_riesgo": 0.8}, {}, modelo,
            variables.head(5), tabla,
        )
        registro_p2.p2_registrar(id_run, "champion")
        with TestClient(app) as cliente:
            yield cliente


@pytest.fixture
def cliente_sin_modelo(tmp_path, monkeypatch):
    """API con un Registry vacío: arranca, pero P2 no está disponible."""
    usar_registry_local(monkeypatch, tmp_path)
    with TestClient(app) as cliente:
        yield cliente


def hora(bin_del_dia: int) -> dict:
    """hora_sin y hora_cos de un bin de 30 min, como los calcula features.py."""
    angulo = 2 * math.pi * bin_del_dia / 48
    return {"hora_sin": math.sin(angulo), "hora_cos": math.cos(angulo)}


def predecir(cliente, registros: list[dict]):
    return cliente.post("/p2/predecir", json={"registros": registros})


def test_predice_cada_registro_en_orden_con_su_estacion_y_hora(cliente):
    registros = [registro(), {**SIN_ACTIVIDAD, "estacion": 7}]
    respuesta = predecir(cliente, registros)

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert (cuerpo["modelo"], cuerpo["version"]) == (cfg.P2_MODELO_REGISTRADO, 1)
    assert cuerpo["umbrales"] == {"sin_bicis": 0.2, "sin_anclajes": 0.1}
    predicciones = cuerpo["predicciones"]
    assert [(p["estacion"], p["hora"]) for p in predicciones] == [(2, "18:00"), (7, "05:00")]
    # Lo que devuelve la API es lo que predice el modelo con los mismos registros
    modelo = app.state.modelos["p2"].modelo
    esperadas = modelo.predict(p2_a_dataframe([P2Registro(**r) for r in registros], modelo))
    clases = p2_decidir_clase(np.asarray(esperadas, dtype="float64"), {1: 0.2, 2: 0.1})
    for p, esperada, clase in zip(predicciones, esperadas, clases):
        assert list(p["probabilidades"].values()) == pytest.approx(esperada, abs=1e-4)
        assert p["clase"] == cfg.P2_CLASES[clase]
    assert cuerpo["por_clase"] == {
        c: [p["clase"] for p in predicciones].count(c) for c in cfg.P2_CLASES
    }


@pytest.mark.parametrize(
    "umbrales, clase",
    [
        # Un umbral casi en 0 vuelve esa clase la de mayor score
        ({"umbral_sin_bicis": 1e-9, "umbral_sin_anclajes": 0.999}, "sin_bicis"),
        ({"umbral_sin_bicis": 0.999, "umbral_sin_anclajes": 1e-9}, "sin_anclajes"),
        # Con umbrales casi en 1 ningún riesgo llega a su umbral
        ({"umbral_sin_bicis": 0.9999, "umbral_sin_anclajes": 0.9999}, "sin_riesgo"),
    ],
)
def test_clase_sale_de_los_umbrales_del_run(cliente, monkeypatch, umbrales, clase):
    monkeypatch.setattr(app.state.modelos["p2"], "parametros", umbrales)
    respuesta = predecir(cliente, [registro()])

    assert respuesta.status_code == 200
    assert respuesta.json()["predicciones"][0]["clase"] == clase


def test_sin_umbrales_en_el_run_responde_503(cliente, monkeypatch):
    monkeypatch.setattr(app.state.modelos["p2"], "parametros", {"data_version": "abc123"})
    respuesta = predecir(cliente, [registro()])

    assert respuesta.status_code == 503
    assert "umbrales" in respuesta.json()["detail"]


@pytest.mark.parametrize("bin_del_dia, valido", [(10, True), (45, True), (9, False), (46, False)])
def test_solo_los_bins_de_las_0500_a_las_2230_tienen_prediccion(bin_del_dia, valido):
    try:
        P2Registro(**registro(**hora(bin_del_dia)))
        aceptado = True
    except ValidationError:
        aceptado = False
    assert aceptado == valido


@pytest.mark.parametrize(
    "cambios, campo",
    [
        pytest.param({"estacion": 0}, "estacion", id="estación 0"),
        pytest.param({"salidas_2": -1}, "salidas_2", id="salidas negativas"),
        pytest.param({"entradas_4": 5000}, "entradas_4", id="entradas sobre el tope"),
        pytest.param({"neto_1_relativo": 50.0}, "neto_1_relativo", id="neto sobre el tope"),
        pytest.param({"latitude": 19.43, "longitude": -99.13}, "latitude", id="Ciudad de México"),
        pytest.param({"es_fin_semana": 2}, "es_fin_semana", id="fin de semana 2"),
        pytest.param({"salidas_2": "3"}, "salidas_2", id="texto en lugar de número"),
    ],
)
def test_valor_fuera_de_rango_o_de_tipo_responde_422(cliente, cambios, campo):
    respuesta = predecir(cliente, [registro(**cambios)])

    assert respuesta.status_code == 422
    assert campo in {str(p) for e in respuesta.json()["detail"] for p in e["loc"]}


@pytest.mark.parametrize(
    "cambios, mensaje",
    [
        # Las 2 h siguientes se salen del día de servicio
        pytest.param(hora(47), "bin de 30 min", id="23:30"),
        pytest.param({"hora_sin": 0.5, "hora_cos": 0.5}, "bin de 30 min", id="hora fuera de bin"),
        # 0.5 x 15 = 7.5, pero 9 - 3 = 6
        pytest.param({"neto_2_relativo": 0.5}, "neto_2_relativo", id="neto incoherente"),
        # Más salidas en 2 bins que en 4, con los netos coherentes
        pytest.param({"salidas_2": 11, "entradas_2": 17}, "ventana de 2 bins", id="ventana 2 > 4"),
        pytest.param(
            {"hist_sin_bicis": 0.7, "hist_sin_anclajes": 0.6}, "pasar de 1", id="historia > 1"
        ),
    ],
)
def test_registro_incoherente_responde_422(cliente, cambios, mensaje):
    respuesta = predecir(cliente, [registro(**cambios)])

    assert respuesta.status_code == 422
    assert mensaje in respuesta.json()["detail"][0]["msg"]


def test_campo_extra_o_faltante_responde_422(cliente):
    con_extra = registro(clase=1)  # el target no es una entrada del modelo
    sin_estacion = {k: v for k, v in P2_EJEMPLO.items() if k != "estacion"}

    assert predecir(cliente, [con_extra]).status_code == 422
    assert predecir(cliente, [sin_estacion]).status_code == 422


def test_peticion_vacia_o_mayor_al_tope_responde_422(cliente):
    assert predecir(cliente, []).status_code == 422
    assert predecir(cliente, [P2_EJEMPLO] * (P2_MAX_REGISTROS + 1)).status_code == 422


def test_sin_modelo_de_p2_responde_503(cliente_sin_modelo):
    respuesta = predecir(cliente_sin_modelo, [P2_EJEMPLO])

    assert respuesta.status_code == 503
    assert "p2" in respuesta.json()["detail"]