"""Pruebas del endpoint de P3: grupos, columnas de la signature y validación de entrada."""

from collections import Counter

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from test_api_p1 import usar_registry_local
from test_p3_registro import crear_candidato
from test_p3_tuning import preparar_experimento_local

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.api.main import app
from tc5061_proyecto_equipo07.api.p3 import (
    P3_EJEMPLO,
    P3_MAX_ESTACIONES,
    P3Estacion,
    p3_a_dataframe,
    p3_columnas_horarias,
)
from tc5061_proyecto_equipo07.modeling import registro_p3, train

COLUMNAS_P3 = (
    p3_columnas_horarias("lab")
    + p3_columnas_horarias("fin")
    + ["log_viajes_por_dia", "neto_sobre_salidas", "prop_circulares", "duracion_mediana_min"]
)
N_HORAS = len(cfg.P3_HORAS_SERVICIO)
# Por grupo: hora de más salidas (posición en el perfil), log de viajes, neto, circulares, minutos
GRUPOS_SINTETICOS = [
    (3, 2.0, -0.2, 0.02, 8.0),
    (14, 4.0, 0.1, 0.06, 12.0),
    (8, 6.0, 0.4, 0.15, 20.0),
]


def perfil_con_pico(rng, pico: int) -> np.ndarray:
    """Proporciones por hora que suman 1, con el mayor peso en la hora `pico`."""
    pesos = rng.dirichlet(np.ones(N_HORAS))
    pesos[pico] += 1.5
    return pesos / pesos.sum()


def tabla_p3_sintetica(por_grupo: int = 10) -> pd.DataFrame:
    """Estaciones con las 44 columnas del champion, en tres grupos bien separados."""
    rng = np.random.default_rng(7)
    filas = []
    for pico, log_viajes, neto, circulares, minutos in GRUPOS_SINTETICOS:
        for _ in range(por_grupo):
            fila = dict(zip(p3_columnas_horarias("lab"), perfil_con_pico(rng, pico)))
            fila |= dict(zip(p3_columnas_horarias("fin"), perfil_con_pico(rng, pico)))
            fila |= {
                "log_viajes_por_dia": log_viajes + rng.normal(0, 0.1),
                "neto_sobre_salidas": neto + rng.normal(0, 0.02),
                "prop_circulares": circulares + rng.normal(0, 0.005),
                "duracion_mediana_min": minutos + rng.normal(0, 0.5),
            }
            filas.append(fila)
    return pd.DataFrame(filas)[COLUMNAS_P3]


def estacion(fila: pd.Series, origen_id: int) -> dict:
    """La fila de la tabla en el formato de la petición (perfiles como arreglos)."""
    return {
        "origen_id": origen_id,
        "perfil_laboral": fila[p3_columnas_horarias("lab")].tolist(),
        "perfil_fin_de_semana": fila[p3_columnas_horarias("fin")].tolist(),
        "log_viajes_por_dia": float(fila["log_viajes_por_dia"]),
        "neto_sobre_salidas": float(fila["neto_sobre_salidas"]),
        "prop_circulares": float(fila["prop_circulares"]),
        "duracion_mediana_min": float(fila["duracion_mediana_min"]),
    }


@pytest.fixture(scope="module")
def tabla():
    return tabla_p3_sintetica()


@pytest.fixture(scope="module")
def cliente(tmp_path_factory, tabla):
    """API con un modelo de P3 registrado como champion, con las 44 columnas del real.

    Se registra una sola vez por archivo: registrar y cargar el modelo es lo más lento.
    """
    carpeta = tmp_path_factory.mktemp("registry_p3")
    with pytest.MonkeyPatch.context() as monkeypatch:
        usar_registry_local(monkeypatch, carpeta)
        monkeypatch.setattr(cfg, "MLFLOW_RUN_NAME", None)
        monkeypatch.setattr(train, "p3_preparar_experimento", preparar_experimento_local(carpeta))
        modelo = Pipeline(
            [("escalador", StandardScaler()), ("kmeans", KMeans(3, n_init=10, random_state=0))]
        ).fit(tabla)
        id_run = crear_candidato(modelo=modelo, ejemplo=tabla.head(3))
        registro_p3.p3_registrar(id_run, "champion")
        with TestClient(app) as cliente:
            yield cliente


@pytest.fixture
def cliente_sin_modelo(tmp_path, monkeypatch):
    """API con un Registry vacío: arranca, pero P3 no está disponible."""
    usar_registry_local(monkeypatch, tmp_path)
    with TestClient(app) as cliente:
        yield cliente


def segmentar(cliente, estaciones: list[dict]):
    return cliente.post("/p3/segmentar", json={"estaciones": estaciones})


def test_asigna_el_grupo_de_cada_estacion_en_orden(cliente, tabla):
    # Una estación de cada grupo sintético (filas 0, 10 y 20) y otra repetida del primero
    filas = [0, 10, 20, 1]
    estaciones = [estacion(tabla.iloc[i], origen_id=100 + n) for n, i in enumerate(filas)]
    respuesta = segmentar(cliente, estaciones)

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert (cuerpo["modelo"], cuerpo["version"]) == (cfg.P3_MODELO_REGISTRADO, 1)
    assert [p["origen_id"] for p in cuerpo["predicciones"]] == [100, 101, 102, 103]
    grupos = [p["grupo"] for p in cuerpo["predicciones"]]
    # Lo que devuelve la API es lo que asigna el modelo con la misma tabla
    modelo = app.state.modelos["p3"].modelo
    assert grupos == modelo.predict(tabla.iloc[filas]).tolist()
    # Los grupos sintéticos están separados: los tres distintos y el repetido con el primero
    assert len(set(grupos[:3])) == 3
    assert grupos[3] == grupos[0]


def test_estaciones_por_grupo_cuenta_las_de_la_peticion(cliente, tabla):
    estaciones = [estacion(tabla.iloc[i], origen_id=i + 1) for i in range(len(tabla))]
    cuerpo = segmentar(cliente, estaciones).json()

    cuenta = Counter(p["grupo"] for p in cuerpo["predicciones"])
    assert {int(g): n for g, n in cuerpo["estaciones_por_grupo"].items()} == dict(cuenta)
    assert sum(cuerpo["estaciones_por_grupo"].values()) == len(tabla)
    # Tres grupos de 10 estaciones
    assert sorted(cuenta.values()) == [10, 10, 10]


def test_dataframe_toma_columnas_y_tipos_de_la_signature(cliente):
    modelo = app.state.modelos["p3"].modelo
    datos = p3_a_dataframe([P3Estacion(**P3_EJEMPLO)], modelo)

    esquema = modelo.metadata.get_input_schema().inputs
    assert list(datos.columns) == [c.name for c in esquema] == COLUMNAS_P3
    assert datos.shape == (1, 44)
    assert {c: datos[c].dtype for c in datos.columns} == {
        c.name: c.type.to_pandas() for c in esquema
    }
    # Los arreglos se abren en el orden de las horas de servicio
    assert datos["lab_h00"].iloc[0] == P3_EJEMPLO["perfil_laboral"][0]
    assert datos["lab_h05"].iloc[0] == P3_EJEMPLO["perfil_laboral"][1]
    assert datos["fin_h23"].iloc[0] == P3_EJEMPLO["perfil_fin_de_semana"][-1]
    assert datos["duracion_mediana_min"].iloc[0] == P3_EJEMPLO["duracion_mediana_min"]


def test_estacion_sin_viajes_de_fin_de_semana_acepta_el_perfil_en_ceros(cliente, tabla):
    sin_fin = estacion(tabla.iloc[0], origen_id=7) | {"perfil_fin_de_semana": [0.0] * N_HORAS}
    respuesta = segmentar(cliente, [sin_fin])

    assert respuesta.status_code == 200
    assert respuesta.json()["predicciones"][0]["origen_id"] == 7


def test_acepta_enteros_donde_el_esquema_pide_decimales(cliente, tabla):
    # JSON no distingue 0 de 0.0: un perfil con ceros enteros debe aceptarse
    fila = estacion(tabla.iloc[0], origen_id=7)
    fila["perfil_laboral"] = [1 if hora == 0 else 0 for hora in range(N_HORAS)]

    assert segmentar(cliente, [fila]).status_code == 200


@pytest.mark.parametrize(
    "cambios",
    [
        pytest.param({"perfil_laboral": P3_EJEMPLO["perfil_laboral"][:-1]}, id="19 horas"),
        pytest.param({"perfil_laboral": P3_EJEMPLO["perfil_laboral"] + [0.0]}, id="21 horas"),
        pytest.param({"perfil_fin_de_semana": [0.5] + [0.0] * (N_HORAS - 1)}, id="suma distinta"),
        pytest.param({"perfil_laboral": [-0.1] + [0.0] * (N_HORAS - 1)}, id="proporción negativa"),
        pytest.param({"salidas": 10}, id="campo de más"),
        pytest.param({"prop_circulares": "0.1"}, id="texto en lugar de número"),
        pytest.param({"prop_circulares": 1.5}, id="circulares mayor que 1"),
        pytest.param({"neto_sobre_salidas": -1.5}, id="neto menor que -1"),
        pytest.param({"log_viajes_por_dia": -1}, id="log de viajes negativo"),
        pytest.param({"duracion_mediana_min": 0}, id="duración cero"),
        pytest.param({"origen_id": 0}, id="estación 0"),
    ],
)
def test_entrada_invalida_responde_422(cliente, cambios):
    respuesta = segmentar(cliente, [{**P3_EJEMPLO, **cambios}])

    assert respuesta.status_code == 422


def test_campo_faltante_responde_422(cliente):
    sin_duracion = {k: v for k, v in P3_EJEMPLO.items() if k != "duracion_mediana_min"}

    assert segmentar(cliente, [sin_duracion]).status_code == 422


def test_estaciones_repetidas_responden_422(cliente):
    respuesta = segmentar(cliente, [P3_EJEMPLO, P3_EJEMPLO])

    assert respuesta.status_code == 422
    assert "repetido" in respuesta.text


def test_peticion_vacia_o_demasiado_grande_responde_422(cliente):
    grande = [{**P3_EJEMPLO, "origen_id": i + 1} for i in range(P3_MAX_ESTACIONES + 1)]

    assert segmentar(cliente, []).status_code == 422
    assert segmentar(cliente, grande).status_code == 422


def test_sin_modelo_de_p3_responde_503(cliente_sin_modelo):
    respuesta = segmentar(cliente_sin_modelo, [P3_EJEMPLO])

    assert respuesta.status_code == 503
    assert "p3" in respuesta.json()["detail"]
