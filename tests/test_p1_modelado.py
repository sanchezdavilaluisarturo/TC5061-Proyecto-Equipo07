"""Pruebas del modelado de P1: transformer de nivel, baseline, métricas y modelos candidatos."""

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.features import P1NivelEstacion
from tc5061_proyecto_equipo07.modeling import train
from tc5061_proyecto_equipo07.modeling.predict import p1_predecir_por_zona
from tc5061_proyecto_equipo07.modeling.train import P1BaselinePromedio, p1_construir_modelo

PROMEDIO = f"viajes_promedio_{cfg.P1_SEMANAS_PROMEDIO}sem"


def p1_tabla_sintetica(n: int = 400, semilla: int = 0) -> pd.DataFrame:
    """Tabla con las columnas de P1_VARIABLES y un target de conteo."""
    rng = np.random.default_rng(semilla)
    tabla = pd.DataFrame(
        {
            "origen_id": rng.integers(1, 6, n),
            "viajes_rezago_1d": rng.poisson(2, n).astype("float32"),
            "viajes_rezago_7d": rng.poisson(2, n).astype("float32"),
            PROMEDIO: rng.gamma(2, 1, n).astype("float32"),
            "hora": rng.choice([5, 8, 13, 18, 23], n),
            "dia_semana": rng.integers(0, 7, n),
            "es_fin_de_semana": 0,
            "mes": rng.integers(1, 13, n),
            "es_festivo": 0,
            "capacidad": 15,
            "latitud": 20.67,
            "longitud": -103.35,
            "zona": pd.Categorical(rng.choice(["GDL", "ZPN"], n)),
            "dias_desde_apertura": rng.integers(0, 900, n),
        }
    )
    tabla["viajes"] = rng.poisson(2, n)
    return tabla


def test_p1_nivel_estacion_solo_aprende_de_fit():
    train_ = p1_tabla_sintetica(semilla=1)
    valid = p1_tabla_sintetica(semilla=2)
    nivel = P1NivelEstacion().fit(train_[cfg.P1_VARIABLES], train_["viajes"])
    antes = nivel.transform(valid[cfg.P1_VARIABLES])["viajes_nivel_estacion"].copy()
    # Transformar otros datos no cambia lo aprendido: el nivel sale solo de train
    nivel.transform(p1_tabla_sintetica(semilla=3)[cfg.P1_VARIABLES])
    despues = nivel.transform(valid[cfg.P1_VARIABLES])["viajes_nivel_estacion"]
    pd.testing.assert_series_equal(antes, despues)
    assert "origen_id" not in nivel.transform(valid[cfg.P1_VARIABLES]).columns


def test_p1_nivel_estacion_es_el_promedio_de_train():
    datos = p1_tabla_sintetica()
    nivel = P1NivelEstacion().fit(datos[cfg.P1_VARIABLES], datos["viajes"])
    fila = datos.iloc[0]
    mismo_grupo = datos[
        (datos["origen_id"] == fila["origen_id"])
        & (datos["dia_semana"] == fila["dia_semana"])
        & (datos["hora"] == fila["hora"])
    ]
    calculado = nivel.transform(datos[cfg.P1_VARIABLES].iloc[[0]])["viajes_nivel_estacion"].iloc[0]
    assert calculado == pytest.approx(mismo_grupo["viajes"].mean(), rel=1e-6)


def test_p1_estacion_nueva_recibe_nivel_del_sistema():
    datos = p1_tabla_sintetica()
    nivel = P1NivelEstacion().fit(datos[cfg.P1_VARIABLES], datos["viajes"])
    nueva = datos[cfg.P1_VARIABLES].iloc[[0]].assign(origen_id=999)  # no existía en train
    salida = nivel.transform(nueva)
    assert not salida["viajes_nivel_estacion"].isna().any()
    assert salida["viajes_nivel_estacion"].iloc[0] == salida["viajes_nivel_sistema"].iloc[0]


def test_p1_baseline_usa_respaldos_cuando_falta_historia():
    datos = p1_tabla_sintetica()
    baseline = P1BaselinePromedio().fit(datos[cfg.P1_VARIABLES], datos["viajes"])
    nueva = datos[cfg.P1_VARIABLES].iloc[[0, 1]].copy()
    nueva[PROMEDIO] = [3.5, np.nan]  # la segunda fila no tiene historia
    nueva["origen_id"] = [nueva["origen_id"].iloc[0], 999]  # estación que no existía en train
    prediccion = baseline.predict(nueva)
    assert prediccion[0] == pytest.approx(3.5)
    nivel_sistema = baseline.nivel_.transform(nueva)["viajes_nivel_sistema"].iloc[1]
    assert prediccion[1] == pytest.approx(nivel_sistema)
    assert not np.isnan(prediccion).any()


def test_p1_metricas_llevan_el_nombre_del_conjunto():
    datos = p1_tabla_sintetica()
    modelo = P1BaselinePromedio().fit(datos[cfg.P1_VARIABLES], datos["viajes"])
    metricas, prediccion = train.p1_evaluar(modelo, datos, "valid")
    assert set(metricas) == {"valid_mae", "valid_rmse"}
    assert len(prediccion) == len(datos)


@pytest.mark.parametrize("clave", ["baseline", "hgb", "xgb"])
def test_p1_modelos_entrenan_y_predicen_conteos(clave):
    datos = p1_tabla_sintetica(n=600)
    modelo = p1_construir_modelo(clave).fit(datos[cfg.P1_VARIABLES], datos["viajes"])
    prediccion = modelo.predict(datos[cfg.P1_VARIABLES].astype({"zona": str}))
    assert prediccion.shape == (len(datos),)
    assert (prediccion >= 0).all()  # pérdida Poisson: nunca predice viajes negativos


def test_p1_modelo_desconocido_falla_con_mensaje_claro():
    with pytest.raises(ValueError, match="desconocido"):
        p1_construir_modelo("random_forest")


def test_p1_prediccion_por_zona_suma_lo_de_cada_estacion():
    datos = pd.DataFrame(
        {
            "hora_inicio": pd.to_datetime(
                ["2025-10-01 08:00", "2025-10-01 18:00", "2025-10-01 08:00", "2025-10-02 08:00"]
            ),
            "zona": pd.Categorical(["GDL", "GDL", "ZPN", "GDL"]),
            "viajes": [3, 1, 2, 5],
        }
    )
    por_zona = p1_predecir_por_zona(datos, np.array([2.0, 1.5, 1.0, 4.0]))
    gdl_dia1 = por_zona[(por_zona["zona"] == "GDL") & (por_zona["dia"] == "2025-10-01")]
    assert gdl_dia1["prediccion"].item() == pytest.approx(3.5)
    assert gdl_dia1["real"].item() == 4
    assert len(por_zona) == 3
    assert por_zona["prediccion"].sum() == pytest.approx(8.5)  # no se pierde ni duplica nada
