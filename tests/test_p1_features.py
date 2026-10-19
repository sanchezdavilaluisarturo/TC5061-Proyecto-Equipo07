"""Pruebas de las funciones p1_* de features.py (target y variables de P1)."""

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo07 import features


def p1_viajes_dos_semanas() -> pd.DataFrame:
    """Estación 1 desde el lunes 2024-01-08; estación 2 abre el 2024-01-15 (una semana después)."""
    inicio = pd.to_datetime(
        ["2024-01-08 08:10", "2024-01-08 08:40"]  # lunes, 2 viajes a las 8:00
        + ["2024-01-09 08:05"] * 5  # martes, 5 viajes a las 8:00
        + ["2024-01-15 08:20"] * 3  # lunes siguiente, 3 viajes a las 8:00
        + ["2024-01-15 02:30"]  # hora sin servicio: no entra a la tabla
        + ["2024-01-15 18:00", "2024-01-20 18:00"]  # estación 2
    )
    return pd.DataFrame(
        {
            "inicio_del_viaje": inicio,
            "origen_id": [1] * 11 + [2, 2],
            "periodo_archivo": "2024-01",
        }
    )


def p1_estaciones() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["(GDL-001) C. Uno / C. Dos", "ex-(ZPN-002) C. Tres / C. Cuatro"],
            "latitude": [20.67, 20.72],
            "longitude": [-103.35, -103.38],
            "dpcapacity": [15, 11],
        }
    )


@pytest.fixture
def tabla() -> pd.DataFrame:
    return features.p1_tabla_modelado(p1_viajes_dos_semanas(), p1_estaciones(), "2024-01", "2024-01")


def fila(tabla: pd.DataFrame, estacion: int, hora: str) -> pd.Series:
    return tabla[(tabla["origen_id"] == estacion) & (tabla["hora_inicio"] == hora)].iloc[0]


def test_p1_target_cuenta_viajes_y_excluye_horas_sin_servicio(tabla):
    assert fila(tabla, 1, "2024-01-09 08:00")["viajes"] == 5
    assert not tabla["hora_inicio"].dt.hour.isin([1, 2, 3, 4]).any()
    assert tabla["viajes"].sum() == 12  # 13 viajes menos el de las 2:30


def test_p1_rezagos_solo_usan_dias_anteriores(tabla):
    martes = fila(tabla, 1, "2024-01-09 08:00")
    assert martes["viajes_rezago_1d"] == 2  # lunes 8:00, no los 5 del propio martes
    lunes_siguiente = fila(tabla, 1, "2024-01-15 08:00")
    assert lunes_siguiente["viajes_rezago_7d"] == 2
    assert lunes_siguiente["viajes_promedio_4sem"] == 2  # solo hay una semana anterior con dato
    assert np.isnan(fila(tabla, 1, "2024-01-08 08:00")["viajes_rezago_1d"])  # antes no operaba


def test_p1_malla_inicia_en_la_apertura_de_cada_estacion(tabla):
    estacion_2 = tabla[tabla["origen_id"] == 2]
    assert estacion_2["hora_inicio"].min() == pd.Timestamp("2024-01-15 18:00")
    assert fila(tabla, 2, "2024-01-16 18:00")["viajes"] == 0  # opera y no tuvo viajes
    assert fila(tabla, 2, "2024-01-16 18:00")["dias_desde_apertura"] == 1
    assert fila(tabla, 2, "2024-01-20 18:00")["zona"] == "ZPN"


def test_p1_rezago_rechaza_horizontes_menores_a_un_dia():
    with pytest.raises(ValueError):
        features.p1_rezago(np.zeros((1, 40), dtype="float32"), 0)


def test_p1_dias_festivos_incluye_lunes_movibles():
    festivos = features.p1_dias_festivos([2020, 2024])
    for dia in ["2020-02-03", "2020-03-16", "2020-11-16", "2024-10-01", "2024-12-25"]:
        assert pd.Timestamp(dia) in festivos
    assert pd.Timestamp("2020-02-10") not in festivos
    assert pd.Timestamp("2025-10-01") not in features.p1_dias_festivos([2025])
