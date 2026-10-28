"""Pruebas de las funciones p1_* de features.py (target y variables de P1)."""

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo07 import features

UNA_SEMANA = 7 * features.P1_HORAS_POR_DIA  # posiciones de la malla en una semana


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
    return features.p1_tabla_modelado(
        p1_viajes_dos_semanas(), p1_estaciones(), "2024-01", "2024-01"
    )


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


def p1_malla_cinco_semanas() -> tuple[np.ndarray, int]:
    """Una estación con 5 semanas de horas en 0 y la posición a revisar en la quinta semana."""
    matriz = np.zeros((1, 5 * UNA_SEMANA), dtype="float32")
    return matriz, 4 * UNA_SEMANA + 3


def test_p1_promedio_usa_la_misma_hora_de_las_semanas_anteriores():
    matriz, posicion = p1_malla_cinco_semanas()
    for semana, viajes in enumerate([4, 3, 2, 1], start=1):
        matriz[0, posicion - semana * UNA_SEMANA] = viajes
    promedio = features.p1_promedio_misma_hora(matriz, semanas=4)
    assert promedio[0, posicion] == pytest.approx(2.5)


def test_p1_promedio_ignora_las_semanas_sin_dato():
    matriz, posicion = p1_malla_cinco_semanas()
    for semana, viajes in enumerate([4, 3, 2, 1], start=1):
        matriz[0, posicion - semana * UNA_SEMANA] = viajes
    matriz[0, posicion - 2 * UNA_SEMANA] = np.nan  # la estación no operaba esa semana
    promedio = features.p1_promedio_misma_hora(matriz, semanas=4)
    assert promedio[0, posicion] == pytest.approx((4 + 2 + 1) / 3)


def test_p1_promedio_sin_semanas_anteriores_es_nan():
    matriz, _ = p1_malla_cinco_semanas()
    promedio = features.p1_promedio_misma_hora(matriz, semanas=4)
    assert np.isnan(promedio[0, :UNA_SEMANA]).all()  # primera semana: no hay historia


def test_p1_promedio_no_usa_la_hora_que_se_predice():
    matriz, posicion = p1_malla_cinco_semanas()
    sin_cambio = features.p1_promedio_misma_hora(matriz, semanas=4)[0, posicion]
    matriz[0, posicion] = 100  # el target de esa hora no debe entrar a su propia variable
    assert features.p1_promedio_misma_hora(matriz, semanas=4)[0, posicion] == sin_cambio


def calendario(*horas: str) -> pd.DataFrame:
    return features.p1_variables_calendario(pd.Series(pd.to_datetime(list(horas))))


def test_p1_calendario_marca_el_fin_de_semana():
    tabla = calendario(
        "2024-01-05 08:00", "2024-01-06 08:00", "2024-01-07 08:00", "2024-01-08 08:00"
    )
    assert tabla["dia_semana"].tolist() == [4, 5, 6, 0]  # viernes, sábado, domingo, lunes
    assert tabla["es_fin_de_semana"].tolist() == [0, 1, 1, 0]


def test_p1_calendario_marca_todo_el_dia_festivo_y_no_el_siguiente():
    # 2024-02-05 es el primer lunes de febrero (Día de la Constitución)
    tabla = calendario("2024-02-05 00:00", "2024-02-05 23:00", "2024-02-06 08:00")
    assert tabla["es_festivo"].tolist() == [1, 1, 0]


def test_p1_calendario_en_el_cambio_de_anio():
    tabla = calendario("2024-12-31 23:00", "2025-01-01 00:00")
    assert tabla["mes"].tolist() == [12, 1]
    assert tabla["hora"].tolist() == [23, 0]
    assert tabla["es_festivo"].tolist() == [0, 1]  # el festivo del año nuevo también se calcula


def test_p1_calendario_conserva_el_indice_y_usa_enteros_pequenos():
    horas = pd.Series(pd.to_datetime(["2024-03-01 08:00", "2024-03-02 18:00"]), index=[7, 9])
    tabla = features.p1_variables_calendario(horas)
    assert list(tabla.index) == [7, 9]
    assert (tabla.dtypes == "int8").all()
