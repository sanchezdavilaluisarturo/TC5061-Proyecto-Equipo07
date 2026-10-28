"""Pruebas de las métricas de evaluación de P1: MAE por hora, por estación y por zona."""

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo07.modeling import train


@pytest.fixture
def datos() -> pd.DataFrame:
    """Dos estaciones de GDL y una de ZPN, a las 8:00 y 18:00 de dos días."""
    filas = [
        (estacion, zona, f"2025-10-0{dia} {hora:02d}:00", viajes)
        for estacion, zona, viajes in [(1, "GDL", 4), (2, "GDL", 2), (3, "ZPN", 6)]
        for dia in (1, 2)
        for hora in (8, 18)
    ]
    tabla = pd.DataFrame(filas, columns=["origen_id", "zona", "hora_inicio", "viajes"])
    tabla["hora_inicio"] = pd.to_datetime(tabla["hora_inicio"])
    tabla["hora"] = tabla["hora_inicio"].dt.hour
    tabla["zona"] = pd.Categorical(tabla["zona"])
    return tabla


def test_p1_prediccion_perfecta_da_error_cero(datos):
    perfecta = datos["viajes"].to_numpy(dtype=float)
    assert (train.p1_mae_por_hora(datos, perfecta) == 0).all()
    assert (train.p1_peores_estaciones(datos, perfecta)["mae"] == 0).all()
    assert (train.p1_mae_por_zona(datos, perfecta)["mae_diario"] == 0).all()


def test_p1_mae_por_hora_promedia_el_error_absoluto_de_cada_hora(datos):
    prediccion = datos["viajes"].to_numpy(dtype=float)
    prediccion[datos["hora"] == 8] += 3  # sobreestima 3 viajes a las 8:00
    prediccion[(datos["hora"] == 18) & (datos["origen_id"] == 1)] -= 4  # subestima en una estación
    mae = train.p1_mae_por_hora(datos, prediccion)
    assert mae.loc[8] == pytest.approx(3)
    assert mae.loc[18] == pytest.approx(4 * 2 / 6)  # 2 de las 6 filas de las 18:00 fallan por 4


def test_p1_peores_estaciones_ordena_de_mayor_a_menor_y_respeta_n(datos):
    error_por_estacion = datos["origen_id"].map({1: 1.0, 2: 5.0, 3: 2.0})
    prediccion = datos["viajes"].to_numpy(dtype=float) + error_por_estacion.to_numpy()
    peores = train.p1_peores_estaciones(datos, prediccion)
    assert list(peores.index) == [2, 3, 1]
    assert peores.loc[2, "mae"] == 5
    assert peores.loc[2, "horas"] == 4
    assert peores.loc[3, "viajes_promedio"] == 6
    assert list(train.p1_peores_estaciones(datos, prediccion, n=1).index) == [2]


def test_p1_peores_estaciones_con_n_mayor_que_las_estaciones_regresa_todas(datos):
    prediccion = np.zeros(len(datos))
    assert len(train.p1_peores_estaciones(datos, prediccion, n=50)) == 3


def test_p1_mae_por_zona_suma_por_dia_antes_de_medir_el_error(datos):
    """El operador planea la flota por zona y día: los errores entre estaciones se compensan."""
    prediccion = datos["viajes"].to_numpy(dtype=float)
    prediccion[datos["origen_id"] == 1] += 2  # sobreestima en una estación de GDL
    prediccion[datos["origen_id"] == 2] -= 2  # y subestima lo mismo en la otra
    prediccion[datos["origen_id"] == 3] += 1  # ZPN: 1 viaje de más en cada hora, 2 por día
    por_zona = train.p1_mae_por_zona(datos, prediccion)
    assert por_zona.loc["GDL", "mae_diario"] == 0
    assert por_zona.loc["ZPN", "mae_diario"] == 2  # promedio de los 2 días, no la suma (4)
    assert por_zona.loc["GDL", "dias"] == 2
    assert por_zona.loc["GDL", "viajes_diarios_promedio"] == (4 + 2) * 2  # 2 horas por día
    # Por estación, en cambio, las dos de GDL sí tienen error
    assert (train.p1_peores_estaciones(datos, prediccion).loc[[1, 2], "mae"] == 2).all()
