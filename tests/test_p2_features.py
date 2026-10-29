"""Pruebas de las funciones p2_* de features.py (limpieza, etiqueta de riesgo y variables de P2)."""

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import features


def p2_estaciones() -> pd.DataFrame:
    """Estaciones 1 y 2 con capacidad 4; la 3 tiene capacidad 0 y P2 la descarta."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "latitude": [20.67, 20.72, 20.70],
            "longitude": [-103.35, -103.38, -103.36],
            "dpcapacity": [4, 4, 0],
        }
    )


def p2_viajes_de_una_manana() -> pd.DataFrame:
    """4 viajes de la estación 1 a la 2 entre 8:10 y 8:20 del lunes 2024-01-08."""
    inicio = pd.to_datetime(["2024-01-08 08:10"] * 4)
    return pd.DataFrame(
        {
            "inicio_del_viaje": inicio,
            "fin_del_viaje": inicio + pd.Timedelta(minutes=10),
            "origen_id": [1] * 4,
            "destino_id": [2] * 4,
        }
    )


def test_p2_solo_conserva_estaciones_con_capacidad():
    con_capacidad = features.p2_estaciones_con_capacidad(p2_estaciones())
    assert con_capacidad["id"].tolist() == [1, 2]


def test_p2_descarta_viajes_con_origen_o_destino_inexistente():
    viajes = pd.DataFrame({"origen_id": [1, 1, 9, 3], "destino_id": [2, 9, 1, 1]})
    estaciones = features.p2_estaciones_con_capacidad(p2_estaciones())
    limpios = features.p2_descartar_estaciones_inexistentes(viajes, estaciones)
    # Se quedan solo los que salen y llegan a una estación con capacidad (la 3 tiene 0)
    assert limpios.index.tolist() == [0]


def test_p2_fecha_limite_es_exclusiva():
    fin = pd.to_datetime(["2026-09-30 23:59:59", "2026-10-01 00:00:00", "2026-10-01 00:10:00"])
    viajes = pd.DataFrame({"fin_del_viaje": fin})
    limpios = features.p2_descartar_fuera_de_rango(viajes, "2026-10-01 00:00:00")
    assert limpios["fin_del_viaje"].tolist() == [fin[0]]


def test_p2_descarta_viajes_que_inician_en_horas_sin_servicio():
    inicio = pd.to_datetime(
        ["2024-01-08 00:59", "2024-01-08 01:00", "2024-01-08 04:59", "2024-01-08 05:00"]
    )
    limpios = features.p2_descartar_horas_sin_servicio(
        pd.DataFrame({"inicio_del_viaje": inicio}), [1, 2, 3, 4]
    )
    assert limpios["inicio_del_viaje"].dt.hour.tolist() == [0, 5]


def test_p2_limpieza_aplica_las_tres_reglas():
    # Reglas:
    # Solo viajes a estaciones con capacidad
    # No incluir viajes que terminen después de MES_FIN
    # No incluir viajes en horarios fuera de servicio
    inicio = pd.to_datetime(
        ["2024-01-08 08:00", "2024-01-08 02:00", "2024-01-08 08:00", "2026-09-30 23:50"]
    )
    viajes = pd.DataFrame(
        {
            "inicio_del_viaje": inicio,
            "fin_del_viaje": inicio + pd.Timedelta(minutes=15),  # el último termina en octubre
            "origen_id": [1, 1, 3, 1],  # la 3 no tiene capacidad
            "destino_id": [2, 2, 1, 2],
        }
    )
    limpios = features.p2_limpiar_viajes(viajes, p2_estaciones())
    assert limpios.index.tolist() == [0]


def test_p2_intervalo_manda_horas_sin_servicio_al_bin_de_las_0():
    ts = pd.Series(
        pd.to_datetime(
            ["2024-01-08 08:40", "2024-01-08 00:45", "2024-01-08 01:10", "2024-01-08 04:59"]
        )
    )
    bins = features.p2_intervalo(ts, 30)
    esperado = pd.to_datetime(
        ["2024-01-08 08:30", "2024-01-08 00:30", "2024-01-08 00:30", "2024-01-08 00:30"]
    )
    assert bins.tolist() == esperado.tolist()


def test_p2_cambios_rellena_con_ceros_los_bins_sin_actividad():
    viajes = pd.DataFrame(
        {
            "inicio_del_viaje": pd.to_datetime(["2024-01-08 08:10", "2024-01-08 08:40"]),
            "fin_del_viaje": pd.to_datetime(["2024-01-08 08:20", "2024-01-08 09:05"]),
            "origen_id": [1, 2],
            "destino_id": [2, 1],
        }
    )
    cambios = features.p2_obtener_dataframe_de_cambios(viajes, 30)
    # Bins de 00:00 a 09:00 sin las horas 1 a 4: 2 (hora 0) + 8 (5:00 a 8:30) + 1 (9:00)
    assert len(cambios) == 2 * 11
    assert not cambios["bin"].dt.hour.isin(cfg.P2_HORAS_SIN_SERVICIO).any()
    assert cambios["salidas"].sum() == cambios["entradas"].sum() == len(viajes)
    fila = cambios.set_index(["estacion", "bin"])
    assert fila.loc[(1, pd.Timestamp("2024-01-08 08:00")), "salidas"] == 1
    assert fila.loc[(1, pd.Timestamp("2024-01-08 09:00")), "entradas"] == 1
    assert fila.loc[(1, pd.Timestamp("2024-01-08 06:00"))].tolist() == [0, 0]


def test_p2_suma_acumulada_reinicia_cada_dia_de_servicio():
    cambios = pd.DataFrame(
        {
            "estacion": [1, 1, 1, 2, 2],
            "bin": pd.to_datetime(
                [
                    "2024-01-08 23:30",
                    "2024-01-09 00:30",  # todavía es el día de servicio del 8
                    "2024-01-09 05:00",  # empieza el día del 9
                    "2024-01-09 05:00",
                    "2024-01-09 05:30",
                ]
            ),
            "salidas": [1, 0, 0, 0, 0],
            "entradas": [0, 2, 1, 0, 0],
        }
    )
    resultado = features.p2_agregar_suma_acumulada_a_cambios(cambios)
    # La estación 2 no tuvo viajes ese día: sus filas se quitan
    assert resultado["estacion"].tolist() == [1, 1, 1]
    assert resultado["dia"].dt.day.tolist() == [8, 8, 9]
    assert resultado["neto"].tolist() == [-1, 2, 1]
    assert resultado["cumsum_dia"].tolist() == [-1, 1, 1]


def test_p2_mayores_cambios_futuros_relativos_a_la_capacidad():
    cambios = pd.DataFrame(
        {
            "estacion": [1] * 5,
            "dia": pd.Timestamp("2024-01-08"),
            "cumsum_dia": [0, -3, -1, 2, 2],
            "dpcapacity": 10,
        }
    )
    vacia, llena = features.p2_obtener_mayores_cambios_en_proximos_h_bins(cambios, 2)
    np.testing.assert_allclose(vacia[:3], [0.3, 0.0, 0.0])
    np.testing.assert_allclose(llena[:3], [0.0, 0.5, 0.3])
    # La ventana de las dos últimas filas se sale del día
    assert vacia[3:].isna().all() and llena[3:].isna().all()


def test_p2_ventana_futura_no_cruza_al_dia_siguiente():
    cambios = pd.DataFrame(
        {
            "estacion": [1, 1, 1],
            "dia": pd.to_datetime(["2024-01-08", "2024-01-08", "2024-01-09"]),
            "cumsum_dia": [0, 0, -10],  # el día siguiente se vacía, pero no cuenta
            "dpcapacity": 10,
        }
    )
    vacia, _ = features.p2_obtener_mayores_cambios_en_proximos_h_bins(cambios, 1)
    assert vacia[0] == 0.0
    assert np.isnan(vacia[1]) and np.isnan(vacia[2])


def test_p2_clase_de_riesgo_con_alfa_exacto_y_ambos_eventos():
    cambios = pd.DataFrame(
        {
            "vacia": [0.1, 0.5, 0.2, 0.6, 0.7, 0.5, 0.49, np.nan],
            "llena": [0.1, 0.0, 0.5, 0.7, 0.6, 0.5, 0.49, 0.2],
        }
    )
    resultado = features.p2_agregar_clase_de_riesgo(cambios, alfa=0.5)
    # alfa cuenta como riesgo; con ambos eventos gana el mayor y el empate va a sin bicis
    assert resultado["clase"].tolist() == [0, 1, 2, 2, 1, 1, 0]
    # La fila sin ventana completa se quita y la tabla original no cambia
    assert len(resultado) == 7
    assert "clase" not in cambios.columns


def test_p2_ventana_suma_los_ultimos_k_bins_del_dia():
    cambios = pd.DataFrame(
        {
            "estacion": [1] * 5,
            "dia": pd.to_datetime(["2024-01-08"] * 3 + ["2024-01-09"] * 2),
            "neto": [1, 2, 3, 4, 5],
        }
    )
    assert features.p2_ventana(cambios, "neto", 2).tolist() == [1, 3, 5, 4, 9]
    assert features.p2_ventana(cambios, "neto", 1).tolist() == [1, 2, 3, 4, 5]


def test_p2_etiqueta_la_ventana_antes_de_vaciarse():
    tabla = features.p2_construir_caracteristicas(
        p2_viajes_de_una_manana(), p2_estaciones(), bin_min=30, h_bins=4, alfa=0.5
    )
    assert list(tabla.columns) == cfg.P2_FEATURES_SIN_HISTORICO + [
        "clase",
        "estacion",
        "dia",
        "hora_idx",
    ]
    assert not tabla[cfg.P2_FEATURES_SIN_HISTORICO].isna().any().any()
    # Bins de 5:00 a 8:00; solo 5:00, 5:30 y 6:00 tienen las 4 ventanas futuras dentro del día
    assert sorted(tabla["hora_idx"].unique()) == [10, 11, 12]
    a_las_6 = tabla[tabla["hora_idx"] == 12].set_index("estacion")["clase"]
    # La estación 1 pierde sus 4 bicis a las 8:00 y la 2 llena sus 4 anclajes
    assert a_las_6.to_dict() == {1: 1, 2: 2}
    assert (tabla.loc[tabla["hora_idx"] < 12, "clase"] == 0).all()


def test_p2_construir_caracteristicas_falla_si_se_pierden_viajes():
    viajes = p2_viajes_de_una_manana()
    viajes.loc[0, "destino_id"] = np.nan  # un viaje sin destino no cuenta como entrada
    with pytest.raises(ValueError, match="no coinciden"):
        features.p2_construir_caracteristicas(viajes, p2_estaciones())