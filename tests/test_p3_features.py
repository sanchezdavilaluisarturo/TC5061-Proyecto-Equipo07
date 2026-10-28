"""Pruebas de las funciones p3_* de features.py.

Primero el esquema y las tablas de P3 (variables de tiempo, perfil, estaciones, demanda y viajes
por mes) y después el perfil de estaciones en sus casos límite: periodos sin un tipo de día, horas
sin servicio, umbrales de salidas y que train no vea lo que pasa en valid, test o brecha.
"""

import numpy as np
import pandas as pd
import pytest
from test_p3_modeling import p3_limpios_sinteticos

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import features

LABORALES = [f"lab_h{h:02d}" for h in cfg.P3_HORAS_SERVICIO]
FINES = [f"fin_h{h:02d}" for h in cfg.P3_HORAS_SERVICIO]


def p3_viajes_laborales() -> pd.DataFrame:
    """Estación 1: 3 salidas laborales (8:00, 8:30, 18:00) y 1 de sábado; estación 2: 1 salida."""
    inicio = pd.to_datetime(
        [
            "2024-01-08 08:00",  # lunes
            "2024-01-09 08:30",  # martes
            "2024-01-10 18:00",  # miércoles
            "2024-01-13 11:00",  # sábado
            "2024-01-08 09:00",  # lunes, estación 2
        ]
    )
    return pd.DataFrame(
        {
            "inicio_del_viaje": inicio,
            "fin_del_viaje": inicio + pd.Timedelta(minutes=10),
            "origen_id": [1, 1, 1, 1, 2],
            "destino_id": [2, 1, 2, 2, 1],
            "duracion_min": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )


def p3_limpios_con_particion() -> pd.DataFrame:
    viajes = p3_viajes_laborales()
    viajes["particion"] = pd.Categorical(
        ["train"] * 5, categories=["train", "valid", "test", "brecha"]
    )
    viajes["periodo_archivo"] = "2024-01"
    return viajes


def viajes(*filas: tuple[str, int, int]) -> pd.DataFrame:
    """Viajes de 10 min a partir de (inicio, origen, destino)."""
    inicio = pd.to_datetime([fila[0] for fila in filas])
    return pd.DataFrame(
        {
            "inicio_del_viaje": inicio,
            "fin_del_viaje": inicio + pd.Timedelta(minutes=10),
            "origen_id": [fila[1] for fila in filas],
            "destino_id": [fila[2] for fila in filas],
            "duracion_min": 10.0,
        }
    )


# --- Esquema y tablas de P3 ---------------------------------------------------------------------


def test_p3_variables_de_tiempo_agrega_columnas_en_su_lugar():
    viajes = p3_viajes_laborales().drop(columns="duracion_min")
    devuelto = features.p3_variables_de_tiempo(viajes)
    assert devuelto is viajes
    assert viajes["duracion_min"].tolist() == [10.0] * 5
    assert viajes["hora"].iloc[0] == 8 and viajes["dia_semana"].iloc[3] == 5  # 13-ene es sábado


def test_p3_perfil_de_estaciones_tiene_el_esquema_esperado():
    perfil = features.p3_perfil_de_estaciones(p3_viajes_laborales())
    n_horas = len(cfg.P3_HORAS_SERVICIO)
    laborales = [c for c in perfil.columns if c.startswith("lab_")]
    fines = [c for c in perfil.columns if c.startswith("fin_")]
    assert len(laborales) == n_horas and len(fines) == n_horas
    assert {"salidas", "log_viajes_por_dia", "neto_sobre_salidas"} <= set(perfil.columns)
    assert set(perfil.index) == {1, 2}

    est1 = perfil.loc[1]
    assert est1["salidas"] == 4
    # Las proporciones de cada bloque suman 1 (hay salidas laborales y de fin de semana)
    assert est1[laborales].sum() == pytest.approx(1.0)
    assert est1[fines].sum() == pytest.approx(1.0)
    assert est1["lab_h08"] == pytest.approx(2 / 3)
    assert est1["lab_h18"] == pytest.approx(1 / 3)
    # Estación 1: 4 salidas y 1 llegada propia (viaje circular) + 1 desde la estación 2
    assert est1["neto_sobre_salidas"] == pytest.approx((2 - 4) / 4)
    assert est1["prop_circulares"] == pytest.approx(1 / 4)


def test_p3_tabla_estaciones_cuenta_solo_entrenamiento():
    estaciones = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "latitude": [20.0, 20.1, 20.2],
            "longitude": [-103.0, -103.1, -103.2],
            "dpcapacity": [15, 0, 20],
        }
    )
    tabla = features.p3_tabla_estaciones(p3_limpios_con_particion(), estaciones)
    assert tabla.loc[1, "salidas_train"] == 4 and tabla.loc[2, "salidas_train"] == 1
    assert bool(tabla.loc[3, "activa_en_train"]) is False
    assert bool(tabla.loc[2, "capacidad_cero"]) is True
    assert tabla.loc[1, "neto_train"] == tabla.loc[1, "llegadas_train"] - 4


def test_p3_demanda_estacion_hora_excluye_horas_sin_servicio_y_no_inventa_ceros():
    limpios = pd.DataFrame(
        {
            "inicio_del_viaje": pd.to_datetime(
                [
                    "2024-01-10 08:30",
                    "2024-01-10 08:50",
                    "2024-01-10 09:10",
                    "2024-01-20 03:15",  # hora sin servicio: queda fuera de la rejilla
                    "2024-01-20 10:00",
                ]
            ),
            "origen_id": pd.array([7, 7, 7, 9, 9], dtype="Int32"),
            "periodo_archivo": pd.Categorical(["2024-01"] * 5),
        }
    )
    tabla, excluidos = features.p3_demanda_estacion_hora(limpios, "2024-01", "2024-01")
    assert excluidos == 1
    assert not tabla["hora_inicio"].dt.hour.isin(cfg.HORAS_SIN_SERVICIO).any()
    assert tabla["viajes"].sum() == 4
    # Los ceros empiezan en la primera hora activa de cada estación, no antes
    primera_7 = tabla.loc[tabla["origen_id"] == 7, "hora_inicio"].min()
    assert primera_7 == pd.Timestamp("2024-01-10 08:00")
    assert (
        tabla.loc[(tabla["origen_id"] == 7) & (tabla["hora_inicio"] == primera_7), "viajes"].item()
        == 2
    )
    assert set(tabla["particion"].unique()) == {"train"}


def test_p3_viajes_por_mes_y_conjunto():
    limpios = p3_limpios_con_particion()
    tabla = features.p3_viajes_por_mes_y_conjunto(limpios)
    assert list(tabla.columns) == ["train", "brecha", "valid", "test"]
    assert tabla.loc["2024-01", "train"] == 5
    assert np.all(tabla[["brecha", "valid", "test"]].to_numpy() == 0)


# --- Perfil de estaciones: casos límite ---------------------------------------------------------


def test_p3_periodo_sin_fin_de_semana_deja_en_cero_todo_el_perfil_de_fin_de_semana():
    """Regresión: antes lanzaba KeyError si el periodo no tenía ningún viaje de ese tipo de día."""
    perfil = features.p3_perfil_de_estaciones(p3_viajes_laborales().query("origen_id == 2"))
    assert (perfil[FINES] == 0).all(axis=None)
    assert perfil.loc[2, "lab_h09"] == pytest.approx(1.0)


def test_p3_periodo_sin_dias_laborales_deja_en_cero_todo_el_perfil_laboral():
    perfil = features.p3_perfil_de_estaciones(viajes(("2024-01-13 11:00", 1, 2)))  # solo sábado
    assert (perfil[LABORALES] == 0).all(axis=None)
    assert perfil.loc[1, "fin_h11"] == pytest.approx(1.0)
    assert list(perfil.columns[: len(LABORALES) + len(FINES)]) == LABORALES + FINES


def test_p3_estacion_solo_de_fin_de_semana_deja_en_cero_el_perfil_laboral():
    perfil = features.p3_perfil_de_estaciones(
        viajes(("2024-01-13 11:00", 1, 2), ("2024-01-08 09:00", 2, 1))  # sábado y lunes
    )
    assert (perfil.loc[1, LABORALES] == 0).all()
    assert perfil.loc[1, "fin_h11"] == pytest.approx(1.0)
    assert perfil.loc[1, "salidas"] == 1


def test_p3_estacion_solo_laboral_deja_en_cero_el_perfil_de_fin_de_semana():
    perfil = features.p3_perfil_de_estaciones(p3_viajes_laborales())  # la 2 solo sale el lunes
    assert (perfil.loc[2, FINES] == 0).all()
    assert perfil.loc[2, "lab_h09"] == pytest.approx(1.0)


def test_p3_las_horas_sin_servicio_no_tienen_columna_y_no_entran_a_la_proporcion():
    perfil = features.p3_perfil_de_estaciones(
        viajes(
            ("2024-01-08 02:00", 1, 2),  # 02:00 sin servicio
            ("2024-01-08 08:00", 1, 2),
            ("2024-01-13 10:00", 2, 1),  # algún viaje de fin de semana en el periodo
        )
    )
    sin_servicio = [f"{prefijo}_h{h:02d}" for prefijo in ("lab", "fin") for h in (1, 2, 3, 4)]
    assert not set(sin_servicio) & set(perfil.columns)
    assert perfil.loc[1, "lab_h08"] == pytest.approx(1.0)  # proporción sobre horas de servicio
    assert perfil.loc[1, "salidas"] == 2  # pero la salida de las 02:00 sí cuenta como salida


def test_p3_el_perfil_solo_incluye_estaciones_con_salidas():
    perfil = features.p3_perfil_de_estaciones(
        viajes(("2024-01-08 08:00", 1, 99), ("2024-01-13 08:00", 1, 99))
    )
    assert perfil.index.tolist() == [1]  # la 99 solo recibe viajes


def test_p3_el_perfil_de_cada_bloque_suma_uno_en_todas_las_estaciones():
    train = features.p3_construir_caracteristicas(p3_limpios_sinteticos(), 1, 1)[0]
    assert train[LABORALES].sum(axis=1).tolist() == pytest.approx([1.0] * len(train))
    assert train[FINES].sum(axis=1).tolist() == pytest.approx([1.0] * len(train))


def test_p3_las_columnas_del_modelo_son_el_perfil_sin_el_conteo_de_salidas():
    _, _, columnas = features.p3_construir_caracteristicas(p3_limpios_sinteticos(), 1, 1)
    assert "salidas" not in columnas
    assert len(columnas) == 2 * len(cfg.P3_HORAS_SERVICIO) + 4  # 44 con 20 horas de servicio
    assert columnas[:1] == [LABORALES[0]]


def test_p3_el_umbral_de_salidas_incluye_a_la_estacion_que_lo_iguala():
    limpios = p3_limpios_sinteticos()
    de_train = limpios.loc[limpios["particion"] == "train", "origen_id"].value_counts()
    de_valid = limpios.loc[limpios["particion"] == "valid", "origen_id"].value_counts()
    umbral_train, umbral_valid = int(de_train.median()), int(de_valid.median())

    train, valid, _ = features.p3_construir_caracteristicas(limpios, umbral_train, umbral_valid)

    assert set(train.index) == set(de_train[de_train >= umbral_train].index)
    assert set(valid.index) == set(de_valid[de_valid >= umbral_valid].index)
    assert (train["salidas"] >= umbral_train).all()
    assert (train["salidas"] == umbral_train).any()  # hay una justo en el límite y se queda


def test_p3_un_umbral_mayor_que_cualquier_estacion_deja_los_perfiles_vacios():
    limpios = p3_limpios_sinteticos()
    train, valid, columnas = features.p3_construir_caracteristicas(limpios, 10**9, 10**9)
    assert train.empty and valid.empty
    assert len(columnas) == 44  # las columnas no dependen de que haya estaciones


def test_p3_el_perfil_de_train_no_cambia_con_lo_que_pase_en_valid_test_o_brecha():
    limpios = p3_limpios_sinteticos()
    esperado = features.p3_construir_caracteristicas(limpios, 50, 20)[0]

    alterado = limpios.copy()
    fuera_de_train = alterado["particion"] != "train"
    alterado.loc[fuera_de_train, "origen_id"] = 1  # todo valid se vuelve de la estación 1
    alterado.loc[fuera_de_train, "duracion_min"] = 999.0
    train = features.p3_construir_caracteristicas(alterado, 50, 20)[0]

    pd.testing.assert_frame_equal(train, esperado)


def test_p3_la_duracion_mediana_ignora_los_viajes_atipicos():
    perfil = features.p3_perfil_de_estaciones(
        viajes(*[("2024-01-08 08:00", 1, 2)] * 4, ("2024-01-13 08:00", 1, 2)).assign(
            duracion_min=[5, 6, 7, 8, 1000.0]
        )
    )
    assert perfil.loc[1, "duracion_mediana_min"] == 7.0
