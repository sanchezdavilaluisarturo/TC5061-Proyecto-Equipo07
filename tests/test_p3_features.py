"""Features de P3: perfiles solo con su periodo, umbrales de salidas y horas sin servicio."""

import pandas as pd
import pytest
from test_features import p3_viajes_laborales
from test_modeling import p3_limpios_sinteticos

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import features

LABORALES = [f"lab_h{h:02d}" for h in cfg.P3_HORAS_SERVICIO]
FINES = [f"fin_h{h:02d}" for h in cfg.P3_HORAS_SERVICIO]


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
    perfil = features.p3_perfil_de_estaciones(p3_viajes_laborales())  # la estación 2 solo sale el lunes
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
