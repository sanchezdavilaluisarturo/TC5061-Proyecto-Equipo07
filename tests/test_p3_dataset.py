"""Pruebas de las funciones p3_* de dataset.py (carga, partición y validación)."""

import pandas as pd
import pytest

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset


def p3_viajes_de_juguete() -> pd.DataFrame:
    """Cuatro viajes limpios de una sola estación, repartidos en train, valid y test."""
    return pd.DataFrame(
        {
            "viaje_id": [1, 2, 3, 4],
            "usuario_id": [10, 11, 12, 13],
            "genero": ["M", "F", "M", None],
            "anio_de_nacimiento": [1990, 1985, None, 2000],
            "inicio_del_viaje": pd.to_datetime(
                ["2024-01-10 08:00", "2024-11-10 08:00", "2026-01-10 08:00", "2024-02-01 09:00"]
            ),
            "fin_del_viaje": pd.to_datetime(
                ["2024-01-10 08:10", "2024-11-10 08:20", "2026-01-10 08:30", "2024-02-01 09:05"]
            ),
            "origen_id": [1, 1, 1, 2],
            "destino_id": [2, 2, 2, 1],
            "periodo_archivo": ["2024-01", "2024-11", "2026-01", "2024-02"],
        }
    )


def test_p3_asignar_particion_en_las_fronteras():
    tiempos = pd.Series(
        pd.to_datetime(
            [
                "2024-09-29 23:59:00",  # último instante de train
                "2024-09-30 00:00:00",  # empieza la brecha de 24 h antes de valid
                "2024-10-01 00:00:00",  # primer instante de valid
                "2025-09-29 23:59:00",  # último instante de valid
                "2025-09-30 00:00:00",  # brecha antes de test
                "2025-10-01 00:00:00",  # primer instante de test
            ]
        )
    )
    particion = dataset.p3_asignar_particion(tiempos)
    assert particion.tolist() == ["train", "brecha", "valid", "valid", "brecha", "test"]
    assert list(particion.cat.categories) == ["train", "valid", "test", "brecha"]


def test_p3_meses_sin_datos_se_calcula_de_los_datos():
    presentes = ["2023-03", "2023-06"]
    assert dataset.p3_meses_sin_datos(presentes, "2023-03", "2023-06") == ["2023-04", "2023-05"]
    assert dataset.p3_meses_sin_datos(["2023-03", "2023-04"], "2023-03", "2023-04") == []


def test_p3_preparar_modelado_no_incluye_datos_personales():
    modelado = dataset.p3_preparar_modelado(p3_viajes_de_juguete())
    assert list(modelado.columns) == [*cfg.P3_COLUMNAS_MODELADO, "particion"]
    for personal in ("usuario_id", "genero", "anio_de_nacimiento"):
        assert personal not in modelado.columns
    assert modelado["duracion_min"].tolist() == pytest.approx([10, 20, 30, 5])
    assert modelado["particion"].tolist() == ["train", "valid", "test", "train"]


def p3_estaciones_de_juguete() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2], "name": ["A", "B"], "dpcapacity": [15, 15]})


def test_p3_validar_modelado_acepta_un_dataset_correcto():
    modelado = dataset.p3_preparar_modelado(p3_viajes_de_juguete())
    dataset.p3_validar_modelado(modelado, p3_estaciones_de_juguete())


@pytest.mark.parametrize(
    "cambio, mensaje",
    [
        (lambda m: m.assign(viaje_id=[1, 1, 3, 4]), "viaje_id repetidos"),
        (lambda m: m.assign(origen_id=[1, 1, 1, 99]), "sin nomenclatura"),
        (lambda m: m.assign(duracion_min=[10.0, 0.0, 30.0, 5.0]), "duraciones"),
        (lambda m: m.assign(duracion_min=[10.0, 25 * 60.0, 30.0, 5.0]), "duraciones"),
    ],
)
def test_p3_validar_modelado_detecta_reglas_incumplidas(cambio, mensaje):
    modelado = cambio(dataset.p3_preparar_modelado(p3_viajes_de_juguete()))
    with pytest.raises(ValueError, match=mensaje):
        dataset.p3_validar_modelado(modelado, p3_estaciones_de_juguete())


def test_p3_cargar_viajes_se_detiene_si_no_coincide_con_el_manifiesto(tmp_path, monkeypatch):
    interim = tmp_path / "interim"
    interim.mkdir()
    viajes = p3_viajes_de_juguete()
    viajes.to_parquet(interim / "viajes_2024-01_2024-12.parquet", index=False)
    manifiesto = pd.DataFrame(
        {
            "archivo": ["datos_abiertos_2024_01.csv"],
            "periodo": ["2024-01"],
            "url": ["x"],
            "registros": [len(viajes) + 5],  # el manifiesto indica más registros de los que hay
            "sha256": ["x"],
        }
    )
    manifiesto.to_csv(tmp_path / "fuentes.csv", index=False)
    monkeypatch.setattr(dataset, "INTERIM_DATA_DIR", interim)
    monkeypatch.setattr(dataset, "RUTA_MANIFIESTO", tmp_path / "fuentes.csv")

    with pytest.raises(ValueError, match="manifiesto"):
        dataset.p3_cargar_viajes("2024-01", "2024-12")
    # Sin validar contra el manifiesto, la lectura sí funciona
    assert len(dataset.p3_cargar_viajes("2024-01", "2024-12", validar_manifiesto=False)) == 4


def test_p3_cargar_viajes_avisa_si_falta_el_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "INTERIM_DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="make data"):
        dataset.p3_cargar_viajes("2020-01", "2020-02")
