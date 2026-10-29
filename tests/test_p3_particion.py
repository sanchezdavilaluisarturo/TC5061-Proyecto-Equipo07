"""Partición temporal de P3: sin viajes sin conjunto, orden, separación mínima y casos límite."""

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset

CATEGORIAS = ["train", "valid", "test", "brecha"]
BRECHA = pd.Timedelta(hours=cfg.P3_BRECHA_HORAS)
FRONTERA_VALID = pd.Timestamp(cfg.P3_FRONTERA_VALID)
FRONTERA_TEST = pd.Timestamp(cfg.P3_FRONTERA_TEST)


def instantes_al_azar(n: int = 5000) -> pd.Series:
    """Viajes repartidos entre 2023 y 2026, que abarcan las tres fronteras."""
    rng = np.random.default_rng(cfg.SEMILLA)
    minutos = rng.integers(0, 3 * 365 * 24 * 60, n)
    return pd.Series(pd.Timestamp("2023-01-01") + pd.to_timedelta(minutos, unit="min"))


@pytest.fixture(scope="module")
def inicio() -> pd.Series:
    return instantes_al_azar()


@pytest.fixture(scope="module")
def particion(inicio) -> pd.Series:
    return dataset.p3_asignar_particion(inicio)


def test_p3_ningun_viaje_queda_sin_conjunto(particion, inicio):
    assert len(particion) == len(inicio)
    assert particion.notna().all()
    assert set(particion.astype(str)) == set(CATEGORIAS)  # los cuatro conjuntos aparecen


def test_p3_los_conjuntos_van_en_orden_y_no_se_superponen(particion, inicio):
    por_conjunto = inicio.groupby(particion, observed=True).agg(["min", "max"])
    assert por_conjunto.loc["train", "max"] < por_conjunto.loc["valid", "min"]
    assert por_conjunto.loc["valid", "max"] < por_conjunto.loc["test", "min"]


def test_p3_entre_conjuntos_hay_al_menos_la_brecha(particion, inicio):
    por_conjunto = inicio.groupby(particion, observed=True).agg(["min", "max"])
    assert por_conjunto.loc["valid", "min"] - por_conjunto.loc["train", "max"] >= BRECHA
    assert por_conjunto.loc["test", "min"] - por_conjunto.loc["valid", "max"] >= BRECHA


def test_p3_la_brecha_son_solo_las_horas_previas_a_cada_frontera(particion, inicio):
    previas = ((inicio >= FRONTERA_VALID - BRECHA) & (inicio < FRONTERA_VALID)) | (
        (inicio >= FRONTERA_TEST - BRECHA) & (inicio < FRONTERA_TEST)
    )
    assert (particion == "brecha").equals(previas)


@pytest.mark.parametrize(
    ("frontera", "antes", "justo_antes_de_la_brecha"),
    [(FRONTERA_VALID, "train", "valid"), (FRONTERA_TEST, "valid", "test")],
)
def test_p3_las_fronteras_se_deciden_al_nanosegundo(frontera, antes, justo_antes_de_la_brecha):
    un_ns = pd.Timedelta(1, "ns")
    instantes = pd.Series(
        [frontera - BRECHA - un_ns, frontera - BRECHA, frontera - un_ns, frontera]
    )
    resultado = dataset.p3_asignar_particion(instantes).tolist()
    assert resultado == [antes, "brecha", "brecha", justo_antes_de_la_brecha]


def test_p3_conserva_el_indice_de_la_serie():
    inicio = pd.Series(
        pd.to_datetime(["2024-01-01", "2024-10-15", "2025-12-01"]), index=[10, 20, 30]
    )
    particion = dataset.p3_asignar_particion(inicio)
    assert particion.index.tolist() == [10, 20, 30]
    assert particion.tolist() == ["train", "valid", "test"]


def test_p3_serie_vacia_regresa_serie_vacia_con_las_categorias():
    vacia = pd.Series([], dtype="datetime64[ns]")
    particion = dataset.p3_asignar_particion(vacia)
    assert len(particion) == 0
    assert list(particion.cat.categories) == CATEGORIAS


def test_p3_sin_brecha_todo_viaje_cae_en_train_valid_o_test(inicio, monkeypatch):
    monkeypatch.setattr(cfg, "P3_BRECHA_HORAS", 0)
    particion = dataset.p3_asignar_particion(inicio)
    assert not (particion == "brecha").any()
    assert (particion[inicio < FRONTERA_VALID] == "train").all()


def test_p3_usa_las_fronteras_de_la_configuracion(monkeypatch):
    monkeypatch.setattr(cfg, "P3_FRONTERA_VALID", "2024-01-01")
    monkeypatch.setattr(cfg, "P3_FRONTERA_TEST", "2024-06-01")
    inicio = pd.Series(pd.to_datetime(["2023-12-01", "2024-03-01", "2024-07-01"]))
    assert dataset.p3_asignar_particion(inicio).tolist() == ["train", "valid", "test"]
