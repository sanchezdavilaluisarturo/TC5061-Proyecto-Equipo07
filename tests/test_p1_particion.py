"""Pruebas de la partición temporal de P1: fronteras, brecha de 24 h y orden de los conjuntos."""

import pandas as pd
import pytest

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import features

FRONTERA_VALID = pd.Timestamp(cfg.FRONTERA_VALID)
FRONTERA_TEST = pd.Timestamp(cfg.FRONTERA_TEST)
BRECHA = pd.Timedelta(hours=cfg.BRECHA_HORAS)
UNA_HORA = pd.Timedelta(hours=1)


def conjunto_de(hora: pd.Timestamp) -> str:
    return features.p1_asignar_particion(pd.Series([hora]))[0]


@pytest.mark.parametrize(
    ("hora", "esperado"),
    [
        (FRONTERA_VALID - BRECHA - UNA_HORA, "train"),  # última hora de train
        (FRONTERA_VALID - BRECHA, "brecha"),  # primera hora de la brecha
        (FRONTERA_VALID - UNA_HORA, "brecha"),  # última hora de la brecha
        (FRONTERA_VALID, "valid"),  # la frontera ya es del conjunto siguiente
        (FRONTERA_TEST - BRECHA - UNA_HORA, "valid"),
        (FRONTERA_TEST - BRECHA, "brecha"),
        (FRONTERA_TEST - UNA_HORA, "brecha"),
        (FRONTERA_TEST, "test"),
    ],
)
def test_p1_cada_hora_cae_en_su_conjunto_en_las_fronteras(hora, esperado):
    assert conjunto_de(hora) == esperado


@pytest.fixture
def particion_de_las_horas() -> pd.DataFrame:
    """Todas las horas de servicio de los meses alrededor de las dos fronteras."""
    horas = features.p1_horas_de_servicio("2024-09", "2025-10").to_series(index=None)
    return pd.DataFrame({"hora": horas, "conjunto": features.p1_asignar_particion(horas)})


def test_p1_los_conjuntos_no_se_traslapan_y_respetan_el_orden(particion_de_las_horas):
    rango = particion_de_las_horas.groupby("conjunto", observed=True)["hora"].agg(["min", "max"])
    assert set(rango.index) == {"train", "valid", "test", "brecha"}
    assert rango.loc["train", "max"] < rango.loc["valid", "min"] < rango.loc["test", "min"]
    assert rango.loc["valid", "max"] < rango.loc["test", "min"]


def test_p1_entre_conjuntos_quedan_al_menos_24_horas(particion_de_las_horas):
    """Con horizonte de 24 h, la etiqueta de la última hora de un conjunto no cruza al siguiente."""
    rango = particion_de_las_horas.groupby("conjunto", observed=True)["hora"].agg(["min", "max"])
    assert rango.loc["valid", "min"] - rango.loc["train", "max"] > BRECHA
    assert rango.loc["test", "min"] - rango.loc["valid", "max"] > BRECHA
