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
