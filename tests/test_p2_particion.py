"""Pruebas de partición en fit, valid y test para P2, incluyendo el cálculo de features históricos hechos con dichas particiones."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import features


def test_p2_particion_en_las_fronteras_exactas():
    valid, test = pd.Timestamp(cfg.FRONTERA_VALID), pd.Timestamp(cfg.FRONTERA_TEST)
    dia = pd.Timedelta(days=1)
    dias = pd.Series([valid - 2 * dia, valid - dia, valid, test - 2 * dia, test - dia, test])
    particion = features.p2_asignar_particion(dias)
    assert particion.tolist() == ["train", "brecha", "valid", "valid", "brecha", "test"]
    assert list(particion.cat.categories) == ["train", "valid", "test", "brecha"]


def p2_historia_tabla(clases: list[int]) -> pd.DataFrame:
    """Misma estación, tipo de día y hora en días consecutivos, más una fila de otra hora."""
    n = len(clases)
    return pd.DataFrame(
        {
            "estacion": [1] * (n + 1),
            "es_fin_semana": [0] * (n + 1),
            "hora_idx": [16] * n + [17],
            "dia": list(pd.date_range("2024-01-08", periods=n)) + [pd.Timestamp("2024-01-08")],
            "clase": clases + [2],
        }
    )


def test_p2_historia_suaviza_hacia_la_frecuencia_global():
    tabla = p2_historia_tabla([1, 0, 1])
    historia = features.P2HistoriaHastaAyer(peso_historico=2).fit(tabla.iloc[:3])
    assert historia.frec_global_ == pytest.approx({"sin_bicis": 2 / 3, "sin_anclajes": 0.0})
    resultado = historia.transform(tabla).sort_values(["hora_idx", "dia"])
    # Día 1 sin historia = global; día 2 vio 1 de 1; día 3 vio 1 de 2 (no su propia clase)
    esperado = [2 / 3, (1 + 4 / 3) / 3, (1 + 4 / 3) / 4]
    np.testing.assert_allclose(resultado["hist_sin_bicis"][:3], esperado)


def test_p2_historia_no_usa_la_etiqueta_del_dia_actual_ni_del_futuro():
    historia = features.P2HistoriaHastaAyer(peso_historico=2).fit(p2_historia_tabla([1, 0, 1]))
    original = historia.transform(p2_historia_tabla([1, 0, 1])).sort_values(["hora_idx", "dia"])
    cambiada = historia.transform(p2_historia_tabla([1, 0, 0])).sort_values(["hora_idx", "dia"])
    columnas = ["hist_sin_bicis", "hist_sin_anclajes"]
    pd.testing.assert_frame_equal(original[columnas], cambiada[columnas])


def test_p2_historia_separa_por_hora():
    historia = features.P2HistoriaHastaAyer(peso_historico=2).fit(p2_historia_tabla([2, 2]))
    resultado = historia.transform(p2_historia_tabla([2, 2]))
    otra_hora = resultado[resultado["hora_idx"] == 17]
    # La fila de las 17 no tiene días previos: solo la frecuencia global
    assert otra_hora["hist_sin_anclajes"].iloc[0] == pytest.approx(1.0)
    assert otra_hora["hist_sin_bicis"].iloc[0] == pytest.approx(0.0)


def test_p2_historia_sin_fit_falla():
    with pytest.raises(NotFittedError):
        features.P2HistoriaHastaAyer().transform(p2_historia_tabla([0]))


def test_p2_preparar_conjuntos_toma_la_frecuencia_global_solo_de_train():
    valid = pd.Timestamp(cfg.FRONTERA_VALID)
    dias = [valid - pd.Timedelta(days=d) for d in (3, 2)] + [valid, valid + pd.Timedelta(days=1)]
    cambios = pd.DataFrame(
        {
            "estacion": 1,
            "es_fin_semana": 0,
            "hora_idx": 16,
            "dia": dias,
            "clase": [1, 1, 2, 2],  # train solo tiene sin bicis
        }
    )
    # p2_preparar_conjuntos pide el día más reciente de test; sin filas de test no hay fecha
    cambios = pd.concat(
        [cambios, cambios.iloc[[3]].assign(dia=pd.Timestamp(cfg.FRONTERA_TEST))],
        ignore_index=True,
    )
    fit, val, prueba = features.p2_preparar_conjuntos(cambios, peso_historico=2)
    assert (len(fit), len(val), len(prueba)) == (2, 2, 1)
    # Primer día de train: sin historia, toda la frecuencia global sale de train (1 y 0)
    primero = fit.sort_values("dia").iloc[0]
    assert primero["hist_sin_bicis"] == pytest.approx(1.0)
    assert primero["hist_sin_anclajes"] == pytest.approx(0.0)
