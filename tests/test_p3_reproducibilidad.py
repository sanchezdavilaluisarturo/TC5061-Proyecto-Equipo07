"""Reproducibilidad de P3: misma semilla, mismos grupos y métricas; otra semilla, otro resultado.

Se usan estaciones sin estructura de grupos (horas al azar). Con grupos bien separados, como los
de `p3_limpios_sinteticos`, K-Means converge al mismo resultado con cualquier semilla y la prueba
de contraste no demostraría nada; con ruido, la semilla sí decide la inercia y las etiquetas.
"""

import numpy as np
import optuna
import pandas as pd
import pytest
from test_modeling import p3_limpios_sinteticos

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.modeling import train, tuning_p3

KW = {"k_minimo": 2, "k_maximo": 4, "min_salidas_train": 50, "min_salidas_valid": 20}
SEMILLA = 42
OTRA_SEMILLA = 7


def limpios_con_ruido() -> pd.DataFrame:
    """Los mismos viajes sintéticos, pero con la estación y la hora de cada viaje al azar."""
    limpios = p3_limpios_sinteticos()
    rng = np.random.default_rng(0)
    limpios["origen_id"] = rng.integers(1, 31, len(limpios))
    limpios["inicio_del_viaje"] = limpios["inicio_del_viaje"].dt.floor("D") + pd.to_timedelta(
        rng.integers(6, 22, len(limpios)), unit="h"
    )
    return limpios


@pytest.fixture(scope="module")
def limpios() -> pd.DataFrame:
    return limpios_con_ruido()


@pytest.fixture(scope="module")
def base(limpios) -> train.P3Baseline:
    return train.p3_ajustar_baseline(limpios, SEMILLA, **KW)


def grupos(base: train.P3Baseline) -> np.ndarray:
    return base.caracteristicas_train["grupo"].to_numpy()


def test_p3_baseline_con_la_misma_semilla_da_los_mismos_grupos_y_metricas(limpios, base):
    de_nuevo = train.p3_ajustar_baseline(limpios, SEMILLA, **KW)
    assert de_nuevo.k_elegido == base.k_elegido
    np.testing.assert_array_equal(grupos(de_nuevo), grupos(base))
    pd.testing.assert_frame_equal(de_nuevo.resultados_k, base.resultados_k)
    # Tolerancia: los hilos de BLAS reparten las sumas distinto y el último decimal puede variar
    assert de_nuevo.metricas == pytest.approx(base.metricas, rel=1e-9)


def test_p3_baseline_con_otra_semilla_cambia_la_inercia_y_los_grupos(limpios, base):
    otra = train.p3_ajustar_baseline(limpios, OTRA_SEMILLA, **KW)
    assert otra.semilla == OTRA_SEMILLA != base.semilla
    # Si la semilla no controlara el azar, el contraste no distinguiría una corrida de otra
    assert otra.metricas["inercia"] != pytest.approx(base.metricas["inercia"], rel=1e-6)
    assert not np.array_equal(grupos(otra), grupos(base))


def test_p3_alternativo_con_la_misma_semilla_elige_y_asigna_igual(base):
    kwargs = {"componentes": [2, 3], "n_semillas": 2, "tam_min_aceptable": 5}
    a = train.p3_entrenar_alternativo(base, **kwargs)
    b = train.p3_entrenar_alternativo(base, **kwargs)
    pd.testing.assert_frame_equal(a.busqueda, b.busqueda)
    pd.testing.assert_series_equal(a.elegido, b.elegido)
    np.testing.assert_array_equal(a.etiquetas, b.etiquetas)
    assert a.metricas_alt == pytest.approx(b.metricas_alt, rel=1e-9)
    assert a.aceptado == b.aceptado


def test_p3_alternativo_con_otra_semilla_evalua_otras_corridas(limpios, base):
    kwargs = {"componentes": [2, 3], "n_semillas": 2, "tam_min_aceptable": 5}
    otra = train.p3_ajustar_baseline(limpios, OTRA_SEMILLA, **KW)
    a = train.p3_entrenar_alternativo(base, **kwargs)
    b = train.p3_entrenar_alternativo(otra, **kwargs)
    # Las semillas de la rejilla salen de la del proyecto: otra base, otras semillas
    assert set(a.busqueda["semilla"]) != set(b.busqueda["semilla"])
    assert a.busqueda["silueta"].to_numpy() != pytest.approx(b.busqueda["silueta"].to_numpy())


def buscar(base: train.P3Baseline, tmp_path) -> optuna.Study:
    """Una búsqueda corta con Optuna dentro de un run padre, con MLflow en SQLite local."""
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("prueba")
    with mlflow.start_run():
        estudio, _ = tuning_p3.p3_buscar(
            base, "alternativo", n_trials=4, timeout_min=0, n_semillas=2, tam_min_aceptable=5
        )
    return estudio


def test_p3_busqueda_de_optuna_corrida_dos_veces_da_los_mismos_trials(base, tmp_path):
    a, b = buscar(base, tmp_path), buscar(base, tmp_path)
    assert [t.params for t in a.trials] == [t.params for t in b.trials]
    assert [t.value for t in a.trials] == pytest.approx([t.value for t in b.trials], rel=1e-9)
    assert a.best_trial.number == b.best_trial.number


def test_p3_busqueda_de_optuna_con_otra_semilla_prueba_otras_configuraciones(
    base, tmp_path, monkeypatch
):
    a = buscar(base, tmp_path)
    monkeypatch.setattr(cfg, "SEMILLA", OTRA_SEMILLA)  # la lee el sampler al crear el estudio
    b = buscar(base, tmp_path)
    assert [t.params for t in a.trials] != [t.params for t in b.trials]
