"""Pruebas de los modelos de P2: baseline, construcción, umbrales, métricas y modelos."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
import typer
from xgboost import XGBClassifier

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.modeling import train

FILAS_POR_DIA = 20


def p2_tabla_sintetica(n: int = 600, semilla: int = 0) -> pd.DataFrame:
    """Variables de P2 con las tres clases: el flujo neto reciente anticipa el riesgo."""
    rng = np.random.default_rng(semilla)
    tabla = pd.DataFrame(
        rng.normal(size=(n, len(cfg.P2_FEATURES_CON_HISTORICO))),
        columns=cfg.P2_FEATURES_CON_HISTORICO,
    )
    ruido = rng.normal(scale=0.3, size=n)
    tabla["clase"] = np.select(
        [tabla["neto_1_relativo"] + ruido < -0.7, tabla["neto_1_relativo"] + ruido > 0.7],
        [1, 2],
        0,
    )
    # La historia "hasta ayer" va de 0 a 0.49 para que el baseline sume 1 sin negativos
    for clase, columna in [(1, "hist_sin_bicis"), (2, "hist_sin_anclajes")]:
        base = 0.1 + 0.25 * (tabla["clase"] == clase)
        tabla[columna] = np.clip(base + rng.normal(scale=0.1, size=n), 0, 0.49)
    tabla["dia"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(n) // 20, "D")
    return tabla


def p2_cambios_sinteticos(filas_por_conjunto: int = 300) -> pd.DataFrame:
    """Tabla con las columnas de `features p2-features` y 15 días en train, valid y test.

    Los días de train terminan antes de la brecha de FRONTERA_VALID, así que ninguna fila se
    descarta. La historia "hasta ayer" no viene en la tabla: la agrega p2_preparar_conjuntos.
    """
    inicios = [
        pd.Timestamp(cfg.FRONTERA_VALID) - pd.Timedelta(days=30),
        pd.Timestamp(cfg.FRONTERA_VALID),
        pd.Timestamp(cfg.FRONTERA_TEST),
    ]
    tabla = p2_tabla_sintetica(len(inicios) * filas_por_conjunto, semilla=1)
    i = np.arange(filas_por_conjunto)
    partes = []
    for k, inicio in enumerate(inicios):
        parte = tabla.iloc[k * filas_por_conjunto : (k + 1) * filas_por_conjunto].copy()
        parte["dia"] = inicio + pd.to_timedelta(i // FILAS_POR_DIA, "D")
        parte["estacion"] = 1 + (i % FILAS_POR_DIA) // 5
        parte["hora_idx"] = 10 + i % 5
        parte["es_fin_semana"] = (parte["dia"].dt.dayofweek >= 5).astype(int)
        partes.append(parte)
    columnas = cfg.P2_FEATURES_SIN_HISTORICO + ["clase", "estacion", "dia", "hora_idx"]
    return pd.concat(partes, ignore_index=True)[columnas]


@pytest.fixture
def figuras_temporales(tmp_path, monkeypatch):
    """Figuras en una carpeta temporal y sin ventana gráfica."""
    plt.switch_backend("Agg")
    monkeypatch.setattr(cfg, "FIGURES_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def cambios_en_disco(tmp_path, monkeypatch):
    """p2-entrenar lee la tabla sintética desde un parquet temporal y deja figuras en tmp_path."""
    plt.switch_backend("Agg")
    monkeypatch.setattr(cfg, "FIGURES_DIR", tmp_path)
    ruta = tmp_path / "p2_cambios_estacion_bin.parquet"
    p2_cambios_sinteticos().to_parquet(ruta, index=False)
    # El valor por defecto de `ruta` se fija al importar: se cambia la función, no cfg
    leer = train.p2_cargar_cambios
    monkeypatch.setattr(train, "p2_cargar_cambios", lambda: leer(ruta))
    return tmp_path


def p2_conjuntos_sinteticos(n: int = 900) -> dict[str, pd.DataFrame]:
    """fit, val y test en orden temporal (60/20/20)."""
    tabla = p2_tabla_sintetica(n)
    corte_val, corte_test = int(n * 0.6), int(n * 0.8)
    return {
        "fit": tabla.iloc[:corte_val].reset_index(drop=True),
        "val": tabla.iloc[corte_val:corte_test].reset_index(drop=True),
        "test": tabla.iloc[corte_test:].reset_index(drop=True),
    }


def test_p2_baseline_usa_la_historia_y_suma_uno():
    x = pd.DataFrame({"hist_sin_bicis": [0.1, 0.6, 0.0], "hist_sin_anclajes": [0.2, 0.1, 0.0]})
    modelo = train.P2BaselineHistorico().fit(x)
    p = modelo.predict_proba(x)
    np.testing.assert_allclose(p, [[0.7, 0.1, 0.2], [0.3, 0.6, 0.1], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert modelo.predict(x).tolist() == [0, 1, 0]
    assert modelo.classes_.tolist() == [0, 1, 2]


def test_p2_construir_modelo_aplica_config_semilla_y_min_samples_leaf():
    hgb = train.p2_construir_modelo("hgb", min_samples_leaf=50, semilla=7)
    assert isinstance(hgb, HistGradientBoostingClassifier)
    assert (hgb.min_samples_leaf, hgb.random_state) == (50, 7)
    assert hgb.max_iter == cfg.P2_PARAMETROS_HGB["max_iter"]

    xgb = train.p2_construir_modelo("xgb", semilla=7)
    assert isinstance(xgb, XGBClassifier)
    assert xgb.get_params()["random_state"] == 7
    assert xgb.get_params()["n_estimators"] == cfg.P2_PARAMETROS_XGB["n_estimators"]

    assert isinstance(train.p2_construir_modelo("baseline"), train.P2BaselineHistorico)


def test_p2_hiperparametros_de_la_busqueda_reemplazan_los_de_config():
    xgb = train.p2_construir_modelo("xgb_optuna", hiperparametros={"max_leaves": 7})
    assert xgb.get_params()["max_leaves"] == 7
    # Lo que la búsqueda no toca se queda como en config.py
    assert xgb.get_params()["learning_rate"] == cfg.P2_PARAMETROS_XGB["learning_rate"]
    hgb = train.p2_construir_modelo("hgb", min_samples_leaf=50, hiperparametros={"max_iter": 5})
    assert (hgb.max_iter, hgb.min_samples_leaf) == (5, 50)


def test_p2_candidato_de_optuna_exige_hiperparametros():
    with pytest.raises(ValueError, match="hiperparámetros"):
        train.p2_construir_modelo("xgb_optuna")


def test_p2_modelo_desconocido_falla():
    with pytest.raises(ValueError, match="desconocido"):
        train.p2_construir_modelo("rf")


def test_p2_parametros_del_baseline_son_sus_umbrales():
    resultado = {"umbrales": {1: 0.25, 2: 0.4}}
    parametros = train.p2_parametros_modelo("baseline", train.P2BaselineHistorico(), resultado)
    assert parametros == {
        "modelo": "baseline",
        "umbral_sin_bicis": 0.25,
        "umbral_sin_anclajes": 0.4,
    }


def test_p2_parametros_de_hgb_incluyen_config_min_samples_leaf_e_iteraciones():
    tabla = p2_tabla_sintetica()
    variables = cfg.P2_FEATURES_CON_HISTORICO
    modelo = train.p2_construir_modelo("hgb", min_samples_leaf=50, hiperparametros={"max_iter": 5})
    modelo.fit(tabla[variables], tabla["clase"])

    parametros = train.p2_parametros_modelo("hgb", modelo, {"umbrales": {1: 0.3, 2: 0.4}})
    assert {f"modelo_{nombre}" for nombre in cfg.P2_PARAMETROS_HGB} <= set(parametros)
    assert parametros["modelo_max_iter"] == 5
    assert parametros["modelo_min_samples_leaf"] == 50
    assert parametros["modelo_n_iter"] == 5  # sin early_stopping hace todas las iteraciones
    assert (parametros["umbral_sin_bicis"], parametros["umbral_sin_anclajes"]) == (0.3, 0.4)


def test_p2_parametros_de_xgb_no_repiten_los_de_la_busqueda():
    modelo = train.p2_construir_modelo("xgb_optuna", hiperparametros={"max_leaves": 7})
    parametros = train.p2_parametros_modelo(
        "xgb_optuna", modelo, {"umbrales": {1: 0.3, 2: 0.3}}, {"max_leaves": 7, "gamma": 1.0}
    )
    assert parametros["modelo_max_leaves"] == 7
    assert parametros["modelo_n_estimators"] == cfg.P2_PARAMETROS_XGB["n_estimators"]
    assert "modelo_gamma" in parametros


def test_p2_comparacion_sin_archivo_previo(tmp_path):
    comparacion = pd.DataFrame({"valid_ap_riesgo": [0.5]}, index=["hgb"])
    resultado = train.p2_combinar_comparacion(comparacion, tmp_path / "no_existe.csv")
    pd.testing.assert_frame_equal(resultado, comparacion)


def test_p2_comparacion_reemplaza_los_modelos_reentrenados_y_conserva_los_demas(tmp_path):
    ruta = tmp_path / "comparacion.csv"
    pd.DataFrame({"valid_ap_riesgo": [0.3, 0.5]}, index=["baseline-historico", "hgb"]).to_csv(ruta)
    nueva = pd.DataFrame({"valid_ap_riesgo": [0.2, 0.6]}, index=["hgb", "xgb"])
    resultado = train.p2_combinar_comparacion(nueva, ruta)
    assert resultado.index.tolist() == ["xgb", "baseline-historico", "hgb"]
    assert resultado.loc["hgb", "valid_ap_riesgo"] == 0.2


def test_p2_nombre_run_con_y_sin_variable(monkeypatch):
    monkeypatch.setattr(cfg, "MLFLOW_RUN_NAME", None)
    assert train.p2_nombre_run("hgb") == "p2-hgb"
    monkeypatch.setattr(cfg, "MLFLOW_RUN_NAME", "fase3")
    assert train.p2_nombre_run("hgb") == "fase3-hgb"


def test_p2_cargar_cambios_sin_tabla_pide_make_features(tmp_path):
    with pytest.raises(FileNotFoundError, match="make features-p2"):
        train.p2_cargar_cambios(tmp_path / "p2_cambios_estacion_bin.parquet")


def test_p2_entrenar_rechaza_candidatos_de_optuna_antes_de_leer_datos():
    with pytest.raises(typer.BadParameter, match="xgb_optuna"):
        train.p2_entrenar(
            modelos="hgb,xgb_optuna", peso_historico=20, min_samples_leaf=1000, registrar=False
        )


def test_p2_entrenar_deja_la_comparacion_de_los_tres_modelos(cambios_en_disco, monkeypatch):
    def no_registrar(*args, **kwargs):
        raise AssertionError("Con registrar=False no se debe escribir en MLflow")

    monkeypatch.setattr(train, "p2_registrar_run", no_registrar)
    train.p2_entrenar(
        modelos="baseline,hgb,xgb", peso_historico=20, min_samples_leaf=20, registrar=False
    )

    comparacion = pd.read_csv(cambios_en_disco / "p2_comparacion_modelos.csv", index_col=0)
    assert set(comparacion.index) == {"baseline-historico", "hgb", "xgb"}
    # El orden es el de selección: valid, no test
    assert comparacion["valid_ap_riesgo"].is_monotonic_decreasing
    assert {"valid_ap_riesgo", "test_ap_riesgo", "segundos_entrenamiento"} <= set(comparacion)
    assert np.isfinite(comparacion.to_numpy()).all()
    for clave in ("baseline", "hgb", "xgb"):
        for archivo in ("matriz_confusion.png", "matriz_confusion.json", "curva_pr.png"):
            assert (cambios_en_disco / f"p2_{clave}_{archivo}").exists()


def test_p2_candidato_aprende_mas_que_el_azar(figuras_temporales):
    conjuntos = p2_conjuntos_sinteticos()
    metricas = train.p2_ajustar_y_evaluar(
        "xgb", conjuntos["fit"], conjuntos["val"], conjuntos["test"], {}, False
    )
    prevalencia = (conjuntos["test"]["clase"] > 0).mean()
    assert metricas["test_ap_riesgo"] > prevalencia + 0.2
