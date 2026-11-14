"""Pruebas del monitoreo de P2: tabla del mes nuevo, conjuntos y drift con Evidently.
Sin red ni servidor de MLflow.
"""

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import features
from tc5061_proyecto_equipo07.modeling import monitoreo_p2
from tc5061_proyecto_equipo07.modeling.train import P2BaselineHistorico

ESTACIONES = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "latitude": [20.67, 20.66, 20.64, 20.70],
        "longitude": [-103.35, -103.40, -103.31, -103.36],
        "dpcapacity": [6, 8, 5, 0],
    }
)


def viajes_sinteticos(desde: str, hasta: str, semilla: int = cfg.SEMILLA) -> pd.DataFrame:
    """Viajes entre las estaciones 1 a 4, de 5:00 a 23:59, con duración de 5 a 40 minutos."""
    rng = np.random.default_rng(semilla)
    minutos = pd.date_range(desde, hasta, freq="7min")
    minutos = minutos[minutos.hour >= 5]
    inicio = minutos[rng.random(len(minutos)) < 0.6]
    n = len(inicio)
    return pd.DataFrame(
        {
            "inicio_del_viaje": inicio,
            "fin_del_viaje": inicio + pd.to_timedelta(rng.integers(5, 41, n), unit="min"),
            "origen_id": rng.integers(1, 5, n),
            "destino_id": rng.integers(1, 5, n),
        }
    )


def tabla_cambios(
    dias: pd.DatetimeIndex,
    estaciones: dict[int, int],
    semilla: int = cfg.SEMILLA,
    horas=range(10, 30),
    escala=None,
) -> pd.DataFrame:
    """Tabla estación-bin con las columnas de P2 (sin historia) y una clase que depende del flujo.

    estaciones: id -> capacidad. escala: función del día que multiplica salidas y entradas
    (para simular que el flujo crece con el tiempo).
    """
    rng = np.random.default_rng(semilla)
    llave = pd.MultiIndex.from_product(
        [list(estaciones), dias, list(horas)], names=["estacion", "dia", "hora_idx"]
    ).to_frame(index=False)
    n = len(llave)
    factor = np.ones(n) if escala is None else escala(llave["dia"]).to_numpy()
    tabla = llave.copy()
    tabla["dpcapacity"] = tabla["estacion"].map(estaciones).astype(float)
    tabla["latitude"] = 20.6 + tabla["estacion"] / 100
    tabla["longitude"] = -103.3 - tabla["estacion"] / 100
    for k in (1, 2, 4, 8):
        tabla[f"neto_{k}_relativo"] = rng.normal(0, 1, n)
    for k in (2, 4):
        tabla[f"salidas_{k}"] = rng.poisson(1.5 * k * factor)
        tabla[f"entradas_{k}"] = rng.poisson(1.5 * k * factor)
    tabla["cumsum_dia_relativo"] = rng.normal(0, 1, n)
    tabla["hora_sin"] = np.sin(2 * np.pi * tabla["hora_idx"] / 48)
    tabla["hora_cos"] = np.cos(2 * np.pi * tabla["hora_idx"] / 48)
    tabla["es_fin_semana"] = (tabla["dia"].dt.dayofweek >= 5).astype(int)
    # Riesgo que el modelo puede aprender: sin bicis si el neto reciente es muy negativo
    neto = tabla["neto_2_relativo"].to_numpy()
    p_bicis = 1 / (1 + np.exp(3 + 2 * neto))
    p_anclajes = 1 / (1 + np.exp(3 - 2 * neto))
    u = rng.random(n)
    tabla["clase"] = np.select([u < p_bicis, u < p_bicis + p_anclajes], [1, 2], 0)
    return tabla[[*cfg.P2_FEATURES_SIN_HISTORICO, "clase", "estacion", "dia", "hora_idx"]]


def con_historia(tabla: pd.DataFrame) -> pd.DataFrame:
    """Agrega hist_sin_bicis y hist_sin_anclajes como en p2_conjuntos (frecuencia global de train)."""
    en_train = features.p2_asignar_particion(tabla["dia"]) == "train"
    historia = features.P2HistoriaHastaAyer(cfg.P2_PESO_HISTORICO).fit(tabla[en_train])
    return historia.transform(tabla).reset_index(drop=True)


@pytest.fixture(scope="module")
def historia_y_mes():
    """Un día de train (para la frecuencia global), septiembre de 2026 y diez días de octubre."""
    dias = pd.DatetimeIndex(
        [pd.Timestamp("2024-09-02")]
        + list(pd.date_range("2026-09-01", "2026-09-29"))
        + list(pd.date_range("2026-10-01", "2026-10-10"))
    )
    tabla = tabla_cambios(dias, {1: 6, 2: 8}, horas=range(10, 14))
    entrenamiento = tabla[tabla["dia"] < cfg.FRONTERA_MONITOREO].reset_index(drop=True)
    mes = tabla[tabla["dia"] >= cfg.FRONTERA_MONITOREO].reset_index(drop=True)
    return entrenamiento, mes


def test_la_tabla_desde_reproduce_la_de_entrenamiento_con_solo_el_margen_de_viajes():
    # La tabla de entrenamiento se construye con toda la historia; la de monitoreo solo con los
    # viajes desde un margen de DURACION_MAXIMA_MIN antes
    viajes = viajes_sinteticos("2026-08-20", "2026-10-31 23:00")
    entrenamiento = features.p2_construir_caracteristicas(
        features.p2_limpiar_viajes(viajes, ESTACIONES),
        features.p2_estaciones_con_capacidad(ESTACIONES),
    )
    desde = pd.Timestamp("2026-09-01")
    margen = pd.Timedelta(minutes=cfg.DURACION_MAXIMA_MIN)
    leidos = viajes[viajes["inicio_del_viaje"] >= desde - margen]
    nueva = monitoreo_p2.p2_tabla_desde(leidos, ESTACIONES, desde, cfg.P2_FECHA_LIMITE_FIN_VIAJE)

    assert nueva["dia"].min() == desde  # los días incompletos del margen se descartan
    assert nueva["dia"].max() < pd.Timestamp(cfg.FRONTERA_MONITOREO)  # respeta la fecha límite
    assert 4 not in set(nueva["estacion"])  # sin capacidad
    previa = entrenamiento[entrenamiento["dia"] >= desde]
    assert monitoreo_p2.p2_comparar_tablas(previa, nueva) == len(previa)


def test_comparar_tablas_detecta_valores_distintos_y_filas_sin_par(historia_y_mes):
    entrenamiento, _ = historia_y_mes
    alterada = entrenamiento.copy()
    alterada.loc[5, "salidas_4"] += 1
    with pytest.raises(ValueError, match="cambian al recalcularse"):
        monitoreo_p2.p2_comparar_tablas(entrenamiento, alterada)
    with pytest.raises(ValueError, match="solo en una de las dos tablas"):
        monitoreo_p2.p2_comparar_tablas(entrenamiento, entrenamiento.iloc[3:])


def test_la_historia_del_mes_nuevo_sigue_a_septiembre_con_la_frecuencia_de_train(historia_y_mes):
    entrenamiento, mes = historia_y_mes
    septiembre = entrenamiento["dia"] >= "2026-09-01"
    evaluacion = monitoreo_p2.p2_agregar_historia(entrenamiento.assign(clase=septiembre * 1), mes)
    assert (evaluacion["dia"] >= cfg.FRONTERA_MONITOREO).all() and len(evaluacion) == len(mes)
    habiles = entrenamiento.drop_duplicates("dia")["dia"].dt.dayofweek < 5
    esperado = (habiles.sum() - 1) / (habiles.sum() + cfg.P2_PESO_HISTORICO)
    primer_dia = evaluacion["dia"] == cfg.FRONTERA_MONITOREO
    assert np.allclose(evaluacion.loc[primer_dia, "hist_sin_bicis"], esperado)


def test_conjuntos_etiqueta_periodos_y_descarta_brechas(tmp_path):
    dias = pd.DatetimeIndex(
        ["2024-09-15", "2024-09-30", "2024-10-01", "2025-09-30", "2025-10-01", "2026-09-30"]
    )
    entrenamiento = tabla_cambios(dias, {1: 6, 2: 8}, horas=range(10, 12))
    mes = tabla_cambios(
        pd.date_range("2026-10-01", "2026-10-03"), {1: 6, 2: 8}, horas=range(10, 12)
    )
    evaluacion = monitoreo_p2.p2_agregar_historia(entrenamiento, mes)
    entrenamiento.to_parquet(tmp_path / "cambios.parquet", index=False)
    evaluacion.to_parquet(tmp_path / "evaluacion.parquet", index=False)

    conjuntos = monitoreo_p2.p2_conjuntos(
        tmp_path / "cambios.parquet", tmp_path / "evaluacion.parquet"
    )
    # Los días previos a cada frontera quedan en brecha, también el previo al mes nuevo
    assert {n: sorted(d["dia"].dt.strftime("%m-%d").unique()) for n, d in conjuntos.items()} == {
        "referencia": ["09-15"],
        "valid": ["10-01"],
        "actual": ["10-01"],
        "evaluacion": ["10-01", "10-02", "10-03"],
    }
    assert all({"hist_sin_bicis", "hist_sin_anclajes"} <= set(d) for d in conjuntos.values())


def test_conjuntos_pide_la_tabla_que_falta(tmp_path):
    with pytest.raises(FileNotFoundError, match="features-p2"):
        monitoreo_p2.p2_conjuntos(tmp_path / "no.parquet", tmp_path / "tampoco.parquet")
    (tmp_path / "cambios.parquet").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="monitoreo-p2-tabla"):
        monitoreo_p2.p2_conjuntos(tmp_path / "cambios.parquet", tmp_path / "tampoco.parquet")


def test_mismo_mes_solo_trae_el_mes_de_evaluacion():
    referencia = pd.DataFrame({"dia": pd.to_datetime(["2022-10-03", "2022-11-03", "2023-10-09"])})
    mismo_mes = monitoreo_p2.p2_mismo_mes(referencia, "2026-10")
    assert mismo_mes["dia"].dt.strftime("%Y-%m").tolist() == ["2022-10", "2023-10"]


def test_metricas_separan_prevalencia_y_calibracion_y_dejan_vacio_con_pocos_positivos():
    y = np.array([0] * 900 + [1] * 50 + [2] * 50)
    proba = np.tile([0.8, 0.1, 0.1], (len(y), 1))  # predice el doble de la prevalencia real
    pred = np.zeros(len(y), dtype=int)
    metricas = monitoreo_p2.p2_metricas(y, proba, pred)
    assert metricas["prevalencia_sin_bicis"] == pytest.approx(0.05)
    assert metricas["calibracion_sin_bicis"] == pytest.approx(2.0)
    assert metricas["auc_sin_bicis"] == pytest.approx(0.5)  # un score constante no ordena

    pocos = np.array([0] * 990 + [1] * 10)
    metricas = monitoreo_p2.p2_metricas(pocos, np.tile([0.9, 0.05, 0.05], (1000, 1)), pred)
    assert metricas["prevalencia_sin_bicis"] == pytest.approx(0.01)
    assert np.isnan(metricas["auc_sin_bicis"]) and np.isnan(metricas["lift_ap_sin_bicis"])
    assert np.isnan(metricas["calibracion_sin_anclajes"])  # sin positivos no hay prevalencia


def test_desempeno_separa_cohortes_de_estaciones_conocidas_y_nuevas():
    # Un día de train para la frecuencia global de la historia, más septiembre y octubre de 2026
    dias = pd.DatetimeIndex(
        [pd.Timestamp("2024-09-02"), *pd.date_range("2026-09-21", "2026-10-10")]
    )
    datos = con_historia(tabla_cambios(dias, {1: 6, 2: 8, 9: 30}, horas=range(10, 20)))
    version = {"modelo": P2BaselineHistorico().fit(datos), "umbrales": {1: 0.1, 2: 0.1}}
    nuevas = datos["estacion"] == 9  # no está en la referencia

    por_conjunto = monitoreo_p2.p2_desempeno(
        {"baseline": version}, {"evaluacion": datos}, np.array([1, 2])
    )
    filas = por_conjunto.set_index("cohorte")["filas"]
    assert filas["todas"] == len(datos)
    assert filas["nuevas"] == nuevas.sum()
    assert filas["conocidas"] + filas["nuevas"] == filas["todas"]
    assert por_conjunto["proporcion_estaciones_nuevas"].iloc[0] == pytest.approx(nuevas.mean())


@pytest.fixture(scope="module")
def proyecto(tmp_path_factory):
    """Tablas de 2023-09 a 2026-10 con flujo creciente y una estación nueva en el mes nuevo, y un
    champion XGBoost pequeño registrado con sus umbrales, igual que registro_p2."""
    import mlflow
    import mlflow.sklearn
    from xgboost import XGBClassifier

    carpeta = tmp_path_factory.mktemp("proyecto_p2")

    def crece(dia: pd.Series) -> pd.Series:  # el flujo pasa de 1x a 3x entre 2023-09 y 2026-10
        return 1 + 2 * (dia - pd.Timestamp("2023-09-01")) / pd.Timedelta(days=1155)

    entrenamiento = tabla_cambios(
        pd.date_range("2023-09-01", "2026-09-30"), {1: 6, 2: 8, 3: 5}, escala=crece
    )
    mes = tabla_cambios(
        pd.date_range("2026-10-01", "2026-10-31"),
        {1: 6, 2: 8, 3: 5, 7: 40},  # la 7 abrió después de train y es más grande
        semilla=cfg.SEMILLA + 1,
        escala=crece,
    )
    entrenamiento.to_parquet(carpeta / "cambios.parquet", index=False)
    monitoreo_p2.p2_agregar_historia(entrenamiento, mes).to_parquet(
        carpeta / "evaluacion.parquet", index=False
    )

    uri = f"sqlite:///{(carpeta / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    # Los artefactos van a la carpeta temporal, no al ./mlruns del directorio de trabajo
    mlflow.create_experiment("prueba-p2", artifact_location=(carpeta / "artefactos").as_uri())
    mlflow.set_experiment("prueba-p2")
    tabla = con_historia(entrenamiento)
    train = tabla[features.p2_asignar_particion(tabla["dia"]) == "train"]
    modelo = XGBClassifier(
        **{**cfg.P2_PARAMETROS_XGB, "n_estimators": 20}, random_state=cfg.SEMILLA
    )
    modelo.fit(train[cfg.P2_FEATURES_CON_HISTORICO], train["clase"])
    with mlflow.start_run():
        mlflow.log_params({"umbral_sin_bicis": 0.2, "umbral_sin_anclajes": 0.2})
        # Igual que p2_registrar_run en train.py
        info = mlflow.sklearn.log_model(modelo, name="modelo", serialization_format="cloudpickle")
    version = mlflow.register_model(f"models:/{info.model_id}", cfg.P2_MODELO_REGISTRADO).version
    mlflow.MlflowClient().set_registered_model_alias(cfg.P2_MODELO_REGISTRADO, "champion", version)
    return {"carpeta": carpeta, "uri": uri, "version": int(version), "modelo_id": info.model_id}


@pytest.fixture
def config_prueba(proyecto, monkeypatch):
    carpeta = proyecto["carpeta"]
    monkeypatch.setattr(cfg, "RUTA_P2_CAMBIOS", carpeta / "cambios.parquet")
    monkeypatch.setattr(cfg, "P2_RUTA_EVALUACION", carpeta / "evaluacion.parquet")
    monkeypatch.setattr(cfg, "MONITOREO_DIR", carpeta / "reportes")
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", proyecto["uri"])
    monkeypatch.setattr(cfg, "EXPERIMENTO_MLFLOW", "prueba-p2")
    for nombre, constante in (
        ("fuentes.csv", "RUTA_MANIFIESTO"),
        ("monitoreo.csv", "RUTA_MANIFIESTO_MONITOREO"),
    ):
        ruta = carpeta / nombre
        ruta.write_text("archivo,periodo,url,registros,sha256\na.csv,2026-10,u,1,abc\n")
        monkeypatch.setattr(cfg, constante, ruta)
    return proyecto


def contextos_del_run(run) -> set[str]:
    """Contextos de los datasets de un run (también sirve para el run del A/B)."""
    return {
        etiqueta.value
        for entrada in run.inputs.dataset_inputs
        for etiqueta in entrada.tags
        if etiqueta.key == "mlflow.data.context"
    }


def test_drift_genera_reportes_tablas_y_el_run_del_champion(config_prueba):
    import mlflow

    monitoreo_p2.drift(muestra=3000)
    carpeta = cfg.MONITOREO_DIR
    for nombre in (
        "p2_drift_train_test",
        "p2_drift_estacional",
        "p2_desempeno_valid_test",
        "p2_desempeno_valid_evaluacion",
    ):
        assert (carpeta / f"{nombre}.html").exists() and (carpeta / f"{nombre}.json").exists()
    assert (carpeta / "p2_serie_mensual.png").exists()

    resumen = pd.read_csv(carpeta / "p2_drift_resumen.csv")
    assert set(resumen["comparacion"]) == {"train_test", "estacional"}
    train_test = resumen[resumen["comparacion"] == "train_test"].set_index("columna")
    # 17 variables, 2 probabilidades, la clase y la predicción
    assert len(train_test) == 21 and "estacion" not in train_test.index
    # El flujo sintético crece con el tiempo; la hora y el fin de semana no cambian
    assert train_test.loc["salidas_4", "drift"] and train_test.loc["entradas_4", "drift"]
    assert not train_test.loc["hora_sin", "drift"]
    assert not train_test.loc["es_fin_semana", "drift"]
    # La estación nueva (capacidad 40) solo aparece en el mes nuevo
    estacional = resumen[resumen["comparacion"] == "estacional"].set_index("columna")
    assert estacional.loc["dpcapacity", "drift"]

    desempeno = pd.read_csv(carpeta / "p2_desempeno.csv")
    assert set(desempeno["modelo"]) == {"champion", "baseline"}
    assert set(desempeno["conjunto"]) == {"valid", "actual", "evaluacion"}
    assert "nuevas" in set(desempeno.query("conjunto == 'evaluacion'")["cohorte"])
    mismo_mes = pd.read_csv(carpeta / "p2_mismo_mes_por_anio.csv")
    assert sorted(set(mismo_mes["mes"])) == ["2023-10", "2024-10", "2025-10", "2026-10"]

    runs = mlflow.search_runs(
        experiment_names=["prueba-p2"], filter_string="tags.tipo_monitoreo = 'drift'"
    )
    run = runs.iloc[0]
    assert run["tags.mlflow.runName"] == "p2-drift"
    assert run["params.champion_version"] == str(config_prueba["version"])
    assert float(run["params.umbral_sin_bicis"]) == pytest.approx(0.2)
    assert 0 <= run["metrics.proporcion_drift_train_test"] <= 1
    completo = mlflow.get_run(run["run_id"])
    assert contextos_del_run(completo) == {"referencia", "valid", "actual", "evaluacion"}
    assert [m.model_id for m in completo.inputs.model_inputs] == [config_prueba["modelo_id"]]
    assert completo.data.tags["mlflow.note.content"].startswith("Monitoreo de P2")
    metricas_modelo = {m.key for m in mlflow.get_logged_model(config_prueba["modelo_id"]).metrics}
    assert "champion_auc_riesgo_evaluacion" in metricas_modelo


def test_challenger_sin_registrar_no_escribe_en_mlflow(config_prueba):
    import mlflow

    antes = len(mlflow.search_runs(experiment_names=["prueba-p2"]))
    modelo, umbrales = monitoreo_p2.challenger(registrar=False)

    assert len(mlflow.search_runs(experiment_names=["prueba-p2"])) == antes
    assert set(umbrales) == {1, 2} and all(0 < u < 1 for u in umbrales.values())
    assert modelo.predict_proba(
        pd.read_parquet(cfg.P2_RUTA_EVALUACION)[cfg.P2_FEATURES_CON_HISTORICO]
    ).shape[1] == len(cfg.P2_CLASES)


def test_challenger_se_registra_sin_mover_el_champion(config_prueba):
    import mlflow

    champion_antes = monitoreo_p2.p2_cargar_alias("champion")
    monitoreo_p2.challenger(registrar=True)

    cliente = mlflow.MlflowClient()
    retador = cliente.get_model_version_by_alias(cfg.P2_MODELO_REGISTRADO, "challenger")
    champion = cliente.get_model_version_by_alias(cfg.P2_MODELO_REGISTRADO, "champion")
    assert int(champion.version) == champion_antes["version"]
    assert int(retador.version) != champion_antes["version"]
    # Linaje: de qué versión sale; no lleva valid_ap_riesgo porque aprendió con valid
    assert retador.tags["base_version"] == str(champion_antes["version"])
    assert retador.tags["base_source_run_id"] == champion_antes["run_id"]
    assert retador.tags["fase"] == cfg.FASE_MONITOREO
    assert "valid_ap_riesgo" not in retador.tags

    run = mlflow.get_run(retador.run_id)
    assert run.data.tags["stage_of_work"] == "challenger"
    assert run.data.params["modelo"] == "xgb_challenger"
    assert run.data.params["rango_datos"].endswith("2026-09")
    # Entrena con referencia, valid y actual sin brechas, nunca con el mes de evaluación
    tabla = monitoreo_p2.p2_tabla_con_historia(cfg.RUTA_P2_CAMBIOS)
    esperadas = tabla["periodo"].isin(["referencia", "valid", "actual"]).sum()
    assert int(run.data.params["filas_fit"]) == esperadas

    # Mismos hiperparámetros que el champion (data iteration) y umbrales que el A/B puede leer
    challenger = monitoreo_p2.p2_cargar_alias("challenger")
    assert {k: str(v) for k, v in challenger["modelo"].get_params().items()} == {
        k: str(v) for k, v in champion_antes["modelo"].get_params().items()
    }
    assert set(challenger["umbrales"]) == {1, 2}
