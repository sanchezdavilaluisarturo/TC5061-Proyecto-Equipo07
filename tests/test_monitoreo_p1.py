"""Pruebas del monitoreo de P1: la tabla del mes nuevo se calcula igual que la de entrenamiento."""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import features
from tc5061_proyecto_equipo07.modeling import monitoreo_p1


def viajes_sinteticos(semilla: int = cfg.SEMILLA) -> pd.DataFrame:
    """Viajes de tres estaciones en septiembre y octubre de 2026.

    La estación 3 deja de operar el 15 de septiembre y vuelve en octubre: en la tabla de
    entrenamiento sus horas posteriores quedan fuera de operación; con octubre, dentro.
    """
    rng = np.random.default_rng(semilla)
    horas = pd.date_range("2026-09-01 06:00", "2026-10-31 22:00", freq="h")
    horas = horas[~horas.hour.isin(cfg.HORAS_SIN_SERVICIO)]
    filas = []
    for estacion in (1, 2, 3):
        elegidas = horas[rng.random(len(horas)) < 0.5]
        if estacion == 3:
            pausa = (elegidas >= "2026-09-15") & (elegidas < "2026-10-05")
            elegidas = elegidas[~pausa]
        filas.append(pd.DataFrame({"origen_id": estacion, "inicio_del_viaje": elegidas}))
    viajes = pd.concat(filas, ignore_index=True)
    viajes["inicio_del_viaje"] += pd.to_timedelta(rng.integers(0, 60, len(viajes)), unit="min")
    viajes["origen_id"] = viajes["origen_id"].astype("Int32")
    viajes["periodo_archivo"] = pd.Categorical(viajes["inicio_del_viaje"].dt.strftime("%Y-%m"))
    return viajes


ESTACIONES = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["(GDL-001) Centro", "(ZPN-002) Chapalita", "(TLQ-003) Centro Tlaquepaque"],
        "dpcapacity": [19, 15, 23],
        "latitude": [20.67, 20.66, 20.64],
        "longitude": [-103.35, -103.40, -103.31],
    }
)


@pytest.fixture(scope="module")
def tablas():
    viajes = viajes_sinteticos()
    hasta_septiembre = viajes[viajes["inicio_del_viaje"] < cfg.FRONTERA_MONITOREO]
    entrenamiento = features.p1_tabla_modelado(
        hasta_septiembre, ESTACIONES, cfg.MES_INICIO, cfg.MES_FIN
    )
    extendida = features.p1_tabla_modelado(viajes, ESTACIONES, cfg.MES_INICIO, cfg.MES_MONITOREO)
    return entrenamiento, extendida


def test_la_extendida_reproduce_la_tabla_de_entrenamiento(tablas):
    entrenamiento, extendida = tablas
    comparacion = monitoreo_p1.p1_comparar_tablas(entrenamiento, extendida)
    assert comparacion["filas_comparadas"] == len(entrenamiento)


def test_las_filas_extra_son_las_horas_tras_el_ultimo_viaje(tablas):
    entrenamiento, extendida = tablas
    comparacion = monitoreo_p1.p1_comparar_tablas(entrenamiento, extendida)
    previa = extendida[extendida["hora_inicio"] < cfg.FRONTERA_MONITOREO]
    unidas = previa.merge(entrenamiento[["origen_id", "hora_inicio"]], how="left", indicator=True)
    extra = unidas[unidas["_merge"] == "left_only"]
    assert len(extra) == comparacion["filas_extra"]
    # Toda fila de más es posterior al último viaje de su estación en la tabla de entrenamiento
    ultima_hora = entrenamiento.groupby("origen_id")["hora_inicio"].max()
    assert (extra["hora_inicio"] > extra["origen_id"].map(ultima_hora)).all()
    # La estación 3 dejó de operar el 15 de septiembre: sus dos semanas de pausa son las más
    assert extra["origen_id"].value_counts().idxmax() == 3
    assert (extra.loc[extra["origen_id"] == 3, "viajes"] == 0).all()


def test_comparar_tablas_detecta_un_valor_distinto(tablas):
    entrenamiento, extendida = tablas
    alterada = entrenamiento.copy()
    alterada.loc[10, "viajes_rezago_7d"] = 99.0
    with pytest.raises(ValueError, match="cambian"):
        monitoreo_p1.p1_comparar_tablas(alterada, extendida)


def test_comparar_tablas_detecta_filas_faltantes(tablas):
    entrenamiento, extendida = tablas
    recortada = extendida.drop(index=extendida.index[:5])
    with pytest.raises(ValueError, match="no están"):
        monitoreo_p1.p1_comparar_tablas(entrenamiento, recortada)


def test_la_evaluacion_solo_trae_el_mes_nuevo(tablas):
    _, extendida = tablas
    evaluacion = monitoreo_p1.p1_evaluacion(extendida)
    assert (evaluacion["hora_inicio"] >= cfg.FRONTERA_MONITOREO).all()
    assert (evaluacion["periodo"] == "evaluacion").all()
    assert "particion" not in evaluacion.columns


def test_los_rezagos_del_mes_nuevo_usan_septiembre(tablas):
    _, extendida = tablas
    evaluacion = monitoreo_p1.p1_evaluacion(extendida)
    primer_dia = evaluacion[evaluacion["hora_inicio"].dt.day == 1]
    # El rezago de 1 día del 1 de octubre viene del 30 de septiembre, que está en la historia
    assert primer_dia.loc[primer_dia["origen_id"] != 3, "viajes_rezago_1d"].notna().all()


def test_conjuntos_etiqueta_periodos_y_descarta_brechas(tablas, tmp_path):
    entrenamiento, extendida = tablas
    ruta_entrenamiento = tmp_path / "entrenamiento.parquet"
    ruta_evaluacion = tmp_path / "evaluacion.parquet"
    entrenamiento.to_parquet(ruta_entrenamiento, index=False)
    monitoreo_p1.p1_evaluacion(extendida).to_parquet(ruta_evaluacion, index=False)

    conjuntos = monitoreo_p1.p1_conjuntos(ruta_entrenamiento, ruta_evaluacion)
    assert set(conjuntos) == {"referencia", "valid", "actual", "evaluacion"}
    actual = conjuntos["actual"]
    # Las 24 h previas al mes nuevo quedan en brecha, igual que en las demás fronteras
    assert actual["hora_inicio"].max() < pd.Timestamp(cfg.FRONTERA_MONITOREO) - pd.Timedelta(
        hours=cfg.BRECHA_HORAS
    )
    assert set(conjuntos["evaluacion"]["periodo"]) == {"evaluacion"}


def test_conjuntos_pide_la_tabla_que_falta(tmp_path):
    with pytest.raises(FileNotFoundError, match="features-p1"):
        monitoreo_p1.p1_conjuntos(tmp_path / "no.parquet", tmp_path / "tampoco.parquet")


def test_la_linea_de_comandos_expone_tabla():
    # Con un solo comando, Typer lo ejecutaría sin su nombre y `make monitoreo-p1-tabla` fallaría
    resultado = CliRunner().invoke(monitoreo_p1.app, ["--help"])
    assert resultado.exit_code == 0
    assert "tabla" in resultado.output


def tabla_larga(semilla: int = cfg.SEMILLA) -> pd.DataFrame:
    """Tabla de P1 de tres estaciones de 2023-09 a 2026-10, para tener todos los periodos."""
    rng = np.random.default_rng(semilla)
    horas = pd.date_range("2023-09-01 05:00", "2026-10-31 23:00", freq="h")
    horas = horas[~horas.hour.isin(cfg.HORAS_SIN_SERVICIO)]
    filas = []
    for estacion, nivel in ((1, 1.5), (2, 0.8), (3, 0.4)):
        # La demanda crece con el tiempo: drift real en rezagos y target
        tendencia = np.linspace(1.0, 1.8, len(horas))
        conteos = rng.poisson(nivel * tendencia)
        repetidas = np.repeat(horas, conteos)
        filas.append(pd.DataFrame({"origen_id": estacion, "inicio_del_viaje": repetidas}))
    viajes = pd.concat(filas, ignore_index=True)
    viajes["origen_id"] = viajes["origen_id"].astype("Int32")
    viajes["periodo_archivo"] = pd.Categorical(viajes["inicio_del_viaje"].dt.strftime("%Y-%m"))
    return features.p1_tabla_modelado(viajes, ESTACIONES, cfg.MES_INICIO, cfg.MES_MONITOREO)


@pytest.fixture(scope="module")
def proyecto(tmp_path_factory):
    """Tablas de entrenamiento y evaluación y un champion de P1 en un MLflow local."""
    import mlflow
    import mlflow.sklearn

    from tc5061_proyecto_equipo07.modeling.train import p1_construir_modelo

    carpeta = tmp_path_factory.mktemp("proyecto")
    extendida = tabla_larga()
    entrenamiento = extendida[extendida["hora_inicio"] < cfg.FRONTERA_MONITOREO]
    entrenamiento.to_parquet(carpeta / "entrenamiento.parquet", index=False)
    monitoreo_p1.p1_evaluacion(extendida).to_parquet(carpeta / "evaluacion.parquet", index=False)

    uri = f"sqlite:///{carpeta / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("prueba-p1")
    train = entrenamiento[entrenamiento["particion"] == "train"]
    modelo = p1_construir_modelo("xgb", hiperparametros=None)
    modelo.set_params(modelo__n_estimators=20)
    modelo.fit(train[cfg.P1_VARIABLES], train["viajes"])
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            serialization_format="cloudpickle",  # igual que train.py: skops rechaza clases propias
        )
    # Igual que registro_p1: registrar desde el modelo registrado del run (llena su model_id)
    version = mlflow.register_model(f"models:/{info.model_id}", cfg.P1_MODELO_REGISTRADO).version
    cliente = mlflow.MlflowClient()
    cliente.set_registered_model_alias(cfg.P1_MODELO_REGISTRADO, "champion", version)
    return {"carpeta": carpeta, "uri": uri, "version": int(version), "modelo_id": info.model_id}


@pytest.fixture
def config_prueba(proyecto, monkeypatch):
    carpeta = proyecto["carpeta"]
    monkeypatch.setattr(cfg, "RUTA_P1_DEMANDA", carpeta / "entrenamiento.parquet")
    monkeypatch.setattr(cfg, "P1_RUTA_EVALUACION", carpeta / "evaluacion.parquet")
    monkeypatch.setattr(cfg, "MONITOREO_DIR", carpeta / "reportes")
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", proyecto["uri"])
    monkeypatch.setattr(cfg, "EXPERIMENTO_MLFLOW", "prueba-p1")
    for nombre, constante in (
        ("fuentes.csv", "RUTA_MANIFIESTO"),
        ("monitoreo.csv", "RUTA_MANIFIESTO_MONITOREO"),
    ):
        ruta = carpeta / nombre
        ruta.write_text("archivo,periodo,url,registros,sha256\na.csv,2026-10,u,1,abc\n")
        monkeypatch.setattr(cfg, constante, ruta)
    return proyecto


def contextos_del_run(run) -> set[str]:
    return {
        etiqueta.value
        for entrada in run.inputs.dataset_inputs
        for etiqueta in entrada.tags
        if etiqueta.key == "mlflow.data.context"
    }


def test_columnas_del_reporte_excluyen_origen_id():
    assert "origen_id" not in monitoreo_p1.P1_NUMERICAS + monitoreo_p1.P1_CATEGORICAS
    assert len(monitoreo_p1.P1_NUMERICAS) + len(monitoreo_p1.P1_CATEGORICAS) == 13


def test_mismo_mes_solo_trae_octubres():
    referencia = pd.DataFrame({"hora_inicio": pd.to_datetime(["2023-10-02", "2023-11-02"])})
    assert monitoreo_p1.p1_mismo_mes(referencia, "2026-10")["hora_inicio"].dt.month.eq(10).all()


def test_cargar_champion_usa_el_alias(config_prueba):
    modelo, version, _, modelo_id = monitoreo_p1.p1_cargar_champion()
    assert version == config_prueba["version"]
    assert hasattr(modelo, "predict")
    assert modelo_id == config_prueba["modelo_id"]


def test_drift_genera_reportes_resumen_y_run(config_prueba):
    import mlflow

    monitoreo_p1.drift(muestra=3000)
    carpeta = cfg.MONITOREO_DIR
    for nombre in ("p1_drift_train_test", "p1_desempeno_valid_test", "p1_drift_estacional"):
        assert (carpeta / f"{nombre}.html").exists()

    resumen = pd.read_csv(carpeta / "p1_drift_resumen.csv")
    assert set(resumen["comparacion"]) == {"train_vs_test", "mismo_mes_vs_evaluacion"}
    train_test = resumen[resumen["comparacion"] == "train_vs_test"].set_index("columna")
    assert "origen_id" not in train_test.index
    # La demanda sintética crece: los rezagos, el target y la predicción se mueven
    assert train_test.loc["viajes_rezago_7d", "drift"]
    assert train_test.loc["prediccion", "drift"]
    # dias_desde_apertura crece con el tiempo: drift por construcción
    assert train_test.loc["dias_desde_apertura", "drift"]
    assert not train_test.loc["latitud", "drift"]

    desempeno = pd.read_csv(carpeta / "p1_desempeno.csv")
    assert set(desempeno["modelo"]) == {"champion", "baseline"}
    assert set(desempeno["conjunto"]) == {"valid", "actual", "evaluacion"}

    runs = mlflow.search_runs(
        experiment_names=["prueba-p1"], filter_string="tags.fase = 'monitoreo'"
    )
    run = runs.iloc[0]
    assert run["tags.mlflow.runName"] == "p1-drift"
    assert run["params.champion_version"] == str(config_prueba["version"])
    assert "metrics.champion_mae_evaluacion" in runs.columns
    assert 0 <= run["metrics.proporcion_drift_train_test"] <= 1
    # Misma calidad que los runs de entrenamiento: datasets, modelo evaluado y descripción
    completo = mlflow.get_run(run["run_id"])
    assert contextos_del_run(completo) == {"referencia", "valid", "actual", "evaluacion"}
    _, _, _, modelo_id = monitoreo_p1.p1_cargar_champion()
    assert [m.model_id for m in completo.inputs.model_inputs] == [modelo_id]
    assert completo.data.tags["mlflow.note.content"].startswith("Monitoreo de P1")
    assert completo.data.tags["autor"] == "Daniel Sánchez Hernández"
    metricas_modelo = {m.key: m.dataset_name for m in mlflow.get_logged_model(modelo_id).metrics}
    assert metricas_modelo["champion_mae_evaluacion"] == "p1_evaluacion"


def test_drift_anual_compara_cada_anio_contra_el_mes_nuevo(config_prueba):
    import mlflow

    monitoreo_p1.drift_anual(muestra=2000)
    tabla = pd.read_csv(cfg.MONITOREO_DIR / "p1_drift_mismo_mes_por_anio.csv")
    # Datos sintéticos de 2023-09 a 2026-10: un octubre por periodo y el mes de evaluación
    assert tabla[["anio", "periodo"]].values.tolist() == [
        [2023, "referencia"],
        [2024, "valid"],
        [2025, "actual"],
        [2026, "evaluacion"],
    ]
    # La demanda sintética crece: mientras más lejano el año, mayor el drift contra 2026
    drift = tabla.set_index("anio")["drift_viajes"].dropna()
    assert drift.is_monotonic_decreasing
    assert pd.isna(tabla.set_index("anio").loc[2026, "drift_viajes"])
    assert (tabla["viajes_promedio"].diff().dropna() > 0).all()

    runs = mlflow.search_runs(
        experiment_names=["prueba-p1"], filter_string="tags.tipo_monitoreo = 'drift_estacional'"
    )
    assert runs.iloc[0]["tags.mlflow.runName"] == "p1-drift-mismo-mes"
    assert "metrics.ventaja_champion_2026" in runs.columns
    completo = mlflow.get_run(runs.iloc[0]["run_id"])
    assert contextos_del_run(completo) == {"mes_2023", "mes_2024", "mes_2025", "evaluacion"}
    assert len(completo.inputs.model_inputs) == 1


def test_modelo_id_sale_del_source_de_la_version():
    from types import SimpleNamespace

    version = SimpleNamespace(model_id=None, source="models:/m-abc123", run_id="r", version="2")
    assert monitoreo_p1.p1_modelo_id(version) == "m-abc123"
    version.model_id = "m-directo"
    assert monitoreo_p1.p1_modelo_id(version) == "m-directo"
