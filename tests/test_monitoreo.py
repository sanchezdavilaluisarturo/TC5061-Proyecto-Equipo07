"""Pruebas de la base de monitoreo: periodos, limpieza del mes nuevo, drift con Evidently y A/B."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset, monitoreo


def test_asignar_periodo_respeta_fronteras_y_brechas():
    fechas = pd.Series(
        pd.to_datetime(
            [
                "2024-09-29 23:00",  # referencia
                "2024-09-30 12:00",  # brecha antes de valid
                "2024-10-01 00:00",  # valid
                "2025-09-30 12:00",  # brecha antes de test
                "2025-10-01 00:00",  # actual
                "2026-09-30 12:00",  # brecha antes del mes nuevo
                "2026-10-01 00:00",  # evaluacion
                "2026-10-31 23:59",  # evaluacion
            ]
        )
    )
    esperado = [
        "referencia",
        "brecha",
        "valid",
        "brecha",
        "actual",
        "brecha",
        "evaluacion",
        "evaluacion",
    ]
    assert monitoreo.asignar_periodo(fechas).tolist() == esperado


def test_asignar_periodo_rechaza_fechas_posteriores_al_mes_nuevo():
    with pytest.raises(ValueError, match="posteriores"):
        monitoreo.asignar_periodo(pd.Series(pd.to_datetime(["2026-11-01 08:00"])))


def viajes_sinteticos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Viajes con un caso por regla de limpieza y la nomenclatura de sus estaciones."""
    inicio = pd.to_datetime(["2026-10-01 08:00"] * 6)
    viajes = pd.DataFrame(
        {
            "viaje_id": pd.array([1, 1, 2, 3, 4, 5], dtype="Int64"),
            "usuario_id": pd.array([10, 10, 11, 12, 13, 14], dtype="Int64"),
            "genero": pd.Categorical(["M", "M", "F", "M", None, "F"]),
            "anio_de_nacimiento": pd.array([1990, 1990, 1850, 1985, 2024, 1970], dtype="Int16"),
            "inicio_del_viaje": inicio,
            "fin_del_viaje": inicio + pd.to_timedelta([15, 15, 20, 0, 10, 2000], unit="min"),
            "origen_id": pd.array([1, 1, 2, 1, 2, 1], dtype="Int32"),
            "destino_id": pd.array([2, 2, 1, 2, 99, 2], dtype="Int32"),
            "periodo_archivo": pd.Categorical(["2026-10"] * 6),
        }
    )
    estaciones = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
    return viajes, estaciones


def test_limpiar_viajes_aplica_las_reglas_de_dataset():
    viajes, estaciones = viajes_sinteticos()
    limpios = monitoreo.limpiar_viajes(viajes, estaciones)
    # Se retiran: el duplicado, la duración cero, la estación 99 y la duración de 2000 min
    assert limpios["viaje_id"].tolist() == [1, 2]
    # 1850 es un año imposible: se anula y el viaje se queda
    assert limpios["anio_de_nacimiento"].isna().tolist() == [False, True]


def test_limpiar_viajes_coincide_con_dataset_limpiar(tmp_path, monkeypatch):
    viajes, estaciones = viajes_sinteticos()
    for carpeta in ("interim", "processed", "estaciones"):
        (tmp_path / carpeta).mkdir()
    viajes.to_parquet(tmp_path / "interim" / "viajes_2026-10_2026-10.parquet", index=False)
    estaciones.to_csv(tmp_path / "estaciones" / "nomenclatura_2026_10.csv", index=False)
    monkeypatch.setattr(dataset, "INTERIM_DATA_DIR", tmp_path / "interim")
    monkeypatch.setattr(dataset, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(dataset, "RUTA_ESTACIONES_CRUDAS", tmp_path / "estaciones")

    dataset.limpiar("2026-10", "2026-10")
    del_proyecto = pd.read_parquet(
        tmp_path / "processed" / "viajes_limpios_2026-10_2026-10.parquet"
    )
    pd.testing.assert_frame_equal(monitoreo.limpiar_viajes(viajes, estaciones), del_proyecto)


def test_estaciones_nuevas_solo_regresa_las_que_no_estaban():
    entrenamiento = pd.DataFrame({"id": [1, 2, 3]})
    monitoreo_nomenclatura = pd.DataFrame({"id": [2, 3, 4, 5]})
    nuevas = monitoreo.estaciones_nuevas(entrenamiento, monitoreo_nomenclatura)
    assert nuevas["id"].tolist() == [4, 5]


def test_unir_con_historia_conserva_filas_columnas_y_categorias():
    mes, _ = viajes_sinteticos()
    historia = mes.assign(periodo_archivo=pd.Categorical(["2026-09"] * len(mes)))
    unidos = monitoreo.unir_con_historia(historia, mes)
    assert len(unidos) == len(historia) + len(mes)
    assert list(unidos.columns) == list(historia.columns)
    assert set(unidos["periodo_archivo"].cat.categories) == {"2026-09", "2026-10"}
    assert unidos["genero"].isna().sum() == 2


def test_muestrear_limita_filas_y_es_reproducible():
    tabla = pd.DataFrame({"x": range(1000)})
    assert len(monitoreo.muestrear(tabla, n=100)) == 100
    assert monitoreo.muestrear(tabla, n=100).equals(monitoreo.muestrear(tabla, n=100))
    assert len(monitoreo.muestrear(tabla, n=5000)) == 1000


def test_definicion_datos_exige_target_y_prediccion_con_tarea():
    with pytest.raises(ValueError, match="target"):
        monitoreo.definicion_datos(["x"], [], tarea="regresion")
    with pytest.raises(ValueError, match="desconocida"):
        monitoreo.definicion_datos(["x"], [], tarea="otra", target="y", prediccion="p")


def tabla_con_drift(n: int, desplazamiento: float, semilla: int) -> pd.DataFrame:
    """Tabla donde solo `con_drift` y `zona` cambian cuando hay desplazamiento."""
    rng = np.random.default_rng(semilla)
    probabilidades = [0.7, 0.2, 0.1] if desplazamiento == 0 else [0.3, 0.4, 0.3]
    y = rng.poisson(3, n).astype(float)
    return pd.DataFrame(
        {
            "con_drift": rng.normal(desplazamiento, 1, n),
            "sin_drift": rng.normal(0, 1, n),
            "zona": rng.choice(["GDL", "ZPN", "TLQ"], n, p=probabilidades),
            "y": y,
            "prediccion": y + rng.normal(0, 0.5, n),
        }
    )


@pytest.fixture(scope="module")
def reporte_regresion(tmp_path_factory):
    referencia = tabla_con_drift(3000, 0.0, 1)
    actual = tabla_con_drift(3000, 1.0, 2)
    definicion = monitoreo.definicion_datos(
        ["con_drift", "sin_drift"], ["zona"], "regresion", "y", "prediccion"
    )
    carpeta = tmp_path_factory.mktemp("monitoreo")
    return monitoreo.generar_reporte(
        referencia, actual, definicion, "prueba", "regresion", carpeta
    )


def test_generar_reporte_guarda_html_y_json(reporte_regresion):
    _, rutas = reporte_regresion
    assert [ruta.suffix for ruta in rutas] == [".html", ".json"]
    assert all(ruta.stat().st_size > 0 for ruta in rutas)


def test_resumen_drift_detecta_solo_las_columnas_desplazadas(reporte_regresion):
    snapshot, _ = reporte_regresion
    resumen = monitoreo.resumen_drift(snapshot).set_index("columna")
    assert resumen.loc["con_drift", "drift"]
    assert resumen.loc["zona", "drift"]
    assert not resumen.loc["sin_drift", "drift"]
    # Con más de 1000 filas Evidently usa distancias, no p-values
    assert resumen.loc["con_drift", "metodo"].startswith("Wasserstein")
    assert resumen.loc["zona", "metodo"].startswith("Jensen-Shannon")


def test_resumen_drift_coincide_con_el_conteo_de_evidently(reporte_regresion):
    snapshot, _ = reporte_regresion
    resumen = monitoreo.resumen_drift(snapshot)
    proporcion = monitoreo.proporcion_drift(snapshot)
    assert resumen["drift"].sum() == proporcion["columnas_con_drift"]
    assert proporcion["proporcion_drift"] == pytest.approx(resumen["drift"].mean())


def test_resumen_drift_usa_p_values_con_muestras_chicas(tmp_path):
    referencia, actual = tabla_con_drift(300, 0.0, 1), tabla_con_drift(300, 1.0, 2)
    definicion = monitoreo.definicion_datos(["con_drift", "sin_drift"], ["zona"])
    snapshot, _ = monitoreo.generar_reporte(
        referencia, actual, definicion, "chica", carpeta=tmp_path
    )
    resumen = monitoreo.resumen_drift(snapshot).set_index("columna")
    assert "p_value" in resumen.loc["con_drift", "metodo"]
    assert resumen.loc["con_drift", "drift"]
    assert resumen["drift"].sum() == monitoreo.proporcion_drift(snapshot)["columnas_con_drift"]


def test_generar_reporte_de_clasificacion(tmp_path):
    rng = np.random.default_rng(cfg.SEMILLA)
    clases = rng.choice(cfg.P2_CLASES, 600)
    tabla = pd.DataFrame({"x": rng.normal(0, 1, 600), "clase": clases, "prediccion_clase": clases})
    definicion = monitoreo.definicion_datos(
        ["x"], [], "clasificacion", "clase", "prediccion_clase"
    )
    snapshot, _ = monitoreo.generar_reporte(
        tabla, tabla, definicion, "clasificacion", "clasificacion", tmp_path
    )
    assert not monitoreo.resumen_drift(snapshot)["drift"].any()


def test_comparar_pareado_elige_la_version_con_menor_perdida():
    rng = np.random.default_rng(cfg.SEMILLA)
    perdida_a = rng.exponential(1.0, 5000)
    perdida_b = perdida_a * 0.8
    resultado = monitoreo.comparar_pareado(perdida_a, perdida_b, n_bootstrap=300)
    assert resultado["ganador"] == "B"
    assert resultado["ic_superior"] < 0
    assert resultado["cambio_relativo"] == pytest.approx(-0.2)


def test_comparar_pareado_sin_diferencia_real_no_declara_ganador():
    rng = np.random.default_rng(cfg.SEMILLA)
    perdida_a = rng.exponential(1.0, 2000)
    perdida_b = rng.permutation(perdida_a)
    resultado = monitoreo.comparar_pareado(perdida_a, perdida_b, n_bootstrap=300)
    assert resultado["ganador"] == "sin diferencia significativa"
    assert resultado["ic_inferior"] < 0 < resultado["ic_superior"]


def test_comparar_pareado_exige_las_mismas_filas():
    with pytest.raises(ValueError, match="mismas filas"):
        monitoreo.comparar_pareado(np.ones(3), np.ones(4))


def test_bootstrap_por_grupos_amplia_el_intervalo_con_filas_correlacionadas():
    rng = np.random.default_rng(cfg.SEMILLA)
    grupos = np.repeat(np.arange(50), 40)
    # Cada grupo comparte un efecto: las filas no son independientes
    diferencias = rng.normal(0, 1, 50)[grupos] + rng.normal(0, 0.1, len(grupos))
    por_filas = monitoreo.bootstrap_media(diferencias, n_bootstrap=500)
    por_grupos = monitoreo.bootstrap_media(diferencias, grupos, n_bootstrap=500)
    assert por_grupos[1] - por_grupos[0] > 3 * (por_filas[1] - por_filas[0])


def test_comparar_aleatorio_no_parte_unidades_entre_grupos():
    rng = np.random.default_rng(cfg.SEMILLA)
    unidades = np.repeat(np.arange(400), 5)
    perdida_a = rng.exponential(1.0, len(unidades))
    resultado = monitoreo.comparar_aleatorio(perdida_a, perdida_a * 0.7, unidades)
    assert resultado["n_a"] % 5 == 0 and resultado["n_b"] % 5 == 0
    assert resultado["n_a"] + resultado["n_b"] == len(unidades)
    assert resultado["ganador"] == "B"


def test_cargar_version_exige_alias_o_version():
    with pytest.raises(ValueError, match="alias"):
        monitoreo.cargar_version("modelo")
    with pytest.raises(ValueError, match="alias"):
        monitoreo.cargar_version("modelo", alias="champion", version=1)


@pytest.fixture
def mlflow_local(tmp_path, monkeypatch):
    """MLflow en SQLite y manifiestos de prueba, para no depender del servidor ni de los datos."""
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    for nombre, constante in (
        ("fuentes.csv", "RUTA_MANIFIESTO"),
        ("monitoreo.csv", "RUTA_MANIFIESTO_MONITOREO"),
    ):
        ruta = tmp_path / nombre
        ruta.write_text("archivo,periodo,url,registros,sha256\na.csv,2026-10,u,1,abc\n")
        monkeypatch.setattr(cfg, constante, ruta)
    return tmp_path


def test_registrar_en_mlflow_guarda_metricas_etiquetas_y_artefactos(mlflow_local):
    reporte = mlflow_local / "reporte.html"
    reporte.write_text("<html></html>")
    id_run = monitoreo.registrar_en_mlflow(
        "p0-drift-prueba",
        "p0",
        [reporte],
        metricas={"proporcion_drift": 0.25},
        descripcion="Prueba de monitoreo",
    )
    run = mlflow.get_run(id_run)
    assert run.data.metrics["proporcion_drift"] == 0.25
    assert run.data.tags["fase"] == cfg.FASE_MONITOREO
    assert run.data.tags["question"] == "p0"
    assert run.data.tags["equipo"] == "07"
    assert run.data.tags["commit"]
    assert run.data.tags["mlflow.note.content"] == "Prueba de monitoreo"
    assert {"data_version", "data_version_monitoreo"} <= set(run.data.params)
    cliente = mlflow.MlflowClient()
    assert [a.path for a in cliente.list_artifacts(id_run, "monitoreo")] == [
        "monitoreo/reporte.html"
    ]
    assert len(cliente.list_artifacts(id_run, "data")) == 2
    configuracion = {Path(a.path).name for a in cliente.list_artifacts(id_run, "config")}
    assert {ruta.name for ruta in monitoreo.archivos_configuracion()} == configuracion
    assert "config.py" in configuracion


def test_registrar_en_mlflow_liga_datasets_y_el_modelo_evaluado(mlflow_local):
    import mlflow.sklearn
    from sklearn.linear_model import LinearRegression

    datos = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0]})
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.EXPERIMENTO_MLFLOW)
    with mlflow.start_run():
        modelo = mlflow.sklearn.log_model(
            LinearRegression().fit(datos[["x"]], datos["y"]), name="m"
        )
    evaluacion = monitoreo.dataset_mlflow(datos, "p0_evaluacion", mlflow_local / "e.parquet", "y")
    referencia = monitoreo.dataset_mlflow(datos, "p0_referencia", mlflow_local / "r.parquet", "y")

    id_run = monitoreo.registrar_en_mlflow(
        "p0-ab-prueba",
        "p0",
        [],
        datasets={"referencia": referencia, "evaluacion": evaluacion},
        modelo_id=modelo.model_id,
        metricas_por_dataset={"evaluacion": {"champion_mae_evaluacion": 0.5}},
    )
    run = mlflow.get_run(id_run)
    contextos = {
        etiqueta.value
        for entrada in run.inputs.dataset_inputs
        for etiqueta in entrada.tags
        if etiqueta.key == "mlflow.data.context"
    }
    assert contextos == {"referencia", "evaluacion"}
    assert [entrada.model_id for entrada in run.inputs.model_inputs] == [modelo.model_id]
    assert run.data.metrics["champion_mae_evaluacion"] == 0.5
    metricas_modelo = mlflow.get_logged_model(modelo.model_id).metrics
    assert [(m.key, m.dataset_name) for m in metricas_modelo] == [
        ("champion_mae_evaluacion", "p0_evaluacion")
    ]
