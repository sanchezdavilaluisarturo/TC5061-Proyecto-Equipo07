"""Pruebas del monitoreo de P3: perfiles por periodo, control estacional, comparación con el
entrenamiento y reportes de Evidently sobre perfiles de estaciones. Sin red ni servidor de MLflow."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset, monitoreo
from tc5061_proyecto_equipo07.modeling import monitoreo_p3 as mp3

ESTACIONES = [1, 2]
MINIMO = 10


def viajes_en(inicio: str, fin: str, por_estacion: int, semilla: int = 0) -> pd.DataFrame:
    """Viajes que salen de cada estación de ESTACIONES entre `inicio` y `fin`, en horas de servicio."""
    rng = np.random.default_rng(semilla)
    filas = []
    for estacion in ESTACIONES:
        dias = pd.date_range(inicio, fin, freq="D")
        salida = rng.choice(dias, por_estacion) + pd.to_timedelta(
            rng.choice(cfg.P3_HORAS_SERVICIO, por_estacion), unit="h"
        )
        filas.append(
            pd.DataFrame(
                {
                    "inicio_del_viaje": salida,
                    "fin_del_viaje": salida
                    + pd.to_timedelta(rng.integers(5, 20, por_estacion), unit="min"),
                    "origen_id": estacion,
                    "destino_id": rng.choice(ESTACIONES, por_estacion),
                }
            )
        )
    return pd.concat(filas, ignore_index=True)


@pytest.fixture
def viajes() -> pd.DataFrame:
    """Un mes de octubre y otro de marzo en la referencia, una hora en la brecha, valid, actual y
    el mes nuevo; la estación 3 solo tiene dos viajes en el mes nuevo (bajo el mínimo)."""
    pocos = pd.DataFrame(
        {
            "inicio_del_viaje": pd.to_datetime(["2026-10-05 08:00", "2026-10-06 09:00"]),
            "fin_del_viaje": pd.to_datetime(["2026-10-05 08:10", "2026-10-06 09:10"]),
            "origen_id": 3,
            "destino_id": 1,
        }
    )
    brecha = pd.DataFrame(
        {
            "inicio_del_viaje": pd.to_datetime(["2024-09-30 12:00"]),
            "fin_del_viaje": pd.to_datetime(["2024-09-30 12:10"]),
            "origen_id": 1,
            "destino_id": 2,
        }
    )
    return pd.concat(
        [
            viajes_en("2021-10-01", "2021-10-31", 60, 1),
            viajes_en("2022-03-01", "2022-03-31", 40, 2),
            brecha,
            viajes_en("2025-03-01", "2025-03-31", 50, 3),
            viajes_en("2026-03-01", "2026-03-31", 50, 4),
            viajes_en("2026-10-02", "2026-10-28", 50, 5),
            pocos,
        ],
        ignore_index=True,
    )


@pytest.fixture(autouse=True)
def minimos_chicos(monkeypatch):
    """Los mínimos de salidas reales (1000 y 300) dejarían sin estaciones a los datos de juguete."""
    for periodo in mp3.P3_PERIODOS:
        monkeypatch.setitem(mp3.P3_MIN_SALIDAS, periodo, MINIMO)
    monkeypatch.setattr(cfg, "P3_MIN_SALIDAS_EVALUACION", MINIMO)
    monkeypatch.setattr(cfg, "P3_MIN_SALIDAS_TRAIN", MINIMO)


def test_el_perfil_tiene_las_44_variables_del_modelo():
    assert len(mp3.P3_COLUMNAS_PERFIL) == 44
    assert mp3.P3_COLUMNAS_PERFIL[:2] == ["lab_h00", "lab_h05"]
    assert mp3.P3_COLUMNAS_PERFIL[-4:] == mp3.P3_INDICADORES


def test_perfiles_por_periodo_separa_los_periodos_y_descarta_la_brecha(viajes):
    perfiles = mp3.p3_perfiles_por_periodo(viajes)
    assert {"referencia", "valid", "actual", "evaluacion"} <= set(perfiles)
    for perfil in perfiles.values():
        assert list(perfil.columns) == [*mp3.P3_COLUMNAS_PERFIL, "salidas"]
    # La estación 1 sale 60 + 40 veces en la referencia; el viaje de la brecha no cuenta
    assert perfiles["referencia"].loc[1, "salidas"] == 100
    assert perfiles["valid"].loc[1, "salidas"] == 50
    assert perfiles["evaluacion"].loc[1, "salidas"] == 50


def test_perfiles_por_periodo_filtra_estaciones_con_pocas_salidas(viajes):
    perfiles = mp3.p3_perfiles_por_periodo(viajes)
    assert 3 not in perfiles["evaluacion"].index
    assert sorted(perfiles["evaluacion"].index) == ESTACIONES


def test_los_perfiles_horarios_suman_uno_por_bloque(viajes):
    perfil = mp3.p3_perfiles_por_periodo(viajes)["referencia"]
    laborales = [c for c in mp3.P3_COLUMNAS_PERFIL if c.startswith("lab_")]
    fines = [c for c in mp3.P3_COLUMNAS_PERFIL if c.startswith("fin_")]
    assert np.allclose(perfil[laborales].sum(axis=1), 1)
    assert np.allclose(perfil[fines].sum(axis=1), 1)


def test_control_estacional_usa_solo_el_mismo_mes_de_los_anios_previos(viajes):
    perfiles = mp3.p3_perfiles_por_periodo(viajes)
    # Solo hay octubre de 2021 antes del mes nuevo; marzo no cuenta ni tampoco octubre de 2026
    assert [n for n in perfiles if n.startswith("mismo_mes_20")] == ["mismo_mes_2021"]
    assert perfiles["mismo_mes_2021"].loc[1, "salidas"] == 60
    assert perfiles["mismo_mes_referencia"].loc[1, "salidas"] == 60


def test_control_estacional_escala_el_minimo_por_los_anios_de_la_referencia(viajes):
    # Un solo año de referencia: el mínimo es el de un mes (10); con 60 salidas pasan las dos
    assert len(mp3.p3_perfiles_por_periodo(viajes)["mismo_mes_referencia"]) == 2
    # Con 70 de mínimo por mes, ninguna estación llega a 60
    otro = cfg.P3_MIN_SALIDAS_EVALUACION
    try:
        cfg.P3_MIN_SALIDAS_EVALUACION = 70
        perfiles = mp3.p3_perfiles_por_periodo(viajes)
    finally:
        cfg.P3_MIN_SALIDAS_EVALUACION = otro
    assert perfiles["mismo_mes_referencia"].empty


def test_comparar_con_entrenamiento_acepta_perfiles_identicos(viajes, monkeypatch):
    original = mp3.features.p3_construir_caracteristicas
    monkeypatch.setattr(
        mp3.features, "p3_construir_caracteristicas", lambda m: original(m, MINIMO, MINIMO)
    )
    modelado = viajes.copy()
    modelado["duracion_min"] = (
        modelado["fin_del_viaje"] - modelado["inicio_del_viaje"]
    ).dt.total_seconds() / 60
    modelado["particion"] = dataset.p3_asignar_particion(modelado["inicio_del_viaje"])
    modelado = modelado[modelado["inicio_del_viaje"] < pd.Timestamp(cfg.FRONTERA_TEST)]

    perfiles = mp3.p3_perfiles_por_periodo(viajes)
    cuentas = mp3.p3_comparar_con_entrenamiento(perfiles, modelado)
    assert cuentas == {"estaciones_referencia": 2, "estaciones_valid": 2}

    # Si el monitoreo calculara distinto un perfil, la comparación lo detecta
    perfiles["referencia"].loc[1, "prop_circulares"] += 0.2
    with pytest.raises(ValueError, match="difieren"):
        mp3.p3_comparar_con_entrenamiento(perfiles, modelado)


def test_comunes_deja_las_estaciones_presentes_en_ambos_conjuntos():
    referencia = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=[1, 2, 3])
    actual = pd.DataFrame({"x": [4.0, 5.0, 6.0]}, index=[2, 3, 4])
    ref, act = mp3.p3_comunes(referencia, actual)
    assert list(ref.index) == list(act.index) == [2, 3]
    assert ref["x"].tolist() == [2.0, 3.0]
    assert act["x"].tolist() == [4.0, 5.0]


class ModeloFalso:
    """Asigna el grupo 1 a las estaciones con más de la mitad de salidas laborales al mediodía."""

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        assert list(x.columns) == mp3.P3_COLUMNAS_PERFIL
        return (x["lab_h12"] > 0.05).astype(int).to_numpy()


def test_tabla_reporte_agrega_el_grupo_como_cadena(viajes):
    perfil = mp3.p3_perfiles_por_periodo(viajes)["referencia"]
    tabla = mp3.p3_tabla_reporte(perfil, ModeloFalso())
    assert list(tabla.columns) == [*mp3.P3_COLUMNAS_PERFIL, "grupo"]
    assert tabla["grupo"].isin(["0", "1"]).all()
    assert list(tabla.index) == [0, 1]
    assert (tabla[mp3.P3_COLUMNAS_PERFIL].dtypes == "float64").all()


def test_conjuntos_lee_los_perfiles_guardados_con_el_indice_de_estacion(viajes, tmp_path):
    perfiles = mp3.p3_perfiles_por_periodo(viajes)
    ruta = tmp_path / "perfiles.parquet"
    pd.concat([p.assign(periodo=n) for n, p in perfiles.items()]).rename_axis(
        "origen_id"
    ).reset_index().to_parquet(ruta, index=False)

    leidos = mp3.p3_conjuntos(ruta)
    assert set(leidos) == set(perfiles)
    assert sorted(leidos["referencia"].index) == ESTACIONES
    assert "periodo" not in leidos["referencia"].columns
    pd.testing.assert_frame_equal(
        leidos["evaluacion"].sort_index(), perfiles["evaluacion"].sort_index(), check_names=False
    )


def test_conjuntos_pide_generar_los_perfiles_si_faltan(tmp_path):
    with pytest.raises(FileNotFoundError, match="monitoreo-p3-perfiles"):
        mp3.p3_conjuntos(tmp_path / "no_existe.parquet")


def test_el_reporte_de_drift_marca_solo_las_variables_desplazadas(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    referencia = pd.DataFrame(
        rng.normal(0, 1, (n, len(mp3.P3_COLUMNAS_PERFIL))), columns=mp3.P3_COLUMNAS_PERFIL
    )
    actual = pd.DataFrame(
        rng.normal(0, 1, (n, len(mp3.P3_COLUMNAS_PERFIL))), columns=mp3.P3_COLUMNAS_PERFIL
    )
    actual["duracion_mediana_min"] += 3
    referencia["grupo"] = rng.choice(["0", "1"], n)
    actual["grupo"] = rng.choice(["0", "1"], n)

    definicion = monitoreo.definicion_datos(mp3.P3_COLUMNAS_PERFIL, ["grupo"])
    snapshot, rutas = monitoreo.generar_reporte(
        referencia, actual, definicion, "p3_prueba", carpeta=tmp_path
    )
    assert all(ruta.exists() for ruta in rutas)

    resumen = monitoreo.resumen_drift(snapshot)
    assert len(resumen) == len(mp3.P3_COLUMNAS_PERFIL) + 1
    assert resumen.loc[resumen["drift"], "columna"].tolist() == ["duracion_mediana_min"]
    # Con 300 estaciones por conjunto Evidently usa p-values (K-S), no distancias
    assert resumen.set_index("columna").loc["duracion_mediana_min", "metodo"] == "K-S p_value"
    assert monitoreo.proporcion_drift(snapshot)["columnas_con_drift"] == 1


def test_perfil_del_challenger_junta_train_valid_y_test_sin_brecha_ni_mes_nuevo(viajes):
    perfil = mp3.p3_perfiles_por_periodo(viajes)["entrenamiento_challenger"]
    # Estación 1: 100 salidas en la referencia, 50 en valid y 50 en test; el viaje de la brecha y
    # los 50 del mes nuevo no cuentan
    assert perfil.loc[1, "salidas"] == 200
    assert sorted(perfil.index) == ESTACIONES


def champion_de_juguete(semilla: int = 0):
    rng = np.random.default_rng(semilla)
    datos = pd.DataFrame(rng.normal(0, 1, (80, 44)), columns=mp3.P3_COLUMNAS_PERFIL)
    modelo = Pipeline(
        [
            ("escalador", StandardScaler()),
            ("pca", PCA(n_components=3, random_state=42)),
            ("kmeans", KMeans(n_clusters=2, n_init=10, random_state=42)),
        ]
    )
    return modelo.fit(datos), datos


def test_reentrenar_conserva_los_hiperparametros_y_cambia_solo_los_datos():
    champion, _ = champion_de_juguete(0)
    _, otros_datos = champion_de_juguete(1)
    challenger = mp3.p3_reentrenar(champion, otros_datos)
    assert challenger is not champion
    assert mp3.p3_hiperparametros(challenger) == mp3.p3_hiperparametros(champion)
    assert not np.allclose(
        challenger.named_steps["escalador"].mean_, champion.named_steps["escalador"].mean_
    )


def test_resumen_del_challenger_con_los_mismos_datos_no_cambia_los_grupos():
    champion, datos = champion_de_juguete(0)
    challenger = mp3.p3_reentrenar(champion, datos)
    resumen = mp3.p3_resumen_challenger(champion, challenger, datos)
    assert resumen["ari_champion_challenger"] == pytest.approx(1.0)
    assert resumen["silueta_challenger"] == pytest.approx(resumen["silueta_champion"])
    assert resumen["estaciones"] == len(datos)


def datos_con_dos_grupos(n: int = 150, semilla: int = 0) -> pd.DataFrame:
    """Estaciones de juguete con dos grupos bien separados en las 44 variables."""
    rng = np.random.default_rng(semilla)
    centros = np.repeat([-3.0, 3.0], n)[:, None]
    datos = rng.normal(0, 1, (2 * n, 44)) + centros
    return pd.DataFrame(datos, columns=mp3.P3_COLUMNAS_PERFIL, index=np.arange(2 * n))


@pytest.fixture
def pocos_remuestreos(monkeypatch):
    original = mp3.p3_bootstrap_silueta
    monkeypatch.setattr(
        mp3, "p3_bootstrap_silueta", lambda *args, **kwargs: original(*args, n_bootstrap=200)
    )


def test_decidir_ab_solo_cambia_de_version_con_mejora_significativa():
    assert mp3.p3_decidir_ab(0.001, 0.01, 0.05, 0.05)[0] == "challenger"
    # Significativa en Wilcoxon, pero el intervalo incluye 0: el champion se queda
    assert mp3.p3_decidir_ab(0.001, -0.01, 0.05, 0.05)[0] == "champion"
    # Intervalo sobre 0 pero p-value no significativo: el champion se queda
    assert mp3.p3_decidir_ab(0.20, 0.01, 0.05, 0.05)[0] == "champion"
    vigente, motivo = mp3.p3_decidir_ab(0.001, -0.05, -0.01, 0.05)
    assert vigente == "champion" and "peor" in motivo


def test_ab_con_versiones_identicas_deja_al_champion(pocos_remuestreos):
    champion, datos = champion_de_juguete(0)
    resultados, por_estacion = mp3.p3_ab(champion, mp3.p3_reentrenar(champion, datos), datos)
    assert resultados["diferencia"] == pytest.approx(0)
    assert resultados["p_wilcoxon"] == 1.0
    assert resultados["version_vigente"] == "champion"
    assert resultados["ari_champion_challenger"] == pytest.approx(1.0)
    assert (por_estacion["diferencia"].abs() < 1e-12).all()


class ChampionAleatorio:
    """Champion de juguete que asigna el grupo al azar, con el escalador de las estaciones."""

    def __init__(self, datos: pd.DataFrame):
        self.named_steps = {"escalador": StandardScaler().fit(datos)}

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.random.default_rng(3).integers(0, 2, len(x))


def test_ab_reemplaza_al_champion_si_el_challenger_separa_mejor(pocos_remuestreos):
    evaluacion = datos_con_dos_grupos()
    champion = ChampionAleatorio(evaluacion)
    challenger = mp3.p3_reentrenar(champion_de_juguete(0)[0], evaluacion)
    resultados, _ = mp3.p3_ab(champion, challenger, evaluacion)
    assert resultados["silueta_challenger"] > resultados["silueta_champion"]
    assert resultados["ic_inferior"] > 0
    assert resultados["version_vigente"] == "challenger"


def test_ab_deja_al_champion_si_es_mejor_que_el_challenger(pocos_remuestreos):
    evaluacion = datos_con_dos_grupos()
    champion = mp3.p3_reentrenar(champion_de_juguete(0)[0], evaluacion)
    challenger = ChampionAleatorio(evaluacion)
    resultados, _ = mp3.p3_ab(champion, challenger, evaluacion)
    assert resultados["diferencia"] < 0
    assert resultados["version_vigente"] == "champion"
    assert "peor" in resultados["motivo"]


def test_ab_compara_las_mismas_estaciones_y_guarda_la_silueta_por_estacion(pocos_remuestreos):
    champion, datos = champion_de_juguete(0)
    resultados, por_estacion = mp3.p3_ab(champion, mp3.p3_reentrenar(champion, datos), datos)
    assert list(por_estacion.index) == list(datos.index)
    assert resultados["estaciones"] == len(datos)
    assert resultados["silueta_champion"] == pytest.approx(por_estacion["silueta_champion"].mean())
    # Reparto aleatorio: cada estación va a una sola versión
    assert resultados["aleatorio_n_champion"] + resultados["aleatorio_n_challenger"] == len(datos)


def test_bootstrap_de_siluetas_es_reproducible_y_cero_si_los_grupos_son_iguales():
    champion, datos = champion_de_juguete(0)
    z = mp3.p3_espacio_comun(champion, datos)
    grupos = champion.predict(datos)
    primero = mp3.p3_bootstrap_silueta(z, grupos, grupos, n_bootstrap=50)
    assert primero == mp3.p3_bootstrap_silueta(z, grupos, grupos, n_bootstrap=50)
    assert primero == pytest.approx((0.0, 0.0))


def test_silueta_por_conjunto_regresa_una_fila_por_conjunto():
    champion, datos = champion_de_juguete(0)
    datos = datos.assign(salidas=100.0)
    tabla = mp3.p3_silueta_por_conjunto(champion, {"a": datos, "b": datos.iloc[:40]})
    assert tabla["conjunto"].tolist() == ["a", "b"]
    assert tabla["estaciones"].tolist() == [80, 40]
    assert (tabla["salidas_mediana"] == 100.0).all()


@pytest.fixture
def registro_local(tmp_path, monkeypatch):
    """MLflow en SQLite, manifiestos y reportes de juguete: `registrar-*` sin servidor ni datos reales."""
    import mlflow

    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    for nombre, constante in (
        ("fuentes.csv", "RUTA_MANIFIESTO"),
        ("monitoreo.csv", "RUTA_MANIFIESTO_MONITOREO"),
    ):
        ruta = tmp_path / nombre
        ruta.write_text("archivo,periodo,url,registros,sha256\na.csv,2026-10,u,1,abc\n")
        monkeypatch.setattr(cfg, constante, ruta)
    reportes = tmp_path / "reportes"
    reportes.mkdir()
    monkeypatch.setattr(cfg, "MONITOREO_DIR", reportes)
    monkeypatch.setattr(cfg, "P3_RUTA_PERFILES_MONITOREO", tmp_path / "perfiles.parquet")
    perfil = pd.DataFrame(
        np.random.default_rng(0).random((6, len(mp3.P3_COLUMNAS_PERFIL))),
        columns=mp3.P3_COLUMNAS_PERFIL,
        index=pd.Index(range(6), name="origen_id"),
    ).assign(salidas=100.0)
    conjuntos = {n: perfil for n in ("referencia", "actual", "evaluacion")}
    monkeypatch.setattr(mp3, "p3_conjuntos", lambda: conjuntos)
    monkeypatch.setattr(mp3, "p3_cargar_champion", lambda: (None, 4, {}, None))
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    return reportes


def test_metricas_de_drift_incluyen_cada_comparacion_y_cada_octubre():
    conteo = pd.DataFrame(
        {
            "comparacion": ["train_test_todas", "estacional_comunes"],
            "columnas_con_drift": [28, 29],
            "proporcion_drift": [0.62, 0.64],
        }
    )
    por_anio = pd.DataFrame({"anio": [2022, 2023], "proporcion_drift": [0.44, 0.49]})
    metricas = mp3.p3_metricas_drift(conteo, por_anio)
    assert metricas["train_test_todas_proporcion_drift"] == 0.62
    assert metricas["estacional_comunes_columnas_con_drift"] == 29.0
    assert metricas["octubre_2023_proporcion_drift"] == 0.49
    assert "octubre_2022_proporcion_drift" in mp3.p3_metricas_drift(conteo, por_anio)
    assert not any(k.startswith("octubre_") for k in mp3.p3_metricas_drift(conteo, None))


def test_registrar_drift_guarda_metricas_etiquetas_y_reportes(registro_local):
    import mlflow

    pd.DataFrame(
        {
            "comparacion": ["train_test_todas"],
            "estaciones_referencia": [362],
            "estaciones_actual": [457],
            "columnas_con_drift": [28],
            "proporcion_drift": [0.62],
            "umbral_proporcion": [0.5],
        }
    ).to_csv(registro_local / "p3_drift_conteo.csv", index=False)
    (registro_local / "p3_drift_train_test_todas.html").write_text("<html></html>")
    mp3.registrar_drift()
    (run,) = mlflow.search_runs(experiment_names=[cfg.EXPERIMENTO_MLFLOW], output_format="list")
    assert run.data.tags["rol"] == "drift" and run.data.tags["fase"] == cfg.FASE_MONITOREO
    assert run.data.metrics["train_test_todas_proporcion_drift"] == 0.62
    archivos = {a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id, "monitoreo")}
    assert archivos == {
        "monitoreo/p3_drift_conteo.csv",
        "monitoreo/p3_drift_train_test_todas.html",
    }


def test_registrar_drift_sin_el_conteo_pide_correr_drift(registro_local):
    with pytest.raises(FileNotFoundError, match="monitoreo-p3-drift"):
        mp3.registrar_drift()


def escribir_resultado_ab(carpeta, champion_version: int = 4):
    filas = {
        "champion_version": champion_version,
        "challenger_version": 5,
        "estaciones": 469,
        "silueta_champion": 0.0728,
        "silueta_challenger": 0.0710,
        "diferencia": -0.0018,
        "p_wilcoxon": 0.83,
        "alfa": 0.05,
        "version_vigente": "champion",
        "motivo": "sin diferencia significativa: el champion se queda",
    }
    pd.Series(filas).to_frame("valor").rename_axis("metrica").to_csv(
        carpeta / "p3_ab_resultado.csv"
    )
    pd.DataFrame({"silueta_champion": [0.1]}).to_csv(carpeta / "p3_ab_estaciones.csv")


def test_registrar_ab_guarda_la_decision_y_las_metricas(registro_local):
    import mlflow

    escribir_resultado_ab(registro_local)
    mp3.registrar_ab()
    (run,) = mlflow.search_runs(experiment_names=[cfg.EXPERIMENTO_MLFLOW], output_format="list")
    assert run.data.tags["rol"] == "ab" and run.data.tags["decision"] == "champion"
    assert run.data.metrics["p_wilcoxon"] == 0.83
    assert run.data.metrics["champion_silueta_evaluacion"] == 0.0728
    assert run.data.params["challenger_version"] == "5"
    assert "champion_version" not in run.data.metrics


def test_registrar_ab_falla_si_el_champion_cambio_desde_el_calculo(registro_local):
    escribir_resultado_ab(registro_local, champion_version=3)
    with pytest.raises(ValueError, match="volver a correr"):
        mp3.registrar_ab()


# ---------------------------------------------------------------------------------------------
# Challenger por ventana de entrenamiento
# ---------------------------------------------------------------------------------------------


def test_inicio_de_ventana_por_inicio_fijo_o_por_meses_hacia_atras():
    assert mp3.p3_inicio_ventana("completa", pd.Period("2026-09", freq="M")) is None
    assert mp3.p3_inicio_ventana("sin_pandemia", pd.Period("2026-09", freq="M")) == pd.Timestamp(
        "2022-01-01"
    )
    # 36 meses terminando en el último mes, incluyéndolo: 2023-10 a 2026-09 y 2022-10 a 2025-09
    assert mp3.p3_inicio_ventana(
        "ultimos_3_anios", pd.Period("2026-09", freq="M")
    ) == pd.Timestamp("2023-10-01")
    assert mp3.p3_inicio_ventana(
        "ultimos_3_anios", pd.Period("2025-09", freq="M")
    ) == pd.Timestamp("2022-10-01")


def test_una_ventana_desconocida_se_rechaza():
    with pytest.raises(ValueError, match="Ventana desconocida"):
        mp3.p3_inicio_ventana("inventada", pd.Period("2026-09", freq="M"))
    with pytest.raises(ValueError, match="Ventana desconocida"):
        mp3.p3_clave_entrenamiento("inventada")


def test_la_ventana_completa_es_el_perfil_del_challenger_actual():
    assert mp3.p3_clave_entrenamiento("completa") == "entrenamiento_challenger"
    assert mp3.p3_clave_entrenamiento("sin_pandemia") == "entrenamiento_challenger_sin_pandemia"


def test_las_ventanas_finales_recortan_el_inicio_del_entrenamiento(viajes):
    perfiles = mp3.p3_perfiles_por_periodo(viajes)
    # Estación 1: 60 salidas en 2021-10, 40 en 2022-03, 50 en valid y 50 en test
    assert perfiles["entrenamiento_challenger"].loc[1, "salidas"] == 200
    assert perfiles["entrenamiento_challenger_sin_pandemia"].loc[1, "salidas"] == 140
    assert perfiles["entrenamiento_challenger_ultimos_3_anios"].loc[1, "salidas"] == 100


def test_el_backtest_solo_usa_datos_anteriores_a_la_frontera_de_test(viajes):
    perfiles = mp3.p3_perfiles_por_periodo(viajes)
    # Sin el test (2026-03), sin la brecha y sin el mes nuevo: 100 de la referencia + 50 de valid
    assert perfiles["backtest_completa"].loc[1, "salidas"] == 150
    assert perfiles["backtest_sin_pandemia"].loc[1, "salidas"] == 90
    # 2022-10 a 2025-09: solo valid
    assert perfiles["backtest_ultimos_3_anios"].loc[1, "salidas"] == 50
    for ventana in cfg.P3_VENTANAS_CHALLENGER:
        assert (
            perfiles[f"backtest_{ventana}"].loc[1, "salidas"]
            <= perfiles["backtest_completa"].loc[1, "salidas"]
        )


def test_estabilidad_es_uno_si_el_modelo_asigna_igual_en_los_dos_conjuntos():
    champion, datos = champion_de_juguete(0)
    assert mp3.p3_estabilidad(champion, datos, datos) == pytest.approx(1.0)


def test_estabilidad_usa_solo_las_estaciones_comunes():
    champion, datos = champion_de_juguete(0)
    evaluacion = datos.iloc[:30]
    assert mp3.p3_estabilidad(champion, datos, evaluacion) == pytest.approx(1.0)


def test_evaluar_ventana_con_los_mismos_datos_no_es_elegible(pocos_remuestreos):
    champion, datos = champion_de_juguete(0)
    fila = mp3.p3_evaluar_ventana(champion, "sin_pandemia", datos, datos, ari_champion=1.0)
    assert fila["ventana"] == "sin_pandemia"
    assert fila["estaciones_entrenamiento"] == len(datos)
    assert fila["diferencia"] == pytest.approx(0)
    assert fila["mejora_significativa"] is False
    assert fila["elegible"] is False
    # Bonferroni: tres ventanas contra el mismo champion
    assert fila["alfa"] == pytest.approx(cfg.MONITOREO_ALFA / len(cfg.P3_VENTANAS_CHALLENGER))


def test_una_restriccion_incumplida_descarta_la_ventana_aunque_mejore(
    pocos_remuestreos, monkeypatch
):
    evaluacion = datos_con_dos_grupos()
    champion = ChampionAleatorio(evaluacion)
    entrenamiento = evaluacion
    # El champion aleatorio no se puede clonar: el modelo de la ventana se arma antes
    modelo_ventana = mp3.p3_reentrenar(champion_de_juguete(0)[0], evaluacion)
    monkeypatch.setattr(mp3, "p3_reentrenar", lambda campeon, perfil: modelo_ventana)
    fila_libre = mp3.p3_evaluar_ventana(champion, "completa", entrenamiento, evaluacion, 0.0)
    assert fila_libre["mejora_significativa"] is True and fila_libre["elegible"] is True
    # Con una estabilidad mínima que el challenger no alcanza, deja de ser elegible
    fila_estricta = mp3.p3_evaluar_ventana(champion, "completa", entrenamiento, evaluacion, 2.0)
    assert fila_estricta["mejora_significativa"] is True
    assert fila_estricta["cumple_estabilidad"] is False and fila_estricta["elegible"] is False
    # Con un tamaño mínimo imposible tampoco
    monkeypatch.setattr(cfg, "P3_TAM_MIN_ACEPTABLE", len(entrenamiento) + 1)
    fila_chica = mp3.p3_evaluar_ventana(champion, "completa", entrenamiento, evaluacion, 0.0)
    assert fila_chica["cumple_tamano"] is False and fila_chica["elegible"] is False


def test_elegir_ventana_toma_la_elegible_de_mayor_diferencia_o_ninguna():
    tabla = pd.DataFrame(
        {
            "ventana": ["completa", "sin_pandemia", "ultimos_3_anios"],
            "diferencia": [0.001, 0.004, 0.009],
            "elegible": [True, True, False],
        }
    )
    assert mp3.p3_elegir_ventana(tabla) == "sin_pandemia"
    assert mp3.p3_elegir_ventana(tabla.assign(elegible=False)) is None


def test_registrar_una_ventana_distinta_de_completa_no_esta_soportado():
    import typer

    with pytest.raises(typer.BadParameter, match="completa"):
        mp3.challenger(registrar=True, ventana="sin_pandemia")


def test_ab_con_ventana_guarda_aparte_y_no_toca_el_ab_de_v5(
    tmp_path, monkeypatch, pocos_remuestreos
):
    champion, datos = champion_de_juguete(0)
    monkeypatch.setattr(cfg, "MONITOREO_DIR", tmp_path)
    conjuntos = {"evaluacion": datos, "entrenamiento_challenger_sin_pandemia": datos}
    monkeypatch.setattr(mp3, "p3_conjuntos", lambda: conjuntos)
    monkeypatch.setattr(mp3, "p3_cargar_champion", lambda: (champion, 4, {}, None))
    mp3.ab(ventana="sin_pandemia")
    resultado = pd.read_csv(tmp_path / "p3_ab_sin_pandemia_resultado.csv", index_col="metrica")
    assert resultado.loc["ventana", "valor"] == "sin_pandemia"
    assert float(resultado.loc["alfa", "valor"]) == pytest.approx(
        cfg.MONITOREO_ALFA / len(cfg.P3_VENTANAS_CHALLENGER)
    )
    assert (tmp_path / "p3_ab_sin_pandemia_estaciones.csv").exists()
    assert not (tmp_path / "p3_ab_resultado.csv").exists()


# ---------------------------------------------------------------------------------------------
# Challenger por variables
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def pocos_remuestreos_ari(monkeypatch):
    original = mp3.p3_ari_bootstrap
    monkeypatch.setattr(
        mp3, "p3_ari_bootstrap", lambda *args, **kwargs: original(*args, n_bootstrap=200, **kwargs)
    )


def perfil_con_picos() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    perfil = pd.DataFrame(
        rng.random((40, len(mp3.P3_COLUMNAS_PERFIL))), columns=mp3.P3_COLUMNAS_PERFIL
    )
    perfil.index.name = "origen_id"
    return perfil


def test_hellinger_aplica_raiz_a_las_horas_y_deja_los_indicadores():
    perfil = perfil_con_picos()
    salida = mp3.p3_raiz_horarias(perfil)
    horarias = [c for c in perfil.columns if c not in mp3.P3_INDICADORES]
    assert np.allclose(salida[horarias], np.sqrt(perfil[horarias]))
    assert np.allclose(salida[mp3.P3_INDICADORES], perfil[mp3.P3_INDICADORES])
    assert list(salida.columns) == list(perfil.columns)
    # No modifica la entrada
    assert not np.allclose(perfil[horarias], salida[horarias])


def test_hellinger_rechaza_proporciones_negativas():
    perfil = perfil_con_picos()
    perfil.loc[0, "lab_h08"] = -0.1
    with pytest.raises(ValueError, match="negativas"):
        mp3.p3_raiz_horarias(perfil)


def test_quitar_horas_vacias_quita_exactamente_las_de_config():
    salida = mp3.p3_quitar_horas_vacias(perfil_con_picos())
    assert set(perfil_con_picos().columns) - set(salida.columns) == set(
        cfg.P3_COLUMNAS_HORAS_VACIAS
    )


def test_la_regla_de_horas_vacias_usa_el_umbral_sobre_la_media():
    perfil = perfil_con_picos()
    perfil["lab_h00"] = 0.001
    perfil["fin_h05"] = 0.0099
    perfil["lab_h06"] = 0.0101
    vacias = mp3.p3_horas_vacias(perfil)
    assert "lab_h00" in vacias and "fin_h05" in vacias and "lab_h06" not in vacias
    # Los indicadores nunca se consideran horas, aunque su media sea baja
    perfil["prop_circulares"] = 0.0
    assert "prop_circulares" not in mp3.p3_horas_vacias(perfil)


@pytest.mark.parametrize("variante", ["identica", *cfg.P3_VARIANTES_VARIABLES])
def test_cada_variante_conserva_pca_y_kmeans_y_predice_con_las_44_columnas(variante):
    champion, datos = champion_de_juguete(0)
    datos = datos.abs()  # son proporciones: la raíz de Hellinger exige valores no negativos
    modelo = mp3.p3_construir_variante(champion, variante).fit(datos)

    def hiper(m, prefijo):
        return {k: v for k, v in mp3.p3_hiperparametros(m).items() if k.startswith(prefijo)}

    for prefijo in ("pca__", "kmeans__"):
        assert hiper(modelo, prefijo) == hiper(champion, prefijo)
    assert len(modelo.predict(datos)) == len(datos)


def test_robusto_cambia_el_escalador_y_las_demas_agregan_pasos_previos():
    champion, _ = champion_de_juguete(0)
    from sklearn.preprocessing import RobustScaler

    assert isinstance(
        mp3.p3_construir_variante(champion, "robusto").named_steps["escalador"], RobustScaler
    )
    assert next(n for n, _ in mp3.p3_construir_variante(champion, "hellinger").steps) == "raiz"
    assert next(n for n, _ in mp3.p3_construir_variante(champion, "sin_horas_vacias").steps) == (
        "sin_vacias"
    )
    assert [n for n, _ in mp3.p3_construir_variante(champion, "combinada").steps][:2] == [
        "raiz",
        "sin_vacias",
    ]
    # El champion no se modifica
    assert [n for n, _ in champion.steps] == ["escalador", "pca", "kmeans"]


def test_una_variante_desconocida_se_rechaza():
    champion, _ = champion_de_juguete(0)
    with pytest.raises(ValueError, match="Variante desconocida"):
        mp3.p3_construir_variante(champion, "inventada")


def test_el_alfa_de_variantes_corrige_por_el_numero_de_variantes():
    assert mp3.p3_alfa_variantes() == pytest.approx(0.05 / 4)


def test_bootstrap_de_ari_es_reproducible_y_cero_si_los_modelos_son_iguales():
    g1, g2 = np.array([0, 1] * 20), np.array([0, 1, 1, 0] * 10)
    primero = mp3.p3_ari_bootstrap(g1, g2, g1, g2, alfa=0.05, n_bootstrap=100)
    assert primero == (0.0, 0.0)
    otro_a, otro_b = np.array([0, 0, 1, 1] * 10), np.array([0, 1, 0, 1] * 10)
    uno = mp3.p3_ari_bootstrap(g1, g2, otro_a, otro_b, alfa=0.05, n_bootstrap=100)
    assert uno == mp3.p3_ari_bootstrap(g1, g2, otro_a, otro_b, alfa=0.05, n_bootstrap=100)


def test_variante_identica_da_diferencia_cero_y_no_es_elegible(
    pocos_remuestreos, pocos_remuestreos_ari
):
    champion, datos = champion_de_juguete(0)
    fila = mp3.p3_evaluar_variante(champion, "identica", datos, datos, datos, datos, alfa=0.0125)
    assert fila["diferencia_ari"] == pytest.approx(0)
    assert fila["ic_inferior_ari"] == pytest.approx(0)
    assert fila["estabilidad_mayor"] is False
    assert fila["elegible"] is False


class ChampionInestable(ChampionAleatorio):
    """Champion de juguete cuyos grupos cambian en cada llamada: su estabilidad entre periodos es nula."""

    def __init__(self, datos: pd.DataFrame):
        super().__init__(datos)
        self.llamadas = 0

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        self.llamadas += 1
        return np.random.default_rng(self.llamadas).integers(0, 2, len(x))


def test_variante_estable_que_cumple_las_restricciones_es_elegible(
    pocos_remuestreos, pocos_remuestreos_ari, monkeypatch
):
    datos = datos_con_dos_grupos()
    champion = ChampionInestable(datos)
    variante = champion_de_juguete(0)[0]
    monkeypatch.setattr(mp3, "p3_construir_variante", lambda campeon, v: variante)
    fila = mp3.p3_evaluar_variante(champion, "hellinger", datos, datos, datos, datos, alfa=0.0125)
    assert fila["ari_variante"] == pytest.approx(1.0)
    assert fila["diferencia_ari"] > 0.5 and fila["ic_inferior_ari"] > 0
    assert fila["cumple_piso_silueta"] is True and fila["cumple_tamano"] is True
    assert fila["elegible"] is True


def test_la_variante_no_es_elegible_si_su_silueta_es_significativamente_peor(
    pocos_remuestreos, pocos_remuestreos_ari, monkeypatch
):
    datos = datos_con_dos_grupos()
    champion = champion_de_juguete(0)[0]
    # La variante asigna grupos al azar: estable por construcción, pero separa peor que el champion
    azar = ChampionAleatorio(datos)

    class Azar:
        def fit(self, x):
            return self

        def predict(self, x):
            return azar.predict(x)

    monkeypatch.setattr(mp3, "p3_construir_variante", lambda campeon, v: Azar())
    fila = mp3.p3_evaluar_variante(champion, "robusto", datos, datos, datos, datos, alfa=0.0125)
    assert fila["cumple_piso_silueta"] is False
    assert fila["elegible"] is False


def test_el_tamano_minimo_de_grupo_descarta_la_variante(
    pocos_remuestreos, pocos_remuestreos_ari, monkeypatch
):
    datos = datos_con_dos_grupos()
    champion = ChampionInestable(datos)
    variante = champion_de_juguete(0)[0]
    monkeypatch.setattr(mp3, "p3_construir_variante", lambda campeon, v: variante)
    monkeypatch.setattr(cfg, "P3_TAM_MIN_ACEPTABLE", len(datos) + 1)
    fila = mp3.p3_evaluar_variante(champion, "hellinger", datos, datos, datos, datos, alfa=0.0125)
    assert fila["cumple_tamano"] is False and fila["elegible"] is False


def test_elegir_variante_toma_la_de_mayor_diferencia_de_estabilidad_o_ninguna():
    tabla = pd.DataFrame(
        {
            "variante": ["hellinger", "robusto", "combinada"],
            "diferencia_ari": [0.05, 0.20, 0.30],
            "elegible": [True, True, False],
        }
    )
    assert mp3.p3_elegir_variante(tabla) == "robusto"
    assert mp3.p3_elegir_variante(tabla.assign(elegible=False)) is None


def conjuntos_de_variables(datos: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {n: datos for n in ("referencia", "mismo_mes_2024", "mismo_mes_2025", "evaluacion")}


def test_backtest_de_variables_escribe_la_tabla_sin_usar_octubre_2026(
    tmp_path, monkeypatch, pocos_remuestreos, pocos_remuestreos_ari
):
    champion, datos = champion_de_juguete(0)
    conjuntos = conjuntos_de_variables(datos.abs())  # proporciones: no negativas
    monkeypatch.setattr(cfg, "MONITOREO_DIR", tmp_path)
    conjuntos["evaluacion"] = None  # si el backtest lo tocara, fallaría
    monkeypatch.setattr(mp3, "p3_conjuntos", lambda: conjuntos)
    monkeypatch.setattr(mp3, "p3_cargar_champion", lambda: (champion, 4, {}, None))
    monkeypatch.setattr(mp3, "p3_horas_vacias", lambda ref: cfg.P3_COLUMNAS_HORAS_VACIAS)
    mp3.backtest_variables()
    tabla = pd.read_csv(tmp_path / "p3_backtest_variables.csv")
    assert list(tabla["variante"]) == cfg.P3_VARIANTES_VARIABLES
    assert tabla["alfa"].eq(0.05 / 4).all()
    assert tabla["elegida"].sum() <= 1


def test_el_backtest_se_detiene_si_la_regla_de_horas_vacias_cambia(monkeypatch):
    datos = champion_de_juguete(0)[1]
    monkeypatch.setattr(mp3, "p3_conjuntos", lambda: conjuntos_de_variables(datos))
    monkeypatch.setattr(mp3, "p3_horas_vacias", lambda ref: ["lab_h00"])
    with pytest.raises(RuntimeError, match="horas vacías"):
        mp3.backtest_variables()


def test_ab_variable_usa_octubre_2025_a_2026_y_guarda_aparte(
    tmp_path, monkeypatch, pocos_remuestreos, pocos_remuestreos_ari
):
    champion, datos = champion_de_juguete(0)
    conjuntos = conjuntos_de_variables(datos.abs())  # proporciones: no negativas
    monkeypatch.setattr(cfg, "MONITOREO_DIR", tmp_path)
    conjuntos["mismo_mes_2024"] = None  # la confirmación no usa octubre 2024
    monkeypatch.setattr(mp3, "p3_conjuntos", lambda: conjuntos)
    monkeypatch.setattr(mp3, "p3_cargar_champion", lambda: (champion, 4, {}, None))
    mp3.ab_variable(variante="robusto")
    resultado = pd.read_csv(tmp_path / "p3_ab_variable_robusto_resultado.csv", index_col="metrica")
    assert resultado.loc["variante", "valor"] == "robusto"
    assert float(resultado.loc["alfa", "valor"]) == pytest.approx(0.05 / 4)
    assert not (tmp_path / "p3_backtest_variables.csv").exists()
