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
