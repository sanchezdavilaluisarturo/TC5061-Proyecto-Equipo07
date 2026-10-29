"""Pruebas de modeling/train.py y modeling/predict.py (con datos sintéticos, sin red)."""

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset
from tc5061_proyecto_equipo07.modeling import predict, train

COLUMNAS = (
    [f"lab_h{h:02d}" for h in cfg.P3_HORAS_SERVICIO]
    + [f"fin_h{h:02d}" for h in cfg.P3_HORAS_SERVICIO]
    + ["log_viajes_por_dia", "neto_sobre_salidas", "prop_circulares", "duracion_mediana_min"]
)


def p3_x_con_tres_grupos(n_por_grupo: int = 30, semilla: int = 0) -> pd.DataFrame:
    """Estaciones sintéticas en tres nubes bien separadas."""
    rng = np.random.default_rng(semilla)
    centros = [0.0, 5.0, 10.0]
    filas = [rng.normal(centro, 0.3, size=(n_por_grupo, len(COLUMNAS))) for centro in centros]
    return pd.DataFrame(np.vstack(filas), columns=COLUMNAS)


def p3_limpios_sinteticos(n_viajes: int = 20000, semilla: int = 0) -> pd.DataFrame:
    """Viajes entre 30 estaciones, en train y en valid, con la estructura del dataset de modelado."""
    rng = np.random.default_rng(semilla)
    inicio_train = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 600 * 24 * 60, n_viajes // 2), unit="min"
    )
    inicio_valid = pd.Timestamp("2024-10-02") + pd.to_timedelta(
        rng.integers(0, 300 * 24 * 60, n_viajes // 2), unit="min"
    )
    inicio = pd.Series(inicio_train.append(inicio_valid))
    # Perfil de mañana para las estaciones 1 a 15 y de tarde para las demás
    origen = rng.integers(1, 31, n_viajes)
    hora = np.where(origen <= 15, rng.choice([7, 8, 9], n_viajes), rng.choice([17, 18, 19], n_viajes))
    inicio = inicio.dt.floor("D") + pd.to_timedelta(hora, unit="h") + pd.to_timedelta(
        rng.integers(0, 59, n_viajes), unit="min"
    )
    limpios = pd.DataFrame(
        {
            "viaje_id": np.arange(n_viajes),
            "inicio_del_viaje": inicio.to_numpy(),
            "origen_id": origen,
            "destino_id": rng.integers(1, 31, n_viajes),
        }
    )
    limpios["fin_del_viaje"] = limpios["inicio_del_viaje"] + pd.Timedelta(minutes=10)
    limpios["duracion_min"] = 10.0
    limpios["periodo_archivo"] = limpios["inicio_del_viaje"].dt.strftime("%Y-%m")
    limpios["particion"] = dataset.p3_asignar_particion(limpios["inicio_del_viaje"])
    return limpios


def test_p3_pesos_por_bloque_igualan_la_varianza_de_cada_bloque():
    pesos = pd.Series(train.p3_pesos_por_bloque(COLUMNAS), index=COLUMNAS)
    n = len(cfg.P3_HORAS_SERVICIO)
    assert (pesos[:n] ** 2).sum() == pytest.approx(1.0)  # bloque laboral
    assert (pesos[n : 2 * n] ** 2).sum() == pytest.approx(1.0)  # fin de semana
    assert (pesos[2 * n :] ** 2).sum() == pytest.approx(1.0)  # indicadores


def test_p3_construir_modelo_agrega_pesos_solo_si_se_pide():
    sin_pesos = train.p3_construir_modelo(COLUMNAS, 5, False, 3, 42)
    con_pesos = train.p3_construir_modelo(COLUMNAS, 5, True, 3, 42)
    assert [n for n, _ in sin_pesos.steps] == ["escalador", "pca", "kmeans"]
    assert [n for n, _ in con_pesos.steps] == ["escalador", "pesos", "pca", "kmeans"]


def test_p3_aplicar_pesos_vive_en_el_modulo_y_no_en_main():
    assert train.p3_aplicar_pesos.__module__ == "tc5061_proyecto_equipo07.modeling.train"


def test_p3_modelo_ponderado_se_serializa_con_skops_sin_depender_de_main():
    skops_io = pytest.importorskip("skops.io")
    x = p3_x_con_tres_grupos()
    modelo = train.p3_construir_modelo(COLUMNAS, 5, True, 3, 42).fit(x)
    contenido = skops_io.dumps(modelo)
    no_confiables = skops_io.get_untrusted_types(data=contenido)
    assert "tc5061_proyecto_equipo07.modeling.train.p3_aplicar_pesos" in no_confiables
    assert not [t for t in no_confiables if t.startswith("__main__")]
    # Con ese tipo declarado como confiable, el modelo se carga y predice igual
    cargado = skops_io.loads(contenido, trusted=no_confiables)
    assert (cargado.predict(x) == modelo.predict(x)).all()


def test_p3_buscar_k_encuentra_los_tres_grupos():
    z = ((p3_x_con_tres_grupos() - 5.0) / 4.0).to_numpy()
    resultados = train.p3_buscar_k(z, 2, 5, 42)
    assert list(resultados.index) == [2, 3, 4, 5]
    assert resultados["silueta"].idxmax() == 3
    assert {"silueta", "davies_bouldin", "calinski_harabasz", "inercia", "tam_min"} <= set(
        resultados.columns
    )


def test_p3_ajustar_baseline_es_determinista_y_coherente():
    limpios = p3_limpios_sinteticos()
    kwargs = {"k_minimo": 2, "k_maximo": 4, "min_salidas_train": 50, "min_salidas_valid": 20}
    a = train.p3_ajustar_baseline(limpios, 42, **kwargs)
    b = train.p3_ajustar_baseline(limpios, 42, **kwargs)
    # Tolerancia: los hilos de BLAS reparten las sumas distinto y el último decimal puede variar
    assert a.metricas == pytest.approx(b.metricas, rel=1e-9)
    assert 2 <= a.k_elegido <= 4
    assert a.tamanos.sum() == len(a.x_train) == a.metricas["estaciones_train"]
    assert a.metricas["ari_valid"] <= 1.0
    # Las estaciones de mañana y de tarde forman dos grupos: el baseline los recupera
    assert a.k_elegido == 2


def test_p3_seleccionar_alternativo_prefiere_las_que_cumplen():
    resumen = pd.DataFrame(
        {
            "ponderar": [False, False, True],
            "componentes": [8, 5, 6],
            "k": [3, 3, 3],
            "silueta": [0.15, 0.14, 0.12],
            "ari_valid": [0.50, 0.70, 0.70],  # la mejor silueta no cumple la estabilidad
            "tam_min": [60, 50, 10],  # y la tercera no cumple el tamaño mínimo
        }
    )
    elegido, cumplen, reserva = train.p3_seleccionar_alternativo(resumen, 0.66, 20)
    assert elegido["componentes"] == 5 and len(cumplen) == 1 and reserva is cumplen
    # Si ninguna cumple, elige entre todas y los criterios la marcarán como no aceptada
    elegido, cumplen, reserva = train.p3_seleccionar_alternativo(resumen, 0.99, 20)
    assert len(cumplen) == 0 and elegido["silueta"] == 0.15


def test_p3_evaluar_criterios():
    metricas = {"silueta": 0.145, "ari_valid": 0.656}
    elegido = pd.Series({"silueta": 0.140, "ari_valid": 0.676, "tam_min": 42})
    criterios, aceptado = train.p3_evaluar_criterios(elegido, metricas, 20)
    assert list(criterios.values()) == [False, True, True]
    assert aceptado is False
    mejor = pd.Series({"silueta": 0.150, "ari_valid": 0.676, "tam_min": 42})
    assert train.p3_evaluar_criterios(mejor, metricas, 20)[1] is True


def test_p3_semillas_busqueda_empieza_con_la_del_proyecto_y_no_repite():
    assert train.p3_semillas_busqueda(42, 5) == [42, 1, 2, 3, 4]
    assert train.p3_semillas_busqueda(2, 4) == [2, 1, 3, 4]
    assert train.p3_semillas_busqueda(42, 1) == [42]


def test_p3_nombre_run_usa_la_variable_de_mlflow(monkeypatch):
    monkeypatch.setattr(cfg, "MLFLOW_RUN_NAME", None)
    assert train.p3_nombre_run("baseline") == "baseline"
    monkeypatch.setattr(cfg, "MLFLOW_RUN_NAME", "ci-123")
    assert train.p3_nombre_run("pca_kmeans") == "ci-123-pca_kmeans"


def test_p3_etiquetas_ci_solo_con_variables_de_github(monkeypatch):
    for variable in ("GITHUB_RUN_ID", "GITHUB_ACTOR", "GITHUB_REF_NAME", "GITHUB_SHA"):
        monkeypatch.delenv(variable, raising=False)
    assert train.p3_etiquetas_ci() == {}
    monkeypatch.setenv("GITHUB_RUN_ID", "77")
    monkeypatch.setenv("GITHUB_SHA", "abcdef1234567")
    assert train.p3_etiquetas_ci() == {"github_run_id": "77", "github_sha": "abcdef1234567"}
    assert train.p3_commit_actual() == "abcdef1"


def test_p3_escribir_resumen_ci(monkeypatch, tmp_path):
    destino = tmp_path / "resumen.md"
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    train.p3_escribir_resumen_ci("sin variable", {"a": 1})  # no hace nada ni falla
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(destino))
    train.p3_escribir_resumen_ci("P3", {"silueta": 0.14491, "k": 2})
    texto = destino.read_text()
    assert "### P3" in texto and "| silueta | 0.1449 |" in texto and "| k | 2 |" in texto


def test_p3_registrar_en_mlflow_exige_la_url(monkeypatch):
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", None)
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        train.p3_registrar_en_mlflow("x", {}, {}, {})


def test_p3_estabilidad_ari_y_reasignacion():
    assert predict.p3_estabilidad_ari([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(1.0)
    x = p3_x_con_tres_grupos()
    modelo = train.p3_construir_modelo(COLUMNAS, 5, False, 3, 42).fit(x)
    grupos = predict.p3_reasignar_grupos(modelo, x, COLUMNAS)
    assert len(grupos) == len(x) and set(grupos) == {0, 1, 2}


def test_p3_nombrar_metricas_antepone_el_conjunto():
    nombradas = train.p3_nombrar_metricas({"silueta": 0.1, "ari_valid": 0.7, "tam_grupo_min": 9})
    assert nombradas == {"train_silueta": 0.1, "valid_ari": 0.7, "train_tam_grupo_min": 9}


def test_p3_nombrar_metricas_exige_que_toda_metrica_tenga_conjunto():
    with pytest.raises(KeyError, match="metrica_nueva"):
        train.p3_nombrar_metricas({"silueta": 0.1, "metrica_nueva": 1.0})


def test_p3_metricas_del_baseline_y_del_alternativo_tienen_conjunto():
    base = train.p3_ajustar_baseline(
        p3_limpios_sinteticos(),
        42,
        k_minimo=2,
        k_maximo=4,
        min_salidas_train=50,
        min_salidas_valid=20,
    )
    nombradas = train.p3_nombrar_metricas(base.metricas)
    assert all(n.startswith(("train_", "valid_")) for n in nombradas)
    assert "valid_ari" in nombradas and "train_silueta" in nombradas
