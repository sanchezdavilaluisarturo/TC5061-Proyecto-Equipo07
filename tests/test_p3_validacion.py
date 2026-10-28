"""Pruebas de la validación de P3: revisiones sobre la tabla completa y el comando p3-validar."""

import numpy as np
import pandas as pd
import pytest
from test_p3_modeling import p3_limpios_sinteticos
from test_p3_tuning import estaciones_sinteticas

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset, features, validacion


@pytest.fixture(scope="module")
def tabla() -> pd.DataFrame:
    """Tabla de modelado de P3 con duraciones variables y el fin coherente con el inicio."""
    modelado = p3_limpios_sinteticos()
    rng = np.random.default_rng(cfg.SEMILLA)
    modelado["duracion_min"] = rng.uniform(5, 40, len(modelado))
    modelado["fin_del_viaje"] = modelado["inicio_del_viaje"] + pd.to_timedelta(
        modelado["duracion_min"], unit="min"
    )
    return modelado


def resultado(revision: pd.DataFrame, nombre: str) -> tuple[str, int]:
    fila = revision.set_index("revision").loc[nombre]
    return fila["resultado"], fila["filas_con_problema"]


def test_p3_tabla_del_pipeline_pasa_todas_las_revisiones(tabla):
    revision = features.p3_revisar_tabla_completa(tabla, estaciones_sinteticas())
    assert (revision["resultado"] == "PASS").all()
    assert len(revision) == 7


def test_p3_sin_catalogo_omite_solo_la_revision_de_estaciones(tabla):
    revision = features.p3_revisar_tabla_completa(tabla)
    assert len(revision) == 6
    assert "origen o destino fuera del catálogo" not in set(revision["revision"])


def test_p3_detecta_viaje_id_repetido(tabla):
    repetidos = pd.concat([tabla, tabla.iloc[:3]], ignore_index=True)
    revision = features.p3_revisar_tabla_completa(repetidos)
    assert resultado(revision, "viaje_id repetido") == ("FAIL", 3)


def test_p3_detecta_duracion_nula_no_positiva_y_demasiado_larga(tabla):
    rota = tabla.copy()
    rota.loc[0, "duracion_min"] = np.nan
    rota.loc[1, "duracion_min"] = 0
    rota.loc[2, "duracion_min"] = -5
    rota.loc[3, "duracion_min"] = cfg.DURACION_MAXIMA_MIN + 1
    revision = features.p3_revisar_tabla_completa(rota)
    assert resultado(revision, "duración nula o no positiva") == ("FAIL", 3)
    assert resultado(revision, f"duración mayor a {cfg.DURACION_MAXIMA_MIN} min") == ("FAIL", 1)


def test_p3_la_duracion_maxima_exacta_es_valida(tabla):
    limite = tabla.copy()
    limite.loc[0, "duracion_min"] = cfg.DURACION_MAXIMA_MIN
    revision = features.p3_revisar_tabla_completa(limite)
    assert resultado(revision, f"duración mayor a {cfg.DURACION_MAXIMA_MIN} min") == ("PASS", 0)


def test_p3_detecta_nulos_y_fin_anterior_al_inicio(tabla):
    rota = tabla.copy()
    rota.loc[0, "origen_id"] = np.nan
    rota.loc[1, "fin_del_viaje"] = pd.NaT
    rota.loc[2, "fin_del_viaje"] = rota.loc[2, "inicio_del_viaje"] - pd.Timedelta(minutes=1)
    revision = features.p3_revisar_tabla_completa(rota)
    assert resultado(revision, "fechas o estaciones nulas") == ("FAIL", 2)
    assert resultado(revision, "fin anterior al inicio") == ("FAIL", 1)


def test_p3_detecta_estaciones_fuera_del_catalogo(tabla):
    rota = tabla.copy()
    rota.loc[0, "origen_id"] = 9999
    rota.loc[1, "destino_id"] = 9999
    rota.loc[2, ["origen_id", "destino_id"]] = 9999  # una sola fila aunque falle en los dos
    revision = features.p3_revisar_tabla_completa(rota, estaciones_sinteticas())
    assert resultado(revision, "origen o destino fuera del catálogo") == ("FAIL", 3)


def test_p3_detecta_viajes_de_valid_dentro_de_entrenamiento(tabla):
    mezclada = tabla.copy()
    primeros_de_train = mezclada.index[mezclada["particion"] == "train"][:4]
    mezclada.loc[primeros_de_train, "particion"] = "valid"  # viajes de 2023 fuera de train
    revision = features.p3_revisar_tabla_completa(mezclada)
    assert resultado(revision, "viajes de valid no posteriores a train") == ("FAIL", 4)


def test_p3_los_viajes_en_brecha_no_cuentan_como_valid_ni_train(tabla):
    con_brecha = tabla.copy()
    con_brecha.loc[con_brecha.index[:5], "particion"] = "brecha"
    revision = features.p3_revisar_tabla_completa(con_brecha)
    assert resultado(revision, "viajes de valid no posteriores a train") == ("PASS", 0)


@pytest.mark.parametrize("sin", ["train", "valid"])
def test_p3_revision_exige_train_y_valid(tabla, sin):
    incompleta = tabla[tabla["particion"] != sin]
    with pytest.raises(ValueError, match=sin):
        features.p3_revisar_tabla_completa(incompleta)


def preparar_comando(tabla: pd.DataFrame, tmp_path, monkeypatch) -> None:
    """Apunta p3-validar a una tabla y a un catálogo sintéticos, y a una carpeta temporal."""
    ruta = tmp_path / "viajes_modelado.parquet"
    tabla.to_parquet(ruta, index=False)
    monkeypatch.setattr(dataset, "p3_ruta_viajes_modelado", lambda *_: ruta)
    monkeypatch.setattr(dataset, "p3_cargar_estaciones", estaciones_sinteticas)
    monkeypatch.setattr(validacion, "VALIDACION_DIR", tmp_path / "validacion")


def test_p3_validar_deja_los_reportes_de_train_vs_valid(tabla, tmp_path, monkeypatch):
    preparar_comando(tabla, tmp_path, monkeypatch)

    features.p3_validar(muestra=2000)

    carpeta = tmp_path / "validacion"
    assert (carpeta / "p3_tabla_completa.csv").exists()
    resumen = pd.read_csv(carpeta / "p3_train_valid_resumen.csv")
    assert (resumen.loc[resumen["critico"], "resultado"] == "PASS").all()
    assert (carpeta / "p3_train_valid_train_test.html").exists()
    assert (carpeta / "p3_train_valid_integridad.html").exists()


def test_p3_validar_se_detiene_si_falla_una_revision_completa(tabla, tmp_path, monkeypatch):
    preparar_comando(pd.concat([tabla, tabla.iloc[:1]], ignore_index=True), tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="revisiones completas"):
        features.p3_validar(muestra=2000)
    # Queda la evidencia de la revisión y no se llega a DeepChecks
    assert (tmp_path / "validacion" / "p3_tabla_completa.csv").exists()
    assert not list((tmp_path / "validacion").glob("*.html"))


def test_p3_validar_avisa_si_falta_la_tabla(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "p3_ruta_viajes_modelado", lambda *_: tmp_path / "no_existe.parquet")
    with pytest.raises(FileNotFoundError, match="features-p3"):
        features.p3_validar(muestra=2000)
