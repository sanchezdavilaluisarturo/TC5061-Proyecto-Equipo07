"""Pruebas del módulo común de validación con DeepChecks, con tablas sintéticas."""

import numpy as np
import pandas as pd
import pytest

from tc5061_proyecto_equipo07 import validacion

N_FILAS = 400
N_TRAIN = 300


def tabla_sintetica(semilla: int = 0) -> pd.DataFrame:
    """Tabla por hora con partición temporal limpia: train son las primeras 300 horas."""
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2025-01-01", periods=N_FILAS, freq="h")
    tabla = pd.DataFrame(
        {
            "fecha": fechas,
            "hora": fechas.hour,
            "zona": rng.choice(["GDL", "ZPN"], N_FILAS),
            "rezago": rng.poisson(2, N_FILAS).astype(float),
            "viajes": rng.poisson(2, N_FILAS),
        }
    )
    tabla["particion"] = np.where(np.arange(N_FILAS) < N_TRAIN, "train", "test")
    return tabla


def validar(tabla: pd.DataFrame, carpeta, **opciones) -> pd.DataFrame:
    return validacion.validar_tabla(
        "prueba", tabla, "particion", "viajes", ["zona"], "fecha", carpeta=carpeta, **opciones
    )


def resultado_de(resumen: pd.DataFrame, chequeo: str) -> str:
    return resumen.loc[resumen["chequeo"] == chequeo, "resultado"].iloc[0]


def detectado(resumen: pd.DataFrame, chequeo: str) -> bool:
    """DeepChecks reporta algunos problemas como WARN y otros como FAIL; ambos cuentan."""
    return resultado_de(resumen, chequeo) in ("WARN", "FAIL")


def test_tabla_limpia_pasa_y_deja_los_reportes(tmp_path):
    resumen = validar(tabla_sintetica(), tmp_path)
    assert (resumen.loc[resumen["critico"], "resultado"] == "PASS").all()
    assert {"suite", "chequeo", "condicion", "resultado", "detalle", "critico"} <= set(resumen)
    for sufijo in ("integridad.html", "train_test.html", "resumen.csv"):
        assert (tmp_path / f"prueba_{sufijo}").exists()


def test_fechas_de_prueba_dentro_de_train_hacen_fallar(tmp_path):
    tabla = tabla_sintetica()
    tabla.loc[N_TRAIN : N_TRAIN + 9, "fecha"] = tabla.loc[:9, "fecha"].to_numpy()
    with pytest.raises(ValueError, match="críticas"):
        validar(tabla, tmp_path)
    # el reporte queda escrito aunque falle, para documentar el problema
    assert (tmp_path / "prueba_resumen.csv").exists()


def test_detecta_duplicados_como_hallazgo(tmp_path):
    tabla = tabla_sintetica()
    tabla = pd.concat([tabla, tabla.iloc[:20]], ignore_index=True)
    resumen = validar(tabla, tmp_path)
    assert detectado(resumen, "Data Duplicates")
    assert not resumen.loc[resumen["chequeo"] == "Data Duplicates", "critico"].iloc[0]


def test_detecta_nulos(tmp_path):
    tabla = tabla_sintetica()
    tabla.loc[:9, "rezago"] = np.nan
    assert detectado(validar(tabla, tmp_path), "PercentOfNulls")


def test_umbral_de_nulos_permite_los_esperados(tmp_path):
    tabla = tabla_sintetica()
    tabla.loc[:9, "rezago"] = np.nan  # 2.5 % de la columna, como los primeros días de rezago
    assert resultado_de(validar(tabla, tmp_path, umbral_nulos=0.05), "PercentOfNulls") == "PASS"


def test_detecta_categoria_nueva_en_prueba(tmp_path):
    tabla = tabla_sintetica()
    tabla.loc[N_TRAIN:, "zona"] = "TLQ"
    assert detectado(validar(tabla, tmp_path), "New Category Train Test")


def test_un_chequeo_declarado_critico_detiene_la_ejecucion(tmp_path):
    tabla = pd.concat([tabla_sintetica(), tabla_sintetica().iloc[:20]], ignore_index=True)
    with pytest.raises(ValueError, match="Data Duplicates"):
        validar(tabla, tmp_path, criticos=("Data Duplicates",))


def test_falla_con_mensaje_claro_si_falta_una_columna(tmp_path):
    with pytest.raises(ValueError, match="Faltan columnas"):
        validar(tabla_sintetica().drop(columns="zona"), tmp_path)


def test_falla_si_un_conjunto_no_tiene_filas(tmp_path):
    with pytest.raises(ValueError, match="valid"):
        validar(tabla_sintetica(), tmp_path, conjuntos=("train", "valid"))


def test_muestrear_es_reproducible_y_respeta_tablas_chicas():
    tabla = tabla_sintetica()
    assert validacion.muestrear(tabla, 50).equals(validacion.muestrear(tabla, 50))
    assert len(validacion.muestrear(tabla, 50)) == 50
    assert validacion.muestrear(tabla, None) is tabla
    assert validacion.muestrear(tabla, 10_000) is tabla
