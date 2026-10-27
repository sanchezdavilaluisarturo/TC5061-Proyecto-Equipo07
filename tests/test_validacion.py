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
