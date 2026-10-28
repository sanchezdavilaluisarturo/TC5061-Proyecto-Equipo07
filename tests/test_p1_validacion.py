"""Pruebas de la validación de P1: revisiones sobre la tabla completa y el comando p1-validar."""

import numpy as np
import pandas as pd
import pytest
from test_p1_features import p1_estaciones

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import features, validacion


@pytest.fixture(scope="module")
def tabla() -> pd.DataFrame:
    """Tabla de P1 hecha con el pipeline real: dos estaciones de sep-2024 a oct-2025."""
    rng = np.random.default_rng(cfg.SEMILLA)
    dias = pd.date_range("2024-09-01", "2025-10-31", freq="D")
    inicio = [
        dia + pd.Timedelta(hours=hora, minutes=10)
        for dia in dias
        for hora in (8, 18)
        for _ in range(rng.integers(1, 4))
    ]
    limpios = pd.DataFrame(
        {
            "inicio_del_viaje": pd.to_datetime(inicio),
            "origen_id": rng.choice([1, 2], len(inicio)),
        }
    )
    limpios["periodo_archivo"] = limpios["inicio_del_viaje"].dt.strftime("%Y-%m")
    return features.p1_tabla_modelado(limpios, p1_estaciones(), "2024-09", "2025-10")


def resultado(revision: pd.DataFrame, nombre: str) -> tuple[str, int]:
    fila = revision.set_index("revision").loc[nombre]
    return fila["resultado"], fila["filas_con_problema"]


def test_p1_tabla_del_pipeline_pasa_todas_las_revisiones(tabla):
    revision = features.p1_revisar_tabla_completa(tabla)
    assert (revision["resultado"] == "PASS").all()
    assert len(revision) == 5


def test_p1_detecta_la_llave_estacion_hora_duplicada(tabla):
    repetidas = tabla.iloc[:3].assign(viajes=99)  # misma estación y hora, otro conteo
    duplicada = pd.concat([tabla, repetidas], ignore_index=True)
    revision = features.p1_revisar_tabla_completa(duplicada)
    assert resultado(revision, "llave estación-hora duplicada") == ("FAIL", 3)


def test_p1_detecta_target_nulo_y_negativo(tabla):
    rota = tabla.astype({"viajes": "float32"})
    rota.loc[0, "viajes"] = np.nan
    rota.loc[1:2, "viajes"] = -1
    revision = features.p1_revisar_tabla_completa(rota)
    assert resultado(revision, "target viajes nulo") == ("FAIL", 1)
    assert resultado(revision, "target viajes negativo") == ("FAIL", 2)


@pytest.mark.parametrize(
    ("marcadas_como", "revision_esperada"),
    [
        ("valid", "horas de valid no posteriores a train"),
        ("test", "horas de test no posteriores a valid"),
    ],
)
def test_p1_detecta_horas_de_evaluacion_dentro_de_entrenamiento(
    tabla, marcadas_como, revision_esperada
):
    mezclada = tabla.copy()
    primeras_de_train = mezclada.index[mezclada["particion"] == "train"][:4]
    mezclada.loc[primeras_de_train, "particion"] = marcadas_como  # horas de 2024 fuera de train
    revision = features.p1_revisar_tabla_completa(mezclada)
    assert resultado(revision, revision_esperada) == ("FAIL", 4)


def test_p1_revision_exige_los_tres_conjuntos(tabla):
    sin_test = tabla[tabla["particion"] != "test"]
    with pytest.raises(ValueError, match="test"):
        features.p1_revisar_tabla_completa(sin_test)


def test_p1_validar_deja_los_reportes_de_train_valid_y_train_test(tabla, tmp_path, monkeypatch):
    ruta = tmp_path / "p1_demanda_estacion_hora.parquet"
    tabla.to_parquet(ruta, index=False)
    monkeypatch.setattr(cfg, "RUTA_P1_DEMANDA", ruta)
    monkeypatch.setattr(validacion, "VALIDACION_DIR", tmp_path / "validacion")

    features.p1_validar(muestra=2000)

    carpeta = tmp_path / "validacion"
    assert (carpeta / "p1_tabla_completa.csv").exists()
    for comparado in ("valid", "test"):
        resumen = pd.read_csv(carpeta / f"p1_train_{comparado}_resumen.csv")
        assert (resumen.loc[resumen["critico"], "resultado"] == "PASS").all()
        assert (carpeta / f"p1_train_{comparado}_train_test.html").exists()


def test_p1_validar_se_detiene_si_falla_una_revision_completa(tabla, tmp_path, monkeypatch):
    ruta = tmp_path / "p1_demanda_estacion_hora.parquet"
    pd.concat([tabla, tabla.iloc[:1]], ignore_index=True).to_parquet(ruta, index=False)
    monkeypatch.setattr(cfg, "RUTA_P1_DEMANDA", ruta)
    monkeypatch.setattr(validacion, "VALIDACION_DIR", tmp_path / "validacion")

    with pytest.raises(ValueError, match="revisiones completas"):
        features.p1_validar(muestra=2000)
    # Queda la evidencia de la revisión y no se llega a DeepChecks
    assert (tmp_path / "validacion" / "p1_tabla_completa.csv").exists()
    assert not list((tmp_path / "validacion").glob("*.html"))
