"""Pruebas de la validación de P2: revisiones sobre la tabla completa y el comando p2-validar."""

import pandas as pd
import pytest
from test_p2_modelado import p2_cambios_sinteticos

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import features, validacion


@pytest.fixture(scope="module")
def tabla() -> pd.DataFrame:
    """fit, valid y test sintéticos armados como los arma p2-validar (con la historia)."""
    return features.p2_tabla_validacion(p2_cambios_sinteticos())


def romper_llave(t):
    return pd.concat([t, t.iloc[:3]], ignore_index=True)


def romper_clase(t):
    return t.assign(clase=t["clase"].where(t.index >= 2, 3))


def romper_historia(t):
    return t.assign(hist_sin_bicis=t["hist_sin_bicis"].where(t.index >= 2, 1.5))


def romper_orden(t):
    return t.assign(particion=t["particion"].where(t.index >= 2, "valid"))  # días de train


def test_p2_tabla_del_pipeline_pasa_todas_las_revisiones(tabla):
    revision = features.p2_revisar_tabla_completa(tabla)
    assert (revision["resultado"] == "PASS").all()
    assert len(revision) == 5


@pytest.mark.parametrize(
    ("romper", "revision_esperada", "filas"),
    [
        (romper_llave, "llave estación-día-bin duplicada", 3),
        (romper_clase, "clase fuera de 0, 1 y 2", 2),
        (romper_historia, "historia fuera de [0, 1]", 2),
        (romper_orden, "días de valid no posteriores a train", 2),
    ],
)
def test_p2_revisiones_detectan_cada_problema(tabla, romper, revision_esperada, filas):
    revision = features.p2_revisar_tabla_completa(romper(tabla)).set_index("revision")
    assert revision.loc[revision_esperada, "resultado"] == "FAIL"
    assert revision.loc[revision_esperada, "filas_con_problema"] == filas
    # Cada caso rompe solo su revisión
    assert (revision["resultado"] == "FAIL").sum() == 1


def test_p2_revision_exige_los_tres_conjuntos(tabla):
    with pytest.raises(ValueError, match="test"):
        features.p2_revisar_tabla_completa(tabla[tabla["particion"] != "test"])


@pytest.fixture
def cambios_en_disco(tmp_path, monkeypatch):
    """p2-validar lee la tabla sintética de un parquet temporal y deja los reportes en tmp_path."""
    monkeypatch.setattr(cfg, "RUTA_P2_CAMBIOS", tmp_path / "p2_cambios_estacion_bin.parquet")
    monkeypatch.setattr(validacion, "VALIDACION_DIR", tmp_path / "validacion")
    return tmp_path


def test_p2_validar_deja_los_reportes_de_train_valid_y_train_test(cambios_en_disco):
    p2_cambios_sinteticos().to_parquet(cfg.RUTA_P2_CAMBIOS, index=False)

    features.p2_validar(muestra=2000)

    carpeta = cambios_en_disco / "validacion"
    assert (carpeta / "p2_tabla_completa.csv").exists()
    for comparado in ("valid", "test"):
        resumen = pd.read_csv(carpeta / f"p2_train_{comparado}_resumen.csv")
        assert (resumen.loc[resumen["critico"], "resultado"] == "PASS").all()
        assert (carpeta / f"p2_train_{comparado}_integridad.html").exists()
        assert (carpeta / f"p2_train_{comparado}_train_test.html").exists()


def test_p2_validar_se_detiene_si_falla_una_revision_completa(cambios_en_disco):
    romper_llave(p2_cambios_sinteticos()).to_parquet(cfg.RUTA_P2_CAMBIOS, index=False)

    with pytest.raises(ValueError, match="revisiones completas"):
        features.p2_validar(muestra=2000)
    # Queda la evidencia de la revisión y no se llega a DeepChecks
    carpeta = cambios_en_disco / "validacion"
    assert (carpeta / "p2_tabla_completa.csv").exists()
    assert not list(carpeta.glob("*.html"))


def test_p2_validar_sin_tabla_pide_make_features(cambios_en_disco):
    with pytest.raises(FileNotFoundError, match="make features-p2"):
        features.p2_validar(muestra=2000)
