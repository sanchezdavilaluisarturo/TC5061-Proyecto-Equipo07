"""Pruebas del monitoreo de P1: la tabla del mes nuevo se calcula igual que la de entrenamiento."""

import numpy as np
import pandas as pd
import pytest

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
