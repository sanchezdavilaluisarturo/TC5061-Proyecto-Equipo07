"""Monitoreo de P2 (riesgo de quedarse sin bicis o sin anclajes): conjuntos, drift, versión
alternativa y A/B testing.

Usa la base común de monitoreo.py. La etiqueta y las variables de flujo de P2 no cruzan el día de
servicio, así que el mes nuevo se calcula solo con sus viajes (más un margen para los que empiezan
antes). La historia "hasta ayer" sí necesita los días previos: se calcula sobre la tabla de
entrenamiento más el mes nuevo.

Uso: python -m tc5061_proyecto_equipo07.modeling.monitoreo_p2 tabla
"""

from loguru import logger
import pandas as pd
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset, features, monitoreo

app = typer.Typer(add_completion=False)

P2_LLAVE = ["estacion", "dia", "hora_idx"]
P2_COLUMNAS = [*cfg.P2_FEATURES_SIN_HISTORICO, "clase"]
P2_COLUMNAS_VIAJES = ["inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]


def p2_leer_viajes(desde: pd.Timestamp, ruta=cfg.RUTA_VIAJES_LIMPIOS_EXTENDIDOS) -> pd.DataFrame:
    """Viajes que pueden caer en los días de servicio desde `desde`.

    Un viaje dura a lo más DURACION_MAXIMA_MIN, así que basta con leer desde ese margen antes;
    los días incompletos del margen se descartan en `p2_tabla_desde`.
    """
    margen = pd.Timedelta(minutes=cfg.DURACION_MAXIMA_MIN)
    return pd.read_parquet(
        ruta, columns=P2_COLUMNAS_VIAJES, filters=[("inicio_del_viaje", ">=", desde - margen)]
    )


def p2_tabla_desde(
    viajes: pd.DataFrame, estaciones: pd.DataFrame, desde: pd.Timestamp, fecha_limite: str
) -> pd.DataFrame:
    """Tabla estación-bin de los días de servicio desde `desde`, con la limpieza de P2."""
    estaciones = features.p2_estaciones_con_capacidad(estaciones)
    limpios = features.p2_limpiar_viajes(viajes, estaciones, fecha_limite)
    tabla = features.p2_construir_caracteristicas(limpios, estaciones)
    return tabla[tabla["dia"] >= desde].reset_index(drop=True)


def p2_comparar_tablas(entrenamiento: pd.DataFrame, nueva: pd.DataFrame) -> int:
    """Comprueba que `nueva` reproduce fila por fila a `entrenamiento` (los mismos días).

    Si no, el mes nuevo se estaría calculando distinto que los datos del champion
    (train-serving skew). Regresa el número de filas comparadas.
    """
    unidas = entrenamiento[P2_LLAVE + P2_COLUMNAS].merge(
        nueva[P2_LLAVE + P2_COLUMNAS],
        on=P2_LLAVE,
        how="outer",
        suffixes=("", "_nueva"),
        indicator=True,
    )
    sin_par = int((unidas["_merge"] != "both").sum())
    if sin_par:
        raise ValueError(f"{sin_par} estación-bins están solo en una de las dos tablas")
    distintas = pd.Series(False, index=unidas.index)
    for columna in P2_COLUMNAS:
        # Los enteros se reducen según el rango de cada tabla: se comparan valores, no tipos
        original = unidas[columna].astype("float64")
        nueva_col = unidas[f"{columna}_nueva"].astype("float64")
        iguales = (original == nueva_col) | (original.isna() & nueva_col.isna())
        if not iguales.all():
            logger.warning(f"{columna}: {int((~iguales).sum())} valores distintos")
        distintas |= ~iguales
    if distintas.any():
        raise ValueError(f"{int(distintas.sum())} estación-bins cambian al recalcularse")
    return len(unidas)


def p2_agregar_historia(
    entrenamiento: pd.DataFrame,
    evaluacion: pd.DataFrame,
    peso_historico: int = cfg.P2_PESO_HISTORICO,
) -> pd.DataFrame:
    """Historia "hasta ayer" del mes nuevo, con la frecuencia global de train (la del champion).

    Cada día del mes usa las etiquetas de todos los días previos, igual que valid y test.
    """
    tabla = pd.concat([entrenamiento, evaluacion], ignore_index=True)
    en_train = features.p2_asignar_particion(tabla["dia"]) == "train"
    historia = features.P2HistoriaHastaAyer(peso_historico).fit(tabla[en_train])
    tabla = historia.transform(tabla)
    columnas = [*cfg.P2_FEATURES_CON_HISTORICO, "clase", "estacion", "dia", "hora_idx"]
    nuevo = tabla["dia"] >= pd.Timestamp(cfg.FRONTERA_MONITOREO)
    return tabla.loc[nuevo, columnas].reset_index(drop=True)


@app.callback()
def p2_ayuda_monitoreo():
    """Monitoreo de P2: tabla del mes nuevo, drift, versión alternativa y A/B testing."""


@app.command()
def tabla():
    """Calcula la tabla del mes nuevo, verificada contra la de entrenamiento, con su historia."""
    for ruta, comando in [
        (cfg.RUTA_VIAJES_LIMPIOS_EXTENDIDOS, "make monitoreo-datos"),
        (cfg.RUTA_P2_CAMBIOS, "make features-p2"),
    ]:
        if not ruta.exists():
            raise FileNotFoundError(f"Falta {ruta.name}; ejecutar `{comando}`")
    inicio_verificacion = pd.Period(cfg.MES_FIN, freq="M").start_time
    inicio_mes = pd.Timestamp(cfg.FRONTERA_MONITOREO)
    viajes = p2_leer_viajes(inicio_verificacion)
    entrenamiento = pd.read_parquet(cfg.RUTA_P2_CAMBIOS)

    # 1) El último mes de entrenamiento, con la nomenclatura y la fecha límite de entrenamiento,
    #    debe salir idéntico a la tabla con la que se entrenó el champion
    verificacion = p2_tabla_desde(
        viajes,
        dataset.p3_cargar_estaciones(),
        inicio_verificacion,
        cfg.P2_FECHA_LIMITE_FIN_VIAJE,
    )
    filas = p2_comparar_tablas(
        entrenamiento[entrenamiento["dia"] >= inicio_verificacion], verificacion
    )
    logger.info(f"{cfg.MES_FIN} reproducido: {filas} estación-bins idénticos")

    # 2) El mes nuevo con la nomenclatura vigente
    fin = str((pd.Period(cfg.MES_MONITOREO, freq="M") + 1).start_time)
    evaluacion = p2_tabla_desde(
        viajes, monitoreo.leer_nomenclatura(cfg.RUTA_MONITOREO_CRUDOS), inicio_mes, fin
    )
    evaluacion = p2_agregar_historia(entrenamiento, evaluacion)
    evaluacion["periodo"] = monitoreo.asignar_periodo(evaluacion["dia"])
    if (evaluacion["periodo"] != "evaluacion").any():
        raise ValueError("La tabla de evaluación trae días fuera del mes de monitoreo")

    evaluacion.to_parquet(cfg.P2_RUTA_EVALUACION, index=False)
    prevalencia = evaluacion["clase"].value_counts(normalize=True).sort_index().round(4)
    logger.success(
        f"{cfg.P2_RUTA_EVALUACION.name}: {len(evaluacion)} estación-bins, "
        f"{evaluacion['estacion'].nunique()} estaciones | prevalencia {prevalencia.to_dict()}"
    )


if __name__ == "__main__":
    app()
