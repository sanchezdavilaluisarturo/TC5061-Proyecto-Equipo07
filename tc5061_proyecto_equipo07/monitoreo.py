"""Monitoreo (Entregable 8): datos del mes nuevo, drift con Evidently y A/B testing de versiones.

Base común de P1, P2 y P3. Cada pregunta arma sus conjuntos y su versión alternativa en su propio
módulo (modeling/monitoreo_p1.py, etc.) y usa de aquí:
    asignar_periodo          referencia, valid, actual, evaluacion o brecha de cada fecha

Uso: python -m tc5061_proyecto_equipo07.monitoreo datos
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset

app = typer.Typer()

PERIODOS = ["referencia", "valid", "actual", "evaluacion", "brecha"]


def asignar_periodo(inicio: pd.Series) -> pd.Series:
    """Regresa el periodo de monitoreo de cada fecha, con la misma brecha de la partición común.

    referencia = train, actual = test y evaluacion = mes nuevo. Las BRECHA_HORAS previas a cada
    frontera quedan en `brecha`, igual que en la partición de P1, P2 y P3.
    """
    brecha = pd.Timedelta(hours=cfg.BRECHA_HORAS)
    fronteras = [
        pd.Timestamp(cfg.FRONTERA_VALID),
        pd.Timestamp(cfg.FRONTERA_TEST),
        pd.Timestamp(cfg.FRONTERA_MONITOREO),
    ]
    fin_monitoreo = pd.Period(cfg.MES_MONITOREO, freq="M").end_time
    condiciones, valores = [], []
    for frontera, nombre in zip(fronteras, ["referencia", "valid", "actual"]):
        condiciones += [inicio < frontera - brecha, inicio < frontera]
        valores += [nombre, "brecha"]
    condiciones.append(inicio <= fin_monitoreo)
    valores.append("evaluacion")
    periodo = np.select(condiciones, valores, default="fuera_de_rango")
    if (periodo == "fuera_de_rango").any():
        raise ValueError(f"Hay fechas posteriores a {cfg.MES_MONITOREO}")
    return pd.Series(pd.Categorical(periodo, categories=PERIODOS), index=inicio.index)


def limpiar_viajes(viajes: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Aplica a un DataFrame las mismas reglas de limpieza de `dataset.limpiar`.

    `dataset.limpiar` trabaja sobre archivos del rango MES_INICIO-MES_FIN y toma la nomenclatura
    de data/raw/estaciones/; aquí se reciben ya cargados para usar la nomenclatura del mes nuevo.
    """
    duracion_min = (viajes["fin_del_viaje"] - viajes["inicio_del_viaje"]).dt.total_seconds() / 60
    con_nomenclatura = viajes["origen_id"].isin(estaciones["id"]) & viajes["destino_id"].isin(
        estaciones["id"]
    )
    reglas = {
        "Filas duplicadas (copias exactas)": ~viajes.duplicated(),
        "Origen o destino sin nomenclatura": con_nomenclatura,
        "Duración cero": duracion_min > 0,
        "Duración mayor a 24 horas": duracion_min <= cfg.DURACION_MAXIMA_MIN,
    }
    conservar = pd.Series(True, index=viajes.index)
    for regla, cumple in reglas.items():
        logger.info(f"{regla}: {int((conservar & ~cumple).sum())} retirados")
        conservar &= cumple
    limpios = viajes[conservar].reset_index(drop=True)

    edad = limpios["inicio_del_viaje"].dt.year - limpios["anio_de_nacimiento"]
    anio_invalido = (limpios["anio_de_nacimiento"] < 1900) | (edad < cfg.EDAD_MINIMA)
    limpios.loc[anio_invalido.fillna(False), "anio_de_nacimiento"] = pd.NA
    logger.info(f"Años de nacimiento inválidos anulados: {int(anio_invalido.sum())}")
    return limpios


def leer_nomenclatura(carpeta: Path) -> pd.DataFrame:
    """Lee la nomenclatura más reciente de la carpeta."""
    ruta = max(carpeta.glob("nomenclatura_*.csv"), default=None)
    if ruta is None:
        raise FileNotFoundError(f"No hay nomenclatura en {carpeta}")
    contenido = ruta.read_bytes()
    return pd.read_csv(ruta, encoding=dataset.detectar_codificacion(contenido))


def estaciones_nuevas(entrenamiento: pd.DataFrame, monitoreo: pd.DataFrame) -> pd.DataFrame:
    """Estaciones de la nomenclatura del mes nuevo que no estaban en la de entrenamiento."""
    return monitoreo[~monitoreo["id"].isin(entrenamiento["id"])].reset_index(drop=True)


def unir_con_historia(historia: pd.DataFrame, mes: pd.DataFrame) -> pd.DataFrame:
    """Une los viajes limpios del rango con los del mes nuevo, con categorías comunes."""
    categoricas = {
        columna: union_categoricals(
            [historia[columna].astype("category"), mes[columna].astype("category")]
        )
        for columna in ("genero", "periodo_archivo")
    }
    viajes = pd.concat(
        [historia.drop(columns=list(categoricas)), mes.drop(columns=list(categoricas))],
        ignore_index=True,
    )
    for columna, valores in categoricas.items():
        viajes[columna] = valores
    return viajes[list(historia.columns)]


@app.command()
def descargar(mes: str = cfg.MES_MONITOREO):
    """Descarga el mes nuevo y la nomenclatura vigente a data/raw/monitoreo/ con su manifiesto.

    Va en una carpeta aparte: el manifiesto del proyecto y data/raw/estaciones/ no cambian.
    """
    enlaces_viajes, enlaces_estaciones = dataset.enlaces_del_portal()
    viajes = enlaces_viajes[enlaces_viajes["periodo"] == mes]
    if viajes.empty:
        raise ValueError(f"El portal todavía no publica los viajes de {mes}")
    archivos = pd.concat([viajes, enlaces_estaciones.iloc[[-1]]], ignore_index=True)

    cfg.RUTA_MONITOREO_CRUDOS.mkdir(parents=True, exist_ok=True)
    rutas = [cfg.RUTA_MONITOREO_CRUDOS / archivo for archivo in archivos["archivo"]]
    for url, ruta in zip(archivos["url"], rutas):
        accion = "descargado" if dataset.descargar_archivo(url, ruta) else "ya existía"
        logger.info(f"{ruta.name}: {accion}")

    manifiesto = archivos[["archivo", "periodo", "url"]].copy()
    manifiesto["registros"] = [dataset.contar_registros(ruta) for ruta in rutas]
    manifiesto["sha256"] = [dataset.calcular_sha256(ruta) for ruta in rutas]
    manifiesto.to_csv(cfg.RUTA_MANIFIESTO_MONITOREO, index=False)
    logger.success(f"Manifiesto: {cfg.RUTA_MANIFIESTO_MONITOREO.name}")


@app.command()
def construir(mes: str = cfg.MES_MONITOREO):
    """Lee, valida y limpia el mes nuevo, y lo une con los viajes limpios del rango del proyecto."""
    manifiesto = pd.read_csv(cfg.RUTA_MANIFIESTO_MONITOREO)
    fila = manifiesto[manifiesto["archivo"].str.startswith("datos_abiertos")].iloc[0]
    ruta = cfg.RUTA_MONITOREO_CRUDOS / fila["archivo"]
    if dataset.calcular_sha256(ruta) != fila["sha256"]:
        raise ValueError(f"{ruta.name} no coincide con el manifiesto; ejecutar 'descargar'")
    viajes = dataset.leer_viajes_csv(ruta, mes)
    if len(viajes) != fila["registros"]:
        raise ValueError(
            f"Se leyeron {len(viajes)} registros y el manifiesto indica {fila['registros']}"
        )

    nomenclatura_monitoreo = leer_nomenclatura(cfg.RUTA_MONITOREO_CRUDOS)
    nuevas = estaciones_nuevas(
        leer_nomenclatura(cfg.RUTA_ESTACIONES_CRUDAS), nomenclatura_monitoreo
    )
    cfg.MONITOREO_DIR.mkdir(parents=True, exist_ok=True)
    nuevas.to_csv(cfg.MONITOREO_DIR / "estaciones_nuevas.csv", index=False)
    logger.info(f"Estaciones nuevas frente a la nomenclatura de entrenamiento: {len(nuevas)}")

    limpios = limpiar_viajes(viajes, nomenclatura_monitoreo)
    limpios.to_parquet(cfg.RUTA_VIAJES_LIMPIOS_MONITOREO, index=False)
    logger.success(f"{cfg.RUTA_VIAJES_LIMPIOS_MONITOREO.name}: {len(limpios)} registros")

    if not cfg.RUTA_VIAJES_LIMPIOS.exists():
        raise FileNotFoundError(f"Falta {cfg.RUTA_VIAJES_LIMPIOS.name}; ejecutar `make data`")
    extendidos = unir_con_historia(pd.read_parquet(cfg.RUTA_VIAJES_LIMPIOS), limpios)
    extendidos.to_parquet(cfg.RUTA_VIAJES_LIMPIOS_EXTENDIDOS, index=False)
    logger.success(f"{cfg.RUTA_VIAJES_LIMPIOS_EXTENDIDOS.name}: {len(extendidos)} registros")


@app.command()
def datos(mes: str = cfg.MES_MONITOREO):
    """Ejecuta descargar y construir en orden."""
    descargar(mes)
    construir(mes)


if __name__ == "__main__":
    app()
