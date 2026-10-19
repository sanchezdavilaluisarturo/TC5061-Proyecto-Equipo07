"""Ingesta reproducible de MiBici: descarga, normalización, versionado y limpieza de los datos.

Etapas (cada una es un comando de la terminal):
    descargar  Portal -> data/raw/ (CSV originales) y manifiesto references/fuentes-datos.csv
    construir  data/raw/ -> data/interim/viajes_AAAA-MM_AAAA-MM.parquet (normalizado y validado)
    limpiar    data/interim/ -> data/processed/viajes_limpios_AAAA-MM_AAAA-MM.parquet
    todo       Las tres etapas en orden

Uso: python -m tc5061_proyecto_equipo07.dataset todo
"""

import hashlib
from pathlib import Path
import re
import time
from urllib.parse import urljoin

from loguru import logger
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.config import (
    COLUMNAS_NORMALIZADAS,
    COLUMNAS_VIAJES,
    DURACION_MAXIMA_MIN,
    EDAD_MINIMA,
    ENCABEZADOS_HTTP,
    INTERIM_DATA_DIR,
    MES_FIN,
    MES_INICIO,
    MESES_SIN_PUBLICAR,
    PROCESSED_DATA_DIR,
    RUTA_ESTACIONES_CRUDAS,
    RUTA_MANIFIESTO,
    RUTA_VIAJES_CRUDOS,
    TIPOS_ENTEROS,
    URL_PORTAL,
)

app = typer.Typer()

# 78 archivos usan ISO; 2023-11, 2024-09 y 2024-10 usan día/mes/año con hora de 1 o 2 dígitos
FORMATO_ISO = "%Y-%m-%d %H:%M:%S"
FORMATO_BARRAS = "%d/%m/%Y %H:%M"


def detectar_codificacion(contenido: bytes) -> str:
    """Regresa utf-8-sig si los bytes son UTF-8 válido; si no, latin-1, que decodifica todo byte.

    utf-8-sig lee igual los archivos con y sin BOM, y quita el BOM del primer encabezado.
    """
    try:
        contenido.decode("utf-8")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "latin-1"


def calcular_sha256(ruta: Path) -> str:
    """Regresa el hash SHA-256 del archivo."""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def contar_registros(ruta: Path) -> int:
    """Cuenta las filas de datos de un CSV sin cargarlo en pandas (líneas menos el encabezado)."""
    contenido = ruta.read_bytes()
    return contenido.count(b"\n") + (0 if contenido.endswith(b"\n") else 1) - 1


def enlaces_del_portal() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee el portal y regresa los enlaces de viajes y de nomenclatura (archivo, periodo, url).

    Las URLs no se pueden construir con el año y el mes: cada archivo vive en una carpeta con un
    identificador distinto, por eso se leen de la página.
    """
    respuesta = requests.get(URL_PORTAL, headers=ENCABEZADOS_HTTP, timeout=30)
    respuesta.raise_for_status()
    tablas = {}
    for prefijo in ("datos_abiertos", "nomenclatura"):
        patron = rf"(?P<archivo>{prefijo}_(?P<anio>\d{{4}})_(?P<mes>\d{{2}})\.csv)"
        enlaces = {}
        for href in re.findall(r"href=['\"]([^'\"]+)['\"]", respuesta.text):
            coincidencia = re.search(patron, href)
            if coincidencia:
                periodo = f"{coincidencia['anio']}-{coincidencia['mes']}"
                enlaces[coincidencia["archivo"]] = (periodo, urljoin(URL_PORTAL, href))
        tablas[prefijo] = pd.DataFrame(
            [(archivo, periodo, url) for archivo, (periodo, url) in enlaces.items()],
            columns=["archivo", "periodo", "url"],
        ).sort_values("periodo", ignore_index=True)
        if tablas[prefijo].empty:
            raise ValueError(f"No se encontraron enlaces de '{prefijo}' en {URL_PORTAL}")
    return tablas["datos_abiertos"], tablas["nomenclatura"]


def descargar_archivo(url: str, destino: Path) -> bool:
    """Descarga el archivo si no existe; regresa True si se descargó.

    Se escribe primero en un .part y se renombra al terminar, para que una descarga
    interrumpida no deje un CSV incompleto que parezca válido.
    """
    if destino.exists():
        return False
    respuesta = requests.get(url, headers=ENCABEZADOS_HTTP, timeout=60)
    respuesta.raise_for_status()
    temporal = destino.with_suffix(".part")
    temporal.write_bytes(respuesta.content)
    temporal.replace(destino)
    return True


def version_datos(ruta_manifiesto: Path = RUTA_MANIFIESTO) -> str:
    """Regresa un identificador corto de la versión de los datos, para registrarlo en MLflow.

    Es el hash de los SHA-256 de los archivos del manifiesto: cambia si cambia cualquier archivo.
    """
    manifiesto = pd.read_csv(ruta_manifiesto).sort_values("archivo")
    return hashlib.sha256("".join(manifiesto["sha256"]).encode()).hexdigest()[:12]


def leer_viajes_csv(ruta: Path, periodo: str) -> pd.DataFrame:
    """Lee un CSV mensual de viajes con nombres, fechas y tipos normalizados.

    Se detiene si aparece un encabezado desconocido, si un valor no se puede convertir o si un
    viaje inicia fuera del mes del archivo (lo que indicaría día y mes intercambiados).
    """
    viajes = pd.read_csv(ruta, encoding=detectar_codificacion(ruta.read_bytes()), dtype=str)
    desconocidas = [columna for columna in viajes.columns if columna not in COLUMNAS_NORMALIZADAS]
    if desconocidas:
        raise ValueError(f"{ruta.name} tiene encabezados desconocidos: {desconocidas}")
    viajes = viajes.rename(columns=COLUMNAS_NORMALIZADAS)
    if list(viajes.columns) != COLUMNAS_VIAJES:
        raise ValueError(f"{ruta.name} no tiene las columnas esperadas: {list(viajes.columns)}")

    primera_fecha = viajes["inicio_del_viaje"].dropna().iloc[0]
    formato = FORMATO_BARRAS if "/" in primera_fecha else FORMATO_ISO
    for columna in COLUMNAS_VIAJES:
        if columna in ("inicio_del_viaje", "fin_del_viaje"):
            convertida = pd.to_datetime(viajes[columna], format=formato, errors="coerce")
        elif columna in TIPOS_ENTEROS:
            convertida = pd.to_numeric(viajes[columna], errors="coerce").astype(
                TIPOS_ENTEROS[columna]
            )
        else:
            continue
        no_convertidos = int((viajes[columna].notna() & convertida.isna()).sum())
        if no_convertidos:
            raise ValueError(
                f"{ruta.name}: {no_convertidos} valores de '{columna}' no se convirtieron"
            )
        viajes[columna] = convertida

    mes = pd.Period(periodo, freq="M")
    inicio = viajes["inicio_del_viaje"]
    fuera_del_mes = int((inicio.notna() & (inicio.dt.to_period("M") != mes)).sum())
    if fuera_del_mes:
        raise ValueError(f"{ruta.name}: {fuera_del_mes} viajes inician fuera de {periodo}")
    viajes["periodo_archivo"] = periodo
    return viajes


@app.command()
def descargar(mes_inicio: str = MES_INICIO, mes_fin: str = MES_FIN):
    """Descarga los CSV del rango y la nomenclatura vigente, y actualiza el manifiesto."""
    enlaces_viajes, enlaces_estaciones = enlaces_del_portal()
    meses_esperados = pd.period_range(mes_inicio, mes_fin, freq="M").astype(str)
    enlaces_viajes = enlaces_viajes[enlaces_viajes["periodo"].isin(meses_esperados)]

    faltantes = sorted(set(meses_esperados) - set(enlaces_viajes["periodo"]))
    inesperados = [mes for mes in faltantes if mes not in MESES_SIN_PUBLICAR]
    if inesperados:
        raise ValueError(
            f"El portal no publica meses no registrados en MESES_SIN_PUBLICAR: {inesperados}"
        )
    if faltantes:
        logger.warning(f"Meses sin publicar (tolerados): {faltantes}")

    RUTA_VIAJES_CRUDOS.mkdir(parents=True, exist_ok=True)
    RUTA_ESTACIONES_CRUDAS.mkdir(parents=True, exist_ok=True)
    enlaces_viajes = enlaces_viajes.assign(carpeta=RUTA_VIAJES_CRUDOS)
    nomenclatura = enlaces_estaciones.iloc[[-1]].assign(carpeta=RUTA_ESTACIONES_CRUDAS)
    archivos = pd.concat([enlaces_viajes, nomenclatura], ignore_index=True)

    descargados = 0
    for fila in tqdm(archivos.itertuples(), total=len(archivos), desc="Descarga"):
        if descargar_archivo(fila.url, fila.carpeta / fila.archivo):
            descargados += 1
            time.sleep(1)  # pausa entre descargas para no saturar el portal
    logger.info(f"Descargados: {descargados} | ya existían: {len(archivos) - descargados}")

    # Manifiesto versionado en Git: si un archivo cambia, el cambio se ve en el diff
    manifiesto = archivos[["archivo", "periodo", "url"]].copy()
    rutas = [
        carpeta / archivo for carpeta, archivo in zip(archivos["carpeta"], archivos["archivo"])
    ]
    manifiesto["registros"] = [contar_registros(ruta) for ruta in rutas]
    manifiesto["sha256"] = [calcular_sha256(ruta) for ruta in rutas]
    manifiesto.to_csv(RUTA_MANIFIESTO, index=False)
    logger.success(f"Manifiesto: {RUTA_MANIFIESTO.name} | versión de los datos: {version_datos()}")


@app.command()
def construir(mes_inicio: str = MES_INICIO, mes_fin: str = MES_FIN):
    """Une los CSV del rango en un Parquet normalizado y valida que no se pierdan registros."""
    manifiesto = pd.read_csv(RUTA_MANIFIESTO)
    manifiesto = manifiesto[manifiesto["archivo"].str.startswith("datos_abiertos")]
    manifiesto = manifiesto[manifiesto["periodo"].between(mes_inicio, mes_fin)]

    partes = []
    for fila in tqdm(manifiesto.itertuples(), total=len(manifiesto), desc="Lectura"):
        ruta = RUTA_VIAJES_CRUDOS / fila.archivo
        if calcular_sha256(ruta) != fila.sha256:
            raise ValueError(f"{fila.archivo} no coincide con el manifiesto; ejecutar 'descargar'")
        partes.append(leer_viajes_csv(ruta, fila.periodo))
    viajes = pd.concat(partes, ignore_index=True)
    del partes
    # A category después de unir, para que todos los meses compartan las mismas categorías
    viajes["genero"] = viajes["genero"].astype("category")
    viajes["periodo_archivo"] = viajes["periodo_archivo"].astype("category")

    esperados = int(manifiesto["registros"].sum())
    if len(viajes) != esperados:
        raise ValueError(f"Se leyeron {len(viajes)} registros y el manifiesto indica {esperados}")

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ruta_salida = INTERIM_DATA_DIR / f"viajes_{mes_inicio}_{mes_fin}.parquet"
    viajes.to_parquet(ruta_salida, index=False)
    logger.success(f"{ruta_salida.name}: {len(viajes)} registros de {len(manifiesto)} archivos")


@app.command()
def limpiar(mes_inicio: str = MES_INICIO, mes_fin: str = MES_FIN):
    """Aplica las reglas de limpieza comunes a P1, P2 y P3 y registra cuánto retira cada una.

    Las reglas propias de una pregunta (horas sin servicio, estaciones con capacidad 0, partición)
    se aplican en features.py o en el notebook de esa pregunta.
    """
    viajes = pd.read_parquet(INTERIM_DATA_DIR / f"viajes_{mes_inicio}_{mes_fin}.parquet")
    ruta_nomenclatura = max(RUTA_ESTACIONES_CRUDAS.glob("nomenclatura_*.csv"))
    contenido = ruta_nomenclatura.read_bytes()
    estaciones = pd.read_csv(ruta_nomenclatura, encoding=detectar_codificacion(contenido))

    duracion_min = (viajes["fin_del_viaje"] - viajes["inicio_del_viaje"]).dt.total_seconds() / 60
    con_nomenclatura = viajes["origen_id"].isin(estaciones["id"]) & viajes["destino_id"].isin(
        estaciones["id"]
    )
    reglas = {
        "Filas duplicadas (copias exactas)": ~viajes.duplicated(),
        "Origen o destino sin nomenclatura": con_nomenclatura,
        "Duración cero": duracion_min > 0,
        "Duración mayor a 24 horas": duracion_min <= DURACION_MAXIMA_MIN,
    }
    conservar = pd.Series(True, index=viajes.index)
    for regla, cumple in reglas.items():
        retirados = int((conservar & ~cumple).sum())
        conservar &= cumple
        logger.info(f"{regla}: {retirados} retirados")
    limpios = viajes[conservar].reset_index(drop=True)

    # Un año imposible es un error del dato de usuario, no del viaje: se anula y el viaje se queda
    edad = limpios["inicio_del_viaje"].dt.year - limpios["anio_de_nacimiento"]
    anio_invalido = (limpios["anio_de_nacimiento"] < 1900) | (edad < EDAD_MINIMA)
    limpios.loc[anio_invalido.fillna(False), "anio_de_nacimiento"] = pd.NA
    logger.info(f"Años de nacimiento inválidos anulados: {int(anio_invalido.sum())}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ruta_salida = PROCESSED_DATA_DIR / f"viajes_limpios_{mes_inicio}_{mes_fin}.parquet"
    limpios.to_parquet(ruta_salida, index=False)
    retirados = len(viajes) - len(limpios)
    logger.success(f"{ruta_salida.name}: {len(limpios)} registros ({retirados} retirados)")


@app.command()
def todo(mes_inicio: str = MES_INICIO, mes_fin: str = MES_FIN):
    """Ejecuta descargar, construir y limpiar en orden."""
    descargar(mes_inicio, mes_fin)
    construir(mes_inicio, mes_fin)
    limpiar(mes_inicio, mes_fin)


# ---------------------------------------------------------------------------------------------
# P3 (clustering de estaciones): carga de las entradas y partición temporal.
# Se agregan sin tocar lo anterior; toda función nueva lleva el prefijo p3_.
# ---------------------------------------------------------------------------------------------


def p3_ruta_viajes_modelado(mes_inicio: str = MES_INICIO, mes_fin: str = MES_FIN) -> Path:
    """Regresa la ruta de data/processed/viajes_modelado_<inicio>_<fin>.parquet."""
    return PROCESSED_DATA_DIR / f"viajes_modelado_{mes_inicio}_{mes_fin}.parquet"


def p3_meses_sin_datos(
    periodos_en_datos, mes_inicio: str = MES_INICIO, mes_fin: str = MES_FIN
) -> list[str]:
    """Regresa los meses 'AAAA-MM' del rango que no están en `periodos_en_datos`.

    Se calculan de los datos y no de una lista fija: si el portal retira o vuelve a publicar un
    mes, el resultado cambia solo.
    """
    presentes = {str(periodo) for periodo in periodos_en_datos}
    rango = pd.period_range(mes_inicio, mes_fin, freq="M")
    return [str(periodo) for periodo in rango if str(periodo) not in presentes]


def p3_cargar_viajes(
    mes_inicio: str = MES_INICIO, mes_fin: str = MES_FIN, validar_manifiesto: bool = True
) -> pd.DataFrame:
    """Lee el Parquet normalizado de data/interim/ (viajes sin limpiar que deja `construir`).

    Con `validar_manifiesto` se detiene si el número de registros no coincide con el del
    manifiesto versionado: sería señal de que `construir` quedó desactualizado.
    """
    ruta = INTERIM_DATA_DIR / f"viajes_{mes_inicio}_{mes_fin}.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta.name}; ejecutar `make data`")
    viajes = pd.read_parquet(ruta)
    if validar_manifiesto:
        manifiesto = pd.read_csv(RUTA_MANIFIESTO)
        es_viaje = manifiesto["archivo"].str.startswith("datos_abiertos")
        manifiesto = manifiesto[es_viaje & manifiesto["periodo"].between(mes_inicio, mes_fin)]
        esperados = int(manifiesto["registros"].sum())
        if len(viajes) != esperados:
            raise ValueError(
                f"{ruta.name} tiene {len(viajes):,} registros y el manifiesto indica "
                f"{esperados:,}; ejecutar `make data`"
            )
    return viajes


def p3_cargar_estaciones() -> pd.DataFrame:
    """Regresa la nomenclatura de estaciones más reciente de data/raw/estaciones/."""
    rutas = list(RUTA_ESTACIONES_CRUDAS.glob("nomenclatura_*.csv"))
    if not rutas:
        raise FileNotFoundError(f"No hay nomenclatura en {RUTA_ESTACIONES_CRUDAS}; `make data`")
    ruta = max(rutas)
    return pd.read_csv(ruta, encoding=detectar_codificacion(ruta.read_bytes()))


def p3_asignar_particion(inicio_del_viaje: pd.Series) -> pd.Series:
    """Regresa train, valid, test o brecha según el instante de inicio (partición temporal).

    La partición es temporal para que ningún conjunto vea información posterior. Las horas
    previas a cada frontera (`P3_BRECHA_HORAS`) quedan en `brecha` y no se usan.
    """
    frontera_valid = pd.Timestamp(cfg.P3_FRONTERA_VALID)
    frontera_test = pd.Timestamp(cfg.P3_FRONTERA_TEST)
    brecha = pd.Timedelta(hours=cfg.P3_BRECHA_HORAS)
    condiciones = [
        inicio_del_viaje < frontera_valid - brecha,
        inicio_del_viaje < frontera_valid,
        inicio_del_viaje < frontera_test - brecha,
        inicio_del_viaje < frontera_test,
    ]
    # Códigos enteros (int8) en lugar de cadenas: con 27 millones de filas ahorra varios GB
    categorias = ["train", "valid", "test", "brecha"]
    codigos = np.select(
        condiciones, [np.int8(0), np.int8(3), np.int8(1), np.int8(3)], default=np.int8(2)
    )
    return pd.Series(
        pd.Categorical.from_codes(codigos, categories=categorias), index=inicio_del_viaje.index
    )


def p3_preparar_modelado(limpios: pd.DataFrame) -> pd.DataFrame:
    """Deja los viajes limpios listos para modelar: columnas de modelado y partición temporal.

    Agrega `duracion_min` y conserva solo `P3_COLUMNAS_MODELADO`: `usuario_id`, `genero` y
    `anio_de_nacimiento` no pasan (son datos personales y ninguna pregunta los usa).
    """
    base = [c for c in cfg.P3_COLUMNAS_MODELADO if c != "duracion_min"]
    modelado = limpios[base].copy()
    modelado["duracion_min"] = (
        modelado["fin_del_viaje"] - modelado["inicio_del_viaje"]
    ).dt.total_seconds() / 60
    modelado = modelado[cfg.P3_COLUMNAS_MODELADO]
    modelado["particion"] = p3_asignar_particion(modelado["inicio_del_viaje"])
    return modelado


def p3_validar_modelado(
    modelado: pd.DataFrame,
    estaciones: pd.DataFrame,
    duracion_maxima_min: float = DURACION_MAXIMA_MIN,
) -> None:
    """Se detiene con ValueError si el dataset de modelado incumple alguna regla de limpieza.

    Reglas: `viaje_id` único, estaciones con nomenclatura, duración en (0, máximo], fechas y
    estaciones sin nulos, y conjuntos de partición ordenados en el tiempo sin superposición.
    """
    problemas = []
    if not modelado["viaje_id"].is_unique:
        problemas.append("quedan viaje_id repetidos")
    ids = estaciones["id"]
    if not (modelado["origen_id"].isin(ids).all() and modelado["destino_id"].isin(ids).all()):
        problemas.append("hay origen o destino sin nomenclatura")
    if not modelado["duracion_min"].between(0, duracion_maxima_min, inclusive="right").all():
        problemas.append("hay duraciones fuera de (0, 24 h]")
    columnas = ["inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]
    if modelado[columnas].isna().any().any():
        problemas.append("hay nulos en fechas o estaciones")
    if modelado["particion"].isna().any():
        problemas.append("hay viajes sin conjunto de partición")
    por_conjunto = modelado.groupby("particion", observed=True)["inicio_del_viaje"].agg(
        ["min", "max"]
    )
    if not (
        por_conjunto.loc["train", "max"] < por_conjunto.loc["valid", "min"]
        and por_conjunto.loc["valid", "max"] < por_conjunto.loc["test", "min"]
    ):
        problemas.append("los conjuntos de la partición se superponen en el tiempo")
    if problemas:
        raise ValueError("El dataset de modelado no es válido: " + "; ".join(problemas))


@app.command()
def p3_particionar(mes_inicio: str = MES_INICIO, mes_fin: str = MES_FIN):
    """Deja viajes_modelado_*.parquet: viajes limpios con columnas de modelado y partición."""
    ruta_limpios = PROCESSED_DATA_DIR / f"viajes_limpios_{mes_inicio}_{mes_fin}.parquet"
    if not ruta_limpios.exists():
        raise FileNotFoundError(f"Falta {ruta_limpios.name}; ejecutar `make data`")
    # Solo las columnas de modelado: así no se cargan usuario_id, genero ni anio_de_nacimiento
    columnas = [c for c in cfg.P3_COLUMNAS_MODELADO if c != "duracion_min"]
    modelado = p3_preparar_modelado(pd.read_parquet(ruta_limpios, columns=columnas))
    p3_validar_modelado(modelado, p3_cargar_estaciones())

    ruta_salida = p3_ruta_viajes_modelado(mes_inicio, mes_fin)
    modelado.to_parquet(ruta_salida, index=False)
    conteo = modelado["particion"].value_counts().reindex(["train", "valid", "test", "brecha"])
    logger.info("Partición: " + ", ".join(f"{k} {v}" for k, v in conteo.items()))
    logger.success(f"{ruta_salida.name}: {len(modelado)} registros")


if __name__ == "__main__":
    app()
