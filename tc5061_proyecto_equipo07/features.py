"""Features de P1 (demanda por estación-hora), P2 y P3 (clustering de estaciones).

P1: lee data/processed/viajes_limpios_AAAA-MM_AAAA-MM.parquet (`make data`) y deja
data/processed/p1_demanda_estacion_hora.parquet con el target y sus variables.
P2: lee data/processed/viajes_limpios_AAAA-MM_AAAA-MM.parquet (`make data`) y deja
data/processed/p2_cambios_estacion_bin.parquet
P3: lee data/processed/viajes_modelado_AAAA-MM_AAAA-MM.parquet (`dataset p3-particionar`) y deja
estaciones_train.parquet y demanda_estacion_hora.parquet en data/processed/.

Uso: python -m tc5061_proyecto_equipo07.features p1-features | p2-features | p3-features
"""

import time

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset, validacion

app = typer.Typer()


def p3_variables_de_tiempo(viajes: pd.DataFrame) -> pd.DataFrame:
    """Agrega duracion_min, hora y dia_semana (0 = lunes); `viajes` se modifica en su lugar."""
    viajes["duracion_min"] = (
        viajes["fin_del_viaje"] - viajes["inicio_del_viaje"]
    ).dt.total_seconds() / 60
    viajes["hora"] = viajes["inicio_del_viaje"].dt.hour.astype("int8")
    viajes["dia_semana"] = viajes["inicio_del_viaje"].dt.dayofweek.astype("int8")
    return viajes


def p3_perfil_de_estaciones(viajes_periodo: pd.DataFrame) -> pd.DataFrame:
    """Regresa una fila por estación con su perfil horario y sus indicadores de uso.

    Variables: proporción de salidas por hora de servicio en días laborales (`lab_hHH`) y en fin
    de semana (`fin_hHH`), volumen diario (`log_viajes_por_dia`), balance neto sobre salidas,
    proporción de viajes circulares y duración mediana. Requiere `duracion_min`.
    """
    inicio = viajes_periodo["inicio_del_viaje"]
    base = pd.DataFrame(
        {
            "estacion": viajes_periodo["origen_id"].to_numpy(),
            "hora": inicio.dt.hour.to_numpy(),
            "laboral": (inicio.dt.dayofweek < 5).to_numpy(),
        }
    )
    conteo = (
        base.groupby(["estacion", "laboral", "hora"])
        .size()
        .unstack("hora", fill_value=0)
        .reindex(columns=cfg.P3_HORAS_SERVICIO, fill_value=0)
    )
    estaciones_ids = conteo.index.get_level_values("estacion").unique()

    def perfil(es_laboral, prefijo):
        # Máscara en lugar de `xs`: si el periodo no tiene ningún día de este tipo, `xs` lanza
        # KeyError; con la máscara el bloque queda vacío y el reindex lo rellena con ceros
        del_tipo = conteo[conteo.index.get_level_values("laboral") == es_laboral]
        bloque = del_tipo.droplevel("laboral").reindex(estaciones_ids, fill_value=0)
        proporcion = bloque.div(bloque.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        proporcion.columns = [f"{prefijo}_h{hora:02d}" for hora in proporcion.columns]
        return proporcion

    perfiles = pd.concat([perfil(True, "lab"), perfil(False, "fin")], axis=1)

    salidas = viajes_periodo["origen_id"].value_counts()
    llegadas = viajes_periodo["destino_id"].value_counts()
    dias_activos = (
        pd.DataFrame(
            {
                "estacion": viajes_periodo["origen_id"].to_numpy(),
                "dia": inicio.dt.floor("D").to_numpy(),
            }
        )
        .groupby("estacion")["dia"]
        .nunique()
    )
    circulares = (
        (viajes_periodo["origen_id"] == viajes_periodo["destino_id"])
        .groupby(viajes_periodo["origen_id"])
        .mean()
    )
    duracion = viajes_periodo.groupby("origen_id", observed=True)["duracion_min"].median()

    indicadores = pd.DataFrame(
        {
            "salidas": salidas,
            "log_viajes_por_dia": np.log1p(salidas / dias_activos),
            "neto_sobre_salidas": (llegadas.reindex(salidas.index).fillna(0) - salidas) / salidas,
            "prop_circulares": circulares,
            "duracion_mediana_min": duracion,
        }
    ).reindex(estaciones_ids)
    indicadores.index = indicadores.index.astype(int)
    perfiles.index = perfiles.index.astype(int)
    return perfiles.join(indicadores)


def p3_construir_caracteristicas(
    limpios: pd.DataFrame,
    min_salidas_train: int = cfg.P3_MIN_SALIDAS_TRAIN,
    min_salidas_valid: int = cfg.P3_MIN_SALIDAS_VALID,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Regresa las variables por estación de train y de valid, y los nombres de las columnas.

    Con pocos viajes el perfil horario es ruido, así que se filtran las estaciones con menos de
    `min_salidas_train` salidas en train (y de `min_salidas_valid` en valid, para la prueba de
    estabilidad). `limpios` debe traer `particion` y `duracion_min`.
    """
    train = p3_perfil_de_estaciones(limpios[limpios["particion"] == "train"])
    train = train[train["salidas"] >= min_salidas_train]
    columnas_modelo = [columna for columna in train.columns if columna != "salidas"]
    valid = p3_perfil_de_estaciones(limpios[limpios["particion"] == "valid"])
    valid = valid[valid["salidas"] >= min_salidas_valid]
    return train, valid, columnas_modelo


def p3_tabla_estaciones(limpios: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Regresa capacidad, ubicación y actividad por estación, calculada solo con train."""
    entrenamiento = limpios[limpios["particion"] == "train"]
    actividad = pd.DataFrame(
        {
            "salidas_train": entrenamiento["origen_id"].value_counts(),
            "llegadas_train": entrenamiento["destino_id"].value_counts(),
        }
    )
    tabla = (
        estaciones.set_index("id")[["name", "latitude", "longitude", "dpcapacity"]]
        .join(actividad)
        .fillna({"salidas_train": 0, "llegadas_train": 0})
    )
    columnas = ["salidas_train", "llegadas_train"]
    tabla[columnas] = tabla[columnas].astype(int)
    tabla["neto_train"] = tabla["llegadas_train"] - tabla["salidas_train"]
    tabla["capacidad_cero"] = tabla["dpcapacity"] == 0
    tabla["activa_en_train"] = tabla["salidas_train"] > 0
    return tabla


def p3_viajes_por_mes_y_conjunto(limpios: pd.DataFrame) -> pd.DataFrame:
    """Regresa los viajes de cada mes (filas) por conjunto de la partición (columnas)."""
    return (
        limpios.groupby(["periodo_archivo", "particion"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "brecha", "valid", "test"], fill_value=0)
    )


def p3_demanda_estacion_hora(
    limpios: pd.DataFrame, mes_inicio: str = cfg.MES_INICIO, mes_fin: str = cfg.MES_FIN
) -> tuple[pd.DataFrame, int]:
    """Regresa (demanda por estación-hora, estación-horas con viajes fuera de la rejilla).

    La rejilla evita inventar datos: los ceros comienzan en la primera hora activa de cada
    estación, y se excluyen las horas sin servicio (1:00 a 4:59) y las de meses sin datos.
    """
    hora_inicio = limpios["inicio_del_viaje"].dt.floor("h")
    demanda = (
        pd.DataFrame({"origen_id": limpios["origen_id"], "hora_inicio": hora_inicio})
        .groupby(["origen_id", "hora_inicio"], observed=True)
        .size()
        .rename("viajes")
        .reset_index()
    )

    # Meses sin datos: se calculan de los datos, no de una lista fija
    meses_sin_datos = dataset.p3_meses_sin_datos(
        limpios["periodo_archivo"].astype(str).unique(), mes_inicio, mes_fin
    )

    # Horas válidas: dentro del rango, con servicio y en meses con datos
    ultima_hora = pd.Timestamp(mes_fin + "-01") + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)
    horas = pd.date_range(f"{mes_inicio}-01", ultima_hora, freq="h")
    meses_hora = horas.strftime("%Y-%m")
    horas_validas = horas[
        ~horas.hour.isin(cfg.HORAS_SIN_SERVICIO) & ~np.isin(meses_hora, meses_sin_datos)
    ]

    # Rejilla: para cada estación, todas las horas válidas desde su primera hora con viajes
    primera_hora = demanda.groupby("origen_id")["hora_inicio"].min()
    ids, instantes = [], []
    for estacion, primera in primera_hora.items():
        posicion = horas_validas.searchsorted(primera)
        ids.append(np.full(len(horas_validas) - posicion, estacion, dtype="int32"))
        instantes.append(horas_validas[posicion:].to_numpy())
    rejilla = pd.DataFrame(
        {"origen_id": np.concatenate(ids), "hora_inicio": np.concatenate(instantes)}
    )

    tabla = rejilla.merge(
        demanda.astype({"origen_id": "int32"}), on=["origen_id", "hora_inicio"], how="left"
    )
    tabla["viajes"] = tabla["viajes"].fillna(0).astype("int16")
    tabla["particion"] = dataset.p3_asignar_particion(tabla["hora_inicio"])

    excluidos = int(
        len(demanda) - demanda.merge(rejilla, on=["origen_id", "hora_inicio"]).shape[0]
    )
    return tabla, excluidos


# Horas de servicio por día: la malla de P1 tiene 20 columnas por día, así que un día = 20 posiciones
P1_HORAS_POR_DIA = 24 - len(cfg.HORAS_SIN_SERVICIO)


def p1_meses_sin_datos(periodos_en_datos, mes_inicio: str, mes_fin: str) -> list[str]:
    """Regresa los meses del rango que no aparecen en los datos (por ejemplo, retirados del portal)."""
    presentes = {str(periodo) for periodo in periodos_en_datos}
    rango = pd.period_range(mes_inicio, mes_fin, freq="M")
    return [str(mes) for mes in rango if str(mes) not in presentes]


def p1_cargar_estaciones() -> pd.DataFrame:
    """Lee la nomenclatura de estaciones más reciente de data/raw/estaciones/."""
    ruta = max(cfg.RUTA_ESTACIONES_CRUDAS.glob("nomenclatura_*.csv"), default=None)
    if ruta is None:
        raise FileNotFoundError(
            f"No hay nomenclatura en {cfg.RUTA_ESTACIONES_CRUDAS}; `make data`"
        )
    return pd.read_csv(ruta, encoding=dataset.detectar_codificacion(ruta.read_bytes()))


def p1_asignar_particion(hora_inicio: pd.Series) -> pd.Series:
    """Regresa train, valid, test o brecha de cada hora según la partición temporal del proyecto.

    Las horas previas a cada frontera (BRECHA_HORAS) quedan en `brecha`: con un horizonte de 24 h,
    sus rezagos y etiquetas cruzarían la frontera entre conjuntos.
    """
    frontera_valid = pd.Timestamp(cfg.FRONTERA_VALID)
    frontera_test = pd.Timestamp(cfg.FRONTERA_TEST)
    brecha = pd.Timedelta(hours=cfg.BRECHA_HORAS)
    conjunto = np.select(
        [
            hora_inicio < frontera_valid - brecha,
            hora_inicio < frontera_valid,
            hora_inicio < frontera_test - brecha,
            hora_inicio < frontera_test,
        ],
        ["train", "brecha", "valid", "brecha"],
        default="test",
    )
    categorias = ["train", "valid", "test", "brecha"]
    return pd.Series(pd.Categorical(conjunto, categories=categorias), index=hora_inicio.index)


def p1_horas_de_servicio(mes_inicio: str, mes_fin: str) -> pd.DatetimeIndex:
    """Regresa todas las horas con servicio del rango (sin las de 1:00 a 4:59, en que no opera).

    Incluye las horas de meses sin datos para que cada día tenga el mismo número de horas y un
    rezago de N días sea siempre un desplazamiento de N * 20 posiciones.
    """
    ultima_hora = pd.Period(mes_fin, "M").end_time.floor("h")
    horas = pd.date_range(f"{mes_inicio}-01", ultima_hora, freq="h")
    return horas[~horas.hour.isin(cfg.HORAS_SIN_SERVICIO)]


def p1_matriz_demanda(
    limpios: pd.DataFrame, horas: pd.DatetimeIndex, meses_sin_datos: list[str]
) -> tuple[np.ndarray, pd.Index, pd.Series]:
    """Cuenta los viajes iniciados por estación y hora en una matriz estaciones x horas.

    Regresa (matriz, ids de estación, primera hora con viajes de cada estación). Una hora sin
    viajes dentro del periodo de operación de la estación vale 0; antes de su primer viaje,
    después del último y en meses sin datos vale NaN, porque ahí no hay demanda que medir.
    """
    hora_inicio = limpios["inicio_del_viaje"].dt.floor("h")
    en_servicio = ~hora_inicio.dt.hour.isin(cfg.HORAS_SIN_SERVICIO)
    conteo = (
        pd.DataFrame(
            {
                "origen_id": limpios.loc[en_servicio, "origen_id"].to_numpy(),
                "hora": hora_inicio[en_servicio].to_numpy(),
            }
        )
        .groupby(["origen_id", "hora"])
        .size()
    )
    ids = conteo.index.get_level_values("origen_id").unique().sort_values()
    columnas = horas.get_indexer(conteo.index.get_level_values("hora"))
    if (columnas < 0).any():
        raise ValueError("Hay viajes fuera del rango de horas; revisar mes_inicio y mes_fin")

    matriz = np.zeros((len(ids), len(horas)), dtype="float32")
    matriz[ids.get_indexer(conteo.index.get_level_values("origen_id")), columnas] = conteo
    operacion = conteo.index.to_frame(index=False).groupby("origen_id")["hora"].agg(["min", "max"])
    operacion = operacion.reindex(ids)
    posicion = np.arange(len(horas))
    inicio = horas.searchsorted(operacion["min"])
    fin = horas.searchsorted(operacion["max"], side="right")
    matriz[(posicion < inicio[:, None]) | (posicion >= fin[:, None])] = np.nan
    matriz[:, np.isin(horas.strftime("%Y-%m"), meses_sin_datos)] = np.nan
    return matriz, ids, operacion["min"]


def p1_rezago(matriz: np.ndarray, dias: int) -> np.ndarray:
    """Regresa la demanda de la misma hora `dias` días antes (NaN si ese momento no existe).

    Con `dias` >= 1 la variable solo usa información disponible al cierre del día anterior, que es
    cuando se hace la predicción del día siguiente: no hay fuga de información del futuro.
    """
    if dias < 1:
        raise ValueError("Los rezagos de P1 deben ser de al menos un día (horizonte de 24 h)")
    posiciones = dias * P1_HORAS_POR_DIA
    rezagada = np.full_like(matriz, np.nan)
    rezagada[:, posiciones:] = matriz[:, :-posiciones]
    return rezagada


def p1_promedio_misma_hora(matriz: np.ndarray, semanas: int) -> np.ndarray:
    """Promedia la demanda de la misma hora y día de la semana en las `semanas` anteriores.

    Una sola estación-hora es ruidosa (autocorrelación semanal mediana de 0.40 en el EDA); el
    promedio de varias semanas estima mejor el nivel esperado. Ignora las semanas sin dato.
    """
    suma = np.zeros_like(matriz)
    validas = np.zeros_like(matriz)
    for semana in range(1, semanas + 1):
        rezagada = p1_rezago(matriz, 7 * semana)
        hay_dato = ~np.isnan(rezagada)
        suma[hay_dato] += rezagada[hay_dato]
        validas += hay_dato
    with np.errstate(invalid="ignore", divide="ignore"):
        return suma / validas  # 0 / 0 = NaN cuando ninguna semana tiene dato


def p1_dias_festivos(anios: list[int]) -> pd.DatetimeIndex:
    """Regresa los días de descanso obligatorio en México (Ley Federal del Trabajo, art. 74).

    En el EDA los días con menos viajes son festivos, por lo que la demanda cambia en esos días.
    """

    def lunes(anio: int, mes: int, numero: int) -> pd.Timestamp:
        dias = pd.date_range(f"{anio}-{mes:02d}-01", periods=7)
        return dias[dias.dayofweek == 0][0] + pd.Timedelta(weeks=numero - 1)

    festivos = []
    for anio in anios:
        festivos += [
            pd.Timestamp(anio, 1, 1),
            lunes(anio, 2, 1),  # Día de la Constitución
            lunes(anio, 3, 3),  # Natalicio de Benito Juárez
            pd.Timestamp(anio, 5, 1),
            pd.Timestamp(anio, 9, 16),
            lunes(anio, 11, 3),  # Día de la Revolución
            pd.Timestamp(anio, 12, 25),
        ]
        if (anio - 2024) % 6 == 0:
            festivos.append(pd.Timestamp(anio, 10, 1))  # Transmisión del Poder Ejecutivo
    return pd.DatetimeIndex(sorted(festivos))


def p1_variables_calendario(hora_inicio: pd.Series) -> pd.DataFrame:
    """Regresa hora, día de la semana, fin de semana, mes y día festivo de cada hora a predecir.

    Son las variables con más señal en el EDA: dos picos entre semana (8:00 y 18:00), un solo pico
    el fin de semana y menos viajes en festivos. Se conocen de antemano, así que no hay fuga.
    """
    anios = range(hora_inicio.dt.year.min(), hora_inicio.dt.year.max() + 1)
    festivos = p1_dias_festivos(list(anios))
    dia_semana = hora_inicio.dt.dayofweek
    return pd.DataFrame(
        {
            "hora": hora_inicio.dt.hour.astype("int8"),
            "dia_semana": dia_semana.astype("int8"),
            "es_fin_de_semana": (dia_semana >= 5).astype("int8"),
            "mes": hora_inicio.dt.month.astype("int8"),
            "es_festivo": hora_inicio.dt.normalize().isin(festivos).astype("int8"),
        },
        index=hora_inicio.index,
    )


def p1_variables_estacion(estaciones: pd.DataFrame) -> pd.DataFrame:
    """Regresa capacidad, ubicación y zona (municipio del código del nombre) por id de estación.

    El nivel de demanda cambia mucho entre estaciones (de 1.51 a 229.91 viajes por día en el
    EDA) y se concentra en el centro de la red, por lo que la ubicación ayuda a distinguirlas.
    """
    zona = estaciones["name"].str.extract(r"([A-Z]{3})-\d{3}", expand=False).fillna("SIN_ZONA")
    return pd.DataFrame(
        {
            "capacidad": estaciones["dpcapacity"].astype("int16").to_numpy(),
            "latitud": estaciones["latitude"].astype("float32").to_numpy(),
            "longitud": estaciones["longitude"].astype("float32").to_numpy(),
            "zona": pd.Categorical(zona),
        },
        index=pd.Index(estaciones["id"].astype("int32"), name="origen_id"),
    )


def p1_tabla_modelado(
    limpios: pd.DataFrame,
    estaciones: pd.DataFrame,
    mes_inicio: str = cfg.MES_INICIO,
    mes_fin: str = cfg.MES_FIN,
) -> pd.DataFrame:
    """Regresa una fila por estación y hora de servicio con el target `viajes` y sus variables.

    Requiere `inicio_del_viaje`, `origen_id` y `periodo_archivo` de los viajes limpios. Cada fila
    trae la partición temporal del proyecto; las horas en `brecha` no se usan para entrenar ni
    evaluar.
    """
    meses_sin_datos = p1_meses_sin_datos(limpios["periodo_archivo"].unique(), mes_inicio, mes_fin)
    horas = p1_horas_de_servicio(mes_inicio, mes_fin)
    matriz, ids, apertura = p1_matriz_demanda(limpios, horas, meses_sin_datos)

    variables = {f"viajes_rezago_{dias}d": p1_rezago(matriz, dias) for dias in cfg.P1_REZAGOS_DIAS}
    promedio = f"viajes_promedio_{cfg.P1_SEMANAS_PROMEDIO}sem"
    variables[promedio] = p1_promedio_misma_hora(matriz, cfg.P1_SEMANAS_PROMEDIO)

    # Solo las estación-horas dentro del periodo de operación (las celdas que no son NaN)
    filas, columnas = np.nonzero(~np.isnan(matriz))
    tabla = pd.DataFrame(
        {
            "origen_id": ids.to_numpy()[filas].astype("int32"),
            "hora_inicio": horas[columnas],
            "viajes": matriz[filas, columnas].astype("int16"),
        }
    )
    for nombre, valores in variables.items():
        tabla[nombre] = valores[filas, columnas]
    del matriz, variables

    tabla = pd.concat([tabla, p1_variables_calendario(tabla["hora_inicio"])], axis=1)
    tabla = tabla.join(p1_variables_estacion(estaciones), on="origen_id")
    dias_operando = tabla["hora_inicio"] - tabla["origen_id"].map(apertura).dt.normalize()
    tabla["dias_desde_apertura"] = dias_operando.dt.days.astype("int16")
    tabla["particion"] = p1_asignar_particion(tabla["hora_inicio"])
    return tabla


# Grupos del nivel histórico: por estación, día y hora; y por día y hora en todo el sistema
P1_GRUPO_ESTACION = ["origen_id", "dia_semana", "hora"]
P1_GRUPO_SISTEMA = ["dia_semana", "hora"]


class P1NivelEstacion(BaseEstimator, TransformerMixin):
    """Agrega el nivel histórico de demanda de cada estación, aprendido solo con los datos de `fit`.

    `fit` promedia los viajes por estación, día de la semana y hora (`viajes_nivel_estacion`) y por
    día de la semana y hora en todo el sistema (`viajes_nivel_sistema`). Una estación que no existía
    en train recibe el nivel del sistema como nivel de estación: en train esa columna nunca está
    vacía, así que un hueco en valid o test haría que el modelo predijera cualquier cosa. Dentro
    de un Pipeline, `fit` solo ve train, así que valid y test no se filtran en el nivel. `transform` quita `origen_id` (el id no es una cantidad) y deja `zona`
    como categoría con las zonas vistas en train, para que el modelo acepte texto al predecir.
    """

    def fit(self, X: pd.DataFrame, y) -> "P1NivelEstacion":
        datos = X[P1_GRUPO_ESTACION].astype("int64")
        datos["viajes"] = np.asarray(y, dtype="float64")
        self.nivel_estacion_ = datos.groupby(P1_GRUPO_ESTACION)["viajes"].mean()
        self.nivel_sistema_ = datos.groupby(P1_GRUPO_SISTEMA)["viajes"].mean()
        self.zonas_ = sorted(X["zona"].astype(str).unique())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        llaves = X[P1_GRUPO_ESTACION].astype("int64")
        estacion = self.nivel_estacion_.reindex(pd.MultiIndex.from_frame(llaves)).to_numpy()
        sistema = self.nivel_sistema_.reindex(
            pd.MultiIndex.from_frame(llaves[P1_GRUPO_SISTEMA])
        ).to_numpy()
        salida = X.drop(columns="origen_id")
        salida["zona"] = pd.Categorical(X["zona"].astype(str), categories=self.zonas_)
        # Estación que no existía en train: usa el nivel del sistema
        estacion = np.where(np.isnan(estacion), sistema, estacion)
        salida["viajes_nivel_estacion"] = estacion.astype("float32")
        salida["viajes_nivel_sistema"] = sistema.astype("float32")
        return salida


@app.callback()
def p3_ayuda_features():
    """Features de P1 (demanda por estación-hora), P2 (riesgo por estación-bin) y P3 (tablas de
    estaciones y de demanda)."""


@app.command()
def p1_features(mes_inicio: str = cfg.MES_INICIO, mes_fin: str = cfg.MES_FIN):
    """Deja p1_demanda_estacion_hora.parquet en data/processed/ (target y variables de P1)."""
    ruta = cfg.PROCESSED_DATA_DIR / f"viajes_limpios_{mes_inicio}_{mes_fin}.parquet"
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta.name}; ejecutar `make data`")
    limpios = pd.read_parquet(ruta, columns=["inicio_del_viaje", "origen_id", "periodo_archivo"])
    tabla = p1_tabla_modelado(limpios, p1_cargar_estaciones(), mes_inicio, mes_fin)

    # Control: la tabla debe contener todos los viajes limpios que inician en horas de servicio
    en_servicio = ~limpios["inicio_del_viaje"].dt.hour.isin(cfg.HORAS_SIN_SERVICIO)
    esperados, en_tabla = int(en_servicio.sum()), int(tabla["viajes"].sum())
    if en_tabla != esperados:
        raise ValueError(f"La tabla suma {en_tabla} viajes y se esperaban {esperados}")

    cfg.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(cfg.RUTA_P1_DEMANDA, index=False)
    por_conjunto = tabla.groupby("particion", observed=True)["viajes"].agg(["size", "sum"])
    logger.info(f"Viajes en horas sin servicio (fuera de P1): {len(limpios) - esperados}")
    logger.info(f"Estación-horas y viajes por conjunto:\n{por_conjunto.to_string()}")
    logger.success(f"{cfg.RUTA_P1_DEMANDA.name}: {len(tabla)} filas y {en_tabla} viajes")


def p1_revisar_tabla_completa(tabla: pd.DataFrame) -> pd.DataFrame:
    """Revisiones que no admiten muestra: llave, target y orden de los conjuntos.

    Un solo caso basta para invalidar la tabla, así que se cuentan sobre todas las filas.
    """
    usados = tabla[tabla["particion"] != "brecha"]
    faltan = {"train", "valid", "test"} - set(usados["particion"].astype(str).unique())
    if faltan:
        raise ValueError(f"La tabla de P1 no tiene filas de {sorted(faltan)}")
    fin = usados.groupby("particion", observed=True)["hora_inicio"].max()
    hora, conjunto = usados["hora_inicio"], usados["particion"]
    valid_antes = (conjunto == "valid") & (hora <= fin["train"])
    test_antes = (conjunto == "test") & (hora <= fin["valid"])
    problemas = {
        "llave estación-hora duplicada": tabla.duplicated(["origen_id", "hora_inicio"]).sum(),
        "target viajes nulo": tabla["viajes"].isna().sum(),
        "target viajes negativo": (tabla["viajes"] < 0).sum(),
        "horas de valid no posteriores a train": valid_antes.sum(),
        "horas de test no posteriores a valid": test_antes.sum(),
    }
    revision = pd.DataFrame(
        {"revision": list(problemas), "filas_con_problema": [int(n) for n in problemas.values()]}
    )
    revision["resultado"] = np.where(revision["filas_con_problema"] == 0, "PASS", "FAIL")
    return revision


@app.command()
def p1_validar(muestra: int = cfg.P1_MUESTRA_VALIDACION):
    """Valida la tabla de P1: revisiones sobre la tabla completa y DeepChecks sobre una muestra."""
    if not cfg.RUTA_P1_DEMANDA.exists():
        raise FileNotFoundError(f"Falta {cfg.RUTA_P1_DEMANDA.name}; ejecutar `make features-p1`")
    inicio = time.perf_counter()
    carpeta = validacion.VALIDACION_DIR
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla = pd.read_parquet(cfg.RUTA_P1_DEMANDA)
    memoria = tabla.memory_usage(deep=True).sum() / 1e9
    logger.info(f"{cfg.RUTA_P1_DEMANDA.name}: {len(tabla)} filas, {memoria:.2f} GB en memoria")

    revision = p1_revisar_tabla_completa(tabla)
    revision.to_csv(carpeta / "p1_tabla_completa.csv", index=False)
    logger.info(f"Revisiones sobre la tabla completa:\n{revision.to_string(index=False)}")
    if (revision["resultado"] == "FAIL").any():
        raise ValueError(
            "La tabla de P1 no pasa las revisiones completas; ver p1_tabla_completa.csv"
        )

    # Las horas en `brecha` no entran a entrenar ni evaluar; se liberan antes de copiar la tabla
    usados = tabla[tabla["particion"] != "brecha"].reset_index(drop=True)
    del tabla
    usados["particion"] = usados["particion"].astype(str)
    for conjunto, filas in usados["particion"].value_counts().items():
        tomadas = min(muestra, filas)
        logger.info(f"Muestra de {conjunto}: {tomadas} de {filas} filas ({tomadas / filas:.1%})")

    # origen_id como categórica: DeepChecks reporta las estaciones que aparecen solo en prueba
    for comparado in ("valid", "test"):
        validacion.validar_tabla(
            f"p1_train_{comparado}",
            usados,
            "particion",
            "viajes",
            ["origen_id", "zona"],
            "hora_inicio",
            conjuntos=("train", comparado),
            umbral_nulos=cfg.P1_UMBRAL_NULOS,
            muestra=muestra,
            carpeta=carpeta,
        )
    logger.success(
        f"Validación de P1 en {(time.perf_counter() - inicio) / 60:.1f} min ({carpeta})"
    )


def p3_revisar_tabla_completa(
    modelado: pd.DataFrame, estaciones: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Revisiones de P3 que no admiten muestra: llave, duración, fechas, catálogo y orden.

    Un solo viaje basta para invalidar la tabla, así que se cuentan sobre todas las filas.
    P3 no usa test, pero la tabla lo trae: solo se exige que train y valid existan.
    """
    usados = modelado[modelado["particion"] != "brecha"]
    faltan = {"train", "valid"} - set(usados["particion"].astype(str).unique())
    if faltan:
        raise ValueError(f"La tabla de P3 no tiene viajes de {sorted(faltan)}")
    fin_train = usados.loc[usados["particion"] == "train", "inicio_del_viaje"].max()
    duracion = modelado["duracion_min"]
    nulos = modelado[["inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]].isna()
    problemas = {
        "viaje_id repetido": modelado["viaje_id"].duplicated().sum(),
        "fechas o estaciones nulas": nulos.any(axis=1).sum(),
        "duración nula o no positiva": (duracion.isna() | (duracion <= 0)).sum(),
        f"duración mayor a {cfg.DURACION_MAXIMA_MIN} min": (
            duracion > cfg.DURACION_MAXIMA_MIN
        ).sum(),
        "fin anterior al inicio": (modelado["fin_del_viaje"] < modelado["inicio_del_viaje"]).sum(),
        "viajes de valid no posteriores a train": (
            (usados["particion"] == "valid") & (usados["inicio_del_viaje"] <= fin_train)
        ).sum(),
    }
    if estaciones is not None:
        catalogo = estaciones["id"]
        problemas["origen o destino fuera del catálogo"] = (
            ~(modelado["origen_id"].isin(catalogo) & modelado["destino_id"].isin(catalogo))
        ).sum()
    revision = pd.DataFrame(
        {"revision": list(problemas), "filas_con_problema": [int(n) for n in problemas.values()]}
    )
    revision["resultado"] = np.where(revision["filas_con_problema"] == 0, "PASS", "FAIL")
    return revision


@app.command()
def p3_validar(muestra: int = cfg.P3_MUESTRA_VALIDACION):
    """Valida la tabla de P3: revisiones sobre la tabla completa y DeepChecks de train vs valid."""
    ruta = dataset.p3_ruta_viajes_modelado()
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta.name}; ejecutar `make features-p3`")
    inicio = time.perf_counter()
    carpeta = validacion.VALIDACION_DIR
    carpeta.mkdir(parents=True, exist_ok=True)
    modelado = pd.read_parquet(ruta)
    memoria = modelado.memory_usage(deep=True).sum() / 1e9
    logger.info(f"{ruta.name}: {len(modelado)} filas, {memoria:.2f} GB en memoria")

    revision = p3_revisar_tabla_completa(modelado, dataset.p3_cargar_estaciones())
    revision.to_csv(carpeta / "p3_tabla_completa.csv", index=False)
    logger.info(f"Revisiones sobre la tabla completa:\n{revision.to_string(index=False)}")
    if (revision["resultado"] == "FAIL").any():
        raise ValueError(
            "La tabla de P3 no pasa las revisiones completas; ver p3_tabla_completa.csv"
        )

    # P3 no usa test y la brecha no entra a ningún conjunto. Se descartan también las columnas
    # que solo identifican o repiten el tiempo: con viaje_id, fin_del_viaje y periodo_archivo,
    # DeepChecks reportaría drift del calendario
    usados = modelado.loc[
        modelado["particion"].isin(["train", "valid"]),
        ["particion", "inicio_del_viaje", "origen_id", "destino_id", "duracion_min"],
    ].reset_index(drop=True)
    del modelado
    usados["particion"] = usados["particion"].astype(str)
    for conjunto, filas in usados["particion"].value_counts().items():
        tomadas = min(muestra, filas)
        logger.info(f"Muestra de {conjunto}: {tomadas} de {filas} filas ({tomadas / filas:.1%})")

    # P3 compara train con valid; origen_id y destino_id como categóricas reportan estaciones nuevas
    validacion.validar_tabla(
        "p3_train_valid",
        usados,
        "particion",
        "duracion_min",
        ["origen_id", "destino_id"],
        "inicio_del_viaje",
        conjuntos=("train", "valid"),
        umbral_nulos=cfg.P3_UMBRAL_NULOS,
        muestra=muestra,
        carpeta=carpeta,
    )
    logger.success(
        f"Validación de P3 en {(time.perf_counter() - inicio) / 60:.1f} min ({carpeta})"
    )


@app.command()
def p3_features(mes_inicio: str = cfg.MES_INICIO, mes_fin: str = cfg.MES_FIN):
    """Deja estaciones_train.parquet y demanda_estacion_hora.parquet en data/processed/."""
    ruta = dataset.p3_ruta_viajes_modelado(mes_inicio, mes_fin)
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta.name}; ejecutar `make data` y `p3-particionar`")
    estaciones = dataset.p3_cargar_estaciones()

    # Cada tabla lee solo las columnas que necesita (menos memoria en equipos pequeños y en CI)
    limpios = pd.read_parquet(ruta, columns=["particion", "origen_id", "destino_id"])
    tabla = p3_tabla_estaciones(limpios, estaciones)
    del limpios
    cfg.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(cfg.RUTA_ESTACIONES_TRAIN)
    logger.info(
        f"Estaciones: {len(tabla)} | activas en train: {int(tabla['activa_en_train'].sum())}"
    )

    limpios = pd.read_parquet(ruta, columns=["inicio_del_viaje", "origen_id", "periodo_archivo"])
    demanda, excluidos = p3_demanda_estacion_hora(limpios, mes_inicio, mes_fin)
    demanda.to_parquet(cfg.RUTA_DEMANDA_ESTACION_HORA, index=False)
    logger.info(f"Demanda estación-hora: {len(demanda)} filas | fuera de la rejilla: {excluidos}")
    logger.success("Features de P3 listas en data/processed/")


@app.command()
def p2_features():
    """Deja p2_cambios_estacion_bin.parquet en data/processed/ (etiqueta y variables de P2)."""
    viajes, estaciones = p2_cargar_viajes()
    cambios = p2_construir_caracteristicas(viajes, estaciones)
    del viajes
    cfg.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cambios.to_parquet(cfg.RUTA_P2_CAMBIOS, index=False)
    logger.success(f"{cfg.RUTA_P2_CAMBIOS.name}: {len(cambios)} filas estación-bin")


# ---------------------------------------------------------------------------------------------
# P2 (clasificación de riesgos): reglas de limpieza propias de la pregunta.
# Entrada: data/processed/viajes_limpios_AAAA-MM_AAAA-MM.parquet (`dataset limpiar`).
# ---------------------------------------------------------------------------------------------


def p2_estaciones_con_capacidad(estaciones: pd.DataFrame) -> pd.DataFrame:
    """Regresa solo las estaciones con capacidad mayor a 0 (P2 analiza la capacidad)."""
    return estaciones[estaciones["dpcapacity"] > 0]


def p2_descartar_estaciones_inexistentes(
    viajes: pd.DataFrame, estaciones: pd.DataFrame
) -> pd.DataFrame:
    """Regresa los viajes cuyo origen y destino están en `estaciones`."""
    ids = estaciones["id"].unique()
    return viajes[viajes["origen_id"].isin(ids) & viajes["destino_id"].isin(ids)]


def p2_descartar_fuera_de_rango(
    viajes: pd.DataFrame, fecha_limite: str = cfg.P2_FECHA_LIMITE_FIN_VIAJE
) -> pd.DataFrame:
    """Regresa los viajes que terminan antes de `fecha_limite`."""
    return viajes[viajes["fin_del_viaje"] < fecha_limite]


def p2_descartar_horas_sin_servicio(
    viajes: pd.DataFrame, horas_sin_servicio=cfg.P2_HORAS_SIN_SERVICIO
) -> pd.DataFrame:
    """Regresa los viajes que inician en horas con servicio (se consideran anómalos los demás)."""
    return viajes[~viajes["inicio_del_viaje"].dt.hour.isin(horas_sin_servicio)]


def p2_limpiar_viajes(viajes: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Aplica en orden las reglas de limpieza de P2."""
    viajes = p2_descartar_estaciones_inexistentes(viajes, p2_estaciones_con_capacidad(estaciones))
    viajes = p2_descartar_fuera_de_rango(viajes)
    return p2_descartar_horas_sin_servicio(viajes)


def p2_cargar_viajes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regresa (viajes limpios para P2, estaciones con capacidad) listos para el modelado."""
    if not cfg.RUTA_VIAJES_LIMPIOS.exists():
        raise FileNotFoundError(f"Falta {cfg.RUTA_VIAJES_LIMPIOS.name}; ejecutar `make data`")
    # Solo las columnas que usa P2: menos memoria y los datos personales no pasan
    viajes = pd.read_parquet(
        cfg.RUTA_VIAJES_LIMPIOS,
        columns=["inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"],
    )
    estaciones = p2_estaciones_con_capacidad(dataset.p3_cargar_estaciones())

    limpios = p2_limpiar_viajes(viajes, estaciones)
    logger.info(f"Viajes P2: {len(limpios)} registros ({len(viajes) - len(limpios)} retirados)")
    logger.success(f"Limpieza de P2 lista: {len(limpios)} viajes y {len(estaciones)} estaciones")
    return limpios, estaciones


def p2_intervalo(ts: pd.Series, bin_min: int) -> pd.Series:
    """Regresa el bin de cada instante; los de horas sin servicio van al último bin de las 0:00."""
    b = ts.dt.floor(f"{bin_min}min")
    ultimo = b.dt.floor("D") + pd.Timedelta(hours=1) - pd.Timedelta(minutes=bin_min)
    return b.where(~b.dt.hour.isin(cfg.P2_HORAS_SIN_SERVICIO), ultimo)


def p2_obtener_dataframe_de_eventos_con_bins(viajes: pd.DataFrame, bin_min: int) -> pd.DataFrame:
    """Regresa salidas y entradas por estación y bin (solo los bins con actividad)."""
    salidas = (
        viajes.groupby(["origen_id", p2_intervalo(viajes["inicio_del_viaje"], bin_min)])
        .size()
        .rename("salidas")
    )
    entradas = (
        viajes.groupby(["destino_id", p2_intervalo(viajes["fin_del_viaje"], bin_min)])
        .size()
        .rename("entradas")
    )
    salidas.index.names = entradas.index.names = ["estacion", "bin"]
    return pd.concat([salidas, entradas], axis=1).fillna(0).astype("int32")


def p2_generar_arreglo_de_bins(eventos: pd.DataFrame, bin_min: int) -> pd.DatetimeIndex:
    """Regresa los bins del primer día al último bin de `eventos`, sin horas sin servicio."""
    bins = pd.date_range(
        eventos.index.get_level_values("bin").min().floor("D"),
        eventos.index.get_level_values("bin").max(),
        freq=f"{bin_min}min",
    )
    return bins[~bins.hour.isin(cfg.P2_HORAS_SIN_SERVICIO)]


def p2_obtener_dataframe_de_cambios(viajes: pd.DataFrame, bin_min: int) -> pd.DataFrame:
    """Regresa salidas y entradas para cada estación y cada bin (los ceros también son filas)."""
    eventos = p2_obtener_dataframe_de_eventos_con_bins(viajes, bin_min)
    bins = p2_generar_arreglo_de_bins(eventos, bin_min)
    rejilla = pd.MultiIndex.from_product(
        [eventos.index.get_level_values("estacion").unique(), bins], names=["estacion", "bin"]
    )
    return eventos.reindex(rejilla, fill_value=0).reset_index()


def p2_agregar_suma_acumulada_a_cambios(cambios: pd.DataFrame) -> pd.DataFrame:
    """Agrega neto, dia (de servicio, 05:00 a 00:59) y cumsum_dia; quita estación-días sin viajes."""
    cambios["neto"] = cambios["entradas"] - cambios["salidas"]
    cambios["dia"] = (cambios["bin"] - pd.Timedelta(hours=5)).dt.floor("D")

    actividad = (
        cambios.groupby(["estacion", "dia"])[["salidas", "entradas"]].transform("sum").sum(axis=1)
    )
    cambios = cambios[actividad > 0]
    cambios = cambios.sort_values(["estacion", "bin"]).reset_index(drop=True)
    cambios["cumsum_dia"] = cambios.groupby(["estacion", "dia"], sort=False)["neto"].cumsum()
    return cambios


def p2_agregar_datos_de_estacion_a_cambios(
    cambios: pd.DataFrame, estaciones: pd.DataFrame
) -> pd.DataFrame:
    """Agrega latitud, longitud y capacidad de cada estación."""
    cambios = cambios.merge(
        estaciones[["id", "latitude", "longitude", "dpcapacity"]],
        left_on="estacion",
        right_on="id",
        how="left",
    )
    return cambios.drop("id", axis=1)


def p2_obtener_mayores_cambios_en_proximos_h_bins(
    cambios: pd.DataFrame, h: int
) -> tuple[pd.Series, pd.Series]:
    """Regresa (vacia, llena): peor salida y peor entrada netas en t+1..t+h, sobre la capacidad.

    Son NaN cuando la ventana futura se sale del día de servicio.
    """
    cumsums = cambios.groupby(["estacion", "dia"], sort=False)["cumsum_dia"]
    cambios_futuros = pd.concat(
        [cumsums.shift(-k) - cambios["cumsum_dia"] for k in range(1, h + 1)], axis=1
    )
    vacia = (-cambios_futuros).max(axis=1, skipna=False).clip(lower=0) / cambios["dpcapacity"]
    llena = cambios_futuros.max(axis=1, skipna=False).clip(lower=0) / cambios["dpcapacity"]
    return vacia, llena


def p2_agregar_clase_de_riesgo(cambios: pd.DataFrame, alfa: float = cfg.P2_ALFA) -> pd.DataFrame:
    """Agrega clase: 0 = sin riesgo, 1 = sin bicis, 2 = sin anclajes.

    Si una ventana tiene ambos eventos gana el de mayor estrés relativo. Se quitan las filas cuya
    ventana futura se sale del día.
    """
    datos = cambios.dropna(subset=["vacia", "llena"]).copy()
    vacia_positiva, llena_positiva = datos["vacia"] >= alfa, datos["llena"] >= alfa
    datos["clase"] = 0
    datos.loc[vacia_positiva, "clase"] = 1
    datos.loc[llena_positiva, "clase"] = 2
    ambas = vacia_positiva & llena_positiva
    datos.loc[ambas, "clase"] = np.where(
        datos.loc[ambas, "vacia"] >= datos.loc[ambas, "llena"], 1, 2
    )
    logger.info(
        f"Ventanas con ambos eventos: {int(ambas.sum())} de "
        f"{int((vacia_positiva | llena_positiva).sum())} positivas"
    )
    logger.info(
        "Distribución de clases: "
        f"{datos['clase'].value_counts(normalize=True).sort_index().round(4).to_dict()}"
    )
    return datos


def p2_ventana(cambios: pd.DataFrame, col: str, k: int) -> pd.Series:
    """Regresa la suma de `col` en los últimos k bins del mismo estación-día (incluye el actual)."""
    c = cambios.groupby(["estacion", "dia"], sort=False)[col].cumsum()
    return c - c.groupby([cambios["estacion"], cambios["dia"]], sort=False).shift(k, fill_value=0)


def p2_agregar_caracteristicas_retrospectivas(cambios: pd.DataFrame) -> pd.DataFrame:
    """Agrega neto relativo a la capacidad (1, 2, 4 y 8 bins) y salidas y entradas (2 y 4 bins)."""
    for k in (1, 2, 4, 8):
        cambios[f"neto_{k}_relativo"] = p2_ventana(cambios, "neto", k) / cambios["dpcapacity"]
    for k in (2, 4):
        cambios[f"salidas_{k}"] = p2_ventana(cambios, "salidas", k)
        cambios[f"entradas_{k}"] = p2_ventana(cambios, "entradas", k)
    return cambios


def p2_agregar_caracteristicas_de_hora(cambios: pd.DataFrame, bin_min: int) -> pd.DataFrame:
    """Agrega hora_idx y su codificación cíclica (hora_sin, hora_cos)."""
    cambios["hora_idx"] = (cambios["bin"].dt.hour * 60 + cambios["bin"].dt.minute) // bin_min
    bins_por_dia = 60 * 24 / bin_min
    cambios["hora_sin"] = np.sin(2 * np.pi * cambios["hora_idx"] / bins_por_dia)
    cambios["hora_cos"] = np.cos(2 * np.pi * cambios["hora_idx"] / bins_por_dia)
    return cambios


def p2_construir_caracteristicas(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
    bin_min: int = cfg.P2_BIN_MIN,
    h_bins: int = cfg.P2_H_BINS,
    alfa: float = cfg.P2_ALFA,
) -> pd.DataFrame:
    """Regresa la tabla estación-bin con las columnas de P2_FEATURES_SIN_HISTORICO y la clase a predecir.
    Limitación importante: por definición del planteamiento, los últimos h_bins registros por día
    no tendrán clase de riesgo a clasificar. Estos registros se filtran.
    """
    cambios = p2_obtener_dataframe_de_cambios(viajes, bin_min)
    # Cada viaje es exactamente una salida y una entrada
    if cambios["salidas"].sum() != len(viajes) or cambios["entradas"].sum() != len(viajes):
        raise ValueError("Las salidas o entradas por bin no coinciden con el número de viajes")

    cambios = p2_agregar_suma_acumulada_a_cambios(cambios)
    cambios = p2_agregar_datos_de_estacion_a_cambios(cambios, estaciones)
    cambios["vacia"], cambios["llena"] = p2_obtener_mayores_cambios_en_proximos_h_bins(
        cambios, h_bins
    )
    cambios = p2_agregar_clase_de_riesgo(cambios, alfa)

    cambios = p2_agregar_caracteristicas_retrospectivas(cambios)
    cambios["cumsum_dia_relativo"] = cambios["cumsum_dia"] / cambios["dpcapacity"]
    cambios = p2_agregar_caracteristicas_de_hora(cambios, bin_min)
    cambios["es_fin_semana"] = (cambios["bin"].dt.dayofweek >= 5).astype(int)

    # Optimización de memoria
    columnas_salida = cfg.P2_FEATURES_SIN_HISTORICO + ["clase", "estacion", "dia", "hora_idx"]
    cambios = cambios[columnas_salida]
    for c in cambios.select_dtypes("integer").columns:
        cambios[c] = pd.to_numeric(cambios[c], downcast="integer")  # sin pérdida

    for c in cambios.select_dtypes("float64").columns:  # decimales a float32
        cambios[c] = cambios[c].astype("float32")

    logger.success(
        f"Features de P2 listas: {len(cambios)} filas estación-bin, "
        f"{cambios['estacion'].nunique()} estaciones, {len(cfg.P2_FEATURES_SIN_HISTORICO)} features"
    )
    return cambios


def p2_asignar_particion(dia: pd.Series) -> pd.Series:
    """Regresa train, valid, test o brecha de cada día de servicio según la partición del proyecto.

    Se asigna por día de servicio (y no por bin) para no partir un día entre dos conjuntos: la
    etiqueta y las ventanas de P2 nunca cruzan el día, así que el día previo a cada frontera
    (BRECHA_HORAS) basta para que ningún conjunto use viajes del siguiente.
    """
    frontera_valid = pd.Timestamp(cfg.FRONTERA_VALID)
    frontera_test = pd.Timestamp(cfg.FRONTERA_TEST)
    brecha = pd.Timedelta(hours=cfg.BRECHA_HORAS)
    conjunto = np.select(
        [
            dia < frontera_valid - brecha,
            dia < frontera_valid,
            dia < frontera_test - brecha,
            dia < frontera_test,
        ],
        ["train", "brecha", "valid", "brecha"],
        default="test",
    )
    categorias = ["train", "valid", "test", "brecha"]
    return pd.Series(pd.Categorical(conjunto, categories=categorias), index=dia.index)


class P2HistoriaHastaAyer(BaseEstimator):
    """Frecuencia de cada riesgo en los días previos de la misma estación, tipo de día y hora,
    suavizada hacia la frecuencia global del periodo de fit. No usa el día actual ni el futuro.

    fit() aprende solo la frecuencia global; transform() calcula la historia sobre la tabla que
    reciba (necesita estacion, es_fin_semana, hora_idx, dia y clase).
    """

    def __init__(self, peso_historico: int = cfg.P2_PESO_HISTORICO):
        self.peso_historico = peso_historico

    def fit(self, X: pd.DataFrame, y=None) -> "P2HistoriaHastaAyer":
        self.frec_global_ = {
            "sin_bicis": float((X["clase"] == 1).mean()),
            "sin_anclajes": float((X["clase"] == 2).mean()),
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "frec_global_")
        X = X.sort_values(["estacion", "es_fin_semana", "hora_idx", "dia"])
        X["es_sin_bicis"] = (X["clase"] == 1).astype("int8")
        X["es_sin_anclajes"] = (X["clase"] == 2).astype("int8")

        # Días previos de cada estación x tipo de día x hora; la suma excluye el día actual y el
        # peso_historico jala la frecuencia hacia la global cuando hay pocos días de historia
        g = X.groupby(["estacion", "es_fin_semana", "hora_idx"], sort=False)
        dias_previos = g.cumcount()
        f = self.frec_global_
        X["hist_sin_bicis"] = (
            g["es_sin_bicis"].cumsum() - X["es_sin_bicis"] + self.peso_historico * f["sin_bicis"]
        ) / (dias_previos + self.peso_historico)
        X["hist_sin_anclajes"] = (
            g["es_sin_anclajes"].cumsum()
            - X["es_sin_anclajes"]
            + self.peso_historico * f["sin_anclajes"]
        ) / (dias_previos + self.peso_historico)
        return X


def p2_separar_fit_val_test(
    cambios: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Regresa (fit, val, test) según la partición del proyecto; los días en `brecha` se descartan."""
    particion = p2_asignar_particion(cambios["dia"])
    return tuple(cambios[particion == conjunto] for conjunto in ["train", "valid", "test"])


def p2_preparar_conjuntos(
    cambios: pd.DataFrame, peso_historico: int = cfg.P2_PESO_HISTORICO
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Agrega la historia "hasta ayer" y regresa (fit, val, test).

    La frecuencia global de la historia sale solo del conjunto de entrenamiento. La historia de
    validación y test usa las etiquetas de días anteriores, que ya se conocen al predecir.
    """
    en_train = p2_asignar_particion(cambios["dia"]) == "train"
    historia = P2HistoriaHastaAyer(peso_historico).fit(cambios[en_train])
    cambios = historia.transform(cambios)
    fit, val, test = p2_separar_fit_val_test(cambios)

    for nombre, conjunto in [("train", fit), ("valid", val), ("test", test)]:
        logger.info(
            f"{nombre}: {conjunto['dia'].min().date()} a {conjunto['dia'].max().date()} | "
            f"{len(conjunto)} filas"
        )
    prevalencia = {cfg.P2_CLASES[k]: round(float((test["clase"] == k).mean()), 4) for k in (1, 2)}
    logger.info(f"Prevalencia en test: {prevalencia}")
    return fit, val, test


def p2_tabla_validacion(cambios: pd.DataFrame) -> pd.DataFrame:
    """fit, valid y test en una sola tabla con la columna `particion`, ya con la historia.

    Se valida lo mismo con lo que se entrena; los días en `brecha` no entran.
    """
    conjuntos = p2_preparar_conjuntos(cambios)
    columnas = cfg.P2_FEATURES_CON_HISTORICO + ["clase", "estacion", "dia", "hora_idx"]
    return pd.concat(
        [c[columnas].assign(particion=n) for n, c in zip(["train", "valid", "test"], conjuntos)],
        ignore_index=True,
    )


def p2_revisar_tabla_completa(tabla: pd.DataFrame) -> pd.DataFrame:
    """Revisiones que no admiten muestra: llave, clase, historia y orden de los conjuntos.

    Un solo caso basta para invalidar la tabla, así que se cuentan sobre todas las filas.
    """
    faltan = {"train", "valid", "test"} - set(tabla["particion"].astype(str).unique())
    if faltan:
        raise ValueError(f"La tabla de P2 no tiene filas de {sorted(faltan)}")
    fin = tabla.groupby("particion", observed=True)["dia"].max()
    dia, conjunto = tabla["dia"], tabla["particion"]
    duplicadas = tabla.duplicated(["estacion", "dia", "hora_idx"])
    historia = tabla[["hist_sin_bicis", "hist_sin_anclajes"]]
    historia_rota = (historia.isna() | (historia < 0) | (historia > 1)).any(axis=1)
    valid_antes = (conjunto == "valid") & (dia <= fin["train"])
    test_antes = (conjunto == "test") & (dia <= fin["valid"])
    problemas = {
        "llave estación-día-bin duplicada": duplicadas.sum(),
        "clase fuera de 0, 1 y 2": (~tabla["clase"].isin([0, 1, 2])).sum(),
        "historia fuera de [0, 1]": historia_rota.sum(),
        "días de valid no posteriores a train": valid_antes.sum(),
        "días de test no posteriores a valid": test_antes.sum(),
    }
    revision = pd.DataFrame(
        {"revision": list(problemas), "filas_con_problema": [int(n) for n in problemas.values()]}
    )
    revision["resultado"] = np.where(revision["filas_con_problema"] == 0, "PASS", "FAIL")
    return revision


@app.command()
def p2_validar(muestra: int = cfg.P2_MUESTRA_VALIDACION):
    """Valida fit, valid y test de P2 (con la historia): revisiones completas y DeepChecks."""
    if not cfg.RUTA_P2_CAMBIOS.exists():
        raise FileNotFoundError(f"Falta {cfg.RUTA_P2_CAMBIOS.name}; ejecutar `make features-p2`")
    inicio = time.perf_counter()
    carpeta = validacion.VALIDACION_DIR
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla = p2_tabla_validacion(pd.read_parquet(cfg.RUTA_P2_CAMBIOS))

    revision = p2_revisar_tabla_completa(tabla)
    revision.to_csv(carpeta / "p2_tabla_completa.csv", index=False)
    logger.info(f"Revisiones sobre la tabla completa:\n{revision.to_string(index=False)}")
    if (revision["resultado"] == "FAIL").any():
        raise ValueError(
            "La tabla de P2 no pasa las revisiones completas; ver p2_tabla_completa.csv"
        )

    # hora_idx ya está en hora_sin y hora_cos; estacion como categórica para ver estaciones nuevas
    tabla = tabla.drop(columns="hora_idx")
    for comparado in ("valid", "test"):
        validacion.validar_tabla(
            f"p2_train_{comparado}",
            tabla,
            "particion",
            "clase",
            ["estacion"],
            "dia",
            conjuntos=("train", comparado),
            muestra=muestra,
            carpeta=carpeta,
        )
    logger.success(
        f"Validación de P2 en {(time.perf_counter() - inicio) / 60:.1f} min ({carpeta})"
    )


if __name__ == "__main__":
    app()
