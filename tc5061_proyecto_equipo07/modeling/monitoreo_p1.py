"""Monitoreo de P1 (demanda por estación-hora): conjuntos, drift, versión alternativa y A/B testing.

Usa la base común de monitoreo.py. Los conjuntos salen de dos tablas:
    referencia, valid y actual   la tabla de entrenamiento (p1_demanda_estacion_hora.parquet),
                                 la misma con la que se entrenó y evaluó el champion
    evaluacion                   el mes nuevo, calculado con el mismo pipeline de features sobre
                                 la historia extendida, porque sus rezagos necesitan los días previos

Uso: python -m tc5061_proyecto_equipo07.modeling.monitoreo_p1 tabla | drift | drift-anual | challenger | ab
"""

import time

from loguru import logger
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import features, monitoreo
from tc5061_proyecto_equipo07.modeling import registro_p1
from tc5061_proyecto_equipo07.modeling.predict import p1_predecir
from tc5061_proyecto_equipo07.modeling.train import (
    P1BaselinePromedio,
    p1_mae_por_zona,
    p1_parametros,
    p1_registrar_run,
)

app = typer.Typer(add_completion=False)

# Columnas que se comparan entre la tabla de entrenamiento y la extendida
P1_COLUMNAS = [*cfg.P1_VARIABLES, "hora_inicio", "viajes"]
P1_LLAVE = ["origen_id", "hora_inicio"]

# Columnas de los reportes de Evidently: las variables del modelo sin origen_id, que es un id y no
# una cantidad. Las banderas van como categóricas: solo toman 0 o 1
P1_CATEGORICAS = ["zona", "es_fin_de_semana", "es_festivo"]
P1_NUMERICAS = [c for c in cfg.P1_VARIABLES if c not in ["origen_id", *P1_CATEGORICAS]]
# Columnas de demanda del desglose estacional por año
P1_DEMANDA = [
    "viajes",
    "viajes_rezago_1d",
    "viajes_rezago_7d",
    f"viajes_promedio_{cfg.P1_SEMANAS_PROMEDIO}sem",
    "prediccion",
]


def p1_comparar_tablas(entrenamiento: pd.DataFrame, extendida: pd.DataFrame) -> dict[str, int]:
    """Comprueba que la tabla extendida reproduce, fila por fila, la tabla de entrenamiento.

    Cada estación-hora de entrenamiento debe estar en la extendida con los mismos valores: si no,
    el mes nuevo se estaría calculando distinto que los datos del champion (train-serving skew).
    La extendida puede traer filas de más antes del mes nuevo: las horas posteriores al último viaje
    de una estación en el rango original, que con el mes nuevo ya caen dentro de su operación.
    """
    fin = pd.Timestamp(cfg.FRONTERA_MONITOREO)
    previa = extendida[extendida["hora_inicio"] < fin]
    unidas = entrenamiento[P1_COLUMNAS].merge(
        previa[P1_COLUMNAS], on=P1_LLAVE, how="left", suffixes=("", "_ext"), indicator=True
    )
    faltantes = int((unidas["_merge"] == "left_only").sum())
    if faltantes:
        raise ValueError(f"{faltantes} estación-horas de entrenamiento no están en la extendida")

    distintas = pd.Series(False, index=unidas.index)
    for columna in P1_COLUMNAS:
        if columna in P1_LLAVE:
            continue
        original, nueva = unidas[columna], unidas[f"{columna}_ext"]
        if isinstance(original.dtype, pd.CategoricalDtype):
            original, nueva = original.astype(str), nueva.astype(str)
        iguales = (original == nueva) | (original.isna() & nueva.isna())
        if not iguales.all():
            logger.warning(f"{columna}: {int((~iguales).sum())} valores distintos")
        distintas |= ~iguales
    if distintas.any():
        raise ValueError(f"{int(distintas.sum())} estación-horas cambian en la tabla extendida")
    return {"filas_comparadas": len(unidas), "filas_extra": len(previa) - len(unidas)}


def p1_evaluacion(extendida: pd.DataFrame) -> pd.DataFrame:
    """Estación-horas del mes de evaluación, etiquetadas con su periodo."""
    evaluacion = extendida[extendida["hora_inicio"] >= pd.Timestamp(cfg.FRONTERA_MONITOREO)]
    evaluacion = evaluacion.drop(columns="particion").reset_index(drop=True)
    evaluacion["periodo"] = monitoreo.asignar_periodo(evaluacion["hora_inicio"])
    if (evaluacion["periodo"] != "evaluacion").any():
        raise ValueError("La tabla de evaluación trae horas fuera del mes de monitoreo")
    return evaluacion


def p1_conjuntos(
    ruta_entrenamiento=cfg.RUTA_P1_DEMANDA, ruta_evaluacion=cfg.P1_RUTA_EVALUACION
) -> dict[str, pd.DataFrame]:
    """Regresa referencia, valid, actual y evaluacion; las horas en `brecha` se descartan.

    Se etiqueta con `monitoreo.asignar_periodo`, no con la partición de entrenamiento: así las
    24 h previas al mes nuevo también quedan en `brecha`, como en las demás fronteras.
    """
    for ruta, comando in [
        (ruta_entrenamiento, "make features-p1"),
        (ruta_evaluacion, "make monitoreo-p1-tabla"),
    ]:
        if not ruta.exists():
            raise FileNotFoundError(f"Falta {ruta.name}; ejecutar `{comando}`")
    tabla = pd.read_parquet(ruta_entrenamiento, columns=P1_COLUMNAS)
    tabla["periodo"] = monitoreo.asignar_periodo(tabla["hora_inicio"])
    evaluacion = pd.read_parquet(ruta_evaluacion, columns=[*P1_COLUMNAS, "periodo"])
    conjuntos = {
        periodo: tabla[tabla["periodo"] == periodo].reset_index(drop=True)
        for periodo in ["referencia", "valid", "actual"]
    }
    conjuntos["evaluacion"] = evaluacion
    return conjuntos


def p1_modelo_id(version) -> str | None:
    """Id del modelo de MLflow (m-...) del que sale una versión del Registry.

    registro_p1 registra desde `models:/<model_id>`, así que el id está en `source`; el atributo
    `model_id` de la versión puede venir vacío. Si la versión salió de `runs:/`, se busca el
    modelo registrado por ese run.
    """
    if version.model_id:
        return version.model_id
    if version.source.startswith("models:/m-"):
        return version.source.removeprefix("models:/").split("/")[0]
    if version.run_id:
        run = mlflow.get_run(version.run_id)
        modelos = mlflow.search_logged_models(
            experiment_ids=[run.info.experiment_id],
            filter_string=f"source_run_id = '{version.run_id}'",
            output_format="list",
        )
        if modelos:
            return modelos[0].model_id
    logger.warning(f"No se encontró el modelo de MLflow de la versión {version.version}")
    return None


def p1_cargar_champion():
    """Carga la versión con alias champion; regresa (Pipeline, versión, etiquetas, model_id)."""
    return p1_cargar_alias("champion")


def p1_cargar_alias(alias: str):
    """Carga la versión con el alias dado; regresa (Pipeline, versión, etiquetas, model_id).

    El model_id liga los runs de monitoreo al modelo que evaluaron, como modelo de entrada.

    Se carga con el sabor sklearn con el que se registró: así se predice con `p1_predecir`, igual
    que al entrenar, sin depender de cómo la firma de pyfunc trate la columna categórica `zona`.
    """
    if cfg.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    version = mlflow.MlflowClient().get_model_version_by_alias(cfg.P1_MODELO_REGISTRADO, alias)
    modelo = mlflow.sklearn.load_model(f"models:/{cfg.P1_MODELO_REGISTRADO}/{version.version}")
    logger.info(f"{cfg.P1_MODELO_REGISTRADO}: {alias} = versión {version.version}")
    return modelo, int(version.version), dict(version.tags), p1_modelo_id(version)


def p1_desempeno(modelos: dict, conjuntos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """MAE y RMSE de cada modelo en cada conjunto, sobre todas sus filas."""
    filas = []
    for nombre, modelo in modelos.items():
        for conjunto, datos in conjuntos.items():
            prediccion = p1_predecir(modelo, datos)
            filas.append(
                {
                    "modelo": nombre,
                    "conjunto": conjunto,
                    "filas": len(datos),
                    "mae": mean_absolute_error(datos["viajes"], prediccion),
                    "rmse": root_mean_squared_error(datos["viajes"], prediccion),
                }
            )
    return pd.DataFrame(filas)


def p1_tabla_reporte(datos: pd.DataFrame, modelo) -> pd.DataFrame:
    """Columnas de los reportes de Evidently: variables, target `viajes` y predicción del modelo."""
    tabla = datos[[*P1_NUMERICAS, "viajes"]].astype("float64")
    for columna in P1_CATEGORICAS:
        tabla[columna] = datos[columna].astype(str)
    tabla["prediccion"] = p1_predecir(modelo, datos)
    return tabla


def p1_mismo_mes(referencia: pd.DataFrame, mes: str = cfg.MES_MONITOREO) -> pd.DataFrame:
    """Filas de la referencia del mismo mes del año que el mes de evaluación (control estacional)."""
    numero = pd.Period(mes, freq="M").month
    return referencia[referencia["hora_inicio"].dt.month == numero].reset_index(drop=True)


def p1_mismo_mes_por_anio(
    conjuntos: dict[str, pd.DataFrame], mes: str = cfg.MES_MONITOREO
) -> dict[int, tuple[str, pd.DataFrame]]:
    """El mismo mes del año en cada año previo, con el periodo al que pertenece.

    Toma referencia, valid y actual: un año en referencia es dato de entrenamiento del champion, así
    que su error ahí es optimista; en valid y actual es error sobre datos que no vio.
    """
    numero = pd.Period(mes, freq="M").month
    meses = {}
    for periodo in ("referencia", "valid", "actual"):
        datos = conjuntos[periodo]
        del_mes = datos[datos["hora_inicio"].dt.month == numero]
        for anio, grupo in del_mes.groupby(del_mes["hora_inicio"].dt.year):
            meses[int(anio)] = (periodo, grupo.reset_index(drop=True))
    return dict(sorted(meses.items()))


def p1_fila_anual(anio: int, periodo: str, datos: pd.DataFrame, modelos: dict) -> dict:
    """Volumen y error de champion y baseline en el mes de un año."""
    desempeno = p1_desempeno(modelos, {periodo: datos}).set_index("modelo")["mae"]
    return {
        "anio": anio,
        "periodo": periodo,
        "filas": len(datos),
        "estaciones": datos["origen_id"].nunique(),
        "viajes_promedio": datos["viajes"].mean(),
        "mae_champion": desempeno["champion"],
        "mae_baseline": desempeno["baseline"],
        "ventaja_champion": 1 - desempeno["champion"] / desempeno["baseline"],
    }


def p1_datasets(conjuntos: dict[str, pd.DataFrame]) -> dict:
    """Conjuntos de monitoreo como datasets de MLflow, con la tabla de la que salen."""
    fuentes = {"evaluacion": cfg.P1_RUTA_EVALUACION}
    return {
        nombre: monitoreo.dataset_mlflow(
            datos[P1_COLUMNAS],
            f"p1_{nombre}",
            fuentes.get(nombre, cfg.RUTA_P1_DEMANDA),
            "viajes",
        )
        for nombre, datos in conjuntos.items()
    }


def p1_separar_metricas_champion(metricas: dict, conjuntos) -> dict[str, dict[str, float]]:
    """Saca de `metricas` las del champion por conjunto, para asociarlas a su modelo y dataset."""
    return {
        conjunto: {
            clave: metricas.pop(clave)
            for clave in (f"champion_mae_{conjunto}", f"champion_rmse_{conjunto}")
            if clave in metricas
        }
        for conjunto in conjuntos
    }


def p1_datos_challenger(conjuntos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Datos del challenger: referencia, valid y actual, sin las brechas ni el mes de evaluación."""
    datos = pd.concat([conjuntos[c] for c in ("referencia", "valid", "actual")], ignore_index=True)
    if datos["hora_inicio"].max() >= pd.Timestamp(cfg.FRONTERA_MONITOREO):
        raise ValueError("Los datos del challenger no pueden incluir el mes de evaluación")
    return datos


def p1_parametros_challenger(champion, etiquetas: dict, version: int, n_train: int) -> dict:
    """Parámetros del run del challenger, con los mismos hiperparámetros que el run del champion.

    Se copian los `modelo_*` del run de origen de la versión, tal como se registraron; si no hay
    run de origen, se toman del regresor del champion.
    """
    parametros = p1_parametros("xgb", n_train)
    parametros.update({"modelo": "xgb_challenger", "base_version": version})
    origen = etiquetas.get("source_run_id")
    if origen:
        del_origen = mlflow.get_run(origen).data.params
        parametros.update({k: v for k, v in del_origen.items() if k.startswith("modelo_")})
        parametros["modelo"] = "xgb_challenger"
    else:
        regresor = champion.named_steps["modelo"].get_params()
        parametros.update(
            {f"modelo_{k}": regresor[k] for k in cfg.P1_PARAMETROS_XGB if k in regresor}
        )
    return parametros


def p1_grupos_estacion_dia(datos: pd.DataFrame) -> np.ndarray:
    """Código de estación-día de cada fila: la unidad del bootstrap y del reparto aleatorio."""
    llave = datos["origen_id"].astype(str) + "_" + datos["hora_inicio"].dt.strftime("%Y-%m-%d")
    return pd.factorize(llave)[0]


def p1_segmentos(evaluacion: pd.DataFrame, referencia: pd.DataFrame) -> dict[str, np.ndarray]:
    """Todas las filas y su división en estaciones que el champion vio al entrenar y nuevas.

    El nivel por estación de P1NivelEstacion se aprendió con train: a una estación que no estaba
    ahí le asigna el nivel del sistema, así que el champion predice sin su historia.
    """
    conocidas = evaluacion["origen_id"].isin(referencia["origen_id"].unique()).to_numpy()
    return {"todas": np.ones(len(evaluacion), bool), "conocidas": conocidas, "nuevas": ~conocidas}


def p1_tabla_ab(
    evaluacion: pd.DataFrame,
    prediccion_a: np.ndarray,
    prediccion_b: np.ndarray,
    prediccion_baseline: np.ndarray,
    segmentos: dict[str, np.ndarray],
) -> pd.DataFrame:
    """A/B por segmento: pareado (decide) y aleatorio (complemento), con el baseline de contexto.

    La pérdida es el error absoluto por fila; la unidad es la estación-día.
    """
    real = evaluacion["viajes"].to_numpy("float64")
    perdida_a, perdida_b = np.abs(real - prediccion_a), np.abs(real - prediccion_b)
    perdida_baseline = np.abs(real - prediccion_baseline)
    grupos = p1_grupos_estacion_dia(evaluacion)
    filas = []
    for segmento, filtro in segmentos.items():
        if not filtro.any():
            continue
        contexto = {
            "segmento": segmento,
            "estaciones": evaluacion.loc[filtro, "origen_id"].nunique(),
            "mae_baseline": perdida_baseline[filtro].mean(),
        }
        pareado = monitoreo.comparar_pareado(perdida_a[filtro], perdida_b[filtro], grupos[filtro])
        filas.append({**contexto, **pareado})
        aleatorio = monitoreo.comparar_aleatorio(
            perdida_a[filtro], perdida_b[filtro], grupos[filtro]
        )
        filas.append({**contexto, **aleatorio})
    return pd.DataFrame(filas)


def p1_tabla_por_zona(
    evaluacion: pd.DataFrame, predicciones: dict[str, np.ndarray]
) -> pd.DataFrame:
    """MAE diario por zona de cada versión (p1_mae_por_zona) y el cambio de B contra A."""
    por_zona = pd.concat(
        {
            nombre: p1_mae_por_zona(evaluacion, prediccion)["mae_diario"]
            for nombre, prediccion in predicciones.items()
        },
        axis=1,
    ).add_prefix("mae_diario_")
    por_zona["cambio_b_vs_a"] = por_zona["mae_diario_b"] / por_zona["mae_diario_a"] - 1
    return por_zona.reset_index()


def p1_mes_previo(frontera: str) -> str:
    """Mes anterior a una frontera ("2024-10-01" -> "2024-09"), para describir los periodos."""
    return str(pd.Period(frontera, freq="M") - 1)


def p1_metricas_drift(snapshots: dict, desempeno: pd.DataFrame) -> dict[str, float]:
    """Métricas del run de drift: proporción de columnas con drift por reporte y MAE por conjunto."""
    metricas = {}
    for nombre, snapshot in snapshots.items():
        proporcion = monitoreo.proporcion_drift(snapshot)
        metricas[f"columnas_con_drift_{nombre}"] = proporcion["columnas_con_drift"]
        metricas[f"proporcion_drift_{nombre}"] = proporcion["proporcion_drift"]
    for fila in desempeno.itertuples():
        metricas[f"{fila.modelo}_mae_{fila.conjunto}"] = fila.mae
        metricas[f"{fila.modelo}_rmse_{fila.conjunto}"] = fila.rmse
    mae = desempeno[desempeno["modelo"] == "champion"].set_index("conjunto")["mae"]
    for conjunto in ("actual", "evaluacion"):
        metricas[f"champion_cambio_mae_{conjunto}_vs_valid"] = mae[conjunto] / mae["valid"] - 1
    return metricas


@app.callback()
def p1_ayuda_monitoreo():
    """Monitoreo de P1: tabla del mes nuevo, drift, versión alternativa y A/B testing."""


@app.command()
def tabla():
    """Calcula las estación-horas del mes nuevo y las verifica contra la tabla de entrenamiento."""
    if not cfg.RUTA_VIAJES_LIMPIOS_EXTENDIDOS.exists():
        raise FileNotFoundError(
            f"Falta {cfg.RUTA_VIAJES_LIMPIOS_EXTENDIDOS.name}; ejecutar `make monitoreo-datos`"
        )
    if not cfg.RUTA_P1_DEMANDA.exists():
        raise FileNotFoundError(f"Falta {cfg.RUTA_P1_DEMANDA.name}; ejecutar `make features-p1`")
    limpios = pd.read_parquet(
        cfg.RUTA_VIAJES_LIMPIOS_EXTENDIDOS,
        columns=["inicio_del_viaje", "origen_id", "periodo_archivo"],
    )
    extendida = features.p1_tabla_modelado(
        limpios, features.p1_cargar_estaciones(), cfg.MES_INICIO, cfg.MES_MONITOREO
    )
    del limpios

    comparacion = p1_comparar_tablas(pd.read_parquet(cfg.RUTA_P1_DEMANDA), extendida)
    logger.info(
        f"Tabla de entrenamiento reproducida: {comparacion['filas_comparadas']} estación-horas "
        f"idénticas; {comparacion['filas_extra']} filas de más antes de {cfg.MES_MONITOREO}"
    )
    evaluacion = p1_evaluacion(extendida)
    evaluacion.to_parquet(cfg.P1_RUTA_EVALUACION, index=False)
    logger.success(
        f"{cfg.P1_RUTA_EVALUACION.name}: {len(evaluacion)} estación-horas, "
        f"{int(evaluacion['viajes'].sum())} viajes, "
        f"{evaluacion['origen_id'].nunique()} estaciones"
    )


@app.command()
def drift(muestra: int = cfg.MONITOREO_MUESTRA):
    """Reportes de drift, desempeño y estacionalidad del champion, registrados como run en MLflow.

    1. Drift y calidad: referencia (train) contra actual (test), con predicción y target.
    2. Desempeño: valid contra actual; train no sirve porque el nivel por estación de
       P1NivelEstacion se aprendió con su propio target y su error sale artificialmente bajo.
    3. Estacionalidad: el mismo mes en la referencia contra el mes de evaluación.
    """
    conjuntos = p1_conjuntos(cfg.RUTA_P1_DEMANDA, cfg.P1_RUTA_EVALUACION)
    referencia = conjuntos["referencia"]
    modelo, version, etiquetas, modelo_id = p1_cargar_champion()
    baseline = P1BaselinePromedio().fit(referencia[cfg.P1_VARIABLES], referencia["viajes"])

    evaluados = {c: conjuntos[c] for c in ("valid", "actual", "evaluacion")}
    desempeno = p1_desempeno({"champion": modelo, "baseline": baseline}, evaluados)
    mae_valid = desempeno.query("modelo == 'champion' and conjunto == 'valid'")["mae"].iloc[0]
    registrado = etiquetas.get("valid_mae")
    if registrado is not None and not np.isclose(mae_valid, float(registrado), atol=1e-3):
        logger.warning(f"MAE de valid {mae_valid:.4f} distinto al registrado ({registrado})")

    def muestra_de(datos: pd.DataFrame) -> pd.DataFrame:
        return p1_tabla_reporte(monitoreo.muestrear(datos, muestra), modelo)

    referencia_m, actual_m = muestra_de(referencia), muestra_de(conjuntos["actual"])
    sin_tarea = monitoreo.definicion_datos(P1_NUMERICAS + ["viajes", "prediccion"], P1_CATEGORICAS)
    con_tarea = monitoreo.definicion_datos(
        P1_NUMERICAS, P1_CATEGORICAS, "regresion", "viajes", "prediccion"
    )
    carpeta = cfg.MONITOREO_DIR
    drift_tt, rutas_tt = monitoreo.generar_reporte(
        referencia_m, actual_m, sin_tarea, "p1_drift_train_test", carpeta=carpeta
    )
    _, rutas_dt = monitoreo.generar_reporte(
        muestra_de(conjuntos["valid"]),
        actual_m,
        con_tarea,
        "p1_desempeno_valid_test",
        "regresion",
        carpeta,
    )
    drift_est, rutas_est = monitoreo.generar_reporte(
        muestra_de(p1_mismo_mes(referencia)),
        muestra_de(conjuntos["evaluacion"]),
        sin_tarea,
        "p1_drift_estacional",
        carpeta=carpeta,
    )

    resumen = pd.concat(
        [
            monitoreo.resumen_drift(drift_tt).assign(comparacion="train_vs_test"),
            monitoreo.resumen_drift(drift_est).assign(comparacion="mismo_mes_vs_evaluacion"),
        ],
        ignore_index=True,
    )
    ruta_resumen = carpeta / "p1_drift_resumen.csv"
    ruta_desempeno = carpeta / "p1_desempeno.csv"
    resumen.to_csv(ruta_resumen, index=False)
    desempeno.to_csv(ruta_desempeno, index=False)
    logger.info(f"Drift por variable:\n{resumen.to_string(index=False)}")
    logger.info(f"Desempeño por conjunto:\n{desempeno.to_string(index=False)}")

    metricas = p1_metricas_drift({"train_test": drift_tt, "estacional": drift_est}, desempeno)
    monitoreo.registrar_en_mlflow(
        "p1-drift",
        "p1",
        [*rutas_tt, *rutas_dt, *rutas_est, ruta_resumen, ruta_desempeno],
        metricas=metricas,
        metricas_por_dataset=p1_separar_metricas_champion(metricas, evaluados),
        datasets=p1_datasets(conjuntos),
        modelo_id=modelo_id,
        descripcion=(
            f"Monitoreo de P1 sobre el champion v{version}: drift de datos (train contra test), "
            f"desempeño (valid contra test) y control estacional ({cfg.MES_MONITOREO} contra el "
            "mismo mes de la referencia), con Evidently. Champion y baseline evaluados en valid, "
            "test y el mes nuevo."
        ),
        parametros={
            "champion_version": version,
            "muestra": muestra,
            "mes_monitoreo": cfg.MES_MONITOREO,
            "referencia": f"{cfg.MES_INICIO} a {p1_mes_previo(cfg.FRONTERA_VALID)}",
            "actual": f"{cfg.FRONTERA_TEST[:7]} a {p1_mes_previo(cfg.FRONTERA_MONITOREO)}",
        },
        etiquetas={
            "champion_version": str(version),
            "tipo_monitoreo": "drift",
            "autor": "Daniel Sánchez Hernández",
        },
    )


@app.command()
def drift_anual(muestra: int = cfg.MONITOREO_MUESTRA):
    """Desglose estacional: el mismo mes de cada año previo contra el mes de evaluación.

    Separa el drift real del de un año atípico (por ejemplo, 2020 con la pandemia) y muestra cómo
    cambia la ventaja del champion sobre el baseline conforme los datos se alejan de su entrenamiento.
    """
    conjuntos = p1_conjuntos(cfg.RUTA_P1_DEMANDA, cfg.P1_RUTA_EVALUACION)
    referencia, evaluacion = conjuntos["referencia"], conjuntos["evaluacion"]
    modelo, version, _, modelo_id = p1_cargar_champion()
    baseline = P1BaselinePromedio().fit(referencia[cfg.P1_VARIABLES], referencia["viajes"])
    modelos = {"champion": modelo, "baseline": baseline}

    definicion = monitoreo.definicion_datos(P1_DEMANDA, [])
    actual_m = p1_tabla_reporte(monitoreo.muestrear(evaluacion, muestra), modelo)[P1_DEMANDA]
    filas, rutas = [], []
    por_anio = p1_mismo_mes_por_anio(conjuntos)
    for anio, (periodo, datos) in por_anio.items():
        referencia_m = p1_tabla_reporte(monitoreo.muestrear(datos, muestra), modelo)[P1_DEMANDA]
        snapshot, rutas_anio = monitoreo.generar_reporte(
            referencia_m, actual_m, definicion, f"p1_drift_mes_{anio}", carpeta=cfg.MONITOREO_DIR
        )
        rutas += rutas_anio
        drift = monitoreo.resumen_drift(snapshot).set_index("columna")["valor"]
        fila = p1_fila_anual(anio, periodo, datos, modelos)
        fila.update({f"drift_{columna}": drift[columna] for columna in P1_DEMANDA})
        filas.append(fila)
    anio_evaluacion = pd.Period(cfg.MES_MONITOREO, freq="M").year
    filas.append(p1_fila_anual(anio_evaluacion, "evaluacion", evaluacion, modelos))

    tabla_anual = pd.DataFrame(filas)
    ruta = cfg.MONITOREO_DIR / "p1_drift_mismo_mes_por_anio.csv"
    tabla_anual.to_csv(ruta, index=False)
    logger.info(
        f"Mismo mes por año contra {cfg.MES_MONITOREO}:\n{tabla_anual.to_string(index=False)}"
    )

    metricas, por_dataset = {}, {}
    for fila in tabla_anual.itertuples():
        contexto = "evaluacion" if fila.periodo == "evaluacion" else f"mes_{fila.anio}"
        metricas[f"ventaja_champion_{fila.anio}"] = fila.ventaja_champion
        metricas[f"baseline_mae_{fila.anio}"] = fila.mae_baseline
        por_dataset[contexto] = {f"champion_mae_{fila.anio}": fila.mae_champion}
        if pd.notna(getattr(fila, "drift_viajes", np.nan)):
            metricas[f"drift_viajes_{fila.anio}"] = fila.drift_viajes
    datasets = p1_datasets(
        {f"mes_{anio}": datos for anio, (_, datos) in por_anio.items()}
        | {"evaluacion": evaluacion}
    )
    monitoreo.registrar_en_mlflow(
        "p1-drift-mismo-mes",
        "p1",
        [*rutas, ruta],
        metricas=metricas,
        metricas_por_dataset=por_dataset,
        datasets=datasets,
        modelo_id=modelo_id,
        descripcion=(
            f"Desglose estacional de P1 sobre el champion v{version}: el mismo mes de cada año "
            f"previo contra {cfg.MES_MONITOREO}, con el drift de la demanda (Evidently) y la "
            "ventaja del champion sobre el baseline por año."
        ),
        parametros={
            "champion_version": version,
            "muestra": muestra,
            "mes_monitoreo": cfg.MES_MONITOREO,
        },
        etiquetas={
            "champion_version": str(version),
            "tipo_monitoreo": "drift_estacional",
            "autor": "Daniel Sánchez Hernández",
        },
    )


@app.command()
def challenger():
    """Entrena y registra la versión alternativa de P1 con alias challenger (data iteration).

    Clona el champion (mismos hiperparámetros y semilla) y lo reentrena con referencia, valid y
    actual, de 2020-01 a 2026-09 sin brechas. El run y el registro usan las mismas funciones que los
    entrenamientos de P1 (p1_registrar_run y registro_p1); el alias champion no se toca.
    """
    conjuntos = p1_conjuntos(cfg.RUTA_P1_DEMANDA, cfg.P1_RUTA_EVALUACION)
    datos = p1_datos_challenger(conjuntos)
    champion, version, etiquetas, modelo_id = p1_cargar_champion()
    modelo = clone(champion)
    logger.info(
        f"Entrenando el challenger con {len(datos)} filas "
        f"({datos['hora_inicio'].min():%Y-%m} a {datos['hora_inicio'].max():%Y-%m})"
    )
    inicio = time.perf_counter()
    modelo.fit(datos[cfg.P1_VARIABLES], datos["viajes"])
    segundos = time.perf_counter() - inicio
    logger.info(f"Challenger entrenado en {segundos / 60:.1f} min")

    rango = f"{cfg.MES_INICIO} a {p1_mes_previo(cfg.FRONTERA_MONITOREO)}"
    id_run = p1_registrar_run(
        "xgb",
        p1_parametros_challenger(champion, etiquetas, version, len(datos)),
        {"segundos_entrenamiento": segundos},
        [],
        modelo,
        datos[cfg.P1_VARIABLES].head(3),
        datos,
        etiquetas_extra={
            "fase": cfg.FASE_MONITOREO,
            "base_version": str(version),
            "base_source_run_id": etiquetas.get("source_run_id", "desconocido"),
            "base_model_id": modelo_id or "desconocido",
            "rango_datos": rango,
        },
        identidad=cfg.P1_CHALLENGER,
        descripcion=cfg.P1_DESCRIPCION_CHALLENGER,
    )
    nueva = registro_p1.p1_registrar(id_run, "challenger")
    registro_p1.p1_verificar("challenger")
    logger.success(f"Challenger = versión {nueva}; champion sigue en la versión {version}")


@app.command()
def ab():
    """A/B testing de P1: champion (A) contra challenger (B) sobre el mes de evaluación.

    Criterio fijado antes de ver resultados: error absoluto por fila; B gana si la prueba pareada
    (Wilcoxon, unidad estación-día) da p < MONITOREO_ALFA y su MAE es menor. La prueba aleatoria y
    el baseline son contexto. El alias champion no se mueve: la decisión queda como recomendación.
    """
    conjuntos = p1_conjuntos(cfg.RUTA_P1_DEMANDA, cfg.P1_RUTA_EVALUACION)
    referencia, evaluacion = conjuntos["referencia"], conjuntos["evaluacion"]
    modelo_a, version_a, _, id_a = p1_cargar_alias("champion")
    modelo_b, version_b, _, id_b = p1_cargar_alias("challenger")
    if version_a == version_b:
        raise ValueError("champion y challenger son la misma versión: no hay nada que comparar")
    baseline = P1BaselinePromedio().fit(referencia[cfg.P1_VARIABLES], referencia["viajes"])

    predicciones = {
        "a": p1_predecir(modelo_a, evaluacion),
        "b": p1_predecir(modelo_b, evaluacion),
        "baseline": p1_predecir(baseline, evaluacion),
    }
    tabla = p1_tabla_ab(
        evaluacion,
        predicciones["a"],
        predicciones["b"],
        predicciones["baseline"],
        p1_segmentos(evaluacion, referencia),
    )
    tabla.insert(0, "version_b", version_b)
    tabla.insert(0, "version_a", version_a)
    por_zona = p1_tabla_por_zona(evaluacion, predicciones)

    carpeta = cfg.MONITOREO_DIR
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_ab, ruta_zona = carpeta / "p1_ab.csv", carpeta / "p1_ab_por_zona.csv"
    tabla.to_csv(ruta_ab, index=False)
    por_zona.to_csv(ruta_zona, index=False)
    logger.info(f"A/B v{version_a} (A) contra v{version_b} (B):\n{tabla.to_string(index=False)}")
    logger.info(f"MAE diario por zona:\n{por_zona.to_string(index=False)}")

    decision = tabla.query("segmento == 'todas' and prueba == 'pareada'").iloc[0]
    metricas = {
        "p_valor_pareado": decision["p_valor"],
        "p_valor_t_pareado": decision["p_valor_t"],
        "diferencia_mae": decision["diferencia"],
        "cambio_relativo_mae": decision["cambio_relativo"],
        "ic_inferior": decision["ic_inferior"],
        "ic_superior": decision["ic_superior"],
        "baseline_mae_evaluacion": decision["mae_baseline"],
    }
    for fila in tabla.query("prueba == 'pareada'").itertuples():
        metricas[f"cambio_relativo_mae_{fila.segmento}"] = fila.cambio_relativo
    aleatoria = tabla.query("segmento == 'todas' and prueba == 'aleatoria'").iloc[0]
    metricas["p_valor_aleatorio"] = aleatoria["p_valor"]

    ganador = decision["ganador"]
    texto = {"A": f"champion v{version_a}", "B": f"challenger v{version_b}"}.get(ganador, ganador)
    monitoreo.registrar_en_mlflow(
        f"p1-ab-v{version_a}-vs-v{version_b}",
        "p1",
        [ruta_ab, ruta_zona],
        metricas=metricas,
        parametros={
            "champion_version": version_a,
            "challenger_version": version_b,
            "criterio": "MAE por fila, prueba pareada de Wilcoxon",
            "alfa": cfg.MONITOREO_ALFA,
            "bootstrap": cfg.MONITOREO_BOOTSTRAP,
            "unidad": "estacion_dia",
        },
        etiquetas={
            "tipo_monitoreo": "ab_testing",
            "ganador": texto,
            "champion_version": str(version_a),
            "challenger_version": str(version_b),
            "autor": "Daniel Sánchez Hernández",
        },
        descripcion=(
            f"A/B testing de P1 sobre {cfg.MES_MONITOREO}: champion v{version_a} (A) contra "
            f"challenger v{version_b} (B), mismo modelo reentrenado con datos más recientes. "
            "Prueba pareada (decide) y aleatoria por estación-día, global y por estaciones "
            "conocidas o nuevas para el champion, más el MAE diario por zona."
        ),
        datasets=p1_datasets({"evaluacion": evaluacion}),
        metricas_por_modelo={
            id_a: {"evaluacion": {"champion_mae_evaluacion": decision["media_a"]}},
            id_b: {"evaluacion": {"challenger_mae_evaluacion": decision["media_b"]}},
        },
    )
    logger.success(
        f"Ganador (prueba pareada, alfa {cfg.MONITOREO_ALFA}): {texto}; "
        f"MAE {decision['media_a']:.4f} -> {decision['media_b']:.4f} "
        f"({decision['cambio_relativo']:+.1%}), p = {decision['p_valor']:.2e}"
    )


if __name__ == "__main__":
    app()
