"""Monitoreo de P1 (demanda por estación-hora): conjuntos, drift, versión alternativa y A/B testing.

Usa la base común de monitoreo.py. Los conjuntos salen de dos tablas:
    referencia, valid y actual   la tabla de entrenamiento (p1_demanda_estacion_hora.parquet),
                                 la misma con la que se entrenó y evaluó el champion
    evaluacion                   el mes nuevo, calculado con el mismo pipeline de features sobre
                                 la historia extendida, porque sus rezagos necesitan los días previos

Uso: python -m tc5061_proyecto_equipo07.modeling.monitoreo_p1 tabla | drift | drift-anual
"""

from loguru import logger
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import features, monitoreo
from tc5061_proyecto_equipo07.modeling.predict import p1_predecir
from tc5061_proyecto_equipo07.modeling.train import P1BaselinePromedio

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
    """Carga la versión con alias champion; regresa (Pipeline, versión, etiquetas, model_id).

    El model_id liga los runs de monitoreo al modelo que evaluaron, como modelo de entrada.

    Se carga con el sabor sklearn con el que se registró: así se predice con `p1_predecir`, igual
    que al entrenar, sin depender de cómo la firma de pyfunc trate la columna categórica `zona`.
    """
    if cfg.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    version = mlflow.MlflowClient().get_model_version_by_alias(
        cfg.P1_MODELO_REGISTRADO, "champion"
    )
    modelo = mlflow.sklearn.load_model(f"models:/{cfg.P1_MODELO_REGISTRADO}/{version.version}")
    logger.info(f"{cfg.P1_MODELO_REGISTRADO}: champion = versión {version.version}")
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


if __name__ == "__main__":
    app()
