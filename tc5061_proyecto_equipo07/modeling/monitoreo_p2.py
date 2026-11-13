"""Monitoreo de P2 (riesgo de quedarse sin bicis o sin anclajes): conjuntos, drift, versión
alternativa y A/B testing.

Usa la base común de monitoreo.py. La etiqueta y las variables de flujo de P2 no cruzan el día de
servicio, así que el mes nuevo se calcula solo con sus viajes (más un margen para los que empiezan
antes). La historia "hasta ayer" sí necesita los días previos: se calcula sobre la tabla de
entrenamiento más el mes nuevo.

Uso: python -m tc5061_proyecto_equipo07.modeling.monitoreo_p2 tabla | drift
"""

from loguru import logger
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset, features, monitoreo
from tc5061_proyecto_equipo07.modeling.predict import p2_decidir_clase
from tc5061_proyecto_equipo07.modeling.train import (
    P2BaselineHistorico,
    p2_calcular_metricas,
    p2_elegir_umbral,
)

app = typer.Typer(add_completion=False)

P2_LLAVE = ["estacion", "dia", "hora_idx"]
P2_COLUMNAS = [*cfg.P2_FEATURES_SIN_HISTORICO, "clase"]
P2_COLUMNAS_VIAJES = ["inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]

# Columnas de los reportes de Evidently: las variables del modelo. es_fin_semana solo toma 0 o 1
P2_CATEGORICAS = ["es_fin_semana"]
P2_NUMERICAS = [c for c in cfg.P2_FEATURES_CON_HISTORICO if c not in P2_CATEGORICAS]
# Probabilidades del modelo: monitorear el score es más sensible que la clase decidida
P2_PROBABILIDADES = ["prob_sin_bicis", "prob_sin_anclajes"]
P2_COLUMNAS_CONJUNTO = [*cfg.P2_FEATURES_CON_HISTORICO, "clase", "estacion", "dia", "hora_idx"]


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


def p2_conjuntos(ruta_entrenamiento=cfg.RUTA_P2_CAMBIOS, ruta_evaluacion=cfg.P2_RUTA_EVALUACION):
    """Regresa referencia, valid, actual y evaluacion con la historia "hasta ayer"

    La historia se calcula como al entrenar (frecuencia global de train y etiquetas de días
    previos), pero el periodo se asigna con `monitoreo.asignar_periodo`: así el día previo al mes
    nuevo también queda en `brecha`, como en las demás fronteras.
    """
    for ruta, comando in [
        (ruta_entrenamiento, "make features-p2"),
        (ruta_evaluacion, "make monitoreo-p2-tabla"),
    ]:
        if not ruta.exists():
            raise FileNotFoundError(f"Falta {ruta.name}; ejecutar `{comando}`")
    cambios = pd.read_parquet(ruta_entrenamiento)
    en_train = features.p2_asignar_particion(cambios["dia"]) == "train"
    historia = features.P2HistoriaHastaAyer(cfg.P2_PESO_HISTORICO).fit(cambios[en_train])
    cambios = historia.transform(cambios)[P2_COLUMNAS_CONJUNTO]
    periodo = monitoreo.asignar_periodo(cambios["dia"])
    conjuntos = {
        nombre: cambios[periodo == nombre].reset_index(drop=True)
        for nombre in ["referencia", "valid", "actual"]
    }
    del cambios
    conjuntos["evaluacion"] = pd.read_parquet(ruta_evaluacion, columns=P2_COLUMNAS_CONJUNTO)
    for nombre, datos in conjuntos.items():
        logger.info(
            f"{nombre}: {datos['dia'].min().date()} a {datos['dia'].max().date()} | "
            f"{len(datos)} filas"
        )
    return conjuntos


def p2_modelo_id(version) -> str | None:
    """Id del modelo de MLflow (m-...) de una versión: registro_p2 registra desde models:/<id>."""
    if version.model_id:
        return version.model_id
    if version.source.startswith("models:/m-"):
        return version.source.removeprefix("models:/").split("/")[0]
    logger.warning(f"No se encontró el modelo de MLflow de la versión {version.version}")
    return None


def p2_cargar_alias(alias: str) -> dict:
    """Carga la versión con el alias dado y los umbrales de su run de origen.

    Regresa modelo, umbrales, version, etiquetas y modelo_id. Los umbrales se leen del run igual
    que en la API: el modelo registrado solo da probabilidades.
    """
    if cfg.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    cliente = mlflow.MlflowClient()
    version = cliente.get_model_version_by_alias(cfg.P2_MODELO_REGISTRADO, alias)
    modelo = mlflow.sklearn.load_model(f"models:/{cfg.P2_MODELO_REGISTRADO}/{version.version}")
    parametros = cliente.get_run(version.run_id).data.params
    umbrales = {k: float(parametros[f"umbral_{cfg.P2_CLASES[k]}"]) for k in (1, 2)}
    logger.info(f"{cfg.P2_MODELO_REGISTRADO}: {alias} = versión {version.version} | {umbrales}")
    return {
        "modelo": modelo,
        "umbrales": umbrales,
        "version": int(version.version),
        "etiquetas": dict(version.tags),
        "modelo_id": p2_modelo_id(version),
    }


def p2_baseline(valid: pd.DataFrame) -> dict:
    """Baseline histórico con sus umbrales elegidos en valid, como al entrenar."""
    modelo = P2BaselineHistorico().fit(valid)
    proba = modelo.predict_proba(valid)
    y = valid["clase"].to_numpy()
    return {
        "modelo": modelo,
        "umbrales": {k: p2_elegir_umbral(y == k, proba[:, k]) for k in (1, 2)},
    }


def p2_predecir(version: dict, datos: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades de cada clase y clase decidida con los umbrales de la versión."""
    proba = version["modelo"].predict_proba(datos[cfg.P2_FEATURES_CON_HISTORICO])
    return proba, p2_decidir_clase(proba, version["umbrales"])


def p2_desempeno(modelos: dict[str, dict], conjuntos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Prevalencia, AP, log-loss y F1 de cada modelo en cada conjunto, sobre todas sus filas."""
    filas = []
    for nombre, version in modelos.items():
        for conjunto, datos in conjuntos.items():
            y = datos["clase"].to_numpy()
            proba, pred = p2_predecir(version, datos)
            _, _, f1, _ = precision_recall_fscore_support(y, pred, labels=[1, 2], zero_division=0)
            filas.append(
                {
                    "modelo": nombre,
                    "conjunto": conjunto,
                    "filas": len(datos),
                    "prevalencia_sin_bicis": float((y == 1).mean()),
                    "prevalencia_sin_anclajes": float((y == 2).mean()),
                    **p2_calcular_metricas(y, proba),
                    "f1_sin_bicis": f1[0],
                    "f1_sin_anclajes": f1[1],
                }
            )
    return pd.DataFrame(filas)


def p2_tabla_reporte(datos: pd.DataFrame, version: dict) -> pd.DataFrame:
    """Columnas de los reportes de Evidently: variables, probabilidades, clase y predicción.

    Clase y predicción van con su nombre (sin_riesgo, sin_bicis, sin_anclajes) para que las
    gráficas se lean sin traducir códigos.
    """
    tabla = datos[P2_NUMERICAS].astype("float64")
    for columna in P2_CATEGORICAS:
        tabla[columna] = datos[columna].astype(str)
    proba, pred = p2_predecir(version, datos)
    tabla["prob_sin_bicis"], tabla["prob_sin_anclajes"] = proba[:, 1], proba[:, 2]
    nombres = np.array(cfg.P2_CLASES)
    tabla["clase"] = nombres[datos["clase"].to_numpy()]
    tabla["prediccion"] = nombres[pred]
    return tabla


def p2_mismo_mes(referencia: pd.DataFrame, mes: str = cfg.MES_MONITOREO) -> pd.DataFrame:
    """Filas de la referencia del mismo mes del año que el mes de evaluación (control estacional)."""
    numero = pd.Period(mes, freq="M").month
    return referencia[referencia["dia"].dt.month == numero].reset_index(drop=True)


def p2_metricas_drift(snapshots: dict, desempeno: pd.DataFrame) -> dict[str, float]:
    """Proporción de columnas con drift por reporte, métricas por modelo y conjunto, y el cambio
    de AP de riesgo del champion respecto a valid."""
    metricas = {}
    for nombre, snapshot in snapshots.items():
        proporcion = monitoreo.proporcion_drift(snapshot)
        metricas[f"columnas_con_drift_{nombre}"] = proporcion["columnas_con_drift"]
        metricas[f"proporcion_drift_{nombre}"] = proporcion["proporcion_drift"]
    valores = [c for c in desempeno.columns if c not in ("modelo", "conjunto", "filas")]
    for fila in desempeno.to_dict("records"):
        for valor in valores:
            metricas[f"{fila['modelo']}_{valor}_{fila['conjunto']}"] = float(fila[valor])
    ap = desempeno[desempeno["modelo"] == "champion"].set_index("conjunto")["ap_riesgo"]
    for conjunto in ("actual", "evaluacion"):
        metricas[f"champion_cambio_ap_riesgo_{conjunto}_vs_valid"] = ap[conjunto] / ap["valid"] - 1
    return metricas


def p2_separar_metricas_champion(metricas: dict, conjuntos) -> dict[str, dict[str, float]]:
    """Saca de `metricas` las del champion por conjunto, para asociarlas a su modelo y dataset."""
    return {
        conjunto: {
            clave: metricas.pop(clave)
            for clave in list(metricas)
            if clave.startswith("champion_")
            and clave.endswith(f"_{conjunto}")
            and "_cambio_" not in clave
        }
        for conjunto in conjuntos
    }


def p2_datasets(conjuntos: dict[str, pd.DataFrame]) -> dict:
    """Conjuntos de monitoreo como datasets de MLflow, con la tabla de la que salen."""
    fuentes = {"evaluacion": cfg.P2_RUTA_EVALUACION}
    return {
        nombre: monitoreo.dataset_mlflow(
            datos, f"p2_{nombre}", fuentes.get(nombre, cfg.RUTA_P2_CAMBIOS), "clase"
        )
        for nombre, datos in conjuntos.items()
    }


def p2_mes_previo(frontera: str) -> str:
    """Mes anterior a una frontera ("2024-10-01" -> "2024-09"), para describir los periodos."""
    return str(pd.Period(frontera, freq="M") - 1)


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


@app.command()
def drift(muestra: int = cfg.MONITOREO_MUESTRA):
    """Reportes de drift, desempeño y estacionalidad del champion, registrados como run en MLflow.

    1. Drift y calidad: referencia (train) contra actual (test), con probabilidades y predicción.
    2. Desempeño: valid contra actual, con la clase real (ClassificationPreset).
    3. Estacionalidad: el mismo mes en la referencia contra el mes de evaluación.
    """
    conjuntos = p2_conjuntos(cfg.RUTA_P2_CAMBIOS, cfg.P2_RUTA_EVALUACION)
    champion = p2_cargar_alias("champion")
    baseline = p2_baseline(conjuntos["valid"])

    evaluados = {c: conjuntos[c] for c in ("valid", "actual", "evaluacion")}
    desempeno = p2_desempeno({"champion": champion, "baseline": baseline}, evaluados)
    ap_valid = desempeno.query("modelo == 'champion' and conjunto == 'valid'")["ap_riesgo"].iloc[0]
    registrado = champion["etiquetas"].get("valid_ap_riesgo")
    if registrado is not None and not np.isclose(ap_valid, float(registrado), atol=1e-3):
        logger.warning(
            f"AP de riesgo en valid {ap_valid:.4f} distinta a la registrada ({registrado})"
        )

    def muestra_de(datos: pd.DataFrame) -> pd.DataFrame:
        return p2_tabla_reporte(monitoreo.muestrear(datos, muestra), champion)

    referencia_m, actual_m = muestra_de(conjuntos["referencia"]), muestra_de(conjuntos["actual"])
    numericas = P2_NUMERICAS + P2_PROBABILIDADES
    sin_tarea = monitoreo.definicion_datos(numericas, P2_CATEGORICAS + ["clase", "prediccion"])
    con_tarea = monitoreo.definicion_datos(
        numericas, P2_CATEGORICAS, "clasificacion", "clase", "prediccion"
    )
    carpeta = cfg.MONITOREO_DIR
    drift_tt, rutas_tt = monitoreo.generar_reporte(
        referencia_m, actual_m, sin_tarea, "p2_drift_train_test", carpeta=carpeta
    )
    _, rutas_dt = monitoreo.generar_reporte(
        muestra_de(conjuntos["valid"]),
        actual_m,
        con_tarea,
        "p2_desempeno_valid_test",
        "clasificacion",
        carpeta,
    )
    drift_est, rutas_est = monitoreo.generar_reporte(
        muestra_de(p2_mismo_mes(conjuntos["referencia"])),
        muestra_de(conjuntos["evaluacion"]),
        sin_tarea,
        "p2_drift_estacional",
        carpeta=carpeta,
    )

    resumen = pd.concat(
        [
            monitoreo.resumen_drift(drift_tt).assign(comparacion="train_vs_test"),
            monitoreo.resumen_drift(drift_est).assign(comparacion="mismo_mes_vs_evaluacion"),
        ],
        ignore_index=True,
    )
    ruta_resumen = carpeta / "p2_drift_resumen.csv"
    ruta_desempeno = carpeta / "p2_desempeno.csv"
    resumen.to_csv(ruta_resumen, index=False)
    desempeno.to_csv(ruta_desempeno, index=False)
    logger.info(f"Drift por variable:\n{resumen.to_string(index=False)}")
    logger.info(f"Desempeño por conjunto:\n{desempeno.to_string(index=False)}")

    metricas = p2_metricas_drift({"train_test": drift_tt, "estacional": drift_est}, desempeno)
    version = champion["version"]
    return monitoreo.registrar_en_mlflow(
        "p2-drift",
        "p2",
        [*rutas_tt, *rutas_dt, *rutas_est, ruta_resumen, ruta_desempeno],
        metricas=metricas,
        metricas_por_dataset=p2_separar_metricas_champion(metricas, evaluados),
        datasets=p2_datasets(conjuntos),
        modelo_id=champion["modelo_id"],
        descripcion=(
            f"Monitoreo de P2 sobre el champion v{version}: drift de datos (train contra test), "
            f"desempeño (valid contra test) y control estacional ({cfg.MES_MONITOREO} contra el "
            "mismo mes de la referencia), con Evidently. Champion y baseline histórico evaluados "
            "en valid, test y el mes nuevo."
        ),
        parametros={
            "champion_version": version,
            "muestra": muestra,
            "referencia": f"{cfg.MES_INICIO} a {p2_mes_previo(cfg.FRONTERA_VALID)}",
            "actual": f"{cfg.FRONTERA_TEST[:7]} a {p2_mes_previo(cfg.FRONTERA_MONITOREO)}",
            **{f"umbral_{cfg.P2_CLASES[k]}": v for k, v in champion["umbrales"].items()},
        },
        etiquetas={
            "champion_version": str(version),
            "tipo_monitoreo": "drift",
            "autor": "Fernando Arreola Treviño",
        },
    )


if __name__ == "__main__":
    app()
