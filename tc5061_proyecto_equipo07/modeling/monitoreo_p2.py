"""Monitoreo de P2 (riesgo de quedarse sin bicis o sin anclajes): conjuntos, drift, versión
alternativa y A/B testing.

Usa la base común de monitoreo.py. La etiqueta y las variables de flujo de P2 no cruzan el día de
servicio, así que el mes nuevo se calcula solo con sus viajes (más un margen para los que empiezan
antes). La historia "hasta ayer" sí necesita los días previos: se calcula sobre la tabla de
entrenamiento más el mes nuevo.

Uso: python -m tc5061_proyecto_equipo07.modeling.monitoreo_p2 tabla | drift
"""

from evidently import DataDefinition, MulticlassClassification
from loguru import logger
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset, features, monitoreo, plots
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

# Con menos casos positivos que esto, AUC y AP de un mes o una cohorte son ruido: quedan vacíos
P2_MIN_POSITIVOS = 30

P2_RIESGOS = {1: "sin_bicis", 2: "sin_anclajes"}
# Reportes de drift de Evidently y el nombre de su comparación en el resumen y las métricas
P2_COMPARACIONES_DRIFT = {"p2_drift_train_test": "train_test", "p2_drift_estacional": "estacional"}


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


def p2_metricas(y: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Métricas de un grupo de filas, separando lo que puede cambiar con el tiempo.

    prevalencia_*   cuánto riesgo hubo (depende de los datos, no del modelo)
    prediccion_*    probabilidad media del modelo; calibracion_* = prediccion / prevalencia (1 = bien)
    auc_*           si ordena bien las filas por riesgo; no depende de la prevalencia
    lift_ap_*       AP entre prevalencia: cuántas veces mejor que el azar
    ap_*, log_loss  las del entrenamiento: dependen de la prevalencia, comparan modelos sobre las
                    mismas filas pero no periodos entre sí
    f1_*            con los umbrales del modelo
    """
    metricas = {}
    for k, riesgo in P2_RIESGOS.items():
        positivo = y == k
        prevalencia, prediccion = positivo.mean(), proba[:, k].mean()
        suficiente = P2_MIN_POSITIVOS <= positivo.sum() < len(y)
        metricas |= {
            f"prevalencia_{riesgo}": prevalencia,
            f"prediccion_{riesgo}": prediccion,
            f"calibracion_{riesgo}": prediccion / prevalencia if prevalencia else np.nan,
            f"auc_{riesgo}": roc_auc_score(positivo, proba[:, k]) if suficiente else np.nan,
            f"lift_ap_{riesgo}": (
                average_precision_score(positivo, proba[:, k]) / prevalencia
                if suficiente
                else np.nan
            ),
        }
    riesgo = y > 0
    suficiente = P2_MIN_POSITIVOS <= riesgo.sum() < len(y)
    metricas["auc_riesgo"] = roc_auc_score(riesgo, 1 - proba[:, 0]) if suficiente else np.nan
    _, _, f1, _ = precision_recall_fscore_support(y, pred, labels=[1, 2], zero_division=0)
    return {
        **metricas,
        **p2_calcular_metricas(y, proba),
        "f1_sin_bicis": f1[0],
        "f1_sin_anclajes": f1[1],
    }


def p2_desempeno(
    modelos: dict[str, dict],
    conjuntos: dict[str, pd.DataFrame],
    estaciones_referencia: np.ndarray,
    por_mes: bool = False,
) -> pd.DataFrame:
    """Métricas de cada modelo por conjunto (o por mes, con `por_mes`) y cohorte de estaciones.

    Cohortes: todas, conocidas (con filas en la referencia) y nuevas (abrieron después de train;
    el modelo nunca las vio y su riesgo, capacidad y flujo pueden ser distintos).
    """
    filas = []
    for conjunto, datos in conjuntos.items():
        y = datos["clase"].to_numpy()
        nuevas = ~datos["estacion"].isin(estaciones_referencia).to_numpy()
        if por_mes:
            grupos = datos["dia"].dt.to_period("M").astype(str).to_numpy()
        else:
            grupos = np.full(len(datos), conjunto)
        for nombre, version in modelos.items():
            proba, pred = p2_predecir(version, datos)
            for grupo in np.unique(grupos):
                del_grupo = grupos == grupo
                cohortes = {"todas": del_grupo, "conocidas": del_grupo & ~nuevas}
                cohortes["nuevas"] = del_grupo & nuevas
                for cohorte, filas_c in cohortes.items():
                    if not (y[filas_c] > 0).any():
                        continue
                    filas.append(
                        {
                            "modelo": nombre,
                            "conjunto": conjunto,
                            **({"mes": grupo} if por_mes else {}),
                            "cohorte": cohorte,
                            "filas": int(filas_c.sum()),
                            "proporcion_estaciones_nuevas": float(nuevas[del_grupo].mean()),
                            **p2_metricas(y[filas_c], proba[filas_c], pred[filas_c]),
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


def p2_definicion_desempeno() -> DataDefinition:
    """Esquema del reporte de desempeño: clase, predicción y la probabilidad de cada clase.

    Evidently empareja cada columna de probabilidad con la clase del mismo nombre; con ellas agrega
    ROC-AUC y log-loss por clase, que no dependen del umbral. `definicion_datos` de la base solo
    admite la clase predicha, así que el esquema se arma aquí.
    """
    return DataDefinition(
        numerical_columns=P2_NUMERICAS + P2_PROBABILIDADES,
        categorical_columns=P2_CATEGORICAS + ["clase", "prediccion"],
        classification=[
            MulticlassClassification(
                target="clase", prediction_labels="prediccion", prediction_probas=cfg.P2_CLASES
            )
        ],
    )


def p2_tabla_desempeno(tabla: pd.DataFrame) -> pd.DataFrame:
    """Agrega a la tabla del reporte una columna de probabilidad por clase, con su nombre."""
    return tabla.assign(
        sin_riesgo=1 - tabla["prob_sin_bicis"] - tabla["prob_sin_anclajes"],
        sin_bicis=tabla["prob_sin_bicis"],
        sin_anclajes=tabla["prob_sin_anclajes"],
    )


def p2_mismo_mes(referencia: pd.DataFrame, mes: str = cfg.MES_MONITOREO) -> pd.DataFrame:
    """Filas de la referencia del mismo mes del año que el mes de evaluación (control estacional)."""
    numero = pd.Period(mes, freq="M").month
    return referencia[referencia["dia"].dt.month == numero].reset_index(drop=True)


def p2_metricas_run(snapshots: dict, desempeno: pd.DataFrame) -> tuple[dict, dict]:
    """Métricas del run de drift: solo las que sostienen la lectura; los CSV traen todo.

    Regresa (del run, del champion por conjunto). Las del champion son su desempeño en cada
    conjunto y van a su página en MLflow; las del run son el drift, la prevalencia, la calibración,
    el AUC por cohorte de estaciones y el baseline como referencia.
    """
    run = {}
    for reporte, comparacion in P2_COMPARACIONES_DRIFT.items():
        proporcion = monitoreo.proporcion_drift(snapshots[reporte])
        run[f"columnas_con_drift_{comparacion}"] = proporcion["columnas_con_drift"]
        run[f"proporcion_drift_{comparacion}"] = proporcion["proporcion_drift"]
    filas = desempeno.set_index(["modelo", "cohorte", "conjunto"])
    del_champion = {}
    for conjunto in desempeno["conjunto"].unique():
        champion, baseline = (
            filas.loc[("champion", "todas", conjunto)],
            filas.loc[("baseline", "todas", conjunto)],
        )
        del_champion[conjunto] = {
            f"champion_{m}_{conjunto}": champion[m]
            for m in ("auc_riesgo", "ap_riesgo", "log_loss", "f1_sin_bicis", "f1_sin_anclajes")
        }
        for m in (
            "proporcion_estaciones_nuevas",
            "prevalencia_sin_bicis",
            "prevalencia_sin_anclajes",
        ):
            run[f"{m}_{conjunto}"] = champion[m]
        for m in ("calibracion_sin_bicis", "calibracion_sin_anclajes"):
            run[f"champion_{m}_{conjunto}"] = champion[m]
        for m in ("auc_riesgo", "ap_riesgo"):
            run[f"baseline_{m}_{conjunto}"] = baseline[m]
        for cohorte in ("conocidas", "nuevas"):
            if ("champion", cohorte, conjunto) in filas.index:
                for m in ("auc_riesgo", "calibracion_sin_bicis"):
                    run[f"champion_{m}_{cohorte}_{conjunto}"] = filas.loc[
                        ("champion", cohorte, conjunto), m
                    ]
    auc = {c: m[f"champion_auc_riesgo_{c}"] for c, m in del_champion.items()}
    for conjunto in ("actual", "evaluacion"):
        run[f"champion_cambio_auc_riesgo_{conjunto}_vs_valid"] = auc[conjunto] / auc["valid"] - 1

    def limpiar(metricas: dict) -> dict[str, float]:
        return {k: float(v) for k, v in metricas.items() if pd.notna(v)}

    return limpiar(run), {c: limpiar(m) for c, m in del_champion.items()}


def p2_datasets(conjuntos: dict[str, pd.DataFrame], referencia_muestra: pd.DataFrame) -> dict:
    """Conjuntos como datasets de MLflow. La referencia va con la muestra que vio Evidently: es lo
    único que se usó de ella, y registrar sus 19 M de filas tomaba unos 4 minutos."""
    fuentes = {"evaluacion": cfg.P2_RUTA_EVALUACION}
    datasets = {
        nombre: monitoreo.dataset_mlflow(
            datos, f"p2_{nombre}", fuentes.get(nombre, cfg.RUTA_P2_CAMBIOS), "clase"
        )
        for nombre, datos in conjuntos.items()
    }
    datasets["referencia"] = monitoreo.dataset_mlflow(
        referencia_muestra, "p2_referencia_muestra", cfg.RUTA_P2_CAMBIOS, "clase"
    )
    return datasets


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
    """Drift, desempeño y serie mensual del champion, registrados como un run en MLflow.

    Reportes de Evidently: drift de referencia (train) contra actual (test), desempeño con
    probabilidades de valid contra actual y contra el mes nuevo, y control estacional (el mismo mes
    de la referencia contra el mes nuevo). Tablas: desempeño por conjunto y por mes, por cohorte de
    estaciones, y el mismo mes de cada año; gráfica de la serie mensual.
    """
    conjuntos = p2_conjuntos(cfg.RUTA_P2_CAMBIOS, cfg.P2_RUTA_EVALUACION)
    # De la referencia solo se usan sus estaciones, una muestra y su mismo mes: se libera el resto
    referencia = conjuntos.pop("referencia")
    estaciones = referencia["estacion"].unique()
    referencia_m, mismo_mes = monitoreo.muestrear(referencia, muestra), p2_mismo_mes(referencia)
    del referencia
    champion = p2_cargar_alias("champion")
    modelos = {"champion": champion, "baseline": p2_baseline(conjuntos["valid"])}
    desempeno = p2_desempeno(modelos, conjuntos, estaciones)
    # El mismo mes de cada año de train entra a la serie como control estacional
    mensual = p2_desempeno(modelos, {"referencia": mismo_mes, **conjuntos}, estaciones, True)
    ap_valid = desempeno.query(
        "modelo == 'champion' and conjunto == 'valid' and cohorte == 'todas'"
    )["ap_riesgo"].iloc[0]
    registrado = champion["etiquetas"].get("valid_ap_riesgo")
    if registrado is not None and not np.isclose(ap_valid, float(registrado), atol=1e-3):
        logger.warning(
            f"AP de riesgo en valid {ap_valid:.4f} distinta a la registrada ({registrado})"
        )

    def muestra_de(datos: pd.DataFrame) -> pd.DataFrame:
        return p2_tabla_reporte(monitoreo.muestrear(datos, muestra), champion)

    actual_m, evaluacion_m = muestra_de(conjuntos["actual"]), muestra_de(conjuntos["evaluacion"])
    valid_m = p2_tabla_desempeno(muestra_de(conjuntos["valid"]))
    sin_tarea = monitoreo.definicion_datos(
        P2_NUMERICAS + P2_PROBABILIDADES, P2_CATEGORICAS + ["clase", "prediccion"]
    )
    con_tarea = p2_definicion_desempeno()
    reportes = {  # nombre: (referencia, actual, esquema, tarea)
        "p2_drift_train_test": (
            p2_tabla_reporte(referencia_m, champion),
            actual_m,
            sin_tarea,
            None,
        ),
        "p2_desempeno_valid_test": (
            valid_m,
            p2_tabla_desempeno(actual_m),
            con_tarea,
            "clasificacion",
        ),
        "p2_desempeno_valid_evaluacion": (
            valid_m,
            p2_tabla_desempeno(evaluacion_m),
            con_tarea,
            "clasificacion",
        ),
        "p2_drift_estacional": (muestra_de(mismo_mes), evaluacion_m, sin_tarea, None),
    }
    carpeta, snapshots, archivos = cfg.MONITOREO_DIR, {}, []
    for nombre, (de_referencia, de_actual, esquema, tarea) in reportes.items():
        snapshots[nombre], rutas = monitoreo.generar_reporte(
            de_referencia, de_actual, esquema, nombre, tarea, carpeta
        )
        archivos += rutas

    resumen = pd.concat(
        [
            monitoreo.resumen_drift(snapshots[reporte]).assign(comparacion=comparacion)
            for reporte, comparacion in P2_COMPARACIONES_DRIFT.items()
        ],
        ignore_index=True,
    )
    mes = f"-{pd.Period(cfg.MES_MONITOREO, freq='M').month:02d}"
    tablas = {
        "p2_drift_resumen": resumen,
        "p2_desempeno": desempeno,
        "p2_serie_mensual": mensual,
        "p2_mismo_mes_por_anio": mensual[
            mensual["mes"].str.endswith(mes) & (mensual["cohorte"] == "todas")
        ],
    }
    for nombre, contenido in tablas.items():
        archivos.append(carpeta / f"{nombre}.csv")
        contenido.to_csv(archivos[-1], index=False)
    figura = plots.p2_grafica_serie_mensual(mensual)
    archivos.append(plots.p3_guardar_figura(figura, carpeta / "p2_serie_mensual.png"))
    columnas = [
        "modelo",
        "conjunto",
        "cohorte",
        "auc_riesgo",
        "calibracion_sin_bicis",
        "ap_riesgo",
    ]
    logger.info(f"Drift por variable:\n{resumen.to_string(index=False)}")
    logger.info(f"Desempeño:\n{desempeno[columnas].round(4).to_string(index=False)}")

    metricas, del_champion = p2_metricas_run(snapshots, desempeno)
    version = champion["version"]
    return monitoreo.registrar_en_mlflow(
        "p2-drift",
        "p2",
        archivos,
        metricas=metricas,
        metricas_por_dataset=del_champion,
        datasets=p2_datasets(conjuntos, referencia_m),
        modelo_id=champion["modelo_id"],
        descripcion=(
            f"Monitoreo de P2 sobre el champion v{version}, con Evidently: drift de datos (train "
            f"contra test), desempeño con probabilidades (valid contra test y contra "
            f"{cfg.MES_MONITOREO}) y control estacional (el mismo mes de la referencia contra "
            f"{cfg.MES_MONITOREO}). Champion y baseline histórico por conjunto y por mes, por "
            "cohorte de estaciones (conocidas en train o nuevas): AUC, calibración, AP, log-loss y F1."
        ),
        parametros={
            "champion_version": version,
            "muestra": muestra,
            "min_positivos": P2_MIN_POSITIVOS,
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
