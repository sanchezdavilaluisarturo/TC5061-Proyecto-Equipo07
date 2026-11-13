"""Monitoreo de P3 (segmentación de estaciones): perfiles por periodo, drift y champion.

Usa la base común de monitoreo.py. P3 no modela filas sino estaciones: cada conjunto es una tabla
con una fila por estación (el perfil horario laboral y de fin de semana, más cuatro indicadores de
uso), calculada con las mismas funciones con las que se entrenó el champion:
    referencia   perfiles de train (2020-01 a 2024-09), con los que se ajustó el champion
    actual       perfiles de test (2025-10 a 2026-09)
    evaluacion   perfiles del mes nuevo (2026-10), que ningún modelo ha visto
valid (2024-10 a 2025-09) también se calcula: es el conjunto con el que se midió la estabilidad.

Uso: python -m tc5061_proyecto_equipo07.modeling.monitoreo_p3 perfiles | drift | drift-anual | challenger | ab | silueta
"""

from loguru import logger
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset, features, monitoreo

app = typer.Typer(add_completion=False)

# Periodos de monitoreo que se convierten en perfiles; la brecha se descarta
P3_PERIODOS = ["referencia", "valid", "actual", "evaluacion"]
# Mínimo de salidas por estación en cada periodo. Referencia y valid son los umbrales del
# entrenamiento; el mes nuevo usa el ritmo mensual de valid (300 salidas en 12 meses), porque
# con 300 en un solo mes se perdería gran parte de las estaciones
P3_MIN_SALIDAS = {
    "referencia": cfg.P3_MIN_SALIDAS_TRAIN,
    "valid": cfg.P3_MIN_SALIDAS_VALID,
    "actual": cfg.P3_MIN_SALIDAS_VALID,
    "evaluacion": cfg.P3_MIN_SALIDAS_EVALUACION,
}
# Columnas del perfil: 40 proporciones horarias (20 horas de servicio, laboral y fin de semana) y 4 indicadores. `salidas` no entra al modelo
P3_INDICADORES = [
    "log_viajes_por_dia",
    "neto_sobre_salidas",
    "prop_circulares",
    "duracion_mediana_min",
]
P3_COLUMNAS_PERFIL = [
    *[f"lab_h{hora:02d}" for hora in cfg.P3_HORAS_SERVICIO],
    *[f"fin_h{hora:02d}" for hora in cfg.P3_HORAS_SERVICIO],
    *P3_INDICADORES,
]


def p3_perfil_filtrado(viajes: pd.DataFrame, minimo_salidas: int) -> pd.DataFrame:
    """Perfil por estación de unos viajes, solo con las estaciones de al menos `minimo_salidas`."""
    viajes = features.p3_variables_de_tiempo(viajes.copy())
    perfil = features.p3_perfil_de_estaciones(viajes)
    return perfil[perfil["salidas"] >= minimo_salidas][[*P3_COLUMNAS_PERFIL, "salidas"]]


def p3_perfiles_por_periodo(viajes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Perfil por estación de cada periodo (índice `origen_id`, columnas del modelo y `salidas`).

    `viajes` trae inicio_del_viaje, fin_del_viaje, origen_id y destino_id. Las horas en `brecha` se
    descartan, igual que en la partición de entrenamiento, y cada periodo se filtra con el mínimo
    de salidas que le corresponde.

    Además del periodo, devuelve el control estacional: el mismo mes del año que el mes nuevo
    (octubre) en cada año previo (`mismo_mes_AAAA`) y los de la referencia juntos
    (`mismo_mes_referencia`). Un mes contra un año completo confunde estacionalidad con drift.
    """
    periodo = monitoreo.asignar_periodo(viajes["inicio_del_viaje"])
    perfiles = {}
    for nombre in P3_PERIODOS:
        perfil = p3_perfil_filtrado(viajes[(periodo == nombre).to_numpy()], P3_MIN_SALIDAS[nombre])
        perfiles[nombre] = perfil
        logger.info(f"{nombre}: {len(perfil)} estaciones")

    # Datos del challenger: train, valid y test juntos (hasta MES_FIN) y el mínimo de salidas del
    # entrenamiento del champion; el mes nuevo y las brechas quedan fuera
    entrenamiento = periodo.isin(["referencia", "valid", "actual"]).to_numpy()
    perfiles["entrenamiento_challenger"] = p3_perfil_filtrado(
        viajes[entrenamiento], cfg.P3_MIN_SALIDAS_TRAIN
    )
    logger.info(
        f"entrenamiento_challenger: {len(perfiles['entrenamiento_challenger'])} estaciones"
    )

    inicio = viajes["inicio_del_viaje"]
    fecha_evaluacion = pd.Period(cfg.MES_MONITOREO, freq="M")
    en_mes = (inicio.dt.month == fecha_evaluacion.month) & (inicio.dt.year < fecha_evaluacion.year)
    for anio in sorted(inicio[en_mes].dt.year.unique()):
        del_anio = viajes[(en_mes & (inicio.dt.year == anio)).to_numpy()]
        perfiles[f"mismo_mes_{anio}"] = p3_perfil_filtrado(del_anio, cfg.P3_MIN_SALIDAS_EVALUACION)
    en_referencia = en_mes & (periodo == "referencia")
    anios_referencia = inicio[en_referencia].dt.year.nunique()
    perfiles["mismo_mes_referencia"] = p3_perfil_filtrado(
        viajes[en_referencia.to_numpy()], cfg.P3_MIN_SALIDAS_EVALUACION * anios_referencia
    )
    for nombre, perfil in perfiles.items():
        if nombre.startswith("mismo_mes"):
            logger.info(f"{nombre}: {len(perfil)} estaciones")
    return perfiles


def p3_comparar_con_entrenamiento(
    perfiles: dict[str, pd.DataFrame], modelado: pd.DataFrame
) -> dict[str, int]:
    """Comprueba que referencia y valid reproducen las variables con las que se entrenó el champion.

    `modelado` es el parquet de entrenamiento (con `particion` y `duracion_min`). Si los perfiles
    del monitoreo difirieran, el drift mediría una diferencia de procesamiento y no de los datos
    (train-serving skew).
    """
    train, valid, columnas = features.p3_construir_caracteristicas(modelado)
    if columnas != P3_COLUMNAS_PERFIL:
        raise ValueError("Las columnas del modelo no coinciden con las del monitoreo")
    for nombre, esperado in (("referencia", train), ("valid", valid)):
        obtenido = perfiles[nombre]
        if set(obtenido.index) != set(esperado.index):
            raise ValueError(f"{nombre}: las estaciones no coinciden con las del entrenamiento")
        if not np.allclose(
            obtenido.loc[esperado.index, columnas].to_numpy(float),
            esperado[columnas].to_numpy(float),
            equal_nan=True,
        ):
            raise ValueError(f"{nombre}: los perfiles difieren de los del entrenamiento")
    return {"estaciones_referencia": len(train), "estaciones_valid": len(valid)}


def p3_conjuntos(ruta=cfg.P3_RUTA_PERFILES_MONITOREO) -> dict[str, pd.DataFrame]:
    """Lee los perfiles guardados por `perfiles` y los separa por periodo."""
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta.name}; ejecutar `make monitoreo-p3-perfiles`")
    tabla = pd.read_parquet(ruta).set_index("origen_id")
    return {nombre: grupo.drop(columns="periodo") for nombre, grupo in tabla.groupby("periodo")}


def p3_cargar_champion():
    """Carga el champion con el sabor sklearn; regresa (Pipeline, versión, etiquetas, model_id)."""
    if cfg.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    version = mlflow.MlflowClient().get_model_version_by_alias(
        cfg.P3_MODELO_REGISTRADO, "champion"
    )
    modelo = mlflow.sklearn.load_model(f"models:/{cfg.P3_MODELO_REGISTRADO}/{version.version}")
    logger.info(f"{cfg.P3_MODELO_REGISTRADO}: champion = versión {version.version}")
    return modelo, int(version.version), dict(version.tags), version.model_id or None


def p3_tabla_reporte(perfil: pd.DataFrame, modelo) -> pd.DataFrame:
    """Columnas de Evidently: las 44 variables y el grupo que el champion asigna a la estación.

    El grupo va como categórica: es una etiqueta de grupo, no una cantidad.
    """
    tabla = perfil[P3_COLUMNAS_PERFIL].astype("float64").reset_index(drop=True)
    tabla["grupo"] = modelo.predict(tabla[P3_COLUMNAS_PERFIL]).astype(str)
    return tabla


@app.command()
def perfiles():
    """Guarda los perfiles de referencia, valid, actual y evaluacion en un solo parquet.

    Requiere `make monitoreo-datos`. Antes de guardar comprueba que referencia y valid son
    idénticos a las variables de entrenamiento del champion.
    """
    for ruta, comando in [
        (cfg.RUTA_VIAJES_LIMPIOS_EXTENDIDOS, "make monitoreo-datos"),
        (cfg.RUTA_VIAJES_MODELADO, "make data"),
    ]:
        if not ruta.exists():
            raise FileNotFoundError(f"Falta {ruta.name}; ejecutar `{comando}`")
    columnas = ["inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]
    viajes = pd.read_parquet(cfg.RUTA_VIAJES_LIMPIOS_EXTENDIDOS, columns=columnas)
    conjuntos = p3_perfiles_por_periodo(viajes)
    del viajes

    modelado = pd.read_parquet(
        cfg.RUTA_VIAJES_MODELADO,
        columns=[
            "inicio_del_viaje",
            "fin_del_viaje",
            "origen_id",
            "destino_id",
            "duracion_min",
            "particion",
        ],
    )
    cuentas = p3_comparar_con_entrenamiento(conjuntos, modelado)
    logger.success(f"Perfiles iguales a los del entrenamiento: {cuentas}")

    tabla = pd.concat(
        [perfil.assign(periodo=nombre) for nombre, perfil in conjuntos.items()]
    ).rename_axis("origen_id")
    cfg.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tabla.reset_index().to_parquet(cfg.P3_RUTA_PERFILES_MONITOREO, index=False)
    logger.success(f"{cfg.P3_RUTA_PERFILES_MONITOREO.name}: {len(tabla)} filas estación-periodo")


def p3_comunes(
    referencia: pd.DataFrame, actual: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deja en los dos conjuntos solo las estaciones que están en ambos.

    Separa el cambio de comportamiento de las estaciones que ya existían del cambio de composición
    (estaciones que abren o cierran, que entran o salen de la red).
    """
    comunes = referencia.index.intersection(actual.index)
    return referencia.loc[comunes], actual.loc[comunes]


@app.command()
def drift():
    """Reportes de drift y calidad de las variables y de la asignación de grupo del champion.

    Cada comparación se corre con todas las estaciones y solo con las comunes a los dos conjuntos:
    1. train_test: referencia (train) contra actual (test), un año completo.
    2. estacional: el mismo mes de la referencia contra el mes nuevo (control estacional).
    3. contraste: toda la referencia contra el mes nuevo; se guarda para mostrar cuánto de su drift
       es estacionalidad, no para decidir con él.
    """
    conjuntos = p3_conjuntos()
    modelo, version, _, _ = p3_cargar_champion()
    definicion = monitoreo.definicion_datos(P3_COLUMNAS_PERFIL, ["grupo"])
    comparaciones = {
        "train_test": ("referencia", "actual"),
        "estacional": ("mismo_mes_referencia", "evaluacion"),
        "contraste": ("referencia", "evaluacion"),
    }
    resumenes, cuentas = [], []
    for nombre, (de, a) in comparaciones.items():
        completas = (conjuntos[de], conjuntos[a])
        for variante, (ref, act) in {
            "todas": completas,
            "comunes": p3_comunes(*completas),
        }.items():
            if nombre == "contraste" and variante == "comunes":
                continue
            clave = f"{nombre}_{variante}"
            snapshot, _ = monitoreo.generar_reporte(
                p3_tabla_reporte(ref, modelo),
                p3_tabla_reporte(act, modelo),
                definicion,
                f"p3_drift_{clave}",
                carpeta=cfg.MONITOREO_DIR,
            )
            resumenes.append(monitoreo.resumen_drift(snapshot).assign(comparacion=clave))
            cuentas.append(
                {
                    "comparacion": clave,
                    "estaciones_referencia": len(ref),
                    "estaciones_actual": len(act),
                    **monitoreo.proporcion_drift(snapshot),
                }
            )
    resumen = pd.concat(resumenes, ignore_index=True)
    tabla = pd.DataFrame(cuentas)
    resumen.to_csv(cfg.MONITOREO_DIR / "p3_drift_resumen.csv", index=False)
    tabla.to_csv(cfg.MONITOREO_DIR / "p3_drift_conteo.csv", index=False)
    logger.info(f"Drift del conjunto (champion v{version}):\n{tabla.to_string(index=False)}")


@app.command()
def drift_anual():
    """El mismo mes de cada año previo contra el mes nuevo: separa el drift real de un año atípico.

    2020 (pandemia) no es comparable con 2026 aunque sea el mismo mes; el desglose lo muestra.
    """
    conjuntos = p3_conjuntos()
    modelo, version, _, _ = p3_cargar_champion()
    definicion = monitoreo.definicion_datos(P3_COLUMNAS_PERFIL, ["grupo"])
    actual = p3_tabla_reporte(conjuntos["evaluacion"], modelo)
    filas = []
    for nombre in sorted(n for n in conjuntos if n.startswith("mismo_mes_20")):
        anio = int(nombre.removeprefix("mismo_mes_"))
        snapshot, _ = monitoreo.generar_reporte(
            p3_tabla_reporte(conjuntos[nombre], modelo),
            actual,
            definicion,
            f"p3_drift_mes_{anio}",
            carpeta=cfg.MONITOREO_DIR,
        )
        filas.append(
            {
                "anio": anio,
                "estaciones": len(conjuntos[nombre]),
                **monitoreo.proporcion_drift(snapshot),
            }
        )
    tabla = pd.DataFrame(filas)
    tabla.to_csv(cfg.MONITOREO_DIR / "p3_drift_mismo_mes_por_anio.csv", index=False)
    logger.info(f"Mismo mes por año contra {cfg.MES_MONITOREO} (champion v{version}):\n{tabla}")


def p3_hiperparametros(pipeline) -> dict:
    """Parámetros escalares de cada paso (`paso__parametro`), sin los objetos de los pasos."""
    return {k: v for k, v in pipeline.get_params().items() if "__" in k}


def p3_reentrenar(champion, perfil: pd.DataFrame):
    """Challenger: el pipeline del champion con los mismos hiperparámetros, ajustado con `perfil`.

    `clone` copia la configuración (escalador, PCA, K-Means, semilla) sin los parámetros ajustados,
    así que la única diferencia entre las dos versiones son los datos de entrenamiento.
    """
    modelo = clone(champion)
    if p3_hiperparametros(modelo) != p3_hiperparametros(champion):
        raise ValueError("El challenger no tiene los hiperparámetros del champion")
    return modelo.fit(perfil[P3_COLUMNAS_PERFIL])


def p3_espacio_comun(champion, perfil: pd.DataFrame) -> np.ndarray:
    """Variables de `perfil` estandarizadas con el escalador del champion.

    Es el espacio en el que se comparan los dos modelos: la silueta de un modelo medida tras su
    propio PCA no es comparable con la del otro. Mismo criterio que `train.p3_evaluar_modelo`.
    """
    return champion.named_steps["escalador"].transform(perfil[P3_COLUMNAS_PERFIL])


def p3_resumen_challenger(champion, challenger, perfil: pd.DataFrame) -> dict[str, float]:
    """Cómo queda el challenger frente al champion sobre las estaciones con las que se entrenó."""
    z = p3_espacio_comun(champion, perfil)
    x = perfil[P3_COLUMNAS_PERFIL]
    grupos_champion, grupos_challenger = champion.predict(x), challenger.predict(x)
    return {
        "estaciones": len(perfil),
        "silueta_champion": float(silhouette_score(z, grupos_champion)),
        "silueta_challenger": float(silhouette_score(z, grupos_challenger)),
        "ari_champion_challenger": float(adjusted_rand_score(grupos_champion, grupos_challenger)),
        "tam_grupo_min_champion": int(np.bincount(grupos_champion).min()),
        "tam_grupo_min_challenger": int(np.bincount(grupos_challenger).min()),
        "varianza_pca_challenger": float(
            challenger.named_steps["pca"].explained_variance_ratio_.sum()
        ),
    }


def p3_registrar_challenger(champion, challenger, perfil, resumen, version_champion: int) -> int:
    """Registra el challenger como versión nueva del modelo de P3 con el alias `challenger`.

    Deja un run con el modelo, sus parámetros y métricas, y copia en las etiquetas de la versión el
    linaje (commit, datos, run de origen). No mueve el alias `champion`. Regresa el número de versión.
    """
    nombre = cfg.P3_MODELO_REGISTRADO
    mlflow.set_experiment(cfg.P3_EXPERIMENTO)
    x = perfil[P3_COLUMNAS_PERFIL].astype("float64")
    commit = monitoreo.commit_actual()
    version_datos = dataset.version_datos(cfg.RUTA_MANIFIESTO)
    descripcion = (
        f"Challenger de P3: los hiperparámetros del champion v{version_champion} ajustados con los "
        f"perfiles de {perfil.shape[0]} estaciones de {cfg.MES_INICIO} a {cfg.MES_FIN} (train, valid "
        "y test juntos). Solo cambian los datos; se compara con el champion en el A/B sobre "
        f"{cfg.MES_MONITOREO}."
    )
    with mlflow.start_run(
        run_name="p3-challenger-datos-2026-09",
        tags={
            "equipo": "07",
            "question": "p3",
            "tarea": "clustering",
            "commit": commit,
            "fase": cfg.FASE_MONITOREO,
            "stage_of_work": "monitoring",
            "rol": "challenger",
            "champion_version": str(version_champion),
            "mlflow.note.content": descripcion,
        },
    ) as run:
        mlflow.log_params(
            {
                **p3_hiperparametros(challenger),
                "data_version": version_datos,
                "rango_datos": f"{cfg.MES_INICIO} a {cfg.MES_FIN}",
                "periodo_train": f"{cfg.MES_INICIO}-01 a 2026-09-29",
                "min_salidas_train": cfg.P3_MIN_SALIDAS_TRAIN,
                "n_variables": len(P3_COLUMNAS_PERFIL),
                "champion_version": version_champion,
            }
        )
        mlflow.log_metrics(
            {
                "train_silueta": resumen["silueta_challenger"],
                "train_silueta_champion": resumen["silueta_champion"],
                "ari_champion_challenger": resumen["ari_champion_challenger"],
                "train_tam_grupo_min": resumen["tam_grupo_min_challenger"],
                "estaciones": resumen["estaciones"],
            }
        )
        info = mlflow.sklearn.log_model(
            challenger,
            name="modelo",
            signature=infer_signature(x, challenger.predict(x)),
            input_example=x.head(5),
        )
    cliente = mlflow.MlflowClient()
    version = mlflow.register_model(f"models:/{info.model_id}", nombre)
    cliente.update_model_version(nombre, version.version, description=descripcion)
    for etiqueta, valor in {
        "question": "p3",
        "fase": cfg.FASE_MONITOREO,
        "rol": "challenger",
        "commit": commit,
        "data_version": version_datos,
        "rango_datos": f"{cfg.MES_INICIO} a {cfg.MES_FIN}",
        "source_run_id": run.info.run_id,
        "champion_version": str(version_champion),
        "train_silueta": f"{resumen['silueta_challenger']:.4f}",
    }.items():
        cliente.set_model_version_tag(nombre, version.version, etiqueta, valor)
    cliente.set_registered_model_alias(nombre, "challenger", version.version)
    logger.success(f"{nombre} v{version.version} (alias challenger), run {run.info.run_id}")
    return int(version.version)


@app.command()
def challenger(
    registrar: bool = typer.Option(
        False, "--registrar", help="Registrarlo en el Registry con el alias challenger"
    ),
):
    """Entrena el challenger con datos hasta MES_FIN y lo compara con el champion.

    Los mismos hiperparámetros del champion; cambian solo los datos (train, valid y test juntos).
    Sin `--registrar` no escribe nada en MLflow.
    """
    perfil = p3_conjuntos()["entrenamiento_challenger"]
    champion, version, _, _ = p3_cargar_champion()
    modelo = p3_reentrenar(champion, perfil)
    resumen = p3_resumen_challenger(champion, modelo, perfil)
    logger.info(f"Champion v{version} contra challenger:\n{pd.Series(resumen).to_string()}")
    if not registrar:
        return
    nueva = p3_registrar_challenger(champion, modelo, perfil, resumen, version)
    cliente = mlflow.MlflowClient()
    alias_champion = cliente.get_model_version_by_alias(cfg.P3_MODELO_REGISTRADO, "champion")
    if int(alias_champion.version) != version:
        raise RuntimeError(
            f"El alias champion cambió: era v{version}, ahora v{alias_champion.version}"
        )
    cargado = mlflow.pyfunc.load_model(f"models:/{cfg.P3_MODELO_REGISTRADO}@challenger")
    grupos = cargado.predict(cargado.input_example)
    logger.success(
        f"challenger = v{nueva} carga y asigna grupo a {len(grupos)} estaciones; champion sigue en v{version}"
    )


# ---------------------------------------------------------------------------------------------
# A/B testing: champion (A) contra challenger (B) sobre las estaciones del mes nuevo
# ---------------------------------------------------------------------------------------------


def p3_bootstrap_silueta(
    z: np.ndarray,
    grupos_a: np.ndarray,
    grupos_b: np.ndarray,
    n_bootstrap: int = cfg.MONITOREO_BOOTSTRAP,
    alfa: float = cfg.MONITOREO_ALFA,
    semilla: int = cfg.SEMILLA,
) -> tuple[float, float]:
    """Intervalo bootstrap de la diferencia de silueta media (B - A), remuestreando estaciones.

    La silueta de una estación depende de los grupos de todas las demás, así que las siluetas por
    estación no son independientes. Por eso cada remuestreo vuelve a calcular la silueta media de
    cada versión, en vez de remuestrear la diferencia ya calculada.
    """
    rng = np.random.default_rng(semilla)
    diferencias = []
    while len(diferencias) < n_bootstrap:
        indices = rng.integers(0, len(z), len(z))
        # La silueta necesita al menos dos grupos en cada versión
        if len(np.unique(grupos_a[indices])) < 2 or len(np.unique(grupos_b[indices])) < 2:
            continue
        diferencias.append(
            silhouette_score(z[indices], grupos_b[indices])
            - silhouette_score(z[indices], grupos_a[indices])
        )
    return tuple(np.quantile(diferencias, [alfa / 2, 1 - alfa / 2]))


def p3_decidir_ab(p_valor: float, ic_inferior: float, ic_superior: float, alfa: float):
    """Regla fijada antes de ver los resultados; regresa (versión vigente, motivo).

    El challenger reemplaza al champion solo si la mejora es significativa (Wilcoxon con p < alfa)
    y el intervalo bootstrap de la diferencia de silueta queda por completo sobre cero. Sin
    evidencia, el champion se queda: cambiar de versión tiene un costo y no se paga sin pruebas.
    """
    if p_valor < alfa and ic_inferior > 0:
        return "challenger", "mejora significativa"
    if p_valor < alfa and ic_superior < 0:
        return "champion", "el challenger es significativamente peor"
    return "champion", "sin diferencia significativa: el champion se queda"


def p3_ab(champion, challenger, perfil: pd.DataFrame, alfa: float = cfg.MONITOREO_ALFA):
    """A/B de las dos versiones sobre las mismas estaciones; regresa (resultados, por estación).

    La métrica es la silueta de cada estación, medida en el espacio estandarizado del champion
    (`p3_espacio_comun`): en el espacio propio de cada modelo (tras su PCA) no serían comparables.
    Las bases comparan una pérdida (menor es mejor), así que se pasa la silueta con signo negativo.
    """
    x = perfil[P3_COLUMNAS_PERFIL]
    z = p3_espacio_comun(champion, perfil)
    grupos_a, grupos_b = champion.predict(x), challenger.predict(x)
    silueta_a, silueta_b = silhouette_samples(z, grupos_a), silhouette_samples(z, grupos_b)

    pareado = (
        monitoreo.comparar_pareado(-silueta_a, -silueta_b, alfa=alfa)
        if not np.allclose(silueta_a, silueta_b)
        else None
    )
    if pareado is None:
        # Dos versiones idénticas: no hay nada que contrastar
        p_valor, p_valor_t = 1.0, 1.0
    else:
        p_valor, p_valor_t = pareado["p_valor"], pareado["p_valor_t"]
    ic_inferior, ic_superior = p3_bootstrap_silueta(z, grupos_a, grupos_b, alfa=alfa)
    aleatorio = monitoreo.comparar_aleatorio(-silueta_a, -silueta_b, alfa=alfa)
    vigente, motivo = p3_decidir_ab(p_valor, ic_inferior, ic_superior, alfa)

    media_a, media_b = silueta_a.mean(), silueta_b.mean()
    resultados = {
        "estaciones": len(perfil),
        "silueta_champion": float(media_a),
        "silueta_challenger": float(media_b),
        "diferencia": float(media_b - media_a),
        "cambio_relativo": float((media_b - media_a) / abs(media_a)),
        "ic_inferior": float(ic_inferior),
        "ic_superior": float(ic_superior),
        "p_wilcoxon": float(p_valor),
        "p_t_pareada": float(p_valor_t),
        "estaciones_challenger_mejor": int((silueta_b > silueta_a).sum()),
        "estaciones_champion_mejor": int((silueta_a > silueta_b).sum()),
        "aleatorio_n_champion": aleatorio["n_a"],
        "aleatorio_n_challenger": aleatorio["n_b"],
        "aleatorio_silueta_champion": float(-aleatorio["media_a"]),
        "aleatorio_silueta_challenger": float(-aleatorio["media_b"]),
        "aleatorio_p_mann_whitney": float(aleatorio["p_valor"]),
        "aleatorio_p_welch": float(aleatorio["p_valor_t"]),
        "ari_champion_challenger": float(adjusted_rand_score(grupos_a, grupos_b)),
        "estaciones_que_cambian_de_grupo": int(
            min((grupos_a != grupos_b).sum(), (grupos_a == grupos_b).sum())
        ),
        "alfa": alfa,
        "version_vigente": vigente,
        "motivo": motivo,
    }
    por_estacion = pd.DataFrame(
        {
            "grupo_champion": grupos_a,
            "grupo_challenger": grupos_b,
            "silueta_champion": silueta_a,
            "silueta_challenger": silueta_b,
            "diferencia": silueta_b - silueta_a,
        },
        index=perfil.index,
    )
    return resultados, por_estacion


@app.command()
def ab():
    """A/B del champion contra el challenger sobre las estaciones de octubre 2026.

    Prueba principal pareada (mismas estaciones) y, como complemento, un reparto aleatorio de las
    estaciones entre las dos versiones. Carga las versiones por alias; no escribe en MLflow.
    """
    evaluacion = p3_conjuntos()["evaluacion"]
    champion, version_a, _, _ = p3_cargar_champion()
    cliente = mlflow.MlflowClient()
    version_b = int(
        cliente.get_model_version_by_alias(cfg.P3_MODELO_REGISTRADO, "challenger").version
    )
    challenger = mlflow.sklearn.load_model(f"models:/{cfg.P3_MODELO_REGISTRADO}/{version_b}")
    resultados, por_estacion = p3_ab(champion, challenger, evaluacion)
    resultados = {"champion_version": version_a, "challenger_version": version_b, **resultados}
    pd.Series(resultados).to_frame("valor").rename_axis("metrica").to_csv(
        cfg.MONITOREO_DIR / "p3_ab_resultado.csv"
    )
    por_estacion.to_csv(cfg.MONITOREO_DIR / "p3_ab_estaciones.csv")
    logger.info(
        f"A/B de P3, champion v{version_a} contra challenger v{version_b}:\n{pd.Series(resultados).to_string()}"
    )


def p3_silueta_por_conjunto(champion, conjuntos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Silueta del champion, en su espacio estandarizado, en cada conjunto de perfiles."""
    filas = []
    for nombre, perfil in conjuntos.items():
        grupos = champion.predict(perfil[P3_COLUMNAS_PERFIL])
        filas.append(
            {
                "conjunto": nombre,
                "estaciones": len(perfil),
                "silueta_champion": float(
                    silhouette_score(p3_espacio_comun(champion, perfil), grupos)
                ),
                "salidas_mediana": float(perfil["salidas"].median()),
            }
        )
    return pd.DataFrame(filas)


@app.command()
def silueta():
    """Silueta del champion por periodo y por octubre de cada año (revisión posterior al A/B).

    Responde si la silueta baja de octubre 2026 (un mes de datos) frente a la de entrenamiento (años
    completos) es degradación o un efecto del tamaño del periodo: se compara octubre con octubres.
    """
    conjuntos = p3_conjuntos()
    champion, version, _, _ = p3_cargar_champion()
    orden = [
        "referencia",
        "valid",
        "actual",
        *sorted(n for n in conjuntos if n.startswith("mismo_mes_20")),
        "evaluacion",
    ]
    tabla = p3_silueta_por_conjunto(champion, {n: conjuntos[n] for n in orden})
    tabla.to_csv(cfg.MONITOREO_DIR / "p3_silueta_por_conjunto.csv", index=False)
    logger.info(f"Silueta del champion v{version} por conjunto:\n{tabla.to_string(index=False)}")


def p3_metricas_drift(conteo: pd.DataFrame, por_anio: pd.DataFrame | None) -> dict[str, float]:
    """Métricas de un run de drift: la proporción y el conteo de variables con drift por comparación."""
    metricas = {}
    for fila in conteo.itertuples():
        metricas[f"{fila.comparacion}_proporcion_drift"] = float(fila.proporcion_drift)
        metricas[f"{fila.comparacion}_columnas_con_drift"] = float(fila.columnas_con_drift)
    if por_anio is not None:
        for fila in por_anio.itertuples():
            metricas[f"octubre_{fila.anio}_proporcion_drift"] = float(fila.proporcion_drift)
    return metricas


def p3_datasets_monitoreo(conjuntos: dict[str, pd.DataFrame], contextos: dict[str, str]) -> dict:
    """Conjuntos de perfiles como datasets de MLflow; `contextos`: contexto -> nombre del conjunto."""
    return {
        contexto: monitoreo.dataset_mlflow(
            conjuntos[nombre].reset_index(),
            f"p3_perfiles_{nombre}",
            cfg.P3_RUTA_PERFILES_MONITOREO,
        )
        for contexto, nombre in contextos.items()
    }


@app.command()
def registrar_drift():
    """Registra en MLflow un run con los reportes de drift ya generados (no los vuelve a correr).

    Lee `p3_drift_*` de reports/monitoreo; si falta el conteo, falla y pide correr `drift` primero.
    Escribe en el experimento del equipo: un run nuevo por cada ejecución (no es idempotente).
    """
    carpeta = cfg.MONITOREO_DIR
    conteo_ruta = carpeta / "p3_drift_conteo.csv"
    if not conteo_ruta.exists():
        raise FileNotFoundError("Falta p3_drift_conteo.csv; ejecutar `make monitoreo-p3-drift`")
    conteo = pd.read_csv(conteo_ruta)
    anual_ruta = carpeta / "p3_drift_mismo_mes_por_anio.csv"
    por_anio = pd.read_csv(anual_ruta) if anual_ruta.exists() else None
    archivos = sorted(carpeta.glob("p3_drift_*"))
    _, version, _, id_modelo = p3_cargar_champion()
    descripcion = (
        f"Drift de P3 (segmentación de estaciones) con Evidently sobre {cfg.MES_MONITOREO}: train "
        "contra test, control estacional (mismo mes de la referencia) y contraste, cada uno con todas "
        f"las estaciones y solo las comunes; champion v{version}. Un porcentaje de variables con "
        "drift bajo el umbral de 50% no es alerta; el desglose por año separa la pandemia."
    )
    id_run = monitoreo.registrar_en_mlflow(
        f"p3-drift-{cfg.MES_MONITOREO}",
        "p3",
        archivos,
        metricas=p3_metricas_drift(conteo, por_anio),
        parametros={
            "champion_version": version,
            "umbral_proporcion": float(conteo["umbral_proporcion"].iloc[0]),
            "n_variables": len(P3_COLUMNAS_PERFIL) + 1,
        },
        etiquetas={"tarea": "clustering", "rol": "drift", "champion_version": str(version)},
        descripcion=descripcion,
        datasets=p3_datasets_monitoreo(
            p3_conjuntos(),
            {"referencia": "referencia", "actual": "actual", "evaluacion": "evaluacion"},
        ),
        modelo_id=id_modelo,
    )
    logger.success(f"Drift de P3 registrado en MLflow: run {id_run}")


@app.command()
def registrar_ab():
    """Registra en MLflow un run con el resultado del A/B ya calculado (no lo vuelve a correr).

    Lee `p3_ab_resultado.csv` y `p3_ab_estaciones.csv`; liga el champion como modelo evaluado y la
    silueta de octubre 2026 como métrica del champion sobre ese conjunto. Escribe en el
    experimento del equipo: un run nuevo por cada ejecución (no es idempotente).
    """
    carpeta = cfg.MONITOREO_DIR
    ruta = carpeta / "p3_ab_resultado.csv"
    if not ruta.exists():
        raise FileNotFoundError("Falta p3_ab_resultado.csv; ejecutar `make monitoreo-p3-ab`")
    resultado = pd.read_csv(ruta, index_col="metrica")["valor"]
    textos = {k: str(resultado[k]) for k in ("version_vigente", "motivo")}
    numericos = resultado.drop(list(textos)).astype(float)
    _, version, _, id_modelo = p3_cargar_champion()
    if int(numericos["champion_version"]) != version:
        raise ValueError(
            f"El A/B se calculó con el champion v{int(numericos['champion_version'])} y el alias "
            f"champion ya apunta a v{version}; volver a correr `make monitoreo-p3-ab`"
        )
    version_challenger = int(numericos["challenger_version"])
    descripcion = (
        f"A/B de P3, champion v{version} contra challenger v{version_challenger} sobre "
        f"{int(numericos['estaciones'])} estaciones de {cfg.MES_MONITOREO}. Silueta por estación en "
        "el espacio del champion; Wilcoxon pareado y bootstrap de la diferencia. Decisión: "
        f"{textos['motivo']}."
    )
    id_run = monitoreo.registrar_en_mlflow(
        f"p3-ab-{cfg.MES_MONITOREO}",
        "p3",
        [carpeta / "p3_ab_resultado.csv", carpeta / "p3_ab_estaciones.csv"],
        metricas=numericos.drop(["champion_version", "challenger_version", "alfa"]).to_dict(),
        parametros={
            "champion_version": version,
            "challenger_version": version_challenger,
            "alfa": float(numericos["alfa"]),
        },
        etiquetas={
            "tarea": "clustering",
            "rol": "ab",
            "champion_version": str(version),
            "decision": textos["version_vigente"],
        },
        descripcion=descripcion,
        datasets=p3_datasets_monitoreo(p3_conjuntos(), {"evaluacion": "evaluacion"}),
        modelo_id=id_modelo,
        metricas_por_dataset={
            "evaluacion": {"champion_silueta_evaluacion": float(numericos["silueta_champion"])}
        },
    )
    logger.success(f"A/B de P3 registrado en MLflow: run {id_run}")


if __name__ == "__main__":
    app()
