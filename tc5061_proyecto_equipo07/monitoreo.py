"""Monitoreo (Entregable 8): datos del mes nuevo, drift con Evidently y A/B testing de versiones.

Base común de P1, P2 y P3. Cada pregunta arma sus conjuntos y su versión alternativa en su propio
módulo (modeling/monitoreo_p1.py, etc.) y usa de aquí:
    asignar_periodo          referencia, valid, actual, evaluacion o brecha de cada fecha
    generar_reporte          reporte de Evidently (drift + calidad, y desempeño si hay target)
    resumen_drift            drift por variable en una tabla (columna, método, valor, drift)
    comparar_pareado         A/B pareado: las dos versiones sobre las mismas filas
    comparar_aleatorio       A/B con asignación aleatoria de unidades a cada versión
    dataset_mlflow           conjunto como dataset de MLflow (esquema, perfil y huella)
    registrar_en_mlflow      run de monitoreo con datasets, modelo evaluado, métricas y reportes

Uso: python -m tc5061_proyecto_equipo07.monitoreo datos
"""

from pathlib import Path
import subprocess  # solo para `git rev-parse` con argumentos fijos  # nosec B404

from evidently import (
    DataDefinition,
    Dataset,
    MulticlassClassification,
    Regression,
    Report,
)
from evidently.presets import (
    ClassificationPreset,
    DataDriftPreset,
    DataSummaryPreset,
    RegressionPreset,
)
from loguru import logger
import mlflow
import mlflow.data
from mlflow.entities import LoggedModelInput
import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals
from scipy import stats
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


def muestrear(
    tabla: pd.DataFrame, n: int = cfg.MONITOREO_MUESTRA, semilla: int = cfg.SEMILLA
) -> pd.DataFrame:
    """Muestra aleatoria de `n` filas (o la tabla completa si tiene menos)."""
    if len(tabla) <= n:
        return tabla.reset_index(drop=True)
    return tabla.sample(n=n, random_state=semilla).reset_index(drop=True)


def definicion_datos(
    numericas: list[str],
    categoricas: list[str],
    tarea: str | None = None,
    target: str | None = None,
    prediccion: str | None = None,
) -> DataDefinition:
    """Esquema de columnas para Evidently; con `tarea` agrega el target y la predicción.

    tarea: "regresion" (P1), "clasificacion" (P2, multiclase con etiquetas) o None (P3).
    """
    if tarea is None:
        return DataDefinition(numerical_columns=numericas, categorical_columns=categoricas)
    if target is None or prediccion is None:
        raise ValueError("Con `tarea` hay que indicar `target` y `prediccion`")
    if tarea == "regresion":
        return DataDefinition(
            numerical_columns=numericas + [target, prediccion],
            categorical_columns=categoricas,
            regression=[Regression(target=target, prediction=prediccion)],
        )
    if tarea == "clasificacion":
        return DataDefinition(
            numerical_columns=numericas,
            categorical_columns=categoricas + [target, prediccion],
            classification=[MulticlassClassification(target=target, prediction_labels=prediccion)],
        )
    raise ValueError(f"Tarea desconocida: {tarea}")


def generar_reporte(
    referencia: pd.DataFrame,
    actual: pd.DataFrame,
    definicion: DataDefinition,
    nombre: str,
    tarea: str | None = None,
    carpeta: Path = cfg.MONITOREO_DIR,
):
    """Corre el reporte de Evidently y lo guarda como HTML y JSON en `carpeta`.

    Incluye drift por variable y calidad de datos; con `tarea`, también el desempeño del modelo.
    Regresa el snapshot y las rutas de los dos archivos.
    """
    presets = [DataDriftPreset(), DataSummaryPreset()]
    if tarea == "regresion":
        presets.append(RegressionPreset())
    elif tarea == "clasificacion":
        presets.append(ClassificationPreset())
    reporte = Report(presets, include_tests=True)
    # Evidently recibe primero el conjunto actual y después el de referencia
    snapshot = reporte.run(
        Dataset.from_pandas(actual, data_definition=definicion),
        Dataset.from_pandas(referencia, data_definition=definicion),
    )
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_html, ruta_json = carpeta / f"{nombre}.html", carpeta / f"{nombre}.json"
    snapshot.save_html(str(ruta_html))
    ruta_json.write_text(snapshot.json(), encoding="utf-8")
    logger.success(f"Reporte de Evidently: {ruta_html.name} y {ruta_json.name}")
    return snapshot, [ruta_html, ruta_json]


def resumen_drift(snapshot) -> pd.DataFrame:
    """Drift por variable a partir del snapshot, ordenado de mayor a menor valor.

    Mismo criterio de Evidently: con p-value hay drift si el valor es menor o igual al umbral; con
    una distancia (Wasserstein, Jensen-Shannon), si es mayor o igual.
    """
    filas = []
    for metrica in snapshot.dict()["metrics"]:
        config = metrica["config"]
        if not config["type"].endswith(":ValueDrift"):
            continue
        es_p_value = "p_value" in config["method"]
        valor, umbral = float(metrica["value"]), float(config["threshold"])
        filas.append(
            {
                "columna": config["column"],
                "metodo": config["method"],
                "umbral": umbral,
                "valor": valor,
                "drift": valor <= umbral if es_p_value else valor >= umbral,
            }
        )
    resumen = pd.DataFrame(filas, columns=["columna", "metodo", "umbral", "valor", "drift"])
    return resumen.sort_values(["drift", "valor"], ascending=False, ignore_index=True)


def proporcion_drift(snapshot) -> dict:
    """Columnas con drift y su proporción, tal como las reporta Evidently (DriftedColumnsCount)."""
    for metrica in snapshot.dict()["metrics"]:
        if metrica["config"]["type"].endswith(":DriftedColumnsCount"):
            return {
                "columnas_con_drift": int(metrica["value"]["count"]),
                "proporcion_drift": float(metrica["value"]["share"]),
                "umbral_proporcion": float(metrica["config"]["drift_share"]),
            }
    raise ValueError("El snapshot no incluye DataDriftPreset")


def bootstrap_media(
    diferencias: np.ndarray,
    grupos: np.ndarray | None = None,
    n_bootstrap: int = cfg.MONITOREO_BOOTSTRAP,
    alfa: float = cfg.MONITOREO_ALFA,
    semilla: int = cfg.SEMILLA,
) -> tuple[float, float]:
    """Intervalo de confianza bootstrap (percentiles) de la media de `diferencias`.

    Con `grupos` se remuestrean grupos completos (por ejemplo, estación-día): las filas de un mismo
    grupo están correlacionadas y remuestrearlas por separado daría un intervalo demasiado angosto.
    """
    rng = np.random.default_rng(semilla)
    if grupos is None:
        sumas, conteos = diferencias, np.ones_like(diferencias)
    else:
        tabla = pd.DataFrame({"grupo": grupos, "dif": diferencias}).groupby("grupo")["dif"]
        sumas, conteos = tabla.sum().to_numpy(), tabla.count().to_numpy()
    medias = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.integers(0, len(sumas), len(sumas))
        medias[i] = sumas[indices].sum() / conteos[indices].sum()
    return tuple(np.quantile(medias, [alfa / 2, 1 - alfa / 2]))


def decidir(media_a: float, media_b: float, p_valor: float, alfa: float) -> str:
    """Versión ganadora con una pérdida (menor es mejor): A, B o empate si no es significativa."""
    if p_valor >= alfa:
        return "sin diferencia significativa"
    return "B" if media_b < media_a else "A"


def comparar_pareado(
    perdida_a,
    perdida_b,
    grupos=None,
    alfa: float = cfg.MONITOREO_ALFA,
    n_bootstrap: int = cfg.MONITOREO_BOOTSTRAP,
    semilla: int = cfg.SEMILLA,
) -> dict:
    """A/B pareado (champion/challenger): las dos versiones predicen las mismas filas.

    `perdida_a` y `perdida_b` son la pérdida por fila de cada versión (por ejemplo, error absoluto
    en P1). Se decide con Wilcoxon de rangos con signo, que no supone normalidad; la t pareada y el
    intervalo bootstrap de la diferencia (B - A) acompañan la decisión.
    """
    perdida_a, perdida_b = np.asarray(perdida_a, float), np.asarray(perdida_b, float)
    if perdida_a.shape != perdida_b.shape:
        raise ValueError("En el A/B pareado las dos versiones deben tener las mismas filas")
    diferencias = perdida_b - perdida_a
    media_a, media_b = perdida_a.mean(), perdida_b.mean()
    # zsplit: las filas sin diferencia (comunes con conteos) cuentan como empates, no se descartan
    p_wilcoxon = stats.wilcoxon(perdida_b, perdida_a, zero_method="zsplit").pvalue
    p_t = stats.ttest_rel(perdida_b, perdida_a).pvalue
    ic_inferior, ic_superior = bootstrap_media(
        diferencias,
        None if grupos is None else np.asarray(grupos),
        n_bootstrap,
        alfa,
        semilla,
    )
    return {
        "prueba": "pareada",
        "n_a": len(perdida_a),
        "n_b": len(perdida_b),
        "media_a": media_a,
        "media_b": media_b,
        "diferencia": media_b - media_a,
        "cambio_relativo": (media_b - media_a) / media_a,
        "ic_inferior": ic_inferior,
        "ic_superior": ic_superior,
        "p_valor": p_wilcoxon,
        "p_valor_t": p_t,
        "ganador": decidir(media_a, media_b, p_wilcoxon, alfa),
    }


def comparar_aleatorio(
    perdida_a,
    perdida_b,
    unidades=None,
    proporcion_b: float = 0.5,
    alfa: float = cfg.MONITOREO_ALFA,
    semilla: int = cfg.SEMILLA,
) -> dict:
    """A/B con asignación aleatoria: cada unidad recibe solo la predicción de una versión.

    Simula el reparto de tráfico de un A/B en producción sobre el mismo conjunto. La unidad de
    asignación (`unidades`, por ejemplo estación-día) evita que filas de una misma unidad queden en
    los dos grupos. Al no ser pareada, necesita más datos que `comparar_pareado` para detectar la
    misma diferencia (Treveil et al., 2020, cap. 7).
    """
    perdida_a, perdida_b = np.asarray(perdida_a, float), np.asarray(perdida_b, float)
    unidades = np.arange(len(perdida_a)) if unidades is None else np.asarray(unidades)
    rng = np.random.default_rng(semilla)
    distintas = np.unique(unidades)
    unidades_b = distintas[rng.random(len(distintas)) < proporcion_b]
    a_b = np.isin(unidades, unidades_b)
    grupo_a, grupo_b = perdida_a[~a_b], perdida_b[a_b]
    media_a, media_b = grupo_a.mean(), grupo_b.mean()
    p_mann_whitney = stats.mannwhitneyu(grupo_b, grupo_a).pvalue
    p_welch = stats.ttest_ind(grupo_b, grupo_a, equal_var=False).pvalue
    return {
        "prueba": "aleatoria",
        "n_a": len(grupo_a),
        "n_b": len(grupo_b),
        "media_a": media_a,
        "media_b": media_b,
        "diferencia": media_b - media_a,
        "cambio_relativo": (media_b - media_a) / media_a,
        "p_valor": p_mann_whitney,
        "p_valor_t": p_welch,
        "ganador": decidir(media_a, media_b, p_mann_whitney, alfa),
    }


def cargar_version(nombre: str, alias: str | None = None, version: int | None = None):
    """Carga una versión del Model Registry por alias o por número; regresa (modelo, versión)."""
    cliente = mlflow.MlflowClient()
    if (alias is None) == (version is None):
        raise ValueError("Indicar `alias` o `version`, no los dos")
    info = (
        cliente.get_model_version_by_alias(nombre, alias)
        if alias
        else cliente.get_model_version(nombre, str(version))
    )
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}/{info.version}")
    logger.info(f"{nombre}: versión {info.version} cargada")
    return modelo, int(info.version)


def commit_actual() -> str:
    """Commit corto con el que se ejecuta (`desconocido` si no hay repositorio Git)."""
    try:
        return subprocess.check_output(  # git con argumentos fijos, sin shell  # nosec B603 B607
            ["git", "rev-parse", "--short", "HEAD"], cwd=cfg.PROJ_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "desconocido"


def archivos_configuracion() -> list[Path]:
    """Archivos con los que corre el run: config.py y, si existen, MLproject y python_env.yaml.

    Los mismos que suben los runs de entrenamiento (p1_archivos_configuracion en train.py).
    """
    candidatos = [
        Path(cfg.__file__),
        cfg.PROJ_ROOT / "MLproject",
        cfg.PROJ_ROOT / "python_env.yaml",
    ]
    return [ruta for ruta in candidatos if ruta.exists()]


def dataset_mlflow(datos: pd.DataFrame, nombre: str, fuente: Path, target: str | None = None):
    """Conjunto como dataset de MLflow: guarda su esquema, perfil y huella (digest)."""
    return mlflow.data.from_pandas(datos, source=str(fuente), name=nombre, targets=target)


def registrar_en_mlflow(
    nombre_run: str,
    pregunta: str,
    archivos: list[Path],
    metricas: dict | None = None,
    parametros: dict | None = None,
    etiquetas: dict | None = None,
    descripcion: str | None = None,
    datasets: dict | None = None,
    modelo_id: str | None = None,
    metricas_por_dataset: dict[str, dict[str, float]] | None = None,
    metricas_por_modelo: dict[str, dict[str, dict[str, float]]] | None = None,
) -> str:
    """Registra un run de monitoreo (drift o A/B) en el experimento del equipo; regresa su ID.

    Con la misma calidad que los runs de entrenamiento: etiquetas de equipo y commit, descripción,
    versión de los datos, manifiestos, archivos de configuración y los conjuntos usados como
    datasets de MLflow (`datasets`: contexto -> dataset_mlflow). Un run de monitoreo no entrena un
    modelo: con `modelo_id` liga como entrada el modelo que evaluó, y `metricas_por_dataset`
    (contexto -> métricas) asocia esas métricas al modelo y al dataset donde se midieron, para que
    aparezcan en la página del modelo. Un A/B evalúa varios modelos: `metricas_por_modelo`
    (model_id -> contexto -> métricas) liga cada uno con sus métricas; las claves deben distinguir
    al modelo (por ejemplo, champion_mae y challenger_mae), porque el run guarda un valor por clave.
    """
    if cfg.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.EXPERIMENTO_MLFLOW)
    base = {
        "equipo": "07",
        "question": pregunta,
        "commit": commit_actual(),
        "fase": cfg.FASE_MONITOREO,
        "stage_of_work": "monitoring",
    }
    if descripcion:
        base["mlflow.note.content"] = descripcion
    datasets = datasets or {}
    with mlflow.start_run(run_name=nombre_run, tags={**base, **(etiquetas or {})}) as run:
        manifiestos = [
            m for m in (cfg.RUTA_MANIFIESTO, cfg.RUTA_MANIFIESTO_MONITOREO) if m.exists()
        ]
        versiones = {
            "data_version" if m == cfg.RUTA_MANIFIESTO else "data_version_monitoreo": (
                dataset.version_datos(m)
            )
            for m in manifiestos
        }
        mlflow.log_params({**versiones, "mes_monitoreo": cfg.MES_MONITOREO, **(parametros or {})})
        for contexto, conjunto in datasets.items():
            mlflow.log_input(conjunto, context=contexto)
        if metricas:
            mlflow.log_metrics(metricas)
        # Las métricas con model_id ligan el modelo evaluado como entrada del run (en MLflow open
        # source, mlflow.log_inputs(models=...) solo funciona en Databricks)
        modelos = {m: dict(c) for m, c in (metricas_por_modelo or {}).items()}
        # Sin modelo_id, las métricas por dataset se registran igual (como métricas del run)
        if modelo_id or metricas_por_dataset:
            modelos.setdefault(modelo_id, {}).update(metricas_por_dataset or {})
        for id_modelo, por_contexto in modelos.items():
            for contexto, valores in por_contexto.items():
                mlflow.log_metrics(valores, model_id=id_modelo, dataset=datasets.get(contexto))
            if not por_contexto:
                mlflow.MlflowClient().log_inputs(
                    run.info.run_id, datasets=[], models=[LoggedModelInput(id_modelo)]
                )
        for archivo in archivos:
            mlflow.log_artifact(str(archivo), artifact_path="monitoreo")
        for manifiesto in manifiestos:
            mlflow.log_artifact(str(manifiesto), artifact_path="data")
        for ruta in archivos_configuracion():
            mlflow.log_artifact(str(ruta), artifact_path="config")
    logger.success(f"Run de monitoreo {nombre_run}: {run.info.run_id}")
    return run.info.run_id


if __name__ == "__main__":
    app()
