"""Entrenamiento de P1 (demanda por estación-hora), P2 (riesgo de quedarse sin bicicletas o sin
anclajes) y P3 (clustering de estaciones), con MLflow.

P1: python -m tc5061_proyecto_equipo07.modeling.train p1-entrenar [--registrar]
Entrada: data/processed/p1_demanda_estacion_hora.parquet (`make features-p1`). Entrena el baseline,
HGB y XGBoost, deja los diagnósticos en reports/figures/p1_* y, con --registrar (o
P1_REGISTRAR_MLFLOW=true), registra un run por modelo en el experimento del proyecto.

P2: python -m tc5061_proyecto_equipo07.modeling.train p2-entrenar [--registrar]
Entrada: data/processed/p2_cambios_estacion_bin.parquet (`make features-p2`). Usa la partición común
del proyecto por día de servicio (validación solo para elegir umbrales; test solo se reporta);
agrega la historia "hasta ayer" (P2_PESO_HISTORICO) y entrena el baseline histórico,
HistGradientBoosting (P2_MIN_SAMPLES_LEAF) y XGBoost. Deja la matriz de confusión y la comparación
en reports/figures/p2_* y, con --registrar (o P2_REGISTRAR_MLFLOW=true), registra un run por modelo
en el experimento del proyecto. Cada parámetro tiene su opción de línea de comandos y su variable
de entorno equivalente (ver config.py).

P3 (cada parámetro tiene una variable de entorno equivalente, ver config.py):
    python -m tc5061_proyecto_equipo07.modeling.train p3-baseline
    python -m tc5061_proyecto_equipo07.modeling.train p3-alternativo

Entrada: data/processed/viajes_modelado_AAAA-MM_AAAA-MM.parquet (`dataset p3-particionar`).
Con P3_REGISTRAR_MLFLOW=true registra el run en MLflow (URL en MLFLOW_TRACKING_URI, nombre en
MLFLOW_RUN_NAME); sin ella solo entrena, evalúa y deja los artefactos en reports/figures/.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time
import warnings

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    log_loss,
    mean_absolute_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import typer
from xgboost import XGBClassifier, XGBRegressor

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset, plots
from tc5061_proyecto_equipo07.features import (
    P1NivelEstacion,
    p2_preparar_conjuntos,
    p3_construir_caracteristicas,
)
from tc5061_proyecto_equipo07.modeling.predict import (
    p1_predecir,
    p1_predecir_por_zona,
    p2_decidir_clase,
    p3_estabilidad_ari,
    p3_reasignar_grupos,
)

# numpy 2.2 con Accelerate (macOS) emite RuntimeWarning falsos en matmul con datos finitos;
# la finitud se verifica con un assert en p3_ajustar_baseline
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")

app = typer.Typer()

# Número de grupos que explora el modelo alternativo: no se fuerza k = 2
P3_K_ALTERNATIVO = range(3, 7)
HORAS_MANANA = [f"lab_h{hora:02d}" for hora in range(6, 10)]
HORAS_TARDE = [f"lab_h{hora:02d}" for hora in range(17, 21)]


# ---------------------------------------------------------------------------------------------
# Modelo base: StandardScaler + K-Means
# ---------------------------------------------------------------------------------------------


@dataclass
class P3Baseline:
    """Resultado del modelo base: modelo, métricas y las tablas que usan las comparaciones."""

    modelo: Pipeline
    k_elegido: int
    resultados_k: pd.DataFrame
    caracteristicas_train: pd.DataFrame  # incluye la columna `grupo`
    caracteristicas_valid: pd.DataFrame
    columnas_modelo: list[str]
    comunes: pd.Index
    x_train: pd.DataFrame
    z_train: np.ndarray
    tamanos: pd.Series
    metricas: dict
    semilla: int
    k_minimo: int
    k_maximo: int
    min_salidas_train: int
    min_salidas_valid: int


def p3_buscar_k(z_train: np.ndarray, k_minimo: int, k_maximo: int, semilla: int) -> pd.DataFrame:
    """Ajusta K-Means para cada `k` y regresa silueta, Davies-Bouldin, Calinski-Harabasz, inercia
    y tamaño del grupo más chico (índice `k`)."""
    filas = []
    for k in range(k_minimo, k_maximo + 1):
        modelo_k = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(z_train)
        filas.append(
            {
                "k": k,
                "silueta": silhouette_score(z_train, modelo_k.labels_),
                "davies_bouldin": davies_bouldin_score(z_train, modelo_k.labels_),
                "calinski_harabasz": calinski_harabasz_score(z_train, modelo_k.labels_),
                "inercia": modelo_k.inertia_,
                "tam_min": int(np.bincount(modelo_k.labels_).min()),
            }
        )
    return pd.DataFrame(filas).set_index("k")


def p3_ajustar_baseline(
    limpios: pd.DataFrame,
    semilla: int = cfg.P3_SEMILLA,
    k_minimo: int = cfg.P3_K_MINIMO,
    k_maximo: int = cfg.P3_K_MAXIMO,
    min_salidas_train: int = cfg.P3_MIN_SALIDAS_TRAIN,
    min_salidas_valid: int = cfg.P3_MIN_SALIDAS_VALID,
) -> P3Baseline:
    """Entrena el modelo base: variables por estación, `k` por silueta y estabilidad en valid.

    Usa solo los viajes de entrenamiento para ajustar; los de validación solo sirven para medir
    cuánto cambian los grupos al reasignar las estaciones con otro periodo. `limpios` es el
    dataset de modelado (con `particion` y `duracion_min`).
    """
    car_train, car_valid, columnas = p3_construir_caracteristicas(
        limpios, min_salidas_train, min_salidas_valid
    )
    x_train = car_train[columnas]
    z_train = StandardScaler().fit_transform(x_train)
    if not np.isfinite(z_train).all():
        raise ValueError("variables escaladas con valores no finitos")

    resultados_k = p3_buscar_k(z_train, k_minimo, k_maximo, semilla)
    k_elegido = int(resultados_k["silueta"].idxmax())

    # Modelo base: escalador + K-Means con el k elegido (un solo objeto para registrar y reutilizar)
    modelo = Pipeline(
        [
            ("escalador", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k_elegido, n_init=10, random_state=semilla)),
        ]
    ).fit(x_train)
    etiquetas = modelo.named_steps["kmeans"].labels_
    car_train = car_train.assign(grupo=etiquetas)
    tamanos = pd.Series(etiquetas).value_counts().sort_index()

    # Estabilidad: reasignación de las estaciones con las variables del año de validación
    comunes = car_train.index.intersection(car_valid.index)
    grupo_original = car_train.loc[comunes, "grupo"]
    grupo_en_valid = p3_reasignar_grupos(modelo, car_valid.loc[comunes], columnas)
    fila = resultados_k.loc[k_elegido]
    metricas = {
        "silueta": float(fila["silueta"]),
        "davies_bouldin": float(fila["davies_bouldin"]),
        "calinski_harabasz": float(fila["calinski_harabasz"]),
        "inercia": float(fila["inercia"]),
        "ari_valid": p3_estabilidad_ari(grupo_original, grupo_en_valid),
        "estaciones_train": len(x_train),
        "estaciones_estabilidad": len(comunes),
        "estaciones_cambian_de_grupo": int((grupo_original.to_numpy() != grupo_en_valid).sum()),
        "tam_grupo_min": int(tamanos.min()),
        "tam_grupo_max": int(tamanos.max()),
    }
    return P3Baseline(
        modelo=modelo,
        k_elegido=k_elegido,
        resultados_k=resultados_k,
        caracteristicas_train=car_train,
        caracteristicas_valid=car_valid,
        columnas_modelo=columnas,
        comunes=comunes,
        x_train=x_train,
        z_train=z_train,
        tamanos=tamanos,
        metricas=metricas,
        semilla=semilla,
        k_minimo=k_minimo,
        k_maximo=k_maximo,
        min_salidas_train=min_salidas_train,
        min_salidas_valid=min_salidas_valid,
    )


def p3_entradas_estabilidad(base: P3Baseline) -> tuple[pd.DataFrame, list[int]]:
    """Regresa las variables de valid de las estaciones comunes y su posición en train."""
    x_valid = base.caracteristicas_valid.loc[base.comunes, base.columnas_modelo]
    posiciones = [base.caracteristicas_train.index.get_loc(e) for e in base.comunes]
    return x_valid, posiciones


def p3_evaluar_modelo(modelo, z_train, x_valid_comunes, posiciones_comunes) -> dict:
    """Métricas internas en el espacio estandarizado del baseline (`z_train`) y estabilidad.

    El espacio común es lo que permite comparar modelos distintos: la silueta de un modelo
    medida en su propio espacio (por ejemplo, tras un PCA) no es comparable con la del baseline.
    """
    etiquetas = modelo.named_steps["kmeans"].labels_
    en_valid = modelo.predict(x_valid_comunes)
    return {
        "silueta": silhouette_score(z_train, etiquetas),
        "davies_bouldin": davies_bouldin_score(z_train, etiquetas),
        "calinski_harabasz": calinski_harabasz_score(z_train, etiquetas),
        "tam_min": int(np.bincount(etiquetas).min()),
        "ari_valid": p3_estabilidad_ari(etiquetas[posiciones_comunes], en_valid),
    }


def p3_baseline_por_k(base: P3Baseline) -> pd.DataFrame:
    """Baseline para cada `k`, evaluado en el mismo espacio y con la misma prueba de estabilidad."""
    x_valid, posiciones = p3_entradas_estabilidad(base)
    filas = []
    for k in range(base.k_minimo, base.k_maximo + 1):
        modelo_k = Pipeline(
            [
                ("escalador", StandardScaler()),
                ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=base.semilla)),
            ]
        ).fit(base.x_train)
        filas.append({"k": k, **p3_evaluar_modelo(modelo_k, base.z_train, x_valid, posiciones)})
    return pd.DataFrame(filas).set_index("k")


def p3_tabla_mapa(caracteristicas: pd.DataFrame, nomenclatura: pd.DataFrame) -> pd.DataFrame:
    """Une las variables por estación con nombre y ubicación (para el mapa y la asignación)."""
    return caracteristicas.join(nomenclatura.set_index("id")[["name", "latitude", "longitude"]])


def p3_resumir_grupos(caracteristicas: pd.DataFrame, nomenclatura: pd.DataFrame) -> pd.DataFrame:
    """Características medianas de cada grupo del baseline (volumen, balance, duración…)."""
    return (
        caracteristicas.join(nomenclatura.set_index("id")[["dpcapacity", "latitude", "longitude"]])
        .groupby("grupo")
        .agg(
            estaciones=("salidas", "size"),
            salidas_medianas=("salidas", "median"),
            viajes_por_dia=("log_viajes_por_dia", lambda s: np.expm1(s).median()),
            neto_sobre_salidas=("neto_sobre_salidas", "median"),
            prop_circulares=("prop_circulares", "median"),
            duracion_mediana_min=("duracion_mediana_min", "median"),
            capacidad_mediana=("dpcapacity", "median"),
        )
        .round(3)
    )


# ---------------------------------------------------------------------------------------------
# Modelo alternativo: pesos por bloque (opcional) + PCA + K-Means
# ---------------------------------------------------------------------------------------------


def p3_pesos_por_bloque(columnas: list[str]) -> np.ndarray:
    """Peso por columna para que cada bloque (laboral, fin de semana, indicadores) aporte la
    misma varianza total."""
    bloque = pd.Series(
        [
            "lab" if c.startswith("lab_") else "fin" if c.startswith("fin_") else "esc"
            for c in columnas
        ]
    )
    return (1 / np.sqrt(bloque.map(bloque.value_counts()))).to_numpy()


def p3_aplicar_pesos(matriz, pesos):
    """Multiplica cada columna por su peso. Vive a nivel de módulo (no en un notebook) para que
    el modelo registrado en MLflow pueda cargarse sin depender de `__main__`."""
    return matriz * pesos


def p3_construir_modelo(
    columnas_modelo: list[str], componentes: int, ponderar: bool, k: int, semilla: int
) -> Pipeline:
    """Escalador, pesos por bloque (opcional), PCA y K-Means en un solo objeto."""
    pasos = [("escalador", StandardScaler())]
    if ponderar:
        # Sin pesos no se agrega el paso: el modelo registrado no depende de funciones propias
        pasos.append(
            (
                "pesos",
                FunctionTransformer(
                    p3_aplicar_pesos, kw_args={"pesos": p3_pesos_por_bloque(columnas_modelo)}
                ),
            )
        )
    pasos += [
        ("pca", PCA(n_components=componentes, random_state=semilla)),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=semilla)),
    ]
    return Pipeline(pasos)


def p3_semillas_busqueda(semilla: int, cantidad: int) -> list[int]:
    """La semilla del proyecto primero y después 1, 2, 3… hasta completar `cantidad`."""
    otras = [s for s in range(1, cantidad + 100) if s != semilla]
    return ([semilla] + otras)[: max(cantidad, 1)]


def p3_evaluar_semillas(
    base: P3Baseline, ponderar: bool, n_comp: int, k: int, n_semillas: int
) -> pd.DataFrame:
    """Ajusta una configuración con cada semilla y regresa una fila de métricas por semilla."""
    x_valid, posiciones = p3_entradas_estabilidad(base)
    filas = []
    for semilla in p3_semillas_busqueda(base.semilla, n_semillas):
        modelo = p3_construir_modelo(base.columnas_modelo, n_comp, ponderar, k, semilla)
        modelo.fit(base.x_train)
        filas.append(
            {
                "ponderar": ponderar,
                "componentes": n_comp,
                "k": k,
                "semilla": semilla,
                **p3_evaluar_modelo(modelo, base.z_train, x_valid, posiciones),
            }
        )
    return pd.DataFrame(filas)


def p3_promediar_semillas(busqueda: pd.DataFrame) -> pd.DataFrame:
    """Una fila por configuración: promedio entre semillas (el peor caso en el tamaño mínimo)."""
    return (
        busqueda.groupby(["ponderar", "componentes", "k"])
        .agg(
            silueta=("silueta", "mean"),
            silueta_min=("silueta", "min"),
            davies_bouldin=("davies_bouldin", "mean"),
            calinski_harabasz=("calinski_harabasz", "mean"),
            ari_valid=("ari_valid", "mean"),
            ari_min=("ari_valid", "min"),
            tam_min=("tam_min", "min"),
        )
        .reset_index()
    )


def p3_buscar_alternativo(
    base: P3Baseline,
    componentes: list[int] = cfg.P3_COMPONENTES,
    ks=P3_K_ALTERNATIVO,
    n_semillas: int = cfg.P3_SEMILLAS_BUSQUEDA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prueba con y sin pesos por bloque, cada número de componentes y de grupos, con varias
    semillas. Regresa (todas las corridas, una fila por configuración con el promedio)."""
    busqueda = pd.concat(
        [
            p3_evaluar_semillas(base, ponderar, n_comp, k, n_semillas)
            for ponderar in (False, True)
            for n_comp in componentes
            for k in ks
        ],
        ignore_index=True,
    )
    return busqueda, p3_promediar_semillas(busqueda)


def p3_seleccionar_alternativo(
    resumen: pd.DataFrame,
    ari_baseline: float,
    tam_min_aceptable: int = cfg.P3_TAM_MIN_ACEPTABLE,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Elige la configuración de mayor silueta entre las que cumplen estabilidad y tamaño mínimo.

    Regresa (elegida, las que cumplen, el conjunto del que se eligió). Si ninguna cumple se elige
    entre todas, y los criterios de aceptación lo dejarán como no aceptado.
    """
    cumplen = resumen[
        (resumen["tam_min"] >= tam_min_aceptable) & (resumen["ari_valid"] >= ari_baseline)
    ]
    reserva = cumplen if len(cumplen) else resumen
    elegido = reserva.sort_values(["silueta", "componentes"], ascending=[False, True]).iloc[0]
    return elegido, cumplen, reserva


def p3_evaluar_criterios(
    elegido: pd.Series, metricas_baseline: dict, tam_min_aceptable: int = cfg.P3_TAM_MIN_ACEPTABLE
) -> tuple[dict[str, bool], bool]:
    """Criterios de aceptación acordados antes de ver los resultados.

    Regresa (criterio -> cumple, aceptado). Falta el cuarto criterio (que cada grupo se pueda
    nombrar), que es cualitativo y se evalúa en la interpretación.
    """
    criterios = {
        f"Silueta > baseline ({metricas_baseline['silueta']:.3f})": bool(
            elegido["silueta"] > metricas_baseline["silueta"]
        ),
        f"ARI de estabilidad >= baseline ({metricas_baseline['ari_valid']:.3f})": bool(
            elegido["ari_valid"] >= metricas_baseline["ari_valid"]
        ),
        f"Tamaño mínimo de grupo >= {tam_min_aceptable}": bool(
            elegido["tam_min"] >= tam_min_aceptable
        ),
    }
    return criterios, all(criterios.values())


@dataclass
class P3Alternativo:
    """Resultado del modelo alternativo y su comparación contra el baseline."""

    base: P3Baseline
    busqueda: pd.DataFrame
    resumen_busqueda: pd.DataFrame
    cumplen: pd.DataFrame
    reserva: pd.DataFrame
    elegido: pd.Series
    modelo: Pipeline
    etiquetas: np.ndarray
    metricas_alt: dict
    varianza_pca: float
    k_alt: int
    baseline_por_k: pd.DataFrame
    comparacion: pd.DataFrame
    criterios: dict
    aceptado: bool
    n_semillas: int


def p3_entrenar_alternativo(
    base: P3Baseline,
    componentes: list[int] = cfg.P3_COMPONENTES,
    n_semillas: int = cfg.P3_SEMILLAS_BUSQUEDA,
    tam_min_aceptable: int = cfg.P3_TAM_MIN_ACEPTABLE,
    busqueda_previa: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> P3Alternativo:
    """Busca, elige, ajusta y compara el modelo alternativo contra el baseline.

    Con `busqueda_previa` (las mismas dos tablas que regresa `p3_buscar_alternativo`, por ejemplo
    las de la búsqueda con Optuna) no corre la rejilla: solo elige entre esas configuraciones.
    """
    baseline_por_k = p3_baseline_por_k(base)
    busqueda, resumen = busqueda_previa or p3_buscar_alternativo(
        base, componentes, n_semillas=n_semillas
    )
    elegido, cumplen, reserva = p3_seleccionar_alternativo(
        resumen, base.metricas["ari_valid"], tam_min_aceptable
    )
    logger.info(
        f"Configuraciones evaluadas: {len(resumen)} | que cumplen ARI y tamaño mínimo: {len(cumplen)}"
    )
    logger.info(
        "Configuración elegida: "
        f"{{'ponderar': {bool(elegido['ponderar'])}, 'componentes': {int(elegido['componentes'])}, "
        f"'k': {int(elegido['k'])}}}"
    )

    k_alt = int(elegido["k"])
    modelo = p3_construir_modelo(
        base.columnas_modelo,
        int(elegido["componentes"]),
        bool(elegido["ponderar"]),
        k_alt,
        base.semilla,
    ).fit(base.x_train)
    x_valid, posiciones = p3_entradas_estabilidad(base)
    metricas_alt = p3_evaluar_modelo(modelo, base.z_train, x_valid, posiciones)
    varianza_pca = float(modelo.named_steps["pca"].explained_variance_ratio_.sum())

    comparacion = pd.DataFrame(
        {
            f"baseline (k={base.k_elegido})": baseline_por_k.loc[base.k_elegido],
            f"baseline (k={k_alt})": baseline_por_k.loc[k_alt],
            f"alternativo (k={k_alt}, promedio de {n_semillas} semillas)": pd.Series(
                {
                    "silueta": elegido["silueta"],
                    "davies_bouldin": elegido["davies_bouldin"],
                    "calinski_harabasz": elegido["calinski_harabasz"],
                    "ari_valid": elegido["ari_valid"],
                    "tam_min": elegido["tam_min"],
                }
            ),
            f"alternativo (k={k_alt}, semilla {base.semilla})": pd.Series(metricas_alt),
        }
    ).T[["silueta", "davies_bouldin", "calinski_harabasz", "ari_valid", "tam_min"]]

    criterios, aceptado = p3_evaluar_criterios(elegido, base.metricas, tam_min_aceptable)
    for nombre, cumple in criterios.items():
        logger.info(("CUMPLE    " if cumple else "NO CUMPLE ") + nombre)
    logger.info(
        "Veredicto cuantitativo: "
        + ("ACEPTADO" if aceptado else "NO ACEPTADO: el baseline sigue siendo el piso")
    )
    logger.info(
        f"Varianza explicada por los {int(elegido['componentes'])} componentes: {varianza_pca:.1%}"
    )
    logger.info(
        f"Silueta entre semillas: min {elegido['silueta_min']:.3f}, promedio {elegido['silueta']:.3f}"
    )
    logger.info(
        f"ARI entre semillas: min {elegido['ari_min']:.3f}, promedio {elegido['ari_valid']:.3f}"
    )

    return P3Alternativo(
        base=base,
        busqueda=busqueda,
        resumen_busqueda=resumen,
        cumplen=cumplen,
        reserva=reserva,
        elegido=elegido,
        modelo=modelo,
        etiquetas=modelo.named_steps["kmeans"].labels_,
        metricas_alt=metricas_alt,
        varianza_pca=varianza_pca,
        k_alt=k_alt,
        baseline_por_k=baseline_por_k,
        comparacion=comparacion,
        criterios=criterios,
        aceptado=aceptado,
        n_semillas=n_semillas,
    )


def p3_asignacion_alternativo(alt: P3Alternativo, nomenclatura: pd.DataFrame) -> pd.DataFrame:
    """Variables por estación con el grupo del modelo alternativo, nombre, ubicación y el % de
    salidas laborales de mañana (6 a 9 h) y de tarde (17 a 20 h)."""
    asignacion = (
        alt.base.caracteristicas_train.drop(columns="grupo")
        .assign(grupo_alt=alt.etiquetas)
        .join(nomenclatura.set_index("id")[["name", "dpcapacity", "latitude", "longitude"]])
    )
    asignacion["pct_manana"] = asignacion[HORAS_MANANA].sum(axis=1) * 100
    asignacion["pct_tarde"] = asignacion[HORAS_TARDE].sum(axis=1) * 100
    return asignacion


def p3_resumir_alternativo(asignacion: pd.DataFrame) -> pd.DataFrame:
    """Características medianas de cada grupo del modelo alternativo."""
    return (
        asignacion.groupby("grupo_alt")
        .agg(
            estaciones=("salidas", "size"),
            pct_salidas_6_a_9h=("pct_manana", "median"),
            pct_salidas_17_a_20h=("pct_tarde", "median"),
            viajes_por_dia=("log_viajes_por_dia", lambda s: np.expm1(s).median()),
            neto_sobre_salidas=("neto_sobre_salidas", "median"),
            prop_circulares=("prop_circulares", "median"),
            duracion_mediana_min=("duracion_mediana_min", "median"),
            capacidad_mediana=("dpcapacity", "median"),
        )
        .round(3)
    )


# ---------------------------------------------------------------------------------------------
# Registro en MLflow y utilidades para CI
# ---------------------------------------------------------------------------------------------


def p3_nombre_run(base: str) -> str:
    """Nombre del run: `<MLFLOW_RUN_NAME>-<base>` si la variable existe; si no, solo `base`."""
    return f"{cfg.MLFLOW_RUN_NAME}-{base}" if cfg.MLFLOW_RUN_NAME else base


def p3_commit_actual() -> str:
    """Commit corto: GITHUB_SHA en CI; si no, `git rev-parse`; si no, `desconocido`."""
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:7]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cfg.PROJ_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "desconocido"


def p3_etiquetas_ci() -> dict[str, str]:
    """Etiquetas de GitHub Actions presentes en el entorno (vacío fuera de CI)."""
    variables = {
        "github_run_id": "GITHUB_RUN_ID",
        "github_actor": "GITHUB_ACTOR",
        "github_ref_name": "GITHUB_REF_NAME",
        "github_sha": "GITHUB_SHA",
    }
    return {etiqueta: os.environ[v] for etiqueta, v in variables.items() if os.environ.get(v)}


def p3_escribir_resumen_ci(titulo: str, valores: dict) -> None:
    """Agrega una tabla de métricas al resumen del job (GITHUB_STEP_SUMMARY), si existe."""
    ruta = os.environ.get("GITHUB_STEP_SUMMARY")
    if not ruta:
        return
    filas = [f"### {titulo}", "", "| métrica | valor |", "|---|---|"]
    for nombre, valor in valores.items():
        texto = f"{valor:.4f}" if isinstance(valor, float) else str(valor)
        filas.append(f"| {nombre} | {texto} |")
    with open(ruta, "a", encoding="utf-8") as archivo:
        archivo.write("\n".join(filas) + "\n\n")


def p3_preparar_experimento(experimento: str = cfg.P3_EXPERIMENTO, uri: str | None = None) -> str:
    """Apunta MLflow al servidor, deja listo el experimento y regresa la URL base de sus runs."""
    import mlflow

    uri = uri or cfg.MLFLOW_TRACKING_URI
    if not uri:
        raise ValueError("MLFLOW_TRACKING_URI no está definida (variable de entorno o .env)")
    mlflow.set_tracking_uri(uri)

    # El experimento debe guardar los artefactos por el proxy del servidor (mlflow-artifacts:/);
    # si apunta a una ruta local del servidor, el cliente no puede escribir en ella.
    existente = mlflow.get_experiment_by_name(experimento)
    if existente is None:
        mlflow.create_experiment(experimento, artifact_location="mlflow-artifacts:/" + experimento)
        existente = mlflow.get_experiment_by_name(experimento)
    if not existente.artifact_location.startswith("mlflow-artifacts:"):
        raise RuntimeError(
            "El experimento no usa el proxy de artefactos: " + existente.artifact_location
        )
    mlflow.set_experiment(experimento)
    return f"{uri.rstrip('/')}/#/experiments/{existente.experiment_id}/runs/"


def p3_etiquetas_run(
    clave: str, origen: str, descripcion: bool = True, **extra: str
) -> dict[str, str]:
    """Etiquetas de la rúbrica de un run de P3 (`clave` es una entrada de cfg.P3_MODELOS)."""
    _, etapa, familia = cfg.P3_MODELOS[clave]
    etiquetas = {
        "equipo": "07",
        "question": "p3",
        "tarea": "clustering",
        "commit": p3_commit_actual(),
        "autor": "lasd",
        "origen": origen,
        "stage_of_work": etapa,
        "model_family": familia,
        "fase": cfg.FASE_ENTREGA,
        **extra,
    }
    if descripcion:
        etiquetas["mlflow.note.content"] = cfg.P3_DESCRIPCIONES[clave]
    return etiquetas


# Conjunto en que se calcula cada métrica de P3, como `valid_mae` y `test_mae` en P1. El modelo se
# ajusta y su estructura (silueta, tamaños, inercia…) se mide con train; valid solo mide la
# estabilidad, al reasignar las estaciones con las variables del año de validación.
P3_NOMBRES_METRICAS = {
    "silueta": "train_silueta",
    "silueta_min": "train_silueta_min",
    "silueta_min_semillas": "train_silueta_min_semillas",
    "silueta_semilla_proyecto": "train_silueta_semilla_proyecto",
    "davies_bouldin": "train_davies_bouldin",
    "calinski_harabasz": "train_calinski_harabasz",
    "inercia": "train_inercia",
    "varianza_pca": "train_varianza_pca",
    "tam_min": "train_tam_grupo_min",
    "tam_grupo_min": "train_tam_grupo_min",
    "tam_grupo_max": "train_tam_grupo_max",
    "estaciones_train": "train_estaciones",
    "delta_silueta_vs_baseline": "train_delta_silueta_vs_baseline",
    "ari_valid": "valid_ari",
    "ari_min": "valid_ari_min",
    "estaciones_estabilidad": "valid_estaciones_estabilidad",
    "estaciones_cambian_de_grupo": "valid_estaciones_cambian_de_grupo",
    "delta_ari_vs_baseline": "valid_delta_ari_vs_baseline",
}


def p3_nombrar_metricas(metricas: dict) -> dict:
    """Pone a cada métrica de P3 el prefijo de su conjunto (`train_` o `valid_`) para MLflow.

    Falla con una métrica que no está en `P3_NOMBRES_METRICAS`: así ninguna se registra sin
    decir en qué conjunto se calculó.
    """
    desconocidas = sorted(set(metricas) - set(P3_NOMBRES_METRICAS))
    if desconocidas:
        raise KeyError(f"Métricas de P3 sin conjunto asignado: {', '.join(desconocidas)}")
    return {P3_NOMBRES_METRICAS[nombre]: valor for nombre, valor in metricas.items()}


def p3_registrar_en_mlflow(
    nombre_run: str,
    etiquetas: dict,
    parametros: dict,
    metricas: dict,
    series: dict[str, pd.Series] | None = None,
    artefactos: list[Path] = (),
    modelo=None,
    ejemplo=None,
    entrada: pd.DataFrame | None = None,
    skops_trusted_types: list[str] | None = None,
    de_nuevo: bool = False,
    experimento: str = cfg.P3_EXPERIMENTO,
    uri: str | None = None,
) -> str:
    """Registra un run en MLflow y regresa su URL. Es idempotente.

    Si ya existe un run terminado con ese nombre en el experimento, no crea otro (salvo
    `de_nuevo`), para no duplicar corridas al re-ejecutar. Un run que quedó a medias no cuenta.
    `entrada` es la tabla con la que se ajustó el modelo (se registra con `log_input`); las
    figuras van a `plots/`, las tablas a `tables/`, y el manifiesto y la configuración a `data/`
    y `config/`.
    """
    import mlflow
    import mlflow.data
    from mlflow.models import infer_signature
    import mlflow.sklearn

    url_base = p3_preparar_experimento(experimento, uri)
    existente = mlflow.get_experiment_by_name(experimento)

    corridas = mlflow.search_runs(
        experiment_ids=[existente.experiment_id],
        filter_string=f"tags.mlflow.runName = '{nombre_run}' and attributes.status = 'FINISHED'",
    )
    if len(corridas) and not de_nuevo:
        url = url_base + str(corridas.iloc[0]["run_id"])
        logger.info(f"Ya existe un run '{nombre_run}' en el experimento '{experimento}': {url}")
        return url

    with mlflow.start_run(run_name=nombre_run) as corrida:
        mlflow.set_tags({**etiquetas, **p3_etiquetas_ci()})
        mlflow.log_params(parametros)
        for nombre, serie in (series or {}).items():
            for paso, valor in serie.items():
                mlflow.log_metric(nombre, float(valor), step=int(paso))
        mlflow.log_metrics(p3_nombrar_metricas(metricas))
        if entrada is not None:
            # La tabla ya está en memoria; MLflow guarda su esquema, perfil y huella (digest)
            mlflow.log_input(
                mlflow.data.from_pandas(
                    entrada, source=str(cfg.RUTA_VIAJES_MODELADO), name="p3_train"
                ),
                context="training",
            )
        for ruta in artefactos:
            mlflow.log_artifact(str(ruta), artifact_path=p1_carpeta_artefacto(ruta))
        mlflow.log_artifact(str(cfg.RUTA_MANIFIESTO), artifact_path="data")
        for ruta in p1_archivos_configuracion():
            mlflow.log_artifact(str(ruta), artifact_path="config")
        if modelo is not None:
            opciones = {"skops_trusted_types": skops_trusted_types} if skops_trusted_types else {}
            firma = (
                infer_signature(ejemplo, modelo.predict(ejemplo)) if ejemplo is not None else None
            )
            mlflow.sklearn.log_model(
                modelo, name=nombre_run, signature=firma, input_example=ejemplo, **opciones
            )
    url = url_base + corrida.info.run_id
    logger.success(f"Run registrado: {url}")
    return url


def p3_parametros_comunes(base: P3Baseline) -> dict:
    """Parámetros de datos y filtros que comparten los runs del baseline y del alternativo."""
    return {
        "semilla": base.semilla,
        "rango_datos": f"{cfg.MES_INICIO} a {cfg.MES_FIN}",
        "periodo_train": "2020-01-01 a 2024-09-29",
        "periodo_estabilidad": "2024-10-01 a 2025-09-29",
        "min_salidas_train": base.min_salidas_train,
        "min_salidas_valid": base.min_salidas_valid,
        "n_variables": len(base.columnas_modelo),
        "data_version": dataset.version_datos(),
    }


def p3_guardar_artefactos_baseline(
    base: P3Baseline, nomenclatura: pd.DataFrame, carpeta: Path = cfg.FIGURES_DIR
) -> list[Path]:
    """Guarda en `carpeta` la asignación por estación, el resumen, las variables y 2 gráficas."""
    carpeta.mkdir(parents=True, exist_ok=True)
    mapa = p3_tabla_mapa(base.caracteristicas_train, nomenclatura)
    asignacion = mapa[["name", "latitude", "longitude", "grupo", "salidas"]].rename(
        columns={"name": "estacion", "salidas": "salidas_train"}
    )
    rutas = [carpeta / "p3_asignacion_grupos.csv", carpeta / "p3_resumen_grupos.csv"]
    asignacion.to_csv(rutas[0], index_label="estacion_id")
    p3_resumir_grupos(base.caracteristicas_train, nomenclatura).to_csv(rutas[1])
    rutas.append(carpeta / "p3_variables.json")
    rutas[2].write_text(json.dumps(base.columnas_modelo, ensure_ascii=False, indent=1))
    fig_k = plots.p3_grafica_silueta_por_k(base.resultados_k, base.k_elegido)
    rutas.append(plots.p3_guardar_figura(fig_k, carpeta / "p3_silueta_por_k.png"))
    fig_m = plots.p3_grafica_mapa_grupos(
        mapa, base.k_elegido, "Estaciones por grupo (baseline)", 1.0
    )
    rutas.append(plots.p3_guardar_figura(fig_m, carpeta / "p3_mapa_grupos.png"))
    return rutas


def p3_registrar_baseline(
    base: P3Baseline,
    nomenclatura: pd.DataFrame,
    de_nuevo: bool = cfg.P3_REGISTRAR_DE_NUEVO,
    origen: str = "modulo",
) -> str:
    """Guarda los artefactos del baseline y registra el run `baseline` en MLflow."""
    artefactos = p3_guardar_artefactos_baseline(base, nomenclatura)
    parametros = {
        "modelo": "StandardScaler + KMeans",
        "k": base.k_elegido,
        "k_minimo": base.k_minimo,
        "k_maximo": base.k_maximo,
        "n_init": 10,
        "horas_excluidas": "1-4",
        **p3_parametros_comunes(base),
    }
    return p3_registrar_en_mlflow(
        nombre_run=p3_nombre_run(cfg.P3_MODELOS["baseline"][0]),
        etiquetas=p3_etiquetas_run("baseline", origen),
        parametros=parametros,
        metricas=base.metricas,
        series={"train_silueta_por_k": base.resultados_k["silueta"]},
        artefactos=artefactos,
        modelo=base.modelo,
        ejemplo=base.x_train.head(3),
        entrada=base.x_train,
        de_nuevo=de_nuevo,
    )


def p3_guardar_artefactos_alternativo(
    alt: P3Alternativo,
    nomenclatura: pd.DataFrame,
    carpeta: Path = cfg.FIGURES_DIR,
    prefijo: str = "p3_pca",
) -> list[Path]:
    """Guarda en `carpeta` la asignación, el resumen, la búsqueda, la comparación y la figura.

    `prefijo` evita que el candidato de Optuna (`p3_pca_optuna`) pise los de la rejilla.
    """
    base = alt.base
    asignacion = p3_asignacion_alternativo(alt, nomenclatura)
    carpeta.mkdir(parents=True, exist_ok=True)
    tablas = {
        f"{prefijo}_asignacion_grupos.csv": asignacion[
            ["name", "latitude", "longitude", "grupo_alt", "salidas"]
        ].rename(columns={"name": "estacion", "salidas": "salidas_train"}),
        f"{prefijo}_resumen_grupos.csv": p3_resumir_alternativo(asignacion),
        f"{prefijo}_busqueda.csv": alt.resumen_busqueda,
        f"{prefijo}_comparacion.csv": alt.comparacion,
    }
    artefactos = []
    for nombre, tabla in tablas.items():
        tabla.to_csv(carpeta / nombre, index_label="id")
        artefactos.append(carpeta / nombre)
    figura = plots.p3_grafica_perfiles_y_mapa(
        asignacion,
        alt.k_alt,
        [c for c in base.columnas_modelo if c.startswith("lab_")],
        [c for c in base.columnas_modelo if c.startswith("fin_")],
        cfg.P3_HORAS_SERVICIO,
    )
    artefactos.append(plots.p3_guardar_figura(figura, carpeta / f"{prefijo}_perfiles_mapa.png"))
    return artefactos


def p3_registrar_alternativo(
    alt: P3Alternativo,
    nomenclatura: pd.DataFrame,
    de_nuevo: bool = cfg.P3_REGISTRAR_DE_NUEVO,
    origen: str = "modulo",
    carpeta: Path = cfg.FIGURES_DIR,
    clave: str = "alternativo",
    etiquetas_extra: dict[str, str] | None = None,
) -> str:
    """Guarda los artefactos del modelo alternativo y registra su run en MLflow.

    `clave` (entrada de cfg.P3_MODELOS) distingue el de la rejilla (`pca_kmeans`) del de la
    búsqueda con Optuna (`pca_kmeans_optuna`). Se registra aunque no se acepte: un resultado
    negativo también es evidencia.
    """
    base = alt.base
    nombre_modelo = cfg.P3_MODELOS[clave][0]
    prefijo = "p3_pca_optuna" if clave.endswith("_optuna") else "p3_pca"
    artefactos = p3_guardar_artefactos_alternativo(alt, nomenclatura, carpeta, prefijo)

    elegido = alt.elegido
    ponderar = bool(elegido["ponderar"])
    parametros = {
        "modelo": "StandardScaler + pesos por bloque + PCA + KMeans"
        if ponderar
        else "StandardScaler + PCA + KMeans",
        "ponderar_bloques": ponderar,
        "componentes": int(elegido["componentes"]),
        "k": alt.k_alt,
        "n_init": 10,
        "semillas_busqueda": alt.n_semillas,
        **p3_parametros_comunes(base),
    }
    metricas = {
        "silueta": float(elegido["silueta"]),
        "silueta_min_semillas": float(elegido["silueta_min"]),
        "silueta_semilla_proyecto": float(alt.metricas_alt["silueta"]),
        "davies_bouldin": float(elegido["davies_bouldin"]),
        "calinski_harabasz": float(alt.metricas_alt["calinski_harabasz"]),
        "ari_valid": float(elegido["ari_valid"]),
        "tam_grupo_min": int(elegido["tam_min"]),
        "varianza_pca": alt.varianza_pca,
        "delta_silueta_vs_baseline": float(elegido["silueta"] - base.metricas["silueta"]),
        "delta_ari_vs_baseline": float(elegido["ari_valid"] - base.metricas["ari_valid"]),
    }
    etiquetas = p3_etiquetas_run(
        clave,
        origen,
        aceptado=str(alt.aceptado).lower(),
        referencia=p3_nombre_run(cfg.P3_MODELOS["baseline"][0]),
        **(etiquetas_extra or {}),
    )
    return p3_registrar_en_mlflow(
        nombre_run=p3_nombre_run(nombre_modelo),
        etiquetas=etiquetas,
        parametros=parametros,
        metricas=metricas,
        artefactos=artefactos,
        modelo=alt.modelo,
        ejemplo=base.x_train.head(3),
        entrada=base.x_train,
        skops_trusted_types=(
            ["tc5061_proyecto_equipo07.modeling.train.p3_aplicar_pesos"] if ponderar else None
        ),
        de_nuevo=de_nuevo,
    )


# ---------------------------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------------------------


def p3_cargar_modelado(mes_inicio: str, mes_fin: str) -> pd.DataFrame:
    """Lee data/processed/viajes_modelado_*.parquet (viajes limpios con partición)."""
    ruta = dataset.p3_ruta_viajes_modelado(mes_inicio, mes_fin)
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta.name}; ejecutar `make data` y `p3-particionar`")
    return pd.read_parquet(ruta)


@app.callback()
def p3_ayuda_train():
    """Entrenamiento de P3: modelo base y modelo alternativo."""


@app.command()
def p3_baseline(
    mes_inicio: str = cfg.MES_INICIO,
    mes_fin: str = cfg.MES_FIN,
    semilla: int = typer.Option(cfg.P3_SEMILLA, envvar="P3_SEMILLA"),
    k_minimo: int = typer.Option(cfg.P3_K_MINIMO, envvar="P3_K_MINIMO"),
    k_maximo: int = typer.Option(cfg.P3_K_MAXIMO, envvar="P3_K_MAXIMO"),
    min_salidas_train: int = typer.Option(cfg.P3_MIN_SALIDAS_TRAIN, envvar="P3_MIN_SALIDAS_TRAIN"),
    min_salidas_valid: int = typer.Option(cfg.P3_MIN_SALIDAS_VALID, envvar="P3_MIN_SALIDAS_VALID"),
    registrar: bool = typer.Option(cfg.P3_REGISTRAR_MLFLOW, envvar="P3_REGISTRAR_MLFLOW"),
    de_nuevo: bool = typer.Option(cfg.P3_REGISTRAR_DE_NUEVO, envvar="P3_REGISTRAR_DE_NUEVO"),
):
    """Entrena y evalúa el modelo base; con --registrar lo registra en MLflow."""
    limpios = p3_cargar_modelado(mes_inicio, mes_fin)
    base = p3_ajustar_baseline(
        limpios, semilla, k_minimo, k_maximo, min_salidas_train, min_salidas_valid
    )
    logger.info(f"k elegido por silueta: {base.k_elegido} | grupos: {base.tamanos.to_dict()}")
    logger.info(
        f"silueta {base.metricas['silueta']:.3f} | Davies-Bouldin {base.metricas['davies_bouldin']:.3f}"
        f" | ARI de estabilidad {base.metricas['ari_valid']:.3f}"
    )
    p3_escribir_resumen_ci(
        "P3 baseline", {"k": base.k_elegido, **p3_nombrar_metricas(base.metricas)}
    )
    nomenclatura = dataset.p3_cargar_estaciones()
    if registrar:
        p3_registrar_baseline(base, nomenclatura, de_nuevo, origen="cli")
    else:
        p3_guardar_artefactos_baseline(base, nomenclatura)
        logger.info("Registro en MLflow desactivado (P3_REGISTRAR_MLFLOW=false)")


@app.command()
def p3_alternativo(
    mes_inicio: str = cfg.MES_INICIO,
    mes_fin: str = cfg.MES_FIN,
    semilla: int = typer.Option(cfg.P3_SEMILLA, envvar="P3_SEMILLA"),
    k_minimo: int = typer.Option(cfg.P3_K_MINIMO, envvar="P3_K_MINIMO"),
    k_maximo: int = typer.Option(cfg.P3_K_MAXIMO, envvar="P3_K_MAXIMO"),
    min_salidas_train: int = typer.Option(cfg.P3_MIN_SALIDAS_TRAIN, envvar="P3_MIN_SALIDAS_TRAIN"),
    min_salidas_valid: int = typer.Option(cfg.P3_MIN_SALIDAS_VALID, envvar="P3_MIN_SALIDAS_VALID"),
    n_semillas: int = typer.Option(cfg.P3_SEMILLAS_BUSQUEDA, envvar="P3_SEMILLAS_BUSQUEDA"),
    registrar: bool = typer.Option(cfg.P3_REGISTRAR_MLFLOW, envvar="P3_REGISTRAR_MLFLOW"),
    de_nuevo: bool = typer.Option(cfg.P3_REGISTRAR_DE_NUEVO, envvar="P3_REGISTRAR_DE_NUEVO"),
    exigir_aceptacion: bool = typer.Option(
        cfg.P3_EXIGIR_ACEPTACION, envvar="P3_EXIGIR_ACEPTACION"
    ),
):
    """Busca, evalúa y compara el modelo alternativo contra el baseline.

    El baseline se recalcula con la misma semilla (es determinista), así que la comparación es
    contra exactamente el mismo resultado. Con --exigir-aceptacion sale con error si no se acepta.
    """
    limpios = p3_cargar_modelado(mes_inicio, mes_fin)
    base = p3_ajustar_baseline(
        limpios, semilla, k_minimo, k_maximo, min_salidas_train, min_salidas_valid
    )
    alt = p3_entrenar_alternativo(base, n_semillas=n_semillas)
    p3_escribir_resumen_ci(
        "P3 modelo alternativo",
        {
            "aceptado": alt.aceptado,
            "k": alt.k_alt,
            "train_silueta (promedio de semillas)": float(alt.elegido["silueta"]),
            "train_silueta baseline": base.metricas["silueta"],
            "valid_ari": float(alt.elegido["ari_valid"]),
            "valid_ari baseline": base.metricas["ari_valid"],
            "train_tam_grupo_min": int(alt.elegido["tam_min"]),
        },
    )
    nomenclatura = dataset.p3_cargar_estaciones()
    if registrar:
        p3_registrar_alternativo(alt, nomenclatura, de_nuevo, origen="cli")
    else:
        p3_guardar_artefactos_alternativo(alt, nomenclatura)
        logger.info("Registro en MLflow desactivado (P3_REGISTRAR_MLFLOW=false)")
    if exigir_aceptacion and not alt.aceptado:
        logger.error("El modelo alternativo no cumple los criterios de aceptación")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------------------------
# P1: demanda de viajes por estación-hora (baseline, HGB y XGBoost)
# ---------------------------------------------------------------------------------------------


# Por qué estos candidatos (según el EDA de P1):
# - El target es un conteo con 44% de ceros y varianza ~4 veces la media: la pérdida Poisson
#   modela conteos y nunca predice viajes negativos.
# - La tabla tiene ~16M de filas: los dos usan histogramas y entrenan en minutos.
# - Los rezagos de estaciones nuevas vienen vacíos (NaN): los dos los manejan sin imputar.
# - La demanda depende de combinaciones (estación × hora × día): los árboles las capturan
#   sin crearlas a mano.
# HGB viene en sklearn; XGBoost es el estándar de la industria y sirve para comparar.


class P1BaselinePromedio(BaseEstimator, RegressorMixin):
    """Predice el promedio de la misma hora en las semanas anteriores (baseline estacional).

    Si una estación-hora no tiene historia (estación nueva), usa su nivel de train y, si tampoco
    existe, el nivel del sistema para esa hora y día. Tiene la misma interfaz `fit`/`predict` que
    los modelos candidatos, así que se evalúa y registra con las mismas funciones.
    """

    def fit(self, X: pd.DataFrame, y) -> "P1BaselinePromedio":
        self.nivel_ = P1NivelEstacion().fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        niveles = self.nivel_.transform(X)
        prediccion = X[f"viajes_promedio_{cfg.P1_SEMANAS_PROMEDIO}sem"].to_numpy(
            "float64", copy=True
        )
        for respaldo in ["viajes_nivel_estacion", "viajes_nivel_sistema"]:
            faltante = np.isnan(prediccion)
            prediccion[faltante] = niveles[respaldo].to_numpy("float64")[faltante]
        return np.nan_to_num(prediccion, nan=0.0)


def p1_construir_modelo(
    clave: str, semilla: int = cfg.SEMILLA, hiperparametros: dict | None = None
):
    """Regresa el modelo sin entrenar: baseline, HGB o XGBoost (estos dos con el nivel de train).

    `hiperparametros` se aplica sobre los de config.py; las claves `*_optuna` lo exigen.
    """
    base = clave.removesuffix("_optuna")
    if clave.endswith("_optuna") and hiperparametros is None:
        raise ValueError(f"{clave} necesita los hiperparámetros de la búsqueda (tuning_p1.py)")
    if base == "baseline":
        return P1BaselinePromedio()
    if base == "hgb":
        regresor = HistGradientBoostingRegressor(
            **{**cfg.P1_PARAMETROS_HGB, **(hiperparametros or {})},
            categorical_features="from_dtype",
            random_state=semilla,
        )
    elif base == "xgb":
        regresor = XGBRegressor(
            **{**cfg.P1_PARAMETROS_XGB, **(hiperparametros or {})},
            enable_categorical=True,
            random_state=semilla,
        )
    else:
        raise ValueError(
            f"Modelo de P1 desconocido: {clave} (opciones: {', '.join(cfg.P1_MODELOS)})"
        )
    return Pipeline([("nivel", P1NivelEstacion()), ("modelo", regresor)])


def p1_cargar_conjuntos(ruta: Path = cfg.RUTA_P1_DEMANDA) -> dict[str, pd.DataFrame]:
    """Lee la tabla de P1 y regresa train, valid y test (las horas en `brecha` se descartan)."""
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta.name}; ejecutar `make features-p1`")
    tabla = pd.read_parquet(
        ruta, columns=[*cfg.P1_VARIABLES, "hora_inicio", "viajes", "particion"]
    )
    return {
        conjunto: tabla[tabla["particion"] == conjunto].reset_index(drop=True)
        for conjunto in ["train", "valid", "test"]
    }


def p1_evaluar(modelo, datos: pd.DataFrame, conjunto: str) -> tuple[dict[str, float], np.ndarray]:
    """Regresa ({conjunto}_mae, {conjunto}_rmse) y las predicciones del modelo en `datos`."""
    prediccion = p1_predecir(modelo, datos)
    metricas = {
        f"{conjunto}_mae": float(mean_absolute_error(datos["viajes"], prediccion)),
        f"{conjunto}_rmse": float(root_mean_squared_error(datos["viajes"], prediccion)),
    }
    return metricas, prediccion


def p1_mae_por_hora(datos: pd.DataFrame, prediccion: np.ndarray) -> pd.Series:
    """MAE por hora del día: el error importa más en los picos de 8:00 y 18:00."""
    return (datos["viajes"] - prediccion).abs().groupby(datos["hora"]).mean()


def p1_peores_estaciones(datos: pd.DataFrame, prediccion: np.ndarray, n: int = 10) -> pd.DataFrame:
    """Las `n` estaciones con mayor MAE: dónde el operador debe confiar menos en la predicción."""
    errores = pd.DataFrame(
        {"origen_id": datos["origen_id"], "error": (datos["viajes"] - prediccion).abs()}
    )
    resumen = errores.groupby("origen_id")["error"].agg(mae="mean", horas="size")
    resumen["viajes_promedio"] = datos.groupby("origen_id")["viajes"].mean()
    return resumen.sort_values("mae", ascending=False).head(n).round(3)


def p1_serie_diaria(datos: pd.DataFrame, prediccion: np.ndarray) -> pd.DataFrame:
    """Suma por día de los viajes reales y predichos de todas las estaciones."""
    dia = datos["hora_inicio"].dt.normalize()
    return pd.DataFrame({"real": datos["viajes"], "prediccion": prediccion}).groupby(dia).sum()


def p1_mae_por_zona(datos: pd.DataFrame, prediccion: np.ndarray) -> pd.DataFrame:
    """MAE diario por zona (municipio): el nivel al que el operador planea la flota."""
    por_zona = p1_predecir_por_zona(datos, prediccion)
    error = (por_zona["real"] - por_zona["prediccion"]).abs()
    resumen = error.groupby(por_zona["zona"], observed=True).agg(mae_diario="mean", dias="size")
    resumen["viajes_diarios_promedio"] = por_zona.groupby("zona", observed=True)["real"].mean()
    return resumen.round(2)


def p1_curva_aprendizaje(
    modelo, train: pd.DataFrame, valid: pd.DataFrame, clave: str, pasos: int = 10
) -> pd.DataFrame:
    """MAE de train y valid cada cierto número de árboles, sobre muestras de ambos conjuntos."""
    nivel, regresor = modelo.named_steps["nivel"], modelo.named_steps["modelo"]
    muestras = {
        nombre: datos.sample(min(len(datos), cfg.P1_MUESTRA_DIAGNOSTICO), random_state=cfg.SEMILLA)
        for nombre, datos in [("train", train), ("valid", valid)]
    }
    entradas = {nombre: nivel.transform(m[cfg.P1_VARIABLES]) for nombre, m in muestras.items()}
    es_xgb = clave.startswith("xgb")
    total = regresor.n_estimators if es_xgb else regresor.n_iter_
    iteraciones = np.unique(np.linspace(1, total, pasos, dtype=int))
    curva = pd.DataFrame({"iteracion": iteraciones})
    for nombre, entrada in entradas.items():
        real = muestras[nombre]["viajes"]
        if es_xgb:
            errores = [
                mean_absolute_error(real, regresor.predict(entrada, iteration_range=(0, int(i))))
                for i in iteraciones
            ]
        else:
            # staged_predict recorre los árboles una sola vez; se guardan solo los pasos pedidos
            pedidas = set(iteraciones.tolist())
            errores = [
                mean_absolute_error(real, prediccion)
                for i, prediccion in enumerate(regresor.staged_predict(entrada), start=1)
                if i in pedidas
            ]
        curva[f"{nombre}_mae"] = errores
    return curva


def p1_importancia(modelo, valid: pd.DataFrame, repeticiones: int = 3) -> pd.Series:
    """Importancia por permutación en una muestra de valid (aumento del MAE al desordenar)."""
    muestra = valid.sample(min(len(valid), cfg.P1_MUESTRA_DIAGNOSTICO), random_state=cfg.SEMILLA)
    resultado = permutation_importance(
        modelo,
        muestra[cfg.P1_VARIABLES],
        muestra["viajes"],
        scoring="neg_mean_absolute_error",
        n_repeats=repeticiones,
        random_state=cfg.SEMILLA,
    )
    return pd.Series(resultado.importances_mean, index=cfg.P1_VARIABLES).sort_values()


def p1_guardar_diagnosticos(
    clave: str, modelo, conjuntos: dict[str, pd.DataFrame], predicciones: dict[str, np.ndarray]
) -> tuple[list[Path], pd.DataFrame | None]:
    """Guarda en reports/figures/ las gráficas y tablas de un modelo; regresa rutas y curva."""
    carpeta, nombre = cfg.FIGURES_DIR, cfg.P1_MODELOS[clave][0]
    valid, test = conjuntos["valid"], conjuntos["test"]
    mae_hora = pd.DataFrame(
        {
            "valid": p1_mae_por_hora(valid, predicciones["valid"]),
            "test": p1_mae_por_hora(test, predicciones["test"]),
        }
    )
    carpeta.mkdir(parents=True, exist_ok=True)
    mae_hora.round(4).to_csv(carpeta / f"p1_{clave}_mae_por_hora.csv")
    peores = p1_peores_estaciones(test, predicciones["test"])
    peores.to_csv(carpeta / f"p1_{clave}_peores_estaciones_test.csv")
    zonas = p1_mae_por_zona(test, predicciones["test"])
    zonas.to_csv(carpeta / f"p1_{clave}_mae_por_zona_test.csv")
    rutas = [
        carpeta / f"p1_{clave}_mae_por_hora.csv",
        carpeta / f"p1_{clave}_peores_estaciones_test.csv",
        carpeta / f"p1_{clave}_mae_por_zona_test.csv",
    ]
    rutas.append(
        plots.p1_guardar_figura(
            plots.p1_grafica_mae_por_hora(
                mae_hora["valid"], mae_hora["test"], f"{nombre}: MAE por hora"
            ),
            carpeta / f"p1_{clave}_mae_por_hora.png",
        )
    )
    rutas.append(
        plots.p1_guardar_figura(
            plots.p1_grafica_real_vs_prediccion(
                test["viajes"].to_numpy("float64"), predicciones["test"], f"{nombre}: test"
            ),
            carpeta / f"p1_{clave}_real_vs_prediccion_test.png",
        )
    )
    rutas.append(
        plots.p1_guardar_figura(
            plots.p1_grafica_serie_diaria(
                p1_serie_diaria(test, predicciones["test"]), f"{nombre}: viajes diarios en test"
            ),
            carpeta / f"p1_{clave}_serie_diaria_test.png",
        )
    )
    curva = None
    if clave != "baseline":
        curva = p1_curva_aprendizaje(modelo, conjuntos["train"], valid, clave)
        rutas.append(
            plots.p1_guardar_figura(
                plots.p1_grafica_curva_aprendizaje(curva, f"{nombre}: curva de aprendizaje"),
                carpeta / f"p1_{clave}_curva_aprendizaje.png",
            )
        )
        rutas.append(
            plots.p1_guardar_figura(
                plots.p1_grafica_importancia(
                    p1_importancia(modelo, valid), f"{nombre}: importancia en valid"
                ),
                carpeta / f"p1_{clave}_importancia.png",
            )
        )
    return rutas, curva


def p1_commit_actual() -> str:
    """Commit corto con el que se ejecuta (`desconocido` si no hay repositorio Git)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cfg.PROJ_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "desconocido"


def p1_parametros(clave: str, n_train: int, hiperparametros: dict | None = None) -> dict:
    """Parámetros que permiten reproducir el run: datos, partición, horizonte y modelo."""
    parametros = {
        "rango_datos": f"{cfg.MES_INICIO} a {cfg.MES_FIN}",
        "data_version": dataset.version_datos(),
        "semilla": cfg.SEMILLA,
        "granularidad": "estacion_hora",
        "horizonte": "dia_siguiente",
        "frontera_valid": cfg.FRONTERA_VALID,
        "frontera_test": cfg.FRONTERA_TEST,
        "brecha_horas": cfg.BRECHA_HORAS,
        "n_variables": len(cfg.P1_VARIABLES),
        "n_train": n_train,
        "modelo": clave,
    }
    if clave == "baseline":
        parametros["semanas_promedio"] = cfg.P1_SEMANAS_PROMEDIO
    else:
        base = cfg.P1_PARAMETROS_HGB if clave.startswith("hgb") else cfg.P1_PARAMETROS_XGB
        hiper = {**base, **(hiperparametros or {})}
        parametros.update({f"modelo_{nombre}": valor for nombre, valor in hiper.items()})
    return parametros


def p1_carpeta_artefacto(ruta: Path) -> str:
    """Subcarpeta del run donde va un artefacto: las figuras en plots/ y las tablas en tables/."""
    return "plots" if ruta.suffix == ".png" else "tables"


def p1_archivos_configuracion() -> list[Path]:
    """Archivos con los que corre el run: config.py y, si existen, MLproject y python_env.yaml."""
    candidatos = [
        Path(cfg.__file__),
        cfg.PROJ_ROOT / "MLproject",
        cfg.PROJ_ROOT / "python_env.yaml",
    ]
    return [ruta for ruta in candidatos if ruta.exists()]


def p1_registrar_run(
    clave: str,
    parametros: dict,
    metricas: dict,
    artefactos: list[Path],
    modelo,
    ejemplo: pd.DataFrame,
    train: pd.DataFrame,
    curva: pd.DataFrame | None = None,
    etiquetas_extra: dict | None = None,
) -> str:
    """Registra un run de P1 en el experimento del proyecto y regresa su id."""
    import mlflow
    import mlflow.data
    from mlflow.models import infer_signature
    import mlflow.sklearn

    if not cfg.MLFLOW_TRACKING_URI:
        raise ValueError("MLFLOW_TRACKING_URI no está definida: copiar .env.example a .env")
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.EXPERIMENTO_MLFLOW)
    nombre_run, etapa, familia = cfg.P1_MODELOS[clave]
    with mlflow.start_run(run_name=nombre_run) as corrida:
        mlflow.set_tags(
            {
                "equipo": "07",
                "question": "p1",
                "commit": p1_commit_actual(),
                "autor": "Daniel Sánchez Hernández",
                "stage_of_work": etapa,
                "model_family": familia,
                "fase": cfg.FASE_ENTREGA,
                "mlflow.note.content": cfg.P1_DESCRIPCIONES[clave],
                **(etiquetas_extra or {}),
            }
        )
        mlflow.log_params(parametros)
        mlflow.log_metrics(metricas)
        if curva is not None:
            for fila in curva.itertuples():
                mlflow.log_metric("train_mae_por_iteracion", fila.train_mae, step=fila.iteracion)
                mlflow.log_metric("valid_mae_por_iteracion", fila.valid_mae, step=fila.iteracion)
        # La tabla ya está en memoria; MLflow guarda su esquema, perfil y huella (digest)
        mlflow.log_input(
            mlflow.data.from_pandas(
                train, source=str(cfg.RUTA_P1_DEMANDA), name="p1_train", targets="viajes"
            ),
            context="training",
        )
        for ruta in artefactos:
            mlflow.log_artifact(str(ruta), artifact_path=p1_carpeta_artefacto(ruta))
        mlflow.log_artifact(str(cfg.RUTA_MANIFIESTO), artifact_path="data")
        for ruta in p1_archivos_configuracion():
            mlflow.log_artifact(str(ruta), artifact_path="config")
        # cloudpickle: el formato por defecto (skops) rechaza las clases propias de P1
        mlflow.sklearn.log_model(
            modelo,
            name=nombre_run,
            signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
            input_example=ejemplo,
            serialization_format="cloudpickle",
        )
    logger.success(f"Run '{nombre_run}' registrado: {corrida.info.run_id}")
    return corrida.info.run_id


def p1_combinar_comparacion(comparacion: pd.DataFrame, ruta: Path) -> pd.DataFrame:
    """Une la comparación de esta corrida con la ya guardada en `ruta`.

    Reemplaza las filas de los modelos entrenados ahora y conserva las demás, así una corrida
    parcial (por ejemplo solo el baseline) no borra la comparación de los otros modelos.
    """
    if not ruta.exists():
        return comparacion
    previas = pd.read_csv(ruta, index_col=0)
    combinada = pd.concat([previas.drop(index=comparacion.index, errors="ignore"), comparacion])
    return combinada.sort_values("valid_mae")


def p1_ajustar_y_evaluar(
    clave: str,
    conjuntos: dict[str, pd.DataFrame],
    registrar: bool,
    hiperparametros: dict | None = None,
    etiquetas_extra: dict | None = None,
) -> dict[str, float]:
    """Ajusta un modelo con todo train, lo evalúa en valid y test y, si se pide, registra su run."""
    train = conjuntos["train"]
    nombre = cfg.P1_MODELOS[clave][0]
    modelo = p1_construir_modelo(clave, hiperparametros=hiperparametros)
    logger.info(f"Entrenando {nombre}")
    inicio = time.perf_counter()
    modelo.fit(train[cfg.P1_VARIABLES], train["viajes"])
    # El run se abre después de entrenar, así que su duración en MLflow no incluye el fit
    metricas = {"segundos_entrenamiento": round(time.perf_counter() - inicio, 1)}
    predicciones = {}
    for conjunto in ["valid", "test"]:
        resultado, predicciones[conjunto] = p1_evaluar(modelo, conjuntos[conjunto], conjunto)
        metricas.update(resultado)
    artefactos, curva = p1_guardar_diagnosticos(clave, modelo, conjuntos, predicciones)
    logger.info(f"{nombre}: {metricas}")
    if registrar:
        p1_registrar_run(
            clave,
            p1_parametros(clave, len(train), hiperparametros),
            metricas,
            artefactos,
            modelo,
            train[cfg.P1_VARIABLES].sample(5, random_state=cfg.SEMILLA).astype({"zona": str}),
            train,
            curva,
            etiquetas_extra,
        )
    return metricas


@app.command()
def p1_entrenar(
    modelos: str = typer.Option("baseline,hgb,xgb", help="Modelos a entrenar, separados por coma"),
    registrar: bool = typer.Option(
        False, "--registrar", envvar="P1_REGISTRAR_MLFLOW", help="Registrar los runs en MLflow"
    ),
):
    """Entrena y evalúa los modelos de P1; elige el mejor por valid_mae (test solo se reporta)."""
    claves = [clave.strip() for clave in modelos.split(",") if clave.strip()]
    conjuntos = p1_cargar_conjuntos()
    logger.info(", ".join(f"{nombre}: {len(datos)} filas" for nombre, datos in conjuntos.items()))
    resumen = {
        cfg.P1_MODELOS[clave][0]: p1_ajustar_y_evaluar(clave, conjuntos, registrar)
        for clave in claves
    }
    comparacion = pd.DataFrame(resumen).T.sort_values("valid_mae").round(4)
    ruta_comparacion = cfg.FIGURES_DIR / "p1_comparacion_modelos.csv"
    p1_combinar_comparacion(comparacion, ruta_comparacion).to_csv(ruta_comparacion)
    logger.info(f"Comparación (ordenada por valid_mae):\n{comparacion.to_string()}")
    logger.success(f"Mejor modelo por valid_mae: {comparacion.index[0]}")
    if not registrar:
        logger.info("Registro en MLflow desactivado (usar --registrar o P1_REGISTRAR_MLFLOW=true)")


# ---------------------------------------------------------------------------------------------
# P2: riesgo de estación (baseline histórico, HistGradientBoosting y XGBoost)
# Uso: python -m tc5061_proyecto_equipo07.modeling.train p2-entrenar [--registrar]
# Lee data/processed/p2_cambios_estacion_bin.parquet (`make features-p2`), agrega la historia
# "hasta ayer", deja la matriz de confusión y la comparación en reports/figures/p2_* y, con
# --registrar (o P2_REGISTRAR_MLFLOW=true), registra un run por modelo.
# ---------------------------------------------------------------------------------------------


class P2BaselineHistorico(BaseEstimator, ClassifierMixin):
    """Probabilidad de cada riesgo = su frecuencia histórica "hasta ayer"; sin_riesgo es lo que
    sobra. Tiene la misma interfaz que el candidato, así que se evalúa y registra igual."""

    def fit(self, X: pd.DataFrame, y=None) -> "P2BaselineHistorico":
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_bicis = X["hist_sin_bicis"].to_numpy()
        p_anclajes = X["hist_sin_anclajes"].to_numpy()
        return np.column_stack([1 - p_bicis - p_anclajes, p_bicis, p_anclajes])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def p2_construir_modelo(
    clave: str,
    min_samples_leaf: int = cfg.P2_MIN_SAMPLES_LEAF,
    semilla: int = cfg.SEMILLA,
    hiperparametros: dict | None = None,
):
    """Regresa el modelo sin entrenar: el baseline histórico, HistGradientBoosting o XGBoost."""
    base = clave.removesuffix("_optuna")
    if clave.endswith("_optuna") and hiperparametros is None:
        raise ValueError(f"{clave} necesita los hiperparámetros de la búsqueda (tuning_p2.py)")
    hiper = hiperparametros or {}
    if base == "baseline":
        return P2BaselineHistorico()
    if base == "hgb":
        return HistGradientBoostingClassifier(
            **{**cfg.P2_PARAMETROS_HGB, "min_samples_leaf": min_samples_leaf, **hiper},
            random_state=semilla,
        )
    if base == "xgb":
        return XGBClassifier(**{**cfg.P2_PARAMETROS_XGB, **hiper}, random_state=semilla)
    raise ValueError(f"Modelo de P2 desconocido: {clave} (opciones: {', '.join(cfg.P2_MODELOS)})")


def p2_calcular_metricas(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """PR-AUC por clase de riesgo y de riesgo en general, y log-loss (no dependen del umbral)."""
    return {
        "ap_sin_bicis": average_precision_score(y == 1, p[:, 1]),
        "ap_sin_anclajes": average_precision_score(y == 2, p[:, 2]),
        "ap_riesgo": average_precision_score(y > 0, 1 - p[:, 0]),
        "log_loss": log_loss(y, p, labels=[0, 1, 2]),
    }


def p2_elegir_umbral(y_binaria: np.ndarray, p: np.ndarray) -> float:
    """Umbral que maximiza el F1 de una clase (se elige con validación)."""
    precision, recall, umbrales = precision_recall_curve(y_binaria, p)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return umbrales[np.argmax(f1[:-1])]  # el último punto de la curva no tiene umbral asociado


def p2_evaluar_modelo(modelo, val: pd.DataFrame, test: pd.DataFrame, variables: list[str]) -> dict:
    """Evalúa un modelo entrenado: los umbrales salen de validación y las métricas de test.

    Regresa las probabilidades de test, la clase predicha, los umbrales, las métricas de test y,
    en `metricas_valid`, las de validación que no dependen del umbral (con ellas se elige).
    """
    y_val, y_test = val["clase"].to_numpy(), test["clase"].to_numpy()
    p_val, p_test = modelo.predict_proba(val[variables]), modelo.predict_proba(test[variables])
    umbrales = {k: p2_elegir_umbral(y_val == k, p_val[:, k]) for k in (1, 2)}
    pred = p2_decidir_clase(p_test, umbrales)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, labels=[1, 2], zero_division=0
    )
    metricas = {
        **p2_calcular_metricas(y_test, p_test),
        "precision_sin_bicis": precision[0],
        "recall_sin_bicis": recall[0],
        "f1_sin_bicis": f1[0],
        "precision_sin_anclajes": precision[1],
        "recall_sin_anclajes": recall[1],
        "f1_sin_anclajes": f1[1],
    }
    return {
        "proba_test": p_test,
        "pred": pred,
        "umbrales": umbrales,
        "metricas": metricas,
        "metricas_valid": p2_calcular_metricas(y_val, p_val),
    }


def p2_reportar_evaluacion(nombre: str, resultado: dict, y_test: np.ndarray) -> None:
    """Deja en el log las métricas, los umbrales, el reporte por clase y la matriz de confusión."""
    umbrales = {cfg.P2_CLASES[k]: round(float(v), 3) for k, v in resultado["umbrales"].items()}
    reporte = classification_report(
        y_test, resultado["pred"], target_names=cfg.P2_CLASES, digits=3, zero_division=0
    )
    matriz = confusion_matrix(y_test, resultado["pred"], labels=[0, 1, 2])
    logger.info(
        f"{nombre}: métricas en test\n{pd.Series(resultado['metricas']).round(4).to_string()}\n"
        f"umbrales (elegidos en validación): {umbrales}\n{reporte}\n"
        f"matriz de confusión (filas = real, columnas = predicho):\n{matriz}"
    )


def p2_archivos_configuracion() -> list[Path]:
    """Archivos con los que corre el run: config.py y, si existen, MLproject y python_env.yaml."""
    candidatos = [
        Path(cfg.__file__),
        cfg.PROJ_ROOT / "MLproject",
        cfg.PROJ_ROOT / "python_env.yaml",
    ]
    return [ruta for ruta in candidatos if ruta.exists()]


def p2_guardar_artefactos(clave: str, resultado: dict, y_test: np.ndarray) -> dict:
    """Guarda en reports/figures/ la matriz de confusión y curva PR de test.
    Además devuelve, por carpeta del run, lo que se registra en MLflow."""
    matriz = confusion_matrix(y_test, resultado["pred"], labels=[0, 1, 2])
    nombre = cfg.P2_MODELOS[clave][0]
    figura = plots.p2_grafica_matriz_confusion(
        matriz, cfg.P2_CLASES, f"Matriz de confusión: {nombre} (test)"
    )
    ruta_cm = plots.p3_guardar_figura(figura, cfg.FIGURES_DIR / f"p2_{clave}_matriz_confusion.png")
    ruta_cm_json = cfg.FIGURES_DIR / f"p2_{clave}_matriz_confusion.json"
    contenido = {
        "clases": cfg.P2_CLASES,
        "filas": "clase real",
        "columnas": "clase predicha",
        "matriz": matriz.tolist(),
    }
    ruta_cm_json.write_text(json.dumps(contenido, ensure_ascii=False, indent=1))

    figura_pr = plots.p2_grafica_curva_pr(
        y_test, resultado["proba_test"], cfg.P2_CLASES, f"Curva precisión-recall: {nombre} (test)"
    )
    ruta_pr = plots.p3_guardar_figura(figura_pr, cfg.FIGURES_DIR / f"p2_{clave}_curva_pr.png")

    rutas_config = p2_archivos_configuracion()
    ruta_manifiesto = cfg.RUTA_MANIFIESTO

    return {
        "plots": [ruta_cm, ruta_pr],
        "tables": [ruta_cm_json],
        "config": rutas_config,
        "data": [ruta_manifiesto],
    }


def p2_nombre_run(base: str) -> str:
    """Nombre del run: `<MLFLOW_RUN_NAME>-<base>` si la variable existe; si no, `p2-<base>`."""
    return f"{cfg.MLFLOW_RUN_NAME}-{base}" if cfg.MLFLOW_RUN_NAME else f"p2-{base}"


def p2_parametros_comunes(
    fit: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, peso_historico: int
) -> dict:
    """Parámetros que comparten los runs de P2: datos, etiqueta, partición e historia."""
    return {
        "rango_datos": f"{cfg.MES_INICIO} a {cfg.MES_FIN}",
        "data_version": dataset.version_datos(),
        "semilla": cfg.SEMILLA,
        "bin_min": cfg.P2_BIN_MIN,
        "h_bins": cfg.P2_H_BINS,
        "alfa": cfg.P2_ALFA,
        "frontera_valid": cfg.FRONTERA_VALID,
        "frontera_test": cfg.FRONTERA_TEST,
        "brecha_horas": cfg.BRECHA_HORAS,
        "peso_historico": peso_historico,
        "n_variables": len(cfg.P2_FEATURES_CON_HISTORICO),
        "variables": ",".join(cfg.P2_FEATURES_CON_HISTORICO),
        "datos_desde": str(fit["dia"].min().date()),
        "datos_hasta": str(test["dia"].max().date()),
        "filas_fit": len(fit),
        "filas_val": len(val),
        "filas_test": len(test),
    }


def p2_parametros_modelo(
    clave: str, modelo, resultado: dict, hiperparametros: dict | None = None
) -> dict:
    """Parámetros propios de un modelo, más los umbrales de decisión elegidos en validación."""
    parametros = {"modelo": clave}
    base = clave.removesuffix("_optuna")
    if base in ("hgb", "xgb"):
        defecto = cfg.P2_PARAMETROS_HGB if base == "hgb" else cfg.P2_PARAMETROS_XGB
        nombres = [*defecto, *(hiperparametros or {})]
        if base == "hgb":
            nombres.append("min_samples_leaf")
        valores = modelo.get_params()
        for nombre in dict.fromkeys(nombres):  # sin repetir los que la búsqueda sobrescribe
            parametros[f"modelo_{nombre}"] = valores[nombre]
    if base == "hgb":
        parametros["modelo_n_iter"] = modelo.n_iter_
    umbrales = resultado["umbrales"]
    parametros.update({f"umbral_{cfg.P2_CLASES[k]}": float(v) for k, v in umbrales.items()})
    return parametros


def p2_registrar_run(
    clave: str,
    parametros: dict,
    metricas: dict,
    artefactos: dict,
    modelo,
    ejemplo: pd.DataFrame,
    fit: pd.DataFrame,
    etiquetas_extra: dict | None = None,
) -> str:
    """Registra un run de P2 en el experimento del proyecto y regresa su id."""
    import mlflow
    import mlflow.data
    from mlflow.data.digest_utils import compute_pandas_digest
    from mlflow.models import infer_signature
    import mlflow.sklearn

    if not cfg.MLFLOW_TRACKING_URI:
        raise ValueError("MLFLOW_TRACKING_URI no está definida: copiar .env.example a .env")
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.EXPERIMENTO_MLFLOW)
    base, etapa, familia = cfg.P2_MODELOS[clave]
    nombre_run = p2_nombre_run(base)
    with mlflow.start_run(run_name=nombre_run) as corrida:
        mlflow.set_tags(
            {
                "equipo": "07",
                "question": "p2",
                "tarea": "clasificacion",
                "commit": p3_commit_actual(),
                "autor": "Fernando Arreola Treviño",
                "stage_of_work": etapa,
                "model_family": familia,
                "fase": cfg.FASE_ENTREGA,
                "mlflow.note.content": cfg.P2_DESCRIPCIONES[clave],
                **p3_etiquetas_ci(),
                **(etiquetas_extra or {}),
            }
        )
        mlflow.log_params(parametros)
        mlflow.log_metrics(metricas)
        for carpeta, rutas in artefactos.items():
            for ruta in rutas:
                mlflow.log_artifact(str(ruta), artifact_path=carpeta)
        huella = compute_pandas_digest(fit.head(10_000))
        mlflow.log_input(
            mlflow.data.from_pandas(
                fit, source=str(cfg.RUTA_P2_CAMBIOS), name="p2_fit", targets="clase", digest=huella
            ),
            context="training",
        )
        # cloudpickle: el formato por defecto (skops) rechaza la clase propia del baseline
        mlflow.sklearn.log_model(
            modelo,
            name=nombre_run,
            signature=infer_signature(ejemplo, modelo.predict_proba(ejemplo)),
            input_example=ejemplo,
            pyfunc_predict_fn="predict_proba",
            serialization_format="cloudpickle",
        )
    logger.success(f"Run '{nombre_run}' registrado: {corrida.info.run_id}")
    return corrida.info.run_id


def p2_cargar_cambios(ruta: Path = cfg.RUTA_P2_CAMBIOS) -> pd.DataFrame:
    """Lee la tabla estación-bin de P2 que deja `features p2-features`."""
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta.name}; ejecutar `make features-p2`")
    return pd.read_parquet(ruta)


def p2_ajustar_y_evaluar(
    clave: str,
    fit: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    comunes: dict,
    registrar: bool,
    min_samples_leaf: int = cfg.P2_MIN_SAMPLES_LEAF,
    hiperparametros: dict | None = None,
    etiquetas_extra: dict | None = None,
) -> dict[str, float]:
    """Ajusta un modelo con fit, lo evalúa en validación y test y, si se pide, registra su run.

    Regresa las métricas del run: valid_* (con ellas se elige), test_* y el tiempo de ajuste.
    """
    variables = cfg.P2_FEATURES_CON_HISTORICO
    base = cfg.P2_MODELOS[clave][0]
    x_fit, y_test = fit[variables], test["clase"].to_numpy()
    modelo = p2_construir_modelo(clave, min_samples_leaf, hiperparametros=hiperparametros)
    logger.info(f"Entrenando {base}")
    inicio = time.perf_counter()
    modelo.fit(x_fit, fit["clase"])
    segundos = round(time.perf_counter() - inicio, 1)
    if list(modelo.classes_) != [0, 1, 2]:
        raise ValueError(f"{base}: clases inesperadas {list(modelo.classes_)}")

    resultado = p2_evaluar_modelo(modelo, val, test, variables)
    p2_reportar_evaluacion(base, resultado, y_test)
    artefactos = p2_guardar_artefactos(clave, resultado, y_test)
    metricas = {
        **{f"valid_{k}": float(v) for k, v in resultado["metricas_valid"].items()},
        **{f"test_{k}": float(v) for k, v in resultado["metricas"].items()},
        "segundos_entrenamiento": segundos,
    }
    if registrar:
        p2_registrar_run(
            clave,
            {**comunes, **p2_parametros_modelo(clave, modelo, resultado, hiperparametros)},
            metricas,
            artefactos,
            modelo,
            x_fit.head(5),
            fit,
            etiquetas_extra,
        )
    return metricas


@app.command()
def p2_entrenar(
    peso_historico: int = typer.Option(cfg.P2_PESO_HISTORICO, envvar="P2_PESO_HISTORICO"),
    min_samples_leaf: int = typer.Option(cfg.P2_MIN_SAMPLES_LEAF, envvar="P2_MIN_SAMPLES_LEAF"),
    registrar: bool = typer.Option(cfg.P2_REGISTRAR_MLFLOW, envvar="P2_REGISTRAR_MLFLOW"),
):
    """Entrena y evalúa el baseline histórico, HistGradientBoosting y XGBoost de P2.

    Los umbrales de decisión salen de validación y el mejor modelo se elige por
    valid_ap_riesgo; test solo se reporta.
    """
    cambios = p2_cargar_cambios()
    fit, val, test = p2_preparar_conjuntos(cambios, peso_historico)
    del cambios

    comunes = p2_parametros_comunes(fit, val, test, peso_historico)
    # Los candidatos `*_optuna` salen de la búsqueda (tuning_p2.py); no se entrenan sin ella
    claves = [clave for clave in cfg.P2_MODELOS if not clave.endswith("_optuna")]
    resumen = {
        cfg.P2_MODELOS[clave][0]: p2_ajustar_y_evaluar(
            clave, fit, val, test, comunes, registrar, min_samples_leaf
        )
        for clave in claves
    }

    comparacion = pd.DataFrame(resumen).T.sort_values("valid_ap_riesgo", ascending=False)
    comparacion = comparacion.round(4)
    comparacion.to_csv(cfg.FIGURES_DIR / "p2_comparacion_modelos.csv")
    logger.info(f"Comparación (ordenada por valid_ap_riesgo):\n{comparacion.to_string()}")
    if not registrar:
        logger.info("Registro en MLflow desactivado (usar --registrar o P2_REGISTRAR_MLFLOW=true)")


if __name__ == "__main__":
    # Se importa con su nombre real: si los comandos corrieran como `__main__`, los objetos que se
    # serializan (modelo en MLflow) apuntarían a `__main__.p3_aplicar_pesos` y no se podrían cargar.
    from tc5061_proyecto_equipo07.modeling.train import app as app_real

    app_real()
