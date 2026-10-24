"""Búsqueda de configuración de P3 con Optuna: un run padre por búsqueda y un run hijo por trial.

Uso (lo invoca MLproject; los parámetros vacíos conservan el valor de config.py):
    mlflow run . -e tune -P question=p3 -P model=alternativo -P n_trials=60

Optuna sustituye a la rejilla del modelo alternativo (`p3_buscar_alternativo`). Cada trial
promedia varias semillas y maximiza la silueta, sujeta a los criterios de aceptación acordados:
ARI de estabilidad >= baseline y tamaño mínimo de grupo. La estabilidad se mide reasignando las
estaciones con el año de validación; P3 no usa test. La configuración se elige con la misma regla
de siempre (`p3_seleccionar_alternativo`) y queda como candidato `pca_kmeans_optuna`.
"""

import os
import time

from loguru import logger
import mlflow
import numpy as np
import optuna
import pandas as pd
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07 import dataset
from tc5061_proyecto_equipo07.modeling.train import (
    P3Baseline,
    p3_ajustar_baseline,
    p3_cargar_modelado,
    p3_entrenar_alternativo,
    p3_escribir_resumen_ci,
    p3_etiquetas_run,
    p3_evaluar_semillas,
    p3_nombrar_metricas,
    p3_nombre_run,
    p3_parametros_comunes,
    p3_preparar_experimento,
    p3_promediar_semillas,
    p3_registrar_alternativo,
    p3_seleccionar_alternativo,
)
from tc5061_proyecto_equipo07.modeling.tuning_p1 import p1_leer_opcion

app = typer.Typer(add_completion=False)

METRICAS_TRIAL = [
    "silueta",
    "silueta_min",
    "davies_bouldin",
    "calinski_harabasz",
    "ari_valid",
    "ari_min",
    "tam_min",
]


def p3_espacio_alternativo(trial: optuna.Trial, limites: dict, max_componentes: int) -> dict:
    """Configuración del modelo alternativo que explora la búsqueda."""
    componentes_min, componentes_max = limites["componentes"]
    k_min, k_max = limites["k"]
    return {
        "ponderar": trial.suggest_categorical("ponderar", [False, True]),
        # El PCA no admite más componentes que variables ni que estaciones
        "componentes": trial.suggest_int(
            "componentes", componentes_min, min(componentes_max, max_componentes)
        ),
        "k": trial.suggest_int("k", k_min, k_max),
    }


ESPACIOS = {"alternativo": p3_espacio_alternativo}


def p3_etiquetas_busqueda(descripcion: str | None = None) -> dict:
    """Etiquetas de los runs de la búsqueda (padre e hijos): stage_of_work=search."""
    extra = {"mlflow.note.content": descripcion} if descripcion else {}
    return p3_etiquetas_run(
        "alternativo_optuna", "optuna", descripcion=False, stage_of_work="search", **extra
    )


def p3_cargar_base() -> P3Baseline:
    """Baseline de P3 con los datos de modelado: es la referencia y el espacio de comparación."""
    limpios = p3_cargar_modelado(cfg.MES_INICIO, cfg.MES_FIN)
    return p3_ajustar_baseline(
        limpios,
        cfg.P3_SEMILLA,
        cfg.P3_K_MINIMO,
        cfg.P3_K_MAXIMO,
        cfg.P3_MIN_SALIDAS_TRAIN,
        cfg.P3_MIN_SALIDAS_VALID,
    )


def p3_buscar(
    base: P3Baseline,
    clave: str,
    n_trials: int,
    timeout_min: int,
    n_semillas: int,
    tam_min_aceptable: int,
) -> tuple[optuna.Study, pd.DataFrame]:
    """Corre la búsqueda dentro del run padre activo; cada trial es un run hijo.

    Regresa el estudio y las corridas por semilla de las configuraciones evaluadas.
    """
    limites = cfg.P3_BUSQUEDA[clave]
    max_componentes = min(len(base.columnas_modelo), len(base.x_train))
    corridas = []

    def objetivo(trial: optuna.Trial) -> float:
        configuracion = ESPACIOS[clave](trial, limites, max_componentes)
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) as hijo:
            mlflow.set_tags(p3_etiquetas_busqueda())
            mlflow.log_params(configuracion)
            trial.set_user_attr("run_id", hijo.info.run_id)
            inicio = time.perf_counter()
            semillas = p3_evaluar_semillas(
                base,
                configuracion["ponderar"],
                configuracion["componentes"],
                configuracion["k"],
                n_semillas,
            )
            segundos = round(time.perf_counter() - inicio, 1)
            fila = p3_promediar_semillas(semillas).iloc[0]
            metricas = {nombre: float(fila[nombre]) for nombre in METRICAS_TRIAL}
            # Violación de cada criterio de aceptación: <= 0 lo cumple
            violaciones = {
                "ari_valid": base.metricas["ari_valid"] - metricas["ari_valid"],
                "tam_min": tam_min_aceptable - metricas["tam_min"],
            }
            mlflow.set_tag("cumple_criterios", str(max(violaciones.values()) <= 0).lower())
            mlflow.log_metrics(
                {**p3_nombrar_metricas(metricas), "segundos_entrenamiento": segundos}
            )
        if not np.isfinite(metricas["silueta"]):
            raise optuna.TrialPruned()
        for criterio, violacion in violaciones.items():
            trial.set_constraint(criterio, violacion)
        corridas.append(semillas)
        return metricas["silueta"]

    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.SEMILLA),
    )
    estudio.optimize(objetivo, n_trials=n_trials, timeout=timeout_min * 60 or None)
    return estudio, pd.concat(corridas, ignore_index=True) if corridas else pd.DataFrame()


@app.command()
def buscar(
    model: str = typer.Option(..., help="alternativo"),
    n_trials: str = "",
    fraccion_train: str = "",
    timeout_min: str = "",
    sin_candidato: bool = typer.Option(False, help="Solo la búsqueda, sin registrar el candidato"),
):
    """Busca la configuración del modelo alternativo y deja la mejor como candidato en MLflow."""
    clave = model.strip().lower()
    if clave not in ESPACIOS:
        raise typer.BadParameter(f"model debe ser alternativo, no '{model}'")
    # En Windows, `mlflow run` pasa los parámetros vacíos como comillas ('' o "")
    if fraccion_train.strip().strip("\"'"):
        raise typer.BadParameter("fraccion_train no aplica a P3: se agrupan todas las estaciones")
    valores = cfg.P3_BUSQUEDA[clave]
    n_trials = p1_leer_opcion(n_trials, int, valores["n_trials"])
    timeout = p1_leer_opcion(timeout_min, int, valores["timeout_min"])
    n_semillas = cfg.P3_SEMILLAS_BUSQUEDA
    tam_min = cfg.P3_TAM_MIN_ACEPTABLE

    # `mlflow run` exporta MLFLOW_RUN_ID; el run padre de la búsqueda debe ser uno propio
    os.environ.pop("MLFLOW_RUN_ID", None)
    p3_preparar_experimento()

    base = p3_cargar_base()
    logger.info(
        f"Búsqueda {clave}: {n_trials} trials, {timeout} min, {n_semillas} semillas por trial | "
        f"baseline: k={base.k_elegido}, silueta {base.metricas['silueta']:.3f}, "
        f"ARI {base.metricas['ari_valid']:.3f}"
    )

    nombre_padre = f"{p3_nombre_run(cfg.P3_MODELOS['alternativo_optuna'][0])}-busqueda"
    with mlflow.start_run(run_name=nombre_padre) as padre:
        mlflow.set_tags(
            p3_etiquetas_busqueda(
                "Búsqueda con Optuna (TPE con restricciones); cada trial es un run hijo. "
                "Objetivo: silueta promedio entre semillas, con ARI de estabilidad >= baseline y "
                "tamaño mínimo de grupo. Test no se usa en P3."
            )
        )
        mlflow.log_params(
            {
                **p3_parametros_comunes(base),
                "objetivo": "train_silueta",
                "n_trials_max": n_trials,
                "timeout_min": timeout,
                "semillas_busqueda": n_semillas,
                "tam_min_aceptable": tam_min,
                "k_baseline": base.k_elegido,
            }
        )
        estudio, semillas = p3_buscar(base, clave, n_trials, timeout, n_semillas, tam_min)
        terminados = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not terminados:
            raise RuntimeError("Ningún trial terminó: revisar los runs hijos en MLflow")

        # TPE puede repetir una configuración: se conserva una sola fila por configuración
        semillas = semillas.drop_duplicates(["ponderar", "componentes", "k", "semilla"])
        resumen = p3_promediar_semillas(semillas)
        elegido, cumplen, _ = p3_seleccionar_alternativo(
            resumen, base.metricas["ari_valid"], tam_min
        )
        mejor = next(
            t
            for t in terminados
            if (t.params["ponderar"], t.params["componentes"], t.params["k"])
            == (bool(elegido["ponderar"]), int(elegido["componentes"]), int(elegido["k"]))
        )
        mlflow.log_params({f"mejor_{nombre}": valor for nombre, valor in mejor.params.items()})
        mlflow.log_param("best_child_run_id", mejor.user_attrs["run_id"])
        mlflow.log_metrics(
            {
                "n_trials_terminados": len(terminados),
                "n_configuraciones_que_cumplen": len(cumplen),
                "mejor_train_silueta": float(elegido["silueta"]),
                "mejor_valid_ari": float(elegido["ari_valid"]),
            }
        )
        mlflow.log_text(resumen.to_csv(index=False), "tables/p3_optuna_configuraciones.csv")
    logger.success(
        f"Mejor configuración (trial {mejor.number}): silueta {elegido['silueta']:.4f} | "
        f"{mejor.params} | cumplen los criterios: {len(cumplen)} de {len(resumen)}"
    )
    p3_escribir_resumen_ci(
        "P3 búsqueda con Optuna",
        {
            "trials terminados": len(terminados),
            "configuraciones que cumplen": len(cumplen),
            "train_silueta": float(elegido["silueta"]),
            "train_silueta baseline": base.metricas["silueta"],
            "valid_ari": float(elegido["ari_valid"]),
            "valid_ari baseline": base.metricas["ari_valid"],
        },
    )

    if not sin_candidato:
        alt = p3_entrenar_alternativo(
            base,
            n_semillas=n_semillas,
            tam_min_aceptable=tam_min,
            busqueda_previa=(semillas, resumen),
        )
        # de_nuevo: el candidato debe ser el de esta búsqueda, no uno anterior con el mismo nombre
        p3_registrar_alternativo(
            alt,
            dataset.p3_cargar_estaciones(),
            de_nuevo=True,
            origen="optuna",
            carpeta=cfg.FIGURES_DIR,
            clave="alternativo_optuna",
            etiquetas_extra={"busqueda_run_id": padre.info.run_id},
        )


if __name__ == "__main__":
    app()
