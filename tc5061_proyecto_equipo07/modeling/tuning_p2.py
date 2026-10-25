"""Búsqueda de hiperparámetros de P2 con Optuna: un run padre por búsqueda y un run hijo por trial.

Uso (lo invoca MLproject; los parámetros vacíos conservan el valor de config.py):
    mlflow run . -e tune -P question=p2 -P model=xgb -P n_trials=15

Cada trial ajusta con una muestra de fit y se califica con valid_ap_riesgo.
El mejor trial se vuelve a entrenar con todo fit y queda como candidato `p2-xgb-optuna`.
"""

import os
import time

from loguru import logger
import mlflow
import optuna
import pandas as pd
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.features import p2_preparar_conjuntos
from tc5061_proyecto_equipo07.modeling.train import (
    p2_ajustar_y_evaluar,
    p2_calcular_metricas,
    p2_cargar_cambios,
    p2_combinar_comparacion,
    p2_commit_actual,
    p2_construir_modelo,
    p2_etiquetas_ci,
    p2_nombre_run,
    p2_parametros_comunes,
)

app = typer.Typer(add_completion=False)


def p2_espacio_xgb(trial: optuna.Trial) -> dict:
    """Hiperparámetros de XGBoost que explora la búsqueda; n_estimators queda en config.py."""
    return {
        "max_leaves": trial.suggest_int("max_leaves", 30, 500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 1000.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }


ESPACIOS = {"xgb": p2_espacio_xgb}
# Caja "Description" del run padre
DESCRIPCION_BUSQUEDA = (
    "Búsqueda con Optuna (TPE); cada trial es un run hijo. Objetivo: valid_ap_riesgo. "
    "Test no se usa para decidir."
)


def p2_etiquetas_busqueda(clave: str, **extra) -> dict:
    """Etiquetas de los runs de la búsqueda (padre e hijos): stage_of_work=search, más `extra`."""
    return {
        "equipo": "07",
        "question": "p2",
        "tarea": "clasificacion",
        "commit": p2_commit_actual(),
        "autor": "Fernando Arreola Treviño",
        "stage_of_work": "search",
        "model_family": cfg.P2_MODELOS[clave][2],
        "fase": cfg.FASE_ENTREGA,
        **p2_etiquetas_ci(),
        **extra,
    }


def p2_leer_opcion(valor: str, tipo, por_defecto):
    """Convierte el parámetro recibido como texto; vacío = el valor de config.py."""
    valor = valor.strip().strip("\"'")
    return tipo(valor) if valor else por_defecto


def p2_preparar_busqueda(fraccion: float) -> dict:
    """Conjuntos de P2, sus parámetros comunes y la muestra de fit con la que entrena cada trial.
    Solo se muestrea fit: valid queda completo para elegir y test solo lo usa el candidato.
    """
    cambios = p2_cargar_cambios()
    fit, val, test = p2_preparar_conjuntos(cambios, cfg.P2_PESO_HISTORICO)
    del cambios
    muestra = fit.sample(frac=fraccion, random_state=cfg.SEMILLA) if fraccion < 1 else fit
    return {
        "fit": fit,
        "val": val,
        "test": test,
        "muestra": muestra,
        "comunes": p2_parametros_comunes(fit, val, test, cfg.P2_PESO_HISTORICO),
    }


def p2_buscar(
    clave: str, muestra: pd.DataFrame, val: pd.DataFrame, n_trials: int, timeout_min: int
) -> optuna.Study:
    """Corre la búsqueda dentro del run padre activo; cada trial es un run hijo."""
    variables = cfg.P2_FEATURES_CON_HISTORICO
    # Se separan una sola vez; todos los trials entrenan y se califican con los mismos datos
    x_muestra, y_muestra = muestra[variables], muestra["clase"]
    x_val, y_val = val[variables], val["clase"].to_numpy()

    def objetivo(trial: optuna.Trial) -> float:
        hiperparametros = ESPACIOS[clave](trial)
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) as hijo:
            mlflow.set_tags(p2_etiquetas_busqueda(clave))
            mlflow.log_params(hiperparametros)
            trial.set_user_attr("run_id", hijo.info.run_id)
            modelo = p2_construir_modelo(f"{clave}_optuna", hiperparametros=hiperparametros)
            inicio = time.perf_counter()
            modelo.fit(x_muestra, y_muestra)
            segundos = round(time.perf_counter() - inicio, 1)
            metricas = {
                f"valid_{k}": float(v)
                for k, v in p2_calcular_metricas(y_val, modelo.predict_proba(x_val)).items()
            }
            mlflow.log_metrics({**metricas, "segundos_entrenamiento": segundos})
        return metricas["valid_ap_riesgo"]

    estudio = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.SEMILLA)
    )
    estudio.optimize(objetivo, n_trials=n_trials, timeout=timeout_min * 60 or None)
    return estudio


@app.command()
def buscar(
    model: str = typer.Option(..., help="xgb"),
    n_trials: str = "",
    fraccion_train: str = "",
    timeout_min: str = "",
    sin_candidato: bool = typer.Option(False, help="Solo la búsqueda, sin re-entrenar el mejor"),
):
    """Busca hiperparámetros del modelo y deja el mejor como candidato en MLflow."""
    clave = model.strip().lower()
    if clave not in ESPACIOS:
        raise typer.BadParameter(f"model debe ser {', '.join(ESPACIOS)}, no '{model}'")
    valores = cfg.P2_BUSQUEDA[clave]
    n_trials = p2_leer_opcion(n_trials, int, valores["n_trials"])
    fraccion = p2_leer_opcion(fraccion_train, float, valores["fraccion_train"])
    timeout = p2_leer_opcion(timeout_min, int, valores["timeout_min"])
    if not 0 < fraccion <= 1:
        raise typer.BadParameter(f"fraccion_train debe estar en (0, 1], no {fraccion}")
    if not cfg.MLFLOW_TRACKING_URI:
        raise ValueError("MLFLOW_TRACKING_URI no está definida: exportarla o copiar .env.example")

    # `mlflow run` exporta MLFLOW_RUN_ID; el run padre de la búsqueda debe ser uno propio
    os.environ.pop("MLFLOW_RUN_ID", None)
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.EXPERIMENTO_MLFLOW)

    datos = p2_preparar_busqueda(fraccion)
    comunes = datos["comunes"]
    logger.info(
        f"Búsqueda {clave}: {n_trials} trials, {timeout} min, {len(datos['muestra'])} filas de fit"
    )

    nombre_padre = f"{p2_nombre_run(cfg.P2_MODELOS[clave + '_optuna'][0])}-busqueda"
    with mlflow.start_run(run_name=nombre_padre) as padre:
        mlflow.set_tags(
            p2_etiquetas_busqueda(clave, **{"mlflow.note.content": DESCRIPCION_BUSQUEDA})
        )
        mlflow.log_params(
            {
                "rango_datos": comunes["rango_datos"],
                "data_version": comunes["data_version"],
                "semilla": cfg.SEMILLA,
                "objetivo": "valid_ap_riesgo",
                "n_trials_max": n_trials,
                "timeout_min": timeout,
                "fraccion_train": fraccion,
                "n_train_busqueda": len(datos["muestra"]),
            }
        )
        estudio = p2_buscar(clave, datos["muestra"], datos["val"], n_trials, timeout)
        terminados = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not terminados:
            raise RuntimeError("Ningún trial terminó: revisar los runs hijos en MLflow")
        mejor = estudio.best_trial
        mlflow.log_params({f"mejor_{nombre}": valor for nombre, valor in mejor.params.items()})
        mlflow.log_param("best_child_run_id", mejor.user_attrs["run_id"])
        mlflow.log_metrics(
            {"n_trials_terminados": len(terminados), "mejor_valid_ap_riesgo": mejor.value}
        )
        trials = estudio.trials_dataframe(attrs=("number", "value", "params", "duration"))
        mlflow.log_text(trials.to_csv(index=False), "tables/trials_busqueda.csv")
    logger.success(
        f"Mejor trial {mejor.number}: valid_ap_riesgo {mejor.value:.4f} | {mejor.params}"
    )

    if not sin_candidato:
        del datos["muestra"]
        metricas = p2_ajustar_y_evaluar(
            f"{clave}_optuna",
            datos["fit"],
            datos["val"],
            datos["test"],
            comunes,
            registrar=True,
            hiperparametros=mejor.params,
            etiquetas_extra={"busqueda_run_id": padre.info.run_id},
        )
        fila = pd.DataFrame({cfg.P2_MODELOS[f"{clave}_optuna"][0]: metricas}).T.round(4)
        ruta = cfg.FIGURES_DIR / "p2_comparacion_modelos.csv"
        p2_combinar_comparacion(fila, ruta).to_csv(ruta)


if __name__ == "__main__":
    # Se importa con su nombre real, como en train.py, para que el modelo serializado se pueda cargar
    from tc5061_proyecto_equipo07.modeling.tuning_p2 import app as app_real

    app_real()
