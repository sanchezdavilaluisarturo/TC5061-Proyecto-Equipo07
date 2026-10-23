"""Punto de entrada `train` del MLflow Project: elige la pregunta y el modelo, aplica los
hiperparámetros recibidos sobre los de config.py y llama a los comandos de modeling/train.py.

Uso (lo invoca MLproject; los parámetros vacíos conservan el valor de config.py):
    mlflow run . -e train -P question=p1 -P model=hgb -P n_estimators=300
"""

import os

from loguru import logger
import typer

from tc5061_proyecto_equipo07 import config as cfg

app = typer.Typer(add_completion=False)

# Hiperparámetro del proyecto -> nombre en el modelo de cada (pregunta, modelo). Lo que no
# aparece no aplica a ese modelo y se rechaza en lugar de ignorarse en silencio.
HIPERPARAMETROS = {
    ("p1", "hgb"): {
        "n_estimators": "max_iter",
        "learning_rate": "learning_rate",
        "max_depth": "max_depth",
        "min_samples_leaf": "min_samples_leaf",
    },
    ("p1", "xgb"): {
        "n_estimators": "n_estimators",
        "learning_rate": "learning_rate",
        "max_depth": "max_depth",
    },
    # min_samples_leaf de P2 es opción del CLI (p2-entrenar), no entrada del dict de config
    ("p2", "hgb"): {
        "n_estimators": "max_iter",
        "learning_rate": "learning_rate",
        "max_depth": "max_depth",
        "min_samples_leaf": "min_samples_leaf",
    },
    ("p2", "xgb"): {
        "n_estimators": "n_estimators",
        "learning_rate": "learning_rate",
        "max_depth": "max_depth",
    },
}
TIPOS = {"n_estimators": int, "learning_rate": float, "max_depth": int, "min_samples_leaf": int}

MODELOS_P3 = {"baseline": "p3-baseline", "alternativo": "p3-alternativo"}


def modelos_validos(pregunta: str) -> list[str]:
    """Modelos que acepta cada pregunta, tomados de config.py (P3: sus dos comandos)."""
    if pregunta == "p1":
        return list(cfg.P1_MODELOS)
    if pregunta == "p2":
        return list(cfg.P2_MODELOS)
    return list(MODELOS_P3)


def leer_hiperparametros(valores: dict[str, str]) -> dict:
    """Convierte los parámetros recibidos como texto; descarta los que vienen vacíos."""
    return {nombre: TIPOS[nombre](valor) for nombre, valor in valores.items() if valor.strip()}


def aplicar_hiperparametros(pregunta: str, modelo: str, recibidos: dict) -> list[str]:
    """Pone los hiperparámetros en los dicts de config.py y regresa las opciones extra del CLI."""
    if not recibidos:
        return []
    permitidos = HIPERPARAMETROS.get((pregunta, modelo))
    if permitidos is None:
        raise typer.BadParameter(f"{pregunta}/{modelo or 'todos'} no recibe hiperparámetros")
    sobran = sorted(set(recibidos) - set(permitidos))
    if sobran:
        raise typer.BadParameter(
            f"{', '.join(sobran)} no aplica a {pregunta}/{modelo} "
            f"(acepta: {', '.join(permitidos)})"
        )
    destino = {
        ("p1", "hgb"): cfg.P1_PARAMETROS_HGB,
        ("p1", "xgb"): cfg.P1_PARAMETROS_XGB,
        ("p2", "hgb"): cfg.P2_PARAMETROS_HGB,
        ("p2", "xgb"): cfg.P2_PARAMETROS_XGB,
    }[(pregunta, modelo)]
    extras = []
    for nombre, valor in recibidos.items():
        if (pregunta, nombre) == ("p2", "min_samples_leaf"):
            extras += ["--min-samples-leaf", str(valor)]
        else:
            destino[permitidos[nombre]] = valor
    return extras


def comandos_de_train(pregunta: str, modelo: str, extras: list[str]) -> list[list[str]]:
    """Comandos de modeling/train.py que se ejecutan; siempre registran en MLflow."""
    if pregunta == "p1":
        return [["p1-entrenar", "--registrar", *(["--modelos", modelo] if modelo else [])]]
    if pregunta == "p2":
        if modelo:
            logger.warning("p2-entrenar entrena el baseline y el HGB juntos; `model` solo valida")
        return [["p2-entrenar", "--registrar", *extras]]
    # --de-nuevo: un `mlflow run` debe dejar su run aunque ya haya uno con la misma configuración
    elegidos = [modelo] if modelo else list(MODELOS_P3)
    return [[MODELOS_P3[nombre], "--registrar", "--de-nuevo"] for nombre in elegidos]


@app.command()
def entrenar(
    question: str = typer.Option(..., help="p1, p2 o p3"),
    model: str = typer.Option("", help="Modelo; vacío = todos los de la pregunta"),
    n_estimators: str = "",
    learning_rate: str = "",
    max_depth: str = "",
    min_samples_leaf: str = "",
):
    """Entrena y registra el modelo de una pregunta; lo no indicado sale de config.py."""
    pregunta = question.strip().lower()
    if pregunta not in ("p1", "p2", "p3"):
        raise typer.BadParameter(f"question debe ser p1, p2 o p3, no '{question}'")
    modelo = model.strip().lower()
    if modelo and modelo not in modelos_validos(pregunta):
        raise typer.BadParameter(
            f"modelo '{modelo}' desconocido para {pregunta} (opciones: "
            f"{', '.join(modelos_validos(pregunta))})"
        )
    recibidos = leer_hiperparametros(
        {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        }
    )
    extras = aplicar_hiperparametros(pregunta, modelo, recibidos)

    # `mlflow run` exporta MLFLOW_RUN_ID; train.py abre sus propios runs con nombre y etiquetas,
    # y con esa variable reanudarían el run del proyecto en lugar de crear los suyos.
    os.environ.pop("MLFLOW_RUN_ID", None)

    # Se importa con su nombre real para que el modelo serializado no apunte a `__main__`
    from tc5061_proyecto_equipo07.modeling.train import app as app_train

    for argumentos in comandos_de_train(pregunta, modelo, extras):
        logger.info(
            f"train.py {' '.join(argumentos)} | hiperparámetros: {recibidos or 'config.py'}"
        )
        app_train(args=argumentos, standalone_mode=False)


if __name__ == "__main__":
    app()
