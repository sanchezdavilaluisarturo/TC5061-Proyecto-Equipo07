# Tiempos de los modelos de P1

Tiempos medidos antes de fijar los límites de la búsqueda con Optuna (`P1_BUSQUEDA` en `config.py`). Se midió el ajuste con todo train (10,531,227 filas) y la predicción sobre valid (2,673,888 filas), con 16 CPUs.

| Modelo | Ajuste con todo train | Predicción en valid |
|---|---:|---:|
| XGBoost Poisson | 117.4 s | 1.7 s |
| HGB Poisson | 970.8 s | 7.7 s |

## Por qué no se usó la prueba con una muestra

La primera estimación se hizo con el 10 % de train: XGBoost tardó 62 s, y extrapolar de forma lineal daba unos 10 minutos por ajuste con todo train. El ajuste real tardó 2 minutos: el tiempo no creció en proporción a las filas. Por eso los límites se fijaron con los tiempos medidos con todo train y no con estimaciones.

## Cómo se fijaron los límites

| Modelo | Trials | Train por trial | Tope | Razón |
|---|---:|---|---:|---|
| XGBoost | 20 | 100 % | 60 min | 20 ajustes de unos 2 min caben en el tope con margen; la búsqueda real tardó unos 35 min |
| HGB | 10 | 15 % | 45 min | Con todo train, 10 ajustes de 16 min no caben; se busca con una muestra de train y el mejor trial se vuelve a entrenar con todo train |

## Script de medición

No registra en MLflow ni escribe archivos. Uso, desde la raíz del repo:

```bash
PYTHONPATH=. python medir_tiempo_p1.py --modelos xgb,hgb
PYTHONPATH=. python medir_tiempo_p1.py --modelos xgb --fraccion 0.1
```

```python
"""Mide cuánto tarda el ajuste de cada modelo de P1; no registra en MLflow ni escribe archivos."""

import os
import time

from loguru import logger
import typer

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.modeling.train import (
    p1_cargar_conjuntos,
    p1_construir_modelo,
    p1_evaluar,
)

app = typer.Typer(add_completion=False)


@app.command()
def medir(
    modelos: str = "xgb,hgb",
    fraccion: float = typer.Option(1.0, min=0.01, max=1.0),
    ruta: str = typer.Option(str(cfg.RUTA_P1_DEMANDA)),
):
    """Ajusta cada modelo sobre train y reporta segundos de ajuste, de predicción y valid_mae."""
    conjuntos = p1_cargar_conjuntos(type(cfg.RUTA_P1_DEMANDA)(ruta))
    train = conjuntos["train"]
    if fraccion < 1.0:
        train = train.sample(frac=fraccion, random_state=cfg.SEMILLA).reset_index(drop=True)
    valid = conjuntos["valid"]
    logger.info(f"CPUs: {os.cpu_count()} | train: {len(train)} filas | valid: {len(valid)} filas")
    for clave in [m.strip() for m in modelos.split(",") if m.strip()]:
        modelo = p1_construir_modelo(clave)
        inicio = time.perf_counter()
        modelo.fit(train[cfg.P1_VARIABLES], train["viajes"])
        segundos_ajuste = time.perf_counter() - inicio
        inicio = time.perf_counter()
        metricas, _ = p1_evaluar(modelo, valid, "valid")
        segundos_prediccion = time.perf_counter() - inicio
        logger.success(
            f"{clave}: ajuste {segundos_ajuste:.1f} s | predicción valid {segundos_prediccion:.1f} s"
            f" | valid_mae {metricas['valid_mae']:.4f}"
        )


if __name__ == "__main__":
    app()
```
