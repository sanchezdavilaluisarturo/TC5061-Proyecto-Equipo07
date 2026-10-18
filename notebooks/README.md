# Notebooks

Convención de nombres: `FASE.NUMERO-INICIALES-descripcion.ipynb`

Ejemplo: `0.01-dsm-exploracion-inicial.ipynb`

Fases: 0 exploración (EDA), 1 limpieza y features, 2 visualizaciones, 3 modelado, 4 publicación y reportes.

Los notebooks son solo para exploración y para documentar hallazgos: la lógica final vive en el módulo `tc5061_proyecto_equipo07` y los runs oficiales se ejecutan con `make` o `python -m` (ver el README principal).

## Notebooks de P1 (demanda por estación-hora) y de la ingesta

| Notebook | Contenido | Dónde quedó la lógica |
|---|---|---|
| `0.02-dsh-exploracion-carga` | Descarga del portal, inventario por archivo (codificación, esquema y fechas), unión y Parquet | `dataset.py` (`make data`) |
| `0.04-dsh-eda-p1` | EDA de P1: calidad, estadísticas, distribuciones, patrones temporales y espaciales, granularidad y horizonte, y formulación | `features.py` (`p1_*`, `make features-p1`) y `modeling/train.py` (`make train-p1`) |

## Notebooks de P2 (riesgo de estación)

| Notebook | Contenido | Dónde quedó la lógica |
|---|---|---|
| `0.01-fat-exploracion-inicial-p2` | EDA sobre Parquet: nulos, años de nacimiento, duración y entradas/salidas por estación | — |
| `0.02-fat-exploracion-y-limpieza-p2` | Exploración y limpieza para P2 | `features.py` (`p2_*`) |
| `0.03-fat-feature-engineering-p2` | Etiqueta de riesgo y variables por estación-bloque | `features.py` (`p2_*`, `make features-p2`) |
| `0.04-fat-model-training-registration-p2` | Exploración del modelado y del registro | `modeling/train.py` (`make train-p2`) |

## Notebooks de P3 (clustering de estaciones)

Se ejecutan en este orden, desde esta carpeta. Cada uno **llama** al módulo `tc5061_proyecto_equipo07` y documenta los hallazgos; la lógica reutilizable no vive en los notebooks.

| Notebook | Paso | Código del módulo que usa |
|---|---|---|
| `0.01-lasd-exploracion-carga` | 1. Carga y descripción | `dataset.p3_cargar_viajes`, `dataset.p3_meses_sin_datos` |
| `0.02-lasd-eda-p3` | 2. EDA | `dataset.p3_cargar_estaciones`, `features.p3_variables_de_tiempo`, `plots.p3_grafica_*` |
| `0.03-lasd-limpieza-p3` | 3. Limpieza, partición y preprocesamiento | `dataset.limpiar`, `dataset.p3_preparar_modelado`, `features.p3_tabla_estaciones`, `features.p3_demanda_estacion_hora` |
| `0.04-lasd-baseline-p3` | 4. Modelo base | `train.p3_ajustar_baseline`, `train.p3_registrar_baseline` |
| `0.05-lasd-mejoramodelo-p3` | 4. Modelo alternativo | `train.p3_entrenar_alternativo`, `train.p3_registrar_alternativo` |

**Convención:** toda función nueva de P3 se llama `p3_<nombre>`. Los parámetros y el registro en MLflow se controlan con variables de entorno (`P3_*`, `MLFLOW_TRACKING_URI`, `MLFLOW_RUN_NAME`); ver la sección *Datos, features y entrenamiento de P3* del README principal. Las celdas de MLflow no escriben en el servidor salvo que se defina `P3_REGISTRAR_MLFLOW=true`.
