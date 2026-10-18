# Diseño del pipeline de ML — Proyecto MiBici (Equipo 7)

> **Estado:** pipeline implementado en la primera entrega. Las cifras salen del EDA (notebooks `0.01-fat`, `0.02-dsh`, `0.01-lasd` y `0.04-dsh`) y de `definicion-datos.md`.

## 1. Vista general

![Pipeline de ML por pregunta](pipeline-ml.png)

*Una limpieza común se bifurca en tres carriles, uno por pregunta. Cada carril recorre agregación, features, partición, entrenamiento y evaluación, y registra en MLflow con el mismo `data_version`. En verde, los controles contra fuga de información del futuro.* El diagrama de componentes y herramientas está en la sección *Arquitectura* del README ([`arquitectura.png`](arquitectura.png)).

El flujo sigue la convención de datos de CCDS (`raw` → `interim` → `processed`): cada etapa lee de una carpeta y escribe en la siguiente, y nunca se modifica lo que ya está en `raw`.

| Etapa | Entrada | Código (CCDS) | Salida |
|---|---|---|---|
| **1. Ingesta** | CSV del portal | `dataset.py` (`make data`) | `data/raw/` con los CSV originales, validados contra el manifiesto `references/fuentes-datos.csv` (registros y SHA-256) |
| **2. Normalización** | `data/raw/` | `dataset.py` | `data/interim/` con Parquet normalizado (columnas, codificación, fechas y tipos) |
| **3. Limpieza** | `data/interim/` | `dataset.py` | `data/processed/viajes_limpios_*.parquet` |
| **4. Agregación, features y partición** | `data/processed/` | `features.py` (`make features`) | Tablas por pregunta en `data/processed/`, con la partición común |
| **5. Entrenamiento y evaluación** | Tablas de features | `modeling/train.py`, `plots.py` (`make train`) | Runs de MLflow (parámetros, métricas, artefactos y modelo) y figuras en `reports/figures/` |
| **6. Predicción** | Modelo ajustado y datos nuevos | `modeling/predict.py` | Predicciones por lotes (P1 por estación y por zona; clase de riesgo en P2; grupo en P3) |

**Trazabilidad:** cada run de MLflow guarda `data_version`, el hash del manifiesto, de modo que cualquier métrica se puede ligar a la versión exacta de los datos.

Los notebooks son solo de exploración: la lógica final vive en el módulo y los runs se ejecutan con `make` o `python -m`.

**Principios comunes**
- **Reproducibilidad:** semilla 42, dependencias con versiones fijadas y manifiesto de datos.
- **Sin fuga:** partición temporal y cortes de información definidos en `definicion-datos.md` (sección 5).
- **Cada decisión de limpieza se justifica con una cifra del EDA.**
- **Cada corrida** registra en MLflow parámetros, métricas, etiquetas (equipo, pregunta, commit) y artefactos.

## 2. Limpieza común a las tres preguntas

| Regla | Cifra | Tratamiento |
|---|---|---|
| Viajes duplicados | 758 filas repetidas completas | Eliminar, conservando una |
| Estaciones sin nomenclatura | 3 ids (124, 405, 407), 3,961 viajes | Excluir: sin capacidad ni ubicación no se pueden usar en P2 ni P3 |
| Duración cero | 55,034 viajes | Excluir (probable desbloqueo fallido) |
| Duración extrema | 159 viajes de más de 24 h (máximo 469 h) | Excluir |
| Estaciones con capacidad 0 | 4 estaciones | Excluir de P2 |
| Meses sin datos | Ninguno: 2023-04 y 2023-05 se republicaron | Si un mes faltara, no se imputa |
| Datos personales | `usuario_id`, `genero`, `anio_de_nacimiento` | Descartar antes del modelado (minimización) |

**Resultado:** 59,912 registros retirados (0.221 %); quedan 27,026,645 viajes.

## 3. P1: demanda de viajes (regresión)

Se predicen los viajes que inician en cada estación por hora de servicio (5:00 a 0:59) **para el día siguiente**: al cierre del día D se predicen las 20 horas de D+1. La zona (municipio: GDL, ZPN o TLQ) es la suma de sus estaciones. Código: `features.py` (funciones `p1_*` y `P1NivelEstacion`), `modeling/train.py` (sección P1) y `modeling/predict.py` (`p1_predecir`, `p1_predecir_por_zona`).

### Etapas

| Etapa | Descripción |
|---|---|
| **Limpieza** | La común del proyecto (`dataset.limpiar`); se descartan los 193 viajes en horas sin servicio (1:00 a 4:59) |
| **Agregación** | Contar viajes por `origen_id` y hora de `inicio_del_viaje`. Las horas sin viajes valen 0 **solo dentro del periodo en que la estación operó** (de su primer a su último viaje); antes y después quedan vacías, para no inventar demanda. Resultado: 16,103,922 estación-horas |
| **Features** | Ver tabla siguiente (`make features-p1`) |
| **Partición** | La común del proyecto, con una brecha de 24 h antes de cada frontera: train 10,531,227 filas, valid 2,673,888 y test 2,884,207 |
| **Entrenamiento** | Baseline, HistGradientBoosting Poisson y XGBoost Poisson, cada uno en un `Pipeline` (`make train-p1`) |
| **Evaluación** | MAE y RMSE en valid y test, MAE por hora, por zona y por estación, real contra predicción y serie diaria; se elige con valid |

### Variables

| Variable | Tipo | Por qué es pertinente |
|---|---|---|
| `hora`, `dia_semana`, `es_fin_de_semana`, `mes` | Calendario | El uso tiene ciclos diarios (picos de 8:00 y 18:00), semanales y estacionales |
| `es_festivo` | Calendario | La demanda cae o cambia de perfil en los días de descanso obligatorio (LFT art. 74) |
| `viajes_rezago_1d`, `viajes_rezago_7d` | Rezago | La misma hora de ayer y de la semana pasada es el mejor predictor de la demanda |
| `viajes_promedio_4sem` | Ventana (solo días anteriores) | Promedio de la misma hora en las 4 semanas previas: suaviza el ruido sin usar el futuro |
| `capacidad`, `latitud`, `longitud`, `zona` | Estación | Ubicación y tamaño explican parte del nivel de demanda |
| `dias_desde_apertura` | Estación | Una estación recién abierta todavía no alcanza su demanda estable |
| `origen_id` | Estación | Solo para aprender el nivel de cada estación (`P1NivelEstacion`, con train); no entra como variable al modelo |

**Restricción de horizonte:** todos los rezagos son de **1 día o más**, porque se predice el día siguiente.

### Modelos y métricas

| Rol | Modelo | Nota |
|---|---|---|
| Baseline | Promedio de la misma hora en las 4 semanas anteriores (si no hay historia, el nivel de la estación en train o el del sistema) | Es el piso que se debe superar |
| Candidato 1 | HistGradientBoosting con pérdida Poisson | Conteo con 44 % de ceros; nunca predice negativos; usa histogramas y maneja rezagos vacíos sin imputar |
| Candidato 2 | XGBoost con pérdida Poisson | Las mismas ventajas; compara dos implementaciones de boosting |

- **Métricas:** MAE (principal, en viajes por estación-hora), RMSE (penaliza los errores en los picos). Se evita MAPE porque el 44 % de las estación-horas tiene 0 viajes.
- **Hiperparámetros:** fijos en `config.py` para esta entrega (sin búsqueda); el modelo se elige con `valid_mae` y test solo se reporta. Resultado: XGBoost (`valid_mae` 0.9761, `test_mae` 0.9952), 4.2 % mejor que el baseline en valid.

## 4. P2: riesgo de quedarse sin bicis o sin anclajes (clasificación)

Para cada estación y bloque de 30 minutos se predice si en las **próximas 2 horas** estará sin riesgo, en riesgo de quedarse **sin bicis** o **sin anclajes**. Código: `features.py` (funciones `p2_*` y `P2HistoriaHastaAyer`), `modeling/train.py` (sección P2) y `modeling/predict.py` (`p2_decidir_clase`).

### Etapas

| Etapa | Descripción |
|---|---|
| **Agregación** | Por estación y bloque de 30 min: salidas (`origen_id`, `inicio_del_viaje`) y llegadas (`destino_id`, `fin_del_viaje`); cambio neto = llegadas − salidas. Se excluyen las 4 estaciones con capacidad 0 |
| **Etiqueta** | En los 4 bloques siguientes (2 h): si la mayor pérdida acumulada alcanza el 50 % de la capacidad (`P2_ALFA`), *sin bicis*; si la mayor ganancia la alcanza, *sin anclajes*; si ocurren los dos, gana el de mayor estrés relativo |
| **Features** | Ver tabla siguiente; todas con información hasta el bloque actual |
| **Partición** | La común del proyecto, con brecha de 24 h; validación solo para elegir umbrales y prueba solo se reporta |
| **Entrenamiento** | Baseline de frecuencia histórica → HistGradientBoostingClassifier (`make train-p2`) |
| **Evaluación** | PR-AUC por clase y de riesgo en general, log-loss, y precisión, recall y F1 por clase con los umbrales de validación; matriz de confusión |

### Variables

| Variable | Por qué es pertinente |
|---|---|
| `neto_{1,2,4,8}_relativo` | Tendencia reciente del cambio neto (30 min a 4 h) sobre la capacidad: −5 bicis es grave en una estación de 15 y no en una de 39 |
| `salidas_{2,4}`, `entradas_{2,4}` | Distingue una estación muy activa de una con movimiento equilibrado |
| `cumsum_dia_relativo` | Estado aproximado de la estación respecto al inicio del día |
| `hora_sin`, `hora_cos`, `es_fin_semana` | Los picos de desbalance siguen la hora y el tipo de día |
| `dpcapacity`, `latitude`, `longitude` | Tamaño y ubicación de la estación |
| `hist_sin_bicis`, `hist_sin_anclajes` | Frecuencia de cada riesgo en los días previos de la misma estación, tipo de día y hora, suavizada hacia la frecuencia global (solo "hasta ayer") |

### Modelos y métricas

| Rol | Modelo | Nota |
|---|---|---|
| Baseline | Frecuencia histórica (`P2BaselineHistorico`) | Lo que se esperaría si la estación repite su historia |
| Candidato | HistGradientBoostingClassifier | Interacciones entre hora, estación y tendencia sin crearlas a mano |

- **Desbalance:** el riesgo es un evento raro, así que no se usa *accuracy*.
- **Métricas:** PR-AUC (*average precision*) por clase y de riesgo en general, log-loss, y precisión, recall y F1 por clase.
- **Umbral:** por clase, el que maximiza el F1 en validación; la clase predicha es el riesgo cuyo score relativo a su umbral sea mayor.

### Limitación declarada

No hay inventario de bicicletas, por lo que la etiqueta es un **proxy**: no conoce el nivel inicial de la estación ni ve el rebalanceo realizado. El modelo detecta **presión de desbalance**, no el inventario real, y debe validarse con el equipo de operación antes de usarse.

## 5. P3: grupos de estaciones (clustering)

Código: funciones `p3_*` de `features.py` y `modeling/train.py` (`make train-p3`).

### Etapas

| Etapa | Descripción |
|---|---|
| **Agregación** | Una fila por estación, calculada **solo con el periodo de entrenamiento** |
| **Filtro** | Estaciones con al menos 1,000 salidas en entrenamiento (362) |
| **Preparación** | Estandarizar las variables |
| **Entrenamiento** | Baseline: K-Means con `k` elegido por silueta entre 2 y 10. Alternativo: PCA + K-Means (`k` de 3 a 6, varias semillas), con criterios de aceptación fijados antes de ver los resultados |
| **Evaluación** | Silueta, Davies-Bouldin y Calinski-Harabasz; estabilidad con el año de validación (ARI); perfil medio y mapa por grupo |

### Variables (44)

| Variable | Por qué es pertinente |
|---|---|
| Perfil horario laboral (20 horas de servicio, normalizado) | Distingue estaciones de commuting (picos de mañana y tarde) de las de ocio |
| Perfil horario de fin de semana (20 horas) | Revela uso recreativo |
| Volumen diario | Separa estaciones de alto y bajo volumen |
| Balance neto sobre salidas | Estaciones que son origen neto o destino neto |
| Proporción de viajes circulares | Origen = destino, típico de uso recreativo |
| Duración mediana | Viajes cortos de conexión frente a recorridos largos |
| `latitud`, `longitud` | **No entran al clustering**; se usan después para ubicar y describir cada grupo |

### Resultado

El baseline elige `k = 2` (silueta 0.145, ARI con validación 0.656): un grupo con pico matutino y más periférico, y otro con pico vespertino, más céntrico, que tiende a perder bicicletas. El alternativo (`k = 3`, 5 componentes) no se acepta porque no mejora la silueta, aunque separa un perfil intermedio de uso mixto. El criterio decisivo es la **utilidad para planeación urbana**, no solo la métrica.

## 6. Comparación entre preguntas

| | P1 | P2 | P3 |
|---|---|---|---|
| **Unidad de análisis** | Estación-hora | Estación-bloque de 30 min | Estación |
| **Objetivo** | Conteo de viajes del día siguiente | Clase de riesgo en las próximas 2 h | Sin objetivo |
| **Baseline** | Promedio de la misma hora en 4 semanas | Frecuencia histórica | K-Means |
| **Modelo candidato** | HGB y XGBoost (Poisson) | HistGradientBoosting | PCA + K-Means |
| **Métrica principal** | MAE | PR-AUC | Silueta + estabilidad |
| **Riesgo principal** | Estaciones nuevas sin historia | Proxy sin inventario | Grupos poco separados |

## 7. Decisiones tomadas

- P1: horizonte del día siguiente, rezagos de 1 día o más y la zona como suma de estaciones.
- P2: umbral del 50 % de la capacidad en una ventana de 2 h, con bloques de 30 min.
- Los 55,034 viajes con duración cero se excluyen.
- P3: mínimo de 1,000 salidas en entrenamiento para entrar al clustering.
- Un baseline y al menos un candidato por pregunta en la primera entrega; el ajuste de hiperparámetros queda para las siguientes.
