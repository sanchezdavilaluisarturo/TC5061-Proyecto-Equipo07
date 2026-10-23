# TC5061-Proyecto-Equipo07

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto integrador de **Operaciones de Aprendizaje Automático (MLOps, TC5061)** del Equipo 7. Propone y construye una solución de Machine Learning sobre los datos abiertos de **MiBici**, el sistema de bicicleta pública del Área Metropolitana de Guadalajara.

## Instalación

Requiere Python 3.11 y [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/sanchezdavilaluisarturo/TC5061-Proyecto-Equipo07.git
cd TC5061-Proyecto-Equipo07
uv venv --python 3.11 .venv
source .venv/bin/activate
make requirements        # uv pip install -r requirements.txt (incluye el paquete con -e .)
cp .env.example .env     # URL de MLflow; el .env no se versiona
```

El `.env` apunta al servidor MLflow del equipo (https://mlflow.equipo7mlops.com), que no pide credenciales: con él se pueden registrar y consultar los runs. Si no responde, `make mlflow-local` levanta un MLflow local de respaldo (SQLite + `mlartifacts/`) y en el `.env` se usa `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`.

## Ejecución

```bash
# Datos: descarga, normaliza, valida contra el manifiesto y limpia
make data

# Features de P1, P2 y P3 (o una sola pregunta: make features-p1, features-p2, features-p3)
make features

# Entrenamiento y evaluación de P1, P2 y P3, sin registrar en MLflow (o make train-p1, train-p2, train-p3)
make train

# Mismo entrenamiento, registrando los runs en MLflow (una variable por pregunta)
P1_REGISTRAR_MLFLOW=true P2_REGISTRAR_MLFLOW=true P3_REGISTRAR_MLFLOW=true make train
```

`make help` lista todos los comandos. `train-p1` depende de `features-p1`, así que siempre entrena con las features al día.

Para reproducir las corridas con `mlflow run` (entorno propio y runs registrados), ver [Ejecución con MLflow Project](#ejecución-con-mlflow-project).

### Comandos por pregunta

Cada etapa es un comando del módulo `tc5061_proyecto_equipo07`; los notebooks solo exploran y documentan los hallazgos (se corren desde `notebooks/`).

| Etapa | P1 | P2 | P3 | Qué deja listo |
|---|---|---|---|---|
| Datos (común) | `make data` | `make data` | `make data` | CSV en `data/raw/`, Parquet normalizado en `data/interim/` y viajes limpios en `data/processed/` |
| Features | `make features-p1` | `make features-p2` | `make features-p3` | La tabla de cada pregunta en `data/processed/`, con la partición común |
| Entrenamiento y evaluación | `make train-p1` | `make train-p2` | `make train-p3` | Métricas, figuras en `reports/figures/pN_*` y, si se pide, runs en MLflow |
| Workflow de GitHub Actions | `p1.yml` | `p2.yml` | `p3.yml` | Lo mismo en un runner de GitHub |

**Convención:** cada pregunta prefija lo que agrega al módulo (`p1_`, `p2_`, `p3_`; por ejemplo, `p3_perfil_de_estaciones`) y el código de una pregunta no depende del de otra. Lo común usa nombres genéricos (`FRONTERA_VALID`, `FRONTERA_TEST`, `BRECHA_HORAS`).

**Variables de entorno.** El mismo comando corre en la terminal, en los notebooks y en GitHub Actions; cada parámetro tiene también su opción de línea de comandos:

| Variable | Qué controla | Por defecto |
|---|---|---|
| `MLFLOW_TRACKING_URI` | URL del servidor MLflow (en CI, el secret del repositorio `MLFLOW_TRACKING_URI`) | `.env` |
| `P1_REGISTRAR_MLFLOW`, `P2_REGISTRAR_MLFLOW`, `P3_REGISTRAR_MLFLOW` | Escribir en MLflow los runs de cada pregunta | `false` |
| `MLFLOW_RUN_NAME` | P2 y P3: prefijo del nombre de los runs (en CI lo fija el workflow). P1 usa nombres fijos | P2: `baseline-historico` y `hgb`; P3: `baseline` y `pca_kmeans` |
| `P2_PESO_HISTORICO`, `P2_MIN_SAMPLES_LEAF` | Parámetros del modelo de P2 | `20`, `500` |
| `P3_REGISTRAR_DE_NUEVO` | P3: crear un run nuevo aunque ya exista uno terminado con ese nombre | `false` |
| `P3_EXIGIR_ACEPTACION` | P3: salir con error si el modelo alternativo no cumple los criterios de aceptación | `false` |
| `P3_EXPERIMENTO` | P3: experimento de MLflow | `mibici-equipo07` (el de `config.py`) |
| `P3_SEMILLA`, `P3_K_MINIMO`, `P3_K_MAXIMO`, `P3_MIN_SALIDAS_TRAIN`, `P3_MIN_SALIDAS_VALID`, `P3_COMPONENTES`, `P3_SEMILLAS_BUSQUEDA` | Parámetros del modelo de P3 | `42`, `2`, `10`, `1000`, `300`, `3,4,5,6,8,10`, `5` |

**Pruebas:** `pytest tests`.

**GitHub Actions:** `TC5061-Proyecto-Equipo07.yml` corre el job de calidad (`ruff` y `pytest`) en cada PR que toca el módulo, las pruebas, el Makefile o los workflows. A mano (*Run workflow*) corre la calidad y luego la acción de cada pregunta (`p1.yml`, `p2.yml` y `p3.yml`), que también se puede correr sola. Las tres descargan los datos con caché por el hash del manifiesto, entrenan con su target de `make` y solo registran en MLflow si se marca `registrar`.

**MLflow del equipo:** https://mlflow.equipo7mlops.com/ (la URL se define en `MLFLOW_TRACKING_URI`). Los runs de la entrega están en el experimento `mibici-equipo07`: `p1-baseline-promedio-4sem`, `p1-hgb-poisson` y `p1-xgb-poisson` (P1), `p2-baseline-historico` y `p2-hgb` (P2), y `p3-baseline` y `p3-pca_kmeans` (P3). En la Fase 2, P1 agrega la búsqueda `p1-xgb-optuna-busqueda` (con sus runs `trial_N`), el candidato `p1-xgb-optuna` y el modelo registrado `mibici-equipo07-p1-demanda` (alias `champion`).

## Ejecución con MLflow Project

Las corridas de la entrega se lanzan con `mlflow run` desde el `MLproject` de la raíz, no desde un notebook. MLflow construye un entorno aislado con las versiones fijas de `requirements.txt` (`python_env.yaml`), ejecuta el punto de entrada pedido y registra el run. Rutas, fechas y semillas no se repiten en la línea de comandos: salen de `tc5061_proyecto_equipo07/config.py`.

### Requisitos

| Requisito | Para qué | Cómo |
|---|---|---|
| CLI de `mlflow` | Lanzar `mlflow run` | Viene en `requirements.txt` (con el `.venv` activo) |
| [pyenv](https://github.com/pyenv/pyenv) | Que MLflow instale Python 3.11 en su entorno aislado | macOS: `brew install pyenv`. WSL/Ubuntu: ver el bloque siguiente |
| `MLFLOW_TRACKING_URI` | Dónde se registran los runs | Ver el paso 1 |

En WSL/Ubuntu, pyenv se instala con su script oficial, después de las dependencias con las que se compila Python:

```bash
sudo apt update && sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev libffi-dev liblzma-dev libncurses-dev tk-dev xz-utils
curl https://pyenv.run | bash
# Solo estas dos líneas en el ~/.bashrc (no las de `pyenv init` que sugiere el instalador)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
# Abrir una terminal nueva y verificar
pyenv --version
which -a python   # el primero debe seguir siendo el del entorno activo
```

`mlflow run` solo necesita el comando `pyenv`. Las líneas `eval "$(pyenv init ...)"` que imprime el instalador ponen los *shims* de pyenv antes que el entorno activo en el `PATH`, y entonces `python` deja de ser el del `.venv` (por ejemplo, `pytest` aparece como no instalado).

Si no se quiere instalar pyenv, se puede usar el `.venv` del proyecto con `--env-manager local` (ver [Entorno de ejecución](#entorno-de-ejecución)).

### Paso a paso

**1. Elegir el servidor de MLflow y exportar su URI.** El servidor del equipo es **https://mlflow.equipo7mlops.com/** (no pide credenciales) y es donde deben quedar los runs de la entrega. `mlflow run` lee la variable del shell, no del `.env`, así que se exporta en la terminal donde se lanzan los runs.

```bash
# Servidor del equipo (el de la entrega)
export MLFLOW_TRACKING_URI=https://mlflow.equipo7mlops.com
```

Si el servidor del equipo no responde, se puede usar uno local de respaldo. En otra terminal se levanta y se deja corriendo; en la terminal de los runs se apunta a él:

```bash
# Terminal 1: servidor local con base de datos
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --port 5000

# Terminal 2: la terminal donde se lanzan los runs
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

`--artifacts-destination` es necesario en el servidor local: sin él, MLflow 3 no puede subir los artefactos (figuras y modelo).

**2. Preparar datos y features** (`ingest`). Descarga, normaliza y limpia los viajes y construye las features de P1, P2 y P3. Es el equivalente de `make data` más `make features`, y se corre una vez por clon. Si `data/processed/` ya tiene las tablas de la pregunta que se va a entrenar, se puede omitir. Para la entrega, desde un clon limpio, se corre con todo:

```bash
mlflow run . -e ingest --experiment-name mibici-equipo07
```

**3. Entrenar y registrar** (`train`). Cada comando entrena el modelo elegido y deja sus runs en el experimento `mibici-equipo07` del servidor al que apunta `MLFLOW_TRACKING_URI`.

```bash
# P1: un modelo con hiperparámetros
mlflow run . -e train -P question=p1 -P model=xgb -P n_estimators=300 --experiment-name mibici-equipo07

# P1: los demás modelos (baseline, hgb)
mlflow run . -e train -P question=p1 -P model=baseline --experiment-name mibici-equipo07
mlflow run . -e train -P question=p1 -P model=hgb --experiment-name mibici-equipo07

# P2: riesgo de estación (baseline y hgb se entrenan juntos)
mlflow run . -e train -P question=p2 -P model=hgb --experiment-name mibici-equipo07
mlflow run . -e train -P question=p2 -P model=hgb -P min_samples_leaf=500 --experiment-name mibici-equipo07

# P3: clustering de estaciones (baseline y alternativo; sin `model` corre los dos)
mlflow run . -e train -P question=p3 -P model=baseline --experiment-name mibici-equipo07
mlflow run . -e train -P question=p3 -P model=alternativo --experiment-name mibici-equipo07
```

**4. Buscar hiperparámetros** (`tune`, P1 y P3; P2 todavía no). Corre una búsqueda con Optuna: un run padre, un run hijo por trial y, al final, el mejor trial re-entrenado como candidato (ver [Fase 2 de P1](#fase-2-búsqueda-de-hiperparámetros-selección-y-registro) y [Búsqueda con Optuna (P3)](#búsqueda-con-optuna-p3)).

```bash
# P1
mlflow run . -e tune -P question=p1 -P model=xgb --experiment-name mibici-equipo07

# P3 (el modelo es `alternativo`; `fraccion_train` no aplica)
mlflow run . -e tune -P question=p3 -P model=alternativo -P n_trials=60 --experiment-name mibici-equipo07
```

**5. Registrar el campeón** (`register`, P1 y P3; P2 todavía no). Registra el modelo de un run como versión nueva en el Model Registry y le pone un alias (`champion` por defecto). El run se elige en la interfaz y se pasa su ID:

```bash
# P1
mlflow run . -e register -P question=p1 -P run_id=ID_DEL_RUN --experiment-name mibici-equipo07

# P3 (ver Model Registry (P3)); con -P alias=challenger queda registrado sin reemplazar al campeón
mlflow run . -e register -P question=p3 -P run_id=ID_DEL_RUN --experiment-name mibici-equipo07
```

### Puntos de entrada

| Punto de entrada | Qué ejecuta | Parámetros |
|---|---|---|
| `ingest` | `dataset todo`, `p1-features`, `p2-features`, `p3-particionar` y `p3-features` | Ninguno: usa el rango de `config.py` |
| `train` | `modeling/run.py`, que llama al comando de `modeling/train.py` de la pregunta y siempre registra en MLflow | Ver la tabla siguiente |
| `tune` | `modeling/tuning_<question>.py`: búsqueda de hiperparámetros con Optuna y runs anidados, y deja el mejor como candidato | `question` (`p1` o `p3`), `model` y, opcionales, `n_trials`, `timeout_min` y `fraccion_train` (solo P1); vacío = `P1_BUSQUEDA` o `P3_BUSQUEDA` de `config.py` |
| `register` | `modeling/registro_<question>.py`: registra el modelo de un run en el Model Registry con un alias | `question`, `run_id` y `alias` (por defecto `champion`) |

`tune` y `register` arman el módulo con el nombre de la pregunta, así que una pregunta nueva solo agrega su `tuning_pN.py` o `registro_pN.py`, sin tocar el `MLproject`.

### Parámetros de `train`

Un parámetro vacío (el valor por defecto) conserva el valor de `config.py`. `question` es el único obligatorio.

| Parámetro | Valores | Aplica a |
|---|---|---|
| `question` | `p1`, `p2`, `p3` | todos |
| `model` | P1: `baseline`, `hgb`, `xgb`. P2: `baseline`, `hgb`. P3: `baseline`, `alternativo`. Vacío = todos los de la pregunta | todos |
| `n_estimators` | entero (en HGB se traduce a `max_iter`) | P1 y P2 `hgb`; P1 `xgb` |
| `learning_rate` | decimal | P1 y P2 `hgb`; P1 `xgb` |
| `max_depth` | entero | P1 y P2 `hgb`; P1 `xgb` |
| `min_samples_leaf` | entero | P1 y P2 `hgb` (no `xgb`) |

- Un hiperparámetro que no aplica al modelo elegido se rechaza con un error claro; no se ignora en silencio.
- P2 entrena el baseline y el HGB en el mismo comando, así que `model` solo se valida.
- P3 no recibe hiperparámetros por esta vía: se ajustan con las variables `P3_*` (ver la tabla de variables de entorno).
- Los valores que se pasan quedan registrados como parámetros del run (por ejemplo `modelo_n_estimators` en `xgb` o `modelo_max_iter` en `hgb`).

### Entorno de ejecución

| Modo | Comando | Cuándo usarlo |
|---|---|---|
| Entorno aislado (por defecto) | `mlflow run . -e train ...` | Entrega y verificación desde un clon limpio. La primera vez tarda varios minutos: instala Python 3.11 y todo `requirements.txt`. Las siguientes reutilizan el entorno en `~/.mlflow/envs/` |
| Entorno activo | `source .venv/bin/activate` y `mlflow run . -e train ... --env-manager local` | Iterar rápido. Debe ser el `.venv` del proyecto (Python 3.11); no sirve el `base` de conda |

### Qué deja cada corrida

- **Runs del modelo** en el experimento `mibici-equipo07`, con parámetros, métricas, etiquetas (`equipo`, `pregunta`, `commit`; en la Fase 2, P1 usa `question`, `stage_of_work`, `model_family` y `fase`), artefactos y el modelo. Sus nombres son los de la sección "MLflow del equipo" más abajo.
- **Un run contenedor** del proyecto (nombre aleatorio, por ejemplo `silent-ray-926`), sin métricas: guarda los parámetros de `mlflow run`. Sale en el experimento que se pasa con `--experiment-name` (sin la opción, en `Default`).
- **Archivos regenerados** en `reports/figures/`. Están versionados, así que conviene revisar `git status` y revertir con `git checkout -- reports/figures` lo que no se quiera commitear.

### Problemas frecuentes

| Mensaje | Causa | Solución |
|---|---|---|
| `FileNotFoundError: ... 'None'` en `_install_python` | pyenv no está instalado y MLflow no puede instalar Python 3.11 | Instalar pyenv (ver Requisitos), o `--env-manager local` con el `.venv` activo |
| `Falta p2_cambios_estacion_bin.parquet` (o `p1_demanda...`, `viajes_modelado...`) | No se han generado las features de esa pregunta | `mlflow run . -e ingest ...` o `make features-p1`, `make features-p2`, `make features-p3` |
| `MLFLOW_TRACKING_URI no está definida` | La variable no está en el shell donde se lanzó | `export MLFLOW_TRACKING_URI=...` (paso 1) |
| `When an mlflow-artifacts URI was supplied, the tracking URI must be ...` | Se usó `sqlite:///...` directo como URI, sin servidor | Levantar `mlflow server` y usar su URL `http://127.0.0.1:5000` |
| `The filesystem tracking backend ... is in maintenance mode` | Se usó una carpeta (`./mlruns`) como backend | Usar el servidor del equipo o el servidor local con SQLite |
| `Invalid value: ... no aplica a p1/xgb` | Un hiperparámetro no corresponde al modelo | Quitarlo o cambiar de modelo (ver la tabla de parámetros) |

Un run fallido deja un run contenedor en estado FAILED en el servidor: se puede borrar desde la interfaz (menú del run, *Delete*).

## Propósito

Responder con ML tres preguntas operativas de MiBici, sobre una base reproducible (ingesta versionada, features reutilizables y experimentos registrados en MLflow) que las siguientes entregas del curso extenderán.

## Preguntas del proyecto

| # | Pregunta de negocio | Tarea | Usuario |
|---|---|---|---|
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Operador del sistema (planeación de flota) |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes? | Clasificación | Equipo de rebalanceo |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | Clustering | Planeación urbana y expansión |

> Los datos no incluyen inventario de bicicletas por estación; para P2 el riesgo se aproxima con el cambio neto de viajes (llegadas − salidas) en las próximas 2 horas frente a la capacidad de la estación.

Detalle de cada pregunta (qué se predice, relevancia y decisión que habilita): [ML Canvas](references/ml-canvas.md).

## Datos

### Fuente

[Datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/), sistema de bicicleta pública del Área Metropolitana de Guadalajara. El portal publica **solo CSV**:

- **Viajes:** `datos_abiertos_AAAA_MM.csv`, un archivo por mes desde dic-2014 (140 archivos publicados al 15-oct-2026).
- **Estaciones:** `nomenclatura_AAAA_MM.csv`, una sola versión vigente (`nomenclatura_2026_09.csv`).
- Los datos **no se versionan en Git** (ver `.gitignore`); se reconstruyen desde el portal y se convierten a Parquet (ver `dataset.py` y los notebooks `0.02-dsh` y `0.01-lasd`).
- **Enlaces, registros y SHA-256 de cada archivo:** [`references/fuentes-datos.csv`](references/fuentes-datos.csv).

### Alcance del análisis

| Concepto | Valor |
|---|---|
| Rango acordado por el equipo | enero 2020 a septiembre 2026 |
| Viajes | **27,086,557** en 81 archivos mensuales |
| Primer y último inicio de viaje | 2020-01-01 06:02:20 a 2026-09-30 23:59:48 |
| Usuarios distintos | 81,245 |
| Estaciones en los viajes | 475 (de 484 en la nomenclatura) |
| Mes con menos viajes | 2020-04 (129,276), inicio de la pandemia de COVID-19 |
| Mes con más viajes | 2026-09 (490,056) |
| Tamaño | 2.3 GB en CSV, 0.6 GB en Parquet |

> **Meses 2023-04 y 2023-05:** desaparecieron del portal sin aviso después del 13-oct-2026 y luego se volvieron a publicar; hoy el rango está completo (81 de 81 meses). El manifiesto `references/fuentes-datos.csv` fija el número de registros y el SHA-256 de cada archivo, así que un archivo distinto se detecta en la ingesta.

Viajes por año (2026 llega solo hasta septiembre):

| 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---:|---:|---:|---:|---:|---:|---:|
| 2,866,072 | 3,184,410 | 4,202,942 | 4,293,115 | 4,645,921 | 4,532,032 | 3,362,065 |

### Esquema de viajes

| Columna (CSV original) | Columna en Parquet | Tipo | Nulos | Descripción |
|---|---|---|---:|---|
| `Viaje_Id` | `viaje_id` | entero | 0 | Identificador del viaje |
| `Usuario_Id` | `usuario_id` | entero | 0 | Identificador del usuario (anonimizado) |
| `Genero` | `genero` | categoría (`M`, `F`) | 135,232 (0.51 %) | Género del usuario |
| `Año_de_nacimiento` | `anio_de_nacimiento` | entero | 33,796 (0.13 %) | Año de nacimiento |
| `Inicio_del_viaje` | `inicio_del_viaje` | fecha y hora | 0 | Inicio del viaje |
| `Fin_del_viaje` | `fin_del_viaje` | fecha y hora | 0 | Fin del viaje |
| `Origen_Id` | `origen_id` | entero | 0 | Estación de origen |
| `Destino_Id` | `destino_id` | entero | 0 | Estación de destino |

### Esquema de estaciones

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | entero | Identificador de la estación (único, de 2 a 519) |
| `name` | texto | Código y cruce de calles, por ejemplo `(GDL-001) C. Epigmenio Glez./ Av. 16 de Sept.` |
| `latitude`, `longitude` | decimal | Ubicación (entre 20.636 y 20.740 N, y entre -103.433 y -103.297 O) |
| `dpcapacity` | entero | Capacidad en anclajes; promedio 16.9, de 0 a 39, **8,171 anclajes en total** |

La nomenclatura tiene **484 estaciones** sin nulos.

### Hallazgos de calidad

Detectados en el EDA. El tratamiento aplicado a cada uno se documenta en *Limpieza y partición*, más abajo.

| Hallazgo | Evidencia | Notebook |
|---|---|---|
| **Nombre de columna corrupto** | 2024-02 y 2024-05 traen `A}äe_nacimiento` en lugar de `Año_de_nacimiento` | 0.02-dsh |
| **Codificación mezclada** | 43 archivos (53.1 %) en latin-1 y 38 (46.9 %) en UTF-8, sin regla por fecha | 0.02-dsh |
| **Formatos de fecha mezclados** | 78 archivos `AAAA-MM-DD hh:mm:ss`; 2023-11, 2024-09 y 2024-10 usan `DD/MM/AAAA hh:mm` | 0.02-dsh |
| **Viajes duplicados** | 758 filas completamente repetidas (1,516 filas comparten un `viaje_id` repetido) | 0.02-dsh, 0.02-lasd |
| **Años de nacimiento imposibles** | 1,255 anteriores a 1900 (mínimo 1) y 2,101 posteriores a 2010 (máximo 2023) | 0.01-fat |
| **Duración cero** | 55,035 viajes con inicio igual a fin (55,034 después de quitar duplicados) | 0.02-lasd |
| **Duración extrema** | 159 viajes de más de 24 horas (máximo 469 horas, cerca de 20 días); la mediana es 9.4 min y el percentil 99, 32.8 min | 0.01-fat, 0.02-lasd |
| **Estaciones sin nomenclatura** | 3 ids (124, 405 y 407) aparecen en viajes y no en la nomenclatura: afecta a 3,961 viajes | 0.01-fat, 0.02-lasd |
| **Estaciones sin capacidad** | 4 estaciones con `dpcapacity = 0`, que afecta el cálculo de riesgo de P2 | 0.02-lasd |
| **Nomenclatura única** | Solo existe la versión 2026-09; estaciones retiradas o reubicadas no se pueden reconstruir para años anteriores | 0.02-dsh |
| **Sin servicio de madrugada** | Solo 194 viajes entre la 1:00 y las 4:59 (193 después de la limpieza); la hora 0 tiene 145,857 (0.5 %) | 0.02-lasd |
| **Estaciones nuevas** | 109 estaciones no tienen viajes en entrenamiento y aparecen en prueba (3.8 % de sus viajes); 12 de la nomenclatura no registran ninguno | 0.03-lasd |
| **Capacidad sin relación con la demanda** | Correlación de Spearman entre salidas y capacidad de −0.02 | 0.02-lasd |
| **Demanda concentrada** | El 10 % de las estaciones (48) concentra el 37.8 % de los viajes; 74 tienen menos de 1,000 salidas | 0.02-lasd |
| **Sin inventario de bicicletas** | Los datos no registran bicis por estación (ver nota en Preguntas) | — |

### Privacidad

Los viajes incluyen `usuario_id`, género y año de nacimiento. Son datos personales aunque estén anonimizados; no se publican valores individuales en los notebooks ni en este README, solo agregados.

Rango de fechas y su justificación, variables y construcción de cada objetivo, condiciones de uso publicadas por MiBici y estrategia de partición temporal: [Definición de los datos](references/definicion-datos.md).

### Patrones del EDA

| Tema | Hallazgo |
|---|---|
| **Tendencia** | El promedio mensual crece de 238,839 viajes (2020) a 387,160 (2024) y se estabiliza: 377,669 en 2025 y 373,563 en 2026 |
| **Pandemia** | Cae de 367,864 viajes (feb-2020) a 129,276 (abr-2020), −64.9 %; el nivel de feb-2020 se supera hasta mar-2022 |
| **Días laborales** | Dos picos: 8:00 (8.3 % de los viajes) y 18:00 (10.5 %) |
| **Fin de semana** | Un solo pico amplio entre 11:00 y 14:00; sábado y domingo concentran el 20.4 % de los viajes |
| **Desbalance** | Acumulan bicicletas estaciones del centro (GDL-049, +148,627) y pierden las de Av. Vallarta (GDL-127, −98,030) |
| **Máximo reciente** | 2026-09 (490,056) es el máximo de la serie y +30 % contra 2025-09; cae en prueba, así que se vigila al interpretar el error |

### Limpieza y partición

Reglas aplicadas por `dataset.limpiar` y documentadas en `0.03-lasd-limpieza-p3`, cada una con su justificación en el notebook:

| Hallazgo | Decisión | Registros |
|---|---|---:|
| Viajes con `viaje_id` repetido (filas idénticas) | Eliminar | 758 |
| Origen o destino sin nomenclatura | Excluir | 3,961 |
| Duración cero | Excluir | 55,034 |
| Duración mayor a 24 h | Excluir | 159 |
| `usuario_id`, `genero`, `anio_de_nacimiento` | No entran a ninguna tabla de modelado (minimización): P1 y P2 leen solo las columnas que usan y P3 parte de `viajes_modelado`, que no los incluye | — |
| Meses sin datos (hoy ninguno: el portal volvió a publicar 2023-04 y 2023-05); horas 1:00 a 4:59 | No imputar; excluir de la tabla de demanda | — |

**Resultado:** 59,912 registros retirados (0.221 %); quedan **27,026,645 viajes** en `data/processed/viajes_limpios_2020-01_2026-09.parquet`. Para P3, `dataset p3-particionar` deja `data/processed/viajes_modelado_2020-01_2026-09.parquet` solo con las columnas que usan las preguntas (sin `usuario_id`, género ni año de nacimiento) y con la partición.

**Partición temporal** con una brecha de 24 horas antes de cada frontera, para que ninguna etiqueta use datos del conjunto siguiente:

| Conjunto | Periodo | Viajes | % |
|---|---|---:|---:|
| train | 2020-01-01 a 2024-09-29 | 17,987,077 | 66.6 |
| valid | 2024-10-01 a 2025-09-29 | 4,520,744 | 16.7 |
| test | 2025-10-01 a 2026-09-30 | 4,489,362 | 16.6 |
| brecha | 24 h antes de cada frontera | 29,462 | 0.1 |

**Preprocesamiento de P3:** `data/processed/estaciones_train.parquet` (484 estaciones, calculada solo con entrenamiento) y `data/processed/demanda_estacion_hora.parquet` (16,179,853 filas para 472 estaciones; 44.2 % de las estación-horas sin viajes), generadas por `features p3-features`.

## Arquitectura

![Arquitectura del sistema](references/arquitectura.png)

| Componente | Herramienta | Justificación | Lectura |
|---|---|---|---|
| Formulación | ML Canvas | Fija para cada pregunta qué se predice, para quién y qué decisión habilita, antes de escribir código | Dorard (2018) |
| Ingesta (CLI) | Python + Typer + loguru (`dataset.py`) | Un comando reconstruye los mismos datos en cualquier computadora, sin pasos manuales | Treveil et al. (2020) |
| Almacenamiento | Parquet (pyarrow) | Formato columnar y comprimido: se leen solo las columnas que usa cada pregunta sobre 27 millones de viajes | Huyen (2022) |
| Versionado de datos | Manifiesto CSV (`references/fuentes-datos.csv`) + `hashlib`, versionado en Git | Registros y SHA-256 por archivo; su hash es el `data_version` que se registra en cada run | Huyen (2022) |
| Procesamiento y features | pandas + scikit-learn (`features.py`) | Suficiente para agregados por estación y hora; los transformers aprenden solo con train, sin fuga | Huyen (2022) |
| Modelado | scikit-learn y XGBoost (`modeling/`) | Un baseline sencillo por pregunta antes de los candidatos, para medir cuánto aporta cada modelo | Huyen (2022) |
| Experimentos | MLflow Tracking, servidor del equipo (https://mlflow.equipo7mlops.com) | Parámetros, métricas, etiquetas (equipo, pregunta, commit), artefactos y modelo en un solo lugar compartido | Treveil et al. (2020); MLflow (s.f.-c) |
| Código y colaboración | Cookiecutter Data Science + Git/GitHub (ramas, PR y revisión) | Estructura estándar y cambios revisados antes de integrarse | Wilson (2022) |
| Automatización | Makefile + GitHub Actions | Calidad (ruff y pytest) en cada PR y entrenamiento de cada pregunta con un comando | Treveil et al. (2020) |

Etapas por pregunta (limpieza, agregaciones, features, entrenamiento y evaluación): [Diseño del pipeline de ML](references/pipeline-ml.md).

## P1: demanda de viajes por estación-hora (regresión)

Notebooks `0.02-dsh-exploracion-carga` (ingesta) y `0.04-dsh-eda-p1` (EDA de P1). Código: funciones `p1_*` y transformer `P1NivelEstacion` en `features.py`, sección P1 de `modeling/train.py`, `p1_predecir` y `p1_predecir_por_zona` en `modeling/predict.py` y gráficas `p1_*` en `plots.py`; pruebas en `tests/test_p1_*.py`.

| Elemento | Decisión |
|---|---|
| **Qué se predice** | Viajes que inician en cada estación por hora de servicio (5:00 a 0:59; de 1:00 a 4:59 el sistema no opera) |
| **Horizonte** | Día siguiente: al cierre del día D se predicen las 20 horas de D+1, así que todas las variables usan datos de 1 día o más atrás |
| **Zona** | Municipio (GDL, ZPN o TLQ, del código en el nombre de la estación). La predicción por zona es la suma de sus estaciones, así las dos vistas son consistentes |
| **Tabla** | 16,103,922 estación-horas con 27,026,452 viajes (los 193 viajes en horas sin servicio quedan fuera) |
| **Partición** | La común del proyecto: train 10,531,227 filas, valid 2,673,888 y test 2,884,207 |
| **Métricas** | MAE como principal (error en viajes por estación-hora, directo para el operador) y RMSE como complementaria (castiga más los errores en las horas pico). Sin MAPE: 44 % de las estación-horas no tienen viajes |

**Variables (14):** `origen_id` (solo para aprender el nivel de cada estación), viajes de la misma hora 1 y 7 días antes, promedio de la misma hora en las 4 semanas anteriores, hora, día de la semana, fin de semana, mes, festivo, capacidad, latitud, longitud, zona y días desde la apertura de la estación.

**Modelos** (los tres en un `Pipeline` de scikit-learn, con la misma interfaz `fit`/`predict`):

| Rol | Modelo | Por qué |
|---|---|---|
| Baseline | Promedio de la misma hora en las 4 semanas anteriores; si la estación no tiene historia, su nivel de train o el del sistema | Es lo que haría un operador sin modelo; los candidatos deben superarlo |
| Candidato | HistGradientBoosting con pérdida Poisson | El target es un conteo con 44 % de ceros: la pérdida Poisson modela conteos y nunca predice viajes negativos. Usa histogramas (16M de filas en minutos) y maneja los rezagos vacíos sin imputar |
| Candidato | XGBoost con pérdida Poisson | Las mismas ventajas; sirve para comparar dos implementaciones de boosting |

Los candidatos reciben además el nivel histórico de cada estación (`P1NivelEstacion`: promedio de train por estación, día de la semana y hora, y el del sistema para las estaciones nuevas), aprendido solo con train.

### Resultados

Se elige por `valid_mae`; test solo se reporta.

| Modelo | valid_mae | valid_rmse | test_mae | test_rmse | Segundos de entrenamiento |
|---|---:|---:|---:|---:|---:|
| **XGBoost Poisson (elegido)** | **0.9761** | 1.5639 | **0.9952** | 1.6781 | 67.2 |
| HGB Poisson | 0.9928 | 1.6157 | 1.0082 | 1.7138 | 72.5 |
| Baseline promedio 4 semanas | 1.0189 | 1.6539 | 1.0146 | 1.7500 | 1.6 |

XGBoost reduce el MAE del baseline 4.2 % en valid y 1.9 % en test, y entrena en un tiempo similar al de HGB. Por zona (suma diaria de las estaciones, XGBoost en test):

| Zona | MAE diario | Viajes diarios promedio | Error relativo |
|---|---:|---:|---:|
| GDL | 1348.2 | 11012.2 | 12 % |
| ZPN | 156.0 | 1144.6 | 14 % |
| TLQ | 23.4 | 142.6 | 16 % |

**Hallazgos de la validación** (prueba en seco con los datos reales, antes de registrar):

1. HGB Poisson divergía: el valor crudo llegó a 2574 en la iteración 287 y las predicciones salieron infinitas. Se corrigió con `l2_regularization=1.0` y `min_samples_leaf=100`; XGBoost ya trae ese freno (`max_delta_step=0.7` por defecto con `count:poisson`).
2. Las estaciones nuevas (109 estaciones y 8.93 % de las filas de test; 9 y 1.80 % en valid) recibían un nivel vacío y XGBoost se disparaba en test (1.164). El transformer ahora les asigna el nivel del sistema. Se detectó al revisar test y se corrigió un error de manejo de datos, sin ajustar hiperparámetros.

### Registro en MLflow (Fase 1)

En la Fase 2 cambiaron las etiquetas y el orden de los artefactos, y se agregaron la búsqueda y el Model Registry: ver [Fase 2](#fase-2-búsqueda-de-hiperparámetros-selección-y-registro).

- **Comando:** `P1_REGISTRAR_MLFLOW=true make train-p1` (o `python -m tc5061_proyecto_equipo07.modeling.train p1-entrenar --registrar`).
- **Experimento:** `mibici-equipo07`, un run por modelo: `p1-baseline-promedio-4sem`, `p1-hgb-poisson` y `p1-xgb-poisson`.
- **Etiquetas:** equipo, pregunta (`P1`), commit, autor, `stage` (baseline o candidato) y `model_family`, más la descripción del run.
- **Parámetros:** rango de datos, `data_version`, semilla, granularidad, horizonte, fronteras de la partición, brecha, número de variables y de filas de train, y los hiperparámetros de cada modelo.
- **Métricas:** `valid_mae`, `valid_rmse`, `test_mae`, `test_rmse`, `segundos_entrenamiento` y, en los candidatos, el MAE de train y valid por iteración.
- **Artefactos (`p1/`):** MAE por hora y por zona, peores estaciones en test, real contra predicción, serie diaria y, en los candidatos, curva de aprendizaje e importancia por permutación. Copia en `reports/figures/p1_*`.
- **Modelo:** el `Pipeline` completo con firma e `input_example`.

Runs de la entrega (commit `59df230`):

| Run | Run ID |
|---|---|
| `p1-baseline-promedio-4sem` | `b31814333aea4a9984401276c1caf850` |
| `p1-hgb-poisson` | `758a7fb096f9497a8d5dc2ed1224c575` |
| `p1-xgb-poisson` | `0cd6ae8976a54e77b71d48f94afaa60d` |

### Fase 2: búsqueda de hiperparámetros, selección y registro

Todo se corre con `mlflow run` desde el `MLproject` (puntos de entrada `train`, `tune` y `register`; ver [Ejecución con MLflow Project](#ejecución-con-mlflow-project)) y queda en el servidor del equipo (MLflow, s.f.-b).

**Búsqueda con Optuna.** Se usa Optuna (sampler TPE, semilla 42; Akiba et al., 2019) en lugar de una búsqueda en malla porque cada ajuste con todo train (10.5 M de filas) cuesta minutos: con 16 CPUs, XGBoost tarda unos 2 min y HGB unos 16 min. TPE usa los resultados de los trials anteriores para proponer los siguientes, así que llega a buenas combinaciones con menos ajustes.

| | XGBoost | HGB |
|---|---|---|
| Trials (máximo) | 20 | 10 |
| Train por trial | 100 % | Muestra al azar del 15 % |
| Tope de tiempo | 60 min | 45 min |
| Hiperparámetros | `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda` | `max_iter`, `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization` |

Los límites están en `P1_BUSQUEDA` (`config.py`) y los rangos en `modeling/tuning_p1.py`; los dos primeros se pueden cambiar en la línea de comandos (`-P n_trials=40 -P timeout_min=120`). En HGB solo se muestrea train: valid queda completo y en orden temporal, y el mejor trial se vuelve a entrenar con todo train. Los rangos de HGB tienen piso en `l2_regularization` y `min_samples_leaf` porque sin ellos la pérdida Poisson diverge (hallazgo 1 de la validación).

Antes de fijar los límites se intentó estimar el tiempo con una prueba en seco sobre el 10 % de train: extrapolarla de forma lineal daba unos 10 min por ajuste de XGBoost, pero con todo train tardó 2 min. Por eso los límites salen de tiempos medidos con todo train ([método y script](references/tiempos-p1.md)).

**Estructura en MLflow:**

- Run padre `p1-<modelo>-optuna-busqueda` (`stage_of_work=search`): parámetros de la búsqueda (`n_trials_max`, `timeout_min`, `fraccion_train`, `data_version`), el mejor resultado (`mejor_*`, `mejor_valid_mae`) y `best_child_run_id`.
- Un run hijo `trial_N` por trial (anidado, `stage_of_work=search`) con sus hiperparámetros, `valid_mae`, `valid_rmse` y segundos de ajuste.
- El candidato `p1-<modelo>-optuna` (`stage_of_work=candidate`): el mejor trial re-entrenado con todo train, con los mismos artefactos que los demás candidatos y la etiqueta `busqueda_run_id` hacia su búsqueda.

El objetivo de Optuna es `valid_mae`; test no interviene en ninguna decisión.

**Reproducibilidad.** Con la semilla fija, TPE propone la misma secuencia de hiperparámetros: el `trial_0` salió idéntico en dos computadoras distintas. Dos límites: el tope de tiempo puede cambiar cuántos trials terminan, y el estudio vive en memoria, así que una búsqueda interrumpida no se puede continuar (sus trials terminados sí quedan en MLflow).

**Etiquetas de los runs de P1 en la Fase 2** (en la Fase 1 eran `pregunta` y `stage`):

| Etiqueta | Valores | Para qué |
|---|---|---|
| `equipo` | `07` | Identificar al equipo en el servidor compartido |
| `question` | `p1` | Filtrar los runs de la pregunta |
| `stage_of_work` | `baseline`, `search` o `candidate` | Separar el baseline, los runs de una búsqueda (padre e hijos) y los candidatos que compiten por el Registry |
| `model_family` | `promedio_historico`, `hist_gradient_boosting` o `xgboost` | Comparar por familia de modelo |
| `fase` | `2` | Distinguir los runs de esta entrega de los de la Fase 1 |
| `commit` | Hash corto del commit | Saber con qué código se ejecutó |
| `autor` | Integrante que lo ejecutó | Responsable del run |
| `busqueda_run_id` | ID del run padre | Solo en los candidatos de Optuna: liga el candidato con su búsqueda |
| `mlflow.note.content` | Texto | Descripción del run (caja *Description*) |

En la interfaz, por ejemplo, `tags.question = "p1" and tags.fase = "2" and tags.stage_of_work = "candidate"` deja solo los candidatos de esta entrega.

**Artefactos:** se ordenan en `plots/`, `tables/`, `data/` (manifiesto de fuentes) y `config/` (`config.py`, `MLproject`, `python_env.yaml`). Cada run registra además su dataset de entrada (`p1_train`, con su digest) y el modelo con firma e `input_example`.

#### Resultados de la Fase 2

Se elige por `valid_mae`; test solo se reporta. Las métricas de baseline, HGB y XGBoost coinciden con las de la Fase 1: `mlflow run` reproduce los resultados desde su entorno aislado. Los segundos de entrenamiento, en cambio, pueden variar entre tablas y corridas: los runs de P1 se ejecutaron en computadoras distintas según dónde se trabajó durante el día, y el tiempo depende del equipo; las métricas no.

| Modelo | valid_mae | valid_rmse | test_mae | test_rmse | Segundos de entrenamiento |
|---|---:|---:|---:|---:|---:|
| **XGBoost con Optuna, 20 trials (champion v1)** | **0.9635** | 1.5177 | **0.9863** | 1.6371 | 106.9 |
| XGBoost Poisson | 0.9761 | 1.5639 | 0.9952 | 1.6781 | 119.5 |
| HGB Poisson | 0.9928 | 1.6157 | 1.0082 | 1.7138 | 988.4 |
| Baseline promedio 4 semanas | 1.0189 | 1.6539 | 1.0146 | 1.7500 | 1.8 |

Por zona (suma diaria de las estaciones, en test):

| Zona | MAE diario XGBoost | MAE diario XGBoost con Optuna | Viajes diarios promedio | Error relativo (Optuna) |
|---|---:|---:|---:|---:|
| GDL | 1348.2 | 1009.9 | 11012.2 | 9 % |
| ZPN | 156.0 | 139.1 | 1144.6 | 12 % |
| TLQ | 23.4 | 27.2 | 142.6 | 19 % |

**Selección del campeón:** `p1-xgb-optuna` (20 trials).

- **Mejora sobre el baseline:** 5.4 % menos MAE en valid y 2.8 % en test; sobre XGBoost con los hiperparámetros por defecto, 1.3 % en valid.
- **Métrica de negocio:** el error diario por zona baja 25 % en GDL (de 12 % a 9 % de los viajes del día) y 11 % en ZPN. En TLQ sube de 23 a 27 viajes por día; es la zona con menos viajes (1 % del total).
- **Estabilidad:** test sigue la misma dirección que valid y el orden de los modelos es el mismo en los dos conjuntos.
- **Costo:** entrena en menos de 2 minutos, como el XGBoost por defecto, y la búsqueda completa de 20 trials tardó unos 35 minutos.

**Model Registry** (MLflow, s.f.-a). El modelo registrado es `mibici-equipo07-p1-demanda`. Cada versión apunta al modelo de un run y lleva en su descripción las métricas, y en sus etiquetas el linaje: `commit`, `data_version`, `source_run_id`, `busqueda_run_id`, `question`, `fase` y `valid_mae`. El alias `champion` señala la versión vigente.

| Versión | Run de origen | Alias |
|---|---|---|
| 1 | `p1-xgb-optuna`, 20 trials (`32b4517a11e54fd686b7f1bd7a84224d`) | `champion` |

`register` no elige el campeón: registra el run que se le indica. Para promover otro, se registra su run (queda como versión nueva y el alias se mueve); con `-P alias=challenger` queda registrado sin reemplazar al campeón. El modelo se carga por alias, así que quien lo usa no cambia su código cuando se promueve una versión nueva:

```python
import mlflow

modelo = mlflow.pyfunc.load_model("models:/mibici-equipo07-p1-demanda@champion")
```

Runs de la Fase 2:

| Run | Run ID | Commit |
|---|---|---|
| `p1-baseline-promedio-4sem` | `42c6908766a0477490fec3fc8dcca19e` | `076053e` |
| `p1-hgb-poisson` | `6d14e64520d4446c9a6ba09a3ac4f24c` | `076053e` |
| `p1-xgb-poisson` | `0090816c11a945809f14beaacbb5245a` | `076053e` |
| `p1-xgb-optuna-busqueda` (20 trials) | `d44b205545a6402392ee30ee63eca411` | `29d4d35` |
| `p1-xgb-optuna` (20 trials) | `32b4517a11e54fd686b7f1bd7a84224d` | `29d4d35` |

## P2: riesgo de estación (clasificación)

Notebooks `0.01-fat-exploracion-inicial-p2`, `0.02-fat-exploracion-y-limpieza-p2`, `0.03-fat-feature-engineering-p2` y `0.04-fat-model-training-registration-p2`. Código: funciones `p2_*` y `P2HistoriaHastaAyer` en `features.py`, sección P2 de `modeling/train.py` y `p2_decidir_clase` en `modeling/predict.py`.

| Elemento | Decisión |
|---|---|
| **Qué se predice** | Para cada estación y bloque de 30 minutos, si en las próximas 2 horas estará en riesgo de quedarse **sin bicis**, **sin anclajes** o **sin riesgo** (3 clases) |
| **Aproximación del riesgo** | Los datos no traen inventario. Se usa el cambio neto de viajes (llegadas − salidas) en la ventana futura frente a la capacidad: si alcanza o supera el 50 % de la capacidad (`P2_ALFA`) hay riesgo; si hay los dos eventos, gana el de mayor estrés relativo |
| **Partición** | La común del proyecto; validación solo para elegir los umbrales de decisión, test solo se reporta |
| **Variables** | Balance neto relativo de los últimos 1, 2, 4 y 8 bloques, salidas y entradas recientes, acumulado del día, capacidad, ubicación, hora cíclica, fin de semana y la frecuencia histórica de cada riesgo "hasta ayer" en la misma estación, tipo de día y hora |
| **Métricas** | Average precision (PR-AUC) por clase de riesgo y de riesgo en general, log-loss, y precisión, recall y F1 por clase con el umbral de cada clase que maximiza el F1 en validación |

| Rol | Modelo |
|---|---|
| Baseline | Frecuencia histórica de cada riesgo en los días previos de la misma estación, tipo de día y hora, suavizada hacia la frecuencia global |
| Candidato | HistGradientBoostingClassifier con las variables anteriores |

Resultados en test:

| Modelo | AP sin bicis | AP sin anclajes | AP riesgo | Recall sin bicis | Recall sin anclajes | F1 sin bicis | F1 sin anclajes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline histórico | 0.2453 | 0.3982 | 0.3329 | 0.4257 | 0.4371 | 0.2419 | 0.4125 |
| HistGradientBoosting | 0.2318 | 0.4029 | 0.3326 | 0.4436 | 0.4467 | 0.2782 | 0.4151 |

Comando: `make train-p2` (con `P2_REGISTRAR_MLFLOW=true` registra los runs `p2-baseline-historico` y `p2-hgb`). Figuras y comparación en `reports/figures/p2_*`.

## P3: clustering de estaciones

### Modelo base

Notebook `0.04-lasd-baseline-p3` (código en `modeling/train.py`). K-Means sobre variables estandarizadas, ajustado solo con el conjunto de entrenamiento.

| Elemento | Valor |
|---|---|
| **Estaciones agrupadas** | 362 (con al menos 1,000 salidas en entrenamiento) |
| **Variables** | 44: perfil horario laboral y de fin de semana (20 horas de servicio cada uno), volumen diario, balance neto sobre salidas, proporción de viajes circulares y duración mediana |
| **`k`** | 2, elegido por silueta entre 2 y 10 |
| **Silueta** | 0.145 (estructura débil: los grupos se solapan) |
| **Davies-Bouldin / Calinski-Harabasz** | 2.12 / 69.2 |
| **Estabilidad (ARI con el año de validación)** | 0.66; 34 de 359 estaciones (9.5 %) cambian de grupo |

| Grupo | Estaciones | Perfil | Volumen mediano |
|---|---:|---|---|
| 0 | 202 | Pico matutino a las 8:00 (10.7 % de las salidas laborales); más periférico (3.0 km del centro de la red); balance +2.9 % | 19.8 viajes por día |
| 1 | 160 | Pico vespertino a las 18:00 (12.6 %); más cerca del centro (2.2 km) y al poniente; tiende a perder bicicletas (−9.3 %) | 29.4 viajes por día |

**Límites:** con `k = 2` la granularidad no basta para decidir ampliaciones o reubicaciones, y la separación se explica casi solo por horario y volumen. Las 109 estaciones nuevas quedan fuera del agrupamiento. Las etiquetas 0 y 1 de K-Means son arbitrarias y pueden invertirse entre corridas. Siguientes pasos: HDBSCAN o mezclas gaussianas. La lectura "residencial frente a zonas de trabajo" es una hipótesis por verificar.

### Modelo alternativo de P3

Notebook `0.05-lasd-mejoramodelo-p3`. PCA sobre las mismas variables y K-Means con `k` de 3 a 6, con criterios de aceptación fijados antes de ver los resultados y cinco semillas por configuración. Resultado con los 81 meses: **no se acepta**.

| Criterio | Baseline (`k = 2`) | Alternativo (`k = 3`, 5 componentes) | ¿Cumple? |
|---|---|---|---|
| Silueta (espacio estandarizado del baseline) | 0.145 | 0.140 | No |
| Estabilidad (ARI con validación) | 0.656 | 0.676 | Sí |
| Estaciones del grupo más chico | 160 | 42 | Sí (mínimo 20) |

A igual `k` (`k = 3`) tampoco hay ventaja clara: el baseline logra ARI 0.689 y grupo mínimo de 68. El valor del alternativo es interpretativo: separa un perfil intermedio de 68 estaciones de uso mixto que el `k = 2` esconde.

### Registro en MLflow

- **Servidor:** la URL de `MLFLOW_TRACKING_URI` (https://mlflow.equipo7mlops.com/) · **Experimento:** `mibici-equipo07` (el de `config.py`) · **Runs:** `<MLFLOW_RUN_NAME>-baseline` y `<MLFLOW_RUN_NAME>-pca_kmeans`; los de la Fase 1 son `p3-baseline` y `p3-pca_kmeans` (sin la etiqueta `fase`, con las métricas sin prefijo), y los de la Fase 2 llevan los mismos nombres con `fase=2` (ver [Runs de la Fase 2](#runs-de-la-fase-2-de-p3)). La búsqueda con Optuna agrega `<MLFLOW_RUN_NAME>-pca_kmeans_optuna-busqueda` y el candidato `<MLFLOW_RUN_NAME>-pca_kmeans_optuna`.
- **Cuándo se registra:** solo con `P3_REGISTRAR_MLFLOW=true`; es idempotente (un run terminado con el mismo nombre no se duplica, y uno que quedó a medias no cuenta).
- **Registrado:** 14 parámetros, 11 métricas (con el conjunto en el nombre: `train_*` para silueta, Davies-Bouldin, Calinski-Harabasz, inercia y tamaños de grupo, medidos con el entrenamiento en el espacio estandarizado del baseline, y `valid_*` para la estabilidad, `valid_ari`, al reasignar las estaciones con el año de validación; P3 no usa test), etiquetas de la rúbrica (`equipo`, `question`, `commit`, `autor`, `stage_of_work`, `model_family`, `fase` y la descripción del run), la tabla de entrada con `log_input`, artefactos (asignación por estación, resumen de grupos, mapa y silueta por `k`) y el modelo (`StandardScaler` + `KMeans`) con firma e `input_example`.
- **Artefactos por carpeta:** `plots/` (figuras), `tables/` (CSV y JSON), `data/` (manifiesto de fuentes) y `config/` (`config.py`, `MLproject` y `python_env.yaml`).
- **Trazabilidad:** el parámetro `data_version` es `dataset.version_datos()` (hash de los SHA-256 de `references/fuentes-datos.csv`); en CI las etiquetas incluyen `github_run_id`, `github_actor`, `github_ref_name` y el commit.
- **Copia local:** los mismos artefactos se guardan en `reports/figures/p3_*` y sí se versionan en Git.

### Búsqueda con Optuna (P3)

Módulo `modeling/tuning_p3.py`. Sustituye la rejilla del modelo alternativo (48 configuraciones) por una búsqueda TPE más amplia: con o sin pesos por bloque, de 2 a 20 componentes y `k` de 2 a 8 (`P3_BUSQUEDA` en `config.py`).

```bash
export MLFLOW_TRACKING_URI=https://mlflow.equipo7mlops.com/
mlflow run . -e tune -P question=p3 -P model=alternativo -P n_trials=60 --experiment-name mibici-equipo07
```

- **Objetivo:** silueta promedio entre las semillas de búsqueda (`P3_SEMILLAS_BUSQUEDA`), medida en el espacio estandarizado del baseline, como en el resto de P3.
- **Restricciones:** son los criterios de aceptación fijados antes de ver resultados: ARI de estabilidad mayor o igual al del baseline y grupo más chico de al menos `P3_TAM_MIN_ACEPTABLE` estaciones. Optuna las trata como restricciones, no como penalización.
- **Selección:** entre las configuraciones evaluadas se aplica la misma regla de la rejilla (`p3_seleccionar_alternativo`): la de mayor silueta que cumple. Si ninguna cumple, se registra la mejor y queda como no aceptada.
- **Sin test:** la estabilidad se mide con el año de validación; P3 no usa test.
- **Runs:** un padre `pca_kmeans_optuna-busqueda` (`stage_of_work=search`), un hijo `trial_<n>` por trial con sus métricas y la etiqueta `cumple_criterios`, y el candidato `pca_kmeans_optuna` (`stage_of_work=candidate`) enlazado al padre con `busqueda_run_id`. Los artefactos del candidato se guardan como `reports/figures/p3_pca_optuna_*`.
- **Presupuesto:** `n_trials` y `timeout_min` vacíos usan `config.py` (60 trials y 20 minutos). `fraccion_train` no aplica a P3 y se rechaza. Con 60 trials basta (ver los resultados siguientes).
- **Reproducibilidad:** con la semilla fija y semillas fijas en cada trial, la secuencia es la misma entre corridas: los primeros 40 trials salieron idénticos (configuración y silueta) en dos corridas independientes, así que una búsqueda de N trials contiene a las de menos trials. TPE puede proponer de nuevo una configuración ya evaluada, y esa repetición se vuelve a calcular.

#### Resultados de la búsqueda

Se comparó el avance de la búsqueda a los 40, 60 y 80 trials, tomando los hijos de una sola corrida de 80 y aplicando la regla de selección del código (mayor silueta entre las que cumplen ARI y tamaño mínimo; en empate, menos componentes). El baseline tiene silueta 0.1449 y ARI 0.6561.

| Trials | Configuraciones únicas | Repetidas | Cumplen los criterios | Mejor configuración | Silueta | Δ vs baseline | ARI | Se halló en el trial |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 40 | 34 | 6 | 21 | sin pesos, 11 componentes, `k = 2` | 0.1454 | +0.0005 | 0.6726 | 16 |
| 60 | 48 | 12 | 30 | sin pesos, 10 componentes, `k = 2` | 0.1455 | +0.0006 | 0.6799 | 48 |
| 80 | 58 | 22 | 36 | sin pesos, 10 componentes, `k = 2` | 0.1455 | +0.0006 | 0.6799 | 48 |

- **Más trials no mejoran el resultado:** el mejor aparece en el trial 48 y los 32 siguientes no lo superan; con 80 trials, 22 de 80 (27 %) son configuraciones repetidas. El ajuste de los 80 trials tarda menos de 1 minuto, así que el criterio para elegir el presupuesto es la convergencia y no el tiempo.
- **El óptimo está en `k = 2`:** solo 2 de las configuraciones que cumplen los criterios superan al baseline, y las dos tienen `k = 2`. Con `k` de 3 a 8 ninguna lo supera, igual que en la rejilla.
- **La mejora es marginal:** +0.0005 o +0.0006 de silueta. Es del orden de la variación entre semillas (en el candidato de 60 trials, la silueta mínima entre semillas fue 0.145, igual que la del baseline), y con `k = 2` el modelo vuelve a dar dos grupos, con el más chico de 160 estaciones en el candidato de 40 trials y en el baseline. De hecho, con la semilla del proyecto (42) el candidato de 40 trials da exactamente las métricas del baseline (silueta 0.1449, ARI 0.6561): la ventaja solo aparece en el promedio de las 5 semillas de búsqueda (`reports/figures/p3_pca_optuna_comparacion.csv`). Se acepta porque el criterio, fijado antes de ver los resultados, es `silueta > baseline`; no se debe leer como una mejora de la segmentación. El valor adicional del alternativo sigue siendo el de la rejilla: con `k = 3` separa un perfil intermedio, pero no supera al baseline en silueta.

#### Runs de la Fase 2 de P3

Corridas desde el commit `4d40d14`: las cuatro primeras con `mlflow run` (`-P n_trials=40` en la búsqueda). Para verlas en la interfaz: `tags.question = "p3" and tags.fase = "2"`.

| Run | Run ID | Aceptado |
|---|---|---|
| `p3-baseline` | `956d6795a7c24d75a847c298da809ffe` | n/a (piso) |
| `p3-pca_kmeans` (rejilla) | `8790436a24c84340b769996251bd6940` | No |
| `p3-pca_kmeans_optuna-busqueda` (40 hijos) | `2c68e9b937d249a383ce49dd245e6646` | n/a |
| `p3-pca_kmeans_optuna` | `872f2218deaa422e80ea197c325f7a6d` | Sí (por 0.0005 de silueta) |
| `p3-n80-pca_kmeans_optuna-busqueda` (exploratorio: se lanzó con `python -m` y `--sin-candidato`; 80 hijos) | `c7ec821702c7464cb2e0f67979e25b51` | n/a |

Cada `mlflow run` deja además un run contenedor con nombre aleatorio (por ejemplo `awesome-hare-758`) y sin métricas; no tiene que ver con el modelo y se puede ignorar en la interfaz.


### Model Registry (P3)

Módulo `modeling/registro_p3.py`, mismo flujo que el de P1: se elige el run del modelo en la interfaz, se copia su ID y se registra con un alias. El punto de entrada `register` resuelve `registro_<question>`.

```bash
mlflow run . -e register -P question=p3 -P run_id=<id del run> -P alias=champion --experiment-name mibici-equipo07
```

- **Modelo registrado:** `mibici-equipo07-p3-segmentos` (`P3_MODELO_REGISTRADO` en `config.py`). Cada versión lleva en sus etiquetas el commit, `data_version`, `source_run_id`, `train_silueta`, `valid_ari`, `aceptado` y, si viene de una búsqueda, `busqueda_run_id`.
- **No decide el campeón:** registra el run que se le indica y comprueba que `models:/<nombre>@<alias>` carga y asigna grupos. Si el run quedó con `aceptado=false`, advierte: el baseline sigue siendo el piso y es el que corresponde a `champion`.
- **Solo runs con métricas por conjunto:** un run anterior a los prefijos `train_` y `valid_` se rechaza con un error claro y debe volver a registrarse.

Estado actual del Registry (`mibici-equipo07-p3-segmentos`):

| Versión | Alias | Run de origen | `train_silueta` | `valid_ari` | Aceptado |
|---:|---|---|---:|---:|---|
| 4 | `champion` | `p3-pca_kmeans_optuna` | 0.1454 | 0.6726 | Sí |
| 3 | `challenger` | `p3-baseline` | 0.1449 | 0.6561 | n/a |
| 1 y 2 | sin alias | `p3-prueba-baseline` y `p3-prueba-pca_kmeans_optuna` (corridas de prueba) | 0.1449 y 0.1455 | 0.6561 y 0.6799 | n/a y sí |

**Selección del campeón.** El campeón es el candidato de la búsqueda con Optuna porque cumple los tres criterios fijados de antemano (silueta mayor que el baseline, ARI de estabilidad mayor o igual y grupo mínimo de al menos 20 estaciones). La ventaja es de 0.0005 de silueta, así que en la práctica las dos versiones son equivalentes y el baseline queda como `challenger`. Si el equipo fija un margen mínimo de silueta para aceptar un modelo, esta promoción tendría que revisarse.

> **Al crear un experimento nuevo en este servidor, usar `artifact_location="mlflow-artifacts:/<nombre>"`.** Sin esa opción el servidor asigna una ruta local suya (`/home/ec2-user/mlflow/artifacts/...`) y la subida de artefactos falla con `Errno 45`. El servidor es MLflow 3.16.1 y el `requirements.txt` fija el cliente en 3.16.0.

## Estructura de carpetas

```
├── data/              <- (ignorado por Git)
│   ├── raw/           <- CSV originales del portal (viajes/ y estaciones/)
│   ├── external/      <- Reservada para datos de terceros (hoy vacía)
│   ├── interim/       <- viajes_2020-01_2026-09.parquet (normalizado, sin limpiar)
│   └── processed/     <- Viajes limpios y tablas de features de P1, P2 y P3 (.parquet)
├── docs/              <- Documentación (mkdocs)
├── models/            <- Modelos serializados (ignorado por Git)
├── notebooks/         <- Convención FASE.NUMERO-INICIALES-descripcion.ipynb
│   ├── 0.01-fat-exploracion-inicial-p2.ipynb           <- P2: EDA inicial (nulos, años de nacimiento, duración)
│   ├── 0.01-lasd-exploracion-carga.ipynb               <- P3: carga y descripción del dataset con dataset.py
│   ├── 0.02-dsh-exploracion-carga.ipynb                <- Ingesta: inventario por archivo, unión y Parquet
│   ├── 0.02-fat-exploracion-y-limpieza-p2.ipynb        <- P2: exploración y limpieza
│   ├── 0.02-lasd-eda-p3.ipynb                          <- P3: EDA (calidad, tiempo y estaciones)
│   ├── 0.03-fat-feature-engineering-p2.ipynb           <- P2: etiqueta de riesgo y variables
│   ├── 0.03-lasd-limpieza-p3.ipynb                     <- P3: limpieza, partición temporal y preprocesamiento
│   ├── 0.04-dsh-eda-p1.ipynb                           <- P1: EDA y formulación
│   ├── 0.04-fat-model-training-registration-p2.ipynb   <- P2: exploración del modelado
│   ├── 0.04-lasd-baseline-p3.ipynb                     <- P3: modelo base y registro en MLflow
│   └── 0.05-lasd-mejoramodelo-p3.ipynb                 <- P3: modelo alternativo y comparación
├── references/        <- Documentos de diseño
│   ├── ml-canvas.md          <- ML Canvas y descripción de P1, P2 y P3
│   ├── definicion-datos.md   <- Rango, fuente, variables, objetivos, condiciones de uso y partición
│   ├── pipeline-ml.md        <- Pipeline por pregunta: limpieza, features, entrenamiento y evaluación
│   ├── pipeline-ml.png       <- Diagrama del pipeline por pregunta (fuente editable: pipeline-ml.svg)
│   ├── fuentes-datos.csv     <- URLs, registros y SHA-256 de los archivos del portal
│   ├── tiempos-p1.md         <- Tiempos medidos de los modelos de P1 y límites de la búsqueda
│   └── arquitectura.png      <- Diagrama de arquitectura (fuente editable: arquitectura.svg)
├── reports/figures/   <- Gráficas y tablas de cada pregunta (p1_*, p2_*, p3_*)
├── tc5061_proyecto_equipo07/   <- Módulo: config, dataset, features, modeling/ (train, predict, run, tuning_p1, tuning_p3, registro_p1, registro_p3), plots
├── tests/             <- Pruebas (pytest) del módulo
├── .github/workflows/
│   ├── TC5061-Proyecto-Equipo07.yml   <- Calidad en los PR y acción que corre las 3 preguntas
│   ├── p1.yml                         <- Acción de P1: entrenamiento y registro en MLflow
│   ├── p2.yml                         <- Acción de P2: entrenamiento y registro en MLflow
│   └── p3.yml                         <- Acción de P3: entrenamiento y registro en MLflow
├── .env.example       <- URL de MLflow (copiar a .env, que no se versiona)
├── Makefile           <- Comandos del proyecto (make help)
├── MLproject          <- Puntos de entrada de `mlflow run` (ingest, train, tune, register)
├── python_env.yaml    <- Entorno aislado de `mlflow run` (Python 3.11 + requirements.txt)
├── pyproject.toml
└── requirements.txt   <- Dependencias con versiones fijadas
```

## Flujo de trabajo con Git (GitHub Flow)

- `main` está protegida: todo cambio entra por Pull Request con al menos una revisión aprobada.
- `EDA` es la rama de integración del equipo. Cada integrante trabaja en su rama (`feat/<tema>`, `fix/<tema>`, `docs/<tema>`) creada desde `EDA`, abre un PR hacia `EDA` y, al cerrar cada entrega, se abre un PR `EDA → main`.
- En la Fase 2 la rama de integración es `mlflow_run` (creada desde `main`): las ramas de cada pregunta (`tuning/p1`, `tuning/p2`, `tuning/p3`) salen de ella y vuelven por PR.
- Commits con [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/) en español, sin alcance entre paréntesis y mencionando la pregunta: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `build:`, `ci:`, `style:`, `chore:`.
- Cada integrante abre al menos un PR y revisa al menos uno de otro integrante.
- En cada PR, GitHub Actions corre `ruff` y `pytest` (`TC5061-Proyecto-Equipo07.yml`).
- El código entra a `EDA` siempre por PR. En el cierre de cada entrega, los ajustes de documentación (README, `references/` y diagramas) se suben directo a `EDA`.

## Buenas prácticas

| Práctica | Cómo se aplica |
|---|---|
| Un experimento por proyecto y runs con nombre descriptivo | Experimento `mibici-equipo07` (`EXPERIMENTO_MLFLOW` en `config.py`); cada run lleva el nombre del enfoque (por ejemplo, `p1-xgb-poisson`), nunca el nombre aleatorio |
| Registrar lo necesario para reproducir el run | Parámetros con el rango de fechas, `data_version` (hash del manifiesto), semilla, partición e hiperparámetros del modelo |
| Ligar el run con su código y su pregunta | Etiquetas `commit`, `pregunta` (`P1`, `P2`, `P3`) y `equipo` en cada run; en la Fase 2, P1 y P3 usan `question`, `stage_of_work` y `fase` |
| Búsqueda de hiperparámetros trazable | Un run padre por búsqueda y un run hijo por trial (Optuna con semilla fija); el candidato final apunta a su búsqueda con `busqueda_run_id` |
| Un modelo oficial por pregunta (hoy P1 y P3) | Model Registry con alias `champion`; cada versión lleva en sus etiquetas el commit, `data_version` y el run de origen |
| Métricas con el conjunto en el nombre | P1 y P2: `valid_mae`, `test_mae`, `test_ap_riesgo`, etc. P3: `train_silueta`, `valid_ari`, etc. (P3 no usa test) |
| Sin credenciales ni datos personales en MLflow | La URL del servidor vive en `.env` (no versionado) o en el secret de GitHub; `usuario_id`, género y año de nacimiento no entran a ninguna tabla de modelado |
| Elegir el modelo sin usar test | La selección se hace con el conjunto de validación; test solo se reporta |
| Notebooks solo para exploración | Nombrados `FASE.NUMERO-INICIALES-descripcion.ipynb`; la lógica final vive en el módulo y los runs se ejecutan con `mlflow run`, `make` o `python -m` |
| Configuración en un solo lugar | Rutas, rango de fechas, fronteras de la partición, semilla y parámetros en `config.py` |
| Dependencias fijas y entorno reproducible | `requirements.txt` con versiones fijadas; instalación desde un clon limpio en *Instalación* |
| Ningún dato, modelo, carpeta de MLflow ni credencial en Git | `.gitignore` excluye `data/`, `models/*`, `*.pkl`, `*.joblib`, `mlruns/`, `mlartifacts/`, `mlflow.db` y `.env` |
| Todo cambio entra a `main` por PR revisado | Rama protegida (ver *Flujo de trabajo con Git*) |

## Equipo

| Integrante | Matrícula | Rol |
|---|---|---|
| Daniel Sánchez Hernández | A01840324 | P1 (regresión de demanda); ingesta y exploración de carga |
| Fernando Arreola Treviño | A01840331 | P2 (riesgo de estación); EDA inicial de calidad de datos |
| Luis Arturo Sánchez Davila | A01840576 | P3 (clustering de estaciones); estructura del repositorio y CI |

## Referencias

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. En *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2623–2631). https://doi.org/10.1145/3292500.3330701

Anthropic. (2026). *Claude* (Claude Opus 5.5 y Claude Sonnet 5.5) [Modelo de lenguaje grande], utilizado como asistente de programación, depuración y redacción de la documentación. https://claude.ai/

Dorard, L. (2018, 9 de marzo). *From data to AI with the Machine Learning Canvas (Part I)*. Own Machine Learning (Medium). https://medium.com/louis-dorard/from-data-to-ai-with-the-machine-learning-canvas-part-i-d171b867b047

Huyen, C. (2022). *Designing machine learning systems: An iterative process for production-ready applications*. O'Reilly Media.

MiBici. (s.f.). *Datos abiertos* [Conjunto de datos]. Sistema de Bicicleta Pública del Área Metropolitana de Guadalajara. https://mibici.net/es/datos-abiertos/

MLflow. (s.f.-a). *ML Model Registry* [Documentación]. https://mlflow.org/docs/latest/ml/model-registry/

MLflow. (s.f.-b). *ML Projects* [Documentación]. https://mlflow.org/docs/latest/ml/projects/

MLflow. (s.f.-c). *MLflow Tracking* [Documentación]. https://mlflow.org/docs/latest/ml/tracking/

Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J., Lavoillotte, A., Miyazaki, M., & Heidmann, L. (2020). *Introducing MLOps: How to scale machine learning in the enterprise*. O'Reilly Media.

Wilson, B. (2022). *Machine learning engineering in action*. Manning Publications.

### Uso de herramientas de inteligencia artificial

Los tres integrantes usaron Claude (Anthropic, 2026) como asistente de programación y de redacción. La tabla resume, por integrante, la herramienta y el apoyo que se recibió.

| Integrante | Herramienta | Para qué se usó |
|---|---|---|
| Daniel Sánchez Hernández (P1) | Claude Code, con los modelos Claude Opus 5.5 y Claude Sonnet 5.5 | Revisión y depuración del código de P1 (features, modelado y registro en MLflow) y su reorganización en el módulo con el mismo patrón que P2 y P3; targets del Makefile y workflow de P1; pruebas y prueba en seco con datos sintéticos antes de cada cambio; actualización de la documentación del equipo (README, `references/` y diagramas) y del borrador del reporte. En la Fase 2: etiquetas y artefactos de los runs de P1, búsqueda con Optuna, registro en el Model Registry y la documentación de esa fase |
| Fernando Arreola Treviño (P2) | Claude Code, con el modelo Claude Sonnet 5.5 | Búsqueda de mejoras en la eficiencia de ejecución, importante porque el proceso es intensivo en memoria; reorganización de la lógica de los notebooks en el módulo para evitar conflictos con el código de los otros integrantes; corrección de ortografía y limpieza de código no utilizado en los notebooks |
| Luis Arturo Sánchez Davila (P3) | Claude Code, con el modelo Claude Sonnet 5.5 | Reorganización de la lógica de los notebooks de P3 en el módulo (funciones `p3_*`), con sus comandos, targets de `make`, pruebas y el workflow de GitHub Actions; redacción de la documentación (`README.md`, `notebooks/README.md`) y de las observaciones de los notebooks. En la Fase 2: etiquetas y artefactos de los runs de P3, métricas con el conjunto en el nombre, búsqueda con Optuna, registro en el Model Registry y la documentación de esa fase |

En las tres preguntas, la formulación del problema, la elección de variables, modelos y métricas, los criterios de aceptación, la ejecución con los datos reales, la validación de los resultados y los commits y Pull Requests son de cada integrante, que revisó y es responsable del contenido. Por acuerdo del equipo, esta declaración vive aquí y no en los mensajes de commit.
