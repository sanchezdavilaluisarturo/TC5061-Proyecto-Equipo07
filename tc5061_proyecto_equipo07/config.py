"""Configuración compartida del proyecto: rutas, fuente de datos, esquema de viajes y MLflow."""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Variables de entorno del archivo .env, si existe (por ejemplo, MLFLOW_TRACKING_URI)
load_dotenv()

# Rutas del proyecto (estructura CCDS)
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REFERENCES_DIR = PROJ_ROOT / "references"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Fuente de los datos
URL_PORTAL = "https://mibici.net/es/datos-abiertos/"
# El portal responde 406 al User-Agent por defecto de requests
ENCABEZADOS_HTTP = {
    "User-Agent": "Mozilla/5.0 (proyecto academico MLOps TC5061)",
    "Accept": "text/html,text/csv,*/*",
}

# Rango acordado por el equipo (meses inclusivos)
MES_INICIO = "2020-01"
MES_FIN = "2026-09"
# El portal retiró estos meses el 14-oct-2026 y los volvió a publicar el 15-oct-2026.
# Se toleran si vuelven a faltar; cualquier otro hueco detiene la ingesta.
MESES_SIN_PUBLICAR = ["2023-04", "2023-05"]

# Partición temporal común a P1, P2 y P3: train hasta 2024-09, valid de 2024-10 a 2025-09 y test
# de 2025-10 a 2026-09. Bloques de 12 meses para que valid y test cubran todas las temporadas.
# Las horas previas a cada frontera quedan en `brecha` (el horizonte de P1 es de 24 h) y no se usan.
FRONTERA_VALID = "2024-10-01"
FRONTERA_TEST = "2025-10-01"
BRECHA_HORAS = 24

# Archivos de datos
RUTA_VIAJES_CRUDOS = RAW_DATA_DIR / "viajes"
RUTA_ESTACIONES_CRUDAS = RAW_DATA_DIR / "estaciones"
RUTA_MANIFIESTO = REFERENCES_DIR / "fuentes-datos.csv"
RUTA_VIAJES_INTERIM = INTERIM_DATA_DIR / f"viajes_{MES_INICIO}_{MES_FIN}.parquet"
RUTA_VIAJES_LIMPIOS = PROCESSED_DATA_DIR / f"viajes_limpios_{MES_INICIO}_{MES_FIN}.parquet"

# Esquema de los viajes: encabezado original -> nombre normalizado.
# 2024-02 y 2024-05 traen la "ñ" dañada en el año de nacimiento.
COLUMNAS_NORMALIZADAS = {
    "Viaje_Id": "viaje_id",
    "Usuario_Id": "usuario_id",
    "Genero": "genero",
    "Año_de_nacimiento": "anio_de_nacimiento",
    "A}äe_nacimiento": "anio_de_nacimiento",
    "Inicio_del_viaje": "inicio_del_viaje",
    "Fin_del_viaje": "fin_del_viaje",
    "Origen_Id": "origen_id",
    "Destino_Id": "destino_id",
}
COLUMNAS_VIAJES = [
    "viaje_id",
    "usuario_id",
    "genero",
    "anio_de_nacimiento",
    "inicio_del_viaje",
    "fin_del_viaje",
    "origen_id",
    "destino_id",
]
TIPOS_ENTEROS = {
    "viaje_id": "Int64",
    "usuario_id": "Int64",
    "anio_de_nacimiento": "Int16",
    "origen_id": "Int32",
    "destino_id": "Int32",
}

# Limpieza común (ver dataset.limpiar)
DURACION_MAXIMA_MIN = 24 * 60
# Un año de nacimiento que implica un usuario de 5 años o menos se considera inválido
EDAD_MINIMA = 6

# Entre la 1:00 y las 4:59 el servicio no opera (194 viajes de 27086557 antes de limpiar)
HORAS_SIN_SERVICIO = [1, 2, 3, 4]

SEMILLA = 42

# MLflow: el experimento incluye el número de equipo; la URI viene del .env
EXPERIMENTO_MLFLOW = "mibici-equipo07"
# Etiqueta `fase` de los runs de esta entrega (la Fase 1 no la tiene)
FASE_ENTREGA = "2"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Paleta Okabe-Ito (apta para daltonismo) para las gráficas del proyecto
AZUL = "#0072B2"
NARANJA = "#E69F00"
VERDE = "#009E73"
AMARILLO = "#F0E442"
BERMELLON = "#D55E00"
GRIS = "#999999"

# ---------------------------------------------------------------------------------------------
# P3 (clustering de estaciones)
# Los parámetros se leen del entorno para que el mismo comando corra en la terminal, en los
# notebooks y en GitHub Actions (variables del repositorio). Si la variable no existe valen los
# defectos de abajo. Estas definiciones solo se agregan: nada de lo anterior cambia.
# ---------------------------------------------------------------------------------------------


def p3_leer_entero(nombre: str, defecto: int) -> int:
    """Regresa la variable de entorno `nombre` como entero, o `defecto` si no existe o está vacía."""
    valor = os.getenv(nombre, "").strip()
    return int(valor) if valor else defecto


def p3_leer_bool(nombre: str, defecto: bool = False) -> bool:
    """Regresa la variable de entorno `nombre` como booleano (1, true, yes, si)."""
    valor = os.getenv(nombre, "").strip().lower()
    if not valor:
        return defecto
    return valor in ("1", "true", "yes", "si", "sí")


def p3_leer_lista_enteros(nombre: str, defecto: list[int]) -> list[int]:
    """Regresa la variable de entorno `nombre` ('3,4,5') como lista de enteros, o `defecto`."""
    valor = os.getenv(nombre, "").strip()
    return [int(parte) for parte in valor.split(",") if parte.strip()] if valor else list(defecto)


# MLflow: la URL es MLFLOW_TRACKING_URI (arriba); el nombre del run es otra variable. Sin
# MLFLOW_RUN_NAME los runs se llaman `baseline` y `pca_kmeans`; con ella, `<valor>-baseline`, etc.
MLFLOW_RUN_NAME = os.getenv("MLFLOW_RUN_NAME") or None
P3_EXPERIMENTO = os.getenv("P3_EXPERIMENTO") or EXPERIMENTO_MLFLOW

P3_SEMILLA = p3_leer_entero("P3_SEMILLA", SEMILLA)
P3_K_MINIMO = p3_leer_entero("P3_K_MINIMO", 2)
P3_K_MAXIMO = p3_leer_entero("P3_K_MAXIMO", 10)
# Estaciones con menos salidas no se agrupan (train) ni entran a la prueba de estabilidad (valid)
P3_MIN_SALIDAS_TRAIN = p3_leer_entero("P3_MIN_SALIDAS_TRAIN", 1000)
P3_MIN_SALIDAS_VALID = p3_leer_entero("P3_MIN_SALIDAS_VALID", 300)

# Búsqueda del modelo alternativo y criterio de tamaño mínimo de grupo
P3_COMPONENTES = p3_leer_lista_enteros("P3_COMPONENTES", [3, 4, 5, 6, 8, 10])
P3_SEMILLAS_BUSQUEDA = p3_leer_entero("P3_SEMILLAS_BUSQUEDA", 5)
P3_TAM_MIN_ACEPTABLE = p3_leer_entero("P3_TAM_MIN_ACEPTABLE", 20)

# Comportamiento en CI: por defecto no se escribe en MLflow ni se falla si no se acepta el modelo
P3_REGISTRAR_MLFLOW = p3_leer_bool("P3_REGISTRAR_MLFLOW")
P3_REGISTRAR_DE_NUEVO = p3_leer_bool("P3_REGISTRAR_DE_NUEVO")
P3_EXIGIR_ACEPTACION = p3_leer_bool("P3_EXIGIR_ACEPTACION")

# Partición temporal: la común del proyecto (definida arriba)
P3_FRONTERA_VALID = FRONTERA_VALID
P3_FRONTERA_TEST = FRONTERA_TEST
P3_BRECHA_HORAS = BRECHA_HORAS

P3_HORAS_SERVICIO = [hora for hora in range(24) if hora not in HORAS_SIN_SERVICIO]
# Columnas que pasan al modelado: usuario_id, genero y anio_de_nacimiento no pasan (privacidad)
P3_COLUMNAS_MODELADO = [
    "viaje_id",
    "inicio_del_viaje",
    "fin_del_viaje",
    "origen_id",
    "destino_id",
    "duracion_min",
    "periodo_archivo",
]

# Archivos de P3 en data/processed/
RUTA_VIAJES_MODELADO = PROCESSED_DATA_DIR / f"viajes_modelado_{MES_INICIO}_{MES_FIN}.parquet"
RUTA_ESTACIONES_TRAIN = PROCESSED_DATA_DIR / "estaciones_train.parquet"
RUTA_DEMANDA_ESTACION_HORA = PROCESSED_DATA_DIR / "demanda_estacion_hora.parquet"


# P1 (regresión): viajes que inician en cada estación por hora de servicio, un día adelante.
# Los rezagos se miden en días completos (>= 1) para no usar información del día que se predice.
P1_REZAGOS_DIAS = [1, 7]
# Semanas anteriores que promedia la variable de demanda a la misma hora
P1_SEMANAS_PROMEDIO = 4
RUTA_P1_DEMANDA = PROCESSED_DATA_DIR / "p1_demanda_estacion_hora.parquet"

# Variables de entrada de los modelos de P1 (origen_id solo lo usa el transformer de nivel)
P1_VARIABLES = [
    "origen_id",
    "viajes_rezago_1d",
    "viajes_rezago_7d",
    f"viajes_promedio_{P1_SEMANAS_PROMEDIO}sem",
    "hora",
    "dia_semana",
    "es_fin_de_semana",
    "mes",
    "es_festivo",
    "capacidad",
    "latitud",
    "longitud",
    "zona",
    "dias_desde_apertura",
]
# Hiperparámetros iniciales fijos (sin ajuste). early_stopping=False: el corte de validación
# interno de HGB toma filas al azar y mezclaría el orden temporal de train.
# l2_regularization y min_samples_leaf frenan los pasos de Poisson en hojas casi sin viajes:
# sin ellos, con los datos reales la predicción se disparó a infinito en la iteración 287.
# (XGBoost ya trae ese freno: max_delta_step=0.7 por defecto con count:poisson.)
P1_PARAMETROS_HGB = {
    "loss": "poisson",
    "max_iter": 300,
    "learning_rate": 0.1,
    "max_leaf_nodes": 63,
    "l2_regularization": 1.0,
    "min_samples_leaf": 100,
    "early_stopping": False,
}
P1_PARAMETROS_XGB = {
    "objective": "count:poisson",
    "n_estimators": 300,
    "learning_rate": 0.1,
    "max_depth": 8,
    "tree_method": "hist",
}
# Filas que se muestrean para los diagnósticos caros (curva de aprendizaje e importancia)
P1_MUESTRA_DIAGNOSTICO = 200_000
# Nombre del run, etapa y familia de cada modelo (etiquetas `stage_of_work` y `model_family`)
P1_MODELOS = {
    "baseline": ("p1-baseline-promedio-4sem", "baseline", "promedio_historico"),
    "hgb": ("p1-hgb-poisson", "candidate", "hist_gradient_boosting"),
    "xgb": ("p1-xgb-poisson", "candidate", "xgboost"),
}

# Descripción de cada run en MLflow (caja "Description")
P1_DESCRIPCIONES = {
    "baseline": "Promedio de la misma hora en las 4 semanas anteriores; si la estación no tiene "
    "historia, usa su nivel de train o el del sistema. Referencia que los candidatos deben superar.",
    "hgb": "HistGradientBoosting con pérdida Poisson sobre el nivel histórico de cada estación "
    "aprendido en train; predice los viajes del día siguiente por estación-hora.",
    "xgb": "XGBoost con pérdida Poisson sobre el nivel histórico de cada estación aprendido en "
    "train; predice los viajes del día siguiente por estación-hora.",
}

# ---------------------------------------------------------------------------------------------
# P2 (clasificacion de riesgos)
# ---------------------------------------------------------------------------------------------

# Horas del día (0-23) en las que MiBici no ofrece servicio (las mismas que HORAS_SIN_SERVICIO)
P2_HORAS_SIN_SERVICIO = HORAS_SIN_SERVICIO

# Los viajes que terminan en esta fecha/hora o después quedan fuera del rango de análisis: es el
# primer instante del mes siguiente a MES_FIN (con MES_FIN = "2026-09", "2026-10-01 00:00:00")
_anio_fin, _mes_fin = map(int, MES_FIN.split("-"))
P2_FECHA_LIMITE_FIN_VIAJE = f"{_anio_fin + _mes_fin // 12}-{_mes_fin % 12 + 1:02d}-01 00:00:00"

# Ventana futura que se evalúa y tamaño de cada intervalo, en minutos
P2_HORIZONTE_MIN = 120
P2_BIN_MIN = 30
P2_H_BINS = P2_HORIZONTE_MIN // P2_BIN_MIN
# Un cambio mayor a esta fracción de la capacidad se considera riesgo
P2_ALFA = 0.5

# P2: historia y modelo (cada uno se puede cambiar con la variable de entorno del mismo nombre).
# La partición es la común del proyecto (FRONTERA_VALID, FRONTERA_TEST y BRECHA_HORAS).
# Peso (en días) de la frecuencia global en la historia "hasta ayer"
P2_PESO_HISTORICO = p3_leer_entero("P2_PESO_HISTORICO", 20)
P2_MIN_SAMPLES_LEAF = p3_leer_entero("P2_MIN_SAMPLES_LEAF", 500)
P2_REGISTRAR_MLFLOW = p3_leer_bool("P2_REGISTRAR_MLFLOW")

P2_PARAMETROS_HGB = {"max_iter": 150, "learning_rate": 0.05, "early_stopping": False}
P2_CLASES = ["sin_riesgo", "sin_bicis", "sin_anclajes"]
# Nombre base del run, etapa y familia (etiquetas `stage` y `model_family` en MLflow)
P2_MODELOS = {
    "baseline": ("baseline-historico", "baseline", "frecuencia_historica"),
    "hgb": ("hgb", "candidato", "hist_gradient_boosting"),
}
P2_DESCRIPCIONES = {  # caja "Description" de cada run
    "baseline": "Frecuencia de cada riesgo en los días previos de la misma estación, tipo de día "
    "y hora, suavizada hacia la frecuencia global. Referencia que el candidato debe superar.",
    "hgb": "HistGradientBoostingClassifier con las features de flujo, hora y estación más la "
    "historia hasta ayer; predice sin riesgo, sin bicis o sin anclajes en las próximas 2 h.",
}


P2_FEATURES_SIN_HISTORICO = (
    [f"neto_{k}_relativo" for k in (1, 2, 4, 8)]
    + [f"{c}_{k}" for c in ("salidas", "entradas") for k in (2, 4)]
    + ["cumsum_dia_relativo", "dpcapacity", "latitude", "longitude", "hora_sin", "hora_cos"]
    + ["es_fin_semana"]
)

P2_FEATURES_CON_HISTORICO = P2_FEATURES_SIN_HISTORICO + ["hist_sin_bicis", "hist_sin_anclajes"]

# Tabla estación-bin con la etiqueta y las variables de P2 (`features p2-features`)
RUTA_P2_CAMBIOS = PROCESSED_DATA_DIR / "p2_cambios_estacion_bin.parquet"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
