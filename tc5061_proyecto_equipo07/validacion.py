"""Validación de datos con DeepChecks, común a las tres preguntas.

Cada pregunta llama a `validar_tabla` con su tabla de modelado y su columna de partición. Se
corren dos suites: integridad (toda la tabla) y train-test (un par de conjuntos), y se deja en
reports/validacion/ el reporte HTML de cada una y un resumen CSV con el resultado de cada
condición.

Hay dos tipos de condición. Las críticas indican un bug del pipeline (por ejemplo, fechas de
prueba dentro de entrenamiento) y hacen fallar la ejecución con cualquier resultado distinto de
PASS, porque DeepChecks marca algunas condiciones (como los duplicados) solo como WARN. Las demás se documentan como
hallazgos: describen el problema real (drift esperado, nulos de los rezagos) y no lo detienen.
"""

from pathlib import Path

from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (
    ConflictingLabels,
    DataDuplicates,
    DateTrainTestLeakageDuplicates,
    DateTrainTestLeakageOverlap,
    FeatureDrift,
    IsSingleValue,
    LabelDrift,
    MixedDataTypes,
    MixedNulls,
    NewCategoryTrainTest,
    PercentOfNulls,
    SpecialCharacters,
)
from loguru import logger
import pandas as pd

from tc5061_proyecto_equipo07 import config as cfg

VALIDACION_DIR = cfg.REPORTS_DIR / "validacion"

# Con partición temporal, que prueba comparta fechas con entrenamiento siempre es un bug
CRITICOS_POR_DEFECTO = (
    "Date Train-Test Leakage (overlap)",
    "Date Train-Test Leakage (duplicates)",
)


def armar_dataset(
    tabla: pd.DataFrame, target: str, categoricas: list[str], fecha: str | None = None
) -> Dataset:
    """Dataset de DeepChecks; la fecha se declara para los chequeos de leakage temporal."""
    return Dataset(tabla, label=target, cat_features=categoricas, datetime_name=fecha)


def suite_integridad(umbral_nulos: float = 0.0) -> Suite:
    """Chequeos de una sola tabla: nulos, duplicados, tipos y valores sospechosos."""
    return Suite(
        "Integridad de datos",
        PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(umbral_nulos),
        MixedNulls().add_condition_different_nulls_less_equal_to(1),
        DataDuplicates().add_condition_ratio_less_or_equal(0),
        MixedDataTypes().add_condition_rare_type_ratio_not_in_range((0.01, 0.1)),
        SpecialCharacters().add_condition_ratio_of_special_characters_less_or_equal(0.001),
        IsSingleValue().add_condition_not_single_value(),
        ConflictingLabels().add_condition_ratio_of_conflicting_labels_less_or_equal(0),
    )


def suite_train_test() -> Suite:
    """Chequeos entre dos conjuntos: leakage temporal, categorías nuevas y drift."""
    return Suite(
        "Validación train-test",
        DateTrainTestLeakageOverlap().add_condition_leakage_ratio_less_or_equal(0),
        DateTrainTestLeakageDuplicates().add_condition_leakage_ratio_less_or_equal(0),
        NewCategoryTrainTest().add_condition_new_categories_less_or_equal(0),
        FeatureDrift().add_condition_drift_score_less_than(),
        LabelDrift().add_condition_drift_score_less_than(),
    )


def resumir_resultado(resultado, conjunto: str) -> pd.DataFrame:
    """Una fila por condición: suite, chequeo, condición, resultado (PASS/WARN/FAIL/ERROR)."""
    filas = []
    for chequeo in resultado.results:
        nombre = chequeo.get_header()
        condiciones = getattr(chequeo, "conditions_results", None)
        if condiciones is None:
            # CheckFailure: el chequeo no pudo correr (por ejemplo, falta la columna de fecha)
            filas.append([conjunto, nombre, "", "ERROR", str(chequeo.exception)])
            continue
        for condicion in condiciones:
            filas.append(
                [conjunto, nombre, condicion.name, condicion.category.name, condicion.details]
            )
    return pd.DataFrame(filas, columns=["suite", "chequeo", "condicion", "resultado", "detalle"])


def muestrear(tabla: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """Muestra con la semilla del proyecto; las tablas de millones de filas no caben en memoria."""
    if n is None or len(tabla) <= n:
        return tabla
    return tabla.sample(n, random_state=cfg.SEMILLA)


def validar_tabla(
    nombre: str,
    tabla: pd.DataFrame,
    particion: str,
    target: str,
    categoricas: list[str],
    fecha: str | None = None,
    conjuntos: tuple[str, str] = ("train", "test"),
    criticos: tuple[str, ...] = CRITICOS_POR_DEFECTO,
    umbral_nulos: float = 0.0,
    muestra: int | None = None,
    carpeta: Path = VALIDACION_DIR,
) -> pd.DataFrame:
    """Corre integridad y train-test, guarda los reportes y falla si una condición crítica falla.

    `particion` es la columna que dice a qué conjunto pertenece cada fila y no entra a los
    chequeos. Regresa el resumen de condiciones (también queda en `<nombre>_resumen.csv`).
    """
    faltantes = {particion, target, *categoricas, *([fecha] if fecha else [])} - set(tabla.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en la tabla de {nombre}: {sorted(faltantes)}")
    for conjunto in conjuntos:
        if not (tabla[particion] == conjunto).any():
            raise ValueError(f"La partición '{conjunto}' no tiene filas en la tabla de {nombre}")

    carpeta.mkdir(parents=True, exist_ok=True)
    sin_particion = tabla.drop(columns=particion)
    entrenamiento, prueba = (
        muestrear(sin_particion[tabla[particion] == c], muestra) for c in conjuntos
    )

    integridad = suite_integridad(umbral_nulos).run(
        armar_dataset(muestrear(sin_particion, muestra), target, categoricas, fecha)
    )
    train_test = suite_train_test().run(
        armar_dataset(entrenamiento, target, categoricas, fecha),
        armar_dataset(prueba, target, categoricas, fecha),
    )
    # connected=True carga plotly desde CDN: el HTML pesa KB en lugar de MB. DeepChecks no
    # sobrescribe (crearía "<nombre> (1).html"), así que se borra el reporte anterior
    for suite, sufijo in ((integridad, "integridad"), (train_test, "train_test")):
        ruta = carpeta / f"{nombre}_{sufijo}.html"
        ruta.unlink(missing_ok=True)
        suite.save_as_html(str(ruta), connected=True)

    resumen = pd.concat(
        [
            resumir_resultado(integridad, "integridad"),
            resumir_resultado(train_test, f"{conjuntos[0]} vs {conjuntos[1]}"),
        ],
        ignore_index=True,
    )
    resumen["critico"] = resumen["chequeo"].isin(criticos)
    resumen.to_csv(carpeta / f"{nombre}_resumen.csv", index=False)

    conteo = resumen["resultado"].value_counts().to_dict()
    logger.info(f"Validación de {nombre}: {conteo} (reportes en {carpeta})")
    no_pasa = resumen["resultado"] != "PASS"
    fallas = resumen[resumen["critico"] & no_pasa]
    if not fallas.empty:
        raise ValueError(
            f"Validación de {nombre}: fallaron condiciones críticas:\n{fallas.to_string()}"
        )
    hallazgos = resumen[~resumen["critico"] & no_pasa]
    for fila in hallazgos.itertuples():
        logger.warning(f"Hallazgo ({fila.suite}) {fila.chequeo}: {fila.detalle}")
    return resumen
