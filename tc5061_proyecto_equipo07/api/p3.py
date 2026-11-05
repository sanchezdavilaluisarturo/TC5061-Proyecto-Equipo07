"""Endpoints de P3: segmentos de estaciones (clustering).

Recibe el perfil de uso de cada estación, tal como lo calcula `features.p3_perfil_de_estaciones`
(proporción de salidas por hora en días laborales y en fin de semana, más cuatro indicadores), y
regresa el grupo al que el modelo champion asigna cada estación.
"""

from typing import Annotated

from fastapi import APIRouter, Request
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.api.modelos import obtener_modelo
from tc5061_proyecto_equipo07.modeling.predict import p3_reasignar_grupos

router = APIRouter(prefix="/p3", tags=["P3: segmentos de estaciones"])

# El modelo recibe 44 columnas (20 horas de servicio x 2 tipos de día + 4 indicadores); la API
# pide los perfiles como arreglos de 20 valores, en el orden de `cfg.P3_HORAS_SERVICIO`
P3_N_HORAS = len(cfg.P3_HORAS_SERVICIO)
# Un día completo de la red cabe en una petición: son ~470 estaciones
P3_MAX_ESTACIONES = 1_000
# Tolerancia de la suma de cada perfil: son proporciones que deben sumar 1
P3_TOLERANCIA_SUMA = 1e-3
# Topes de cordura de los indicadores. `neto_sobre_salidas` no tiene techo natural (una estación
# puede recibir muchas más bicis de las que despacha); el piso es -1 (no recibe ninguna)
P3_MAX_LOG_VIAJES = 10
P3_MAX_NETO = 10
P3_MAX_DURACION_MIN = 1440

# Primera fila del input_example del champion (versión 4)
P3_EJEMPLO = {
    "origen_id": 1,
    "perfil_laboral": [
        0.000798, 0.006649, 0.028238, 0.074473, 0.119295, 0.081107, 0.058488, 0.047801,
        0.040446, 0.059225, 0.060991, 0.062281, 0.068976, 0.060546, 0.086619, 0.064108,
        0.042903, 0.024185, 0.007601, 0.005267,
    ],
    "perfil_fin_de_semana": [
        0.019705, 0.002723, 0.025393, 0.039811, 0.052147, 0.066725, 0.056312, 0.05495,
        0.057834, 0.079221, 0.089234, 0.072333, 0.056232, 0.051826, 0.053829, 0.068568,
        0.074495, 0.051746, 0.013698, 0.013217,
    ],
    "log_viajes_por_dia": 3.823337,
    "neto_sobre_salidas": 0.092065,
    "prop_circulares": 0.067815,
    "duracion_mediana_min": 10.316667,
}  # fmt: skip

# Proporción de salidas de una hora: entre 0 y 1
Proporcion = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
DESCRIPCION_PERFIL = (
    f"Proporción de salidas en cada una de las {P3_N_HORAS} horas de servicio "
    "(0 y de 5 a 23), en ese orden. Suma 1; todo ceros solo si la estación no tuvo ese tipo de día"
)


def p3_validar_suma(perfil: list[float]) -> list[float]:
    """La suma es 1 (proporciones) o 0 (la estación no tuvo viajes de ese tipo de día)."""
    suma = sum(perfil)
    if suma != 0 and abs(suma - 1) > P3_TOLERANCIA_SUMA:
        raise ValueError(f"las proporciones deben sumar 1 (suman {suma:.4f})")
    return perfil


class P3Estacion(BaseModel):
    """Una estación con las 44 variables de P3, tal como las calcula `features.py`."""

    # strict: "8" (texto) no se convierte en silencio a 8; forbid: un campo de más es un error
    model_config = ConfigDict(
        extra="forbid", strict=True, json_schema_extra={"examples": [P3_EJEMPLO]}
    )

    origen_id: int = Field(ge=1, description="Id de la estación")
    perfil_laboral: list[Proporcion] = Field(
        min_length=P3_N_HORAS, max_length=P3_N_HORAS, description=DESCRIPCION_PERFIL
    )
    perfil_fin_de_semana: list[Proporcion] = Field(
        min_length=P3_N_HORAS, max_length=P3_N_HORAS, description=DESCRIPCION_PERFIL
    )
    log_viajes_por_dia: float = Field(
        ge=0, le=P3_MAX_LOG_VIAJES, allow_inf_nan=False, description="log(1 + salidas por día)"
    )
    neto_sobre_salidas: float = Field(
        ge=-1,
        le=P3_MAX_NETO,
        allow_inf_nan=False,
        description="(llegadas - salidas) / salidas; negativo si despacha más de lo que recibe",
    )
    prop_circulares: float = Field(
        ge=0, le=1, allow_inf_nan=False, description="Proporción de viajes que regresan al origen"
    )
    duracion_mediana_min: float = Field(
        gt=0, le=P3_MAX_DURACION_MIN, allow_inf_nan=False, description="Minutos"
    )

    _suma_laboral = field_validator("perfil_laboral")(p3_validar_suma)
    _suma_fin_de_semana = field_validator("perfil_fin_de_semana")(p3_validar_suma)


class P3Peticion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estaciones: list[P3Estacion] = Field(min_length=1, max_length=P3_MAX_ESTACIONES)

    @model_validator(mode="after")
    def p3_sin_estaciones_repetidas(self) -> "P3Peticion":
        ids = [estacion.origen_id for estacion in self.estaciones]
        repetidas = sorted({i for i in ids if ids.count(i) > 1})
        if repetidas:
            raise ValueError(f"origen_id repetido: {repetidas[:5]}")
        return self


class P3Prediccion(BaseModel):
    origen_id: int
    grupo: int = Field(ge=0, description="Grupo asignado por el modelo (0, 1, ...)")


class P3Respuesta(BaseModel):
    modelo: str
    version: int
    predicciones: list[P3Prediccion]
    # Estaciones de cada grupo dentro de esta petición (no de toda la red)
    estaciones_por_grupo: dict[int, int]


def p3_columnas_horarias(prefijo: str) -> list[str]:
    """Nombres de las columnas del modelo para un tipo de día: lab_h00, lab_h05... lab_h23."""
    return [f"{prefijo}_h{hora:02d}" for hora in cfg.P3_HORAS_SERVICIO]


def p3_a_dataframe(estaciones: list[P3Estacion], modelo) -> pd.DataFrame:
    """DataFrame con las columnas y los tipos de la signature del modelo (una fila por estación).

    Los perfiles llegan como arreglos y aquí se abren en `lab_hHH` y `fin_hHH`. Las columnas se
    toman de la signature y no de una lista escrita aquí: si se reentrena con otras, la API las usa.
    """
    laborales = pd.DataFrame(
        [e.perfil_laboral for e in estaciones], columns=p3_columnas_horarias("lab")
    )
    fines = pd.DataFrame(
        [e.perfil_fin_de_semana for e in estaciones], columns=p3_columnas_horarias("fin")
    )
    indicadores = pd.DataFrame(
        [
            e.model_dump(exclude={"origen_id", "perfil_laboral", "perfil_fin_de_semana"})
            for e in estaciones
        ]
    )
    datos = pd.concat([laborales, fines, indicadores], axis=1)
    esquema = modelo.metadata.get_input_schema()
    if esquema is None:
        return datos
    return datos[[c.name for c in esquema.inputs]].astype(
        {c.name: c.type.to_pandas() for c in esquema.inputs}
    )


@router.post("/segmentar", response_model=P3Respuesta)
def p3_segmentar(peticion: P3Peticion, request: Request) -> P3Respuesta:
    """Grupo de cada estación según el modelo champion de P3."""
    servido = obtener_modelo(request, "p3")
    datos = p3_a_dataframe(peticion.estaciones, servido.modelo)
    grupos = p3_reasignar_grupos(servido.modelo, datos, list(datos.columns))

    predicciones = [
        P3Prediccion(origen_id=e.origen_id, grupo=int(g))
        for e, g in zip(peticion.estaciones, grupos)
    ]
    por_grupo = pd.Series(grupos).value_counts().sort_index()
    return P3Respuesta(
        modelo=servido.nombre,
        version=servido.version,
        predicciones=predicciones,
        estaciones_por_grupo={int(g): int(n) for g, n in por_grupo.items()},
    )
