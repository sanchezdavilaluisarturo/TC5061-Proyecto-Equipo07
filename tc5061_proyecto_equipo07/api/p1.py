"""Endpoint de P1: viajes esperados por estación-hora del día siguiente (regresión).

Recibe las features ya calculadas por `features.py` (son batch features: se calculan al cierre
de cada día) y regresa la predicción por estación-hora y su suma por zona.
"""

from typing import Literal

from fastapi import APIRouter, Request
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.api.modelos import obtener_modelo
from tc5061_proyecto_equipo07.modeling.predict import p1_predecir as p1_predecir_modelo

router = APIRouter(prefix="/p1", tags=["P1: demanda por estación-hora"])

# Un día completo de la red cabe en una petición: 472 estaciones x 20 horas de servicio = 9,440
P1_MAX_REGISTROS = 10_000
# Rango de las estaciones (latitud 20.64 a 20.74, longitud -103.43 a -103.30) con ~10 km de
# margen para estaciones nuevas del área metropolitana de Guadalajara
P1_LATITUD = (20.55, 20.85)
P1_LONGITUD = (-103.55, -103.20)
# Tope de cordura para los rezagos: el máximo observado en una estación-hora es 207 viajes
P1_MAX_VIAJES = 1000
P1_ZONAS = Literal["GDL", "ZPN", "TLQ"]

# Primera fila del input_example del champion (versión 2)
P1_EJEMPLO = {
    "origen_id": 168,
    "viajes_rezago_1d": 0.0,
    "viajes_rezago_7d": 0.0,
    "viajes_promedio_4sem": 0.75,
    "hora": 23,
    "dia_semana": 6,
    "es_fin_de_semana": 1,
    "mes": 9,
    "es_festivo": 0,
    "capacidad": 19,
    "latitud": 20.6896896362,
    "longitud": -103.3748626709,
    "zona": "GDL",
    "dias_desde_apertura": 256,
}

DESCRIPCION_REZAGO = "Null si la estación aún no tiene historia (el modelo maneja el vacío)"


class P1Registro(BaseModel):
    """Una estación-hora con las 14 features de P1, tal como las calcula `features.py`."""

    # strict: "8" (texto) no se convierte en silencio a 8; forbid: un campo de más es un error
    model_config = ConfigDict(
        extra="forbid", strict=True, json_schema_extra={"examples": [P1_EJEMPLO]}
    )

    origen_id: int = Field(ge=1, description="Id de la estación de origen")
    viajes_rezago_1d: float | None = Field(
        ge=0, le=P1_MAX_VIAJES, allow_inf_nan=False, description=DESCRIPCION_REZAGO
    )
    viajes_rezago_7d: float | None = Field(
        ge=0, le=P1_MAX_VIAJES, allow_inf_nan=False, description=DESCRIPCION_REZAGO
    )
    viajes_promedio_4sem: float | None = Field(
        ge=0, le=P1_MAX_VIAJES, allow_inf_nan=False, description=DESCRIPCION_REZAGO
    )
    hora: int = Field(ge=0, le=23, description="Hora de servicio: 0 y de 5 a 23")
    dia_semana: int = Field(ge=0, le=6, description="0 = lunes, 6 = domingo")
    es_fin_de_semana: Literal[0, 1]
    mes: int = Field(ge=1, le=12)
    es_festivo: Literal[0, 1]
    capacidad: int = Field(ge=1, le=100, description="Anclajes de la estación")
    latitud: float = Field(ge=P1_LATITUD[0], le=P1_LATITUD[1], allow_inf_nan=False)
    longitud: float = Field(ge=P1_LONGITUD[0], le=P1_LONGITUD[1], allow_inf_nan=False)
    zona: P1_ZONAS = Field(description="Municipio: Guadalajara, Zapopan o Tlaquepaque")
    dias_desde_apertura: int = Field(ge=0)

    @field_validator("hora")
    @classmethod
    def p1_hora_de_servicio(cls, hora: int) -> int:
        if hora in cfg.HORAS_SIN_SERVICIO:
            raise ValueError(f"MiBici no opera de 1:00 a 4:59 (hora {hora})")
        return hora

    @model_validator(mode="after")
    def p1_fin_de_semana_coherente(self) -> "P1Registro":
        if self.es_fin_de_semana != int(self.dia_semana >= 5):
            raise ValueError(
                f"es_fin_de_semana={self.es_fin_de_semana} no corresponde a "
                f"dia_semana={self.dia_semana} (sábado = 5, domingo = 6)"
            )
        return self


class P1Peticion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registros: list[P1Registro] = Field(min_length=1, max_length=P1_MAX_REGISTROS)


class P1Prediccion(BaseModel):
    origen_id: int
    hora: int
    zona: P1_ZONAS
    viajes_esperados: float = Field(ge=0)


class P1Respuesta(BaseModel):
    modelo: str
    version: int
    predicciones: list[P1Prediccion]
    # Suma de las estaciones de cada zona dentro de esta petición
    por_zona: dict[P1_ZONAS, float]


def p1_a_dataframe(registros: list[P1Registro], modelo) -> pd.DataFrame:
    """DataFrame con los tipos de la signature del modelo (JSON llega como int64 y float64).

    Los tipos se leen del modelo y no se escriben aquí: si se reentrena con otros, la API los toma.
    Pydantic ya validó los rangos, así que pasar a un tipo más chico no desborda ningún valor.
    """
    datos = pd.DataFrame([registro.model_dump() for registro in registros])[cfg.P1_VARIABLES]
    esquema = modelo.metadata.get_input_schema()
    if esquema is None:
        return datos
    return datos.astype({columna.name: columna.type.to_pandas() for columna in esquema.inputs})


@router.post("/predecir", response_model=P1Respuesta)
def p1_predecir(peticion: P1Peticion, request: Request) -> P1Respuesta:
    """Viajes esperados por estación-hora (y su suma por zona) con el modelo champion de P1."""
    servido = obtener_modelo(request, "p1")
    datos = p1_a_dataframe(peticion.registros, servido.modelo)
    # float64 antes de redondear: en float32, 1.1808 se vuelve 1.180799961...
    viajes = p1_predecir_modelo(servido.modelo, datos).astype("float64").round(4)

    predicciones = [
        P1Prediccion(origen_id=r.origen_id, hora=r.hora, zona=r.zona, viajes_esperados=v)
        for r, v in zip(peticion.registros, viajes.tolist())
    ]
    por_zona = (
        pd.Series(viajes, index=[r.zona for r in peticion.registros])
        .groupby(level=0)
        .sum()
        .round(4)
    )
    return P1Respuesta(
        modelo=servido.nombre,
        version=servido.version,
        predicciones=predicciones,
        por_zona=por_zona.to_dict(),
    )
