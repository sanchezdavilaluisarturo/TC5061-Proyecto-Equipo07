"""Endpoint de P2: riesgo de que una estación se quede sin bicis o sin anclajes en las próximas 2 h.

Recibe las features ya calculadas por `features.py`: las de flujo se calculan al cierre de cada bin
de 30 minutos y la historia (`hist_*`) al cierre del día anterior. Regresa la probabilidad de cada
clase y la clase decidida con los umbrales elegidos en validación, que se leen del run de origen.
"""

import math
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Request
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.api.modelos import ModeloServido, obtener_modelo
from tc5061_proyecto_equipo07.api.p1 import P1_LATITUD, P1_LONGITUD
from tc5061_proyecto_equipo07.modeling.predict import p2_decidir_clase

router = APIRouter(prefix="/p2", tags=["P2: riesgo de estación"])

# La red completa en un bin cabe en una petición (472 estaciones), con margen para estaciones nuevas
P2_MAX_REGISTROS = 1000
# Bins con predicción: de 05:00 (10) a 22:30 (45). De 23:00 a 00:30 las 2 h siguientes se salen del
# día de servicio (termina a la 01:00), así que esos bins no tienen etiqueta ni se entrenaron
P2_BINS_POR_DIA = 24 * 60 // cfg.P2_BIN_MIN
P2_BINS_CON_PREDICCION = range(10, 46)
# El máximo de viajes observado en 4 bins es 392, y el de un neto relativo, 8.9
P2_MAX_VIAJES = 1000
P2_MAX_RELATIVO = 20.0
# Admite decimales redondeados por el cliente (4 decimales x 39 anclajes = 0.002)
P2_TOLERANCIA = 0.01
P2_CLASE = Literal["sin_riesgo", "sin_bicis", "sin_anclajes"]

# Fila real de train (estación 2, viernes 2023-12-15 a las 18:00) con su historia
P2_EJEMPLO = {
    "estacion": 2,
    "neto_1_relativo": 0.2,
    "neto_2_relativo": 0.4,
    "neto_4_relativo": 0.666667,
    "neto_8_relativo": 1.0,
    "salidas_2": 3,
    "salidas_4": 10,
    "entradas_2": 9,
    "entradas_4": 20,
    "cumsum_dia_relativo": 0.266667,
    "dpcapacity": 15,
    "latitude": 20.666378,
    "longitude": -103.348824,
    "hora_sin": -1.0,
    "hora_cos": 0.0,
    "es_fin_semana": 0,
    "hist_sin_bicis": 0.000227,
    "hist_sin_anclajes": 0.147563,
}

Relativo = Annotated[float, Field(ge=-P2_MAX_RELATIVO, le=P2_MAX_RELATIVO, allow_inf_nan=False)]
Viajes = Annotated[int, Field(ge=0, le=P2_MAX_VIAJES)]
Frecuencia = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


def p2_bin_del_dia(hora_sin: float, hora_cos: float) -> int:
    """Bin de 30 min (0 a 47) más cercano al ángulo de la codificación cíclica de la hora."""
    angulo = math.atan2(hora_sin, hora_cos)
    return round(angulo * P2_BINS_POR_DIA / (2 * math.pi)) % P2_BINS_POR_DIA


class P2Registro(BaseModel):
    """Una estación en un bin de 30 min con las 17 features de P2, como las calcula `features.py`."""

    model_config = ConfigDict(
        extra="forbid", strict=True, json_schema_extra={"examples": [P2_EJEMPLO]}
    )

    estacion: int = Field(ge=1, description="Id de la estación; no entra al modelo")
    neto_1_relativo: Relativo = Field(description="(entradas - salidas) del bin / capacidad")
    neto_2_relativo: Relativo = Field(description="Lo mismo en los últimos 2 bins (1 h)")
    neto_4_relativo: Relativo = Field(description="Lo mismo en los últimos 4 bins (2 h)")
    neto_8_relativo: Relativo = Field(description="Lo mismo en los últimos 8 bins (4 h)")
    salidas_2: Viajes
    salidas_4: Viajes
    entradas_2: Viajes
    entradas_4: Viajes
    cumsum_dia_relativo: Relativo = Field(description="Neto acumulado desde las 05:00 / capacidad")
    dpcapacity: int = Field(ge=1, le=100, description="Anclajes de la estación")
    latitude: float = Field(ge=P1_LATITUD[0], le=P1_LATITUD[1], allow_inf_nan=False)
    longitude: float = Field(ge=P1_LONGITUD[0], le=P1_LONGITUD[1], allow_inf_nan=False)
    hora_sin: float = Field(ge=-1, le=1, description="sin(2π·bin/48), bin de 30 min")
    hora_cos: float = Field(ge=-1, le=1, description="cos(2π·bin/48)")
    es_fin_semana: Literal[0, 1]
    hist_sin_bicis: Frecuencia
    hist_sin_anclajes: Frecuencia

    @model_validator(mode="after")
    def p2_registro_coherente(self) -> "P2Registro":
        # Se obtiene el bin de 30 min mas cercano a la hora dada
        b = p2_bin_del_dia(self.hora_sin, self.hora_cos)
        angulo = 2 * math.pi * b / P2_BINS_POR_DIA
        # Se rechaza si la hora dada no corresponde con un bin de 30 min (con margen de tolerancia)
        # o si se sale del rango de horas de servicio
        if (
            b not in P2_BINS_CON_PREDICCION
            or abs(self.hora_sin - math.sin(angulo)) > P2_TOLERANCIA
            or abs(self.hora_cos - math.cos(angulo)) > P2_TOLERANCIA
        ):
            raise ValueError("hora_sin y hora_cos deben ser un bin de 30 min de 05:00 a 22:30")
        # Se rechaza si neto_k_relativo no encaja con el neto y las capacidades dadas
        for k in (2, 4):
            neto = getattr(self, f"entradas_{k}") - getattr(self, f"salidas_{k}")
            if abs(getattr(self, f"neto_{k}_relativo") * self.dpcapacity - neto) > P2_TOLERANCIA:
                raise ValueError(f"neto_{k}_relativo x dpcapacity != entradas_{k} - salidas_{k}")
        if self.salidas_2 > self.salidas_4 or self.entradas_2 > self.entradas_4:
            raise ValueError("La ventana de 2 bins no puede tener más viajes que la de 4")
        if self.hist_sin_bicis + self.hist_sin_anclajes > 1:
            raise ValueError("hist_sin_bicis + hist_sin_anclajes no puede pasar de 1")
        return self


class P2Peticion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registros: list[P2Registro] = Field(min_length=1, max_length=P2_MAX_REGISTROS)


class P2Prediccion(BaseModel):
    estacion: int
    hora: str = Field(pattern=r"^\d{2}:\d{2}$", description="Inicio del bin (HH:MM)")
    probabilidades: dict[P2_CLASE, Annotated[float, Field(ge=0, le=1)]]
    clase: P2_CLASE


class P2Respuesta(BaseModel):
    modelo: str
    version: int
    # Umbrales con los que se decidió la clase (elegidos en validación, del run de origen)
    umbrales: dict[Literal["sin_bicis", "sin_anclajes"], float]
    predicciones: list[P2Prediccion]
    por_clase: dict[P2_CLASE, int]


def p2_umbrales(servido: ModeloServido) -> dict[int, float]:
    """Umbrales del run de origen; sin ellos la clase no se puede decidir (503)."""
    try:
        return {k: float(servido.parametros[f"umbral_{cfg.P2_CLASES[k]}"]) for k in (1, 2)}
    except (KeyError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"El modelo de p2 (versión {servido.version}) no tiene sus umbrales: {error}",
        ) from error


def p2_hora(registro: P2Registro) -> str:
    """Hora de inicio del bin del registro, como HH:MM."""
    minutos = p2_bin_del_dia(registro.hora_sin, registro.hora_cos) * cfg.P2_BIN_MIN
    return f"{minutos // 60:02d}:{minutos % 60:02d}"


def p2_a_dataframe(registros: list[P2Registro], modelo) -> pd.DataFrame:
    """DataFrame con las features en el orden y los tipos de la signature del modelo."""
    datos = pd.DataFrame([registro.model_dump() for registro in registros])
    datos = datos[cfg.P2_FEATURES_CON_HISTORICO]
    esquema = modelo.metadata.get_input_schema()
    if esquema is None:
        return datos
    return datos.astype({columna.name: columna.type.to_pandas() for columna in esquema.inputs})


@router.post("/predecir", response_model=P2Respuesta)
def p2_predecir(peticion: P2Peticion, request: Request) -> P2Respuesta:
    """Probabilidad de cada riesgo en las próximas 2 h y la clase decidida con los umbrales."""
    servido = obtener_modelo(request, "p2")
    umbrales = p2_umbrales(servido)
    datos = p2_a_dataframe(peticion.registros, servido.modelo)
    probabilidades = np.asarray(servido.modelo.predict(datos), dtype="float64")
    clases = [cfg.P2_CLASES[k] for k in p2_decidir_clase(probabilidades, umbrales)]
    probabilidades = probabilidades.round(4)

    return P2Respuesta(
        modelo=servido.nombre,
        version=servido.version,
        umbrales={cfg.P2_CLASES[k]: round(v, 4) for k, v in umbrales.items()},
        predicciones=[
            P2Prediccion(
                estacion=r.estacion,
                hora=p2_hora(r),
                probabilidades=dict(zip(cfg.P2_CLASES, p.tolist())),
                clase=c,
            )
            for r, p, c in zip(peticion.registros, probabilidades, clases)
        ],
        por_clase={c: clases.count(c) for c in cfg.P2_CLASES},
    )
