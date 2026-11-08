"""API REST de MiBici: sirve los modelos de P1, P2 y P3 registrados en el Model Registry.

Uso local (sin contenedor): `make api-local`; documentación interactiva en /docs.
"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel

from tc5061_proyecto_equipo07.api import p1, p2, p3
from tc5061_proyecto_equipo07.api.modelos import cargar_modelos


class EstadoModelo(BaseModel):
    nombre: str
    alias: str
    version: int | None
    listo: bool
    error: str | None


class RespuestaSalud(BaseModel):
    estado: Literal["ok", "degradado"]
    modelos: dict[str, EstadoModelo]


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Carga los modelos una sola vez al arrancar, no en cada petición."""
    app.state.modelos = cargar_modelos()
    yield


app = FastAPI(
    title="API MiBici - Equipo 07",
    description="Predicciones de P1 (demanda), P2 (riesgo de estación) y P3 (segmentos de "
    "estaciones) con los modelos del Model Registry de MLflow.",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)
app.include_router(p1.router)
app.include_router(p2.router)
app.include_router(p3.router)


@app.get("/salud", response_model=RespuestaSalud, tags=["Servicio"])
def salud(request: Request) -> RespuestaSalud:
    """Health check: la API está viva y qué versión de cada modelo cargó."""
    modelos = {
        pregunta: EstadoModelo(
            nombre=m.nombre, alias=m.alias, version=m.version, listo=m.listo, error=m.error
        )
        for pregunta, m in request.app.state.modelos.items()
    }
    estado = "ok" if all(m.listo for m in modelos.values()) else "degradado"
    return RespuestaSalud(estado=estado, modelos=modelos)
