# Imagen de la API de MiBici (FastAPI). Sigue el estándar OCI: se construye igual con
# `docker build` o `podman build`. Los modelos no van en la imagen: se descargan del Model
# Registry al arrancar, con la URI de MLFLOW_TRACKING_URI (ver README, API REST en contenedor).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# XGBoost necesita la librería de OpenMP del sistema, que la imagen slim no trae
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias antes que el código: esta capa se reutiliza mientras no cambien los requirements.
# requirements.txt fija las versiones (restricciones); su línea `-e .` no aplica aquí
COPY requirements.txt requirements-api.txt ./
RUN grep -v '^-e' requirements.txt > restricciones.txt \
    && pip install -r requirements-api.txt -c restricciones.txt

COPY tc5061_proyecto_equipo07/ tc5061_proyecto_equipo07/

# Sin privilegios de root dentro del contenedor
RUN useradd --create-home api
USER api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/salud')"

CMD ["uvicorn", "tc5061_proyecto_equipo07.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
