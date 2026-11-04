#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = tc5061_proyecto_equipo07
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.dataset todo


## Make features de las tres preguntas
.PHONY: features
features: features-p1 features-p2 features-p3


## Make P1 features (viajes por estación-hora con rezagos, calendario y estación)
.PHONY: features-p1
features-p1:
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.features p1-features


## Validación de P1 con DeepChecks sobre la tabla de features (reportes en reports/validacion)
.PHONY: validar-p1
validar-p1:
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.features p1-validar


## Validación de P2 con DeepChecks sobre la tabla de features (reportes en reports/validacion)
.PHONY: validar-p2
validar-p2:
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.features p2-validar


## Validación de P3 con DeepChecks sobre la tabla de modelado (reportes en reports/validacion)
.PHONY: validar-p3
validar-p3:
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.features p3-validar


## Make P2 features (limpieza propia de P2, etiqueta de riesgo y variables por estación-bin)
.PHONY: features-p2
features-p2:
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.features p2-features


## Make P3 features (partición temporal, estaciones y demanda por estación-hora)
.PHONY: features-p3
features-p3:
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.dataset p3-particionar
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.features p3-features


## Train de las tres preguntas
.PHONY: train
train: train-p1 train-p2 train-p3


## Train P1 (baseline, HGB y XGBoost; con P1_REGISTRAR_MLFLOW=true registra en MLflow)
.PHONY: train-p1
train-p1: features-p1
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.modeling.train p1-entrenar


## Train P2 (baseline histórico y HGB; con P2_REGISTRAR_MLFLOW=true registra en MLflow)
.PHONY: train-p2
train-p2:
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.modeling.train p2-entrenar


## Train P3 (modelo base y modelo alternativo; con P3_REGISTRAR_MLFLOW=true registra en MLflow)
.PHONY: train-p3
train-p3:
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.modeling.train p3-baseline
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.modeling.train p3-alternativo


## MLflow local de respaldo si el servidor del equipo no responde (usar MLFLOW_TRACKING_URI=http://127.0.0.1:5000)
.PHONY: mlflow-local
mlflow-local:
	mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts \
		--host 127.0.0.1 --port 5000


## Pruebas de tests/ (PyTest, incluye la validación sintética con DeepChecks)
.PHONY: pruebas
pruebas:
	$(PYTHON_INTERPRETER) -m pytest tests


# Vulnerabilidades revisadas y aceptadas: mlflow (3), nltk (1) y setuptools (3).
# La justificación de cada una está en el README (Chequeos de seguridad)
PIP_AUDIT_IGNORAR = \
	--ignore-vuln CVE-2026-53995 --ignore-vuln CVE-2026-115886 --ignore-vuln CVE-2026-116691 \
	--ignore-vuln PYSEC-2026-3740 \
	--ignore-vuln PYSEC-2025-49 --ignore-vuln PYSEC-2026-3447 --ignore-vuln GHSA-xmj3-9gf7-h52w

## Seguridad: bandit sobre el código y pip-audit sobre requirements.txt (reportes en reports/seguridad)
.PHONY: seguridad
seguridad:
	mkdir -p reports/seguridad
	bandit -r tc5061_proyecto_equipo07 -f json -o reports/seguridad/bandit.json --exit-zero
	bandit -r tc5061_proyecto_equipo07
	-pip-audit -r requirements.txt --no-deps --disable-pip --progress-spinner off \
		$(PIP_AUDIT_IGNORAR) -f json -o reports/seguridad/pip_audit.json
	pip-audit -r requirements.txt --no-deps --disable-pip --progress-spinner off $(PIP_AUDIT_IGNORAR)


# API REST (FastAPI) en contenedor. Funciona con Docker (por defecto) o Podman: make api-build MOTOR=podman
MOTOR ?= docker
IMAGEN_API = tc5061-equipo07-api
PUERTO_API = 8000

## API sin contenedor, con recarga al guardar (http://127.0.0.1:8000/docs)
.PHONY: api-local
api-local:
	uvicorn tc5061_proyecto_equipo07.api.main:app --reload --port $(PUERTO_API)

## Construye la imagen de la API con el Dockerfile
.PHONY: api-build
api-build:
	$(MOTOR) build -t $(IMAGEN_API) .

## Levanta el contenedor de la API (toma MLFLOW_TRACKING_URI del .env)
.PHONY: api-run
api-run:
	$(MOTOR) run --rm --name $(IMAGEN_API) -p $(PUERTO_API):8000 --env-file .env $(IMAGEN_API)

## Consulta /salud de la API levantada: versión y alias de cada modelo
.PHONY: api-probar
api-probar:
	curl -s http://127.0.0.1:$(PUERTO_API)/salud | $(PYTHON_INTERPRETER) -m json.tool

## Predicción de P1 con la petición de ejemplo (references/api_p1_peticion.json)
.PHONY: api-probar-p1
api-probar-p1:
	curl -s -X POST http://127.0.0.1:$(PUERTO_API)/p1/predecir -H "Content-Type: application/json" \
		-d @references/api_p1_peticion.json | $(PYTHON_INTERPRETER) -m json.tool

## Smoke test de la API levantada: /salud en "ok", P1 predice y una petición inválida da 422
.PHONY: api-smoke
api-smoke:
	curl -sf http://127.0.0.1:$(PUERTO_API)/salud | $(PYTHON_INTERPRETER) -c \
		'import json, sys; s = json.load(sys.stdin); print(s); sys.exit(s["estado"] != "ok")'
	curl -sf -X POST http://127.0.0.1:$(PUERTO_API)/p1/predecir -H "Content-Type: application/json" \
		-d @references/api_p1_peticion.json | $(PYTHON_INTERPRETER) -c \
		'import json, sys; r = json.load(sys.stdin); print(r); sys.exit(len(r["predicciones"]) != 5)'
	test "$$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:$(PUERTO_API)/p1/predecir \
		-H 'Content-Type: application/json' -d '{"registros": []}')" = 422
	@echo "Smoke test de la API: OK"


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z0-9_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
