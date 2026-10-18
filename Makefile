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


## Make P3 features (partición temporal, estaciones y demanda por estación-hora)
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.dataset p3-particionar
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.features p3-features


## Make P1 features (viajes por estación-hora con rezagos, calendario y estación)
.PHONY: features-p1
features-p1:
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.features p1-features


## Make P2 features (limpieza propia de P2, etiqueta de riesgo y variables por estación-bin)
.PHONY: features-p2
features-p2:
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.features p2-features


## Train P2 (baseline histórico y HGB; con P2_REGISTRAR_MLFLOW=true registra en MLflow)
.PHONY: train-p2
train-p2:
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.modeling.train p2-entrenar


## Train P3 (modelo base y modelo alternativo; con P3_REGISTRAR_MLFLOW=true registra en MLflow)
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.modeling.train p3-baseline
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo07.modeling.train p3-alternativo


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
