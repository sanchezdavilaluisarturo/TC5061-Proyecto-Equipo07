"""Configuración común de las pruebas."""

import os

# Las pruebas entrenan con unos cientos de filas: con todos los hilos del CPU, coordinar OpenMP
# cuesta más que el cálculo (en WSL2 con 16 hilos, 4 pruebas pasaron de 5 s a 7 min). Se fija
# antes de importar sklearn y xgboost; no afecta `make train` ni `mlflow run`.
os.environ.setdefault("OMP_NUM_THREADS", "1")
