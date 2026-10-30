"""Pruebas de la lectura de variables de entorno (config.py) y de la convención p3_."""

import ast
from pathlib import Path

import pytest

from tc5061_proyecto_equipo07 import config as cfg

PAQUETE = Path(cfg.__file__).parent

# Funciones que ya existían en dataset.py antes de P3 (acuerdo del equipo): no se renombran
DATASET_ORIGINAL = {
    "detectar_codificacion",
    "calcular_sha256",
    "contar_registros",
    "enlaces_del_portal",
    "descargar_archivo",
    "version_datos",
    "leer_viajes_csv",
    "descargar",
    "construir",
    "limpiar",
    "todo",
}


def test_p3_leer_entero(monkeypatch):
    monkeypatch.delenv("P3_PRUEBA", raising=False)
    assert cfg.p3_leer_entero("P3_PRUEBA", 7) == 7
    monkeypatch.setenv("P3_PRUEBA", "  12 ")
    assert cfg.p3_leer_entero("P3_PRUEBA", 7) == 12
    monkeypatch.setenv("P3_PRUEBA", "")
    assert cfg.p3_leer_entero("P3_PRUEBA", 7) == 7


@pytest.mark.parametrize(
    "valor, esperado",
    [("true", True), ("1", True), ("SI", True), ("false", False), ("0", False), ("", False)],
)
def test_p3_leer_bool(monkeypatch, valor, esperado):
    monkeypatch.setenv("P3_PRUEBA", valor)
    assert cfg.p3_leer_bool("P3_PRUEBA") is esperado


def test_p3_leer_lista_enteros(monkeypatch):
    monkeypatch.delenv("P3_PRUEBA", raising=False)
    assert cfg.p3_leer_lista_enteros("P3_PRUEBA", [3, 4]) == [3, 4]
    monkeypatch.setenv("P3_PRUEBA", "5, 6,8")
    assert cfg.p3_leer_lista_enteros("P3_PRUEBA", [3, 4]) == [5, 6, 8]


def test_defectos_de_p3_coinciden_con_los_acordados():
    assert (cfg.P3_K_MINIMO, cfg.P3_K_MAXIMO) == (2, 10)
    assert (cfg.P3_MIN_SALIDAS_TRAIN, cfg.P3_MIN_SALIDAS_VALID) == (1000, 300)
    assert cfg.P3_COMPONENTES == [3, 4, 5, 6, 8, 10]
    assert cfg.P3_REGISTRAR_MLFLOW is False  # por defecto no se escribe en MLflow
    assert cfg.P3_EXPERIMENTO == cfg.EXPERIMENTO_MLFLOW


def p3_funciones_de(ruta: Path) -> list[str]:
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    return [n.name for n in arbol.body if isinstance(n, ast.FunctionDef)]


@pytest.mark.parametrize(
    "archivo", ["features.py", "plots.py", "modeling/train.py", "modeling/predict.py", "config.py"]
)
def test_toda_funcion_nueva_lleva_el_prefijo_de_su_pregunta(archivo):
    prefijos = ("p1_", "p2_", "p3_")
    sin_prefijo = [f for f in p3_funciones_de(PAQUETE / archivo) if not f.startswith(prefijos)]
    assert sin_prefijo == []


def test_dataset_solo_agrega_funciones_con_prefijo_p3():
    nuevas = set(p3_funciones_de(PAQUETE / "dataset.py")) - DATASET_ORIGINAL
    assert nuevas, "dataset.py debería tener funciones p3_"
    assert all(nombre.startswith("p3_") for nombre in nuevas)
    # y lo anterior sigue existiendo, sin renombrar
    assert DATASET_ORIGINAL <= set(p3_funciones_de(PAQUETE / "dataset.py"))
