"""Pruebas del registro de P3: versión, linaje, alias y carga por alias."""

from loguru import logger
import mlflow
import pytest
from test_p3_modeling import COLUMNAS, p3_x_con_tres_grupos
from test_p3_tuning import base_sintetica, preparar_experimento_local

from tc5061_proyecto_equipo07 import config as cfg
from tc5061_proyecto_equipo07.modeling import registro_p3, train


@pytest.fixture
def servidor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(train, "p3_preparar_experimento", preparar_experimento_local(tmp_path))
    mlflow.set_tracking_uri(uri)
    return uri


def crear_candidato(etiquetas_extra=None, metricas=None, modelo=None, ejemplo=None, trusted=None):
    """Registra un run de P3 como lo hace `p3_registrar_en_mlflow`, sin escribir figuras."""
    base = base_sintetica()
    url = train.p3_registrar_en_mlflow(
        nombre_run="candidato",
        etiquetas=train.p3_etiquetas_run("alternativo", "prueba", **(etiquetas_extra or {})),
        parametros={"data_version": "abc123", "k": base.k_elegido},
        metricas=metricas or {"silueta": 0.14, "ari_valid": 0.68, "tam_min": 40},
        modelo=base.modelo if modelo is None else modelo,
        ejemplo=base.x_train.head(3) if ejemplo is None else ejemplo,
        entrada=base.x_train,
        skops_trusted_types=trusted,
        de_nuevo=True,
    )
    return url.rsplit("/", 1)[-1]


def test_registrar_deja_version_linaje_y_alias(servidor):
    id_run = crear_candidato({"busqueda_run_id": "padre123", "aceptado": "true"})

    version = registro_p3.p3_registrar(id_run, "champion")

    cliente = mlflow.MlflowClient(servidor)
    nombre = cfg.P3_MODELO_REGISTRADO
    assert int(cliente.get_model_version_by_alias(nombre, "champion").version) == version
    registrada = cliente.get_model_version(nombre, version)
    assert registrada.tags["source_run_id"] == id_run
    assert registrada.tags["data_version"] == "abc123"
    assert registrada.tags["busqueda_run_id"] == "padre123"
    assert registrada.tags["aceptado"] == "true"
    assert registrada.tags["question"] == "p3"
    assert registrada.tags["fase"] == cfg.FASE_ENTREGA
    assert registrada.tags["train_silueta"] == "0.1400" and registrada.tags["valid_ari"] == "0.6800"
    assert "train_silueta=0.1400" in registrada.description
    assert cliente.get_registered_model(nombre).description


def test_el_alias_se_mueve_a_la_version_nueva(servidor):
    primera = registro_p3.p3_registrar(crear_candidato(), "champion")
    segunda = registro_p3.p3_registrar(crear_candidato(), "champion")

    alias = mlflow.MlflowClient(servidor).get_model_version_by_alias(
        cfg.P3_MODELO_REGISTRADO, "champion"
    )
    assert (primera, segunda) == (1, 2)
    assert int(alias.version) == 2


def test_verificar_carga_por_alias_y_asigna_grupos(servidor):
    registro_p3.p3_registrar(crear_candidato(), "champion")

    assert registro_p3.p3_verificar("champion") == 3


def test_verificar_carga_un_modelo_con_pesos_por_bloque(servidor):
    """El modelo ponderado lleva una función propia: debe cargar por alias sin depender de main."""
    x = p3_x_con_tres_grupos()
    modelo = train.p3_construir_modelo(COLUMNAS, 5, True, 3, 42).fit(x)
    id_run = crear_candidato(
        modelo=modelo,
        ejemplo=x.head(3),
        trusted=["tc5061_proyecto_equipo07.modeling.train.p3_aplicar_pesos"],
    )
    registro_p3.p3_registrar(id_run, "champion")

    assert registro_p3.p3_verificar("champion") == 3


def test_advierte_si_el_run_no_fue_aceptado(servidor):
    mensajes = []
    sumidero = logger.add(lambda m: mensajes.append(str(m)), level="WARNING")
    try:
        registro_p3.p3_registrar(crear_candidato({"aceptado": "false"}), "challenger")
    finally:
        logger.remove(sumidero)

    assert any("aceptado=false" in m for m in mensajes)


def test_no_registra_un_run_sin_modelo(servidor):
    mlflow.set_experiment(cfg.EXPERIMENTO_MLFLOW)
    with mlflow.start_run() as corrida:
        mlflow.log_metric("train_silueta", 0.1)

    with pytest.raises(ValueError, match="no tiene un modelo"):
        registro_p3.p3_registrar(corrida.info.run_id, "champion")


def test_no_registra_un_run_con_metricas_sin_conjunto(servidor):
    """Un run anterior al cambio de nombres (silueta, ari_valid) debe volver a registrarse."""
    mlflow.set_experiment(cfg.EXPERIMENTO_MLFLOW)
    base = base_sintetica()
    with mlflow.start_run() as corrida:
        mlflow.log_metric("silueta", 0.14)
        mlflow.sklearn.log_model(base.modelo, name="viejo", input_example=base.x_train.head(3))

    with pytest.raises(ValueError, match="train_silueta, valid_ari"):
        registro_p3.p3_registrar(corrida.info.run_id, "champion")


def test_no_registra_un_run_que_no_termino(servidor):
    mlflow.set_experiment(cfg.EXPERIMENTO_MLFLOW)
    with mlflow.start_run() as corrida:
        pass
    mlflow.MlflowClient(servidor).set_terminated(corrida.info.run_id, status="FAILED")

    with pytest.raises(ValueError, match="no terminó"):
        registro_p3.p3_registrar(corrida.info.run_id, "champion")


def test_champion_y_challenger_conviven_en_versiones_distintas(servidor):
    campeon = registro_p3.p3_registrar(crear_candidato(), "champion")
    retador = registro_p3.p3_registrar(crear_candidato(), "challenger")

    cliente = mlflow.MlflowClient(servidor)
    por_alias = {
        alias: int(cliente.get_model_version_by_alias(cfg.P3_MODELO_REGISTRADO, alias).version)
        for alias in ("champion", "challenger")
    }
    assert por_alias == {"champion": campeon, "challenger": retador} and campeon != retador


def test_verificar_con_un_alias_inexistente_falla(servidor):
    registro_p3.p3_registrar(crear_candidato(), "champion")

    with pytest.raises(mlflow.exceptions.MlflowException):
        registro_p3.p3_verificar("no_existe")


def test_etiquetas_de_la_version_usan_valores_por_defecto_si_el_run_no_los_trae():
    from types import SimpleNamespace

    corrida = SimpleNamespace(
        info=SimpleNamespace(run_id="abc"),
        data=SimpleNamespace(
            tags={}, params={}, metrics={"train_silueta": 0.5, "valid_ari": 0.25}
        ),
    )
    etiquetas = registro_p3.p3_etiquetas_version(corrida)
    assert etiquetas["commit"] == "desconocido" and etiquetas["data_version"] == "desconocido"
    assert etiquetas["question"] == "p3" and etiquetas["fase"] == cfg.FASE_ENTREGA
    assert etiquetas["source_run_id"] == "abc"
    # Las etiquetas de linaje opcionales solo aparecen si el run las tiene
    assert not {"aceptado", "referencia", "busqueda_run_id"} & set(etiquetas)


def test_descripcion_de_la_version_omite_las_metricas_que_el_run_no_tiene():
    from types import SimpleNamespace

    corrida = SimpleNamespace(
        info=SimpleNamespace(run_id="abc", run_name="candidato"),
        data=SimpleNamespace(tags={}, metrics={"train_silueta": 0.5}),
    )
    descripcion = registro_p3.p3_descripcion_version(corrida)
    assert "train_silueta=0.5000" in descripcion and "valid_ari" not in descripcion
    assert descripcion.startswith("candidato:") and descripcion.endswith("Run de origen: abc.")


def test_registrar_sin_servidor_de_mlflow_se_detiene(monkeypatch):
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", "")
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        registro_p3.registrar(run_id="abc", alias="champion")
