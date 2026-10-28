"""Evaluación de P3: criterios de aceptación en sus bordes, selección y métricas por semilla."""

import pandas as pd
import pytest
from test_tuning_p3 import base_sintetica

from tc5061_proyecto_equipo07.modeling import train

BASELINE = {"silueta": 0.145, "ari_valid": 0.656}


def elegido(silueta=0.150, ari_valid=0.676, tam_min=42) -> pd.Series:
    return pd.Series({"silueta": silueta, "ari_valid": ari_valid, "tam_min": tam_min})


def criterios(elegido_: pd.Series, tam_min_aceptable: int = 20) -> tuple[list[bool], bool]:
    resultado, aceptado = train.p3_evaluar_criterios(elegido_, BASELINE, tam_min_aceptable)
    return list(resultado.values()), aceptado


def test_p3_aceptado_solo_si_cumple_los_tres_criterios():
    assert criterios(elegido()) == ([True, True, True], True)


def test_p3_la_silueta_debe_superar_al_baseline_no_igualarlo():
    assert criterios(elegido(silueta=BASELINE["silueta"]))[0][0] is False
    assert criterios(elegido(silueta=BASELINE["silueta"] + 1e-9))[0][0] is True


def test_p3_la_estabilidad_puede_igualar_al_baseline():
    assert criterios(elegido(ari_valid=BASELINE["ari_valid"]))[0][1] is True
    assert criterios(elegido(ari_valid=BASELINE["ari_valid"] - 1e-9)) == ([True, False, True], False)


def test_p3_el_tamano_minimo_puede_igualar_al_umbral():
    assert criterios(elegido(tam_min=20), 20)[0][2] is True
    assert criterios(elegido(tam_min=19), 20) == ([True, True, False], False)


def test_p3_el_nombre_de_cada_criterio_lleva_su_referencia():
    nombres = list(train.p3_evaluar_criterios(elegido(), BASELINE, 20)[0])
    assert "0.145" in nombres[0] and "0.656" in nombres[1] and "20" in nombres[2]


def resumen_de_configuraciones() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ponderar": [False, True, False],
            "componentes": [8, 4, 6],
            "k": [3, 3, 3],
            "silueta": [0.20, 0.20, 0.10],
            "ari_valid": [0.70, 0.70, 0.70],
            "tam_min": [25, 25, 25],
        }
    )


def test_p3_con_silueta_empatada_gana_la_de_menos_componentes():
    resultado, _, _ = train.p3_seleccionar_alternativo(resumen_de_configuraciones(), 0.5, 20)
    assert resultado["componentes"] == 4


def test_p3_cumplir_es_inclusivo_en_estabilidad_y_en_tamano():
    resumen = resumen_de_configuraciones().assign(ari_valid=0.5, tam_min=20)
    _, cumplen, _ = train.p3_seleccionar_alternativo(resumen, 0.5, 20)
    assert len(cumplen) == 3  # justo en el borde: ARI igual al del baseline y tamaño igual al umbral


def test_p3_un_solo_candidato_se_elige_aunque_no_cumpla():
    unico = resumen_de_configuraciones().iloc[[2]].assign(tam_min=1)
    resultado, cumplen, reserva = train.p3_seleccionar_alternativo(unico, 0.99, 20)
    assert resultado["componentes"] == 6 and cumplen.empty and len(reserva) == 1


def test_p3_promediar_semillas_promedia_y_toma_el_peor_caso_en_tamano():
    corridas = pd.DataFrame(
        {
            "ponderar": [False] * 4,
            "componentes": [5, 5, 8, 8],
            "k": [3, 3, 3, 3],
            "semilla": [42, 1, 42, 1],
            "silueta": [0.10, 0.20, 0.30, 0.50],
            "davies_bouldin": [1.0, 2.0, 1.0, 1.0],
            "calinski_harabasz": [10.0, 20.0, 30.0, 30.0],
            "tam_min": [30, 12, 25, 40],
            "ari_valid": [0.6, 0.8, 0.9, 0.7],
        }
    )
    resumen = train.p3_promediar_semillas(corridas).set_index("componentes")
    assert len(resumen) == 2
    assert resumen.loc[5, "silueta"] == pytest.approx(0.15)
    assert resumen.loc[5, "silueta_min"] == 0.10
    assert resumen.loc[5, "tam_min"] == 12  # el peor caso, no el promedio (21)
    assert resumen.loc[8, "ari_valid"] == pytest.approx(0.8)
    assert resumen.loc[8, "ari_min"] == 0.7


@pytest.mark.parametrize(
    ("semilla", "cantidad", "esperadas"),
    [
        (42, 3, [42, 1, 2]),
        (2, 3, [2, 1, 3]),  # la del proyecto va primero y no se repite después
        (500, 2, [500, 1]),  # una semilla fuera de 1..n no desplaza a las demás
        (42, 1, [42]),
        (42, 0, [42]),  # nunca queda sin semillas
    ],
)
def test_p3_semillas_de_la_busqueda(semilla, cantidad, esperadas):
    assert train.p3_semillas_busqueda(semilla, cantidad) == esperadas


def test_p3_evaluar_modelo_coincide_con_las_metricas_del_baseline():
    base = base_sintetica()
    x_valid, posiciones = train.p3_entradas_estabilidad(base)
    metricas = train.p3_evaluar_modelo(base.modelo, base.z_train, x_valid, posiciones)
    assert metricas["silueta"] == pytest.approx(base.metricas["silueta"])
    assert metricas["ari_valid"] == pytest.approx(base.metricas["ari_valid"])
    assert metricas["tam_min"] == base.tamanos.min()
    assert -1 <= metricas["silueta"] <= 1 and -1 <= metricas["ari_valid"] <= 1


def test_p3_las_estaciones_de_estabilidad_estan_en_train_y_en_valid():
    base = base_sintetica()
    x_valid, posiciones = train.p3_entradas_estabilidad(base)
    assert len(x_valid) == len(posiciones) == len(base.comunes)
    assert set(base.comunes) <= set(base.caracteristicas_train.index)
    assert set(base.comunes) <= set(base.caracteristicas_valid.index)
    assert list(x_valid.columns) == base.columnas_modelo
