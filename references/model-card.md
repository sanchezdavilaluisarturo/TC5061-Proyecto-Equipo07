# Model Card: Proyecto MiBici (Equipo 07)

Documenta los tres modelos que el proyecto sirve en producción, uno por pregunta, con la estructura de las *model cards* de Mitchell et al. (2019): detalles del modelo, uso previsto, datos, métricas, consideraciones éticas, limitaciones y mantenimiento. Las cifras salen de los runs de MLflow y de las secciones del [README](../README.md) que se enlazan en cada apartado; no hay resultados nuevos aquí.

Estado del Model Registry al 24-nov-2026 (verificado en la interfaz del servidor del equipo, https://mlflow.equipo7mlops.com):

| Pregunta | Modelo registrado | Tarea | `champion` | `challenger` | Endpoint de la API |
|---|---|---|---|---|---|
| P1 | `mibici-equipo07-p1-demanda` | Regresión | v3 | Ninguno | `POST /p1/predecir` |
| P2 | `mibici-equipo07-p2-riesgo` | Clasificación (3 clases) | v4 | Ninguno | `POST /p2/predecir` |
| P3 | `mibici-equipo07-p3-segmentos` | Clustering | v4 | v5 | `POST /p3/segmentar` |

## Común a los tres modelos

| Apartado | Detalle |
|---|---|
| **Desarrolladores** | Equipo 07 del curso Operaciones de Aprendizaje Automático (TC5061), Tecnológico de Monterrey: Daniel Sánchez Hernández (P1), Fernando Arreola Treviño (P2) y Luis Arturo Sánchez Davila (P3) |
| **Fecha y versión del documento** | 24-nov-2026; describe las versiones de la tabla anterior |
| **Datos** | [Datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/), enero 2020 a septiembre 2026: 27,026,645 viajes después de la limpieza (59,912 registros retirados, 0.221 %). Partición temporal común con una brecha de 24 h antes de cada frontera: train 2020-01 a 2024-09, valid 2024-10 a 2025-09 y test 2025-10 a 2026-09. El mes de monitoreo y de los A/B es octubre 2026, que ningún modelo vio. Detalle en [Datos](../README.md#datos) y [Definición de los datos](definicion-datos.md) |
| **Versión de los datos** | `data_version` es el hash de los SHA-256 del manifiesto [`fuentes-datos.csv`](fuentes-datos.csv); va como parámetro en cada run y como etiqueta en cada versión del Registry. Un archivo distinto del portal se detecta en la ingesta |
| **Linaje en el Registry** | Cada versión lleva en sus etiquetas `commit`, `data_version`, `source_run_id`, `question`, `fase` y, si viene de una búsqueda, `busqueda_run_id`. Las versiones reentrenadas en el monitoreo agregan `base_version` y `base_source_run_id`; una promoción deja `ab_run_id` y `champion_anterior` en la versión promovida y `reemplazada_por` en la anterior. Detalle en [Cargar el modelo registrado](../README.md#cargar-el-modelo-registrado) |
| **Estado de producción** | Se usan alias y no etapas: `champion` es la versión que carga la API al arrancar (`models:/<nombre>@champion`) y `challenger` una versión que compite sin reemplazarla. Promover o regresar a una versión anterior (*rollback*) es mover el alias, sin reconstruir la imagen |
| **Privacidad** | Los viajes traen `usuario_id`, género y año de nacimiento. Ninguna de esas columnas entra a las tablas de modelado (minimización): los tres modelos trabajan con agregados por estación y hora, y no se publican valores individuales. Detalle en [Privacidad](../README.md#privacidad) |
| **Condiciones de uso de los datos** | El portal no publica licencia; se asume uso académico, no comercial y con cita de la fuente, y los datos no se suben al repositorio (`.gitignore` excluye `data/`) |
| **Monitoreo** | Drift con Evidently contra la distribución de entrenamiento y con control estacional (el mismo mes de años previos), y A/B testing pareado entre versiones con un protocolo fijado antes de ver los resultados. Detalle en [Monitoreo y drift](../README.md#monitoreo-y-drift) |
| **Costo de operación** | La API corre en un EC2 t3.medium (USD 0.0416 por hora) junto con MLflow; el costo por unidad de cada modelo está en su sección. Detalle en el [README](../README.md#finops-costo-por-inferencia) |
| **Limitaciones comunes** | Los datos no registran inventario de bicicletas por estación. Solo existe la nomenclatura de 2026-09, así que no se reconstruyen estaciones retiradas o reubicadas. La API no guarda un registro de sus predicciones: el monitoreo usa el portal, no el tráfico real. Cada A/B se evaluó con un solo mes (octubre 2026) |

## P1: demanda de viajes por estación-hora

| Apartado | Detalle |
|---|---|
| **Modelo** | `mibici-equipo07-p1-demanda` v3: `Pipeline` de scikit-learn con XGBoost y pérdida Poisson. Usa los hiperparámetros de la búsqueda con Optuna de 40 trials (la v2) y el nivel histórico de cada estación (`P1NivelEstacion`) |
| **Uso previsto** | Planeación de flota del operador: al cierre del día D predice los viajes que iniciarán en cada estación en cada una de las 20 horas de servicio de D+1 (5:00 a 0:59). La predicción por zona (municipio: GDL, Guadalajara; ZPN, Zapopan; TLQ, Tlaquepaque) es la suma de sus estaciones |
| **Fuera de alcance** | Horizontes de más de un día, predicciones dentro del mismo día, horas sin servicio (1:00 a 4:59) y el inventario de bicicletas. No se usa para decisiones sobre usuarios |
| **Entradas y salida** | 14 variables por estación-hora: `origen_id` (solo para el nivel de la estación), viajes de la misma hora 1 y 7 días antes, promedio de la misma hora en las 4 semanas anteriores, hora, día de la semana, fin de semana, mes, festivo, capacidad, latitud, longitud, zona y días desde la apertura. Salida: viajes esperados (conteo, nunca negativo) |
| **Datos de entrenamiento** | La v3 es un `clone` de la v2 (mismos hiperparámetros y semilla) entrenado desde cero con 2020-01 a 2026-09 sin brechas: 16,080,870 estación-horas |
| **Métrica principal** | MAE por estación-hora (error en viajes, directo para el operador); RMSE como complementaria. Sin MAPE: el 44 % de las estación-horas no tiene viajes |
| **Resultados** | Fase 2 (v2, entrenada con train): MAE de 0.9617 en valid y 0.9861 en test, contra 1.0189 y 1.0146 del baseline (promedio de 4 semanas). La v3 se entrenó con valid y test, así que su evaluación fuera de muestra es el A/B de octubre 2026: MAE de 1.1425 contra 1.2277 de la v2 (−6.9 %) y 1.2005 del baseline. En estaciones nuevas, 0.7095 contra 0.8293. Error diario en GDL (Guadalajara): 1581 viajes contra 2148.9 del baseline |
| **Linaje** | Run de origen `p1-xgb-challenger` (`2c8af50671024701a8a05c6cb6c23838`, commit `bd596b7`); versión base v2 (`eae9b6f4f3244161a13b3d230f32f70f`); promovida con el A/B `p1-ab-v2-vs-v3` (`c71d095c02194a6e85263f8b61145375`); `data_version` `c3b84e1ab511`, con la ventana en `rango_datos` |
| **Limitaciones** | Una estación sin historia recibe el nivel del sistema, así que se predice peor hasta acumular datos. El A/B es de un solo mes. La demanda de 2026 es la más alta de la serie (2026-09 es el máximo): un cambio de nivel así degradó a la v2 en octubre 2026 |
| **Costo** | USD 0.0020 por millón de predicciones (USD 0.0030 con el recargo de CPU del t3), medido con 40 usuarios en el EC2. Es una cota pesimista: el límite fue la red del generador de carga y no el servidor |
| **Mantenimiento** | Reentrenar cada mes con los mismos hiperparámetros cuando el portal publique el mes nuevo, y promover solo si el A/B lo respalda (`make monitoreo-p1-challenger`, `monitoreo-p1-ab` y `monitoreo-p1-promover`). Disparadores: drift de la demanda contra el mismo mes del año anterior o estaciones sin historia en entrenamiento. Detalle en [Monitoreo de P1](../README.md#monitoreo-de-p1) |

## P2: riesgo de estación

| Apartado | Detalle |
|---|---|
| **Modelo** | `mibici-equipo07-p2-riesgo` v4: `XGBClassifier` (`multi:softprob`) con los hiperparámetros de la búsqueda con Optuna de 15 trials (300 árboles, `max_leaves` 133, `learning_rate` 0.027) |
| **Uso previsto** | Equipo de rebalanceo: para cada estación y bloque de 30 minutos, la probabilidad de quedarse **sin bicis** o **sin anclajes** en las 2 horas siguientes. La clase final se decide con los umbrales de la v4 (0.296 sin bicis y 0.269 sin anclajes) |
| **Fuera de alcance** | El inventario real de la estación: el riesgo se aproxima con el cambio neto de viajes en la ventana futura frente a la capacidad (riesgo si alcanza el 50 %). Las estaciones con capacidad 0 no tienen una etiqueta válida |
| **Entradas y salida** | 17 variables: balance neto relativo de los últimos 1, 2, 4 y 8 bloques, salidas y entradas recientes, acumulado del día, capacidad, ubicación, hora cíclica, fin de semana y la frecuencia histórica de cada riesgo "hasta ayer" en la misma estación, tipo de día y hora. Salida: probabilidad de cada clase (sin riesgo, sin bicis y sin anclajes) y la clase decidida |
| **Datos de entrenamiento** | La v4 es un `clone` de la v3 entrenado con 2020-01 a 2026-09 sin brechas: 28,796,364 estación-bins (52 % más que la v3). Sus umbrales los eligió un modelo auxiliar igual, entrenado con el 25 % de los datos |
| **Métrica principal** | Average precision (PR-AUC) de riesgo, porque las clases son raras (en test, 0.74 % sin bicis y 1.03 % sin anclajes). En el A/B decide el log-loss por fila, que no depende de los umbrales |
| **Resultados** | Fase 2 (v3): AP de riesgo de 0.4324 en valid y 0.3342 en test, prácticamente empatada con el baseline histórico en test (0.3329). A/B de octubre 2026: log-loss de 0.0664 contra 0.0715 de la v3 (−7.1 %), AP de riesgo de 0.418 contra 0.353 (baseline 0.355), AUC de 0.941 contra 0.934, y calibración de sin bicis de 0.94 contra 1.39 |
| **Linaje** | Run de origen `p2-xgb-challenger` (`20be0d7ce2374ac49b2dff58a7c67ec1`, commit `792a843`); versión base v3 (`db1ced6a7d8b4487974deb64bc82b047`); promovida con el A/B `p2-ab-v3-vs-v4` (`52f0d024c2a84978a34ab56185af0ffe`) |
| **Limitaciones** | El drift principal es de la etiqueta (*label shift*): cada vez hay menos episodios de riesgo, y un modelo sin reentrenar sobreestima el riesgo. En las estaciones nuevas el modelo subestima el riesgo sin anclajes: F1 de 0.20 contra 0.36 del baseline histórico. La etiqueta es una aproximación, no una medición del inventario |
| **Costo** | USD 0.0042 por millón de predicciones (USD 0.0061 con el recargo), en el punto de operación de 40 usuarios con p95 por debajo de 400 ms |
| **Mantenimiento** | Reentrenar cada mes con los mismos hiperparámetros y promover solo con el A/B (`make monitoreo-p2-challenger`, `monitoreo-p2-ab` y `monitoreo-p2-promover`). Disparadores: la ventaja de AP sobre el baseline llega a cero o la calibración se aleja de 1 varios meses seguidos. Para las estaciones nuevas, revisar variables que no dependan de su propia historia. Detalle en [Monitoreo de P2](../README.md#monitoreo-de-p2-riesgo-de-estación) |

## P3: segmentación de estaciones

| Apartado | Detalle |
|---|---|
| **Modelo** | `mibici-equipo07-p3-segmentos` v4: `Pipeline` de scikit-learn con `StandardScaler`, PCA de 11 componentes y K-Means con `k = 2` (semilla 42), elegido con la búsqueda con Optuna de 40 trials |
| **Uso previsto** | Planeación urbana y expansión: agrupar estaciones con patrones de uso parecidos y ubicarlas en el mapa. En el modelo base, un grupo tiene pico matutino a las 8:00 y es más periférico; el otro, pico vespertino a las 18:00, más cerca del centro y tiende a perder bicicletas |
| **Fuera de alcance** | Decidir ampliaciones o reubicaciones solo con dos grupos: la granularidad no basta. Las etiquetas 0 y 1 de K-Means son arbitrarias, y la lectura "residencial frente a zonas de trabajo" es una hipótesis por verificar |
| **Entradas y salida** | 44 variables por estación: proporción de salidas por hora en días laborales y en fin de semana (20 horas de servicio cada uno), volumen diario, balance neto sobre salidas, proporción de viajes circulares y duración mediana. Salida: el grupo de cada estación |
| **Datos de entrenamiento** | Perfiles de train de las 362 estaciones con al menos 1,000 salidas. Las estaciones abiertas después de train no entraron al ajuste, pero la API les asigna grupo a partir de su perfil |
| **Métrica principal** | Silueta, medida en el espacio estandarizado del baseline; estabilidad con el ARI al reasignar las estaciones con el año de validación. P3 no usa test |
| **Resultados** | Silueta de 0.1454 en train contra 0.1449 del baseline, y ARI de 0.6726 contra 0.6561. La ventaja es marginal (+0.0005, del orden de la variación entre semillas): se aceptó porque cumple los criterios fijados de antemano, no porque segmente mejor. A/B de octubre 2026 contra la v5: silueta de 0.0728 contra 0.0710, Wilcoxon p = 0.83, así que la v4 se queda |
| **Linaje** | Run de origen `p3-pca_kmeans_optuna` (`872f2218deaa422e80ea197c325f7a6d`, commit `4d40d14`), de la búsqueda `2c68e9b937d249a383ce49dd245e6646`. La v5 (`challenger`, run `p3-challenger-datos-2026-09`, `2b3696b1f97f48cd86923849b5f8eac2`, commit `868f52e`) es la v4 reentrenada con 399 estaciones de 2020-01 a 2026-09 (silueta de 0.1618 sobre esas estaciones, no comparable con la de la v4); al registrarla, el alias `challenger` pasó de la v3 (el baseline) a la v5 |
| **Limitaciones** | La estructura es débil (silueta 0.145): las estaciones forman más un continuo que dos grupos separados. La silueta baja por periodo (0.145 en train, 0.132 en valid y 0.103 en test), y con estos datos no se puede separar el cambio de comportamiento del cambio de composición de la red (de 362 a 457 estaciones) |
| **Costo** | USD 0.0048 por millón de estaciones enviando la red completa en una llamada con 5 usuarios, y USD 0.0011 en el tope; con una llamada por estación sube a USD 0.448 |
| **Mantenimiento** | No reentrenar por ahora: ni datos más recientes, ni otra ventana de entrenamiento, ni otra preparación de las variables superaron a la v4. Revisión mensual del drift y la silueta, disparadores definidos de antemano y reentrenamiento anual. La promoción sigue siendo una decisión humana con el A/B delante, porque un modelo sin etiqueta no se valida solo. Detalle en [Monitoreo de P3](../README.md#monitoreo-de-p3-segmentación-de-estaciones) |

## Consideraciones éticas y gobernanza

- **Ninguna decisión es sobre personas.** Los tres modelos trabajan con agregados por estación y no reciben datos personales. Sus decisiones (flota, rebalanceo y planeación) afectan a zonas de la ciudad, no a usuarios.
- **Riesgo de inequidad entre zonas.** Los errores no son iguales en toda la red: la zona con menos viajes (TLQ, Tlaquepaque) tiene el mayor error relativo de P1, y en las estaciones nuevas, sin historia propia, el champion de P1 llegó a quedar por debajo del baseline (v2 en octubre 2026) y el de P2 sigue por debajo en sin anclajes. Usar las predicciones sin revisión podría llevar menos recursos a donde hay menos historia. Por eso las predicciones apoyan la decisión del operador y no la sustituyen.
- **Criterios fijados antes de ver los resultados.** La métrica de selección, los criterios de aceptación y la regla de cada A/B se fijaron antes de correr, test no se usó para elegir y, en P3, el nivel se corrigió por el número de candidatos probados.
- **Promoción con evidencia.** Un modelo nuevo solo reemplaza al vigente si gana el A/B sobre datos que ninguno vio; la promoción es un paso aparte que deja su evidencia en el Registry.
- **Trazabilidad.** Cada versión se puede reproducir con su commit, su `data_version` y su run de origen.

## Referencias

Huyen, C. (2022). *Designing machine learning systems: An iterative process for production-ready applications*. O'Reilly Media.

MiBici. (s.f.). *Datos abiertos* [Conjunto de datos]. Sistema de Bicicleta Pública del Área Metropolitana de Guadalajara. https://mibici.net/es/datos-abiertos/

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). https://doi.org/10.1145/3287560.3287596

Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J., Lavoillotte, A., Miyazaki, M., & Heidmann, L. (2020). *Introducing MLOps: How to scale machine learning in the enterprise*. O'Reilly Media.
