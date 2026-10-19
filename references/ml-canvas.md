# ML Canvas — Proyecto MiBici (Equipo 7)

Formulación del problema del proyecto integrador de **Operaciones de Aprendizaje Automático (TC5061)**. Para cada pregunta se describe: 1) qué se predice o se agrupa, con qué granularidad y horizonte; 2) por qué es relevante para la operación de MiBici; 3) quién usaría el resultado y qué decisión tomaría con él.

> **Estado:** formulación de la primera entrega; las decisiones del equipo están en *Decisiones tomadas*, al final.

## ML Canvas del proyecto

| Bloque | Contenido |
|---|---|
| **Propuesta de valor** | Pasar de operar MiBici de forma reactiva a anticipar la demanda, los faltantes y los patrones de uso por estación |
| **Decisiones** | Cuántas bicis y camiones de rebalanceo asignar, qué estaciones atender primero y dónde ampliar o reubicar |
| **Tarea de ML** | P1 regresión, P2 clasificación y P3 clustering (no supervisado) |
| **Fuentes de datos** | Viajes mensuales (`datos_abiertos_AAAA_MM`, 27,086,557 viajes de ene-2020 a sep-2026) y nomenclatura de estaciones (484 estaciones con coordenadas y capacidad). Ver la sección *Datos* del README |
| **Recolección de datos** | Descarga mensual del portal de datos abiertos de MiBici. El archivo del mes tarda en publicarse y el portal puede retirar meses (2023-04 y 2023-05) |
| **Predicciones** | Por lotes (por ejemplo, una vez al día), no en tiempo real |
| **Construcción de modelos** | Un baseline sencillo por pregunta y al menos un candidato: HGB y XGBoost Poisson (P1), HistGradientBoosting (P2), K-Means y PCA + K-Means (P3) |
| **Features** | Estación, hora, día de la semana, mes, festivos, rezagos de demanda, balance neto, capacidad y ubicación |
| **Evaluación offline** | Partición **temporal** común (entrenar con el pasado, probar con el futuro): MAE y RMSE para P1; PR-AUC, recall y F1 para P2; silueta, estabilidad e interpretación para P3 |
| **Evaluación y monitoreo en vivo** | Error semanal contra lo observado, deriva de la demanda y alerta si un mes llega incompleto |

## P1: demanda de viajes (regresión)

**1) Qué se predice**
- **Variable:** número de viajes que **inician** en una estación.
- **Granularidad:** estación × hora de servicio (5:00 a 0:59). La zona (municipio: GDL, ZPN o TLQ) es la suma de sus estaciones.
- **Horizonte:** el día siguiente: al cierre del día D se predicen las 20 horas de D+1.
- **Objetivo:** `COUNT(viaje_id)` agrupado por `origen_id` y hora de `inicio_del_viaje`.

**2) Por qué es relevante**
- La demanda varía mucho (de 129,276 viajes en abril de 2020 a 490,056 en septiembre de 2026), así que un plan fijo desperdicia flota o deja usuarios sin servicio.

**3) Quién lo usa y qué decide**
- **Usuario:** operador del sistema (planeación de flota).
- **Decisión:** cuántas bicis tener disponibles por estación y turno.

## P2: riesgo de quedarse sin bicis o sin anclajes (clasificación)

**1) Qué se predice**
- **Variable:** si una estación llegará a faltante de bicis o de anclajes.
- **Granularidad:** estación × bloque de 30 minutos.
- **Horizonte:** las próximas 2 horas (4 bloques).
- **Objetivo:** tres clases (`sin riesgo`, `sin bicis`, `sin anclajes`): hay riesgo si el cambio neto (llegadas − salidas) en la ventana alcanza el 50 % de `dpcapacity` (`P2_ALFA`); si ocurren los dos, gana el de mayor estrés relativo.

**2) Por qué es relevante**
- Una estación vacía pierde un viaje y una llena impide terminarlo; ambas dañan la percepción del servicio.

**3) Quién lo usa y qué decide**
- **Usuario:** equipo de rebalanceo.
- **Decisión:** a qué estaciones enviar bicis o retirarlas, y en qué orden.

**Limitaciones**
- Los datos **no traen inventario**, por lo que la etiqueta es un proxy: el balance acumulado necesita un punto de partida desconocido y no registra los movimientos de rebalanceo que ya hizo la operación.
- Las 4 estaciones con `dpcapacity = 0` se excluyen.
- Dado que se evalúa el riesgo por estación con una ventana de 2 horas en el futuro, no es posible clasificar el riesgo de estaciones en las últimas 2 horas hábiles diarias.

## P3: grupos de estaciones con patrones de uso (clustering)

**1) Qué se agrupa**
- **Entidad:** estaciones con al menos 1,000 salidas en entrenamiento (362 de 475 con viajes).
- **Variables:** perfil horario laboral y de fin de semana, volumen, balance neto, viajes circulares y duración mediana; la ubicación no entra al clustering y se usa para describir los grupos.
- **Horizonte:** no es predictivo; se recalcula por periodo y se mide su estabilidad con el año de validación.
- **Salida:** etiqueta de grupo por estación.

**2) Por qué es relevante**
- Permite aplicar una política común a estaciones parecidas (por ejemplo, las que sirven al trabajo, las residenciales o las de ocio).

**3) Quién lo usa y qué decide**
- **Usuario:** planeación urbana y expansión.
- **Decisión:** dónde ampliar capacidad, qué estaciones reubicar o abrir, y qué reglas de rebalanceo aplicar por grupo.

## Resumen

| | P1 | P2 | P3 |
|---|---|---|---|
| **Tarea** | Regresión | Clasificación | Clustering |
| **Entidad** | Estación-hora | Estación-bloque de 30 min | Estación |
| **Horizonte** | Día siguiente | 2 h | Por periodo |
| **Usuario** | Planeación de flota | Rebalanceo | Planeación urbana |
| **Métrica** | MAE, RMSE | PR-AUC, recall, F1 | Silueta, ARI |

## Decisiones tomadas

- Horizonte de P1: el día siguiente, con rezagos de 1 día o más.
- Riesgo de P2: el cambio neto (llegadas − salidas) en las próximas 2 horas supera el 50 % de la capacidad (`P2_ALFA`).
- 2023-04 y 2023-05: el portal los volvió a publicar; el rango de 81 meses está completo.
- Una pregunta por integrante: P1 Daniel Sánchez Hernández, P2 Fernando Arreola Treviño y P3 Luis Arturo Sánchez Davila.
