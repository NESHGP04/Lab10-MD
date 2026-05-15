# Fase 3 — Diseño Experimental

## 3.1 Protocolo de evaluación

- **Split:** 70% train / 30% test, estratificado por clase con `train_test_split(stratify=y)`.
- **Justificación del 70/30:** El dataset cuenta con ~13,543 filas, por lo que el 30% de test corresponde a ~4,063 muestras, una cantidad estadísticamente significativa para estimar métricas de generalización con precisión. El 70% de train (~9,480 muestras) es suficiente para que los modelos semi-supervisados tengan material de aprendizaje incluso cuando solo se revelan el 5% de las etiquetas (~474 etiquetas visibles). La estratificación protege especialmente a la clase minoritaria BOMBAY (522 instancias totales), garantizando que esté representada de forma proporcional tanto en train como en test.
- **Semillas:** 5 seeds (`random_state ∈ {0, 1, 2, 3, 4}`) por configuración. Justificación: con 5 semillas independientes se puede reportar media ± desviación estándar, lo cual captura la variabilidad debida a la inicialización aleatoria de los modelos (especialmente relevante para GMM y K-means, que son sensibles al punto de partida). Usar más semillas ofrecería estimaciones más precisas pero con coste computacional proporcional; 5 es el mínimo razonable para detectar alta varianza sin que el experimento resulte impracticable en un notebook interactivo.
- **Simulación semi-supervisada:** se aplica **exclusivamente al conjunto de train**. El test mantiene todas sus etiquetas para servir como ground truth objetivo. La función `make_semi_supervised` estratifica la selección de etiquetas visibles por clase, asegurando que cada una de las 7 clases tenga al menos 1 etiqueta visible incluso a la fracción más baja (5%).

---

## 3.2 Fracciones de etiquetas evaluadas

Se evalúan tres niveles de supervisión parcial para estudiar cómo varía el desempeño a medida que se revelan más etiquetas:

| Fracción | Etiquetas visibles en train (aprox.) | Contexto |
|---|---|---|
| 5%  | ~474 / 9,480 | Escenario de supervisión mínima |
| 10% | ~948 / 9,480 | Escenario de referencia principal |
| 20% | ~1,896 / 9,480 | Escenario con supervisión moderada |

La fracción del **10%** es el punto de referencia central porque es el más utilizado en la literatura de aprendizaje semi-supervisado para conjuntos de datos tabulares de tamaño mediano. Las fracciones del 5% y 20% permiten estudiar el comportamiento de los modelos en los extremos y trazar una curva de aprendizaje en función de la supervisión.

---

## 3.3 Métricas

- **Accuracy**: métrica principal por compatibilidad con el enunciado del laboratorio y con la mayoría de papers de referencia en clustering semi-supervisado. Intuitiva e interpretable directamente.
- **F1-macro**: promedia el F1 de cada clase con el mismo peso independientemente del tamaño de clase. Es fundamental para este dataset porque BOMBAY tiene ~4× menos instancias que DERMASON; la accuracy puede ser alta aunque el modelo abandone completamente la clase minoritaria, pero el F1-macro lo detectaría.
- **Recall-macro**: complementa al F1-macro focalizando en la capacidad de encontrar instancias de cada clase. Útil para detectar el fenómeno de "colapso de cluster" donde un modelo asigna casi todas las predicciones a unas pocas clases dominantes.
- **Tiempo de entrenamiento (s)**: medido con `time.perf_counter()` alrededor del bloque `fit`. Permite discutir el coste computacional adicional de los algoritmos semi-supervisados frente al baseline supervisado puro. Se espera que GMM-SS sea considerablemente más lento que la regresión logística baseline.

---

## 3.4 Variables controladas vs variadas

| Controladas (fijas) | Variadas (factores experimentales) |
|---|---|
| Split 70/30 estratificado | `label_fraction ∈ {0.05, 0.10, 0.20}` |
| `random_state` (5 seeds por config) | Modelo: baseline supervisado / GMM-SS / CKM |
| Preprocesamiento (StandardScaler ya aplicado) | Hiperparámetros del modelo (grid search aparte) |
| Dataset completo sin modificaciones | — |
| Función `make_semi_supervised` con `stratify=True` | — |

La separación entre variables controladas y variadas es crítica para interpretar correctamente los resultados: cualquier diferencia de desempeño entre modelos puede atribuirse al algoritmo (y no a diferencias en el split o el preprocesamiento), y cualquier diferencia entre fracciones puede atribuirse a la cantidad de supervisión disponible.

---

## 3.5 Análisis de sensibilidad

A fracción fija de **10%** se barren los siguientes hiperparámetros para entender la robustez de cada modelo frente a sus opciones de configuración:

**GMM semi-supervisado** — `run_grid_search(SemiSupervisedGMM, GMM_GRID, X, y, label_fraction=0.10, n_seeds=3)`:

```python
GMM_GRID = {
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
    'reg_covar': [1e-6, 1e-4, 1e-2],
    'n_init': [1, 3],
}
# 4 × 3 × 2 = 24 combinaciones × 3 seeds = 72 corridas
```

- `covariance_type`: controla la flexibilidad geométrica de las gaussianas. `full` es el más expresivo pero requiere más datos; `spherical` el más restrictivo pero más estable numéricamente.
- `reg_covar`: regularización para evitar matrices de covarianza singulares. Valores bajos dan más libertad al modelo pero pueden causar colapso numérico.
- `n_init`: número de inicializaciones del EM. Más inicializaciones reducen la probabilidad de quedar atrapado en un mínimo local.

**Constrained K-means** — `run_grid_search(ConstrainedKMeans, CKM_GRID, X, y, label_fraction=0.10, n_seeds=3)`:

```python
CKM_GRID = {
    'max_constraints': [50, 200, 500, 1000],
    'max_iter': [100, 300],
}
# 4 × 2 = 8 combinaciones × 3 seeds = 24 corridas
```

- `max_constraints`: número máximo de restricciones must-link/cannot-link derivadas de las etiquetas visibles. Mayor número de restricciones debería mejorar la calidad del clustering pero incrementa el coste por iteración.
- `max_iter`: número máximo de iteraciones del algoritmo. Relevante para detectar si el modelo converge prematuramente.

---

## 3.6 Detección de overfitting/underfitting

Para cada configuración del grid search se reporta accuracy en train (calculado sobre las muestras etiquetadas del train) y en test. Las tres situaciones posibles son:

| Patrón | Diagnóstico | Acción sugerida |
|---|---|---|
| `acc_train >> acc_test` | Overfitting | Aumentar `reg_covar`, reducir `n_components`, agregar regularización |
| Ambos bajos (< 0.6) | Underfitting | Aumentar `n_components`, usar `covariance_type='full'`, más iteraciones |
| `acc_train ≈ acc_test` y altos | Bien ajustado | Configuración candidata para el experimento principal |

Este análisis es particularmente relevante para GMM porque la estimación de la covarianza completa en 16 dimensiones puede sobreajustar con pocos puntos etiquetados (5% ≈ 474 puntos).

---

## 3.7 Outputs

Los siguientes archivos CSV se generan durante la experimentación:

| Archivo | Generado por | Contenido |
|---|---|---|
| `results/metrics/metrics_smoke.csv` | Equipo | Verificación rápida del harness: 2 modelos × 2 fracciones × 2 seeds = 8 filas |
| `results/metrics/metrics_full.csv` | Equipo | Experimento principal: 3 modelos × 3 fracciones × 5 seeds = 45 filas |
| `results/metrics/sensitivity_gmm.csv` | Equipo | Grid search GMM: 24 combinaciones × 3 seeds = 72 filas |
| `results/metrics/sensitivity_ckm.csv` | Equipo | Grid search CKM: 8 combinaciones × 3 seeds = 24 filas |

Todos los CSV incluyen las columnas `model`, `label_fraction`, `seed`, `accuracy`, `f1_macro`, `recall_macro`, `train_time_s`, `error`. Los archivos de grid search incluyen adicionalmente columnas por cada hiperparámetro de la grilla correspondiente.

---

## 3.8 Baseline y comparación

El experimento compara tres modelos:

1. **Baseline supervisado (LogisticRegression)**: entrena únicamente con las muestras etiquetadas (ignora las `-1`). Establece el piso de desempeño: cualquier algoritmo semi-supervisado debe superar este resultado para justificar su complejidad adicional.
2. **GMM semi-supervisado (EM con etiquetas parciales)**: modelo generativo que usa tanto los puntos etiquetados como los no etiquetados durante el EM.
3. **Constrained K-means**: clustering con restricciones must-link/cannot-link derivadas de las etiquetas visibles.

La hipótesis principal del laboratorio es que los modelos semi-supervisados superarán al baseline supervisado, especialmente en la fracción baja (5%), donde el baseline dispone de muy poca información de supervisión.
