# Lab 10 — Minería de Datos: Métodos generativos y clustering con restricciones

**Grupo 5.** Algoritmos asignados:
- Gaussian Mixture Models semi-supervisados (EM con datos parcialmente etiquetados).
- Constrained K-means (must-link / cannot-link).

## Dataset

[Dry Bean Dataset (UCI)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).
13,543 instancias × 16 features numéricas × 7 clases (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA).

El preprocesamiento (estandarización + label encoding) ya está aplicado en `data/processed/dataset_clean.csv`. Notebooks originales de EDA y preprocesamiento en `raw_data/`.

## Estructura del repo

```
.
├── data/processed/         Dataset limpio (input de los notebooks)
├── raw_data/               EDA y preprocesamiento originales
├── notebooks/              Notebooks de implementación y experimentación
│   ├── 00_data_loader.ipynb        Carga + simulación semi-supervisada
│   ├── 03_implementacion.ipynb     (Xavi) Modelos: GMM-SS, CKM, baseline
│   ├── 04_diseno_experimental.ipynb Harness experimental
│   ├── 05_sensibilidad.ipynb       (Ian) Análisis de hiperparámetros
│   └── 06_visualizacion.ipynb      (Javi) Gráficas finales
├── results/
│   ├── figures/            PNGs de las visualizaciones
│   └── metrics/            CSVs con métricas crudas y agregadas
├── docs/                   Marco teórico y notas de cada fase
├── requirements.txt
└── README.md
```

## Cómo reproducir

```bash
pip install -r requirements.txt
jupyter notebook
# Ejecutar en orden: 00 → 03 → 04 → 05 → 06
```

## Notas sobre dependencias

- `cop-kmeans`: paquete para Constrained K-means. Si la instalación falla, Xavi puede usar `active-semi-supervised-clustering` como alternativa.

## Equipo

- Camila — Marco teórico
- Marines — Dataset, EDA, preprocesamiento
- Esteban — Setup del repo, harness experimental, presentación
- Xavi — Implementación de modelos
- Ian — Experimentación y análisis de sensibilidad
- Javi — Visualización y discusión crítica
