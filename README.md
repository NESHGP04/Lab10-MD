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
│   ├── 03_Implementacion.ipynb     Modelos: GMM-SS, CKM, baseline
│   ├── 04_diseno_experimental.ipynb Harness experimental
│   ├── 05_Sensibilidad.ipynb       Análisis de hiperparámetros
│   └── 06_Visualizacion.ipynb      Gráficas finales y discusión
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

## Visualización y análisis de resultados

Se realizó la visualización final de resultados para comparar el baseline supervisado, GMM semi-supervisado y Constrained K-means en los escenarios de 5%, 10% y 20% de datos etiquetados.

- Notebook final: `notebooks/06_Visualizacion.ipynb`
- Figuras generadas: `results/figures/`
- Discusión crítica: `docs/fase5_visualizacion_discusion.md`

Para ejecutar esta parte:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/06_Visualizacion.ipynb
```

## Notas sobre dependencias

- `cop-kmeans`: paquete para Constrained K-means. Si la instalación falla, se puede usar `active-semi-supervised-clustering` como alternativa.
