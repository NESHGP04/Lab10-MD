"""Harness experimental para Lab 10, Fase 4."""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_repo_root() -> Path:
    """Resuelve la raiz del repo desde notebooks, src o el directorio raiz."""
    cwd = Path.cwd()
    if cwd.name in {"notebooks", "src", "docs"}:
        return cwd.parent
    return cwd


def load_dataset(repo_root: Path | None = None):
    """Carga Dry Bean preprocesado y su mapeo de clases."""
    repo_root = repo_root or get_repo_root()
    data_path = repo_root / "data" / "processed" / "dataset_clean.csv"
    mapping_path = repo_root / "data" / "processed" / "class_mapping.json"

    df = pd.read_csv(data_path)
    feature_names = [c for c in df.columns if c not in ("Class", "Class_name")]
    X = df[feature_names].to_numpy()
    y = df["Class"].to_numpy(dtype=int)

    with mapping_path.open() as f:
        class_names = {int(k): v for k, v in json.load(f).items()}

    return X, y, feature_names, class_names


def make_semi_supervised(y, label_fraction, random_state=42, stratify=True):
    """Oculta etiquetas y deja visibles solo `label_fraction`."""
    rng = np.random.default_rng(random_state)
    n = len(y)
    labeled_mask = np.zeros(n, dtype=bool)

    if stratify:
        for c in np.unique(y):
            idx_c = np.where(y == c)[0]
            n_keep = max(1, int(round(label_fraction * len(idx_c))))
            chosen = rng.choice(idx_c, size=n_keep, replace=False)
            labeled_mask[chosen] = True
    else:
        n_keep = max(1, int(round(label_fraction * n)))
        chosen = rng.choice(n, size=n_keep, replace=False)
        labeled_mask[chosen] = True

    y_partial = y.copy()
    y_partial[~labeled_mask] = -1
    return y_partial, labeled_mask


def make_semi_supervised_split(
    X,
    y,
    label_fraction,
    test_size=0.30,
    random_state=42,
):
    """Split estratificado y simulacion semi-supervisada solo sobre train."""
    X_train, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    y_train_partial, _ = make_semi_supervised(
        y_train_full,
        label_fraction=label_fraction,
        random_state=random_state,
        stratify=True,
    )
    return X_train, X_test, y_train_partial, y_train_full, y_test


def evaluate_model(y_true, y_pred, prefix=""):
    """Calcula metricas multiclase estandar."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if prefix:
        return {f"{prefix}_{key}": value for key, value in metrics.items()}
    return metrics


def _run_single(model, X_train, y_train_partial, y_train_full, X_test, y_test):
    """Entrena, predice train/test y devuelve metricas y tiempo."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train_partial)
    train_time = time.perf_counter() - t0

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    metrics = evaluate_model(y_test, y_pred_test)
    metrics.update(evaluate_model(y_train_full, y_pred_train, prefix="train"))
    metrics["generalization_gap"] = metrics["train_accuracy"] - metrics["accuracy"]
    metrics["train_time_s"] = train_time
    metrics["error"] = None
    return metrics


def run_experiment(
    model_factory: Callable,
    model_name,
    X,
    y,
    label_fractions=(0.05, 0.10, 0.20),
    n_seeds=5,
    test_size=0.30,
    verbose=True,
):
    """Ejecuta el experimento principal por fraccion y seed."""
    rows = []
    pbar = tqdm(
        total=len(label_fractions) * n_seeds,
        desc=model_name,
        disable=not verbose,
    )

    for label_fraction in label_fractions:
        for seed in range(n_seeds):
            X_train, X_test, y_train_partial, y_train_full, y_test = (
                make_semi_supervised_split(
                    X,
                    y,
                    label_fraction=label_fraction,
                    test_size=test_size,
                    random_state=seed,
                )
            )
            row = {
                "model": model_name,
                "label_fraction": label_fraction,
                "seed": seed,
            }
            try:
                metrics = _run_single(
                    model_factory(seed),
                    X_train,
                    y_train_partial,
                    y_train_full,
                    X_test,
                    y_test,
                )
            except Exception as exc:
                metrics = {
                    "accuracy": np.nan,
                    "f1_macro": np.nan,
                    "recall_macro": np.nan,
                    "train_accuracy": np.nan,
                    "train_f1_macro": np.nan,
                    "train_recall_macro": np.nan,
                    "generalization_gap": np.nan,
                    "train_time_s": np.nan,
                    "error": str(exc),
                }
            rows.append({**row, **metrics})
            pbar.update(1)

    pbar.close()
    return pd.DataFrame(rows)


def run_grid_search(
    model_factory: Callable,
    model_name,
    hyperparam_grid,
    X,
    y,
    label_fraction=0.10,
    n_seeds=3,
    test_size=0.30,
    verbose=True,
):
    """Ejecuta sensibilidad para una grilla de hiperparametros."""
    keys = list(hyperparam_grid.keys())
    combos = list(product(*[hyperparam_grid[key] for key in keys]))
    rows = []
    pbar = tqdm(
        total=len(combos) * n_seeds,
        desc=f"{model_name} sensitivity",
        disable=not verbose,
    )

    for combo in combos:
        params = dict(zip(keys, combo))
        for seed in range(n_seeds):
            X_train, X_test, y_train_partial, y_train_full, y_test = (
                make_semi_supervised_split(
                    X,
                    y,
                    label_fraction=label_fraction,
                    test_size=test_size,
                    random_state=seed,
                )
            )
            row = {
                "model": model_name,
                "label_fraction": label_fraction,
                "seed": seed,
                **params,
            }
            try:
                metrics = _run_single(
                    model_factory(seed=seed, **params),
                    X_train,
                    y_train_partial,
                    y_train_full,
                    X_test,
                    y_test,
                )
            except Exception as exc:
                metrics = {
                    "accuracy": np.nan,
                    "f1_macro": np.nan,
                    "recall_macro": np.nan,
                    "train_accuracy": np.nan,
                    "train_f1_macro": np.nan,
                    "train_recall_macro": np.nan,
                    "generalization_gap": np.nan,
                    "train_time_s": np.nan,
                    "error": str(exc),
                }
            rows.append({**row, **metrics})
            pbar.update(1)

    pbar.close()
    return pd.DataFrame(rows)


def summarize_metrics(df, group_cols):
    """Agrega medias y desviaciones para reportar tablas compactas."""
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            recall_macro_mean=("recall_macro", "mean"),
            train_accuracy_mean=("train_accuracy", "mean"),
            gap_mean=("generalization_gap", "mean"),
            train_time_s_mean=("train_time_s", "mean"),
            errors=("error", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
