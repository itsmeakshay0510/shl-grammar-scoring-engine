# ============================================================
# SHL Grammar Scoring Engine — Metric Utilities
# ============================================================
# Centralized metric functions used by ALL experiments.
# No experiment should compute RMSE or Pearson inline —
# always call from here to guarantee consistency.
# ============================================================

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Root Mean Squared Error.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground-truth labels.
    y_pred : array-like of shape (n,)
        Model predictions.

    Returns
    -------
    float
        RMSE ≥ 0. Lower is better.

    Notes
    -----
    Uses population (n) rather than sample (n-1) denominator,
    consistent with Kaggle's standard RMSE formulation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Pearson product-moment correlation coefficient.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground-truth labels.
    y_pred : array-like of shape (n,)
        Model predictions.

    Returns
    -------
    float
        Pearson r ∈ [−1, 1]. Higher is better.
        Returns 0.0 if either array has zero variance.

    Notes
    -----
    Uses scipy.stats.pearsonr for numerical stability.
    The p-value is intentionally discarded — we care only
    about the correlation coefficient, and n < 1000 makes
    the p-value unreliable as an effect-size measure.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def evaluate(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    label: str = "",
    verbose: bool = True,
) -> dict[str, float]:
    """
    Compute and optionally print both competition metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Model predictions.
    label : str, optional
        Descriptive tag printed alongside the results (e.g., "OOF").
    verbose : bool, optional
        If True, prints a formatted summary. Default True.

    Returns
    -------
    dict with keys "rmse" and "pearson".
    """
    r = rmse(y_true, y_pred)
    p = pearson(y_true, y_pred)
    if verbose:
        tag = f"[{label}] " if label else ""
        print(f"{tag}RMSE: {r:.4f}  |  Pearson: {p:.4f}")
    return {"rmse": r, "pearson": p}


def summarize_cv_results(
    fold_rmse: list[float],
    fold_pearson: list[float],
    *,
    label: str = "CV",
    verbose: bool = True,
) -> dict[str, float]:
    """
    Aggregate per-fold metrics into mean ± std summary.

    Parameters
    ----------
    fold_rmse : list of float
        RMSE value for each fold.
    fold_pearson : list of float
        Pearson r for each fold.
    label : str, optional
        Label for display.
    verbose : bool, optional
        If True, prints a formatted summary.

    Returns
    -------
    dict with keys:
        rmse_mean, rmse_std, pearson_mean, pearson_std
    """
    rmse_arr = np.asarray(fold_rmse)
    pearson_arr = np.asarray(fold_pearson)
    result = {
        "rmse_mean": float(rmse_arr.mean()),
        "rmse_std": float(rmse_arr.std()),
        "pearson_mean": float(pearson_arr.mean()),
        "pearson_std": float(pearson_arr.std()),
    }
    if verbose:
        print(
            f"[{label}]"
            f"  RMSE: {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}"
            f"  |  Pearson: {result['pearson_mean']:.4f} ± {result['pearson_std']:.4f}"
        )
    return result


def mean_baseline_rmse(y_true: ArrayLike) -> float:
    """
    RMSE of the trivial mean predictor (always predict the training mean).

    This is the reference point (E000) for the experiment log.
    Any model that cannot beat this is not learning anything.

    Parameters
    ----------
    y_true : array-like
        Training labels.

    Returns
    -------
    float
        RMSE of mean predictor evaluated on y_true.
    """
    y_true = np.asarray(y_true, dtype=float)
    mean_pred = np.full_like(y_true, fill_value=y_true.mean())
    return rmse(y_true, mean_pred)


# ----------------------------------------------------------
# Self-test
# ----------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y_t = rng.uniform(0, 5, 769)
    y_p = y_t + rng.normal(0, 0.5, 769)

    print("=== Metric Self-Test ===")
    evaluate(y_t, y_p, label="Synthetic")
    summarize_cv_results([0.82, 0.79, 0.85, 0.81, 0.83], [0.71, 0.74, 0.70, 0.73, 0.72])
    print(f"Mean-baseline RMSE (synthetic): {mean_baseline_rmse(y_t):.4f}")
