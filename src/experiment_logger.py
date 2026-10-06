# ============================================================
# SHL Grammar Scoring Engine — Experiment Logger
# ============================================================
# All experiments must log results here before any decision.
# Never rely on notebook variables — the log is the record.
# ============================================================

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import EXPERIMENT_LOG, EXPERIMENT_LOG_COLUMNS


def _ensure_log_exists(log_path: Path = EXPERIMENT_LOG) -> None:
    """Create the experiment log CSV with headers if it doesn't exist."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EXPERIMENT_LOG_COLUMNS)
            writer.writeheader()


def log_experiment(
    experiment_id: str,
    description: str,
    hypothesis: str,
    features: str,
    representation: str,
    model: str,
    hyperparameters: str | dict[str, Any],
    cv_strategy: str,
    seed: int,
    train_rmse: float | str,
    oof_rmse: float | str,
    oof_pearson: float | str,
    oof_rmse_std: float | str,
    oof_pearson_std: float | str,
    status: str = "complete",
    decision: str = "",
    notes: str = "",
    data_version: str = "v1",
    log_path: Path = EXPERIMENT_LOG,
) -> None:
    """
    Append one experiment result row to the experiment log.

    Parameters
    ----------
    experiment_id : str
        Unique identifier, e.g. "E001".
    description : str
        One-line plain-English description of the experiment.
    hypothesis : str
        Why we expected this to work.
    features : str
        Comma-separated list of feature groups used.
    representation : str
        Which cached representation was used (e.g., "whisper-large-v3").
    model : str
        Model class name and key settings (e.g., "Ridge(alpha=1.0)").
    hyperparameters : str or dict
        Hyperparameter dictionary or its string representation.
    cv_strategy : str
        Description of CV (e.g., "RepeatedKFold(5×5, seed=42)").
    seed : int
        Primary seed used for this experiment.
    train_rmse : float or str
        RMSE on the training set (mandatory for final submission).
    oof_rmse : float or str
        Mean OOF RMSE across all folds.
    oof_pearson : float or str
        Mean OOF Pearson across all folds.
    oof_rmse_std : float or str
        Standard deviation of OOF RMSE across folds.
    oof_pearson_std : float or str
        Standard deviation of OOF Pearson across folds.
    status : str
        "complete", "failed", "partial", "running".
    decision : str
        "keep", "discard", "investigate", "baseline".
    notes : str
        Free-text observations.
    data_version : str
        Data version tag (default "v1").
    log_path : Path
        Path to the log CSV. Defaults to config.EXPERIMENT_LOG.
    """
    _ensure_log_exists(log_path)

    row = {
        "experiment_id": experiment_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "description": description,
        "hypothesis": hypothesis,
        "data_version": data_version,
        "features": features,
        "representation": representation,
        "model": model,
        "hyperparameters": str(hyperparameters),
        "cv_strategy": cv_strategy,
        "seed": seed,
        "train_rmse": _fmt(train_rmse),
        "oof_rmse": _fmt(oof_rmse),
        "oof_pearson": _fmt(oof_pearson),
        "oof_rmse_std": _fmt(oof_rmse_std),
        "oof_pearson_std": _fmt(oof_pearson_std),
        "status": status,
        "decision": decision,
        "notes": notes,
    }

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_LOG_COLUMNS)
        writer.writerow(row)

    print(
        f"[experiment_log] {experiment_id} logged — "
        f"OOF RMSE={_fmt(oof_rmse)}, Pearson={_fmt(oof_pearson)}, "
        f"Decision={decision}"
    )


def _fmt(value: float | str) -> str:
    """Format a numeric metric value to 4 decimal places, or pass through str."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def read_log(log_path: Path = EXPERIMENT_LOG) -> "pd.DataFrame":  # noqa: F821
    """
    Read the experiment log into a DataFrame for analysis.

    Returns
    -------
    pd.DataFrame or empty DataFrame if log doesn't exist.
    """
    import pandas as pd  # lazy import

    if not log_path.exists():
        return pd.DataFrame(columns=EXPERIMENT_LOG_COLUMNS)
    return pd.read_csv(log_path)
