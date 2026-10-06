# ============================================================
# SHL Grammar Scoring Engine — General Utilities
# ============================================================

from __future__ import annotations

import os
import random
from typing import Sequence

import numpy as np


def set_seeds(seed: int = 42) -> None:
    """
    Set all relevant random seeds for reproducibility.

    Covers: Python built-in random, NumPy, and (if installed)
    PyTorch CPU and CUDA. Call this at the top of every script
    or notebook cell that involves stochastic operations.

    Parameters
    ----------
    seed : int
        Seed value. Defaults to PROJECT_SEED (42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed; that's fine for non-DL phases


def clip_predictions(
    predictions: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 5.0,
) -> np.ndarray:
    """
    Clip predictions to the valid score range [min_val, max_val].

    Must be applied to all predictions before submission.
    Never apply before evaluation (would inflate OOF metrics
    by correcting model errors at the boundary).

    Parameters
    ----------
    predictions : np.ndarray
        Raw model output.
    min_val, max_val : float
        Valid score boundaries from config.MIN_SCORE / MAX_SCORE.

    Returns
    -------
    np.ndarray
        Clipped predictions.
    """
    return np.clip(predictions, min_val, max_val)


def check_no_leakage(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
) -> None:
    """
    Assert that no ID appears in both train and validation sets.

    Call this inside any CV loop before fitting a model to
    enforce the anti-leakage constitution.

    Parameters
    ----------
    train_ids : sequence of str
        Filename IDs in the training fold.
    val_ids : sequence of str
        Filename IDs in the validation fold.

    Raises
    ------
    AssertionError
        If any ID appears in both sets.
    """
    overlap = set(train_ids) & set(val_ids)
    assert len(overlap) == 0, (
        f"LEAKAGE DETECTED: {len(overlap)} IDs appear in both "
        f"train and validation folds: {sorted(overlap)[:10]}"
    )


def make_submission(
    ids: Sequence[str],
    predictions: np.ndarray,
    id_column: str = "filename",
    target_column: str = "label",
) -> "pd.DataFrame":  # noqa: F821
    """
    Construct a submission DataFrame ready for .to_csv(index=False).

    Parameters
    ----------
    ids : sequence of str
        Test set filenames in the correct order.
    predictions : np.ndarray
        Predicted scores (same length as ids).
    id_column, target_column : str
        Column names matching the competition submission format.

    Returns
    -------
    pd.DataFrame with columns [id_column, target_column].
    """
    import pandas as pd  # lazy import to keep this file lightweight

    predictions = clip_predictions(np.asarray(predictions))
    return pd.DataFrame({id_column: ids, target_column: predictions})
