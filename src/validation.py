# ============================================================
# SHL Grammar Scoring Engine — Validation Strategy
# ============================================================
# Defines the CV objects used across ALL experiments.
# Every experiment must use these to guarantee comparable folds.
# ============================================================

from __future__ import annotations

from sklearn.model_selection import KFold, RepeatedKFold

from src.config import N_FOLDS, N_REPEATS, PROJECT_SEED


def get_cv(
    *,
    repeated: bool = True,
    n_folds: int = N_FOLDS,
    n_repeats: int = N_REPEATS,
    seed: int = PROJECT_SEED,
) -> RepeatedKFold | KFold:
    """
    Return the standard CV splitter for this project.

    Parameters
    ----------
    repeated : bool
        If True (default), returns RepeatedKFold with n_repeats.
        If False, returns a single KFold (useful for quick iteration).
    n_folds : int
        Number of folds. Default: N_FOLDS from config.
    n_repeats : int
        Number of repeats. Only used when repeated=True.
        Default: N_REPEATS from config.
    seed : int
        Random state. Default: PROJECT_SEED from config.

    Returns
    -------
    RepeatedKFold or KFold

    Notes
    -----
    Validation philosophy (from PROJECT_SPEC.md §6):
    - Development CV: repeated=True for stable estimates.
    - Quick iteration: repeated=False for speed.
    - Confirmation CV: a fresh repeated=True run with frozen
      hyperparameters, run once before freezing the final model.
    - The public leaderboard is NEVER used as a validation set.
    """
    if repeated:
        return RepeatedKFold(
            n_splits=n_folds,
            n_repeats=n_repeats,
            random_state=seed,
        )
    return KFold(n_splits=n_folds, shuffle=True, random_state=seed)
