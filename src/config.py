# ============================================================
# SHL Grammar Scoring Engine — Central Configuration
# ============================================================
# ALL global constants and project-wide defaults live here.
# Import this module wherever a configurable value is needed.
# Do NOT scatter magic numbers across scripts.
# ============================================================

from pathlib import Path

# ----------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------

PROJECT_SEED: int = 42

RANDOM_SEEDS: list[int] = [42, 123, 456, 789, 2026]
"""
Five independent seeds for repeated cross-validation.
Using multiple seeds guards against accidentally optimistic
or pessimistic results from a single lucky/unlucky split.
"""

# ----------------------------------------------------------
# Data schema
# ----------------------------------------------------------

TARGET_COLUMN: str = "label"
ID_COLUMN: str = "filename"

MIN_SCORE: float = 0.0
MAX_SCORE: float = 5.0
"""
Valid label range per competition specification.
Predictions clipped to [MIN_SCORE, MAX_SCORE] before submission.
"""

# ----------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------

N_FOLDS: int = 5
N_REPEATS: int = 5
"""
RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=PROJECT_SEED)
produces 25 evaluation runs — enough for stable variance estimates
without prohibitive compute on expensive representations.

NOTE: The actual CV object is constructed in src/validation.py.
      This config controls only the hyperparameters.
"""

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
"""Absolute path to shl-grammar-scoring/ regardless of cwd."""

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
METADATA_DIR: Path = DATA_DIR / "metadata"

CACHE_DIR: Path = ROOT_DIR / "cache"
CACHE_AUDIO_DIR: Path = CACHE_DIR / "audio"
CACHE_ASR_DIR: Path = CACHE_DIR / "asr"
CACHE_TEXT_DIR: Path = CACHE_DIR / "text"
CACHE_ACOUSTIC_DIR: Path = CACHE_DIR / "acoustic"
CACHE_EMBEDDINGS_DIR: Path = CACHE_DIR / "embeddings"

EXPERIMENTS_DIR: Path = ROOT_DIR / "experiments"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"

# ----------------------------------------------------------
# Expected raw data filenames (competition-provided)
# ----------------------------------------------------------

TRAIN_CSV: Path = RAW_DIR / "train.csv"
TEST_CSV: Path = RAW_DIR / "test.csv"
TRAIN_AUDIO_DIR: Path = RAW_DIR / "audios_train"
TEST_AUDIO_DIR: Path = RAW_DIR / "audios_test"
"""
Adjust these names if the competition archive uses different names.
All downstream modules import from here — so one edit propagates.
"""

# ----------------------------------------------------------
# Submission
# ----------------------------------------------------------

SUBMISSION_CSV: Path = PREDICTIONS_DIR / "submission.csv"

# ----------------------------------------------------------
# Experiment log
# ----------------------------------------------------------

EXPERIMENT_LOG: Path = EXPERIMENTS_DIR / "experiment_log.csv"

EXPERIMENT_LOG_COLUMNS: list[str] = [
    "experiment_id",
    "date",
    "description",
    "hypothesis",
    "data_version",
    "features",
    "representation",
    "model",
    "hyperparameters",
    "cv_strategy",
    "seed",
    "train_rmse",
    "oof_rmse",
    "oof_pearson",
    "oof_rmse_std",
    "oof_pearson_std",
    "status",
    "decision",
    "notes",
]

# ----------------------------------------------------------
# Audio preprocessing defaults
# ----------------------------------------------------------
# NOTE: These are PLACEHOLDER values only.
# The actual target sample rate, channel strategy, and
# normalization method will be decided in Phase 01
# after inspecting the raw data.
# ----------------------------------------------------------

AUDIO_TARGET_SR: int = 16_000
"""
16 kHz is the standard input rate for most speech models
(Whisper, wav2vec 2.0, HuBERT). Confirmed as default pending
Phase 01 audio inspection.
"""

# ----------------------------------------------------------
# ASR configuration
# ----------------------------------------------------------
# Intentionally left sparse.
# Whisper model size, beam width, and language are NOT set here.
# Those decisions belong to Phase 02 — ASR / Transcription,
# after we know the available compute environment.
# ----------------------------------------------------------

# WHISPER_MODEL = ???   ← decided in Phase 02

# ----------------------------------------------------------
# Model hyperparameters
# ----------------------------------------------------------
# NOT defined here.
# Each experiment module owns its own hyperparameter dict and
# logs it to the experiment log.
# Centralized tuned hyperparameters are added here only after
# a phase freeze.
# ----------------------------------------------------------

# ----------------------------------------------------------
# Utility: ensure all project directories exist
# ----------------------------------------------------------

ALL_DIRS: list[Path] = [
    RAW_DIR,
    METADATA_DIR,
    CACHE_AUDIO_DIR,
    CACHE_ASR_DIR,
    CACHE_TEXT_DIR,
    CACHE_ACOUSTIC_DIR,
    CACHE_EMBEDDINGS_DIR,
    EXPERIMENTS_DIR,
    PREDICTIONS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    NOTEBOOKS_DIR,
]


def create_project_dirs() -> None:
    """Create all project directories if they do not already exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_project_dirs()
    print("Project directories verified / created.")
    for d in ALL_DIRS:
        status = "✓" if d.exists() else "✗"
        print(f"  {status}  {d.relative_to(ROOT_DIR)}")
