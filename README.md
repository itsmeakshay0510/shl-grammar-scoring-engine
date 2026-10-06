# SHL Grammar Scoring Engine

> Kaggle Competition | Single-Participant | Reproducible ML Pipeline

---

## Overview

This repository implements a reproducible machine-learning pipeline for the **SHL Grammar Scoring Engine** competition. The task is to predict the grammar score of spoken English audio samples as a continuous value in the range **0–5**.

**Evaluation metrics:** RMSE (lower is better) + Pearson correlation (higher is better)

| Split    | Samples |
|----------|---------|
| Training | 769     |
| Test     | 216     |

---

## Project Structure

```
shl-grammar-scoring/
│
├── README.md                  ← This file
├── PROJECT_SPEC.md            ← Competition constitution and design rules
├── requirements.txt           ← Python dependencies
│
├── data/
│   ├── raw/                   ← Original competition data (audio + CSVs)
│   └── metadata/              ← Derived metadata (duration, SR, etc.)
│
├── cache/
│   ├── audio/                 ← Preprocessed audio (resampled, normalized)
│   ├── asr/                   ← ASR transcripts, versioned by model
│   ├── text/                  ← Text features (LanguageTool, readability)
│   ├── acoustic/              ← Acoustic feature sets (MFCC, prosody, etc.)
│   └── embeddings/            ← Dense embeddings, versioned by model
│
├── src/
│   ├── config.py              ← Central configuration (seeds, paths, constants)
│   ├── data.py                ← Data loading and path resolution
│   ├── validation.py          ← CV strategy and fold generation
│   ├── metrics.py             ← RMSE, Pearson, and logging utilities
│   ├── features.py            ← Feature extraction pipeline
│   ├── models.py              ← Model definitions and training loops
│   ├── ensemble.py            ← Ensembling and stacking logic
│   └── utils.py               ← General utilities (seeding, I/O, logging)
│
├── experiments/
│   └── experiment_log.csv     ← Structured log of all experiments
│
├── outputs/
│   ├── predictions/           ← OOF and test predictions (CSV)
│   ├── plots/                 ← Figures and visualizations
│   └── reports/               ← Per-phase summary reports
│
└── notebooks/
    └── competition.ipynb      ← Final submission notebook
```

---

## Quickstart

```bash
# 1. Clone and enter the project
cd shl-grammar-scoring

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate.bat     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place competition data
# Copy train/ test/ train.csv test.csv into data/raw/

# 5. Run validation (Phase 01+)
python -m src.validation
```

---

## Experiment Log

All experiments are tracked in [`experiments/experiment_log.csv`](experiments/experiment_log.csv).

Every experiment records:
- Hypothesis
- Feature set and representation
- Model and hyperparameters
- CV strategy and seeds
- Train RMSE, OOF RMSE, OOF Pearson
- Variance across folds
- Decision and reasoning

---

## Rules at a Glance

| Rule | Detail |
|------|--------|
| External data | ❌ Not allowed |
| External pretrained models | ✅ Allowed |
| External LLMs / APIs | ✅ Allowed |
| Leaderboard as validation | ❌ Forbidden |
| Filename as feature | ❌ Forbidden |
| One-variable experiments | ✅ Required when possible |
| Mandatory training RMSE | ✅ Required in final notebook |

For the full specification, see [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

---

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 00 | Project Constitution | ✅ Complete |
| 01 | Dataset Integrity | ⬜ Pending |
| 02 | ASR / Transcription | ⬜ Pending |
| 03 | Acoustic Baseline | ⬜ Pending |
| 04 | Linguistic Features | ⬜ Pending |
| 05 | Text Embeddings | ⬜ Pending |
| ... | ... | ... |

---

## Reproducibility

- Global seed: `42`
- All seeds are set via `src/utils.set_seeds()`
- All expensive representations are cached under versioned paths
- Re-running any pipeline stage with identical inputs and config produces identical outputs

---

*SHL Grammar Scoring Engine — Kaggle Competition*
