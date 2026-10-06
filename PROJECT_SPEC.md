# SHL Grammar Scoring Engine — Project Specification

> **Version:** 0.1.0 — Phase 00 Constitution
> **Date:** 2026-10-06
> **Status:** Active

---

## 1. Objective

Develop a reproducible machine-learning pipeline that predicts the **grammar score** of a spoken English sample from its audio, producing a **continuous score in the range 0–5**.

The model should prioritize generalization and linguistic signal while maintaining a clean, reproducible implementation.

---

## 2. Dataset

| Split    | Samples | Labels      |
|----------|---------|-------------|
| Training | 769     | Provided    |
| Test     | 216     | Hidden      |

- **Target column:** `label` (continuous, range 0–5)
- **ID column:** `filename`
- **Input:** Raw `.wav` audio files

> **Note:** Questions about zero-valued labels, train/test filename collisions, and label distribution are deferred to **Phase 01 — Dataset Integrity**. No assumptions are made here.

---

## 3. Evaluation Metrics

### 3.1 Local Metrics

All experiments report **both** metrics independently:

**RMSE** (Root Mean Squared Error):

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

- **Direction:** Lower is better.
- **Mandatory:** Training RMSE must be reported in the final notebook.

**Pearson Correlation:**

$$r = \frac{\sum(y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum(y_i - \bar{y})^2 \cdot \sum(\hat{y}_i - \bar{\hat{y}})^2}}$$

- **Direction:** Higher is better (range: −1 to 1).

### 3.2 Kaggle Leaderboard Formula

> **The exact Kaggle leaderboard scoring formulation has NOT been assumed.**
>
> Local experiments will report RMSE and Pearson independently. The exact leaderboard formula will be verified through competition documentation and calibration submissions before any leaderboard-directed optimization.

---

## 4. Competition Constraints

```
Team size:              1 participant
External data:          NOT ALLOWED
External pretrained:    ALLOWED
External LLMs/APIs:     ALLOWED
Max submissions/day:    100
Final submissions:      2
Training RMSE:          MANDATORY in final notebook
Dataset:                Competition use only
```

### 4.1 Allowed External Resources

Pretrained models are **allowed** because the rules permit external model weights. Examples:

```
Whisper          (OpenAI) — ASR / acoustic
wav2vec 2.0      (Meta)   — acoustic representations
HuBERT           (Meta)   — acoustic representations
DeBERTa          (MS)     — text representations
sentence-transformers     — semantic embeddings
```

> **Policy:** Antigravity must **not** assume a particular external model is competition-legal without checking the current competition rules page. When a new external model is introduced, document its source and license in the experiment log.

### 4.2 Prohibited External Resources

```
External grammar-score datasets
External labeled speech datasets
External demographic datasets
External competition datasets not provided by organizers
Any resource that supplements the training labels
```

---

## 5. Anti-Leakage Constitution

### 5.1 Forbidden

We will **never**:

- Use test set labels (directly or inferred)
- Manually infer test labels from train label patterns
- Use filename/ID as a predictive feature
- Use information derived from the hidden test target
- Use external labeled data to augment training
- Tune repeatedly against the public leaderboard score
- Select a final model primarily on test-set predictions
- Allow the same recording to appear in both train and validation folds

### 5.2 Allowed

We **can**:

- Use pretrained model weights
- Use unsupervised representations from external models
- Use competition-provided audio (train + test) for unsupervised steps
- Perform cross-validation on the training set
- Use OOF (out-of-fold) predictions for stacking/ensembling
- Perform feature engineering on competition-provided data
- Ensemble models trained solely on training data

---

## 6. Validation Philosophy

```
Development CV
      ↓
  Feature / model selection
      ↓
      Freeze
      ↓
  Fresh confirmation CV
      ↓
  Final model trained on full training set
      ↓
  Submission generation
```

**Key principles:**

1. The public leaderboard score is **not** a validation set. It will not be used to guide model selection.
2. OOF performance (CV) is the primary decision criterion.
3. An improvement must be **consistent across folds and seeds** to be accepted — not just the result of one lucky split.
4. Hyperparameters are frozen before confirmation CV runs.

---

## 7. Experiment Rules

### 7.1 Each Experiment Must Answer

| Question | Example |
|----------|---------|
| **What are we testing?** | Does adding ASR confidence improve the acoustic baseline? |
| **Why might it work?** | Poor ASR confidence may correlate with unclear or grammatically difficult speech. |
| **How will we test it?** | Same folds, seeds, model — only ASR confidence features changed. |
| **What determines success?** | Consistent improvement across folds/repeats, not a single lucky split. |

### 7.2 The One-Variable Rule

When possible, change **one meaningful thing at a time**.

**Bad experiment design:**
```
Baseline + Whisper + LanguageTool + DeBERTa + new scaler + new model
```
If it improves, we have no idea why.

**Good experiment design:**
```
E001 = Acoustic baseline
E002 = E001 + ASR confidence
E003 = E001 + linguistic features
E004 = E001 + transcript embedding
E005 = E003 + transcript embedding
```
Now each contribution is attributable.

### 7.3 The No Unnecessary Complexity Rule

> **A component is included in the final pipeline only if there is evidence that it improves generalization or provides necessary functionality. Complexity alone is not considered a benefit.**

For every proposed component, answer:

```
1. What does it do?
2. Why should it help?
3. How was it evaluated?
4. Did it improve OOF performance?
5. Is the improvement stable across folds and seeds?
6. Does it introduce leakage or reproducibility problems?
```

If we cannot answer all six questions: **do not add it.**

---

## 8. Artifact and Cache Versioning

### 8.1 Cache Namespace Strategy

Expensive representations are stored under a versioned path:

```
cache/asr/<model_name>/
cache/embeddings/<model_name>/
cache/acoustic/<feature_set>/
```

Examples:
```
cache/asr/whisper-large-v3/
cache/asr/whisper-small/
cache/embeddings/deberta-v3-base/
cache/embeddings/all-mpnet-base-v2/
```

### 8.2 Immutability Rule

- **Never overwrite** a cached artifact silently.
- If a model or preprocessing step changes, use a **new versioned subdirectory**.
- The old artifact is retained until the experiment comparing them is resolved.

### 8.3 Artifact Inventory

Each cache directory should contain a `manifest.json` describing:
```json
{
  "model": "<model name>",
  "version": "<version>",
  "date_created": "<ISO date>",
  "input_data": "<data source>",
  "preprocessing": "<steps applied>",
  "checksum": "<optional SHA256>"
}
```

---

## 9. Final Notebook Requirements

The final `notebooks/competition.ipynb` must contain the following sections:

| Section | Required |
|---------|----------|
| Problem description | ✅ |
| Competition constraints | ✅ |
| Data exploration | ✅ |
| Methodology | ✅ |
| Preprocessing | ✅ |
| Feature engineering | ✅ |
| Model architecture | ✅ |
| Validation methodology | ✅ |
| Results | ✅ |
| Visualizations | ✅ |
| Error analysis | ✅ |
| **Training RMSE** | ✅ **Mandatory** |
| OOF RMSE | ✅ |
| OOF Pearson correlation | ✅ |
| Final predictions | ✅ |
| Submission generation | ✅ |

---

## 10. Reproducibility Guarantees

1. All random operations use seeds from the centralized `RANDOM_SEEDS` list in `src/config.py`.
2. All seeds are set at the start of every script and notebook cell that involves randomness.
3. All feature extraction results are cached; re-running a pipeline with the same inputs and config produces identical outputs.
4. All experiment results are logged to `experiments/experiment_log.csv` before any decision is made.
5. External model checkpoints are pinned by name/version; if a model is updated upstream, a new cache namespace is used.

---

## 11. Open Questions (Deferred to Phase 01)

The following questions are **explicitly deferred** and must not be assumed in Phase 00:

- [ ] What is the label distribution? Are the 37 zero-label samples valid or artifacts?
- [ ] Do the 212 train/test filename collisions represent actual duplicate recordings?
- [ ] Is the train/test split random or stratified by speaker, label, or session?
- [ ] Is there speaker overlap across train and validation folds?
- [ ] What is the recording quality distribution (sample rate, duration, silence ratio)?
- [ ] What is the exact Kaggle leaderboard scoring formula?
- [ ] What compute environment is available (affects Whisper model size selection)?

---

*Document maintained under the SHL Grammar Scoring Engine project. All decisions must reference this specification.*
