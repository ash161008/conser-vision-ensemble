# Conser-vision Ensemble Pipeline

A reproducible batch pipeline that blends per-fold model predictions into a
competition submission. Placed **12th** on the DrivenData Conser-vision wildlife
camera-trap leaderboard (multi-class log-loss **0.7644**).

This repository is the **blending and validation stage** of a larger computer
vision pipeline, extracted from Kaggle notebooks and made runnable anywhere.

```
raw camera-trap images
        │
        ▼
  MegaDetector v5          detect the animal, crop to the highest-confidence box
        │
        ▼
  ConvNeXT-Base-22k        5-fold, site-grouped
  DINOv2-Base              5-fold, site-grouped
        │
        ▼
  10 fold prediction CSVs  ◄── this repository starts here
        │
        ▼
  align → weight → prune → blend → validate
        │
        ▼
  submission.csv
```

## Why the blending stage is its own pipeline

In production, models are expensive to run. They are typically executed as batch
jobs that write predictions to intermediate files; a separate, cheap, frequently
re-run pipeline then aligns, combines and validates those files. That separation
is what this repository implements. The blend runs in under a second on CPU and
can be re-run with different strategies without touching a GPU.

## Quick start

```bash
pip install -r requirements.txt

# place fold CSVs in data/predictions/ - see data/README.md for the layout
python run.py                          # geometric_pruned, the 0.7644 configuration
python run.py --strategy all           # write every strategy for comparison
python run.py --data-dir /mnt/preds --out sub.csv
pytest -q                              # 16 tests
```

## Engineering decisions

**Alignment is a join, not a stack.** Fold CSVs come from separate training runs
and are not guaranteed to share row or column order. Stacking them positionally
would average model A's `bird` column with model B's `blank` column and still
emit a valid-looking submission. Every fold is reindexed onto
`submission_format.csv`'s index and columns, so combination happens by image id
and species name. Any id present in one file and absent from another becomes
NaN, which `src/load.py` raises on rather than silently blending.

Two tests cover exactly this — `test_shuffled_ids_are_realigned_not_stacked`
and `test_shuffled_columns_are_realigned` — because it is the failure mode that
produces a wrong answer without producing an error.

**Geometric mean over arithmetic.** Arithmetic averaging is forgiving of
disagreement: if one model says 0.9 and another 0.1, the average is 0.5. The
geometric mean gives 0.3, so a class needs support from most members to survive.
That suppresses confident single-model errors, which is what log-loss punishes
hardest. On this data it was worth roughly 0.02 over arithmetic weighting.

The trade-off is real and documented in the tests: the geometric mean is not
globally "less confident". By moving mass off contested classes it often ends up
*more* peaked on the consensus class. The win is specifically about crushing
classes that only one member supports.

**Inverse-log-loss weighting with pruning.** Members are weighted by
`(1 / val_log_loss) ** 2` and any member above 0.95 validation log-loss is
dropped entirely. This removes the two weakest ConvNeXT folds, leaving 8 of 10
members. Both the exponent and the threshold live in `config.py`.

**Validation gates before writing.** Shape against the submission format, no
NaNs, all values in [0, 1], every row summing to 1. These began as inline
notebook asserts; as a module they run identically from the CLI and the test
suite, and a failure names the offending rows instead of raising a bare
`AssertionError`.

**Configuration is centralised.** Paths, weighting exponent and prune threshold
live in `config.py`, overridable by environment variable or CLI flag. Nothing in
`src/` knows where the data lives.

## Results

| Strategy | Members | Leaderboard |
|---|---|---|
| Simple average | 10 | — |
| Inverse-loss weighted (arithmetic) | 10 | 0.7857 |
| Geometric mean | 10 | 0.7818 |
| **Geometric mean + pruning** | **8** | **0.7644** |

Blending strategy mattered roughly ten times more than retraining individual
members — a finding that only became visible because the strategies are cheap to
compare in this form.

## Layout

```
config.py              paths and hyperparameters
src/load.py            read fold CSVs, align to submission format, guard NaN
src/blend.py           normalise / arithmetic / geometric, weighting, pruning
src/validate.py        pre-submission gates
run.py                 CLI entrypoint with per-stage logging
tests/test_pipeline.py 16 tests
```

## Scope and honest limits

- **This repository covers the blending stage only.** Detection (MegaDetector)
  and training (ConvNeXT, DINOv2) ran as separate Kaggle notebooks; model
  weights were not retained, only the fold predictions they produced.
- **Validation used site-grouped k-fold.** Camera sites in the test set appear
  nowhere in training. An early random split scored 0.42 in validation and 1.92
  on the leaderboard — the gap was site leakage, and every subsequent split was
  grouped by site.
- **Diminishing returns were measured, not assumed.** Two further backbones were
  trained and evaluated (EVA-02, and Google's camera-trap-specific SpeciesNet).
  Neither improved the blend: EVA-02 scored 0.7685 blended in, SpeciesNet 0.7700.
  Both were dropped. The two-model ensemble was already close to the ceiling of
  this approach.
- **The remaining bottleneck is detection, not classification.** Of the worst
  out-of-fold errors, 23 of 25 were images where the detector found no animal at
  all, so the classifier received a background crop and predicted `blank`. That
  is a detection problem; no downstream classifier fixes it.
