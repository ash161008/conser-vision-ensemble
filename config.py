"""Central configuration. No paths or hyperparameters are hardcoded elsewhere."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# --- Input locations -------------------------------------------------------
# Override with the CONSER_DATA_DIR env var, or --data-dir on the CLI.
DATA_DIR = Path(os.environ.get("CONSER_DATA_DIR", BASE_DIR / "data"))

# Directory holding the per-fold prediction CSVs (one row per test image,
# columns = the 8 species). Each file is a single fold of a single backbone.
PREDICTIONS_DIR = DATA_DIR / "predictions"

# submission_format.csv from the competition. Defines the canonical row order
# (test image ids) and column order (species) that every fold is aligned to.
SUBMISSION_FORMAT = DATA_DIR / "submission_format.csv"

# Optional: per-model validation log-loss, used for inverse-loss weighting and
# pruning. CSV with columns: model,log_loss. If absent, models are weighted
# equally and no pruning is applied.
MODEL_SCORES = DATA_DIR / "model_scores.csv"

OUTPUT_DIR = Path(os.environ.get("CONSER_OUTPUT_DIR", BASE_DIR / "output"))

# --- Blend hyperparameters -------------------------------------------------
# Members are weighted by (1 / val_log_loss) ** POWER, then normalised.
POWER = 2.0

# Members with val log-loss above this are dropped entirely. 0.95 drops the two
# weakest ConvNeXT folds, leaving 8 of 10 members - this is the configuration
# that scored 0.7644 on the leaderboard.
PRUNE_AT = 0.95

# Numerical floors. ARITHMETIC_EPS guards against log(0) in the metric;
# GEOMETRIC_EPS is the floor applied before taking logs in the geometric mean.
ARITHMETIC_EPS = 1e-4
GEOMETRIC_EPS = 1e-6

DEFAULT_STRATEGY = "geometric_pruned"
