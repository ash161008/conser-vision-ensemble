"""Load per-fold prediction CSVs and align them to the submission format.

The alignment step is the important one. Fold CSVs are written by separate
training runs and are not guaranteed to share a row order or column order. If
they were stacked by position, model A's `bird` column could be averaged with
model B's `blank` column and the blend would be silently wrong - it would still
produce a valid-looking submission.

Reindexing every fold onto submission_format's index and columns forces a join
on image id and species name. Any id present in one file but not another
becomes NaN, which the guard below turns into a hard failure rather than a
quietly corrupt blend.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_submission_format(path: Path) -> pd.DataFrame:
    """Read submission_format.csv - the canonical row and column order."""
    if not path.exists():
        raise FileNotFoundError(
            f"submission_format.csv not found at {path}. "
            "Set CONSER_DATA_DIR or pass --data-dir."
        )
    fmt = pd.read_csv(path, index_col="id")
    logger.info(
        "submission format: %d rows x %d classes", len(fmt), fmt.shape[1]
    )
    return fmt


def load_preds(pred_dir: Path, submission_format: pd.DataFrame) -> dict:
    """Load every prediction CSV in `pred_dir`, aligned to the submission format.

    Returns {model_name: ndarray of shape (n_images, n_classes)} where model_name
    is the filename stem.
    """
    if not pred_dir.exists():
        raise FileNotFoundError(f"predictions directory not found: {pred_dir}")

    files = sorted(pred_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no prediction CSVs found in {pred_dir}")

    index, columns = submission_format.index, submission_format.columns
    preds = {}

    for fp in files:
        df = pd.read_csv(fp, index_col="id")

        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"{fp.name} is missing expected class columns: {sorted(missing_cols)}"
            )

        aligned = df.reindex(index=index, columns=columns)

        n_nan = int(aligned.isna().sum().sum())
        if n_nan:
            missing_ids = aligned.index[aligned.isna().any(axis=1)][:5].tolist()
            raise ValueError(
                f"{fp.name}: {n_nan} NaN cells after aligning to the submission "
                f"format - the file is missing rows for ids such as {missing_ids}. "
                "Blending would silently corrupt these rows."
            )

        preds[fp.stem] = aligned.values.astype(np.float64)
        logger.info("loaded %-32s %s", fp.name, aligned.shape)

    logger.info("loaded %d fold predictions", len(preds))
    return preds


def load_scores(path: Path, model_names) -> pd.Series:
    """Per-model validation log-loss, used for weighting and pruning.

    Missing file, or models absent from it, fall back to equal weighting.
    """
    if not path.exists():
        logger.warning(
            "no model_scores.csv at %s - falling back to equal weights, no pruning",
            path,
        )
        return pd.Series(1.0, index=list(model_names))

    scores = pd.read_csv(path).set_index("model")["log_loss"]
    unknown = [m for m in model_names if m not in scores.index]
    if unknown:
        logger.warning("no score for %s - weighting these equally", unknown)
        for m in unknown:
            scores[m] = scores.mean()

    return scores.reindex(list(model_names))
