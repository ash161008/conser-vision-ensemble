"""Validation gates run before any submission is written.

These were inline asserts in the notebook. Promoting them to a module means the
same checks run from the CLI and from the test suite, and a failure names the
offending rows instead of just raising AssertionError.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when a blend fails a pre-submission check."""


def check_shape(blended, submission_format):
    expected = submission_format.shape
    if blended.shape != expected:
        raise ValidationError(
            f"shape mismatch: blend is {blended.shape}, submission format expects {expected}"
        )
    logger.info("shape ok: %s", blended.shape)


def check_rows_sum_to_one(blended, tol=1e-6):
    sums = blended.sum(axis=1)
    bad = np.where(np.abs(sums - 1.0) > tol)[0]
    if len(bad):
        raise ValidationError(
            f"{len(bad)} rows do not sum to 1 (tolerance {tol}). "
            f"First offenders at positions {bad[:5].tolist()} with sums {sums[bad[:5]].round(6).tolist()}"
        )
    logger.info("all %d rows sum to 1", len(sums))


def check_probabilities(blended):
    if np.isnan(blended).any():
        raise ValidationError(f"{int(np.isnan(blended).sum())} NaN values in blend")
    if (blended < 0).any() or (blended > 1).any():
        raise ValidationError("probabilities outside [0, 1]")
    logger.info("no NaNs, all values in [0, 1]")


def validate(blended, submission_format):
    """Run every gate. Raises ValidationError on the first failure."""
    check_shape(blended, submission_format)
    check_probabilities(blended)
    check_rows_sum_to_one(blended)
    logger.info("validation passed")
    return True


def to_submission(blended, submission_format):
    """Wrap the array back into a DataFrame with the canonical index/columns."""
    return pd.DataFrame(
        blended, index=submission_format.index, columns=submission_format.columns
    )
