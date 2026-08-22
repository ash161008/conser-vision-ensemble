"""Test suite. The notebook's inline asserts, promoted to real tests, plus the
failure modes that inline asserts never covered - misaligned ids and shuffled
columns, which are the bugs that would silently corrupt a submission.

    pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from src import blend, load, validate  # noqa: E402

CLASSES = ["antelope_duiker", "bird", "blank", "civet_genet",
           "hog", "leopard", "monkey_prosimian", "rodent"]
IDS = [f"ZJ{i:06d}" for i in range(20)]


def _random_preds(seed=0, index=None, columns=None):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(len(CLASSES)), size=len(index or IDS))
    return pd.DataFrame(p, index=index or IDS, columns=columns or CLASSES)


@pytest.fixture
def fmt():
    return _random_preds(seed=99)


@pytest.fixture
def preds():
    return {f"model_{i}": _random_preds(seed=i).values for i in range(3)}


@pytest.fixture
def scores():
    return pd.Series({"model_0": 0.80, "model_1": 0.90, "model_2": 1.10})


# --- the notebook's asserts ------------------------------------------------

def test_rows_sum_to_one_after_arithmetic(preds):
    out = blend.arithmetic(preds, list(preds))
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)


def test_rows_sum_to_one_after_geometric(preds):
    out = blend.geometric(preds, list(preds))
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)


def test_normalise_clips_and_renormalises():
    p = np.array([[0.0, 1.0], [0.5, 0.5]])
    out = blend.normalise(p)
    assert (out > 0).all(), "zeros must be clipped away - log(0) is -inf"
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)


def test_output_shape_matches_submission_format(preds, fmt):
    out = blend.geometric(preds, list(preds))
    validate.check_shape(out, fmt)


# --- weighting and pruning -------------------------------------------------

def test_inverse_loss_weights_favour_better_models(scores):
    w = blend.inverse_loss_weights(scores)
    assert w[0] > w[1] > w[2], "lower log-loss must earn a larger weight"


def test_prune_drops_weak_members(scores):
    kept, dropped = blend.prune(list(scores.index), scores, threshold=0.95)
    assert kept == ["model_0", "model_1"]
    assert dropped == ["model_2"]


def test_prune_raises_if_everything_dropped(scores):
    with pytest.raises(ValueError, match="removed every member"):
        blend.prune(list(scores.index), scores, threshold=0.1)


def test_geometric_suppresses_lone_dissenter():
    """The property that actually matters: a class backed by only one member
    survives arithmetic averaging but is crushed by the geometric mean.

    Note this does NOT mean geometric is globally less confident - by moving
    mass off contested classes it often ends up MORE peaked on the consensus
    class. The win is specifically about suppressing confident single-model
    errors, which is the failure mode log-loss punishes hardest.
    """
    confident = np.array([[0.90, 0.05, 0.05]])
    dissent = np.array([[0.02, 0.49, 0.49]])
    preds = {"a": confident, "b": dissent, "c": dissent}

    a = blend.arithmetic(preds, ["a", "b", "c"])[0, 0]
    g = blend.geometric(preds, ["a", "b", "c"])[0, 0]
    assert g < a, "geometric must penalise the class only one member supports"


# --- alignment: the bugs inline asserts never caught -----------------------

def test_shuffled_ids_are_realigned_not_stacked(tmp_path, fmt):
    """A fold written in a different row order must still align by id."""
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    df = _random_preds(seed=1)
    df.loc[list(reversed(IDS))].to_csv(pred_dir / "shuffled.csv",
                                       index_label="id")

    loaded = load.load_preds(pred_dir, fmt)["shuffled"]
    np.testing.assert_allclose(loaded, df.loc[fmt.index].values)


def test_shuffled_columns_are_realigned(tmp_path, fmt):
    """Species columns in a different order must not be silently mixed."""
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    df = _random_preds(seed=2)
    df[list(reversed(CLASSES))].to_csv(pred_dir / "cols.csv", index_label="id")

    loaded = load.load_preds(pred_dir, fmt)["cols"]
    np.testing.assert_allclose(loaded, df[fmt.columns].values)


def test_missing_ids_raise_rather_than_producing_nan(tmp_path, fmt):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _random_preds(seed=3).iloc[:10].to_csv(pred_dir / "short.csv",
                                           index_label="id")

    with pytest.raises(ValueError, match="NaN cells"):
        load.load_preds(pred_dir, fmt)


def test_missing_class_column_raises(tmp_path, fmt):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _random_preds(seed=4).drop(columns=["hog"]).to_csv(
        pred_dir / "nohog.csv", index_label="id")

    with pytest.raises(ValueError, match="missing expected class columns"):
        load.load_preds(pred_dir, fmt)


def test_empty_predictions_dir_raises(tmp_path, fmt):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="no prediction CSVs"):
        load.load_preds(pred_dir, fmt)


# --- validation gates ------------------------------------------------------

def test_validation_catches_rows_not_summing_to_one(fmt):
    bad = np.full(fmt.shape, 0.5)
    with pytest.raises(validate.ValidationError, match="do not sum to 1"):
        validate.check_rows_sum_to_one(bad)


def test_validation_catches_nan(fmt):
    bad = np.full(fmt.shape, 0.125)
    bad[0, 0] = np.nan
    with pytest.raises(validate.ValidationError, match="NaN"):
        validate.check_probabilities(bad)


def test_full_pipeline_passes_validation(preds, scores, fmt):
    blended, members = blend.build(preds, scores, "geometric_pruned")
    assert validate.validate(blended, fmt)
    assert len(members) == 2, "model_2 (1.10) should be pruned at 0.95"
