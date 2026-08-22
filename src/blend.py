"""Blending strategies.

`normalise`, `arithmetic` and `geometric` are lifted unchanged from the Task 5
notebook that produced the 0.7644 leaderboard score - only the module-level
`P` dictionary has become an explicit argument.

Why geometric beats arithmetic here: arithmetic averaging is forgiving of
disagreement - if one model says 0.9 and another 0.1, the average is 0.5. The
geometric mean gives 0.3, so a class needs support from most members to
survive. That suppresses confident single-model errors, which is what log-loss
punishes hardest. The risk is the mirror image: it can also suppress a correct
minority opinion. On this data it was worth ~0.02 over arithmetic weighted.
"""

import logging

import numpy as np

import config

logger = logging.getLogger(__name__)


def normalise(p, eps=config.ARITHMETIC_EPS):
    """Clip to avoid infinite log-loss, then renormalise rows to sum to 1."""
    p = np.clip(p, eps, 1 - eps)
    return p / p.sum(axis=1, keepdims=True)


def arithmetic(preds, names_subset, weights=None):
    stack = np.stack([preds[k] for k in names_subset])
    if weights is None:
        weights = np.ones(len(names_subset))
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return normalise(np.tensordot(w, stack, axes=(0, 0)))


def geometric(preds, names_subset, weights=None):
    stack = np.stack(
        [np.clip(preds[k], config.GEOMETRIC_EPS, 1.0) for k in names_subset]
    )
    if weights is None:
        weights = np.ones(len(names_subset))
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    log_avg = np.tensordot(w, np.log(stack), axes=(0, 0))
    return normalise(np.exp(log_avg))


def inverse_loss_weights(scores, power=config.POWER):
    """Weight members by (1 / val_log_loss) ** power. Better members count more."""
    return (1.0 / np.asarray(scores, dtype=float)) ** power


def prune(names, scores, threshold=config.PRUNE_AT):
    """Drop members whose validation log-loss exceeds `threshold`."""
    kept = [k for k in names if scores[k] <= threshold]
    dropped = [k for k in names if k not in kept]
    if dropped:
        logger.info("pruning at %.2f - dropped %d: %s", threshold, len(dropped), dropped)
    if not kept:
        raise ValueError(
            f"pruning at {threshold} removed every member; raise PRUNE_AT"
        )
    return kept, dropped


def build(preds, scores, strategy=config.DEFAULT_STRATEGY):
    """Dispatch to a named strategy. Returns (blended_array, members_used)."""
    names = list(preds.keys())
    inv_w = inverse_loss_weights(scores[names])

    if strategy == "simple_average":
        return arithmetic(preds, names), names

    if strategy == "weighted":
        return arithmetic(preds, names, inv_w), names

    if strategy == "weighted_pruned":
        kept, _ = prune(names, scores)
        return arithmetic(preds, kept, inverse_loss_weights(scores[kept])), kept

    if strategy == "geometric":
        return geometric(preds, names, inv_w), names

    if strategy == "geometric_pruned":
        # the 0.7644 configuration
        kept, _ = prune(names, scores)
        return geometric(preds, kept, inverse_loss_weights(scores[kept])), kept

    raise ValueError(
        f"unknown strategy {strategy!r}. Available: simple_average, weighted, "
        "weighted_pruned, geometric, geometric_pruned"
    )


def diagnostics(blended):
    """Confidence diagnostics. Lower mean top-probability is usually safer for
    log-loss - it means the blend is less willing to bet everything on one class."""
    return {
        "mean_top_prob": float(blended.max(1).mean()),
        "mean_entropy": float((-(blended * np.log(blended)).sum(1)).mean()),
    }
