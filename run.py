#!/usr/bin/env python3
"""Blend per-fold predictions into a competition submission.

    python run.py                                  # geometric_pruned (the 0.7644 config)
    python run.py --strategy geometric
    python run.py --strategy all                   # write every strategy, compare
    python run.py --data-dir /path/to/data --out submission.csv
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import config
from src import blend, load, validate

STRATEGIES = [
    "simple_average",
    "weighted",
    "weighted_pruned",
    "geometric",
    "geometric_pruned",
]


def setup_logging(verbose=False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", default=config.DEFAULT_STRATEGY,
                   choices=STRATEGIES + ["all"])
    p.add_argument("--data-dir", type=Path, default=None,
                   help="overrides CONSER_DATA_DIR / config.DATA_DIR")
    p.add_argument("--out", type=Path, default=None, help="output CSV path")
    p.add_argument("--power", type=float, default=config.POWER)
    p.add_argument("--prune-at", type=float, default=config.PRUNE_AT)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    log = logging.getLogger("run")
    t0 = time.time()

    data_dir = args.data_dir or config.DATA_DIR
    pred_dir = data_dir / "predictions"
    fmt_path = data_dir / "submission_format.csv"
    scores_path = data_dir / "model_scores.csv"

    config.POWER, config.PRUNE_AT = args.power, args.prune_at

    log.info("=== Conser-vision ensemble pipeline ===")
    log.info("data dir: %s", data_dir)

    try:
        fmt = load.load_submission_format(fmt_path)
        preds = load.load_preds(pred_dir, fmt)
        scores = load.load_scores(scores_path, preds.keys())
    except (FileNotFoundError, ValueError) as e:
        log.error("ingestion failed: %s", e)
        return 1

    targets = STRATEGIES if args.strategy == "all" else [args.strategy]
    out_dir = args.out.parent if args.out else config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for strategy in targets:
        log.info("--- %s ---", strategy)
        try:
            blended, members = blend.build(preds, scores, strategy)
            validate.validate(blended, fmt)
        except (ValueError, validate.ValidationError) as e:
            log.error("%s failed: %s", strategy, e)
            continue

        diag = blend.diagnostics(blended)
        log.info("members: %d | mean top prob: %.4f | mean entropy: %.4f",
                 len(members), diag["mean_top_prob"], diag["mean_entropy"])

        out_path = (args.out if args.out and len(targets) == 1
                    else out_dir / f"submission_{strategy}.csv")
        validate.to_submission(blended, fmt).to_csv(out_path)
        log.info("wrote %s", out_path)
        results[strategy] = diag

    if not results:
        log.error("no strategy completed successfully")
        return 1

    log.info("=== done in %.1fs | %d submission(s) written ===",
             time.time() - t0, len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
