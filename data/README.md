# Expected layout

    data/
    ├── submission_format.csv     # from the competition - defines row/column order
    ├── model_scores.csv          # model,log_loss - one row per fold prediction file
    └── predictions/
        ├── convnext_fold0.csv    # id + 8 species columns
        ├── ...
        └── dinov2_fold4.csv

`model_scores.csv` drives inverse-loss weighting and pruning. Without it the
pipeline falls back to equal weights and no pruning, and logs a warning.

Contents are gitignored - they are large and regenerable from the training
notebooks.
