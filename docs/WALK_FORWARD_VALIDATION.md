"""Walk-forward validation documentation for AlphaForge."""

# Walk-Forward / Temporal Robustness Validation

This document explains the purpose and design of the walk-forward
validation implemented in `src/research/walk_forward.py`.

Key points:
- Expanding-window design: train window grows with each fold, validation
  windows advance forward chronologically.
- Final holdout TEST (last 15%) is never touched by fold construction.
- No random shuffling; all splits preserve chronological order.
- Scaling is fitted per-fold on TRAIN only.

Walk-forward results are research artifacts and should not be used to
replace the production `champion.json`.

Walk-forward reports are written to `reports/research/`:
- `walk_forward_results.csv` - fold-level results
- `walk_forward_summary.csv` - per-model summary statistics
- `walk_forward_feature_importance.csv` - fold-level feature importances
- `TEMPORAL_ROBUSTNESS_REPORT.md` - human-readable summary

"""
