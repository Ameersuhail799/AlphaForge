# Champion System

The champion system ranks model comparison results and produces two artifacts in `reports/`:

- `champion.json` stores the selected champion model and its summary metrics.
- `model_leaderboard.csv` lists all models with rank order.

## Champion Selection

`ChampionManager` reads `reports/model_comparison.csv` and selects the best model using this priority order:

1. ROC-AUC
2. F1
3. Accuracy

The top-ranked model is written to `reports/champion.json` with its metrics, a UTC timestamp, and a selection reason.

## Leaderboard

`LeaderboardManager` sorts the same comparison data by ROC-AUC in descending order and adds a `Rank` column starting at 1.

The resulting leaderboard is saved to `reports/model_leaderboard.csv`.

## Ranking Logic

The ranking rules are deterministic:

- Higher ROC-AUC always wins.
- If ROC-AUC is tied, higher F1 wins.
- If both ROC-AUC and F1 are tied, higher Accuracy wins.

This keeps champion selection aligned with the strongest general classification signal while preserving stable tie-breaking.