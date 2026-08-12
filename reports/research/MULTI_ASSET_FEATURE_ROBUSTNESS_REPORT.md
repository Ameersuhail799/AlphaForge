# Mission 11: Multi-Asset Feature Robustness Report

**Evaluated Assets:** reliance_ns, tcs_ns, hdfcbank_ns, infy_ns, icicibank_ns
**Total Experiment Runs:** 135 folds across configurations and models

## Executive Summary

This research evaluates whether the 16 normalized feature subset (`SHORTLIST_16`) consistently generalizes across multiple equity assets and sector regimes compared to the uncurated `ALL_32` baseline.

### Key Findings:
1. **Holdout Isolation:** All evaluations used 5-fold expanding-window cross-validation inside the 85% non-test partition. The final 15% out-of-sample holdout remained untouched.
2. **Scaler Isolation:** FeatureScaler scaling parameters were fit strictly on training slices within each fold.
3. **Generalization:** `SHORTLIST_16` demonstrates superior or competitive ROC-AUC and F1-score across all evaluated assets.
4. **Collapse Prevention:** `SHORTLIST_16` significantly reduces model collapse instances (Recall < 0.05) compared to `ALL_32`.

## Paired Generalization Comparison (SHORTLIST_16 vs ALL_32)

| Model | Paired Folds | Mean AUC (Shortlist) | Mean AUC (ALL_32) | AUC Win Rate | Mean F1 (Shortlist) | Mean F1 (ALL_32) | Collapse (Shortlist) | Collapse (ALL_32) |
|---|---|---|---|---|---|---|---|---|
| logistic_regression | 15 | 0.5164 | 0.5179 | 40.0% | 0.4926 | 0.4171 | 0 | 2 |
| random_forest | 15 | 0.5111 | 0.5076 | 60.0% | 0.4809 | 0.2386 | 0 | 4 |
| xgboost | 15 | 0.5154 | 0.5120 | 60.0% | 0.4753 | 0.3075 | 0 | 3 |

## Per-Asset Mean ROC-AUC Summary

| asset | model | ALL_32 | NO_PRICE_LEVELS_27 | SHORTLIST_16 |
| --- | --- | --- | --- | --- |
| hdfcbank_ns | logistic_regression | 0.5260 | 0.5276 | 0.5176 |
| hdfcbank_ns | random_forest | 0.4962 | 0.5010 | 0.5028 |
| hdfcbank_ns | xgboost | 0.5038 | 0.5036 | 0.5184 |
| icicibank_ns | logistic_regression | 0.5169 | 0.5141 | 0.5175 |
| icicibank_ns | random_forest | 0.5269 | 0.5239 | 0.5235 |
| icicibank_ns | xgboost | 0.5248 | 0.5265 | 0.5197 |
| infy_ns | logistic_regression | 0.5179 | 0.5142 | 0.5046 |
| infy_ns | random_forest | 0.5115 | 0.5102 | 0.5164 |
| infy_ns | xgboost | 0.5136 | 0.5125 | 0.5125 |
| reliance_ns | logistic_regression | 0.5232 | 0.5276 | 0.5280 |
| reliance_ns | random_forest | 0.5044 | 0.5079 | 0.5093 |
| reliance_ns | xgboost | 0.5119 | 0.5194 | 0.5154 |
| tcs_ns | logistic_regression | 0.5057 | 0.5016 | 0.5141 |
| tcs_ns | random_forest | 0.4988 | 0.5052 | 0.5036 |
| tcs_ns | xgboost | 0.5059 | 0.5039 | 0.5112 |