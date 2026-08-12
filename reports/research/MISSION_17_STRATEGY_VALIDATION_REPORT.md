# Mission 17 — Candidate Refinement, Regime Robustness & Trading Strategy Validation Report

**Total Outer Evaluations:** 375 across 5 assets, 3 models, 5 outer folds, and 5 configurations (C0, C5, C7, C57, C8)

## Executive Summary

This experiment evaluates regime robustness, PPR signal bias, distribution shifts, pre-registered confidence filtering, and 10-day trading strategy performance across all asset/model/configuration combinations.

## Overall Configuration Summary

| config_id | total_folds | mean_roc_auc | std_roc_auc | mean_pr_auc | mean_mcc | mean_f1 | mean_ppr | mean_return_spread_pct | net_spread_10bps_pct | net_spread_50bps_pct | pos_auc_folds | pos_mcc_folds | pos_spread_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C5 | 75 | 0.5317 | 0.0483 | 0.5949 | 0.0361 | 0.5081 | 0.4881 | 0.7794 | 0.6794 | 0.2794 | 52 | 54 | 54 |
| C57 | 75 | 0.5307 | 0.0487 | 0.5934 | 0.0366 | 0.5099 | 0.4906 | 0.7109 | 0.6109 | 0.2109 | 52 | 51 | 53 |
| C8 | 75 | 0.5279 | 0.0550 | 0.5944 | 0.0303 | 0.5258 | 0.5187 | 0.5402 | 0.4402 | 0.0402 | 51 | 47 | 47 |
| C7 | 75 | 0.5253 | 0.0417 | 0.5870 | 0.0398 | 0.5229 | 0.5201 | 0.6175 | 0.5175 | 0.1175 | 55 | 57 | 56 |
| C0 | 75 | 0.5210 | 0.0401 | 0.5854 | 0.0321 | 0.5175 | 0.5186 | 0.4289 | 0.3289 | -0.0711 | 53 | 54 | 59 |
