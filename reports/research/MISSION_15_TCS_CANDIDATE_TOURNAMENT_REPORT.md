# Mission 15 — Step 3: TCS Candidate Tournament & Deep Validation Report

## Executive Summary

This experiment deep-validates the top four TCS candidate combinations across 5 chronological expanding folds to test temporal consistency, worst-fold risk, and return spreads.

## Candidate Tournament Ranking & Final Verdicts

| candidate_id | target_name | model | stability_score | pos_auc_folds | pos_mcc_folds | pos_spread_folds | mean_roc_auc | std_roc_auc | min_roc_auc | mean_mcc | mean_return_spread_pct | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Candidate_B | TARGET_B | xgboost | 92.0291 | 4 | 5 | 5 | 0.5274 | 0.0194 | 0.4966 | 0.0503 | 0.3943 | 🟢 STRONG CANDIDATE |
| Candidate_C | TARGET_D | random_forest | 91.7924 | 5 | 4 | 5 | 0.5607 | 0.0242 | 0.5224 | 0.0796 | 1.1478 | 🟢 STRONG CANDIDATE |
| Candidate_A | TARGET_B | random_forest | 79.3273 | 4 | 4 | 4 | 0.5203 | 0.0135 | 0.4961 | 0.0239 | 0.2438 | 🟢 STRONG CANDIDATE |
| Candidate_D | TARGET_D | xgboost | 77.7273 | 4 | 4 | 4 | 0.5674 | 0.0455 | 0.4814 | 0.0944 | 0.9828 | 🟢 STRONG CANDIDATE |

## Fold-by-Fold Performance Matrix

| candidate_id | fold | roc_auc | pr_auc | mcc | f1 | ppr | mean_realized_ret_buy | mean_realized_ret_sell | return_spread | cum_strategy_return | max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Candidate_A | 1 | 0.4961 | 0.4977 | -0.0072 | 0.4739 | 0.4650 | 0.0001 | 0.0007 | -0.0006 | -0.2319 | 0.8278 |
| Candidate_A | 2 | 0.5197 | 0.6000 | 0.0513 | 0.3369 | 0.2150 | 0.0111 | 0.0041 | 0.0070 | 5.7268 | 0.1481 |
| Candidate_A | 3 | 0.5276 | 0.5946 | 0.0016 | 0.5571 | 0.5594 | 0.0026 | 0.0023 | 0.0003 | 1.8355 | 0.3291 |
| Candidate_A | 4 | 0.5213 | 0.5661 | 0.0186 | 0.5714 | 0.5829 | 0.0034 | 0.0018 | 0.0016 | 3.5414 | 0.4532 |
| Candidate_A | 5 | 0.5366 | 0.5602 | 0.0552 | 0.4759 | 0.3910 | 0.0041 | 0.0002 | 0.0039 | 2.4657 | 0.2877 |
| Candidate_B | 1 | 0.4966 | 0.4807 | 0.0012 | 0.5269 | 0.5670 | 0.0014 | -0.0008 | 0.0022 | 0.2526 | 0.7821 |
| Candidate_B | 2 | 0.5136 | 0.5905 | 0.0282 | 0.3799 | 0.2720 | 0.0099 | 0.0040 | 0.0059 | 7.1639 | 0.1884 |
| Candidate_B | 3 | 0.5443 | 0.5908 | 0.0660 | 0.5979 | 0.5867 | 0.0033 | 0.0011 | 0.0022 | 3.3762 | 0.2445 |
| Candidate_B | 4 | 0.5349 | 0.5928 | 0.0579 | 0.5736 | 0.5498 | 0.0040 | 0.0012 | 0.0027 | 4.3786 | 0.3946 |
| Candidate_B | 5 | 0.5475 | 0.5727 | 0.0984 | 0.4218 | 0.2832 | 0.0066 | -0.0002 | 0.0067 | 3.3354 | 0.2814 |
| Candidate_C | 1 | 0.5224 | 0.5376 | -0.0048 | 0.6715 | 0.9442 | 0.0010 | -0.0106 | 0.0116 | -0.6592 | 0.9998 |
| Candidate_C | 2 | 0.5763 | 0.6972 | 0.1138 | 0.4415 | 0.2803 | 0.0338 | 0.0123 | 0.0215 | 1761.2983 | 0.2801 |
| Candidate_C | 3 | 0.5941 | 0.6497 | 0.1277 | 0.5544 | 0.4418 | 0.0152 | 0.0024 | 0.0128 | 196.1597 | 0.5084 |
| Candidate_C | 4 | 0.5511 | 0.5974 | 0.0889 | 0.5327 | 0.4301 | 0.0140 | 0.0048 | 0.0092 | 110.2935 | 0.6713 |
| Candidate_C | 5 | 0.5597 | 0.5964 | 0.0722 | 0.5268 | 0.4419 | 0.0072 | 0.0049 | 0.0023 | 7.9200 | 0.8550 |
| Candidate_D | 1 | 0.4814 | 0.5093 | -0.0359 | 0.6562 | 0.9098 | 0.0001 | 0.0026 | -0.0025 | -0.8305 | 0.9999 |
| Candidate_D | 2 | 0.6108 | 0.7080 | 0.1553 | 0.4209 | 0.2399 | 0.0371 | 0.0124 | 0.0246 | 1183.1861 | 0.2552 |
| Candidate_D | 3 | 0.5901 | 0.6507 | 0.1391 | 0.5052 | 0.3551 | 0.0145 | 0.0045 | 0.0100 | 58.1418 | 0.4467 |
| Candidate_D | 4 | 0.5643 | 0.6157 | 0.0689 | 0.5139 | 0.4159 | 0.0124 | 0.0061 | 0.0062 | 51.9181 | 0.7246 |
| Candidate_D | 5 | 0.5904 | 0.6287 | 0.1448 | 0.5338 | 0.3934 | 0.0124 | 0.0017 | 0.0108 | 39.0280 | 0.8471 |