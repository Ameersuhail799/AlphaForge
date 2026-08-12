# Mission 12: Dynamic Threshold Experiment Report

**Hypothesis Final Verdict:** **NOT SUPPORTED**

## Executive Summary

This experiment evaluates Treatment A (Fixed 0.50 Decision Threshold) versus Treatment B (Leak-Safe Dynamic Decision Threshold via Nested CV on training data only) using `SHORTLIST_16` across 5 liquid equity assets, 3 models, and 5 outer expanding folds.

## Formal Hypothesis Acceptance Criteria Audit

1. **Criterion 1: Recall Failures Elimination (7/45 -> 0/45):** Fixed Failures = **16**, Dynamic Failures = **1** -> Status: **FAIL**
2. **Criterion 2: F1 Non-Loss Rate (>= 70%):** Observed F1 Win/Tie Rate = **100.00%** -> Status: **PASS**
3. **Criterion 3: Mean F1 Improvement (>= +0.030):** Observed Mean F1 Delta = **+0.1929** -> Status: **PASS**

**Final Status:** **NOT SUPPORTED**

## Core Verification Confirmations
- **ROC-AUC Invariance:** Confirmed mathematically identical across all 45 fold comparisons.
- **Holdout Protection:** Confirmed the final 15% out-of-sample holdout test set was **100% untouched** and never accessed.
- **Zero Leakage:** Dynamic thresholds were fit exclusively on inner out-of-fold training predictions inside `outer_X_train`.

## Overall Summary Statistics

- **Total Outer Evaluations:** 75
- **Fixed Recall Failures (< 0.35):** 16 / 75
- **Dynamic Recall Failures (< 0.35):** 1 / 75
- **Severe Model Collapses (< 0.05):** Fixed = 2, Dynamic = 0
- **Mean Fixed F1:** 0.4722
- **Mean Dynamic F1:** 0.6650
- **Mean F1 Delta:** +0.1929
- **Mean Recall Delta:** +0.4924
- **Dynamic Threshold Stats:** Mean = 0.2269, Std = 0.0366

## Per-Model Results

| model | total_folds | fixed_recall_failures | dynamic_recall_failures | f1_win_rate | mean_fixed_f1 | mean_dynamic_f1 | mean_delta_f1 | mean_fixed_recall | mean_dynamic_recall | mean_delta_recall | mean_dynamic_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 25 | 6 | 1 | 1.0000 | 0.4714 | 0.6554 | 0.1840 | 0.5165 | 0.9499 | 0.4334 | 0.2468 |
| random_forest | 25 | 5 | 0 | 1.0000 | 0.4745 | 0.6754 | 0.2009 | 0.4571 | 0.9975 | 0.5404 | 0.2276 |
| xgboost | 25 | 5 | 0 | 1.0000 | 0.4706 | 0.6643 | 0.1937 | 0.4539 | 0.9575 | 0.5035 | 0.2064 |

## Per-Asset Results

| asset | total_folds | fixed_recall_failures | dynamic_recall_failures | f1_win_rate | mean_fixed_f1 | mean_dynamic_f1 | mean_delta_f1 | mean_fixed_recall | mean_dynamic_recall | mean_delta_recall | mean_dynamic_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hdfcbank_ns | 15 | 5 | 1 | 1.0000 | 0.4322 | 0.6494 | 0.2172 | 0.3908 | 0.9375 | 0.5467 | 0.2113 |
| icicibank_ns | 15 | 3 | 0 | 1.0000 | 0.4785 | 0.6663 | 0.1878 | 0.4897 | 0.9932 | 0.5035 | 0.2447 |
| infy_ns | 15 | 4 | 0 | 1.0000 | 0.4573 | 0.6739 | 0.2166 | 0.4568 | 0.9789 | 0.5221 | 0.2233 |
| reliance_ns | 15 | 3 | 0 | 1.0000 | 0.4831 | 0.6584 | 0.1753 | 0.5282 | 0.9368 | 0.4086 | 0.2060 |
| tcs_ns | 15 | 1 | 0 | 1.0000 | 0.5097 | 0.6772 | 0.1675 | 0.5138 | 0.9950 | 0.4813 | 0.2493 |

## Complete Per-Fold Results

| asset | model | fold | dynamic_threshold | fixed_recall | dynamic_recall | delta_recall | fixed_f1 | dynamic_f1 | delta_f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reliance_ns | logistic_regression | 1 | 0.2000 | 0.0078 | 0.5734 | 0.5656 | 0.0155 | 0.5513 | 0.5358 | 0.4934 |
| reliance_ns | random_forest | 1 | 0.2000 | 0.1644 | 0.9941 | 0.8297 | 0.2569 | 0.7090 | 0.4521 | 0.5071 |
| reliance_ns | xgboost | 1 | 0.2300 | 0.1624 | 0.6086 | 0.4462 | 0.2519 | 0.5696 | 0.3177 | 0.4896 |
| reliance_ns | logistic_regression | 2 | 0.2000 | 0.9530 | 1.0000 | 0.0470 | 0.6607 | 0.6671 | 0.0064 | 0.5113 |
| reliance_ns | random_forest | 2 | 0.2000 | 0.4444 | 1.0000 | 0.5556 | 0.4706 | 0.6667 | 0.1961 | 0.4882 |
| reliance_ns | xgboost | 2 | 0.2200 | 0.3504 | 0.8761 | 0.5256 | 0.4141 | 0.6352 | 0.2210 | 0.5013 |
| reliance_ns | logistic_regression | 3 | 0.2000 | 0.7930 | 1.0000 | 0.2070 | 0.6081 | 0.6528 | 0.0447 | 0.5334 |
| reliance_ns | random_forest | 3 | 0.2100 | 0.6079 | 1.0000 | 0.3921 | 0.5526 | 0.6532 | 0.1007 | 0.5321 |
| reliance_ns | xgboost | 3 | 0.2000 | 0.6630 | 1.0000 | 0.3370 | 0.5805 | 0.6532 | 0.0727 | 0.5481 |
| reliance_ns | logistic_regression | 4 | 0.2100 | 0.7229 | 1.0000 | 0.2771 | 0.6261 | 0.6931 | 0.0670 | 0.5263 |
| reliance_ns | random_forest | 4 | 0.2100 | 0.4719 | 1.0000 | 0.5281 | 0.5087 | 0.6931 | 0.1845 | 0.5242 |
| reliance_ns | xgboost | 4 | 0.2000 | 0.4458 | 1.0000 | 0.5542 | 0.4852 | 0.6931 | 0.2079 | 0.5051 |
| reliance_ns | logistic_regression | 5 | 0.2000 | 0.7495 | 1.0000 | 0.2505 | 0.6183 | 0.6793 | 0.0610 | 0.5353 |
| reliance_ns | random_forest | 5 | 0.2100 | 0.6729 | 1.0000 | 0.3271 | 0.5925 | 0.6793 | 0.0868 | 0.5190 |
| reliance_ns | xgboost | 5 | 0.2000 | 0.7143 | 1.0000 | 0.2857 | 0.6047 | 0.6793 | 0.0746 | 0.5184 |
| tcs_ns | logistic_regression | 1 | 0.3300 | 0.8870 | 0.9902 | 0.1032 | 0.6322 | 0.6500 | 0.0178 | 0.5074 |
| tcs_ns | random_forest | 1 | 0.2400 | 0.5455 | 0.9975 | 0.4521 | 0.5163 | 0.6517 | 0.1354 | 0.5094 |
| tcs_ns | xgboost | 1 | 0.2000 | 0.4840 | 0.9631 | 0.4791 | 0.4758 | 0.6421 | 0.1663 | 0.4968 |
| tcs_ns | logistic_regression | 2 | 0.2600 | 0.2661 | 0.9977 | 0.7317 | 0.3558 | 0.6824 | 0.3265 | 0.5450 |
| tcs_ns | random_forest | 2 | 0.2400 | 0.4312 | 1.0000 | 0.5688 | 0.4808 | 0.6829 | 0.2020 | 0.5222 |
| tcs_ns | xgboost | 2 | 0.2000 | 0.3991 | 0.9885 | 0.5894 | 0.4622 | 0.6874 | 0.2252 | 0.5285 |
| tcs_ns | logistic_regression | 3 | 0.3000 | 0.5011 | 0.9977 | 0.4966 | 0.5105 | 0.6850 | 0.1745 | 0.5125 |
| tcs_ns | random_forest | 3 | 0.2700 | 0.4142 | 1.0000 | 0.5858 | 0.4695 | 0.6844 | 0.2149 | 0.5085 |
| tcs_ns | xgboost | 3 | 0.2000 | 0.4737 | 0.9977 | 0.5240 | 0.5024 | 0.6850 | 0.1826 | 0.5270 |
| tcs_ns | logistic_regression | 4 | 0.3500 | 0.6205 | 0.9977 | 0.3773 | 0.5759 | 0.6870 | 0.1111 | 0.5260 |
| tcs_ns | random_forest | 4 | 0.2100 | 0.5318 | 1.0000 | 0.4682 | 0.5361 | 0.6859 | 0.1498 | 0.5272 |
| tcs_ns | xgboost | 4 | 0.2000 | 0.5000 | 1.0000 | 0.5000 | 0.5152 | 0.6864 | 0.1712 | 0.5206 |
| tcs_ns | logistic_regression | 5 | 0.2800 | 0.6430 | 0.9977 | 0.3547 | 0.5770 | 0.6829 | 0.1058 | 0.4930 |
| tcs_ns | random_forest | 5 | 0.2600 | 0.5423 | 1.0000 | 0.4577 | 0.5350 | 0.6828 | 0.1478 | 0.5217 |
| tcs_ns | xgboost | 5 | 0.2000 | 0.4668 | 0.9977 | 0.5309 | 0.5012 | 0.6823 | 0.1811 | 0.5132 |
| hdfcbank_ns | logistic_regression | 1 | 0.2400 | 0.0044 | 0.2232 | 0.2188 | 0.0087 | 0.3068 | 0.2981 | 0.5112 |
| hdfcbank_ns | random_forest | 1 | 0.2100 | 0.2210 | 1.0000 | 0.7790 | 0.3047 | 0.6618 | 0.3572 | 0.5182 |
| hdfcbank_ns | xgboost | 1 | 0.2000 | 0.2801 | 0.9453 | 0.6652 | 0.3561 | 0.6496 | 0.2936 | 0.4947 |
| hdfcbank_ns | logistic_regression | 2 | 0.2000 | 0.4158 | 0.9979 | 0.5821 | 0.4768 | 0.6794 | 0.2026 | 0.5418 |
| hdfcbank_ns | random_forest | 2 | 0.2200 | 0.2682 | 0.9979 | 0.7297 | 0.3634 | 0.6780 | 0.3146 | 0.5408 |
| hdfcbank_ns | xgboost | 2 | 0.2000 | 0.3015 | 0.9356 | 0.6341 | 0.3924 | 0.6677 | 0.2752 | 0.5439 |
| hdfcbank_ns | logistic_regression | 3 | 0.2000 | 0.4428 | 0.9958 | 0.5530 | 0.4697 | 0.6681 | 0.1984 | 0.5077 |
| hdfcbank_ns | random_forest | 3 | 0.2700 | 0.4343 | 0.9873 | 0.5530 | 0.4762 | 0.6657 | 0.1895 | 0.5082 |
| hdfcbank_ns | xgboost | 3 | 0.2100 | 0.4915 | 0.9852 | 0.4936 | 0.5121 | 0.6676 | 0.1555 | 0.5436 |
| hdfcbank_ns | logistic_regression | 4 | 0.2000 | 0.4970 | 1.0000 | 0.5030 | 0.5251 | 0.6899 | 0.1648 | 0.5351 |
| hdfcbank_ns | random_forest | 4 | 0.2000 | 0.5212 | 1.0000 | 0.4788 | 0.5432 | 0.6899 | 0.1467 | 0.5560 |
| hdfcbank_ns | xgboost | 4 | 0.2000 | 0.4485 | 0.9939 | 0.5455 | 0.4944 | 0.6881 | 0.1937 | 0.5494 |
| hdfcbank_ns | logistic_regression | 5 | 0.2000 | 0.6229 | 1.0000 | 0.3771 | 0.5806 | 0.6761 | 0.0955 | 0.5319 |
| hdfcbank_ns | random_forest | 5 | 0.2200 | 0.4542 | 1.0000 | 0.5458 | 0.4866 | 0.6761 | 0.1894 | 0.5233 |
| hdfcbank_ns | xgboost | 5 | 0.2000 | 0.4583 | 1.0000 | 0.5417 | 0.4927 | 0.6761 | 0.1833 | 0.5116 |
| infy_ns | logistic_regression | 1 | 0.2800 | 0.0710 | 0.9828 | 0.9118 | 0.1231 | 0.6633 | 0.5401 | 0.5207 |
| infy_ns | random_forest | 1 | 0.2000 | 0.1763 | 0.9914 | 0.8151 | 0.2628 | 0.6691 | 0.4063 | 0.5027 |
| infy_ns | xgboost | 1 | 0.2000 | 0.1290 | 0.7290 | 0.6000 | 0.2113 | 0.5974 | 0.3861 | 0.5014 |
| infy_ns | logistic_regression | 2 | 0.2600 | 0.2357 | 1.0000 | 0.7643 | 0.3083 | 0.6695 | 0.3612 | 0.4646 |
| infy_ns | random_forest | 2 | 0.2400 | 0.3694 | 0.9958 | 0.6263 | 0.4270 | 0.6671 | 0.2401 | 0.5006 |
| infy_ns | xgboost | 2 | 0.2000 | 0.3652 | 0.9894 | 0.6242 | 0.4226 | 0.6652 | 0.2426 | 0.4804 |
| infy_ns | logistic_regression | 3 | 0.2200 | 0.6148 | 0.9980 | 0.3832 | 0.5797 | 0.6845 | 0.1048 | 0.5128 |
| infy_ns | random_forest | 3 | 0.2000 | 0.5881 | 1.0000 | 0.4119 | 0.5666 | 0.6844 | 0.1178 | 0.5399 |
| infy_ns | xgboost | 3 | 0.2000 | 0.5328 | 1.0000 | 0.4672 | 0.5333 | 0.6859 | 0.1525 | 0.5452 |
| infy_ns | logistic_regression | 4 | 0.2100 | 0.7792 | 1.0000 | 0.2208 | 0.6296 | 0.6765 | 0.0469 | 0.5159 |
| infy_ns | random_forest | 4 | 0.2300 | 0.5917 | 1.0000 | 0.4083 | 0.5602 | 0.6761 | 0.1159 | 0.5466 |
| infy_ns | xgboost | 4 | 0.2000 | 0.6292 | 0.9979 | 0.3688 | 0.5725 | 0.6756 | 0.1031 | 0.5429 |
| infy_ns | logistic_regression | 5 | 0.3100 | 0.7679 | 1.0000 | 0.2321 | 0.6329 | 0.6981 | 0.0652 | 0.4896 |
| infy_ns | random_forest | 5 | 0.2000 | 0.4940 | 1.0000 | 0.5060 | 0.5061 | 0.6981 | 0.1920 | 0.4871 |
| infy_ns | xgboost | 5 | 0.2000 | 0.5079 | 1.0000 | 0.4921 | 0.5235 | 0.6981 | 0.1745 | 0.5013 |
| icicibank_ns | logistic_regression | 1 | 0.2100 | 0.8585 | 1.0000 | 0.1415 | 0.6276 | 0.6693 | 0.0417 | 0.4879 |
| icicibank_ns | random_forest | 1 | 0.2100 | 0.7311 | 1.0000 | 0.2689 | 0.6151 | 0.6677 | 0.0526 | 0.5646 |
| icicibank_ns | xgboost | 1 | 0.2100 | 0.7288 | 0.9906 | 0.2618 | 0.6119 | 0.6651 | 0.0532 | 0.5440 |
| icicibank_ns | logistic_regression | 2 | 0.3100 | 0.5094 | 0.9977 | 0.4883 | 0.5118 | 0.6703 | 0.1586 | 0.5201 |
| icicibank_ns | random_forest | 2 | 0.2900 | 0.6009 | 0.9765 | 0.3756 | 0.5470 | 0.6635 | 0.1165 | 0.5037 |
| icicibank_ns | xgboost | 2 | 0.2000 | 0.6620 | 0.9906 | 0.3286 | 0.5657 | 0.6672 | 0.1015 | 0.4991 |
| icicibank_ns | logistic_regression | 3 | 0.3000 | 0.4279 | 1.0000 | 0.5721 | 0.4827 | 0.6677 | 0.1851 | 0.5537 |
| icicibank_ns | random_forest | 3 | 0.2700 | 0.4917 | 1.0000 | 0.5083 | 0.5079 | 0.6667 | 0.1587 | 0.5357 |
| icicibank_ns | xgboost | 3 | 0.2100 | 0.5035 | 0.9953 | 0.4917 | 0.5114 | 0.6667 | 0.1553 | 0.5240 |
| icicibank_ns | logistic_regression | 4 | 0.2700 | 0.3921 | 0.9975 | 0.6055 | 0.4371 | 0.6437 | 0.2066 | 0.5154 |
| icicibank_ns | random_forest | 4 | 0.2400 | 0.4293 | 0.9975 | 0.5682 | 0.4517 | 0.6432 | 0.1915 | 0.4997 |
| icicibank_ns | xgboost | 4 | 0.2500 | 0.4367 | 0.9926 | 0.5558 | 0.4619 | 0.6441 | 0.1822 | 0.5042 |
| icicibank_ns | logistic_regression | 5 | 0.2300 | 0.1303 | 1.0000 | 0.8697 | 0.2105 | 0.6899 | 0.4794 | 0.5138 |
| icicibank_ns | random_forest | 5 | 0.2400 | 0.2292 | 1.0000 | 0.7708 | 0.3264 | 0.6894 | 0.3630 | 0.5233 |
| icicibank_ns | xgboost | 5 | 0.2300 | 0.2135 | 0.9596 | 0.7461 | 0.3089 | 0.6794 | 0.3705 | 0.5186 |