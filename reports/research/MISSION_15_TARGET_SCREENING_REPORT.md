# Mission 15 — Step 2: Target Horizon Screening Report

**Total Outer Evaluations:** 450 across 5 assets, 3 models, 5 outer folds, and 6 candidate targets (A–F)

## Executive Summary

This experiment evaluates candidate target horizons (TARGET_A through TARGET_F) holding feature set constant at `SHORTLIST_16` to isolate target reformulation effects.

## Target Summary Table

| target_name | total_folds | mean_roc_auc | mean_pr_auc | mean_mcc | mean_f1 | mean_precision | mean_recall | mean_accuracy | mean_ppr | mean_realized_ret_buy_pct | mean_realized_ret_sell_pct | return_spread_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TARGET_B | 75 | 0.5245 | 0.5564 | 0.0363 | 0.4786 | 0.5518 | 0.4681 | 0.5102 | 0.4519 | 0.6580 | 0.3379 | 0.3201 |
| TARGET_D | 75 | 0.5210 | 0.5854 | 0.0321 | 0.5175 | 0.5863 | 0.5316 | 0.5088 | 0.5186 | 0.6353 | 0.3337 | 0.3016 |
| TARGET_A | 75 | 0.5171 | 0.5270 | 0.0266 | 0.4726 | 0.5249 | 0.4759 | 0.5106 | 0.4638 | 0.6418 | 0.4061 | 0.2356 |
| TARGET_C | 75 | 0.5149 | 0.5606 | 0.0187 | 0.4961 | 0.5643 | 0.5050 | 0.5035 | 0.4969 | 0.6901 | 0.3354 | 0.3547 |
| TARGET_E | 75 | 0.5149 | 0.5606 | 0.0187 | 0.4961 | 0.5643 | 0.5050 | 0.5035 | 0.4969 | 0.6901 | 0.3354 | 0.3547 |
| TARGET_F | 75 | 0.5000 | 0.4082 | 0.0203 | 0.2888 | 0.3439 | 0.3435 | 0.3435 | 0.1246 | 0.5411 | 0.4650 | 0.0762 |

## Per-Model Summary

| model | target_name | total_folds | mean_roc_auc | mean_pr_auc | mean_mcc | mean_f1 | mean_ppr | return_spread_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | TARGET_A | 25 | 0.5166 | 0.5233 | 0.0292 | 0.4710 | 0.5014 | 0.1409 |
| logistic_regression | TARGET_B | 25 | 0.5225 | 0.5513 | 0.0281 | 0.4886 | 0.5132 | 0.1453 |
| logistic_regression | TARGET_C | 25 | 0.5098 | 0.5537 | 0.0118 | 0.5099 | 0.5815 | 0.4917 |
| logistic_regression | TARGET_D | 25 | 0.5074 | 0.5692 | 0.0090 | 0.5322 | 0.6181 | 0.1780 |
| logistic_regression | TARGET_E | 25 | 0.5098 | 0.5537 | 0.0118 | 0.5099 | 0.5815 | 0.4917 |
| logistic_regression | TARGET_F | 25 | 0.5000 | 0.4184 | 0.0095 | 0.2597 | 0.1129 | -0.2528 |
| random_forest | TARGET_A | 25 | 0.5182 | 0.5296 | 0.0260 | 0.4749 | 0.4464 | 0.3214 |
| random_forest | TARGET_B | 25 | 0.5261 | 0.5586 | 0.0364 | 0.4800 | 0.4286 | 0.3808 |
| random_forest | TARGET_C | 25 | 0.5185 | 0.5637 | 0.0261 | 0.4955 | 0.4569 | 0.3067 |
| random_forest | TARGET_D | 25 | 0.5261 | 0.5914 | 0.0464 | 0.5205 | 0.4827 | 0.3823 |
| random_forest | TARGET_E | 25 | 0.5185 | 0.5637 | 0.0261 | 0.4955 | 0.4569 | 0.3067 |
| random_forest | TARGET_F | 25 | 0.5000 | 0.4081 | 0.0288 | 0.3057 | 0.1356 | 0.1630 |
| xgboost | TARGET_A | 25 | 0.5164 | 0.5281 | 0.0246 | 0.4718 | 0.4435 | 0.2445 |
| xgboost | TARGET_B | 25 | 0.5250 | 0.5594 | 0.0443 | 0.4671 | 0.4140 | 0.4342 |
| xgboost | TARGET_C | 25 | 0.5163 | 0.5645 | 0.0181 | 0.4829 | 0.4523 | 0.2657 |
| xgboost | TARGET_D | 25 | 0.5295 | 0.5956 | 0.0409 | 0.4999 | 0.4551 | 0.3447 |
| xgboost | TARGET_E | 25 | 0.5163 | 0.5645 | 0.0181 | 0.4829 | 0.4523 | 0.2657 |
| xgboost | TARGET_F | 25 | 0.5000 | 0.3979 | 0.0227 | 0.3011 | 0.1253 | 0.3183 |

## Per-Asset Summary

| asset | target_name | total_folds | mean_roc_auc | mean_pr_auc | mean_mcc | mean_f1 | mean_ppr | return_spread_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hdfcbank_ns | TARGET_A | 15 | 0.5239 | 0.5276 | 0.0275 | 0.4227 | 0.3667 | 0.2700 |
| hdfcbank_ns | TARGET_B | 15 | 0.5225 | 0.5588 | 0.0329 | 0.4314 | 0.3884 | 0.3776 |
| hdfcbank_ns | TARGET_C | 15 | 0.5004 | 0.5552 | 0.0000 | 0.4775 | 0.4898 | 0.8216 |
| hdfcbank_ns | TARGET_D | 15 | 0.4966 | 0.5710 | -0.0020 | 0.5220 | 0.5366 | 0.1521 |
| hdfcbank_ns | TARGET_E | 15 | 0.5004 | 0.5552 | 0.0000 | 0.4775 | 0.4898 | 0.8216 |
| hdfcbank_ns | TARGET_F | 15 | 0.5000 | 0.4153 | 0.0293 | 0.2977 | 0.1166 | 0.3188 |
| icicibank_ns | TARGET_A | 15 | 0.5208 | 0.5230 | 0.0280 | 0.4773 | 0.4711 | 0.1114 |
| icicibank_ns | TARGET_B | 15 | 0.5249 | 0.5382 | 0.0404 | 0.4616 | 0.4108 | 0.4211 |
| icicibank_ns | TARGET_C | 15 | 0.5246 | 0.5397 | 0.0360 | 0.4769 | 0.4350 | 0.2668 |
| icicibank_ns | TARGET_D | 15 | 0.5308 | 0.5683 | 0.0436 | 0.4589 | 0.4055 | 0.2982 |
| icicibank_ns | TARGET_E | 15 | 0.5246 | 0.5397 | 0.0360 | 0.4769 | 0.4350 | 0.2668 |
| icicibank_ns | TARGET_F | 15 | 0.5000 | 0.3814 | 0.0116 | 0.2915 | 0.1575 | -0.4059 |
| infy_ns | TARGET_A | 15 | 0.5098 | 0.5256 | 0.0177 | 0.4605 | 0.4529 | 0.1575 |
| infy_ns | TARGET_B | 15 | 0.5351 | 0.5777 | 0.0434 | 0.4924 | 0.4701 | 0.3525 |
| infy_ns | TARGET_C | 15 | 0.5123 | 0.5734 | 0.0164 | 0.5035 | 0.5283 | 0.3147 |
| infy_ns | TARGET_D | 15 | 0.5091 | 0.6012 | 0.0179 | 0.5376 | 0.5832 | 0.3018 |
| infy_ns | TARGET_E | 15 | 0.5123 | 0.5734 | 0.0164 | 0.5035 | 0.5283 | 0.3147 |
| infy_ns | TARGET_F | 15 | 0.5000 | 0.4323 | 0.0072 | 0.2699 | 0.1225 | 0.4193 |
| reliance_ns | TARGET_A | 15 | 0.5129 | 0.5286 | 0.0292 | 0.4831 | 0.5143 | 0.1835 |
| reliance_ns | TARGET_B | 15 | 0.5206 | 0.5487 | 0.0326 | 0.4972 | 0.5010 | 0.0571 |
| reliance_ns | TARGET_C | 15 | 0.5117 | 0.5610 | 0.0048 | 0.5157 | 0.5619 | -0.0690 |
| reliance_ns | TARGET_D | 15 | 0.5293 | 0.5891 | 0.0431 | 0.5132 | 0.5206 | 0.2020 |
| reliance_ns | TARGET_E | 15 | 0.5117 | 0.5610 | 0.0048 | 0.5157 | 0.5619 | -0.0690 |
| reliance_ns | TARGET_F | 15 | 0.5000 | 0.3869 | 0.0127 | 0.2853 | 0.1197 | -0.6251 |
| tcs_ns | TARGET_A | 15 | 0.5179 | 0.5301 | 0.0306 | 0.5193 | 0.5138 | 0.4557 |
| tcs_ns | TARGET_B | 15 | 0.5196 | 0.5586 | 0.0321 | 0.5102 | 0.4894 | 0.3921 |
| tcs_ns | TARGET_C | 15 | 0.5254 | 0.5737 | 0.0361 | 0.5068 | 0.4696 | 0.4393 |
| tcs_ns | TARGET_D | 15 | 0.5392 | 0.5972 | 0.0580 | 0.5560 | 0.5472 | 0.5541 |
| tcs_ns | TARGET_E | 15 | 0.5254 | 0.5737 | 0.0361 | 0.5068 | 0.4696 | 0.4393 |
| tcs_ns | TARGET_F | 15 | 0.5000 | 0.4249 | 0.0408 | 0.2998 | 0.1066 | 0.6737 |