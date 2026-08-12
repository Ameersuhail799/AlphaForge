# Mission 18 — Production-Grade Paper Trading Engine Report

## Executive Summary

This report presents the validated performance of AlphaForge's **Production-Grade Paper Trading Engine** evaluating `TCS + Random Forest + C57` across Mode A (Single Position 100%) and Mode B (10-Slot Portfolio 10%).

## 1. Overall Portfolio Mode Performance Summary (10 bps Default)

| mode | entry_timing | mean_cum_return_pct | median_cum_return_pct | mean_return_ex_fold2_pct | mean_wealth_multiple | mean_max_drawdown_pct | profitable_folds_count | total_trades_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MODE_A | NEXT_BAR_OPEN | 65.1965 | 60.5226 | 28.8504 | 1.6520 | 29.2158 | 4 | 247 |
| MODE_A | SAME_BAR_CLOSE | 59.9816 | 53.6989 | 35.1105 | 1.5998 | 28.1895 | 4 | 264 |
| MODE_B | NEXT_BAR_OPEN | 26.5220 | 34.8567 | 19.2037 | 1.2652 | 24.1231 | 4 | 1486 |
| MODE_B | SAME_BAR_CLOSE | 30.5768 | 33.8653 | 23.6942 | 1.3058 | 21.9550 | 4 | 1550 |

## 2. Fold 2 Dependency & Regime Breakdown

| fold | mode | entry_timing | cum_return_pct | wealth_multiple | max_drawdown_pct | total_trades |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | MODE_A | SAME_BAR_CLOSE | -19.3806 | 0.8062 | 65.0230 | 80 |
| 1 | MODE_A | NEXT_BAR_OPEN | -44.6703 | 0.5533 | 72.1855 | 73 |
| 1 | MODE_B | SAME_BAR_CLOSE | -19.9053 | 0.8009 | 57.4290 | 642 |
| 1 | MODE_B | NEXT_BAR_OPEN | -37.4363 | 0.6256 | 65.2203 | 607 |
| 2 | MODE_A | SAME_BAR_CLOSE | 159.4659 | 2.5947 | 16.4410 | 36 |
| 2 | MODE_A | NEXT_BAR_OPEN | 210.5813 | 3.1058 | 19.8367 | 35 |
| 2 | MODE_B | SAME_BAR_CLOSE | 58.1075 | 1.5811 | 14.5147 | 104 |
| 2 | MODE_B | NEXT_BAR_OPEN | 55.7950 | 1.5580 | 14.3795 | 102 |
| 3 | MODE_A | SAME_BAR_CLOSE | 53.6989 | 1.5370 | 15.6064 | 43 |
| 3 | MODE_A | NEXT_BAR_OPEN | 70.4899 | 1.7049 | 11.5281 | 42 |
| 3 | MODE_B | SAME_BAR_CLOSE | 33.8653 | 1.3387 | 8.9027 | 226 |
| 3 | MODE_B | NEXT_BAR_OPEN | 31.1332 | 1.3113 | 9.1059 | 220 |
| 4 | MODE_A | SAME_BAR_CLOSE | 74.4035 | 1.7440 | 20.8028 | 50 |
| 4 | MODE_A | NEXT_BAR_OPEN | 60.5226 | 1.6052 | 17.9656 | 46 |
| 4 | MODE_B | SAME_BAR_CLOSE | 47.2386 | 1.4724 | 9.2234 | 256 |
| 4 | MODE_B | NEXT_BAR_OPEN | 48.2612 | 1.4826 | 10.7474 | 246 |
| 5 | MODE_A | SAME_BAR_CLOSE | 31.7204 | 1.3172 | 23.0741 | 55 |
| 5 | MODE_A | NEXT_BAR_OPEN | 29.0592 | 1.2906 | 24.5633 | 51 |
| 5 | MODE_B | SAME_BAR_CLOSE | 33.5782 | 1.3358 | 19.7053 | 322 |
| 5 | MODE_B | NEXT_BAR_OPEN | 34.8567 | 1.3486 | 21.1626 | 311 |