# Mission 25 — Adaptive Exit & Asymmetric Risk Engine Report

## 1. Final Decision & Verdict

### **FINAL DECISION VERDICT: PROMISING**

**Executive Summary:**
Adaptive exits demonstrated risk-reduction benefits, but requires further refinement before replacing fixed 10D exits.

---

## 2. Locked Deterministic Control Baseline vs. Best Candidate Exit

* **Control Baseline:** `Candidate 1: Control Baseline (Fixed 10D)`
* **Winning Strategy:** `Candidate 5: Model Probability Deterioration`
* **Daily Equity Curve Sharpe:** `0.74` (Baseline) vs **`0.99`** (Best Candidate).
* **Signal Win Rate:** `59.09%` (Baseline) vs **`62.54%`** (Best Candidate).
* **Payoff Ratio (Avg Win / Avg Loss):** `1.20` (Baseline) vs **`1.35`** (Best Candidate).
* **Net Trade Expectancy:** `+1.16%` (Baseline) vs **`+0.74%`** per trade.
* **Maximum Drawdown:** `31.20%` (Baseline) vs **`25.83%`**.

---

## 3. Adaptive Exit Mechanism Candidate Comparison Matrix

| strategy | cum_return_pct | daily_sharpe | win_rate_pct | avg_win_pct | avg_loss_pct | payoff_ratio | profit_factor | expectancy_pct | max_drawdown_pct | total_trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Candidate 1: Control Baseline (Fixed 10D) | 67.34 | 0.74 | 59.09 | 4.64 | -3.94 | 1.20 | 1.90 | 1.16 | 31.20 | 55.20 |
| Candidate 2: Profit Target (+3.0%) + 10D Time Stop | 61.33 | 0.90 | 73.00 | 2.73 | -4.64 | 0.66 | 1.82 | 0.80 | 30.08 | 83.00 |
| Candidate 3: ATR Protective Stop (1.5x ATR14) | 75.25 | 0.77 | 52.88 | 5.12 | -3.50 | 1.49 | 1.87 | 1.03 | 30.41 | 68.40 |
| Candidate 4: Trailing ATR Exit (1.5x ATR14) | 16.96 | 0.28 | 44.18 | 4.46 | -2.81 | 1.58 | 1.31 | 0.28 | 33.72 | 79.60 |
| Candidate 5: Model Probability Deterioration | 65.04 | 0.99 | 62.54 | 2.88 | -2.75 | 1.35 | 2.20 | 0.74 | 25.83 | 77.00 |
| Candidate 6: Expected Return Decay Exit | 56.84 | 0.85 | 59.79 | 2.63 | -2.27 | 1.23 | 1.93 | 0.62 | 27.23 | 91.80 |
| Candidate 7: Combined Adaptive Exit | 46.36 | 0.89 | 64.04 | 2.27 | -2.83 | 0.97 | 1.71 | 0.44 | 27.69 | 120.60 |

## 4. Exit Reason Distribution Breakdown

| strategy | exit_reason | trade_id |
| --- | --- | --- |
| Candidate 1: Control Baseline (Fixed 10D) | TIME_LIMIT_10D | 276 |
| Candidate 2: Profit Target (+3.0%) + 10D Time Stop | PROFIT_TARGET | 282 |
| Candidate 2: Profit Target (+3.0%) + 10D Time Stop | TIME_LIMIT_10D | 133 |
| Candidate 3: ATR Protective Stop (1.5x ATR14) | STOP_LOSS | 125 |
| Candidate 3: ATR Protective Stop (1.5x ATR14) | TIME_LIMIT_10D | 217 |
| Candidate 4: Trailing ATR Exit (1.5x ATR14) | TIME_LIMIT_10D | 131 |
| Candidate 4: Trailing ATR Exit (1.5x ATR14) | TRAILING_STOP | 267 |
| Candidate 5: Model Probability Deterioration | PROB_DETERIORATION | 289 |
| Candidate 5: Model Probability Deterioration | TIME_LIMIT_10D | 96 |
| Candidate 6: Expected Return Decay Exit | RETURN_DECAY | 344 |
| Candidate 6: Expected Return Decay Exit | TIME_LIMIT_10D | 115 |
| Candidate 7: Combined Adaptive Exit | PROB_DETERIORATION | 165 |
| Candidate 7: Combined Adaptive Exit | PROFIT_TARGET | 279 |
| Candidate 7: Combined Adaptive Exit | STOP_LOSS | 126 |
| Candidate 7: Combined Adaptive Exit | TIME_LIMIT_10D | 33 |

## 5. Scientific Recommendation & Next Steps

* **Exit Engine Recommendation:** Exit candidate `Candidate 5: Model Probability Deterioration` provides optimal asymmetric risk control.
* **Production Integrity:** `config/champion.json` and production models remain **100% UNTOUCHED**.
* **Holdout Protection:** Final 15% out-of-sample holdout test partition remained **100% UNTOUCHED**.