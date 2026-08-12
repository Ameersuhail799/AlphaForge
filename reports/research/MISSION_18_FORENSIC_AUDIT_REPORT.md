# Mission 18 Forensic Audit Report — Discrepancy Investigation & Corrections

## Executive Summary

This independent forensic audit investigated the three specific discrepancies identified in Mission 18. All root causes were isolated, verified via unit tests, and corrected against raw market data without altering production code, feature pipelines, target definitions, or model hyperparameters.

---

## 1. Discrepancy #1: Mode A 10-bps Return (+59.98% vs +62.59%)

* **Reported Value in Table 1 & 2:** **`+59.98%`**
* **Reported Value in Cost Sensitivity Table:** **`+62.59%`**
* **Root Cause:** In `paper_trading_engine.py`, the cost sensitivity aggregation code grouped results by `cost_bps` without filtering by `entry_timing == 'SAME_BAR_CLOSE'`. For `cost_bps = 10.0`, the dataframe contained TWO entries per fold: `SAME_BAR_CLOSE` (**`+59.98%`**) and `NEXT_BAR_OPEN` (**`+65.20%`**). The cost sensitivity table averaged these two timing modes together: $\frac{59.9816 + 65.1965}{2} = 62.5891\%$ (**`+62.59%`**).
* **Corrected Mode A 10-bps (SAME_BAR_CLOSE) Return:** **`+59.98%`** (`1.60x` wealth multiple)
* **Corrected Mode A 10-bps (NEXT_BAR_OPEN) Return:** **`+65.20%`** (`1.65x` wealth multiple)

---

## 2. Discrepancy #2: Buy & Hold TCS Benchmark Wealth Multiple (227,154x vs True Price Ratio)

* **Reported Value in Mission 18:** **`227,154x`** (+2.27e+07%)
* **Root Cause:** The Buy & Hold benchmark in `mission17_strategy_validation.py` and `paper_trading_engine.py` was calculated using `np.cumprod(1.0 + realized_ret_10d)`, which compounded 10-day forward return percentages on consecutive daily bars $(1 + r_{10d})^{843}$, creating an artificial huge compounding multiplier.
* **True Buy & Hold Calculation:** The true Buy & Hold return across a validation window from $P_{\text{start}}$ to $P_{\text{end}}$ is simply $\frac{P_{\text{end}} - P_{\text{start}}}{P_{\text{start}}}$.
* **Corrected True Buy & Hold Mean Return:** **`+102.38%`** (**`2.02x`** wealth multiple / +0.28x profit multiple).

---

## 3. Discrepancy #3: 4-Tests vs 5-Verification-Items Reconciled

* **Root Cause:** `test_paper_trading_engine.py` contained 4 test functions (`test_01` to `test_04`), which executed 5 distinct verification assertions (1. Future price mutation, 2. Mode A single position limit, 3. Mode B 10-slot limit, 4. Accounting balance reconciliation, 5. Holdout protection).
* **Correction:** Expanded unit test suite with 2 dedicated forensic tests (`test_05_true_buy_and_hold_calculation` and `test_06_cost_sensitivity_entry_timing_isolation`), bringing the test suite to **6 total test functions**.

---

## 4. Corrected Performance Matrix Across All Configurations

| fold | mode_a_same_bar_close_return_pct | mode_a_next_bar_open_return_pct | mode_b_same_bar_close_return_pct | random_signal_return_pct | true_buy_hold_return_pct | true_buy_hold_wealth_multiple |
| --- | --- | --- | --- | --- | --- | --- |
| 1.00 | -19.38 | -44.67 | -19.91 | -6.36 | -20.19 | 0.80 |
| 2.00 | 159.47 | 210.58 | 58.11 | 398.48 | 316.87 | 4.17 |
| 3.00 | 53.70 | 70.49 | 33.87 | 14.43 | 67.21 | 1.67 |
| 4.00 | 74.40 | 60.52 | 47.24 | 43.20 | 100.30 | 2.00 |
| 5.00 | 31.72 | 29.06 | 33.58 | 85.69 | 47.69 | 1.48 |

## 5. Corrected Cost Sensitivity (SAME_BAR_CLOSE Only)

| Cost Scenario (bps) | Corrected Mode A Mean Return (%) |
| --- | --- |
| 0.0 bps | +67.86% |
| 5.0 bps | +63.87% |
| 10.0 bps | +59.98% |
| 20.0 bps | +52.50% |
| 50.0 bps | +32.19% |