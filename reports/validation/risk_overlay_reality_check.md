# Risk Overlay Reality Check Report: RELIANCE.NS & TCS.NS

## Overview

This report evaluates a **Risk Overlay Strategy** on `RELIANCE.NS` and `TCS.NS` independently. The overlay uses the exact existing Dual-Agreement model signals ($P(\text{up})$ classifier + continuous return regressor trained on `C59_COLS` features) as a macro tail-risk protection overlay:

* **Default State:** 100% invested in the single stock (like Buy-and-Hold).
* **Daily Check:** BEARISH condition: $P(\text{up}) \le 0.45 \text{ OR } \text{predicted 10-day return} < -0.5\%$.
* **If BEARISH:** Move 100% to cash (sell position, realistic 2026 NSE delivery exit fees applied).
* **If NOT BEARISH (HOLD or BUY):** Stay 100% invested. If coming from cash, buy back in (realistic 2026 NSE delivery entry fees applied).
* **Dynamic Holding:** Evaluated daily without forced 10-day exit. Positions and cash held for arbitrary lengths.

Performance is evaluated side-by-side against **Plain Buy-and-Hold** and the **Original Champion 10-Day-Cycling Strategy** over the identical 23-year evaluation period (**August 12, 2003 to August 6, 2026 — 5,695 trading days**) with realistic 2026 NSE delivery costs.

---

## Three-Way Strategy Comparison Tables

### 1. RELIANCE (`RELIANCE.NS`)

| Metric | Risk Overlay Strategy | Plain Buy-and-Hold | Original Champion 10-Day Strategy |
|---|---|---|---|
| **Total Cumulative Return** | **+1,819.22%** | **+5,099.69%** | **+522.60%** |
| **CAGR** | **13.97%** | **19.10%** | **8.33%** |
| **Daily Sharpe Ratio** | **0.32** | **0.43** | **0.50** |
| **Daily Sortino Ratio** | **1.18** | **1.08** | **0.43** |
| **Maximum Drawdown** | **52.81%** | **77.59%** | **51.81%** |
| **Total Position Transitions / Trades** | **770 Transitions** (385 Cash $\leftrightarrow$ Stock cycles) | 1 Trade | 200 Trades |

#### Plain Verdict & Whipsaw Analysis for RELIANCE.NS
* **Verdict:** No, the Risk Overlay did not hit buy-and-hold-like CAGR on RELIANCE (achieving 13.97% CAGR vs. Buy-and-Hold's 19.10%, a **5.13 percentage point gap**), although it did meaningfully reduce maximum drawdown by **24.78 percentage points** (52.81% vs. 77.59%).
* **Whipsaw Analysis:** The overlay executed **770 total transitions** (385 buy/sell round-trips over 23 years, averaging ~33 transitions per year), indicating severe whipsawing where frequent BEARISH signal flips triggered repeated ₹15.93 DP fees and 0.20% STT charges that dragged CAGR down by 5.13%.

---

### 2. TCS (`TCS.NS`)

| Metric | Risk Overlay Strategy | Plain Buy-and-Hold | Original Champion 10-Day Strategy |
|---|---|---|---|
| **Total Cumulative Return** | **+2,743.75%** | **+6,280.99%** | **+568.64%** |
| **CAGR** | **15.97%** | **20.01%** | **8.66%** |
| **Daily Sharpe Ratio** | **0.44** | **0.51** | **0.52** |
| **Daily Sortino Ratio** | **0.87** | **1.12** | **0.45** |
| **Maximum Drawdown** | **59.55%** | **66.36%** | **66.96%** |
| **Total Position Transitions / Trades** | **736 Transitions** (368 Cash $\leftrightarrow$ Stock cycles) | 1 Trade | 192 Trades |

#### Plain Verdict & Whipsaw Analysis for TCS.NS
* **Verdict:** No, the Risk Overlay did not hit buy-and-hold-like CAGR on TCS (achieving 15.97% CAGR vs. Buy-and-Hold's 20.01%, a **4.04 percentage point gap**) and failed to provide meaningful drawdown protection, reducing maximum drawdown by only **6.81 percentage points** (59.55% vs. 66.36%).
* **Whipsaw Analysis:** The overlay executed **736 total transitions** (368 buy/sell round-trips over 23 years, averaging ~32 transitions per year), causing significant friction from constant signal oscillation without yielding substantial drawdown protection.

---

## Summary Comparison Across Strategies

| Asset | Strategy | Total Return | CAGR | Sharpe | Sortino | Max DD | Transitions / Trades | CAGR Gap vs B&H | Max DD Cut vs B&H |
|---|---|---|---|---|---|---|---|---|---|
| **RELIANCE** | **Risk Overlay** | **+1,819.22%** | **13.97%** | **0.32** | **1.18** | **52.81%** | **770** | -5.13% | **-24.78%** |
| **RELIANCE** | Plain Buy & Hold | +5,099.69% | 19.10% | 0.43 | 1.08 | 77.59% | 1 | 0.00% | 0.00% |
| **RELIANCE** | Champion 10D | +522.60% | 8.33% | 0.50 | 0.43 | 51.81% | 200 | -10.77% | -25.78% |
| **TCS** | **Risk Overlay** | **+2,743.75%** | **15.97%** | **0.44** | **0.87** | **59.55%** | **736** | -4.04% | **-6.81%** |
| **TCS** | Plain Buy & Hold | +6,280.99% | 20.01% | 0.51 | 1.12 | 66.36% | 1 | 0.00% | 0.00% |
| **TCS** | Champion 10D | +568.64% | 8.66% | 0.52 | 0.45 | 66.96% | 192 | -11.35% | +0.60% |

---

## Reproducibility Script & Artifacts

* **Deliverable Report:** [`reports/validation/risk_overlay_reality_check.md`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/reports/validation/risk_overlay_reality_check.md)
* **Summary CSV Ledger:** [`reports/validation/risk_overlay_summary.csv`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/reports/validation/risk_overlay_summary.csv)
* **Measurement Script:** [`src/research/risk_overlay_reality_check.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/src/research/risk_overlay_reality_check.py)
