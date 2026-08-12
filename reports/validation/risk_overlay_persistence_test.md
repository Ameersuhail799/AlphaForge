# Risk Overlay Persistence Test ($N=3$ Consecutive Days) Report

## Overview

This report evaluates an $N=3$ **Persistence-Filtered Risk Overlay Strategy** on `RELIANCE.NS` and `TCS.NS` independently. The strategy tests a fixed, pre-committed $N=3$ consecutive-day persistence filter designed to solve the high-frequency whipsawing problem identified in the plain risk overlay audit:

* **State Tracking:** Tracks consecutive day streaks for BEARISH and NOT-BEARISH signals ($P(\text{up}) \le 0.45 \text{ OR } \text{predicted return} < -0.5\%$).
* **INVESTED $\to$ CASH:** Requires **3 consecutive trading days** of BEARISH signals before executing a sell-to-cash transition (realistic 2026 NSE delivery exit fees applied).
* **CASH $\to$ INVESTED:** Requires **3 consecutive trading days** of NOT-BEARISH signals before re-entering a 100% position (realistic 2026 NSE delivery entry fees applied).
* **Fixed Parameter:** $N=3$ is fixed and pre-committed (no parameter sweeping or post-hoc tuning).

Evaluated across the identical 23-year window (**August 12, 2003 to August 6, 2026 — 5,695 trading days**) with realistic 2026 NSE delivery costs.

---

## Four-Way Strategy Comparison Tables

### 1. RELIANCE (`RELIANCE.NS`)

| Metric | $N=3$ Persistence-Filtered Overlay | Plain Risk Overlay (No Filter) | Plain Buy-and-Hold | Original Champion 10-Day Strategy |
|---|---|---|---|---|
| **Total Cumulative Return** | **+974.67%** | **+1,819.22%** | **+5,099.69%** | **+522.60%** |
| **CAGR** | **11.08%** | **13.97%** | **19.10%** | **8.33%** |
| **Daily Sharpe Ratio** | **0.59** | **0.32** | **0.43** | **0.50** |
| **Daily Sortino Ratio** | **0.66** | **1.18** | **1.08** | **0.43** |
| **Maximum Drawdown** | **47.43%** | **52.81%** | **77.59%** | **51.81%** |
| **Total Position Transitions / Trades** | **194 Transitions** | **770 Transitions** | 1 Trade | 200 Trades |

#### Transitions Removed & Plain Verdict for RELIANCE.NS
* **Transitions Removed:** The $N=3$ filter removed **576 transitions** (reduced from 770 down to 194, a **74.8% reduction** in transaction churn).
* **Plain Verdict:** No, the $N=3$ persistence-filtered overlay did not land within 2-3 percentage points of buy-and-hold's CAGR on RELIANCE (achieving **11.08% CAGR** vs. Buy-and-Hold's **19.10% CAGR**, an **8.02 percentage point gap**), despite eliminating 576 churn transitions (74.8% reduction) and cutting maximum drawdown by 30.16 percentage points (**47.43%** vs. **77.59%**).

---

### 2. TCS (`TCS.NS`)

| Metric | $N=3$ Persistence-Filtered Overlay | Plain Risk Overlay (No Filter) | Plain Buy-and-Hold | Original Champion 10-Day Strategy |
|---|---|---|---|---|
| **Total Cumulative Return** | **+4,427.52%** | **+2,743.75%** | **+6,280.99%** | **+568.64%** |
| **CAGR** | **18.38%** | **15.97%** | **20.01%** | **8.66%** |
| **Daily Sharpe Ratio** | **0.48** | **0.44** | **0.51** | **0.52** |
| **Daily Sortino Ratio** | **0.89** | **0.87** | **1.12** | **0.45** |
| **Maximum Drawdown** | **65.74%** | **59.55%** | **66.36%** | **66.96%** |
| **Total Position Transitions / Trades** | **172 Transitions** | **736 Transitions** | 1 Trade | 192 Trades |

#### Transitions Removed & Plain Verdict for TCS.NS
* **Transitions Removed:** The $N=3$ filter removed **564 transitions** (reduced from 736 down to 172, a **76.6% reduction** in transaction churn).
* **Plain Verdict:** YES, the $N=3$ persistence-filtered overlay landed within 2-3 percentage points of buy-and-hold's CAGR on TCS (**18.38% CAGR** vs. Buy-and-Hold's **20.01% CAGR**, a tight **1.63 percentage point gap** after eliminating 564 churn transitions), but it failed to meaningfully cut maximum drawdown (**65.74%** vs. **66.36%**, only a 0.62 percentage point reduction).

---

## Overall Summary Matrix

| Asset | Strategy | Total Return | CAGR | Sharpe | Sortino | Max DD | Transitions | Churn Reduction | CAGR Gap vs B&H | Max DD Cut vs B&H |
|---|---|---|---|---|---|---|---|---|---|---|
| **RELIANCE** | **$N=3$ Filtered Overlay** | **+974.67%** | **11.08%** | **0.59** | **0.66** | **47.43%** | **194** | **-576 (-74.8%)** | -8.02% | **-30.16%** |
| **RELIANCE** | Plain Risk Overlay | +1,819.22% | 13.97% | 0.32 | 1.18 | 52.81% | 770 | Baseline | -5.13% | -24.78% |
| **RELIANCE** | Plain Buy & Hold | +5,099.69% | 19.10% | 0.43 | 1.08 | 77.59% | 1 | N/A | 0.00% | 0.00% |
| **RELIANCE** | Champion 10D | +522.60% | 8.33% | 0.50 | 0.43 | 51.81% | 200 | N/A | -10.77% | -25.78% |
| **TCS** | **$N=3$ Filtered Overlay** | **+4,427.52%** | **18.38%** | **0.48** | **0.89** | **65.74%** | **172** | **-564 (-76.6%)** | **-1.63%** | **-0.62%** |
| **TCS** | Plain Risk Overlay | +2,743.75% | 15.97% | 0.44 | 0.87 | 59.55% | 736 | Baseline | -4.04% | -6.81% |
| **TCS** | Plain Buy & Hold | +6,280.99% | 20.01% | 0.51 | 1.12 | 66.36% | 1 | N/A | 0.00% | 0.00% |
| **TCS** | Champion 10D | +568.64% | 8.66% | 0.52 | 0.45 | 66.96% | 192 | N/A | -11.35% | +0.60% |

---

## Reproducibility Artifacts

* **Deliverable Markdown Report:** [`reports/validation/risk_overlay_persistence_test.md`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/reports/validation/risk_overlay_persistence_test.md)
* **Summary CSV Ledger:** [`reports/validation/risk_overlay_persistence_summary.csv`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/reports/validation/risk_overlay_persistence_summary.csv)
* **Measurement Script:** [`src/research/risk_overlay_persistence_test.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/src/research/risk_overlay_persistence_test.py)
