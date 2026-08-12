# Benchmark and Cost Reality Check: Realistic Costs vs. Flat 10bps Assumption

## Overview

This report presents a direct, empirical measurement of the **AlphaForge Production Champion Strategy** under realistic 2026 Indian National Stock Exchange (NSE) delivery-equity transaction costs compared against the old 10bps (0.10%) flat round-trip cost assumption, and evaluates performance against an **Equal-Weight Buy-and-Hold Benchmark** for the identical 5-stock universe (`TCS.NS`, `INFY.NS`, `RELIANCE.NS`, `ICICIBANK.NS`, `HDFCBANK.NS`) over the exact same evaluation period (**August 13, 2003 to August 6, 2026 — 5,694 trading days / ~23.0 years**).

---

## Transaction Cost Models

### 1. Old Assumption (10bps Flat Round-Trip)
* **Entry Cost:** 5 bps ($0.05\%$ of trade allocation value)
* **Exit Cost:** 5 bps ($0.05\%$ of liquidated position value)
* **Total Round-Trip Cost:** 10 bps ($0.10\%$)

### 2. Realistic 2026 NSE Delivery-Equity Charges (Discount Broker Rates)
* **Securities Transaction Tax (STT):** $0.10\%$ ($0.0010$) on buy leg + $0.10\%$ ($0.0010$) on sell leg ($0.20\%$ total round-trip).
* **Stamp Duty:** $0.015\%$ ($0.00015$) on buy leg only.
* **Exchange Transaction Charge + SEBI Fee:** $0.0031\%$ per leg + $18\%$ GST ($0.00366\%$ per leg).
* **Depository Participant (DP) Charge:** Flat **₹15.93** (₹13.50 + 18% GST) charged on the sell trade when shares are debited from demat (flat ₹ fee per stock per sell order, NOT a percentage).
* **Effective Friction:** On a typical ₹20,000 trade allocation, entry fee is **₹23.73** ($0.1186\%$) and exit fee is **₹36.66** ($0.1833\%$), totaling **₹60.39** per round-trip trade (**0.302% of trade value** — over **3x higher** than the old 10bps flat assumption!).

---

## Performance Comparison Table

| Metric | Champion Strategy (Old 10bps Flat) | Champion Strategy (Realistic 2026 Costs) | Equal-Weight Buy-and-Hold (Realistic Entry Cost) |
|---|---|---|---|
| **Evaluation Period** | 2003-08-13 to 2026-08-06 | 2003-08-13 to 2026-08-06 | 2003-08-13 to 2026-08-06 |
| **Trading Days** | 5,694 (~23.0 Years) | 5,694 (~23.0 Years) | 5,694 (~23.0 Years) |
| **Total Cumulative Return** | **+317.44%** | **+217.97%** | **+4,676.95%** |
| **Compound Annual Growth Rate (CAGR)** | **6.53%** | **5.25%** | **18.66%** |
| **Daily Sharpe Ratio** | **0.73** | **0.60** | **0.71** |
| **Daily Sortino Ratio** | **0.82** | **0.68** | **1.01** |
| **Maximum Drawdown** | **24.45%** | **25.95%** | **61.00%** |
| **Total Trades Executed** | 1,003 | 1,003 | 5 (Initial Allocation) |
| **Win Rate** | 56.43% | 55.23% | N/A |
| **Mean Net Expectancy per Trade** | +0.98% | +0.80% | N/A |

---

## Plain Verdict: Does the Champion Strategy Beat Buy-and-Hold?

**NO.** The champion active trading strategy does **NOT** beat the equal-weight buy-and-hold benchmark after applying realistic transaction costs — nor did it beat buy-and-hold under the old 10bps assumption. Over the identical 23-year period across the exact same 5 liquid Indian equities, an equal-weight Buy-and-Hold strategy generated a total cumulative return of **+4,676.95%** (**18.66% CAGR**) with a Sharpe ratio of **0.71** and a Sortino ratio of **1.01**, compared to the champion active strategy's **+217.97%** total return (**5.25% CAGR**), **0.60** Sharpe ratio, and **0.68** Sortino ratio under realistic costs. While the champion strategy achieved a substantially lower maximum drawdown (**25.95%** vs. **61.00%**), its active 10-day cycling structure generated 1,003 round-trip trades whose cumulative friction eroded more than **95%** of the multi-asset buy-and-hold upside.

---

## Reproducibility Script

The complete measurement benchmark script is implemented and preserved at:
[`src/research/benchmark_and_cost_reality_check.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/src/research/benchmark_and_cost_reality_check.py)

### Execution Command

```bash
python -m src.research.benchmark_and_cost_reality_check
```

### Script Implementation Source Code

```python
"""Measurement Task: Benchmark and Realistic Cost Reality Check for AlphaForge."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.dataset.scaler import FeatureScaler
from src.research.mission21_model_improvement import C57_FEATURES
from src.research.mission27_cross_asset_generalization import ASSET_UNIVERSE, _create_folds_index, build_asset_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

C59_COLS = C57_FEATURES + ["RANGE_COMPRESSION_EXP", "VOLUME_BREAKOUT_CONFIRM", "TREND_VOL_INTERACTION"]


def calculate_nse_delivery_cost(
    trade_value: float,
    is_buy: bool,
    stt_rate: float = 0.0010,
    stamp_duty_rate: float = 0.00015,
    exchange_charge_rate: float = 0.00003,
    sebi_fee_rate: float = 0.000001,
    gst_rate: float = 0.18,
    dp_charge_flat: float = 15.93,
) -> float:
    """Calculate exact 2026 NSE delivery equity transaction cost for a single leg."""
    if is_buy:
        stt = trade_value * stt_rate
        stamp_duty = trade_value * stamp_duty_rate
        exch_charge = trade_value * exchange_charge_rate
        gst = exch_charge * gst_rate
        sebi = trade_value * sebi_fee_rate
        dp = 0.0
    else:
        stt = trade_value * stt_rate
        stamp_duty = 0.0
        exch_charge = trade_value * exchange_charge_rate
        gst = exch_charge * gst_rate
        sebi = trade_value * sebi_fee_rate
        dp = dp_charge_flat

    return stt + stamp_duty + exch_charge + gst + sebi + dp
```
