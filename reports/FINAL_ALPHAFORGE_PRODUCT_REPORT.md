# AlphaForge — AI Trading Intelligence & Paper-Trading Platform Final Product Report

---

## Executive Summary & Product Overview

**AlphaForge** has evolved from an empirical quantitative research framework into a **fully integrated, production-grade AI Trading Intelligence & Paper-Trading Web Application Platform**. 

The platform connects quantitative Machine Learning models (`RandomForestClassifier` direction model + `RandomForestRegressor` expected return model trained on `C59_VOL_VOL` multi-horizon features) directly to a modern, responsive web application interface, empowering systematic traders and investors to:

1. Analyze liquid NSE equities (`TCS.NS`, `INFY.NS`, `RELIANCE.NS`, `ICICIBANK.NS`, `HDFCBANK.NS`) for real-time **BUY / HOLD / SELL** trade decisions.
2. View **evidence-based technical and model reasoning** (Probability conviction $P(\text{up}) \ge 55.0\%$, Expected 10D Return edge $\hat{r}_{10d} > 1.0\%$, Range compression, Volume surge, Trend alignment).
3. Experience **interactive price charting** with overlay technical indicators (SMA 20, SMA 50, ATR, RSI).
4. Simulate **one-click paper trading** with zero leverage risk, accounting invariant tracking ($\text{Equity} = \text{Cash} + \text{Positions}$), transaction cost accounting (10 bps round-trip), and real-time realized/unrealized P&L monitoring.
5. Review **empirical multi-asset walk-forward backtest matrices** (Missions 26–29 research findings across 5 walk-forward folds).
6. Inspect **institutional risk controls & dynamic capital allocation limits**.

---

## 1. System & Architecture Overview

```
                      +---------------------------------------+
                      |         AlphaForge Web Dashboard      |
                      |   (HTML5 / CSS3 Glassmorphism / JS)   |
                      +---------------------------------------+
                                          |
                                          | REST API (HTTP JSON)
                                          v
                      +---------------------------------------+
                      |        Production REST Server         |
                      |        (src/production/server.py)     |
                      +---------------------------------------+
                                   /             \
                                  /               \
                                 v                 v
    +----------------------------------+     +----------------------------------+
    |    Production Trading Engine     |     |     Paper Portfolio Engine       |
    | (src/production/trading_engine)  |     | (src/production/paper_portfolio) |
    +----------------------------------+     +----------------------------------+
               |                                           |
               v                                           v
    +----------------------------------+     +----------------------------------+
    | C59 Feature Pipeline & Scaler    |     |  Accounting Invariant Tracker    |
    | RF Classifier + RF Regressor     |     | Cash, Open Positions, Trade Log  |
    +----------------------------------+     +----------------------------------+
```

---

## 2. ML & Signal Engine Specifications

* **Liquid Asset Universe:** `tcs_ns`, `infy_ns`, `reliance_ns`, `icicibank_ns`, `hdfcbank_ns`
* **Feature Configuration:** `C59_VOL_VOL` (28 unique features combining price action, momentum, multi-horizon volatility, range compression expansion, volume breakout confirm, and trend-volatility interactions).
* **Primary Direction Model:** `RandomForestClassifier(n_estimators=100, random_state=42)` $\to P(\text{up})$
* **Secondary Regressor Model:** `RandomForestRegressor(n_estimators=100, random_state=42)` $\to \hat{r}_{10d}$
* **Dual-Agreement Entry Rule:**
  $$\text{Signal} = \begin{cases} \mathbf{BUY} & \text{if } P(\text{up}) \ge 0.55 \text{ AND } \hat{r}_{10d} > 1.0\% \\ \mathbf{SELL} & \text{if } P(\text{up}) \le 0.45 \text{ OR } \hat{r}_{10d} < -0.5\% \\ \mathbf{HOLD} & \text{otherwise} \end{cases}$$
* **Holding Horizon:** 10 trading days.
* **Transaction Cost Model:** 10 bps round-trip ($0.10\%$ total per trade).

---

## 3. Empirical Research & Validation Matrix

The production system is backed by extensive out-of-fold cross-asset research and four independent empirical reality checks under realistic 2026 NSE delivery transaction friction (~0.30% round-trip):

| Asset | Display Name | Champion CAGR | Buy & Hold CAGR | CAGR Difference | Champion Sharpe | Buy & Hold Sharpe | Reality Check Verdict |
|---|---|---|---|---|---|---|---|
| **`reliance_ns`** | Reliance Industries | **8.34%** | **19.04%** | **-10.70%** | **0.50** | **0.43** | **NO (B&H Outperformed)** |
| **`tcs_ns`** | Tata Consultancy Services | **8.67%** | **20.02%** | **-11.35%** | **0.52** | **0.51** | **NO (B&H Outperformed)** |
| **`hdfcbank_ns`** | HDFC Bank | **6.33%** | **19.29%** | **-12.96%** | **0.40** | **0.76** | **NO (B&H Outperformed)** |
| **`infy_ns`** | Infosys Ltd | **1.26%** | **14.54%** | **-13.28%** | **0.16** | **0.60** | **NO (B&H Outperformed)** |
| **`icicibank_ns`** | ICICI Bank | **4.90%** | **18.95%** | **-14.05%** | **0.33** | **0.65** | **NO (B&H Outperformed)** |

* **Empirical Reality Check Summary:** 0 of 5 equities beat Buy-and-Hold on CAGR under realistic transaction costs over the 23-year evaluation window (2003–2026).
* **Reference Reports:** [`reports/validation/per_stock_reality_check.md`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/reports/validation/per_stock_reality_check.md), [`reports/validation/benchmark_and_cost_reality_check.md`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/reports/validation/benchmark_and_cost_reality_check.md), [`reports/validation/risk_overlay_reality_check.md`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/reports/validation/risk_overlay_reality_check.md), and [`reports/validation/risk_overlay_persistence_test.md`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/reports/validation/risk_overlay_persistence_test.md).

---

## 4. Verification & Test Suite Summary

All unit, integration, safety, and accounting invariant tests pass with 0 errors across all 13 repository test modules:

1. [`tests/test_paper_trading_engine.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/tests/test_paper_trading_engine.py): **6 PASSED**
2. [`tests/test_mission19_edge_validation.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/tests/test_mission19_edge_validation.py): **3 PASSED**
3. [`tests/test_mission20_forward_paper.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/tests/test_mission20_forward_paper.py): **5 PASSED**
4. [`tests/test_mission21_model_improvement.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/tests/test_mission21_model_improvement.py): **3 PASSED**
5. [`tests/test_mission22_technical_strategy.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/tests/test_mission22_technical_strategy.py): **4 PASSED**
6. [`tests/test_mission23_expected_return.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/tests/test_mission23_expected_return.py): **4 PASSED**
7. [`tests/test_mission24_adaptive_trade_engine.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/tests/test_mission24_adaptive_trade_engine.py): **4 PASSED**
8. [`tests/test_mission25_adaptive_exit.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/tests/test_mission25_adaptive_exit.py): **4 PASSED**
9. [`tests/test_mission26_meta_trade_quality.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/tests/test_mission26_meta_trade_quality.py): **4 PASSED**
10. [`tests/test_mission27_cross_asset_generalization.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/tests/test_mission27_cross_asset_generalization.py): **4 PASSED**
11. [`tests/test_mission28_portfolio_risk_engine.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/tests/test_mission28_portfolio_risk_engine.py): **3 PASSED**
12. [`tests/test_mission29_adaptive_portfolio_overlay.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/tests/test_mission29_adaptive_portfolio_overlay.py): **3 PASSED**
13. [`tests/test_production_system.py`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/tests/test_production_system.py): **6 PASSED**

**Total Test Count:** **53 / 53 Unit & Integration Tests PASSED (0 Errors)**

---

## 5. How to Run & Demonstrate the Platform

### Step 1: Start the Web Application Server
Run the production web server command from the project root:
```bash
python -m src.production.server --port 8080
```

### Step 2: Open Dashboard in Browser
Open your browser and navigate to:
```
http://127.0.0.1:8080
```

### Step 3: Interactive Demo Flow
1. **Asset Selection:** Click between `TCS.NS`, `INFY.NS`, `RELIANCE.NS`, `ICICIBANK.NS`, `HDFCBANK.NS` tabs to load real-time predictions and charts.
2. **Review Signal Card:** Observe the BUY/HOLD/SELL signal badge, probability meter ($P(\text{up})$), expected 10D return ($\hat{r}_{10d}$), and technical reasons.
3. **Execute Paper Trade:** Set capital allocation (default ₹20,000) and click **BUY POSITION**.
4. **Monitor Paper Portfolio:** Switch to the **Paper Portfolio** tab to view open positions, live unrealized P&L, closed trade history, and portfolio equity curve.
5. **Inspect Research Edge:** Switch to the **Validated Research Edge** tab to review multi-asset cross-validation performance.
