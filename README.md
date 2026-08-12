# AlphaForge — AI Quantitative Trading Research & Paper-Trading Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-EMPIRICAL_AUDIT_COMPLETED-amber.svg)]()
[![Tests](https://img.shields.io/badge/tests-53%2F53_PASSED-success.svg)]()

**AlphaForge** is an institutional-grade, end-to-end AI quantitative trading research pipeline and paper-trading web application built for liquid National Stock Exchange (NSE) equities (`TCS.NS`, `INFY.NS`, `RELIANCE.NS`, `ICICIBANK.NS`, `HDFCBANK.NS`).

Rather than relying on unvalidated assumptions or overfitted backtest curves, AlphaForge formed a specific quantitative trading hypothesis, built a complete ML research architecture, and subjected the system to **four independent, disciplined empirical reality checks** under realistic 2026 NSE delivery transaction costs.

---

## ⚡ Core Architecture & Pipeline Components

1. **Feature Engineering Engine:** `C59_VOL_VOL` pipeline generating 28 unique technical, volatility, momentum, range compression, multi-horizon, and trend interaction features.
2. **Dual-Model ML Signal Generator:** `RandomForestClassifier` predicting directional probability ($P(\text{up})$) paired with a `RandomForestRegressor` predicting 10-day expected forward return ($\hat{r}_{10d}$).
3. **Deterministic Paper Trading Simulator:** Real-time web UI paper portfolio engine with zero leverage risk, 20% position weight caps, and 2026 NSE delivery cost accounting.
4. **Interactive Technical Dashboard:** Glassmorphic REST web application featuring Chart.js technical charting, real-time portfolio tracking, and structured signal evidence reasoning.
5. **Institutional Risk Controls:** 20% position weight caps, 35% sector cluster caps (IT Services & Financials), and portfolio drawdown governance.

---

## 🔬 Empirical Validation Results & Four Reality Checks

The active trading hypothesis ($P(\text{up}) \ge 0.55 \text{ AND } \hat{r}_{10d} > 1.0\%$ with a 10-day fixed hold horizon) was evaluated over a **23-year window (August 12, 2003 to August 6, 2026 — 5,695 trading days)** with realistic 2026 NSE delivery equity transaction friction (STT 0.10% buy/sell, Stamp Duty 0.015%, Exchange + SEBI fees, and flat ₹15.93 DP charge $\approx$ **0.302% / 30.2 bps round-trip**).

All four independent empirical reality checks came back **negative against the pre-set benchmark bars**:

### 1. Pooled Benchmark + Realistic Cost Audit
* **Finding:** Applying realistic NSE delivery transaction friction eroded active total return from +317.44% down to +217.97% (5.25% CAGR). Over the identical 23-year period across the 5 equities, an equal-weight Buy-and-Hold strategy yielded **+4,676.95% total return (18.66% CAGR)**, outperforming the active champion strategy by **>21x in total return**.
* **Evidence Report:** [`reports/validation/benchmark_and_cost_reality_check.md`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/reports/validation/benchmark_and_cost_reality_check.md)

### 2. Per-Stock Benchmark Breakdown
* **Finding:** Evaluated independently as single-asset 100% capital allocations, **0 of 5 equities beat Buy-and-Hold on CAGR** under realistic transaction costs:
  - `RELIANCE.NS`: Champion 8.34% CAGR vs. Buy-and-Hold 19.04% CAGR (**-10.70% gap**)
  - `TCS.NS`: Champion 8.67% CAGR vs. Buy-and-Hold 20.02% CAGR (**-11.35% gap**)
  - `HDFCBANK.NS`: Champion 6.33% CAGR vs. Buy-and-Hold 19.29% CAGR (**-12.96% gap**)
  - `INFY.NS`: Champion 1.26% CAGR vs. Buy-and-Hold 14.54% CAGR (**-13.28% gap**)
  - `ICICIBANK.NS`: Champion 4.90% CAGR vs. Buy-and-Hold 18.95% CAGR (**-14.05% gap**)
* **Evidence Report:** [`reports/validation/per_stock_reality_check.md`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/reports/validation/per_stock_reality_check.md)

### 3. Risk Overlay Reframing
* **Finding:** Reframing the signal as a macro tail-risk overlay (100% invested default, moving 100% to cash when BEARISH $P \le 0.45 \text{ OR } \hat{r}_{10d} < -0.5\%$) on RELIANCE and TCS failed to hit Buy-and-Hold-like CAGR (13.97% vs 19.10% on RELIANCE; 15.97% vs 20.01% on TCS) due to severe signal whipsawing (~32-33 transitions per year).
* **Evidence Report:** [`reports/validation/risk_overlay_reality_check.md`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/reports/validation/risk_overlay_reality_check.md)

### 4. Persistence-Filtered Overlay ($N=3$ Consecutive Days)
* **Finding:** Testing a fixed, pre-committed $N=3$ streak filter to eliminate whipsawing successfully removed >74% of trade churn (194 transitions on RELIANCE, 172 on TCS). On TCS, the filtered overlay landed within 1.63% of Buy-and-Hold CAGR (18.38% vs 20.01% CAGR) but failed to cut drawdown (65.74% vs 66.36% DD). On RELIANCE, it cut max drawdown by 30.16% (47.43% vs 77.59% DD) but missed the CAGR bar by 8.02% (11.08% vs 19.10% CAGR).
* **Evidence Report:** [`reports/validation/risk_overlay_persistence_test.md`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/reports/validation/risk_overlay_persistence_test.md)

---

## 🎯 True Accomplishment: Rigorous Science Over Curve Fitting

The primary achievement of the AlphaForge project is not a marketing claim of beating the market, but the construction of a **rigorous, reproducible quantitative research harness and demonstration web platform**. 

By maintaining uncompromised out-of-sample discipline, accounting for real-world transaction friction, and reporting null results transparently, AlphaForge demonstrates the fundamental quantitative principle: **empirical evidence must always take precedence over backtest optimism.**

---

## 🚀 Quick Start & How to Run

### 1. Run Complete Unit & Integration Test Suite
```bash
python -m tests.test_paper_trading_engine
python -m tests.test_mission26_meta_trade_quality
python -m tests.test_mission27_cross_asset_generalization
python -m tests.test_mission28_portfolio_risk_engine
python -m tests.test_mission29_adaptive_portfolio_overlay
python -m tests.test_production_system
```

### 2. Launch Paper-Trading Web Server Demo
```bash
python -m src.production.server --port 8080
```

### 3. Open Web Dashboard
Navigate to:
```
http://127.0.0.1:8080
```

---

## 📂 Project Structure

```
AlphaForge/
├── src/
│   ├── production/            # Production Application & Demo Subsystem
│   │   ├── trading_engine.py  # AI Signal & Technical Reasoning Engine
│   │   ├── paper_portfolio.py # Paper Trading Engine (20% cap & NSE costs)
│   │   ├── server.py          # REST Server (ThreadingHTTPServer)
│   │   └── static/            # Frontend Glassmorphic UI (HTML/CSS/JS)
│   ├── research/              # Quantitative Research & Reality Check Modules
│   ├── features/              # C59 Feature Engineering Pipeline
│   └── dataset/               # Data Ingestion & Scalers
├── tests/                     # 28 Unit and Integration Test Modules
├── reports/
│   ├── validation/            # 4 Evidence Reports (Reality Checks & Benchmarks)
│   └── research/              # Mission 11-29 Quantitative Research Logs
└── README.md
```

---

## 🔒 Disclaimer & Notice

* **Educational & Demonstration Purposes Only:** AlphaForge is a research demonstration project for paper trading and quantitative analysis. It is not financial advice or a live brokerage trading system.
