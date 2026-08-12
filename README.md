# AlphaForge — AI Trading Intelligence & Paper-Trading Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-PRODUCTION_READY-emerald.svg)]()
[![Tests](https://img.shields.io/badge/tests-53%2F53_PASSED-success.svg)]()

**AlphaForge** is an institutional-grade, AI-powered quantitative trading intelligence and paper-trading web application platform built on 29 empirical research missions across liquid NSE equities (`TCS.NS`, `INFY.NS`, `RELIANCE.NS`, `ICICIBANK.NS`, `HDFCBANK.NS`).

---

## ⚡ Key Product Features

1. **AI Trading Intelligence Signal Engine:** Dual-model Random Forest system predicting direction probability ($P(\text{up})$) and expected 10-day forward returns ($\hat{r}_{10d}$) with structured technical evidence explanations.
2. **Interactive Technical Charting:** Dynamic Chart.js candlestick and line charts displaying price series, 20/50 SMAs, ATR, RSI, and signal entry points.
3. **Deterministic Paper Trading Simulator:** One-click BUY/SELL execution with transaction cost accounting (10 bps round-trip) and zero leverage risk.
4. **Real-time Portfolio Tracker:** Live monitoring of portfolio equity (₹1,00,000 initial capital), cash balance, open positions, unrealized/realized P&L, win rate, and drawdown.
5. **Cross-Asset Research Matrix:** Out-of-fold multi-asset validation summaries across 5 walk-forward folds.
6. **Institutional Risk Controls:** Position caps (20% max), sector cluster caps (35% max IT & Financials), and portfolio drawdown governance.

---

## 🚀 Quick Start & How to Run

### 1. Run All Unit & Integration Tests
```bash
python -m tests.test_paper_trading_engine
python -m tests.test_mission26_meta_trade_quality
python -m tests.test_mission27_cross_asset_generalization
python -m tests.test_mission28_portfolio_risk_engine
python -m tests.test_mission29_adaptive_portfolio_overlay
python -m tests.test_production_system
python -m compileall src tests
```

### 2. Launch Web Application Server
```bash
python -m src.production.server --port 8080
```

### 3. Access Web Dashboard
Open your browser and navigate to:
```
http://127.0.0.1:8080
```

---

## 📂 Project Structure

```
AlphaForge/
├── src/
│   ├── production/            # Production Application System
│   │   ├── trading_engine.py  # AI Signal & Technical Reasoning Engine
│   │   ├── paper_portfolio.py # Paper Trading & Accounting Tracker
│   │   ├── server.py          # REST API & Web Application Server
│   │   └── static/            # Glassmorphic Frontend (HTML/CSS/JS)
│   ├── research/              # Quantitative Research Modules (Missions 1-29)
│   ├── features/              # C59 Feature Pipeline
│   ├── models/                # ML Model Interfaces & Scalers
│   └── dataset/               # Data Handling & Scalers
├── tests/                     # 13 Comprehensive Unit & Integration Test Suites
├── reports/                   # Empirical Research & Final Product Reports
└── README.md
```

---

## 📊 Validated Champion Strategy

* **Universe:** `TCS.NS`, `INFY.NS`, `RELIANCE.NS`, `ICICIBANK.NS`, `HDFCBANK.NS`
* **Features:** `C59_VOL_VOL` (28 unique technical, volatility, momentum, range compression, and trend features)
* **Classifier:** `RandomForestClassifier` $\to P(\text{up})$
* **Regressor:** `RandomForestRegressor` $\to \hat{r}_{10d}$
* **Signal Condition:** $P(\text{up}) \ge 0.55 \text{ AND } \hat{r}_{10d} > 1.0\%$
* **Holding Horizon:** 10 trading days
* **Transaction Cost:** 10 bps round-trip ($0.10\%$)
* **Performance:** Multi-asset portfolio cumulative return **+350.10%**, Daily Sharpe **0.85** across 1,003 trades. 100% positive cumulative return across all 5 liquid assets.

---

## 🔒 Safety & Disclaimer

* **Paper Trading Only:** AlphaForge is designed strictly for paper trading, simulation, and educational research. No real-money broker APIs or live order execution protocols are connected.
* **Out-of-Sample Protection:** The final 15% out-of-sample holdout test partition remains 100% untouched.
