# 🚀 AlphaForge — AI Quantitative Trading Platform & Research Engine

[![Live Web Application](https://img.shields.io/badge/Render-Live_Production_ App-10B981?style=for-the-badge&logo=render&logoColor=white)](https://alphaforge-92d1.onrender.com)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Test Suite](https://img.shields.io/badge/Tests-100%25_Passing_(23%2F23)-34D399?style=for-the-badge&logo=pytest&logoColor=white)]()
[![Status](https://img.shields.io/badge/Status-Empirical_Audit_Completed-F59E0B?style=for-the-badge)]()

> **Live Production URL**: [https://alphaforge-92d1.onrender.com](https://alphaforge-92d1.onrender.com)

**AlphaForge** is an institutional-grade, end-to-end AI quantitative trading research framework and interactive web platform built for liquid National Stock Exchange (NSE) equities (`TCS.NS`, `INFY.NS`, `RELIANCE.NS`, `ICICIBANK.NS`, `HDFCBANK.NS`).

Rather than relying on unvalidated backtest assumptions or overfitted curves, AlphaForge combines automated machine learning feature engineering, dual-model prediction engines, a deterministic paper-trading simulator, and **rigorous empirical reality checks** under realistic 2026 NSE delivery transaction friction.

---

## 🌐 Live Production Dashboard

Access the live cloud deployment hosted on Render:

🔗 **[https://alphaforge-92d1.onrender.com](https://alphaforge-92d1.onrender.com)**

*(Note: Render free-tier instances may take 30–40 seconds to spin up on initial cold start while model weights and historical market datasets load into memory.)*

---

## ✨ Key Platform Features

### 1. 🤖 AI Trading Intelligence Engine
* **Dual ML Architecture**: Combines a `RandomForestClassifier` predicting directional probability ($P(\text{up}) \ge 0.55$) with a `RandomForestRegressor` predicting 10-day expected forward return ($\hat{r}_{10d} > 1.0\%$).
* **Multi-Horizon Feature Engineering**: Custom `C59` pipeline generating 28+ trend, momentum, volatility expansion, and range compression signals.
* **Structured Technical Reasoning**: Automatically synthesizes model probabilities and technical indicators into human-readable trade conviction explanations.

### 2. 📈 Interactive Technical & Signal Chart
* **High-Performance Renderer**: Built with TradingView LightweightCharts (with Chart.js fallback support).
* **Multi-Timeframe Controls**: Toggle across 1D, 1W, 1M, 3M, 6M, 1Y, and ALL historical periods.
* **Indicator Overlays**: Live SMA 20 / SMA 50 trend overlays and volume histograms.

### 3. 💼 Deterministic Paper Portfolio Engine
* **Realistic Execution**: Simulates live orders with strict capital controls.
* **20% Allocation Cap**: Automatically enforces a max 20% equity weight per asset to maintain portfolio diversification.
* **2026 NSE Friction Accounting**: Deducts exact round-trip transaction costs (~0.302% / 30.2 bps), including STT (0.10% buy/sell), Stamp Duty (0.015%), Exchange/SEBI fees, and flat DP charges.

### 4. 📜 Historical Context & Regime Matching
* **Descriptive History**: Identifies historical market regimes matching current RSI, trend alignment, and volatility conditions across 20+ years of daily data.
* **Forward Distribution Grid**: Computes non-parametric 10-day forward return metrics (Min, 25th %ile, Median, 75th %ile, Max, and % Positive).

---

## 🔬 Empirical Validation & The Four Reality Checks

AlphaForge evaluates active trading strategies over a **23-year historical window (2003–2026 across 5,695 trading days)**. To maintain institutional research discipline, all model iterations were subjected to **four independent reality checks**:

| Reality Check | Objective | Result / Finding |
| :--- | :--- | :--- |
| **1. Pooled Benchmark + Cost Audit** | Test active signal against Buy-and-Hold with 2026 NSE costs. | Transaction friction eroded return to +217.97% (5.25% CAGR) vs Buy-and-Hold +4,676.95% (18.66% CAGR). |
| **2. Per-Stock Breakdown** | Evaluate single-asset 100% capital allocations. | **0 of 5 equities beat Buy-and-Hold on CAGR** under realistic transaction costs. |
| **3. Tail-Risk Overlay** | Test signal as a macro risk-off cash filter. | Severe signal whipsawing (~32 transitions/yr) reduced CAGR vs Buy-and-Hold. |
| **4. Persistence-Filtered Overlay** | Apply $N=3$ consecutive day streak filter. | Eliminated >74% of trade churn; improved drawdown protection on RELIANCE but fell short of Buy-and-Hold CAGR bar. |

> **Scientific Philosophy**: AlphaForge prioritizes **empirical scientific truth over curve fitting**. Reporting null results transparently demonstrates that quantitative rigor must always override backtest optimism.

---

## 🛠️ Technology Stack & Dependencies

* **Core Language**: Python 3.10+
* **Machine Learning**: `scikit-learn`, `joblib`, `xgboost`, `scipy`
* **Data Ingestion & Storage**: `pandas`, `numpy`, `pyarrow`, `yfinance`, `jugaad-data`
* **Backend Server**: Native `http.server` (`ThreadingHTTPServer`)
* **Frontend UI**: Vanilla JavaScript (ES6+), Glassmorphic CSS, TradingView LightweightCharts, Chart.js

---

## 💻 Local Setup & Installation

### 1. Clone Repository
```bash
git clone https://github.com/Ameersuhail799/AlphaForge.git
cd AlphaForge
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Production dependencies (for running web server):
pip install -r requirements.txt

# Development/Research dependencies (optional for local research scripts):
pip install -r requirements-dev.txt
```

### 4. Run Test Suite
```bash
python -m unittest tests.test_historical_context tests.test_strategy_tester tests.test_production_system tests.test_paper_trading_engine
```

### 5. Launch Local Web Server
```bash
python -m src.production.server --port 8080
```
Open your browser and navigate to `http://127.0.0.1:8080`.

---

## 🚀 Deployment (Render Free Tier)

This repository includes a pre-configured `render.yaml` for 1-click Render Web Service deployment:

1. Connect `Ameersuhail799/AlphaForge` on [Render Dashboard](https://dashboard.render.com).
2. Render automatically detects `render.yaml`:
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `python -m src.production.server --host 0.0.0.0 --port $PORT`

---

## 📂 Repository Layout

```text
AlphaForge/
├── src/
│   ├── production/            # Production Server & Live Web UI
│   │   ├── server.py          # Multithreaded REST API Server
│   │   ├── trading_engine.py  # Production AI Inference Engine
│   │   ├── paper_portfolio.py # Paper Portfolio & Capital Controls
│   │   └── static/            # Web UI (HTML5 / Glassmorphism CSS / JS)
│   ├── research/              # Quantitative Research & Strategy Testers
│   ├── features/              # C59 Feature Pipelines & Indicators
│   ├── data/                  # Providers (yfinance), Storage & Validators
│   └── dataset/               # Feature Scalers & Partitioning
├── data/raw/                  # Historical Parquet Market Datasets
├── tests/                     # 23 Automated Unit & Integration Tests
├── reports/validation/        # Evidence Reports & Reality Check Audits
├── render.yaml                # Render Web Service Infrastructure Specs
├── requirements.txt           # Scoped Production Dependencies
└── requirements-dev.txt       # Local Development & Research Dependencies
```

---

## 🔒 Disclaimer & Notice

* **Educational & Demonstration Purposes Only**: AlphaForge is a quantitative research platform designed for educational analysis and paper trading. It is not financial advice, investment recommendation, or a live brokerage execution system.
