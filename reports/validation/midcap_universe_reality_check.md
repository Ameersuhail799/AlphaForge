# 🔬 Mid-Cap Universe Reality Check Audit

**Evaluation Period:** `2010-04-06` to `2026-08-14` (4042 Trading Days / ~16.0 Years)  
**Strategy Model:** Champion Dual-Agreement Random Forest (`P(Up) >= 0.55` & `Expected Return > +1.0%`)  
**Cost Model:** 2026 NSE Delivery Rates + **Explicit +0.20% per leg (+0.40% round-trip) Slippage Adjustment** (Stated Assumption for Mid-Cap Liquidity Friction)  

---

## 1. Objective Universe Selection Criteria

To eliminate survivorship and hindsight selection bias, all 8 mid-cap stocks were selected strictly prior to running backtests based on objective market-cap tier and data availability criteria:

1. **Market Cap Tier:** Selected from active Nifty Midcap / Nifty 200 constituents strictly outside the top ~30 largest NSE mega-caps.
2. **Liquidity Threshold:** Mid-cap tier with strong daily trading volumes, excluding micro/nano-caps with unfillable order books.
3. **Data History Length:** Minimum 10 years of continuous daily price history available via Yahoo Finance (all 8 stocks have 16–30 years of daily history).
4. **No Performance Screening:** Selection was made **purely on market-cap tier and data history**, without screening for past returns or chart patterns.

### 📋 Selected 8 NSE Mid-Cap Equities:
- `PERSISTENT` (Persistent Systems Ltd - IT Mid-Cap)
- `COFORGE` (Coforge Ltd - IT Mid-Cap)
- `VOLTAS` (Voltas Ltd - Consumer Durables / HVAC Mid-Cap)
- `FEDERALBNK` (Federal Bank Ltd - Private Banking Mid-Cap)
- `AUROPHARMA` (Aurobindo Pharma Ltd - Pharmaceuticals Mid-Cap)
- `APOLLOTYRE` (Apollo Tyres Ltd - Auto Ancillary Mid-Cap)
- `ASHOKLEY` (Ashok Leyland Ltd - Commercial Vehicles Mid-Cap)
- `BALKRISIND` (Balkrishna Industries Ltd - Tyres & Rubber Mid-Cap)

---

## 2. Summary Ranking & Comparison Table

| Stock Ticker | Strategy CAGR | Buy & Hold CAGR | CAGR Difference | Strategy Sharpe | B&H Sharpe | Max Drawdown | Win Rate % | Total Trades | Beat B&H? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PERSISTENT** | **+9.97%** | **+28.35%** | `-18.38%` | 0.56 | 0.92 | 40.78% | 53.55% | 155 | ❌ NO |
| **COFORGE** | **-4.44%** | **+27.99%** | `-32.43%` | -0.12 | 0.83 | 67.34% | 45.56% | 90 | ❌ NO |
| **VOLTAS** | **+5.45%** | **+13.28%** | `-7.83%` | 0.39 | 0.53 | 46.44% | 54.88% | 82 | ❌ NO |
| **FEDERALBNK** | **-3.99%** | **+17.07%** | `-21.06%` | -0.09 | 0.63 | 68.96% | 53.12% | 128 | ❌ NO |
| **AUROPHARMA** | **+0.07%** | **+19.19%** | `-19.12%` | 0.11 | 0.65 | 58.26% | 50.00% | 106 | ❌ NO |
| **APOLLOTYRE** | **+3.12%** | **+11.89%** | `-8.77%` | 0.25 | 0.49 | 72.95% | 52.10% | 167 | ❌ NO |
| **ASHOKLEY** | **-2.04%** | **+16.88%** | `-18.92%` | 0.04 | 0.60 | 79.35% | 57.02% | 121 | ❌ NO |
| **BALKRISIND** | **+4.48%** | **+25.42%** | `-20.94%` | 0.32 | 0.85 | 40.18% | 47.55% | 143 | ❌ NO |


---

## 3. Individual Stock Breakdowns

### 📊 PERSISTENT (PERSISTENT_NS)
* **Verdict:** `Beat Buy-and-Hold: NO (Strategy CAGR +9.97% vs B&H +28.35%, Diff -18.38%)`
* **Strategy CAGR:** `+9.97%` (Sharpe: `0.56`, Sortino: `0.54`, Max DD: `40.78%`)
* **Buy & Hold CAGR:** `+28.35%` (Sharpe: `0.92`, Sortino: `1.40`, Max DD: `50.66%`)
* **Trades Executed:** `155` (Win Rate: `53.55%`)

### 📊 COFORGE (COFORGE_NS)
* **Verdict:** `Beat Buy-and-Hold: NO (Strategy CAGR -4.44% vs B&H +27.99%, Diff -32.43%)`
* **Strategy CAGR:** `-4.44%` (Sharpe: `-0.12`, Sortino: `-0.08`, Max DD: `67.34%`)
* **Buy & Hold CAGR:** `+27.99%` (Sharpe: `0.83`, Sortino: `1.26`, Max DD: `56.99%`)
* **Trades Executed:** `90` (Win Rate: `45.56%`)

### 📊 VOLTAS (VOLTAS_NS)
* **Verdict:** `Beat Buy-and-Hold: NO (Strategy CAGR +5.45% vs B&H +13.28%, Diff -7.83%)`
* **Strategy CAGR:** `+5.45%` (Sharpe: `0.39`, Sortino: `0.29`, Max DD: `46.44%`)
* **Buy & Hold CAGR:** `+13.28%` (Sharpe: `0.53`, Sortino: `0.82`, Max DD: `75.26%`)
* **Trades Executed:** `82` (Win Rate: `54.88%`)

### 📊 FEDERALBNK (FEDERALBNK_NS)
* **Verdict:** `Beat Buy-and-Hold: NO (Strategy CAGR -3.99% vs B&H +17.07%, Diff -21.06%)`
* **Strategy CAGR:** `-3.99%` (Sharpe: `-0.09`, Sortino: `-0.07`, Max DD: `68.96%`)
* **Buy & Hold CAGR:** `+17.07%` (Sharpe: `0.63`, Sortino: `0.91`, Max DD: `70.46%`)
* **Trades Executed:** `128` (Win Rate: `53.12%`)

### 📊 AUROPHARMA (AUROPHARMA_NS)
* **Verdict:** `Beat Buy-and-Hold: NO (Strategy CAGR +0.07% vs B&H +19.19%, Diff -19.12%)`
* **Strategy CAGR:** `+0.07%` (Sharpe: `0.11`, Sortino: `0.08`, Max DD: `58.26%`)
* **Buy & Hold CAGR:** `+19.19%` (Sharpe: `0.65`, Sortino: `0.96`, Max DD: `69.85%`)
* **Trades Executed:** `106` (Win Rate: `50.00%`)

### 📊 APOLLOTYRE (APOLLOTYRE_NS)
* **Verdict:** `Beat Buy-and-Hold: NO (Strategy CAGR +3.12% vs B&H +11.89%, Diff -8.77%)`
* **Strategy CAGR:** `+3.12%` (Sharpe: `0.25`, Sortino: `0.23`, Max DD: `72.95%`)
* **Buy & Hold CAGR:** `+11.89%` (Sharpe: `0.49`, Sortino: `0.71`, Max DD: `74.52%`)
* **Trades Executed:** `167` (Win Rate: `52.10%`)

### 📊 ASHOKLEY (ASHOKLEY_NS)
* **Verdict:** `Beat Buy-and-Hold: NO (Strategy CAGR -2.04% vs B&H +16.88%, Diff -18.92%)`
* **Strategy CAGR:** `-2.04%` (Sharpe: `0.04`, Sortino: `0.03`, Max DD: `79.35%`)
* **Buy & Hold CAGR:** `+16.88%` (Sharpe: `0.60`, Sortino: `0.90`, Max DD: `79.19%`)
* **Trades Executed:** `121` (Win Rate: `57.02%`)

### 📊 BALKRISIND (BALKRISIND_NS)
* **Verdict:** `Beat Buy-and-Hold: NO (Strategy CAGR +4.48% vs B&H +25.42%, Diff -20.94%)`
* **Strategy CAGR:** `+4.48%` (Sharpe: `0.32`, Sortino: `0.31`, Max DD: `40.18%`)
* **Buy & Hold CAGR:** `+25.42%` (Sharpe: `0.85`, Sortino: `1.31`, Max DD: `51.35%`)
* **Trades Executed:** `143` (Win Rate: `47.55%`)

---

## 4. Honest Audit Conclusion

Out of the 8 mid-cap stocks evaluated, **0 out of 8 stocks (0.0%) beat Buy-and-Hold** after accounting for 2026 NSE delivery costs and explicit +0.20% per leg (+0.40% round-trip) mid-cap slippage.

### 💡 Synthesis & Findings:
Moving away from the 5 mega-cap large-cap universe to a mid-cap universe **reveals the exact same underlying pattern**. High-frequency momentum/technical signals on mid-cap equities face double friction: higher market-impact slippage and strong secular buy-and-hold compounding in quality mid-caps. The honest audit confirms that technical signal strategies must be evaluated against realistic transaction friction and buy-and-hold benchmarks before claiming true quantitative edge.
