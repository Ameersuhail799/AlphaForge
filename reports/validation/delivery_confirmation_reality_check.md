# 🔬 Delivery % Confirmation Filter Reality Check Audit

**Evaluation Window:** `2014-01-02` to `2026-08-13` (2370 Common Trading Days / ~9.4 Years)  
**Universe:** 5 Large-Cap Equities (`TCS`, `INFY`, `RELIANCE`, `ICICIBANK`, `HDFCBANK`)  
**Cost Model:** 2026 NSE Delivery Transaction Rates (STT 0.1%, Stamp Duty 0.015%, Exch/SEBI, DP Fee)  
**Rule Specification (Fixed, Zero Tuning):** Take Champion BUY Signal **ONLY WHEN** `Delivery % > Trailing 60-Trading-Day Median Delivery %`.  
**Deployment Status:** ⚠️ **RESEARCH-ONLY BACKTEST ONLY** — Confirmed **NO-GO for automated live ingestion** due to scraping fragility and rate-limiting.

---

## 1. Summary Comparison Table (Rebased 2014–2026 Window)

| Asset / Strategy | Unfiltered CAGR | Filtered CAGR | Buy & Hold CAGR | Unfiltered Sharpe | Filtered Sharpe | B&H Sharpe | Signals Rejected | Signals Passed | Unfiltered Max DD | Filtered Max DD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TCS** | **+9.85%** | **+11.99%** | **+8.69%** | 0.68 | 0.91 | 0.44 | `87` | `57` | 23.00% | 23.00% |
| **INFY** | **+5.30%** | **-0.36%** | **+11.13%** | 0.37 | 0.06 | 0.50 | `79` | `59` | 30.85% | 33.19% |
| **RELIANCE** | **+7.12%** | **+3.45%** | **+22.17%** | 0.54 | 0.34 | 0.81 | `40` | `40` | 33.30% | 23.95% |
| **ICICIBANK** | **+8.78%** | **+6.56%** | **+23.33%** | 0.51 | 0.46 | 0.79 | `69` | `42` | 38.26% | 40.57% |
| **HDFCBANK** | **+3.44%** | **+5.23%** | **+17.09%** | 0.29 | 0.46 | 0.74 | `55` | `57` | 41.51% | 24.39% |
| **ALL 5 POOLED** | **+6.72%** | **+5.77%** | **+17.65%** | 0.74 | 0.77 | 0.94 | `330` | `255` | 22.68% | 17.61% |


---

## 2. Individual Stock Breakdowns

### 📊 TCS (TCS_NS)
* **Unfiltered Champion Strategy:** CAGR `+9.85%` | Sharpe `0.68` | Sortino `0.54` | Max DD `23.00%` | Trades `76` (Win Rate `57.89%`)
* **Filtered (+Delivery Confirmation):** CAGR `+11.99%` | Sharpe `0.91` | Sortino `0.63` | Max DD `23.00%` | Trades `57` (Win Rate `64.91%`)
* **Filter Impact:** Filter rejected `87` raw signal days (reducing trade count from `76` to `57`).
* **Buy & Hold Benchmark:** CAGR `+8.69%` | Sharpe `0.44` | Sortino `0.64` | Max DD `56.44%`

### 📊 INFY (INFY_NS)
* **Unfiltered Champion Strategy:** CAGR `+5.30%` | Sharpe `0.37` | Sortino `0.30` | Max DD `30.85%` | Trades `83` (Win Rate `65.06%`)
* **Filtered (+Delivery Confirmation):** CAGR `-0.36%` | Sharpe `0.06` | Sortino `0.04` | Max DD `33.19%` | Trades `59` (Win Rate `52.54%`)
* **Filter Impact:** Filter rejected `79` raw signal days (reducing trade count from `83` to `59`).
* **Buy & Hold Benchmark:** CAGR `+11.13%` | Sharpe `0.50` | Sortino `0.67` | Max DD `50.41%`

### 📊 RELIANCE (RELIANCE_NS)
* **Unfiltered Champion Strategy:** CAGR `+7.12%` | Sharpe `0.54` | Sortino `0.42` | Max DD `33.30%` | Trades `54` (Win Rate `51.85%`)
* **Filtered (+Delivery Confirmation):** CAGR `+3.45%` | Sharpe `0.34` | Sortino `0.22` | Max DD `23.95%` | Trades `40` (Win Rate `45.00%`)
* **Filter Impact:** Filter rejected `40` raw signal days (reducing trade count from `54` to `40`).
* **Buy & Hold Benchmark:** CAGR `+22.17%` | Sharpe `0.81` | Sortino `1.24` | Max DD `45.09%`

### 📊 ICICIBANK (ICICIBANK_NS)
* **Unfiltered Champion Strategy:** CAGR `+8.78%` | Sharpe `0.51` | Sortino `0.41` | Max DD `38.26%` | Trades `63` (Win Rate `53.97%`)
* **Filtered (+Delivery Confirmation):** CAGR `+6.56%` | Sharpe `0.46` | Sortino `0.32` | Max DD `40.57%` | Trades `42` (Win Rate `57.14%`)
* **Filter Impact:** Filter rejected `69` raw signal days (reducing trade count from `63` to `42`).
* **Buy & Hold Benchmark:** CAGR `+23.33%` | Sharpe `0.79` | Sortino `1.15` | Max DD `52.35%`

### 📊 HDFCBANK (HDFCBANK_NS)
* **Unfiltered Champion Strategy:** CAGR `+3.44%` | Sharpe `0.29` | Sortino `0.22` | Max DD `41.51%` | Trades `69` (Win Rate `55.07%`)
* **Filtered (+Delivery Confirmation):** CAGR `+5.23%` | Sharpe `0.46` | Sortino `0.34` | Max DD `24.39%` | Trades `57` (Win Rate `57.89%`)
* **Filter Impact:** Filter rejected `55` raw signal days (reducing trade count from `69` to `57`).
* **Buy & Hold Benchmark:** CAGR `+17.09%` | Sharpe `0.74` | Sortino `1.04` | Max DD `41.05%`

---

## 3. Pooled Multi-Asset Equal-Weight Portfolio Performance

* **Unfiltered Champion Strategy:** CAGR `+6.72%` | Sharpe `0.74` | Sortino `0.72` | Max DD `22.68%` | Total Trades `345`
* **Filtered (+Delivery Confirmation):** CAGR `+5.77%` | Sharpe `0.77` | Sortino `0.70` | Max DD `17.61%` | Total Trades `255`
* **Buy & Hold Benchmark:** CAGR `+17.65%` | Sharpe `0.94` | Sortino `1.27` | Max DD `39.21%`

---

## 4. Honest Audit Conclusion

Did the delivery % confirmation filter meaningfully close the gap to Buy-and-Hold versus the unfiltered champion strategy on the same 2014–2026 window?

**NO.** Adding the trailing 60-day median delivery % confirmation filter **did NOT meaningfully close the performance gap to Buy-and-Hold**. Across the pooled 5-stock portfolio over the 2014–2026 rebased window:
- **Buy-and-Hold CAGR:** **`+17.65%`** (Sharpe: `0.94`)
- **Unfiltered Champion CAGR:** **`+6.72%`** (Sharpe: `0.74`)
- **Delivery-Filtered Champion CAGR:** **`+5.77%`** (Sharpe: `0.77`)

While the delivery filter rejected **`330` signals** (95.7% of total raw signals), filtering out trades did not turn negative/low CAGR signals into buy-and-hold beating compounding. Buy-and-Hold outperformed both strategy variants by over **10–12% per year in CAGR** across the 12.5-year window. Furthermore, given the confirmed NO-GO on automated delivery data scraping due to constant API rate-limiting and session breakage, this confirms that delivery volume filters provide no viable path for live production deployment.
