# Mission 15 — Target & Feature Representation Research Plan

---

## 1. Research Question

> **"Does replacing the 1-day directional target (`NEXT_DAY_DIRECTION`) with economically meaningful multi-horizon targets and a normalized multi-horizon feature representation unlock genuine predictive discrimination ($\text{ROC-AUC} \ge 0.550$, $\text{PR-AUC} \ge 0.550$, $\text{MCC} \ge 0.050$) across liquid NSE assets without temporal leakage?"**

---

## 2. Motivation from Missions 11–14

Missions 11 through 14 systematically established the boundaries of downstream post-processing:

* **Mission 11:** Identified that `SHORTLIST_16` eliminated severe model collapses found in `ALL_32`, but failed formal multi-asset generalization criteria.
* **Mission 12:** Unconstrained F1 threshold optimization generated a degenerate "always BUY" heuristic (predicting positive $>96\%$ of the time).
* **Mission 13:** Robust threshold constraints (Precision-Constrained F1, MCC, Youden's J) eliminated degenerate prediction bias, but failed all production criteria due to low raw model precision (~50.3%–51.6%).
* **Mission 14:** Post-hoc probability calibration (Platt Sigmoid / Isotonic Regression) reduced Brier Score and Log Loss, but failed to improve underlying rank discrimination ($\text{ROC-AUC} \approx 0.518$) and introduced rank inversion issues on noisy out-of-fold predictions.

**Key Scientific Conclusion:** Downstream threshold tuning and probability calibration cannot manufacture predictive alpha when the underlying target/feature representation lacks directional separation. Mission 15 pivots upstream to target formulation and normalized multi-horizon feature engineering.

---

## 3. Existing Target Diagnosis (Step 1A Empirical Findings)

An empirical inspection of `NEXT_DAY_DIRECTION = Close[t+1] > Close[t]` across 5 liquid NSE equity assets (`reliance_ns`, `tcs_ns`, `hdfcbank_ns`, `infy_ns`, `icicibank_ns`) revealed the fundamental flaw in the 1-day prediction target:

| Asset Name | Total Daily Rows | Positive Class Rate | Mean 1-Day Return | Std Dev 1-Day Return | Return $\le \pm 0.25\%$ | Return $\le \pm 0.50\%$ | Return $\le \pm 1.00\%$ |
|---|---|---|---|---|---|---|---|
| **`reliance_ns`** | 6,633 | 51.06% | +0.1244% | 4.718% | 14.55% | 27.09% | 48.44% |
| **`tcs_ns`** | 5,955 | 47.96% | +0.1101% | 3.530% | 19.61% | 33.37% | 54.79% |
| **`hdfcbank_ns`** | 6,637 | 50.26% | +0.0851% | 1.897% | 17.30% | 32.70% | 55.58% |
| **`infy_ns`** | 6,637 | 50.46% | +0.0594% | 2.245% | 14.75% | 27.47% | 48.62% |
| **`icicibank_ns`** | 5,984 | 50.32% | +0.0953% | 2.345% | 13.70% | 25.62% | 45.27% |

### Key Diagnostic Takeaways:
1. **Extreme Noise Ratio:** Across all assets, **25.6% to 33.4%** of daily price movements lie within $\pm 0.50\%$, and **45.3% to 55.6%** lie within $\pm 1.00\%$. Over half of all binary 1-day labels represent noise smaller than transaction costs and execution slippage.
2. **Artificial Class Symmetry:** Class balance sits almost perfectly at **$50.3\% - 51.1\%$** ($47.96\%$ for TCS). 1-day daily direction behaves as a near-random coin flip, explaining why model probabilities cluster near 0.500.

---

## 4. Candidate Target Definitions (Step 1B)

| Target Identifier | Forecast Horizon ($h$) | Target Definition / Formula | Class Type | Neutral Band / Threshold | Expected Trading Suitability |
|---|---|---|---|---|---|
| **`TARGET_A`** *(Baseline)* | 1 Day | $y_t = \mathbb{I}(Close_{t+1} > Close_t)$ | Binary (0/1) | None | **Low** (Noisy, transaction cost sensitive) |
| **`TARGET_B`** | 3 Days | $y_t = \mathbb{I}(Close_{t+3} > Close_t)$ | Binary (0/1) | None | **Moderate** (Captures short swing) |
| **`TARGET_C`** | 5 Days | $y_t = \mathbb{I}(Close_{t+5} > Close_t)$ | Binary (0/1) | None | **High** (Weekly momentum swing) |
| **`TARGET_D`** | 10 Days | $y_t = \mathbb{I}(Close_{t+10} > Close_t)$ | Binary (0/1) | None | **High** (2-week trend direction) |
| **`TARGET_E`** | 5 Days | $y_t = \frac{Close_{t+5} - Close_t}{Close_t} / \frac{\text{ATR}_{14, t}}{Close_t}$ | Continuous / Binary | $\pm 0.5 \times \sigma_{\text{norm}}$ | **Very High** (Volatility-normalized magnitude) |
| **`TARGET_F`** *(Neutral Band)* | 5 Days | $y_t = \begin{cases} +1, & r_{t \to t+5} > +0.75 \times \text{ATR}_{14} / P_t \\ -1, & r_{t \to t+5} < -0.75 \times \text{ATR}_{14} / P_t \\ 0, & \text{Otherwise (Neutral)} \end{cases}$ | Tri-class (+1, 0, -1) | $\pm 0.75 \times \text{ATR}_{14} / P_t$ | **Very High** (Eliminates noise; trades only strong moves) |

---

## 5. Candidate Feature Families & Inventory Gap Analysis (Step 1C)

### Existing Feature Inventory (32 Features):
* **Price & Candlestick (6):** `GAP_PCT`, `OPEN_CLOSE_PCT`, `HIGH_LOW_PCT`, `BODY_SIZE`, `UPPER_WICK`, `LOWER_WICK`
* **Momentum (6):** `DAILY_RETURN`, `PRICE_CHANGE_PCT`, `MOMENTUM_10`, `MOMENTUM_20`, `ROC_12`, `RSI_14`
* **Volatility (5):** `TRUE_RANGE`, `ATR_14`, `HIST_VOL_20`, `ROLLING_STD_20`, `DAILY_RANGE_PCT`
* **Volume (5):** `VOLUME_SMA_20`, `VOLUME_EMA_20`, `VOLUME_RATIO`, `VOLUME_CHANGE_PCT`, `OBV`
* **Trend (4):** `SMA_20`, `SMA_50`, `EMA_20`, `EMA_50`

### Proposed Multi-Horizon Normalized Feature Candidates:
To support multi-day target forecasting, the following normalized, stationary feature families will be evaluated:

1. **Price Normalization Ratios:**
   * `CLOSE_TO_SMA20` $= Close_t / \text{SMA}_{20, t} - 1$
   * `CLOSE_TO_SMA50` $= Close_t / \text{SMA}_{50, t} - 1$
   * `CLOSE_TO_SMA200` $= Close_t / \text{SMA}_{200, t} - 1$
2. **Multi-Horizon Return Ratios:**
   * `RETURN_3D` $= (Close_t - Close_{t-3}) / Close_{t-3}$
   * `RETURN_5D` $= (Close_t - Close_{t-5}) / Close_{t-5}$
   * `RETURN_10D` $= (Close_t - Close_{t-10}) / Close_{t-10}$
   * `RETURN_20D` $= (Close_t - Close_{t-20}) / Close_{t-20}$
3. **Multi-Horizon Volatility & Ratio Normalization:**
   * `VOLATILITY_5D` $= \text{Std}(\text{LogReturn})_{5d} \times \sqrt{252}$
   * `VOLATILITY_60D` $= \text{Std}(\text{LogReturn})_{60d} \times \sqrt{252}$
   * `ATR_TO_PRICE` $= \text{ATR}_{14, t} / Close_t$
   * `RANGE_TO_ATR` $= (High_t - Low_t) / \text{ATR}_{14, t}$
4. **Trend Slope & Spread Ratios:**
   * `SMA20_SMA50_SPREAD` $= (\text{SMA}_{20} - \text{SMA}_{50}) / \text{SMA}_{50}$
   * `SMA50_SMA200_SPREAD` $= (\text{SMA}_{50} - \text{SMA}_{200}) / \text{SMA}_{200}$
   * `SMA20_SLOPE_5D` $= (\text{SMA}_{20, t} - \text{SMA}_{20, t-5}) / \text{SMA}_{20, t-5}$
5. **Volume Multi-Horizon Ratios:**
   * `VOLUME_RATIO_5` $= Volume_t / \text{SMA}(Volume, 5)_t$
   * `VOLUME_CHANGE_5D` $= (Volume_t - Volume_{t-5}) / Volume_{t-5}$
6. **Price Extremes & Range Position:**
   * `DISTANCE_TO_20D_HIGH` $= (Close_t - \text{Max}(High)_{20d}) / \text{Max}(High)_{20d}$
   * `DISTANCE_TO_20D_LOW` $= (Close_t - \text{Min}(Low)_{20d}) / \text{Min}(Low)_{20d}$
   * `DISTANCE_TO_52W_HIGH` $= (Close_t - \text{Max}(High)_{252d}) / \text{Max}(High)_{252d}$
   * `DISTANCE_TO_52W_LOW` $= (Close_t - \text{Min}(Low)_{252d}) / \text{Min}(Low)_{252d}$

---

## 6. Leakage Audit & Safety Protocol

1. **Strict Lookahead Isolation:** All candidate features at row $t$ use OHLCV data from index $\le t$ only.
2. **Target Isolation:** Forward return targets ($Close_{t+h}$) are used **EXCLUSIVELY** as target labels $y_t$ and never enter feature matrix $X_t$.
3. **Scaling Isolation:** `FeatureScaler` is fit exclusively on training data partitions (`outer_X_train`).
4. **Holdout Protection:** The final 15% out-of-sample holdout test partition is strictly excluded and untouched.

---

## 7. Controlled Staged Experiment Matrix

```
+-----------------------------------------------------------------------------------+
| Stage 1: Baseline Audit                                                           |
| Target: TARGET_A (1-Day Direction) | Features: SHORTLIST_16                        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Stage 2: Target Horizon Screening                                                 |
| Evaluate TARGET_B (3D), TARGET_C (5D), TARGET_D (10D), TARGET_E (5D Norm),        |
| and TARGET_F (5D Neutral Band) holding feature set constant at SHORTLIST_16.      |
| Goal: Identify target horizon with strongest ROC-AUC & PR-AUC separation.         |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Stage 3: Feature Representation Enhancement                                       |
| Combine winning Stage 2 target(s) with proposed multi-horizon normalized          |
| features. Evaluate ROC-AUC gain, PR-AUC, and MCC stability across 75 folds.       |
+-----------------------------------------------------------------------------------+
```

---

## 8. Mission 15 Acceptance Criteria

A target/feature combination in Stage 3 is declared **CERTIFIED FOR PRODUCTION DEVELOPMENT** if it satisfies **ALL 4** criteria across 75 outer fold evaluations:

| Criterion | Target Requirement | Scientific Rationale |
|---|---|---|
| **1. Discrimination Edge** | Mean $\text{ROC-AUC} \ge \mathbf{0.5500}$ | Proves model achieves genuine ranking separation above random chance (0.500). |
| **2. Precision-Recall Area** | Mean $\text{PR-AUC} \ge \mathbf{0.5500}$ | Validates strong precision across all classification probability thresholds. |
| **3. Correlation Edge** | Mean $\text{MCC} \ge \mathbf{+0.0500}$ | Confirms positive correlation between predictions and realized trading outcome. |
| **4. Non-Degenerate PPR** | Mean PPR between **35% and 65%** (0 folds $>80\%$ or $<20\%$) | Prevents degenerate positive/negative prediction heuristics. |

---

## 9. Risks & Mitigations

* **Risk 1: Autocorrelation in Multi-Day Horizons.**  
  * *Mitigation:* Expanding-window fold gaps ($h$ days) will be enforced to prevent overlap between validation fold boundaries.
* **Risk 2: Multi-Class Neutral Band Imbalance.**  
  * *Mitigation:* `TARGET_F` will use symmetric $\pm 0.75 \times \text{ATR}_{14}$ threshold to maintain ~33%/33%/33% or ~25%/50%/25% balanced class distributions.

---

## 10. Expected Research Artifacts

1. `reports/research/mission15_target_diagnostics.csv` *(Generated in Step 1A)*
2. `docs/MISSION_15_TARGET_REPRESENTATION_RESEARCH.md` *(This design document)*
3. `reports/research/MISSION_15_TARGET_EXPERIMENT_REPORT.md`
4. `reports/research/mission15_target_summary.csv`
5. `reports/research/mission15_target_by_asset.csv`

---

## 11. Reproduction Commands
```bash
# Diagnostic inspection
python -c "import pandas as pd; print(pd.read_csv('reports/research/mission15_target_diagnostics.csv'))"

# Code compilation check
python -m compileall src/research tests
```
