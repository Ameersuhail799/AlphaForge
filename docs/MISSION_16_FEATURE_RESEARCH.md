# Mission 16 — Multi-Horizon Trading Feature Research Plan

---

## 1. Research Objective

> **"Can a stationary, multi-horizon normalized feature representation—engineered to capture trend slopes, price relative distance, multi-period momentum, volatility regimes, breakout locations, and volume confirmation—enhance the verified TCS directional signal without introducing temporal lookahead leakage or feature redundancy?"**

---

## 2. Verified Control Baselines

All candidate feature sets evaluated in Mission 16 will be benchmarked directly against the untouched 15% out-of-sample holdout test partition and verified Mission 15 control baselines:

1. **Primary Control Baseline:**
   * Asset: `tcs_ns` | Target: `TARGET_D` (10-Day Direction) | Model: `random_forest` | Features: `SHORTLIST_16`
   * Metrics: Mean $\text{ROC-AUC} = \mathbf{0.5607}$, Mean $\text{PR-AUC} = \mathbf{0.6156}$, Mean $\text{MCC} = \mathbf{+0.0796}$, Mean Return Spread = $\mathbf{+1.15\%}$, Stability Score = $\mathbf{91.79/100}$.

2. **Secondary Control Baseline:**
   * Asset: `tcs_ns` | Target: `TARGET_B` (3-Day Direction) | Model: `xgboost` | Features: `SHORTLIST_16`
   * Metrics: Mean $\text{ROC-AUC} = \mathbf{0.5274}$, Mean $\text{PR-AUC} = \mathbf{0.5655}$, Mean $\text{MCC} = \mathbf{+0.0503}$, Mean Return Spread = $\mathbf{+0.39\%}$, Stability Score = $\mathbf{92.03/100}$.

---

## 3. Existing Feature Pipeline Audit vs Proposed Feature Families

### Existing Pipeline Inventory (32 Total / `SHORTLIST_16` Subset):
* **Price & Candlestick:** `GAP_PCT`, `OPEN_CLOSE_PCT`, `HIGH_LOW_PCT`, `BODY_SIZE`, `UPPER_WICK`, `LOWER_WICK`
* **Momentum:** `DAILY_RETURN`, `PRICE_CHANGE_PCT`, `MOMENTUM_10`, `MOMENTUM_20`, `ROC_12`, `RSI_14`
* **Volatility:** `TRUE_RANGE`, `ATR_14`, `HIST_VOL_20`, `ROLLING_STD_20`, `DAILY_RANGE_PCT`
* **Volume:** `VOLUME_SMA_20`, `VOLUME_EMA_20`, `VOLUME_RATIO`, `VOLUME_CHANGE_PCT`, `OBV`
* **Trend:** `SMA_20`, `SMA_50`, `EMA_20`, `EMA_50`

---

## 4. Proposed 31 Multi-Horizon Normalized Features (New Feature Pool)

| Group | Feature Name | Mathematical Formula | Economic / Trading Rationale | Expected Signal Role |
|---|---|---|---|---|
| **A. Price / Trend Position** | `CLOSE_TO_SMA20` | $Close_t / \text{SMA}_{20, t} - 1$ | Distance from short-term trend line | Mean Reversion / Trend Stretch |
| | `CLOSE_TO_SMA50` | $Close_t / \text{SMA}_{50, t} - 1$ | Distance from medium-term trend line | Medium Trend Position |
| | `CLOSE_TO_SMA200` | $Close_t / \text{SMA}_{200, t} - 1$ | Distance from long-term trend line | Macro Regime Filter |
| | `SMA20_SMA50_SPREAD` | $(\text{SMA}_{20, t} - \text{SMA}_{50, t}) / \text{SMA}_{50, t}$ | Moving average convergence / divergence | Trend Alignment |
| | `SMA50_SMA200_SPREAD` | $(\text{SMA}_{50, t} - \text{SMA}_{200, t}) / \text{SMA}_{200, t}$ | Golden / Death cross distance | Macro Trend Spread |
| **B. Multi-Horizon Returns** | `RETURN_3D` | $(Close_t - Close_{t-3}) / Close_{t-3}$ | 3-day percentage return | Short Swing Momentum |
| | `RETURN_5D` | $(Close_t - Close_{t-5}) / Close_{t-5}$ | 5-day percentage return | Weekly Momentum |
| | `RETURN_10D` | $(Close_t - Close_{t-10}) / Close_{t-10}$ | 10-day percentage return | Fortnightly Momentum |
| | `RETURN_20D` | $(Close_t - Close_{t-20}) / Close_{t-20}$ | 20-day percentage return | Monthly Trend Return |
| **C. Trend Slope** | `SMA20_SLOPE_5D` | $(\text{SMA}_{20, t} - \text{SMA}_{20, t-5}) / \text{SMA}_{20, t-5}$ | 5-day trajectory of 20-day SMA | Short Trend Velocity |
| | `SMA50_SLOPE_10D` | $(\text{SMA}_{50, t} - \text{SMA}_{50, t-10}) / \text{SMA}_{50, t-10}$ | 10-day trajectory of 50-day SMA | Medium Trend Velocity |
| | `EMA20_SLOPE_5D` | $(\text{EMA}_{20, t} - \text{EMA}_{20, t-5}) / \text{EMA}_{20, t-5}$ | 5-day trajectory of 20-day EMA | Fast Trend Velocity |
| **D. Volatility Regime** | `ATR_TO_PRICE` | $\text{ATR}_{14, t} / Close_t$ | Price-normalized average true range | Volatility Scale Normalization |
| | `VOLATILITY_5D` | $\text{Std}(\ln(C_t/C_{t-1}))_{5d} \times \sqrt{252}$ | Short-term realized volatility | Volatility Spike Detection |
| | `VOLATILITY_20D` | $\text{Std}(\ln(C_t/C_{t-1}))_{20d} \times \sqrt{252}$ | Medium-term annualized volatility | Monthly Volatility Baseline |
| | `VOLATILITY_60D` | $\text{Std}(\ln(C_t/C_{t-1}))_{60d} \times \sqrt{252}$ | Long-term annualized volatility | Quarterly Volatility Baseline |
| | `RANGE_TO_ATR` | $(High_t - Low_t) / \text{ATR}_{14, t}$ | Daily bar range relative to 14D ATR | Expansion / Compression Bar |
| | `VOLATILITY_RATIO_SHORT_LONG` | $\text{VOLATILITY}_{5D} / (\text{VOLATILITY}_{60D} + 1e-8)$ | Volatility ratio (short vs long term) | Volatility Regime Shift |
| **E. Breakout / Location** | `DISTANCE_TO_20D_HIGH` | $(Close_t - \text{Max}(H)_{20d}) / \text{Max}(H)_{20d}$ | Distance to 20-day high (always $\le 0$) | Resistance Proximity |
| | `DISTANCE_TO_20D_LOW` | $(Close_t - \text{Min}(L)_{20d}) / \text{Min}(L)_{20d}$ | Distance to 20-day low (always $\ge 0$) | Support Proximity |
| | `DISTANCE_TO_52W_HIGH` | $(Close_t - \text{Max}(H)_{252d}) / \text{Max}(H)_{252d}$ | Distance to 52-week high | Macro Resistance Proximity |
| | `DISTANCE_TO_52W_LOW` | $(Close_t - \text{Min}(L)_{252d}) / \text{Min}(L)_{252d}$ | Distance to 52-week low | Macro Support Proximity |
| | `POSITION_IN_20D_RANGE` | $(Close_t - \text{Min}(L)_{20d}) / (\text{Max}(H)_{20d} - \text{Min}(L)_{20d} + 1e-8)$ | Relative position in 20D Channel $[0, 1]$ | Stochastic Range Location |
| **F. Volume Confirmation** | `VOLUME_RATIO_5` | $Volume_t / \text{SMA}(Vol, 5)_t$ | Volume relative to 5-day SMA | Short Volume Spike |
| | `VOLUME_RATIO_20` | $Volume_t / \text{SMA}(Vol, 20)_t$ | Volume relative to 20-day SMA | Monthly Volume Spike |
| | `VOLUME_CHANGE_5D` | $(Volume_t - Volume_{t-5}) / Volume_{t-5}$ | 5-day volume growth rate | Volume Acceleration |
| | `VOLUME_TREND_RATIO` | $\text{SMA}(Vol, 5)_t / (\text{SMA}(Vol, 20)_t + 1e-8)$ | 5D vs 20D volume moving average | Institutional Volume Inflow |
| **G. Momentum / Mean Rev** | `RSI_NEUTRAL_DIFF` | $\text{RSI}_{14, t} - 50.0$ | RSI distance from neutral center line | Directional RSI Bias |
| | `MOMENTUM_ACCELERATION` | $\text{RETURN}_{5D, t} - \text{RETURN}_{5D, t-5}$ | 5-day return acceleration | Momentum Velocity Change |
| | `MOMENTUM_SPREAD_SHORT_LONG` | $\text{RETURN}_{5D, t} - \text{RETURN}_{20D, t}$ | Short vs long momentum spread | Swing vs Trend Divergence |
| | `PRICE_ZSCORE_20D` | $(Close_t - \text{SMA}_{20, t}) / (\text{ROLLING\_STD}_{20, t} + 1e-8)$ | Volatility-standardized price distance | Mean Reversion Z-Score |

---

## 5. Temporal Leakage Audit Checklist

Every proposed feature MUST strictly satisfy the following lookahead isolation properties:
1. **Historical Bar Indexing:** Feature at row $t$ dereferences ONLY OHLCV data from index $\le t$.
2. **Backward-Looking Windows:** Rolling window functions (`rolling(window).mean()`, `std()`, `max()`, `min()`) use backward-looking windows.
3. **No Target Contamination:** Target forward returns ($Close_{t+h}$) are built separately and NEVER enter feature columns.
4. **Scaler Partition Isolation:** `FeatureScaler` is fit exclusively on outer training data (`outer_X_train`).

---

## 6. Proposed Controlled Experiment Matrix for Mission 16

```
+-----------------------------------------------------------------------------------+
| Baseline Control                                                                 |
| SHORTLIST_16 (16 features)                                                        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Group Feature Additions (Ablation)                                                |
| Evaluate SHORTLIST_16 + Group A (Price/Trend)                                     |
| Evaluate SHORTLIST_16 + Group B (Multi-Horizon Returns)                           |
| Evaluate SHORTLIST_16 + Group C (Trend Slope)                                     |
| Evaluate SHORTLIST_16 + Group D (Volatility Regime)                               |
| Evaluate SHORTLIST_16 + Group E (Breakout/Channel Position)                       |
| Evaluate SHORTLIST_16 + Group F (Volume Confirmation)                             |
| Evaluate SHORTLIST_16 + Group G (Momentum & Z-Score)                              |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Final Combined Feature Pool Selection                                             |
| Evaluate top non-redundant feature combinations against verified control baselines |
+-----------------------------------------------------------------------------------+
```
