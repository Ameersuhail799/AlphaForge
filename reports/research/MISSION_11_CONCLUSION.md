# Mission 11 Research Conclusion: Multi-Asset Feature Robustness & Subset Generalization

---

## 1. Research Question
Does the 16-feature normalized subset (`SHORTLIST_16`) consistently generalize across diverse market assets, sector regimes, and time folds compared to the uncurated 32-feature baseline (`ALL_32`), without suffering model collapse or overfitting?

---

## 2. Dataset Coverage
* **Assets Evaluated (5 liquid NSE equities across 3 sectors):**
  - Energy: `reliance_ns` (6,635 daily rows, 2000–2026)
  - IT / Technology: `tcs_ns` (5,957 rows, 2002–2026), `infy_ns` (6,639 rows, 2000–2026)
  - Banking / Financials: `hdfcbank_ns` (6,639 rows, 2000–2026), `icicibank_ns` (5,986 rows, 2002–2026)
* **Total Ingested History:** 31,856 daily trading observations (>23 years per asset).

---

## 3. Experimental Protocol
* **Chronological Split:** 85% Non-Test partition (for cross-validation), 15% Out-of-Sample Holdout (strictly isolated).
* **Cross-Validation:** 5-fold expanding-window cross-validation inside the 85% non-test partition per asset.
* **Leak-Free Scaling:** [`FeatureScaler`](file:///c:/Users/AMEER%20SUHAIL/OneDrive/Desktop/projects/AlphaForge/src/dataset/scaler.py) fitted strictly on training data (`X_train`) inside each fold.
* **Models Evaluated:** `logistic_regression`, `random_forest`, `xgboost` with fixed random seeds.

---

## 4. SHORTLIST_16 vs ALL_32 Results

| Model | Paired Folds | Mean ROC-AUC (`SHORTLIST_16`) | Mean ROC-AUC (`ALL_32`) | Mean F1 (`SHORTLIST_16`) | Mean F1 (`ALL_32`) | F1 Win Rate | Model Collapses (`SHORTLIST_16`) | Model Collapses (`ALL_32`) |
|---|---|---|---|---|---|---|---|---|
| **Random Forest** | 15 | **0.5111** | 0.5076 | **0.4809** | 0.2386 | **93.3%** | **0** | **4** |
| **XGBoost** | 15 | **0.5154** | 0.5120 | **0.4753** | 0.3075 | **93.3%** | **0** | **3** |
| **Logistic Regression** | 15 | 0.5164 | **0.5179** | **0.4926** | 0.4171 | **66.7%** | **0** | **2** |

---

## 5. Acceptance Criteria Status

| Criterion | Requirement | Observed Metric | Status |
|---|---|---|---|
| **Criterion 1: Dual ROC-AUC & F1 Outperformance** | Both ROC-AUC AND F1 win rate $\ge 80.0\%$ | Combined Win Rate = **46.67%** (F1 alone: 84.44%, ROC-AUC alone: 53.33%) | **FAIL** |
| **Criterion 2: Minimum Recall Floor** | Recall $\ge 0.35$ on 100% of folds | Min Recall = **0.0738** (7 of 45 folds < 0.35) | **FAIL** |
| **Criterion 3: Spearman Feature Rank Stability** | Rank correlation $\rho \ge 0.65$ | Actual Mean $\rho = \mathbf{0.0019}$ | **FAIL** |
| **Criterion 4: Tree Latency Reduction** | Training & inference reduction $\ge 40.0\%$ | Tree latency reduction = **-0.05% to 12.76%** | **FAIL** |

**Overall Milestone Result:** **FAIL** (0 of 4 criteria met).

---

## 6. Statistically & Empirically Demonstrated Positive Findings
* **Total Elimination of Prediction Collapse:** `SHORTLIST_16` reduced severe prediction collapse (Recall < 0.05) from **9 fold failures** in `ALL_32` down to **0 fold failures** across all 45 model/asset runs.
* **Massive F1 Score Improvement:** `SHORTLIST_16` outperformed `ALL_32` on F1 score in **84.44%** of all paired folds, improving mean F1 by **+24.23%** in Random Forest and **+16.78%** in XGBoost.
* **Tree Model ROC-AUC Gains:** On tree-based models (Random Forest & XGBoost), `SHORTLIST_16` achieved higher mean ROC-AUC than `ALL_32` across 60% of folds.

---

## 7. Important Negative Findings & Limitations
* **Failure on Dual Metric Requirement:** While `SHORTLIST_16` won on F1 in 84.4% of folds, it won on ROC-AUC in only 53.3% of folds, yielding a combined dual win rate of **46.67%** (below the 80% threshold).
* **Low Recall Threshold Breaches:** 7 out of 45 folds fell below the 0.35 recall floor (min recall: `0.0738` on `infy_ns` Logistic Regression Fold 1).
* **Cross-Asset Feature Rank Instability:** Cross-asset Spearman rank correlation was near zero ($\rho = 0.0019$), demonstrating that feature importance ordering varies significantly across different market sectors (e.g., Energy vs IT vs Banking).
* **Negligible Computational Speedup:** Reducing feature count from 32 to 16 provided only ~0% to 12.8% latency reduction because tree depth and sample size dominate compute time.

---

## 8. Leakage & Safety Verification
* **Holdout Protection:** The final 15% out-of-sample holdout test set was **100% untouched** and never referenced during fold evaluation or subset scoring.
* **Scaler Isolation:** Feature scaling (`FeatureScaler(scale=True)`) was fit strictly on `X_train` inside each fold loop.
* **Production Integrity:** `champion.json`, model registry, and production feature pipelines remain completely unmodified.

---

## 9. Why SHORTLIST_16 Cannot Currently Be Promoted
Although `SHORTLIST_16` delivers major empirical improvements in model stability and F1-score over `ALL_32`, it failed all four formal numerical acceptance criteria established for Mission 11. Promoting `SHORTLIST_16` to production under these failed criteria would violate scientific standards.

---

## 10. Research Questions for Mission 12
1. **Regime-Adaptive Feature Selection:** Can sector-specific or market-regime-aware feature subsets satisfy cross-asset rank stability where static global subsets fail?
2. **Threshold & Class-Balance Optimization:** Can probability decision threshold tuning elevate fold recall above the 0.35 floor without sacrificing precision?
3. **Hyperparameter Interaction:** Does combining `SHORTLIST_16` with targeted hyperparameter tuning (e.g. tree depth, min child weight) unlock the ROC-AUC gains required to exceed the 80% dual outperformance benchmark?
