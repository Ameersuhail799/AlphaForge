"""Mission 16 Step 1D: Feature Redundancy & Correlation Analysis.

Calculates cross-correlations among SHORTLIST_16 and the 31 proposed multi-horizon features
on TCS non-test market data, classifying features into Redundant, Complementary, or Potentially Useful.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.data.storage import StorageEngine
from src.features.feature_pipeline import FeaturePipeline
from src.research.multi_horizon_feature_generator import (
    PROPOSED_31_FEATURES,
    MultiHorizonFeatureGenerator,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

SHORTLIST_16 = [
    "GAP_PCT",
    "OPEN_CLOSE_PCT",
    "HIGH_LOW_PCT",
    "BODY_SIZE",
    "UPPER_WICK",
    "LOWER_WICK",
    "ROC_12",
    "RSI_14",
    "PRICE_CHANGE_PCT",
    "DAILY_RETURN",
    "ROLLING_STD_20",
    "HIST_VOL_20",
    "ATR_14",
    "DAILY_RANGE_PCT",
    "VOLUME_RATIO",
    "VOLUME_CHANGE_PCT",
]


def run_feature_redundancy_analysis(
    asset_name: str = "tcs_ns",
    output_dir: Path | str = Path("reports") / "research",
) -> pd.DataFrame:
    """Execute feature redundancy and correlation classification."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    storage = StorageEngine()
    fp = FeaturePipeline()
    gen_mh = MultiHorizonFeatureGenerator()

    raw = storage.load_dataset(asset_name)
    df_base = fp.generate(raw.copy())
    df_full = gen_mh.generate(df_base)

    # Exclude test partition
    total_rows = len(df_full)
    test_size = max(1, int(total_rows * 0.15))
    non_test = df_full.iloc[:-test_size].copy()

    all_features = [c for c in SHORTLIST_16 + PROPOSED_31_FEATURES if c in non_test.columns]
    valid_df = non_test[all_features].replace([np.inf, -np.inf], np.nan).dropna()

    corr_matrix = valid_df.corr().abs()

    records = []
    for col in PROPOSED_31_FEATURES:
        if col not in corr_matrix.columns:
            continue

        # Highest correlation with any SHORTLIST_16 feature
        shortlist_corrs = corr_matrix.loc[col, [c for c in SHORTLIST_16 if c in corr_matrix.columns]]
        max_shortlist_corr = float(shortlist_corrs.max())
        most_similar_shortlist_feat = str(shortlist_corrs.idxmax())

        # Highest correlation with any other proposed feature
        other_proposed = [c for c in PROPOSED_31_FEATURES if c != col and c in corr_matrix.columns]
        other_corrs = corr_matrix.loc[col, other_proposed]
        max_other_corr = float(other_corrs.max()) if not other_corrs.empty else 0.0
        most_similar_other_feat = str(other_corrs.idxmax()) if not other_corrs.empty else "None"

        # Classification logic
        if max_shortlist_corr >= 0.85:
            classification = "Redundant (High Base Correlation)"
            reasoning = f"Correlated with base feature {most_similar_shortlist_feat} (|r|={max_shortlist_corr:.2f} >= 0.85)"
        elif max_shortlist_corr <= 0.30:
            classification = "Complementary (Orthogonal Signal)"
            reasoning = f"Low correlation with base features (Max |r|={max_shortlist_corr:.2f} with {most_similar_shortlist_feat})"
        else:
            classification = "Potentially Useful (Moderate Correlation)"
            reasoning = f"Moderate correlation (|r|={max_shortlist_corr:.2f}) with base feature {most_similar_shortlist_feat}"

        records.append({
            "proposed_feature": col,
            "max_shortlist_corr": max_shortlist_corr,
            "most_similar_base_feature": most_similar_shortlist_feat,
            "max_other_proposed_corr": max_other_corr,
            "most_similar_other_feature": most_similar_other_feat,
            "classification": classification,
            "reasoning": reasoning,
        })

    df_res = pd.DataFrame(records).sort_values("max_shortlist_corr", ascending=False).reset_index(drop=True)
    df_res.to_csv(output_dir / "mission16_feature_redundancy.csv", index=False)
    logger.info("Feature redundancy analysis saved to %s", output_dir / "mission16_feature_redundancy.csv")

    return df_res


if __name__ == "__main__":
    run_feature_redundancy_analysis()
