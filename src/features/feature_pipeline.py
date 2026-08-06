"""
AlphaForge Feature Pipeline
"""

from __future__ import annotations

import pandas as pd

from src.features.momentum import MomentumFeatureGenerator
from src.features.price import PriceFeatureGenerator
from src.features.trend import TrendFeatureGenerator
from src.features.volatility import VolatilityFeatureGenerator
from src.features.volume import VolumeFeatureGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePipeline:

    def __init__(self):

        self.trend = TrendFeatureGenerator()
        self.momentum = MomentumFeatureGenerator()
        self.volatility = VolatilityFeatureGenerator()
        self.volume = VolumeFeatureGenerator()
        self.price = PriceFeatureGenerator()

        self.generators = [
            self.trend,
            self.momentum,
            self.volatility,
            self.volume,
            self.price,
        ]

    def generate(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info("=" * 60)
        logger.info("Starting Feature Engineering Pipeline")
        logger.info("=" * 60)

        for generator in self.generators:
            df = generator.generate(df)

        logger.info("Feature engineering completed.")

        return df
