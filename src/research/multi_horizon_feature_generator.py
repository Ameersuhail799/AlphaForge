"""Mission 16 Step 1: Multi-Horizon Trading Feature Generator.

Research-isolated feature generator implementing 31 stationary multi-horizon features
covering price relative distance, trend slopes, multi-period momentum, volatility regimes,
breakout structure, volume confirmation, and mean reversion Z-scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_GROUP_A = [
    "CLOSE_TO_SMA20",
    "CLOSE_TO_SMA50",
    "CLOSE_TO_SMA200",
    "SMA20_SMA50_SPREAD",
    "SMA50_SMA200_SPREAD",
]

FEATURE_GROUP_B = [
    "RETURN_3D",
    "RETURN_5D",
    "RETURN_10D",
    "RETURN_20D",
]

FEATURE_GROUP_C = [
    "SMA20_SLOPE_5D",
    "SMA50_SLOPE_10D",
    "EMA20_SLOPE_5D",
]

FEATURE_GROUP_D = [
    "ATR_TO_PRICE",
    "VOLATILITY_5D",
    "VOLATILITY_20D",
    "VOLATILITY_60D",
    "RANGE_TO_ATR",
    "VOLATILITY_RATIO_SHORT_LONG",
]

FEATURE_GROUP_E = [
    "DISTANCE_TO_20D_HIGH",
    "DISTANCE_TO_20D_LOW",
    "DISTANCE_TO_52W_HIGH",
    "DISTANCE_TO_52W_LOW",
    "POSITION_IN_20D_RANGE",
]

FEATURE_GROUP_F = [
    "VOLUME_RATIO_5",
    "VOLUME_RATIO_20",
    "VOLUME_CHANGE_5D",
    "VOLUME_TREND_RATIO",
]

FEATURE_GROUP_G = [
    "RSI_NEUTRAL_DIFF",
    "MOMENTUM_ACCELERATION",
    "MOMENTUM_SPREAD_SHORT_LONG",
    "PRICE_ZSCORE_20D",
]

PROPOSED_31_FEATURES = (
    FEATURE_GROUP_A
    + FEATURE_GROUP_B
    + FEATURE_GROUP_C
    + FEATURE_GROUP_D
    + FEATURE_GROUP_E
    + FEATURE_GROUP_F
    + FEATURE_GROUP_G
)


class MultiHorizonFeatureGenerator:
    """Generates 31 proposed multi-horizon normalized trading features."""

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 31 multi-horizon features on input market dataframe.
        
        Uses strictly historical/backward-looking windows at row index t.
        """
        self._validate(df)
        df = df.copy()

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        open_price = df["Open"]
        volume = df["Volume"]

        # Base intermediate indicators (backward-looking only)
        sma20 = close.rolling(window=20).mean()
        sma50 = close.rolling(window=50).mean()
        sma200 = close.rolling(window=200).mean()

        ema20 = close.ewm(span=20, adjust=False).mean()

        # True Range and ATR_14
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = np.maximum.reduce([tr1, tr2, tr3])
        atr14 = pd.Series(true_range, index=df.index).rolling(14).mean()

        # Log returns
        log_ret = np.log(close / close.shift(1))

        # --- GROUP A: Price / Trend Position ---
        df["CLOSE_TO_SMA20"] = (close / sma20) - 1.0
        df["CLOSE_TO_SMA50"] = (close / sma50) - 1.0
        df["CLOSE_TO_SMA200"] = (close / sma200) - 1.0
        df["SMA20_SMA50_SPREAD"] = (sma20 - sma50) / (sma50 + 1e-8)
        df["SMA50_SMA200_SPREAD"] = (sma50 - sma200) / (sma200 + 1e-8)

        # --- GROUP B: Multi-Horizon Returns ---
        df["RETURN_3D"] = (close - close.shift(3)) / close.shift(3)
        df["RETURN_5D"] = (close - close.shift(5)) / close.shift(5)
        df["RETURN_10D"] = (close - close.shift(10)) / close.shift(10)
        df["RETURN_20D"] = (close - close.shift(20)) / close.shift(20)

        # --- GROUP C: Trend Slope ---
        df["SMA20_SLOPE_5D"] = (sma20 - sma20.shift(5)) / (sma20.shift(5) + 1e-8)
        df["SMA50_SLOPE_10D"] = (sma50 - sma50.shift(10)) / (sma50.shift(10) + 1e-8)
        df["EMA20_SLOPE_5D"] = (ema20 - ema20.shift(5)) / (ema20.shift(5) + 1e-8)

        # --- GROUP D: Volatility Regime ---
        df["ATR_TO_PRICE"] = atr14 / close
        df["VOLATILITY_5D"] = log_ret.rolling(5).std() * np.sqrt(252)
        df["VOLATILITY_20D"] = log_ret.rolling(20).std() * np.sqrt(252)
        df["VOLATILITY_60D"] = log_ret.rolling(60).std() * np.sqrt(252)
        df["RANGE_TO_ATR"] = (high - low) / (atr14 + 1e-8)
        df["VOLATILITY_RATIO_SHORT_LONG"] = df["VOLATILITY_5D"] / (df["VOLATILITY_60D"] + 1e-8)

        # --- GROUP E: Breakout / Price Location ---
        high_20d = high.rolling(20).max()
        low_20d = low.rolling(20).min()
        high_52w = high.rolling(252).max()
        low_52w = low.rolling(252).min()

        df["DISTANCE_TO_20D_HIGH"] = (close - high_20d) / (high_20d + 1e-8)
        df["DISTANCE_TO_20D_LOW"] = (close - low_20d) / (low_20d + 1e-8)
        df["DISTANCE_TO_52W_HIGH"] = (close - high_52w) / (high_52w + 1e-8)
        df["DISTANCE_TO_52W_LOW"] = (close - low_52w) / (low_52w + 1e-8)
        df["POSITION_IN_20D_RANGE"] = (close - low_20d) / (high_20d - low_20d + 1e-8)

        # --- GROUP F: Volume Confirmation ---
        vol_sma5 = volume.rolling(5).mean()
        vol_sma20 = volume.rolling(20).mean()

        df["VOLUME_RATIO_5"] = volume / (vol_sma5 + 1e-8)
        df["VOLUME_RATIO_20"] = volume / (vol_sma20 + 1e-8)
        df["VOLUME_CHANGE_5D"] = (volume - volume.shift(5)) / (volume.shift(5) + 1e-8)
        df["VOLUME_TREND_RATIO"] = vol_sma5 / (vol_sma20 + 1e-8)

        # --- GROUP G: Momentum / Mean Reversion ---
        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi14 = 100.0 - (100.0 / (1.0 + rs))

        rolling_std_20 = close.rolling(20).std()

        df["RSI_NEUTRAL_DIFF"] = rsi14 - 50.0
        df["MOMENTUM_ACCELERATION"] = df["RETURN_5D"] - df["RETURN_5D"].shift(5)
        df["MOMENTUM_SPREAD_SHORT_LONG"] = df["RETURN_5D"] - df["RETURN_20D"]
        df["PRICE_ZSCORE_20D"] = (close - sma20) / (rolling_std_20 + 1e-8)

        logger.info("Multi-horizon feature generation completed: 31 features added.")
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required market columns: {missing}")
