"""Production Trading Engine for AlphaForge.

Handles end-to-end signal generation, inference, technical reasoning, and trade decision making.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.data.storage import StorageEngine
from src.dataset.scaler import FeatureScaler
from src.features.feature_pipeline import FeaturePipeline
from src.research.mission21_model_improvement import add_group_h_features, C57_FEATURES
from src.research.multi_horizon_feature_generator import MultiHorizonFeatureGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_ASSETS = ["tcs_ns", "infy_ns", "reliance_ns", "icicibank_ns", "hdfcbank_ns"]
ASSET_DISPLAY_NAMES = {
    "tcs_ns": "Tata Consultancy Services (TCS)",
    "infy_ns": "Infosys Ltd (INFY)",
    "reliance_ns": "Reliance Industries Ltd (RELIANCE)",
    "icicibank_ns": "ICICI Bank Ltd (ICICIBANK)",
    "hdfcbank_ns": "HDFC Bank Ltd (HDFCBANK)",
}

C59_COLS = C57_FEATURES + ["RANGE_COMPRESSION_EXP", "VOLUME_BREAKOUT_CONFIRM", "TREND_VOL_INTERACTION"]


class ProductionTradingEngine:
    """Production AI Trading Intelligence Engine."""

    def __init__(self, assets: Optional[List[str]] = None) -> None:
        self.assets = assets or SUPPORTED_ASSETS
        self.storage = StorageEngine()
        self.fp = FeaturePipeline()
        self.gen_mh = MultiHorizonFeatureGenerator()

        self.clf_models: Dict[str, RandomForestClassifier] = {}
        self.reg_models: Dict[str, RandomForestRegressor] = {}
        self.scalers: Dict[str, FeatureScaler] = {}
        self.processed_df_dict: Dict[str, pd.DataFrame] = {}

        self._initialize_and_train()

    def _initialize_and_train(self) -> None:
        """Fit models on historical dataset for production inference."""
        logger.info("Initializing Production Trading Engine across assets: %s", self.assets)

        for asset in self.assets:
            raw = self.storage.load_dataset(asset)
            df_base = self.fp.generate(raw.copy())
            df_mh = self.gen_mh.generate(df_base)
            df_full = add_group_h_features(df_mh)

            close = df_full["Close"]
            ret_10d = (close.shift(-10) - close) / close
            df_full["TARGET_D"] = (ret_10d > 0).astype(int)
            df_full["REALIZED_RET_10D"] = ret_10d

            train_required_cols = list(C59_COLS) + ["TARGET_D", "REALIZED_RET_10D", "Close", "Open", "High", "Low", "ATR_14", "HIST_VOL_20"]
            df_train = df_full.replace([np.inf, -np.inf], np.nan).dropna(subset=train_required_cols).copy()

            X_train = df_train[C59_COLS].copy()
            y_train = df_train["TARGET_D"]
            r_train = df_train["REALIZED_RET_10D"]

            scaler = FeatureScaler(scale=True)
            X_scaled = scaler.fit_transform_train(X_train)

            clf = RandomForestClassifier(n_estimators=25, n_jobs=-1, random_state=42)
            clf.fit(X_scaled, y_train)

            reg = RandomForestRegressor(n_estimators=25, n_jobs=-1, random_state=42)
            reg.fit(X_scaled, r_train)

            # Store inference dataframe (retains latest completed trading days up to today)
            infer_required_cols = list(C59_COLS) + ["Close", "Open", "High", "Low", "ATR_14", "HIST_VOL_20"]
            df_infer = df_full.replace([np.inf, -np.inf], np.nan).dropna(subset=infer_required_cols).copy()

            self.clf_models[asset] = clf
            self.reg_models[asset] = reg
            self.scalers[asset] = scaler
            self.processed_df_dict[asset] = df_infer

            logger.info("Asset %s trained cleanly. Total inference bars: %d", asset, len(df_infer))

    def get_asset_market_data(self, asset_symbol: str, limit: int = 150) -> Dict[str, Any]:
        """Retrieve market data and indicators for chart plotting."""
        symbol = asset_symbol.lower()
        if symbol not in self.processed_df_dict:
            raise KeyError(f"Asset '{asset_symbol}' not supported.")

        df = self.processed_df_dict[symbol].tail(limit)

        dates = [d.strftime("%Y-%m-%d") for d in df.index]
        close = df["Close"].values.tolist()
        open_p = df["Open"].values.tolist()
        high_p = df["High"].values.tolist()
        low_p = df["Low"].values.tolist()
        volume = df["Volume"].values.tolist()
        sma20 = df["SMA_20"].values.tolist() if "SMA_20" in df.columns else close
        sma50 = df["SMA_50"].values.tolist() if "SMA_50" in df.columns else close
        rsi14 = df["RSI_14"].values.tolist() if "RSI_14" in df.columns else [50.0] * len(df)
        atr14 = df["ATR_14"].values.tolist() if "ATR_14" in df.columns else [10.0] * len(df)

        return {
            "symbol": symbol,
            "display_name": ASSET_DISPLAY_NAMES.get(symbol, symbol.upper()),
            "dates": dates,
            "close": close,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "volume": volume,
            "sma20": sma20,
            "sma50": sma50,
            "rsi14": rsi14,
            "atr14": atr14,
            "last_price": close[-1],
            "price_change_pct": ((close[-1] - close[-2]) / close[-2]) * 100.0 if len(close) > 1 else 0.0,
        }

    def predict_trade_signal(self, asset_symbol: str) -> Dict[str, Any]:
        """Generate production AI trading signal and structured technical explanations."""
        symbol = asset_symbol.lower()
        if symbol not in self.processed_df_dict:
            raise KeyError(f"Asset '{asset_symbol}' not supported.")

        df = self.processed_df_dict[symbol]
        latest_row = df.iloc[-1:]

        X_latest = latest_row[C59_COLS]
        scaler = self.scalers[symbol]
        X_scaled = scaler.transform(X_latest)

        clf = self.clf_models[symbol]
        reg = self.reg_models[symbol]

        prob_up = float(clf.predict_proba(X_scaled)[0, 1])
        exp_ret = float(reg.predict(X_scaled)[0])

        last_close = float(latest_row["Close"].values[0])
        atr_val = float(latest_row["ATR_14"].values[0]) if "ATR_14" in latest_row.columns else last_close * 0.02
        hist_vol = float(latest_row["HIST_VOL_20"].values[0]) if "HIST_VOL_20" in latest_row.columns else 0.015
        range_comp = float(latest_row["RANGE_COMPRESSION_EXP"].values[0]) if "RANGE_COMPRESSION_EXP" in latest_row.columns else 1.0
        vol_breakout = float(latest_row["VOLUME_BREAKOUT_CONFIRM"].values[0]) if "VOLUME_BREAKOUT_CONFIRM" in latest_row.columns else 0.0
        bullish_regime = int(latest_row["BULLISH_TREND_REGIME"].values[0]) if "BULLISH_TREND_REGIME" in latest_row.columns else 1

        # Trade Decision Engine
        if prob_up >= 0.55 and exp_ret > 0.010:
            signal_type = "BUY"
            signal_color = "#10B981"  # Emerald Green
        elif prob_up <= 0.45 or exp_ret < -0.005:
            signal_type = "SELL"
            signal_color = "#EF4444"  # Red
        else:
            signal_type = "HOLD"
            signal_color = "#F59E0B"  # Amber

        # Risk Level Assessment
        rel_atr = (atr_val / last_close) * 100.0
        if rel_atr > 3.0 or hist_vol > 0.025:
            risk_level = "HIGH"
        elif rel_atr > 1.8 or hist_vol > 0.015:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        confidence_pct = float(np.clip(prob_up * 100.0, 50.0, 95.0))
        trade_score = float(np.clip((prob_up * exp_ret * 100.0) / (rel_atr / 100.0 + 1e-8), 0.0, 10.0))

        # Evidence & Technical Reasoning
        reasons = []
        if prob_up >= 0.55:
            reasons.append(f"Model direction probability P(up) is {prob_up*100.0:.1f}%, exceeding the 55.0% confidence threshold.")
        else:
            reasons.append(f"Model direction probability P(up) is {prob_up*100.0:.1f}%, below entry conviction requirement.")

        if exp_ret > 0.010:
            reasons.append(f"Predicted 10-day expected return is +{exp_ret*100.0:.2f}%, exceeding the +1.0% minimum edge requirement.")
        elif exp_ret > 0.0:
            reasons.append(f"Predicted 10-day expected return (+{exp_ret*100.0:.2f}%) is positive but below conviction threshold.")
        else:
            reasons.append(f"Predicted 10-day return ({exp_ret*100.0:.2f}%) indicates downside / stagnant price action.")

        if bullish_regime == 1:
            reasons.append("Causal trend alignment: Price > SMA50 and SMA20 > SMA50 confirm a structural bullish trend regime.")
        else:
            reasons.append("Macro trend structure indicates consolidation or bearish trend pressure.")

        if range_comp > 1.05:
            reasons.append("Volatility expansion detected (Range Compression Ratio > 1.05), supporting breakout momentum.")

        if vol_breakout > 0.0:
            reasons.append("Volume breakout confirmation active: Trading volume surge aligns with price action.")

        return {
            "symbol": symbol,
            "display_name": ASSET_DISPLAY_NAMES.get(symbol, symbol.upper()),
            "timestamp": latest_row.index[0].strftime("%Y-%m-%d"),
            "signal_as_of_date": latest_row.index[0].strftime("%Y-%m-%d"),
            "signal_timestamp_text": f"Signal calculated as of confirmed daily close: {latest_row.index[0].strftime('%Y-%m-%d')}",
            "last_price": last_close,
            "signal": signal_type,
            "signal_color": signal_color,
            "prob_up": prob_up,
            "prob_up_pct": prob_up * 100.0,
            "expected_return": exp_ret,
            "expected_return_pct": exp_ret * 100.0,
            "confidence_pct": confidence_pct,
            "risk_level": risk_level,
            "trade_score": trade_score,
            "horizon_days": 10,
            "atr_14": atr_val,
            "hist_vol_20": hist_vol,
            "reasons": reasons,
        }

    def get_live_price_data(self, asset_symbol: str) -> Dict[str, Any]:
        """Fetch current/live price and market hours status using YahooProvider."""
        symbol = asset_symbol.lower()
        fallback_p = 0.0
        if symbol in self.processed_df_dict:
            fallback_p = float(self.processed_df_dict[symbol]["Close"].iloc[-1])

        from src.data.providers.yahoo_provider import YahooProvider
        provider = YahooProvider()
        return provider.get_live_price(symbol, fallback_close=fallback_p)
