"""Production Paper Trading & Portfolio Engine for AlphaForge.

Maintains persistent portfolio state, positions, cash, execution costs, and P&L tracking.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PaperPortfolioEngine:
    """Production Paper Trading & Portfolio Tracker."""

    def __init__(self, initial_capital: float = 100000.0) -> None:
        self.initial_capital = initial_capital
        self.reset_portfolio(initial_capital)

    def reset_portfolio(self, initial_capital: float = 100000.0) -> None:
        """Reset portfolio back to initial state."""
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.equity_history: List[Dict[str, Any]] = [
            {"timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "portfolio_equity": initial_capital, "cash": initial_capital}
        ]

    def execute_trade(
        self,
        asset_symbol: str,
        signal_type: str,
        current_price: float,
        capital_alloc: float = 20000.0,
        cost_bps: float = 0.0010,
    ) -> Dict[str, Any]:
        """Execute a paper trade (BUY or SELL/CLOSE)."""
        symbol = asset_symbol.lower()

        if signal_type == "BUY":
            if symbol in self.open_positions:
                return {"status": "ERROR", "message": f"Active position already exists for {symbol.upper()}."}

            # Enforce 20% position allocation cap relative to current total equity
            pos_val_sum = sum(p["units"] * p["current_price"] for p in self.open_positions.values())
            total_equity = self.cash + pos_val_sum
            max_20pct_alloc = total_equity * 0.20

            alloc_cash = min(capital_alloc, self.cash, max_20pct_alloc)
            if alloc_cash <= 1000.0:
                return {"status": "ERROR", "message": f"Cash allocation (INR {alloc_cash:,.2f}) below minimum limit or exceeds 20% equity cap (INR {max_20pct_alloc:,.2f})."}

            from src.research.benchmark_and_cost_reality_check import calculate_nse_delivery_cost
            entry_cost = calculate_nse_delivery_cost(alloc_cash, is_buy=True)
            units = (alloc_cash - entry_cost) / current_price

            self.cash -= alloc_cash

            pos = {
                "asset": symbol,
                "entry_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "entry_price": current_price,
                "allocated_cash": alloc_cash,
                "units": units,
                "entry_cost": entry_cost,
                "current_price": current_price,
                "unrealized_pnl": -entry_cost,
                "unrealized_pnl_pct": (-entry_cost / alloc_cash) * 100.0,
            }
            self.open_positions[symbol] = pos
            self.update_market_prices({symbol: current_price})

            logger.info("Paper BUY executed for %s: %f units at INR %.2f (Allocated INR %.2f)", symbol.upper(), units, current_price, alloc_cash)
            return {"status": "SUCCESS", "action": "BUY", "position": pos}

        elif signal_type == "SELL" or signal_type == "CLOSE":
            if symbol not in self.open_positions:
                return {"status": "ERROR", "message": f"No active position found for {symbol.upper()}."}

            pos = self.open_positions.pop(symbol)
            alloc_cash = pos["allocated_cash"]
            entry_p = pos["entry_price"]
            units = pos["units"]
            entry_cost = pos["entry_cost"]

            from src.research.benchmark_and_cost_reality_check import calculate_nse_delivery_cost
            exit_cost = calculate_nse_delivery_cost(units * current_price, is_buy=False)

            gross_pnl = units * (current_price - entry_p)
            net_pnl = gross_pnl - entry_cost - exit_cost

            gross_ret = (current_price - entry_p) / entry_p
            net_ret = net_pnl / alloc_cash

            self.cash += alloc_cash + net_pnl

            trade_record = {
                "trade_id": len(self.trade_history) + 1,
                "asset": symbol,
                "entry_date": pos["entry_date"],
                "exit_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "entry_price": entry_p,
                "exit_price": current_price,
                "units": units,
                "allocated_cash": alloc_cash,
                "net_pnl": net_pnl,
                "net_return_pct": net_ret * 100.0,
                "is_win": (net_pnl > 0),
            }
            self.trade_history.append(trade_record)
            self.update_market_prices({})

            logger.info("Paper SELL executed for %s: Net PnL INR %.2f (%.2f%%)", symbol.upper(), net_pnl, net_ret * 100.0)
            return {"status": "SUCCESS", "action": "SELL", "trade": trade_record}

        return {"status": "ERROR", "message": "Invalid signal type."}

    def update_market_prices(self, current_prices: Dict[str, float]) -> None:
        """Update active positions and recalculate portfolio equity."""
        pos_value_sum = 0.0
        for symbol, pos in self.open_positions.items():
            if symbol in current_prices:
                pos["current_price"] = current_prices[symbol]
            curr_p = pos["current_price"]
            entry_p = pos["entry_price"]
            units = pos["units"]
            alloc_cash = pos["allocated_cash"]

            unreal_pnl = units * (curr_p - entry_p) - pos["entry_cost"]
            pos["unrealized_pnl"] = unreal_pnl
            pos["unrealized_pnl_pct"] = (unreal_pnl / alloc_cash) * 100.0

            pos_value_sum += units * curr_p

        total_equity = self.cash + pos_value_sum
        self.equity_history.append({
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "portfolio_equity": total_equity,
            "cash": self.cash,
        })

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Return complete portfolio summary and metrics."""
        pos_value_sum = sum(pos["units"] * pos["current_price"] for pos in self.open_positions.values())
        total_equity = self.cash + pos_value_sum
        total_pnl = total_equity - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100.0

        realized_pnl = sum(tr["net_pnl"] for tr in self.trade_history)
        unrealized_pnl = sum(pos["unrealized_pnl"] for pos in self.open_positions.values())

        total_trades = len(self.trade_history)
        wins = [tr for tr in self.trade_history if tr["is_win"]]
        win_rate_pct = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0

        # Maximum Drawdown calculation from equity history
        eq_vals = [h["portfolio_equity"] for h in self.equity_history]
        pk = np.maximum.accumulate(eq_vals)
        dd = (pk - eq_vals) / pk
        max_dd_pct = float(np.max(dd)) * 100.0 if len(dd) > 0 else 0.0

        return {
            "initial_capital": self.initial_capital,
            "current_equity": total_equity,
            "cash_balance": self.cash,
            "positions_value": pos_value_sum,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_trades": total_trades,
            "winning_trades": len(wins),
            "win_rate_pct": win_rate_pct,
            "max_drawdown_pct": max_dd_pct,
            "open_positions": list(self.open_positions.values()),
            "trade_history": self.trade_history,
            "equity_history": self.equity_history,
        }
