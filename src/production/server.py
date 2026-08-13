"""Production REST API & Web Application Server for AlphaForge.

Serves backend JSON API endpoints and static frontend UI assets.
Zero external server dependencies required (uses standard library http.server + json).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
import urllib.parse

from src.production.paper_portfolio import PaperPortfolioEngine
from src.production.trading_engine import ASSET_DISPLAY_NAMES, SUPPORTED_ASSETS, ProductionTradingEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Global Instances for Production Server
ENGINE: Optional[ProductionTradingEngine] = None
PORTFOLIO: Optional[PaperPortfolioEngine] = None


def get_engine() -> ProductionTradingEngine:
    global ENGINE
    if ENGINE is None:
        logger.info("Initializing Production Trading Engine models...")
        ENGINE = ProductionTradingEngine()
    return ENGINE


def get_portfolio() -> PaperPortfolioEngine:
    global PORTFOLIO
    if PORTFOLIO is None:
        PORTFOLIO = PaperPortfolioEngine(initial_capital=100000.0)
    return PORTFOLIO


class AlphaForgeRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for AlphaForge API and Web UI."""

    def _send_json(self, data: Dict[str, Any], status_code: int = 200) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path) -> None:
        resolved_path = file_path.resolve()
        resolved_static = STATIC_DIR.resolve()
        try:
            resolved_path.relative_to(resolved_static)
        except ValueError:
            self._send_json({"error": True, "message": "Access Denied: Path Traversal", "code": "ACCESS_DENIED"}, status_code=403)
            return

        if not resolved_path.exists() or not resolved_path.is_file():
            self._send_json({"error": True, "message": f"File not found: {file_path.name}", "code": "NOT_FOUND"}, status_code=404)
            return

        mime_type, _ = mimetypes.guess_type(str(resolved_path))
        mime_type = mime_type or "application/octet-stream"

        content = resolved_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        try:
            if path == "/api/health":
                self._send_json({"status": "ONLINE", "version": "1.0.0", "name": "AlphaForge Trading Platform"})
            elif path == "/api/portfolio":
                portfolio = get_portfolio()
                summary = portfolio.get_portfolio_summary()
                self._send_json(summary)
            elif path == "/api/backtests":
                self._send_json(_get_backtest_research_summaries())
            elif path == "/api/live-price":
                engine = get_engine()
                symbol = query.get("symbol", [engine.assets[0]])[0]
                live_info = engine.get_live_price_data(symbol)
                self._send_json(live_info)
            elif path == "/api/risk-summary":
                from src.research.benchmark_and_cost_reality_check import calculate_nse_delivery_cost
                sample_trade_cap = 20000.0
                entry_c = calculate_nse_delivery_cost(sample_trade_cap, is_buy=True)
                exit_c = calculate_nse_delivery_cost(sample_trade_cap, is_buy=False)
                round_trip_cost_pct = ((entry_c + exit_c) / sample_trade_cap) * 100.0

                self._send_json({
                    "max_position_weight_pct": 20.0,
                    "max_portfolio_exposure_pct": 100.0,
                    "sector_caps": {"IT_Services": 35.0, "Banking_Financials": 35.0, "Energy": 35.0},
                    "drawdown_governor_status": "ACTIVE (Hysteresis Buffer 20% -> 10%)",
                    "transaction_cost_bps": round(round_trip_cost_pct * 100.0, 1),
                    "transaction_cost_pct": round(round_trip_cost_pct, 3),
                    "transaction_cost_model": "2026 NSE Delivery Rates (STT 0.10% buy/sell + Stamp Duty 0.015% + Exch 0.00354% + DP Flat ₹15.93: ~0.302% / 30.2 bps round-trip)",
                })
            elif path == "/api/assets":
                engine = get_engine()
                assets_data = []
                for a in engine.assets:
                    m_data = engine.get_asset_market_data(a, limit=2)
                    live_p = engine.get_live_price_data(a)
                    assets_data.append({
                        "symbol": a,
                        "display_name": ASSET_DISPLAY_NAMES.get(a, a.upper()),
                        "last_price": live_p["current_price"],
                        "change_pct": live_p["change_pct"],
                        "is_market_open": live_p["is_market_open"],
                    })
                self._send_json({"assets": assets_data})
            elif path == "/api/signal":
                engine = get_engine()
                symbol = query.get("symbol", [engine.assets[0]])[0]
                sig = engine.predict_trade_signal(symbol)
                self._send_json(sig)
            elif path == "/api/market-data":
                engine = get_engine()
                symbol = query.get("symbol", [engine.assets[0]])[0]
                m_data = engine.get_asset_market_data(symbol, limit=150)
                m_data["live_price_info"] = engine.get_live_price_data(symbol)
                self._send_json(m_data)
            else:
                # Serve Static Web UI Files
                target_path = STATIC_DIR / ("index.html" if path == "/" or path == "" else path.lstrip("/"))
                self._send_file(target_path)
        except KeyError as k_err:
            self._send_json({"error": True, "message": f"Asset symbol error: {str(k_err)}", "code": "INVALID_SYMBOL"}, status_code=400)
        except Exception as err:
            logger.error("Error handling GET %s: %s", path, str(err))
            self._send_json({"error": True, "message": str(err), "code": "INTERNAL_SERVER_ERROR"}, status_code=500)

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            body_json = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            body_json = {}

        try:
            if path == "/api/paper-trade":
                engine = get_engine()
                portfolio = get_portfolio()

                symbol = body_json.get("symbol", "tcs_ns")
                action = body_json.get("action", "BUY")
                capital = float(body_json.get("capital", 20000.0))

                if capital <= 0:
                    self._send_json({"error": True, "message": "Capital allocation must be positive", "code": "INVALID_CAPITAL"}, status_code=400)
                    return

                # Enforce 20% max position weight cap relative to portfolio equity
                port_summary = portfolio.get_portfolio_summary()
                max_20pct_cap = port_summary["current_equity"] * 0.20
                capital = min(capital, max_20pct_cap)

                m_data = engine.get_asset_market_data(symbol, limit=1)
                curr_price = m_data["last_price"]

                res = portfolio.execute_trade(symbol, action, curr_price, capital_alloc=capital)
                if res.get("status") == "ERROR":
                    self._send_json({"error": True, "message": res["message"], "code": "TRADE_EXECUTION_ERROR"}, status_code=400)
                else:
                    self._send_json(res)
            elif path == "/api/portfolio/reset":
                portfolio = get_portfolio()
                cap = float(body_json.get("initial_capital", 100000.0))
                portfolio.reset_portfolio(cap)
                self._send_json({"status": "SUCCESS", "message": f"Portfolio reset to INR {cap:,.2f}"})
            else:
                self._send_json({"error": True, "message": f"POST path '{path}' not found", "code": "NOT_FOUND"}, status_code=404)
        except KeyError as k_err:
            self._send_json({"error": True, "message": f"Asset symbol error: {str(k_err)}", "code": "INVALID_SYMBOL"}, status_code=400)
        except Exception as err:
            logger.error("Error handling POST %s: %s", path, str(err))
            self._send_json({"error": True, "message": str(err), "code": "INTERNAL_SERVER_ERROR"}, status_code=500)


def _get_backtest_research_summaries() -> Dict[str, Any]:
    """Return key validated research summaries and realistic cost reality check benchmark results."""
    return {
        "status": "HONEST_VALIDATION_COMPLETED",
        "verdict": "0 of 5 equities beat Buy-and-Hold on CAGR under realistic transaction costs",
        "evaluation_period": "2003-08-12 to 2026-08-06 (5,695 trading days / ~23.0 years)",
        "per_stock_reality_check": [
            {"asset": "reliance_ns", "display_name": "RELIANCE", "champion_cagr": 8.34, "bh_cagr": 19.04, "cagr_diff": -10.70, "champion_sharpe": 0.50, "bh_sharpe": 0.43, "verdict": "NO"},
            {"asset": "tcs_ns", "display_name": "TCS", "champion_cagr": 8.67, "bh_cagr": 20.02, "cagr_diff": -11.35, "champion_sharpe": 0.52, "bh_sharpe": 0.51, "verdict": "NO"},
            {"asset": "hdfcbank_ns", "display_name": "HDFCBANK", "champion_cagr": 6.33, "bh_cagr": 19.29, "cagr_diff": -12.96, "champion_sharpe": 0.40, "bh_sharpe": 0.76, "verdict": "NO"},
            {"asset": "infy_ns", "display_name": "INFY", "champion_cagr": 1.26, "bh_cagr": 14.54, "cagr_diff": -13.28, "champion_sharpe": 0.16, "bh_sharpe": 0.60, "verdict": "NO"},
            {"asset": "icicibank_ns", "display_name": "ICICIBANK", "champion_cagr": 4.90, "bh_cagr": 18.95, "cagr_diff": -14.05, "champion_sharpe": 0.33, "bh_sharpe": 0.65, "verdict": "NO"},
        ],
        "pooled_reality_check": {
            "champion_cagr": 5.25,
            "champion_return": 217.97,
            "champion_sharpe": 0.60,
            "bh_cagr": 18.66,
            "bh_return": 4676.95,
            "bh_sharpe": 0.71,
            "verdict": "NO (Buy-and-Hold outperformed Champion strategy by >21x in total return)",
        },
        "mission26_champion": {
            "signal": "P(up) >= 0.55 AND Expected Return > 1.0% (Superseded)",
            "mean_cum_return_pct": 79.84,
            "daily_sharpe": 1.09,
            "positive_folds": "5 / 5 (Superseded — See Reality Check)",
        },
        "mission27_cross_asset": [
            {"asset": "tcs_ns", "display_name": "TCS", "cum_return_pct": 79.84, "sharpe": 1.09, "expectancy_pct": 2.09, "positive_folds": "5/5 (Superseded)"},
            {"asset": "infy_ns", "display_name": "INFY", "cum_return_pct": 11.99, "sharpe": 0.31, "expectancy_pct": 2.84, "positive_folds": "4/5 (Superseded)"},
            {"asset": "reliance_ns", "display_name": "RELIANCE", "cum_return_pct": 53.35, "sharpe": 0.70, "expectancy_pct": 1.60, "positive_folds": "5/5 (Superseded)"},
            {"asset": "icicibank_ns", "display_name": "ICICIBANK", "cum_return_pct": 44.98, "sharpe": 0.45, "expectancy_pct": 1.19, "positive_folds": "4/5 (Superseded)"},
            {"asset": "hdfcbank_ns", "display_name": "HDFCBANK", "cum_return_pct": 42.74, "sharpe": 0.54, "expectancy_pct": 1.23, "positive_folds": "4/5 (Superseded)"},
        ],
        "superseded_historical_matrix": [
            {"asset": "tcs_ns", "display_name": "TCS", "cum_return_pct": 79.84, "sharpe": 1.09, "expectancy_pct": 2.09, "positive_folds": "5/5 (Superseded)"},
            {"asset": "infy_ns", "display_name": "INFY", "cum_return_pct": 11.99, "sharpe": 0.31, "expectancy_pct": 2.84, "positive_folds": "4/5 (Superseded)"},
            {"asset": "reliance_ns", "display_name": "RELIANCE", "cum_return_pct": 53.35, "sharpe": 0.70, "expectancy_pct": 1.60, "positive_folds": "5/5 (Superseded)"},
            {"asset": "icicibank_ns", "display_name": "ICICIBANK", "cum_return_pct": 44.98, "sharpe": 0.45, "expectancy_pct": 1.19, "positive_folds": "4/5 (Superseded)"},
            {"asset": "hdfcbank_ns", "display_name": "HDFCBANK", "cum_return_pct": 42.74, "sharpe": 0.54, "expectancy_pct": 1.23, "positive_folds": "4/5 (Superseded)"},
        ],
    }


def start_server(host: str = "0.0.0.0", port: int = 8080) -> ThreadingHTTPServer:
    """Launch production ThreadingHTTPServer immediately and pre-warm models asynchronously."""
    import threading
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, AlphaForgeRequestHandler)

    def _async_prewarm():
        try:
            from src.data.downloader import refresh_all_datasets
            logger.info("Checking and refreshing raw OHLCV datasets on server startup...")
            refresh_all_datasets()
        except Exception as err:
            logger.warning("Auto dataset refresh skipped / failed (using existing local parquets): %s", str(err))

        logger.info("Pre-warming Trading Engine models in background...")
        get_engine()
        logger.info("Trading Engine models ready.")

    threading.Thread(target=_async_prewarm, daemon=True).start()
    logger.info("AlphaForge Server running at http://127.0.0.1:%d", port)
    return httpd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaForge Production Web Application Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    args = parser.parse_args()

    httpd = start_server(args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("AlphaForge Server stopped by user.")
        httpd.server_close()
