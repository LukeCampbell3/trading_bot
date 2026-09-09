"""Recent V14.3 win-rate validation using REAL Alpaca option bars.

This is stronger than the old delta-proxy backtest because P&L is generated from
actual historical option contract bars. It is still NOT a quote/fill proof:
option bars do not contain historical bid/ask, queue position, or fill latency.
Entry and exit values are therefore stressed by configurable haircuts.

The validator is timestamp-safe on the underlying: signal -> watch -> confirm ->
next minute option-bar entry. It uses only contracts Alpaca currently exposes,
so it is intended for recent-window validation (default 5 trading days), not a
long-horizon survivorship-free historical study.

Usage:
    python validate_v14_3_recent_options.py --symbol COIN --days 7
    python validate_v14_3_recent_options.py --symbol TSLA --days 7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from alpaca_config import AlpacaConfig
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, OptionBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from strategy.route_conditioning import RouteConditioner
from strategy.confirmation_engine import ConfirmationEngine
from strategy.watch_ticket import WatchTicket, TicketSide
from strategy.option_chain_fetcher import OptionChainFetcher
from strategy.v14_3_highvol_config import get_v14_3_config, can_trade_symbol


@dataclass
class ValidationTrade:
    symbol: str
    route: str
    side: str
    mode: str
    watch_time: str
    confirm_time: str
    entry_time: str
    exit_time: str
    long_contract: str
    short_contract: str
    entry_debit: float
    exit_credit_core: float
    exit_credit_runner: float
    quantity: int
    pnl: float
    pnl_pct_on_debit: float
    winner: bool
    exit_reason: str


def _stock_frame(client, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed.IEX,
        limit=10000,
    )
    bars = client.get_stock_bars(req)
    df = bars.df.copy()
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)
    return df.sort_index()


def _option_frame(client, contracts, start: datetime, end: datetime) -> pd.DataFrame:
    req = OptionBarsRequest(
        symbol_or_symbols=list(contracts),
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        limit=10000,
    )
    bars = client.get_option_bars(req)
    return bars.df.copy()


def _with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    typical = (high + low + close) / 3.0
    df["vwap_calc"] = (typical * volume).groupby(df.index.date).cumsum() / volume.groupby(df.index.date).cumsum().clip(lower=1)
    prev = close.shift(1)
    tr = pd.concat([(high-low), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    df["atr_calc"] = tr.rolling(14, min_periods=5).mean()
    df["vol_ratio"] = volume / volume.rolling(10, min_periods=5).mean().clip(lower=1)
    trend = np.zeros(len(df), dtype=float)
    arr = close.to_numpy(dtype=float)
    xs = np.arange(20, dtype=float)
    for i in range(19, len(df)):
        trend[i] = float(np.polyfit(xs, arr[i-19:i+1], 1)[0])
    df["trend_slope"] = trend
    # Intraday HOD/LOD without crossing days.
    df["hod"] = high.groupby(df.index.date).cummax()
    df["lod"] = low.groupby(df.index.date).cummin()
    return df


def _row_price(df: pd.DataFrame, i: int, ago: int) -> float:
    j = max(0, i - ago)
    return float(df["close"].iloc[j])


def _aligned_option_values(option_df: pd.DataFrame, long_symbol: str, short_symbol: str):
    try:
        long_df = option_df.xs(long_symbol, level=0).sort_index()
        short_df = option_df.xs(short_symbol, level=0).sort_index()
    except Exception:
        return None
    joined = long_df[["close"]].rename(columns={"close": "long_close"}).join(
        short_df[["close"]].rename(columns={"close": "short_close"}), how="inner"
    ).dropna()
    return joined if not joined.empty else None


def _first_index_at_or_after(index, ts):
    pos = index.searchsorted(ts, side="left")
    return int(pos) if pos < len(index) else None


def _estimate_pwin(symbol: str, route_score: float, mfe_velocity: float) -> float:
    prior = {"COIN": 0.75, "TSLA": 0.357}.get(symbol, 0.50)
    q = max(0.0, min(1.0, 0.75 * route_score + 0.25 * min(1.0, mfe_velocity)))
    return max(0.20, min(0.90, 0.55 * prior + 0.45 * q))


def _exit_trade(
    cfg: dict,
    stock_df: pd.DataFrame,
    option_joined: pd.DataFrame,
    entry_pos: int,
    entry_debit: float,
    mode: str,
    qty: int,
    core_qty: int,
    runner_qty: int,
    side: str,
    max_hold: int,
    exit_haircut: float,
):
    core_closed = False
    core_credit = 0.0
    runner_credit = 0.0
    realized = 0.0
    reason = "time_exit"
    end_pos = min(len(option_joined) - 1, entry_pos + max_hold)

    for j in range(entry_pos + 1, end_pos + 1):
        raw_credit = float(option_joined["long_close"].iloc[j] - option_joined["short_close"].iloc[j])
        credit = max(0.01, raw_credit * (1.0 - exit_haircut))
        pnl_pct = (credit - entry_debit) / entry_debit

        if mode == "FALLBACK":
            if pnl_pct <= cfg["single_spread_initial_stop_pct"]:
                reason = "fallback_stop"
            elif pnl_pct >= cfg["soft_greed_target_pct"]:
                reason = "fallback_target"
            else:
                continue
            realized = (credit - entry_debit) * 100.0 * qty
            core_credit = runner_credit = credit
            return j, credit, credit, realized, reason

        if not core_closed:
            if pnl_pct <= cfg["package_stop_pct"]:
                reason = "package_stop"
                realized = (credit - entry_debit) * 100.0 * qty
                return j, credit, credit, realized, reason
            if pnl_pct >= cfg["core_target_pct"]:
                core_closed = True
                core_credit = credit
                realized += (credit - entry_debit) * 100.0 * core_qty
                if runner_qty <= 0:
                    return j, core_credit, credit, realized, "core_target_all"
                continue
        else:
            if pnl_pct >= cfg["runner_target_pct"]:
                reason = "runner_target"
            elif pnl_pct <= cfg["runner_lock_pct"]:
                reason = "runner_lock"
            else:
                continue
            runner_credit = credit
            realized += (credit - entry_debit) * 100.0 * runner_qty
            return j, core_credit, runner_credit, realized, reason

    raw_credit = float(option_joined["long_close"].iloc[end_pos] - option_joined["short_close"].iloc[end_pos])
    credit = max(0.01, raw_credit * (1.0 - exit_haircut))
    if mode == "PACKAGE" and core_closed:
        runner_credit = credit
        realized += (credit - entry_debit) * 100.0 * runner_qty
    else:
        core_credit = runner_credit = credit
        realized = (credit - entry_debit) * 100.0 * qty
    return end_pos, core_credit, runner_credit, realized, reason


def validate(symbol: str, days: int, starting_equity: float, entry_slippage: float, exit_haircut: float):
    cfg = get_v14_3_config()
    allowed, policy_reason, size_mult = can_trade_symbol(symbol)
    if not allowed:
        raise RuntimeError(f"symbol policy blocks {symbol}: {policy_reason}")

    AlpacaConfig.validate()
    trading = TradingClient(AlpacaConfig.API_KEY, AlpacaConfig.API_SECRET, paper=True, url_override=AlpacaConfig.BASE_URL)
    stock_client = StockHistoricalDataClient(AlpacaConfig.API_KEY, AlpacaConfig.API_SECRET, url_override=AlpacaConfig.DATA_URL)
    option_client = OptionHistoricalDataClient(AlpacaConfig.API_KEY, AlpacaConfig.API_SECRET)
    chain = OptionChainFetcher(trading, option_client)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    stock_df = _with_indicators(_stock_frame(stock_client, symbol, start, end))
    if len(stock_df) < 100:
        raise RuntimeError(f"insufficient stock bars: {len(stock_df)}")

    route_engine = RouteConditioner(cfg)
    confirm_engine = ConfirmationEngine(cfg)
    option_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    trades = []
    equity = float(starting_equity)
    watch: Optional[WatchTicket] = None
    watch_high = watch_low = 0.0
    next_free_time = stock_df.index[0]
    trades_by_day: Dict[str, int] = {}

    for i in range(30, len(stock_df) - 2):
        ts = stock_df.index[i]
        if ts < next_free_time:
            continue
        row = stock_df.iloc[i]
        if not math.isfinite(float(row["atr_calc"])) or float(row["atr_calc"]) <= 0:
            continue
        day_key = str(ts.date())
        if trades_by_day.get(day_key, 0) >= cfg["max_trades_per_day"]:
            continue

        price = float(row["close"])
        atr = float(row["atr_calc"])
        vwap = float(row["vwap_calc"])

        if watch is None:
            candidates = route_engine.evaluate_routes(
                price=price,
                vwap=vwap,
                atr=atr,
                high_of_day=float(row["hod"]),
                low_of_day=float(row["lod"]),
                trend_slope=float(row["trend_slope"]),
                volume_ratio=float(row["vol_ratio"]),
                price_5m_ago=_row_price(stock_df, i, 5),
                price_15m_ago=_row_price(stock_df, i, 15),
                option_liquidity=0.75,
                iv_percentile=0.5,
            )
            candidates = [c for c in candidates if c.ic_spread >= cfg["watch_min_ic_spread"] and c.ev_over_debit >= cfg["watch_min_ev_over_debit"]]
            if not candidates:
                continue
            c = candidates[0]
            watch = WatchTicket(
                symbol=symbol,
                route=c.route,
                side=TicketSide.CALL if c.side == "CALL" else TicketSide.PUT,
                timestamp_created=ts.to_pydatetime(),
                underlying_price_at_watch=price,
                vwap_at_watch=vwap,
                atr_at_watch=atr,
                route_score=c.score,
                ic_spread=c.ic_spread,
                expected_ev_over_debit=c.ev_over_debit,
                option_liquidity_score=0.75,
                expected_move_to_target=c.expected_move,
            )
            watch_high = float(row["high"])
            watch_low = float(row["low"])
            continue

        # Expire watches after 20 bars.
        age_minutes = (ts.to_pydatetime() - watch.timestamp_created).total_seconds() / 60.0
        if age_minutes > 20:
            watch = None
            continue
        watch_high = max(watch_high, float(row["high"]))
        watch_low = min(watch_low, float(row["low"]))
        favorable = price - watch.underlying_price_at_watch if watch.side == TicketSide.CALL else watch.underlying_price_at_watch - price
        mfe_velocity = favorable / atr
        conf = confirm_engine.check_confirmation(
            watch, price, vwap, atr, watch_high, watch_low,
            mfe_velocity=mfe_velocity,
            env_stress=0.10,
            route_score_now=watch.route_score,
            option_quote_valid=True,
            timestamp=ts.to_pydatetime(),
        )
        if not conf.confirmed:
            continue
        confirm_engine.apply_confirmation(watch, conf, ts.to_pydatetime())

        side = "CALL" if watch.side == TicketSide.CALL else "PUT"
        pair = chain.get_spread_contracts(symbol, price, side, dte_min=5, dte_max=14, strike_width=5.0)
        if not pair:
            watch = None
            continue
        long_contract, short_contract = pair
        key = (long_contract.symbol, short_contract.symbol)
        if key not in option_cache:
            try:
                option_cache[key] = _option_frame(option_client, key, start, end)
            except Exception:
                option_cache[key] = pd.DataFrame()
        joined = _aligned_option_values(option_cache[key], *key)
        if joined is None:
            watch = None
            continue

        # Strictly NEXT minute after confirmation.
        next_ts = ts + pd.Timedelta(minutes=1)
        entry_pos = _first_index_at_or_after(joined.index, next_ts)
        if entry_pos is None:
            watch = None
            continue
        entry_time = joined.index[entry_pos]
        raw_debit = float(joined["long_close"].iloc[entry_pos] - joined["short_close"].iloc[entry_pos])
        if raw_debit <= 0:
            watch = None
            continue
        entry_debit = raw_debit * (1.0 + entry_slippage)
        per_contract = entry_debit * 100.0
        budget = equity * size_mult * cfg["max_open_debit_exposure_pct"]
        qty = max(0, min(5, int(budget / per_contract)))
        if qty <= 0:
            watch = None
            continue

        pwin = _estimate_pwin(symbol, watch.route_score, watch.mfe_velocity)
        package_ok = watch.route in cfg["routes_allowed"] and pwin >= cfg["package_min_pwin"] and qty >= 2
        mode = "PACKAGE" if package_ok else "FALLBACK"
        if mode == "PACKAGE":
            core_qty = max(1, round(qty * cfg["core_fraction"]))
            runner_qty = max(1, qty - core_qty)
            if core_qty + runner_qty > qty:
                core_qty = qty - runner_qty
        else:
            core_qty, runner_qty = 0, 0

        exit_pos, core_credit, runner_credit, pnl, exit_reason = _exit_trade(
            cfg, stock_df, joined, entry_pos, entry_debit, mode, qty,
            core_qty, runner_qty, side, max_hold=60,
            exit_haircut=exit_haircut,
        )
        exit_time = joined.index[exit_pos]
        equity += pnl
        total_debit = per_contract * qty
        trades.append(ValidationTrade(
            symbol=symbol,
            route=watch.route,
            side=side,
            mode=mode,
            watch_time=watch.timestamp_created.isoformat(),
            confirm_time=watch.timestamp_confirmed.isoformat(),
            entry_time=entry_time.isoformat(),
            exit_time=exit_time.isoformat(),
            long_contract=long_contract.symbol,
            short_contract=short_contract.symbol,
            entry_debit=entry_debit,
            exit_credit_core=core_credit,
            exit_credit_runner=runner_credit,
            quantity=qty,
            pnl=pnl,
            pnl_pct_on_debit=pnl / total_debit if total_debit > 0 else 0.0,
            winner=pnl > 0,
            exit_reason=exit_reason,
        ))
        trades_by_day[day_key] = trades_by_day.get(day_key, 0) + 1
        next_free_time = exit_time
        watch = None

    wins = sum(t.winner for t in trades)
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    result = {
        "symbol": symbol,
        "validation_type": "REAL_OPTION_BARS_WITH_FILL_STRESS_NOT_QUOTES",
        "days_requested": days,
        "stock_bars": len(stock_df),
        "trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": wins / len(trades) if trades else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0),
        "net_pnl": sum(t.pnl for t in trades),
        "starting_equity": starting_equity,
        "ending_equity": equity,
        "return_pct": equity / starting_equity - 1.0,
        "entry_slippage": entry_slippage,
        "exit_haircut": exit_haircut,
        "size_multiplier": size_mult,
        "limitations": [
            "historical option bars do not contain bid/ask queue/fill data",
            "current contract master is used for recent historical window",
            "IEX underlying bars are not full SIP market coverage",
        ],
    }
    return result, trades


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="COIN", choices=["COIN", "TSLA"])
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--starting-equity", type=float, default=1000.0)
    p.add_argument("--entry-slippage", type=float, default=0.025)
    p.add_argument("--exit-haircut", type=float, default=0.025)
    args = p.parse_args()

    result, trades = validate(args.symbol, args.days, args.starting_equity, args.entry_slippage, args.exit_haircut)
    out = Path("HFT/logs/v14_3_options/validation")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.symbol.lower()}_recent_option_bar_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out / f"{args.symbol.lower()}_recent_option_bar_trades.csv", "w", newline="") as f:
        if trades:
            w = csv.DictWriter(f, fieldnames=list(asdict(trades[0]).keys()))
            w.writeheader()
            for t in trades:
                w.writerow(asdict(t))

    print(json.dumps(result, indent=2))
    if result["trades"] < 5:
        print("WARNING: fewer than 5 option-bar trades; win rate is not decision-grade.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
