#!/usr/bin/env python3
"""
Continuous futures market scanner with GUI updates (BUY/SELL/HOLD) using:
- VWAP
- EMA 9 vs EMA 21
- RSI 14

This script is informational only and does not place trades.
"""

from __future__ import annotations

import csv
import json
import logging
import signal
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.colors import is_color_like
    from matplotlib.patches import Rectangle
    MATPLOTLIB_CORE_OK = True
except Exception:
    mdates = None
    plt = None
    is_color_like = None
    Rectangle = None
    MATPLOTLIB_CORE_OK = False

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # Some headless IDEs may not support GUI.
    tk = None
    ttk = None

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_TK_OK = MATPLOTLIB_CORE_OK
except Exception:
    FigureCanvasTkAgg = None
    MATPLOTLIB_TK_OK = False


@dataclass
class Config:
    timeframe: str = "5m"  # Supported: 1m, 5m, 15m, 1h
    scan_interval_seconds: int = 1
    min_bars_required: int = 50
    sr_lookback_bars: int = 50
    target_notional_per_trade: float = 25000.0
    csv_log_file: str = "futures_scans.csv"
    error_log_file: str = "futures_scanner_errors.log"
    paper_state_file: str = "paper_trading_state.json"


@dataclass
class PaperSizingSettings:
    mode: str = "Fixed Dollar"  # Fixed Dollar | Percent of Account | Fixed Contracts
    fixed_dollar: float = 1000.0
    percent_of_account: float = 2.0
    fixed_contracts: int = 1


@dataclass
class PaperPosition:
    symbol: str
    side: str  # LONG | SHORT
    entry_price: float
    current_price: float
    quantity: int
    position_size: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    trade_mode: str  # Manual | Auto
    open_time: str
    stop_loss: float = np.nan


@dataclass
class PaperClosedTrade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    realized_pnl: float
    realized_pnl_pct: float
    open_time: str
    close_time: str
    trade_mode: str


@dataclass
class PaperAccountState:
    starting_balance: float = 50000.0
    current_balance: float = 50000.0
    available_buying_power: float = 50000.0
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0


CONFIG = Config()

# Mini equity index futures only.
PRIMARY_SYMBOL_MAP: Dict[str, str] = {
    "ES": "ES=F",   # E-mini S&P 500
    "NQ": "NQ=F",   # E-mini Nasdaq 100
    "YM": "YM=F",   # E-mini Dow
    "RTY": "RTY=F", # E-mini Russell 2000
}

FALLBACK_SYMBOL_MAP: Dict[str, str] = {
    "ES": "SPY",
    "NQ": "QQQ",
    "YM": "DIA",
    "RTY": "IWM",
}

TIMEFRAME_MAP: Dict[str, Tuple[str, str]] = {
    "1m": ("1m", "2d"),
    "5m": ("5m", "7d"),
    "15m": ("15m", "30d"),
    "30m": ("30m", "60d"),
    "60m": ("60m", "60d"),
    "1h": ("60m", "60d"),
}

SHUTDOWN_REQUESTED = False
SCAN_SYMBOLS: List[str] = list(PRIMARY_SYMBOL_MAP.keys())
CUSTOM_SYMBOL_MAP: Dict[str, str] = {}
SCAN_SYMBOLS_LOCK = threading.Lock()
ACTIVE_SCAN_TIMEFRAME = CONFIG.timeframe
ACTIVE_SCAN_TIMEFRAME_LOCK = threading.Lock()
CHART_INTERVAL_MAX_DAYS: Dict[str, int] = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
}


def setup_logging() -> None:
    logging.basicConfig(
        filename=CONFIG.error_log_file,
        level=logging.ERROR,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def handle_signal(signum: int, _frame) -> None:
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print(f"\nReceived signal {signum}. Shutting down gracefully...")


def validate_config() -> None:
    if CONFIG.timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe '{CONFIG.timeframe}'. Use one of: {', '.join(TIMEFRAME_MAP)}")
    if CONFIG.scan_interval_seconds <= 0:
        raise ValueError("scan_interval_seconds must be greater than 0")


def get_scan_symbols() -> List[str]:
    with SCAN_SYMBOLS_LOCK:
        return list(SCAN_SYMBOLS)


def get_active_scan_timeframe() -> str:
    with ACTIVE_SCAN_TIMEFRAME_LOCK:
        return ACTIVE_SCAN_TIMEFRAME


def set_active_scan_timeframe(timeframe: str) -> None:
    if timeframe not in TIMEFRAME_MAP:
        return
    global ACTIVE_SCAN_TIMEFRAME
    with ACTIVE_SCAN_TIMEFRAME_LOCK:
        ACTIVE_SCAN_TIMEFRAME = timeframe


def add_custom_symbol(symbol: str) -> Tuple[bool, str]:
    symbol = symbol.strip().upper()
    if not symbol:
        return False, "Ticker cannot be empty."

    with SCAN_SYMBOLS_LOCK:
        if symbol in SCAN_SYMBOLS:
            return False, f"{symbol} is already in the scan list."
        CUSTOM_SYMBOL_MAP[symbol] = symbol
        SCAN_SYMBOLS.append(symbol)

    return True, f"Added {symbol} to scan list."


def remove_symbol(symbol: str) -> Tuple[bool, str]:
    symbol = symbol.strip().upper()
    if not symbol:
        return False, "Ticker cannot be empty."

    with SCAN_SYMBOLS_LOCK:
        if symbol not in SCAN_SYMBOLS:
            return False, f"{symbol} is not in the scan list."
        SCAN_SYMBOLS.remove(symbol)
        CUSTOM_SYMBOL_MAP.pop(symbol, None)

    return True, f"Removed {symbol} from scan list."


def _download_symbol(symbol: str) -> pd.DataFrame:
    interval, period = TIMEFRAME_MAP[get_active_scan_timeframe()]
    ticker = yf.Ticker(symbol)
    # prepost=True helps include extended-hours data where available.
    df = ticker.history(interval=interval, period=period, auto_adjust=False, prepost=True)

    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol}")

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing OHLCV columns for {symbol}")

    df = df.dropna(subset=["High", "Low", "Close", "Volume"]).copy()
    if len(df) < CONFIG.min_bars_required:
        raise ValueError(f"Insufficient bars for {symbol}: {len(df)}")

    return df


def fetch_market_data(symbol: str) -> Tuple[pd.DataFrame, str, str]:
    if symbol in CUSTOM_SYMBOL_MAP:
        custom_ticker = CUSTOM_SYMBOL_MAP[symbol]
        return _download_symbol(custom_ticker), custom_ticker, "CUSTOM"

    primary = PRIMARY_SYMBOL_MAP[symbol]
    fallback = FALLBACK_SYMBOL_MAP[symbol]

    try:
        return _download_symbol(primary), primary, "FUTURES"
    except Exception as primary_error:
        logging.error("Primary symbol failed for %s (%s): %s", symbol, primary, primary_error)

    return _download_symbol(fallback), fallback, "FALLBACK"


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    volume = df["Volume"].replace(0, np.nan)
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_trendline(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    x = np.arange(len(series), dtype=float)
    y = series.astype(float).values
    slope, intercept = np.polyfit(x, y, 1)
    trend = intercept + (slope * x)
    return pd.Series(trend, index=series.index)


def get_short_term_support_resistance(df: pd.DataFrame, lookback: int = 15) -> Tuple[float, float]:
    recent = df.tail(lookback + 1)
    if len(recent) < 3:
        return float(df["Low"].iloc[-1]), float(df["High"].iloc[-1])
    short_support = float(recent["Low"].iloc[:-1].min())
    short_resistance = float(recent["High"].iloc[:-1].max())
    return short_support, short_resistance


def detect_breakout(
    last_price: float,
    short_resistance: float,
    last_vwap: float,
    last_ema9: float,
    last_ema21: float,
    last_rsi: float,
) -> bool:
    # Breakout logic added: bullish break above short-term resistance with confirmation.
    return (
        last_price > short_resistance
        and last_price > last_vwap
        and last_ema9 > last_ema21
        and last_rsi >= 52
    )


def detect_breakdown(
    last_price: float,
    short_support: float,
    last_vwap: float,
    last_ema9: float,
    last_ema21: float,
    last_rsi: float,
) -> bool:
    # Breakout logic added: bearish break below short-term support with confirmation.
    return (
        last_price < short_support
        and last_price < last_vwap
        and last_ema9 < last_ema21
        and last_rsi <= 48
    )


def detect_bullish_continuation(close: pd.Series, last_vwap: float, last_ema9: float, last_ema21: float) -> bool:
    # Continuation logic added: pullback then push higher in a bullish structure.
    if len(close) < 5:
        return False
    c = close.tail(5)
    pullback_then_resume = c.iloc[-3] > c.iloc[-2] and c.iloc[-1] > c.iloc[-2] and c.iloc[-1] > c.iloc[-4]
    return bool(pullback_then_resume and c.iloc[-1] > last_vwap and last_ema9 > last_ema21)


def detect_bearish_continuation(close: pd.Series, last_vwap: float, last_ema9: float, last_ema21: float) -> bool:
    # Continuation logic added: pullback then continuation lower in a bearish structure.
    if len(close) < 5:
        return False
    c = close.tail(5)
    pullback_then_resume = c.iloc[-3] < c.iloc[-2] and c.iloc[-1] < c.iloc[-2] and c.iloc[-1] < c.iloc[-4]
    return bool(pullback_then_resume and c.iloc[-1] < last_vwap and last_ema9 < last_ema21)


def calculate_bar_signals(df: pd.DataFrame, lookback: int = 30) -> pd.Series:
    close = df["Close"]
    vwap = calculate_vwap(df)
    ema9 = calculate_ema(close, 9)
    ema21 = calculate_ema(close, 21)
    rsi14 = calculate_rsi(close, 14)
    rolling_support = df["Low"].rolling(lookback, min_periods=5).min()
    rolling_resistance = df["High"].rolling(lookback, min_periods=5).max()
    short_support = df["Low"].rolling(15, min_periods=5).min().shift(1)
    short_resistance = df["High"].rolling(15, min_periods=5).max().shift(1)
    rolling_range = (rolling_resistance - rolling_support).replace(0, np.nan)
    dip_zone = close <= (rolling_support + (0.30 * rolling_range))
    rally_zone = close >= (rolling_resistance - (0.30 * rolling_range))

    bullish = ((close > vwap).astype(int) + (ema9 > ema21).astype(int) + (rsi14 > 50).astype(int))
    bearish = ((close < vwap).astype(int) + (ema9 < ema21).astype(int) + (rsi14 < 50).astype(int))

    signal = pd.Series("HOLD", index=df.index, dtype=object)
    # Chart markers updated: include breakout/breakdown and continuation setups.
    breakout_buy = (close > short_resistance) & (close > vwap) & (ema9 > ema21) & (rsi14 >= 52)
    breakdown_sell = (close < short_support) & (close < vwap) & (ema9 < ema21) & (rsi14 <= 48)
    cont_buy = (
        (bullish > bearish)
        & (close > vwap)
        & (ema9 > ema21)
        & (close.shift(3) > close.shift(2))
        & (close.shift(1) < close.shift(2))
        & (close > close.shift(1))
    )
    cont_sell = (
        (bearish > bullish)
        & (close < vwap)
        & (ema9 < ema21)
        & (close.shift(3) < close.shift(2))
        & (close.shift(1) > close.shift(2))
        & (close < close.shift(1))
    )
    mr_buy = (bullish > bearish) & dip_zone & (rsi14 < 62)
    mr_sell = (bearish > bullish) & rally_zone & (rsi14 > 38)

    signal[mr_buy | breakout_buy | cont_buy] = "BUY"
    signal[mr_sell | breakdown_sell | cont_sell] = "SELL"
    return signal


def enforce_right_anchor(ax, latest_x: float) -> None:
    """
    Keep the right edge fixed at the latest candle while preserving current X-range.
    This makes zoom/pan behave like a trading platform where new bars stay visible.
    """
    x_min, x_max = ax.get_xlim()
    width = x_max - x_min
    if width <= 0:
        return
    ax.set_xlim(latest_x - width, latest_x)


def evaluate_signal(df: pd.DataFrame) -> Dict[str, object]:
    close = df["Close"]
    vwap = calculate_vwap(df)
    ema9 = calculate_ema(close, 9)
    ema21 = calculate_ema(close, 21)
    rsi14 = calculate_rsi(close, 14)

    last_price = float(close.iloc[-1])
    last_vwap = float(vwap.iloc[-1])
    last_ema9 = float(ema9.iloc[-1])
    last_ema21 = float(ema21.iloc[-1])
    last_rsi = float(rsi14.iloc[-1])

    bullish_points = 0
    bearish_points = 0
    reasons: List[str] = []

    if last_price > last_vwap:
        bullish_points += 1
        vwap_status = "Price above VWAP"
        reasons.append("above VWAP")
    elif last_price < last_vwap:
        bearish_points += 1
        vwap_status = "Price below VWAP"
        reasons.append("below VWAP")
    else:
        vwap_status = "Price at VWAP"
        reasons.append("at VWAP")

    if last_ema9 > last_ema21:
        bullish_points += 1
        ema_status = "EMA9 above EMA21"
        reasons.append("EMA9>EMA21")
    elif last_ema9 < last_ema21:
        bearish_points += 1
        ema_status = "EMA9 below EMA21"
        reasons.append("EMA9<EMA21")
    else:
        ema_status = "EMA9 equal EMA21"
        reasons.append("EMA9=EMA21")

    if last_rsi > 50:
        bullish_points += 1
        reasons.append(f"RSI {last_rsi:.2f} > 50")
    elif last_rsi < 50:
        bearish_points += 1
        reasons.append(f"RSI {last_rsi:.2f} < 50")
    else:
        reasons.append("RSI=50")

    lookback_df = df.tail(CONFIG.sr_lookback_bars)
    support = float(lookback_df["Low"].min())
    resistance = float(lookback_df["High"].max())
    short_support, short_resistance = get_short_term_support_resistance(df, lookback=15)
    sr_range = max(resistance - support, 0.0001)
    dip_zone_limit = support + (0.30 * sr_range)
    rally_zone_limit = resistance - (0.30 * sr_range)
    in_dip_zone = last_price <= dip_zone_limit
    in_rally_zone = last_price >= rally_zone_limit

    is_breakout_buy = detect_breakout(last_price, short_resistance, last_vwap, last_ema9, last_ema21, last_rsi)
    is_breakdown_sell = detect_breakdown(last_price, short_support, last_vwap, last_ema9, last_ema21, last_rsi)
    is_cont_buy = (
        bullish_points > bearish_points
        and last_price > last_vwap
        and last_ema9 > last_ema21
        and detect_bullish_continuation(close, last_vwap, last_ema9, last_ema21)
    )
    is_cont_sell = (
        bearish_points > bullish_points
        and last_price < last_vwap
        and last_ema9 < last_ema21
        and detect_bearish_continuation(close, last_vwap, last_ema9, last_ema21)
    )
    is_mr_buy = bullish_points > bearish_points and in_dip_zone and last_rsi < 62
    is_mr_sell = bearish_points > bullish_points and in_rally_zone and last_rsi > 38
    strong_trend = (bullish_points >= 2 and last_price > last_vwap and last_ema9 > last_ema21) or (
        bearish_points >= 2 and last_price < last_vwap and last_ema9 < last_ema21
    )
    recent_swing_low = float(df["Low"].tail(5).min())
    recent_swing_high = float(df["High"].tail(5).max())

    # New signal-to-action mapping added: mirror the latest chart bar signal in table output.
    last_bar_signal = calculate_bar_signals(df).iloc[-1]

    # Old logic is bypassed when new signals exist (BUY/SELL from bar signals).
    if last_bar_signal == "BUY":
        signal = "BUY"
        direction = "UPTREND"
        recommended_action = "BUY"
        entry = last_price
        stop_loss = min(short_support, recent_swing_low) - (0.05 * sr_range)
        risk = max(entry - stop_loss, 0.0001)
        take_profit = max(short_resistance, resistance, entry + (1.8 * risk))
        if is_breakout_buy:
            trade_mode = "Breakout"
        elif is_cont_buy:
            trade_mode = "Continuation"
        else:
            trade_mode = "Continuation"
        reasons.append("bullish continuation or breakout signal")
    elif last_bar_signal == "SELL":
        signal = "SELL"
        direction = "DOWNTREND"
        recommended_action = "SELL/SHORT"
        entry = last_price
        stop_loss = max(short_resistance, recent_swing_high) + (0.05 * sr_range)
        risk = max(stop_loss - entry, 0.0001)
        take_profit = min(short_support, support, entry - (1.8 * risk))
        if is_breakdown_sell:
            trade_mode = "Breakout"
        elif is_cont_sell:
            trade_mode = "Continuation"
        else:
            trade_mode = "Continuation"
        reasons.append("bearish continuation or breakdown signal")

    elif is_breakout_buy:
        signal = "BUY"
        direction = "UPTREND"
        trade_mode = "Breakout"
        recommended_action = "BUY"
        entry = last_price
        stop_loss = short_resistance - (0.10 * sr_range)
        take_profit = resistance + (0.70 * sr_range)
        reasons.append("bullish breakout above short-term resistance")
    elif is_breakdown_sell:
        signal = "SELL"
        direction = "DOWNTREND"
        trade_mode = "Breakout"
        recommended_action = "SELL/SHORT"
        entry = last_price
        stop_loss = short_support + (0.10 * sr_range)
        take_profit = support - (0.70 * sr_range)
        reasons.append("bearish breakdown below short-term support")
    elif is_mr_buy:
        signal = "BUY"
        direction = "UPTREND"
        trade_mode = "Mean Reversion"
        recommended_action = "BUY"
        entry = max(last_price, support + (0.20 * sr_range))
        stop_loss = support - (0.10 * sr_range)
        take_profit = resistance + (0.50 * sr_range)
        reasons.append("buy-the-dip zone")
    elif is_mr_sell:
        signal = "SELL"
        direction = "DOWNTREND"
        trade_mode = "Mean Reversion"
        recommended_action = "SELL/SHORT"
        entry = min(last_price, resistance - (0.20 * sr_range))
        stop_loss = resistance + (0.10 * sr_range)
        take_profit = support - (0.50 * sr_range)
        reasons.append("sell-the-rally zone")
    elif is_cont_buy:
        signal = "BUY"
        direction = "UPTREND"
        trade_mode = "Continuation"
        recommended_action = "BUY"
        entry = last_price
        stop_loss = float(df["Low"].tail(5).min()) - (0.05 * sr_range)
        take_profit = resistance + (0.40 * sr_range)
        reasons.append("bullish continuation above VWAP with EMA9>EMA21")
    elif is_cont_sell:
        signal = "SELL"
        direction = "DOWNTREND"
        trade_mode = "Continuation"
        recommended_action = "SELL/SHORT"
        entry = last_price
        stop_loss = float(df["High"].tail(5).max()) + (0.05 * sr_range)
        take_profit = support - (0.40 * sr_range)
        reasons.append("bearish continuation below VWAP with EMA9<EMA21")
    else:
        signal = "HOLD"
        direction = "MIXED"
        trade_mode = "Hold / No Trade"
        recommended_action = "DO NOT BUY"
        entry = np.nan
        stop_loss = np.nan
        take_profit = np.nan
        # No-trade-zone logic refined: only mark mid-range if truly inside with no directional setup.
        if last_bar_signal == "HOLD" and not in_dip_zone and not in_rally_zone and not strong_trend:
            reasons.append("mid-range (no trade zone)")
        else:
            reasons.append("setup not strong enough at edge")

    estimated_next_buy = support + (0.15 * sr_range)
    estimated_next_short = resistance - (0.15 * sr_range)
    if signal == "BUY" and not np.isnan(entry):
        estimated_next_buy = min(float(entry), estimated_next_buy)
    if signal == "SELL" and not np.isnan(entry):
        estimated_next_short = max(float(entry), estimated_next_short)

    return {
        "last_price": last_price,
        "vwap_status": vwap_status,
        "ema_status": ema_status,
        "rsi_value": last_rsi,
        "direction": direction,
        "signal": signal,
        "support": support,
        "resistance": resistance,
        "recommended_action": recommended_action,
        "trade_mode": trade_mode,
        "entry": entry,
        "estimated_next_buy": estimated_next_buy,
        "estimated_next_short": estimated_next_short,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reason": f"Bullish={bullish_points}, Bearish={bearish_points}; " + ", ".join(reasons),
    }


def ensure_csv_header(path: Path) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_utc",
                "symbol",
                "data_symbol",
                "source_type",
                "last_price",
                "vwap_status",
                "ema_status",
                "rsi_value",
                "support",
                "resistance",
                "trend",
                "signal",
                "recommended_action",
                "trade_mode",
                "entry",
                "estimated_next_buy",
                "estimated_next_short",
                "stop_loss",
                "take_profit",
                "reason",
            ])


def save_scan_row(
    path: Path,
    timestamp_utc: str,
    symbol: str,
    data_symbol: str,
    source_type: str,
    result: Dict[str, object],
) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp_utc,
            symbol,
            data_symbol,
            source_type,
            f"{result['last_price']:.6f}",
            result["vwap_status"],
            result["ema_status"],
            f"{result['rsi_value']:.2f}",
            f"{result['support']:.6f}",
            f"{result['resistance']:.6f}",
            result["direction"],
            result["signal"],
            result["recommended_action"],
            result["trade_mode"],
            f"{result['entry']:.6f}" if not np.isnan(result["entry"]) else "",
            f"{result['estimated_next_buy']:.6f}",
            f"{result['estimated_next_short']:.6f}",
            f"{result['stop_loss']:.6f}" if not np.isnan(result["stop_loss"]) else "",
            f"{result['take_profit']:.6f}" if not np.isnan(result["take_profit"]) else "",
            result["reason"],
        ])


def run_single_scan(csv_path: Path, timestamp_utc: str) -> Tuple[List[Dict[str, str]], Dict[str, pd.DataFrame]]:
    rows: List[Dict[str, str]] = []
    latest_frames: Dict[str, pd.DataFrame] = {}
    for symbol in get_scan_symbols():
        try:
            df, data_symbol, source_type = fetch_market_data(symbol)
            result = evaluate_signal(df)
            save_scan_row(csv_path, timestamp_utc, symbol, data_symbol, source_type, result)
            latest_frames[symbol] = df.tail(120).copy()
            rows.append({
                "symbol": symbol,
                "data": f"{source_type}:{data_symbol}",
                "last_price": f"{result['last_price']:.2f}",
                "vwap": str(result["vwap_status"]),
                "ema": str(result["ema_status"]),
                "rsi": f"{result['rsi_value']:.2f}",
                "support": f"{result['support']:.2f}",
                "resistance": f"{result['resistance']:.2f}",
                "direction": str(result["direction"]),
                "signal": str(result["signal"]),
                "action": str(result["recommended_action"]),
                "trade_mode": str(result["trade_mode"]),
                "entry": "-" if np.isnan(result["entry"]) else f"{result['entry']:.2f}",
                "estimated_next_buy": f"{result['estimated_next_buy']:.2f}",
                "estimated_next_short": f"{result['estimated_next_short']:.2f}",
                "stop_loss": "-" if np.isnan(result["stop_loss"]) else f"{result['stop_loss']:.2f}",
                "take_profit": "-" if np.isnan(result["take_profit"]) else f"{result['take_profit']:.2f}",
                "reason": str(result["reason"]),
            })
        except Exception as exc:
            err = f"{symbol} failed on primary+fallback symbols: {exc}"
            logging.error(err, exc_info=True)
            rows.append({
                "symbol": symbol,
                "data": "ERROR",
                "last_price": "-",
                "vwap": "ERROR",
                "ema": "ERROR",
                "rsi": "-",
                "support": "-",
                "resistance": "-",
                "direction": "MIXED",
                "signal": "HOLD",
                "action": "DO NOT BUY",
                "trade_mode": "Hold / No Trade",
                "entry": "-",
                "estimated_next_buy": "-",
                "estimated_next_short": "-",
                "stop_loss": "-",
                "take_profit": "-",
                "reason": err,
            })
    return rows, latest_frames


def run_console_scanner() -> None:
    csv_path = Path(CONFIG.csv_log_file)
    ensure_csv_header(csv_path)
    print("Starting console scanner (Ctrl+C to stop).")

    while not SHUTDOWN_REQUESTED:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print("\n" + "=" * 160)
        print(f"FUTURES SCAN | {stamp} | timeframe={get_active_scan_timeframe()} | interval={CONFIG.scan_interval_seconds}s")
        print("=" * 160)
        print(
            f"{'Symbol':<7} {'Data':<14} {'Last':>10} {'Support':>10} {'Resistance':>11} "
            f"{'Action':<12} {'Mode':<26} {'Entry':>10} {'BuyDip':>10} {'ShortSpike':>10} {'Stop':>10} {'Target':>10} {'Signal':<7} Reason"
        )
        print("-" * 200)

        rows, _latest_frames = run_single_scan(csv_path, stamp)
        for row in rows:
            print(
                f"{row['symbol']:<7} {row['data']:<14} {row['last_price']:>10} {row['support']:>10} {row['resistance']:>11} "
                f"{row['action']:<12} {row['trade_mode']:<26} {row['entry']:>10} {row['estimated_next_buy']:>10} {row['estimated_next_short']:>10} {row['stop_loss']:>10} {row['take_profit']:>10} "
                f"{row['signal']:<7} {row['reason']}"
            )

        print("-" * 200)
        print(f"Next scan in {CONFIG.scan_interval_seconds} seconds...")
        slept = 0
        while slept < CONFIG.scan_interval_seconds and not SHUTDOWN_REQUESTED:
            time.sleep(1)
            slept += 1


class FuturesScannerGUI:
    def __init__(self) -> None:
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter GUI is not available in this environment.")

        self.csv_path = Path(CONFIG.csv_log_file)
        ensure_csv_header(self.csv_path)

        self.queue: Queue = Queue()
        self.root = tk.Tk()
        self.root.title("Futures Scanner - BUY/SELL/HOLD")
        # Window sizing improved: start much larger and allow full-screen-style maximize.
        self.root.geometry("2400x1400")
        self.root.minsize(1600, 950)
        self.root.resizable(True, True)
        try:
            self.root.state("zoomed")
        except Exception:
            try:
                self.root.attributes("-zoomed", True)
            except Exception:
                pass

        self.status_var = tk.StringVar(value="Starting scanner...")
        self.info_var = tk.StringVar()
        # Paper trading feature state (paper only, no real broker execution).
        self.paper_enabled_var = tk.BooleanVar(value=False)
        self.paper_auto_trade_var = tk.BooleanVar(value=False)
        self.paper_auto_close_opposite_var = tk.BooleanVar(value=False)
        self.paper_account = PaperAccountState()
        self.paper_sizing = PaperSizingSettings()
        self.paper_open_positions: Dict[str, PaperPosition] = {}
        self.paper_closed_trades: List[PaperClosedTrade] = []
        self.paper_state_path = Path(CONFIG.paper_state_file)
        self.selected_symbol: str = ""
        self.latest_frames: Dict[str, pd.DataFrame] = {}
        self.last_signal_by_symbol: Dict[str, str] = {}
        self.recent_buy_signals_by_symbol: Dict[str, deque] = {}
        self.recent_short_signals_by_symbol: Dict[str, deque] = {}
        self.chart_period_var = tk.StringVar(value="90d")
        self.chart_interval_var = tk.StringVar(value="60m")
        self.sync_scan_timeframe_with_chart()
        self.starting_balance_var = tk.StringVar(value=f"{self.paper_account.starting_balance:.2f}")
        self.fixed_dollar_var = tk.StringVar(value=f"{self.paper_sizing.fixed_dollar:.2f}")
        self.percent_account_var = tk.StringVar(value=f"{self.paper_sizing.percent_of_account:.2f}")
        self.fixed_contracts_var = tk.StringVar(value=str(self.paper_sizing.fixed_contracts))
        self.sizing_mode_var = tk.StringVar(value=self.paper_sizing.mode)
        self.account_balance_var = tk.StringVar(value="0.00")
        self.account_buying_power_var = tk.StringVar(value="0.00")
        self.account_realized_var = tk.StringVar(value="0.00")
        self.account_unrealized_var = tk.StringVar(value="0.00")
        self.user_zoom_active = False
        self.latest_x_right: float | None = None
        self.full_x_min: float | None = None
        self.full_x_max: float | None = None
        self.chart_colors = {
            "bull_candle": "green",
            "bear_candle": "red",
            "vwap": "cyan",
            "ema9": "orange",
            "ema21": "magenta",
            "buy": "limegreen",
            "sell": "crimson",
            "current_price": "deepskyblue",
        }
        self.color_vars = {k: tk.StringVar(value=v) for k, v in self.chart_colors.items()}
        self.common_colors = (
            "red", "green", "blue", "orange", "purple", "cyan", "magenta",
            "yellow", "white", "black", "gray", "gold", "pink", "brown",
            "navy", "teal", "lime", "maroon", "olive", "deepskyblue",
            "crimson", "limegreen",
        )

        info = tk.Label(
            self.root,
            textvariable=self.info_var,
            font=("Arial", 11, "bold"),
        )
        info.pack(pady=6)

        # Resizable layout improved: add vertical PanedWindow so table/paper section and
        # chart section can be drag-resized, with chart getting dominant space by default.
        self.main_split = ttk.Panedwindow(self.root, orient="vertical")
        self.main_split.pack(fill="both", expand=True, padx=6, pady=4)
        self.top_panel = tk.Frame(self.main_split)
        self.chart_panel = tk.Frame(self.main_split)
        self.main_split.add(self.top_panel, weight=3)
        self.main_split.add(self.chart_panel, weight=7)

        add_frame = tk.Frame(self.top_panel)
        add_frame.pack(fill="x", padx=8, pady=4)
        tk.Label(add_frame, text="Add ticker:").pack(side="left")
        self.add_symbol_var = tk.StringVar()
        add_entry = tk.Entry(add_frame, textvariable=self.add_symbol_var, width=16)
        add_entry.pack(side="left", padx=6)
        add_entry.bind("<Return>", lambda _event: self.add_ticker_from_input())
        tk.Button(add_frame, text="Add", command=self.add_ticker_from_input).pack(side="left")
        tk.Button(add_frame, text="Remove Typed", command=self.remove_ticker_from_input).pack(side="left", padx=6)
        tk.Button(add_frame, text="Remove Selected", command=self.remove_selected_ticker).pack(side="left")

        color_frame = tk.Frame(self.top_panel)
        color_frame.pack(fill="x", padx=8, pady=2)
        tk.Label(color_frame, text="Chart Colors:").pack(side="left")
        for key, label in (
            ("bull_candle", "Bull"),
            ("bear_candle", "Bear"),
            ("buy", "Buy"),
            ("sell", "Sell"),
            ("vwap", "VWAP"),
            ("ema9", "EMA9"),
            ("ema21", "EMA21"),
        ):
            tk.Label(color_frame, text=label).pack(side="left", padx=(8, 2))
            ttk.Combobox(
                color_frame,
                textvariable=self.color_vars[key],
                values=self.common_colors,
                width=10,
            ).pack(side="left")
        tk.Button(color_frame, text="Apply Colors", command=self.apply_chart_colors).pack(side="left", padx=8)

        columns = (
            "symbol",
            "data",
            "last_price",
            "support",
            "resistance",
            "action",
            "trade_mode",
            "entry",
            "estimated_next_buy",
            "estimated_next_short",
            "stop_loss",
            "take_profit",
            "signal",
            "reason",
        )
        self.tree = ttk.Treeview(self.top_panel, columns=columns, show="headings", height=12)

        headings = {
            "symbol": "Symbol",
            "data": "Data",
            "last_price": "Last Price",
            "support": "Support",
            "resistance": "Resistance",
            "action": "Action",
            "trade_mode": "Trade Mode",
            "entry": "Entry",
            "estimated_next_buy": "Buy Dip",
            "estimated_next_short": "Short Spike",
            "stop_loss": "Stop Loss",
            "take_profit": "Take Profit",
            "signal": "Signal",
            "reason": "Reason",
        }
        widths = {
            "symbol": 70,
            "data": 120,
            "last_price": 95,
            "support": 95,
            "resistance": 95,
            "action": 120,
            "trade_mode": 170,
            "entry": 90,
            "estimated_next_buy": 95,
            "estimated_next_short": 95,
            "stop_loss": 90,
            "take_profit": 90,
            "signal": 80,
            "reason": 720,
        }

        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor="w")

        self.tree.pack(fill="both", expand=True, padx=8, pady=4)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        recent_frame = tk.Frame(self.top_panel)
        recent_frame.pack(fill="x", padx=8, pady=(0, 4))
        buy_box = tk.LabelFrame(recent_frame, text="Recent BUY Signals (Last 5)")
        buy_box.pack(side="left", fill="both", expand=True, padx=(0, 4))
        short_box = tk.LabelFrame(recent_frame, text="Recent SHORT Signals (Last 5)")
        short_box.pack(side="left", fill="both", expand=True, padx=(4, 0))

        recent_cols = ("time", "symbol", "entry", "stop", "target")
        self.buy_tree = ttk.Treeview(buy_box, columns=recent_cols, show="headings", height=5)
        self.short_tree = ttk.Treeview(short_box, columns=recent_cols, show="headings", height=5)
        for tree in (self.buy_tree, self.short_tree):
            tree.heading("time", text="Time")
            tree.heading("symbol", text="Symbol")
            tree.heading("entry", text="Entry")
            tree.heading("stop", text="Stop Loss")
            tree.heading("target", text="Take Profit")
            tree.column("time", width=170, anchor="w")
            tree.column("symbol", width=70, anchor="w")
            tree.column("entry", width=90, anchor="e")
            tree.column("stop", width=100, anchor="e")
            tree.column("target", width=100, anchor="e")
            tree.pack(fill="both", expand=True)

        # Paper trading controls/positions/history section (paper only, no live orders).
        paper_frame = tk.LabelFrame(self.top_panel, text="Paper Trading (Simulation Only - No Real Broker Orders)")
        paper_frame.pack(fill="x", padx=8, pady=(0, 4))
        controls_row = tk.Frame(paper_frame)
        controls_row.pack(fill="x", padx=4, pady=3)
        tk.Checkbutton(controls_row, text="Enable Paper Trading", variable=self.paper_enabled_var).pack(side="left")
        tk.Checkbutton(controls_row, text="Auto Trade", variable=self.paper_auto_trade_var).pack(side="left", padx=6)
        tk.Checkbutton(controls_row, text="Auto Close Opposite", variable=self.paper_auto_close_opposite_var).pack(side="left")
        tk.Button(controls_row, text="Manual BUY", command=lambda: self.manual_open_trade("LONG")).pack(side="left", padx=8)
        tk.Button(controls_row, text="Manual SELL/SHORT", command=lambda: self.manual_open_trade("SHORT")).pack(side="left")
        tk.Button(controls_row, text="Manual CLOSE POSITION", command=self.manual_close_trade).pack(side="left", padx=8)

        account_row = tk.Frame(paper_frame)
        account_row.pack(fill="x", padx=4, pady=3)
        tk.Label(account_row, text="Starting Balance:").pack(side="left")
        tk.Entry(account_row, textvariable=self.starting_balance_var, width=10).pack(side="left", padx=4)
        tk.Label(account_row, text="Sizing Mode:").pack(side="left", padx=(10, 0))
        ttk.Combobox(
            account_row,
            textvariable=self.sizing_mode_var,
            values=("Fixed Dollar", "Percent of Account", "Fixed Contracts"),
            state="readonly",
            width=16,
        ).pack(side="left", padx=4)
        tk.Label(account_row, text="Fixed $:").pack(side="left")
        tk.Entry(account_row, textvariable=self.fixed_dollar_var, width=8).pack(side="left", padx=2)
        tk.Label(account_row, text="% Acct:").pack(side="left")
        tk.Entry(account_row, textvariable=self.percent_account_var, width=6).pack(side="left", padx=2)
        tk.Label(account_row, text="Contracts:").pack(side="left")
        tk.Entry(account_row, textvariable=self.fixed_contracts_var, width=6).pack(side="left", padx=2)
        tk.Button(account_row, text="Apply Account Settings", command=self.apply_account_settings).pack(side="left", padx=8)
        tk.Button(account_row, text="Reset Account", command=self.reset_paper_account).pack(side="left")

        metrics_row = tk.Frame(paper_frame)
        metrics_row.pack(fill="x", padx=4, pady=3)
        tk.Label(metrics_row, text="Current Balance:").pack(side="left")
        tk.Label(metrics_row, textvariable=self.account_balance_var, fg="blue").pack(side="left", padx=4)
        tk.Label(metrics_row, text="Buying Power:").pack(side="left", padx=(10, 0))
        tk.Label(metrics_row, textvariable=self.account_buying_power_var, fg="blue").pack(side="left", padx=4)
        tk.Label(metrics_row, text="Realized P&L:").pack(side="left", padx=(10, 0))
        tk.Label(metrics_row, textvariable=self.account_realized_var, fg="blue").pack(side="left", padx=4)
        tk.Label(metrics_row, text="Unrealized P&L:").pack(side="left", padx=(10, 0))
        tk.Label(metrics_row, textvariable=self.account_unrealized_var, fg="blue").pack(side="left", padx=4)

        paper_tables_row = tk.Frame(paper_frame)
        paper_tables_row.pack(fill="x", padx=4, pady=3)
        open_box = tk.LabelFrame(paper_tables_row, text="Open Paper Positions")
        open_box.pack(side="left", fill="both", expand=True, padx=(0, 4))
        closed_box = tk.LabelFrame(paper_tables_row, text="Closed Paper Trades")
        closed_box.pack(side="left", fill="both", expand=True, padx=(4, 0))

        open_cols = ("symbol", "side", "entry", "current", "qty", "size", "upnl", "upnl_pct", "mode", "open_time")
        self.paper_open_tree = ttk.Treeview(open_box, columns=open_cols, show="headings", height=5)
        for c, t, w in (
            ("symbol", "Symbol", 70), ("side", "Side", 60), ("entry", "Entry", 80), ("current", "Current", 80),
            ("qty", "Qty", 60), ("size", "Size $", 90), ("upnl", "Unrealized $", 100), ("upnl_pct", "Unrealized %", 100),
            ("mode", "Mode", 70), ("open_time", "Open Time", 140),
        ):
            self.paper_open_tree.heading(c, text=t)
            self.paper_open_tree.column(c, width=w, anchor="w")
        self.paper_open_tree.pack(fill="both", expand=True)

        closed_cols = ("symbol", "side", "entry", "exit", "qty", "rpnl", "rpnl_pct", "open_time", "close_time", "mode")
        self.paper_closed_tree = ttk.Treeview(closed_box, columns=closed_cols, show="headings", height=5)
        for c, t, w in (
            ("symbol", "Symbol", 70), ("side", "Side", 60), ("entry", "Entry", 80), ("exit", "Exit", 80),
            ("qty", "Qty", 60), ("rpnl", "Realized $", 95), ("rpnl_pct", "Realized %", 95),
            ("open_time", "Open Time", 140), ("close_time", "Close Time", 140), ("mode", "Mode", 70),
        ):
            self.paper_closed_tree.heading(c, text=t)
            self.paper_closed_tree.column(c, width=w, anchor="w")
        self.paper_closed_tree.pack(fill="both", expand=True)
        self.load_paper_trading_state()
        self.refresh_paper_tables_and_metrics()

        # PanedWindow split added: chart lives in dedicated resizable container.
        chart_frame = tk.Frame(self.chart_panel)
        chart_frame.pack(fill="both", expand=True, padx=8, pady=4)
        chart_controls = tk.Frame(chart_frame)
        chart_controls.pack(fill="x", pady=(0, 4))
        tk.Label(chart_controls, text="Chart Range:").pack(side="left")
        period_box = ttk.Combobox(
            chart_controls,
            textvariable=self.chart_period_var,
            values=("90d", "60d", "30d", "7d", "1d"),
            state="readonly",
            width=8,
        )
        period_box.pack(side="left", padx=(6, 16))
        tk.Label(chart_controls, text="Chart Interval:").pack(side="left")
        interval_box = ttk.Combobox(
            chart_controls,
            textvariable=self.chart_interval_var,
            values=("60m", "30m", "15m", "5m", "1m"),
            state="readonly",
            width=8,
        )
        interval_box.pack(side="left", padx=(6, 16))
        tk.Button(chart_controls, text="Apply Chart Settings", command=self.refresh_selected_chart).pack(side="left")
        tk.Button(chart_controls, text="Auto Zoom", command=self.auto_zoom_latest).pack(side="left", padx=(8, 0))
        period_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh_selected_chart())
        interval_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh_selected_chart())

        self.chart_available = MATPLOTLIB_TK_OK
        if self.chart_available:
            self.figure, (self.ax, self.ax_rsi) = plt.subplots(
                2,
                1,
                figsize=(20, 8.5),
                sharex=True,
                # Chart frame resizing improved: taller main chart, shorter RSI panel.
                gridspec_kw={"height_ratios": [5, 1], "hspace": 0.03},
            )
            self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            self.price_hover_line = None
            self.price_hover_label = None
            self.current_price_line = None
            self.current_price_label = None
            self.trade_hover_label = None
            self.buy_marker_points: np.ndarray = np.empty((0, 2))
            self.sell_marker_points: np.ndarray = np.empty((0, 2))
            self.buy_marker_meta: List[Dict[str, object]] = []
            self.sell_marker_meta: List[Dict[str, object]] = []
            self.buy_marker_total = 0
            self.sell_marker_total = 0
            self.marker_hover_tolerance_px = 14.0
            self.last_hover_y: float | None = None
            self.last_mouse_px: float | None = None
            self.last_mouse_py: float | None = None
            self.canvas.mpl_connect("motion_notify_event", self.on_chart_mouse_move)
            self.canvas.mpl_connect("axes_leave_event", self.on_chart_mouse_leave)
            self.canvas.mpl_connect("figure_leave_event", self.on_chart_mouse_leave)
            self.canvas.mpl_connect("scroll_event", self.on_chart_scroll)
            self.ax.callbacks.connect("xlim_changed", self.on_main_xlim_changed)
        else:
            self.chart_label = tk.Label(
                chart_frame,
                text="Chart unavailable: matplotlib Tk backend not installed in this environment.",
                fg="red",
            )
            self.chart_label.pack(anchor="w")

        status_lbl = tk.Label(self.top_panel, textvariable=self.status_var, anchor="w", fg="blue")
        status_lbl.pack(fill="x", padx=8, pady=4)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.thread = threading.Thread(target=self.scan_worker, daemon=True)
        self.thread.start()
        self.root.after(500, self.process_queue)
        self.root.after(250, self.configure_initial_layout)

    def configure_initial_layout(self) -> None:
        try:
            # Resizable split defaults: keep chart dominant while preserving tables/controls.
            total_h = max(self.main_split.winfo_height(), self.root.winfo_height())
            self.main_split.sashpos(0, int(total_h * 0.40))
        except Exception:
            pass

    def on_close(self) -> None:
        global SHUTDOWN_REQUESTED
        SHUTDOWN_REQUESTED = True
        self.status_var.set("Stopping scanner...")
        self.save_paper_trading_state()
        self.root.after(500, self.root.destroy)

    def record_signal_hit(self, timestamp: str, row: Dict[str, str], frame: pd.DataFrame | None = None) -> None:
        symbol = row["symbol"]
        signal = row["signal"]
        # Prefer freshest bar signal so recent lists match plotted chart markers.
        if frame is not None and not frame.empty:
            try:
                signal = str(calculate_bar_signals(frame).iloc[-1])
            except Exception:
                signal = row["signal"]
        previous_signal = self.last_signal_by_symbol.get(symbol, "HOLD")
        self.last_signal_by_symbol[symbol] = signal

        # Only append new BUY/SELL ticks (signal transition), then keep only last 5.
        if signal == previous_signal or signal not in ("BUY", "SELL"):
            return

        entry = row.get("entry", "-")
        stop = row.get("stop_loss", "-")
        target = row.get("take_profit", "-")
        payload = {
            "time": timestamp,
            "symbol": symbol,
            "entry": entry,
            "stop": stop,
            "target": target,
        }
        if signal == "BUY":
            if symbol not in self.recent_buy_signals_by_symbol:
                self.recent_buy_signals_by_symbol[symbol] = deque(maxlen=5)
            self.recent_buy_signals_by_symbol[symbol].appendleft(payload)
        else:
            if symbol not in self.recent_short_signals_by_symbol:
                self.recent_short_signals_by_symbol[symbol] = deque(maxlen=5)
            self.recent_short_signals_by_symbol[symbol].appendleft(payload)

    def refresh_recent_signal_tables(self) -> None:
        for tree in (self.buy_tree, self.short_tree):
            for item in tree.get_children():
                tree.delete(item)

        selected = self.selected_symbol.strip().upper() if self.selected_symbol else ""
        buy_items = self.recent_buy_signals_by_symbol.get(selected, deque())
        short_items = self.recent_short_signals_by_symbol.get(selected, deque())

        for i, item in enumerate(buy_items):
            self.buy_tree.insert(
                "",
                "end",
                iid=f"buy_{i}",
                values=(item["time"], item["symbol"], item["entry"], item["stop"], item["target"]),
            )
        for i, item in enumerate(short_items):
            self.short_tree.insert(
                "",
                "end",
                iid=f"short_{i}",
                values=(item["time"], item["symbol"], item["entry"], item["stop"], item["target"]),
            )

    # ===== Paper trading features added below (simulation only) =====
    def calculate_position_size(self, price: float) -> Tuple[int, float]:
        if price <= 0:
            return 0, 0.0
        mode = self.paper_sizing.mode
        available = max(self.paper_account.available_buying_power, 0.0)

        if mode == "Fixed Contracts":
            qty = max(int(self.paper_sizing.fixed_contracts), 1)
            notional = qty * price
        elif mode == "Percent of Account":
            dollars = self.paper_account.current_balance * (max(self.paper_sizing.percent_of_account, 0.0) / 100.0)
            dollars = min(dollars, available)
            qty = int(dollars / price)
            notional = qty * price
        else:
            dollars = min(max(self.paper_sizing.fixed_dollar, 0.0), available)
            qty = int(dollars / price)
            notional = qty * price
        if qty <= 0 or notional <= 0:
            return 0, 0.0
        return qty, notional

    def open_paper_position(
        self,
        symbol: str,
        side: str,
        price: float,
        trade_mode: str,
        stop_loss: float | None = None,
    ) -> bool:
        if symbol in self.paper_open_positions:
            return False
        qty, notional = self.calculate_position_size(price)
        if qty <= 0 or notional > self.paper_account.available_buying_power:
            self.status_var.set(f"Paper trade blocked for {symbol}: insufficient paper capital.")
            return False
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.paper_open_positions[symbol] = PaperPosition(
            symbol=symbol,
            side=side,
            entry_price=price,
            current_price=price,
            quantity=qty,
            position_size=notional,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            trade_mode=trade_mode,
            open_time=now,
            stop_loss=float(stop_loss) if stop_loss is not None else np.nan,
        )
        self.paper_account.available_buying_power -= notional
        self.update_open_positions_mark_to_market([])
        self.save_paper_trading_state()
        return True

    def close_paper_position(self, symbol: str, price: float, close_time: str | None = None) -> bool:
        pos = self.paper_open_positions.get(symbol)
        if pos is None:
            return False
        if close_time is None:
            close_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        if pos.side == "LONG":
            realized = (price - pos.entry_price) * pos.quantity
        else:
            realized = (pos.entry_price - price) * pos.quantity
        realized_pct = (realized / pos.position_size) * 100 if pos.position_size > 0 else 0.0
        self.paper_account.current_balance += realized
        self.paper_account.total_realized_pnl += realized
        self.paper_account.available_buying_power += pos.position_size
        self.paper_closed_trades.insert(0, PaperClosedTrade(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            realized_pnl=realized,
            realized_pnl_pct=realized_pct,
            open_time=pos.open_time,
            close_time=close_time,
            trade_mode=pos.trade_mode,
        ))
        if len(self.paper_closed_trades) > 200:
            self.paper_closed_trades = self.paper_closed_trades[:200]
        self.paper_open_positions.pop(symbol, None)
        self.update_open_positions_mark_to_market([])
        self.save_paper_trading_state()
        return True

    def update_open_positions_mark_to_market(self, rows: List[Dict[str, str]]) -> None:
        price_map: Dict[str, float] = {}
        for row in rows:
            try:
                price_map[row["symbol"]] = float(row["last_price"])
            except Exception:
                continue

        total_unreal = 0.0
        reserved = 0.0
        stop_hits: List[Tuple[str, float]] = []
        for symbol, pos in self.paper_open_positions.items():
            if symbol in price_map:
                pos.current_price = price_map[symbol]
            if pos.side == "LONG":
                pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity
                if not np.isnan(pos.stop_loss) and pos.current_price <= pos.stop_loss:
                    stop_hits.append((symbol, pos.current_price))
            else:
                pos.unrealized_pnl = (pos.entry_price - pos.current_price) * pos.quantity
                if not np.isnan(pos.stop_loss) and pos.current_price >= pos.stop_loss:
                    stop_hits.append((symbol, pos.current_price))
            pos.unrealized_pnl_pct = (pos.unrealized_pnl / pos.position_size) * 100 if pos.position_size > 0 else 0.0
            total_unreal += pos.unrealized_pnl
            reserved += pos.position_size
        self.paper_account.total_unrealized_pnl = total_unreal
        self.paper_account.available_buying_power = max(self.paper_account.current_balance - reserved, 0.0)

        for symbol, exit_price in stop_hits:
            closed = self.close_paper_position(symbol, exit_price)
            if closed:
                self.status_var.set(f"Auto stop-loss close executed for {symbol} at {exit_price:.2f}.")
        self.refresh_paper_tables_and_metrics()

    def handle_auto_trade_signals(
        self,
        row: Dict[str, str],
        timestamp: str,
        frame: pd.DataFrame | None = None,
    ) -> None:
        if not self.paper_enabled_var.get() or not self.paper_auto_trade_var.get():
            return
        symbol = row["symbol"]
        signal = row["signal"]
        # Use fresh bar signal when available so auto-trade follows chart/scanner momentum.
        if frame is not None and not frame.empty:
            try:
                signal = str(calculate_bar_signals(frame).iloc[-1])
            except Exception:
                signal = row["signal"]
        # If scanner row is HOLD, use bullish/bearish bias from reason text as directional fallback.
        if signal == "HOLD":
            try:
                reason = row.get("reason", "")
                parts = reason.split(";")[0]
                bullish_val = int(parts.split("Bullish=")[1].split(",")[0])
                bearish_val = int(parts.split("Bearish=")[1].split(",")[0])
                if bullish_val > bearish_val:
                    signal = "BUY"
                elif bearish_val > bullish_val:
                    signal = "SELL"
            except Exception:
                pass
        try:
            price = float(row["last_price"])
        except Exception:
            return
        stop_loss_value: float | None = None
        try:
            raw_stop = row.get("stop_loss", "-")
            if raw_stop not in ("", "-", None):
                stop_loss_value = float(raw_stop)
        except Exception:
            stop_loss_value = None
        current = self.paper_open_positions.get(symbol)
        if self.paper_auto_close_opposite_var.get() and current is not None:
            if (current.side == "LONG" and signal == "SELL") or (current.side == "SHORT" and signal == "BUY"):
                self.close_paper_position(symbol, price, close_time=timestamp)
                current = None
        if current is not None:
            return
        if signal == "BUY":
            self.open_paper_position(symbol, "LONG", price, "Auto", stop_loss=stop_loss_value)
        elif signal == "SELL":
            self.open_paper_position(symbol, "SHORT", price, "Auto", stop_loss=stop_loss_value)

    def manual_open_trade(self, side: str) -> None:
        if not self.paper_enabled_var.get():
            self.status_var.set("Enable Paper Trading first (simulation only).")
            return
        symbol = self.selected_symbol or self.tree.focus()
        if not symbol or not self.tree.exists(symbol):
            self.status_var.set("Select a ticker row first.")
            return
        values = self.tree.item(symbol, "values")
        if not values:
            self.status_var.set("Unable to read selected ticker price.")
            return
        try:
            price = float(values[2])
        except Exception:
            self.status_var.set("Selected ticker has no valid last price.")
            return
        stop_loss_value: float | None = None
        try:
            if len(values) > 10 and values[10] not in ("", "-"):
                stop_loss_value = float(values[10])
        except Exception:
            stop_loss_value = None
        ok = self.open_paper_position(symbol, side, price, "Manual", stop_loss=stop_loss_value)
        self.status_var.set(f"Paper {side} opened for {symbol}." if ok else f"Paper {side} not opened for {symbol}.")

    def manual_close_trade(self) -> None:
        symbol = self.selected_symbol or self.tree.focus()
        if not symbol:
            self.status_var.set("Select a ticker row first.")
            return
        pos = self.paper_open_positions.get(symbol)
        if pos is None:
            self.status_var.set(f"No open paper position for {symbol}.")
            return
        price = pos.current_price
        if self.tree.exists(symbol):
            vals = self.tree.item(symbol, "values")
            try:
                price = float(vals[2])
            except Exception:
                pass
        ok = self.close_paper_position(symbol, price)
        self.status_var.set(f"Paper position closed for {symbol}." if ok else f"Failed to close paper position for {symbol}.")

    def refresh_paper_tables_and_metrics(self) -> None:
        self.account_balance_var.set(f"{self.paper_account.current_balance:,.2f}")
        self.account_buying_power_var.set(f"{self.paper_account.available_buying_power:,.2f}")
        self.account_realized_var.set(f"{self.paper_account.total_realized_pnl:,.2f}")
        self.account_unrealized_var.set(f"{self.paper_account.total_unrealized_pnl:,.2f}")

        for item in self.paper_open_tree.get_children():
            self.paper_open_tree.delete(item)
        for item in self.paper_closed_tree.get_children():
            self.paper_closed_tree.delete(item)

        for i, pos in enumerate(self.paper_open_positions.values()):
            self.paper_open_tree.insert(
                "", "end", iid=f"op_{i}",
                values=(
                    pos.symbol, pos.side, f"{pos.entry_price:.2f}", f"{pos.current_price:.2f}", pos.quantity,
                    f"{pos.position_size:.2f}", f"{pos.unrealized_pnl:.2f}", f"{pos.unrealized_pnl_pct:.2f}%",
                    pos.trade_mode, pos.open_time,
                ),
            )
        for i, tr in enumerate(self.paper_closed_trades[:200]):
            self.paper_closed_tree.insert(
                "", "end", iid=f"cl_{i}",
                values=(
                    tr.symbol, tr.side, f"{tr.entry_price:.2f}", f"{tr.exit_price:.2f}", tr.quantity,
                    f"{tr.realized_pnl:.2f}", f"{tr.realized_pnl_pct:.2f}%",
                    tr.open_time, tr.close_time, tr.trade_mode,
                ),
            )

    def apply_account_settings(self) -> None:
        try:
            start = float(self.starting_balance_var.get())
            fixed_d = float(self.fixed_dollar_var.get())
            pct = float(self.percent_account_var.get())
            fixed_c = int(float(self.fixed_contracts_var.get()))
            mode = self.sizing_mode_var.get()
        except Exception:
            self.status_var.set("Invalid paper account/sizing input values.")
            return
        self.paper_sizing.mode = mode
        self.paper_sizing.fixed_dollar = max(fixed_d, 0.0)
        self.paper_sizing.percent_of_account = max(pct, 0.0)
        self.paper_sizing.fixed_contracts = max(fixed_c, 1)
        # Apply settings only: do NOT reset current account/trade history here.
        # Full reset behavior is reserved for "Reset Account" button only.
        self.paper_account.starting_balance = max(start, 0.0)
        self.update_open_positions_mark_to_market([])
        self.save_paper_trading_state()
        self.status_var.set("Paper account settings applied (no account reset).")

    def reset_paper_account(self) -> None:
        self.paper_open_positions.clear()
        self.paper_closed_trades.clear()
        try:
            starting_balance = float(self.starting_balance_var.get())
        except Exception:
            starting_balance = 50000.0
        self.paper_account = PaperAccountState(starting_balance=starting_balance)
        self.paper_account.current_balance = self.paper_account.starting_balance
        self.paper_account.available_buying_power = self.paper_account.starting_balance
        self.last_signal_by_symbol.clear()
        self.refresh_paper_tables_and_metrics()
        self.save_paper_trading_state()
        self.status_var.set("Paper account reset complete.")

    def save_paper_trading_state(self) -> None:
        data = {
            "paper_account": asdict(self.paper_account),
            "paper_sizing": asdict(self.paper_sizing),
            "paper_open_positions": {k: asdict(v) for k, v in self.paper_open_positions.items()},
            "paper_closed_trades": [asdict(t) for t in self.paper_closed_trades[:200]],
            "paper_enabled": self.paper_enabled_var.get(),
            "paper_auto_trade": self.paper_auto_trade_var.get(),
            "paper_auto_close_opposite": self.paper_auto_close_opposite_var.get(),
        }
        try:
            self.paper_state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logging.error("Failed saving paper trading state: %s", exc)

    def load_paper_trading_state(self) -> None:
        if not self.paper_state_path.exists():
            self.refresh_paper_tables_and_metrics()
            return
        try:
            data = json.loads(self.paper_state_path.read_text(encoding="utf-8"))
            self.paper_account = PaperAccountState(**data.get("paper_account", {}))
            self.paper_sizing = PaperSizingSettings(**data.get("paper_sizing", {}))
            self.paper_open_positions = {
                k: PaperPosition(**v) for k, v in data.get("paper_open_positions", {}).items()
            }
            self.paper_closed_trades = [PaperClosedTrade(**x) for x in data.get("paper_closed_trades", [])]
            self.paper_enabled_var.set(bool(data.get("paper_enabled", False)))
            self.paper_auto_trade_var.set(bool(data.get("paper_auto_trade", False)))
            self.paper_auto_close_opposite_var.set(bool(data.get("paper_auto_close_opposite", False)))
            self.starting_balance_var.set(f"{self.paper_account.starting_balance:.2f}")
            self.fixed_dollar_var.set(f"{self.paper_sizing.fixed_dollar:.2f}")
            self.percent_account_var.set(f"{self.paper_sizing.percent_of_account:.2f}")
            self.fixed_contracts_var.set(str(self.paper_sizing.fixed_contracts))
            self.sizing_mode_var.set(self.paper_sizing.mode)
        except Exception as exc:
            logging.error("Failed loading paper trading state: %s", exc)
        self.update_open_positions_mark_to_market([])

    def add_ticker_from_input(self) -> None:
        symbol = self.add_symbol_var.get()
        ok, msg = add_custom_symbol(symbol)
        self.status_var.set(msg)
        if ok:
            self.add_symbol_var.set("")

    def remove_ticker_from_input(self) -> None:
        symbol = self.add_symbol_var.get()
        ok, msg = remove_symbol(symbol)
        self.status_var.set(msg)
        if ok:
            self.add_symbol_var.set("")
            symbol = symbol.strip().upper()
            if self.tree.exists(symbol):
                self.tree.delete(symbol)

    def remove_selected_ticker(self) -> None:
        selected = self.tree.focus()
        if not selected:
            self.status_var.set("Select a ticker row first.")
            return
        ok, msg = remove_symbol(selected)
        self.status_var.set(msg)
        if ok and self.tree.exists(selected):
            self.tree.delete(selected)
        if ok and selected in self.latest_frames:
            self.latest_frames.pop(selected, None)

    def on_tree_select(self, _event=None) -> None:
        selected = self.tree.focus()
        if not selected:
            return
        self.selected_symbol = selected
        self.refresh_recent_signal_tables()
        self.draw_candles(selected)

    def refresh_selected_chart(self) -> None:
        # Keep list/table scan timeframe aligned with the chart timeframe selection.
        self.sync_scan_timeframe_with_chart()
        if self.selected_symbol:
            self.draw_candles(self.selected_symbol)

    def sync_scan_timeframe_with_chart(self) -> None:
        selected_interval = self.chart_interval_var.get().strip()
        if selected_interval in TIMEFRAME_MAP:
            set_active_scan_timeframe(selected_interval)
        elif selected_interval == "60m":
            set_active_scan_timeframe("1h")
        active_tf = get_active_scan_timeframe()
        self.info_var.set(f"Timeframe: {active_tf} | Scan every {CONFIG.scan_interval_seconds}s")

    def apply_chart_colors(self) -> None:
        for key, var in self.color_vars.items():
            value = var.get().strip()
            if value and (is_color_like is None or is_color_like(value)):
                self.chart_colors[key] = value
            elif value:
                self.status_var.set(f"Ignored invalid color for {key}: {value}")
        self.status_var.set("Chart colors updated.")
        self.refresh_selected_chart()

    def auto_zoom_latest(self) -> None:
        if (
            not self.chart_available
            or self.latest_x_right is None
            or self.full_x_min is None
            or self.full_x_max is None
        ):
            return
        window_width = (self.full_x_max - self.full_x_min) * 0.35
        if window_width <= 0:
            return
        self.user_zoom_active = True
        self.ax.set_xlim(self.latest_x_right - window_width, self.latest_x_right)
        self.ax_rsi.set_xlim(self.latest_x_right - window_width, self.latest_x_right)
        y_min, y_max = self.ax.get_ylim()
        self.ax.set_ylim(y_min, y_max)
        self.canvas.draw_idle()

    def _resolve_data_symbol_for_chart(self, symbol: str) -> str:
        if symbol in CUSTOM_SYMBOL_MAP:
            return CUSTOM_SYMBOL_MAP[symbol]
        if symbol in PRIMARY_SYMBOL_MAP:
            return PRIMARY_SYMBOL_MAP[symbol]
        return symbol

    def fetch_chart_data(self, symbol: str) -> Tuple[pd.DataFrame | None, str | None]:
        chart_symbol = self._resolve_data_symbol_for_chart(symbol)
        interval = self.chart_interval_var.get()
        period = self.chart_period_var.get()
        if interval not in CHART_INTERVAL_MAX_DAYS:
            interval = "60m"
        try:
            requested_days = int(period[:-1])
        except Exception:
            requested_days = 90
        max_days = CHART_INTERVAL_MAX_DAYS[interval]
        actual_days = min(requested_days, max_days)
        actual_period = f"{actual_days}d"

        try:
            chart_df = yf.Ticker(chart_symbol).history(
                interval=interval,
                period=actual_period,
                auto_adjust=False,
                prepost=True,
            )
            chart_df = chart_df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
            if chart_df.empty:
                return None, f"No chart data for {chart_symbol} ({interval}, {actual_period})."
            note = None
            if actual_period != period:
                note = f"Adjusted chart range to {actual_period} (max for {interval})."
            return chart_df, note
        except Exception as exc:
            return None, f"Chart fetch failed for {chart_symbol}: {exc}"

    def draw_candles(self, symbol: str) -> None:
        if not self.chart_available:
            return

        # Chart refresh logic update: fetch once per redraw, then calculate studies once.
        df, chart_note = self.fetch_chart_data(symbol)
        if df is None or df.empty:
            if chart_note:
                self.status_var.set(chart_note)
            return

        previous_xlim = self.ax.get_xlim()
        previous_width = max(previous_xlim[1] - previous_xlim[0], 0.0)
        plot_df = df.tail(80).copy()
        vwap = calculate_vwap(plot_df)
        ema9 = calculate_ema(plot_df["Close"], 9)
        ema21 = calculate_ema(plot_df["Close"], 21)
        rsi14 = calculate_rsi(plot_df["Close"], 14)

        self.ax.clear()
        self.ax_rsi.clear()
        self.ax.set_title(
            f"{symbol} Candlestick ({self.chart_period_var.get()} / {self.chart_interval_var.get()})"
            f" - latest {len(plot_df)} bars"
        )
        self.ax.set_ylabel("Price")
        self.ax.yaxis.tick_right()
        self.ax.yaxis.set_label_position("right")
        self.ax.spines["left"].set_visible(False)
        self.ax.spines["right"].set_visible(True)

        x = mdates.date2num(plot_df.index.to_pydatetime())
        candle_width = (x[1] - x[0]) * 0.7 if len(x) > 1 else 0.0005

        for xi, (_, row) in zip(x, plot_df.iterrows()):
            open_p = float(row["Open"])
            high_p = float(row["High"])
            low_p = float(row["Low"])
            close_p = float(row["Close"])

            color = self.chart_colors["bull_candle"] if close_p >= open_p else self.chart_colors["bear_candle"]
            self.ax.plot([xi, xi], [low_p, high_p], color=color, linewidth=1)
            body_low = min(open_p, close_p)
            body_height = abs(close_p - open_p)
            if body_height == 0:
                body_height = max((high_p - low_p) * 0.01, 0.0001)
            rect = Rectangle(
                (xi - candle_width / 2, body_low),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=0.8,
            )
            self.ax.add_patch(rect)

        # Study overlays added: VWAP, EMA9, EMA21 on the main candlestick panel.
        self.ax.plot(x, vwap.values, color=self.chart_colors["vwap"], linewidth=1.2, linestyle="-", label="VWAP")
        self.ax.plot(x, ema9.values, color=self.chart_colors["ema9"], linewidth=1.2, linestyle="-", label="EMA9")
        self.ax.plot(x, ema21.values, color=self.chart_colors["ema21"], linewidth=1.2, linestyle="-", label="EMA21")

        close_trend = calculate_trendline(plot_df["Close"])
        high_trend = calculate_trendline(plot_df["High"])
        low_trend = calculate_trendline(plot_df["Low"])
        self.ax.plot(x, close_trend.values, color="gold", linewidth=1.6, linestyle="--", label="Close Trend")
        self.ax.plot(x, high_trend.values, color="dodgerblue", linewidth=1.1, linestyle=":", label="Resistance Trend")
        self.ax.plot(x, low_trend.values, color="mediumpurple", linewidth=1.1, linestyle=":", label="Support Trend")

        bar_signal = calculate_bar_signals(plot_df)
        signal_change = bar_signal.ne(bar_signal.shift(1))
        buy_idx = bar_signal[(bar_signal == "BUY") & signal_change].index
        sell_idx = bar_signal[(bar_signal == "SELL") & signal_change].index
        self.update_recent_from_chart_markers(symbol, plot_df, buy_idx, sell_idx)

        if len(buy_idx) > 0:
            buy_x = mdates.date2num(buy_idx.to_pydatetime())
            offset = max(float((plot_df["High"] - plot_df["Low"]).median()) * 0.12, 0.0001)
            buy_y = plot_df.loc[buy_idx, "Low"].values - offset
            self.ax.scatter(buy_x, buy_y, marker="^", color=self.chart_colors["buy"], s=60, zorder=5, label="Buy")
            self.buy_marker_points = np.column_stack((buy_x, buy_y))
            self.buy_marker_meta = self.build_marker_meta(plot_df, buy_idx, "BUY")
            self.buy_marker_total = len(buy_idx)
        else:
            self.buy_marker_points = np.empty((0, 2))
            self.buy_marker_meta = []
            self.buy_marker_total = 0
        if len(sell_idx) > 0:
            sell_x = mdates.date2num(sell_idx.to_pydatetime())
            offset = max(float((plot_df["High"] - plot_df["Low"]).median()) * 0.12, 0.0001)
            sell_y = plot_df.loc[sell_idx, "High"].values + offset
            self.ax.scatter(sell_x, sell_y, marker="v", color=self.chart_colors["sell"], s=60, zorder=5, label="Short/Sell")
            self.sell_marker_points = np.column_stack((sell_x, sell_y))
            self.sell_marker_meta = self.build_marker_meta(plot_df, sell_idx, "SHORT")
            self.sell_marker_total = len(sell_idx)
        else:
            self.sell_marker_points = np.empty((0, 2))
            self.sell_marker_meta = []
            self.sell_marker_total = 0

        self.ax.xaxis_date()
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper left", fontsize=8)

        latest_close = float(plot_df["Close"].iloc[-1])
        self.current_price_line = self.ax.axhline(
            y=latest_close,
            color=self.chart_colors["current_price"],
            linewidth=1.1,
            linestyle="-.",
            alpha=0.9,
        )
        self.current_price_label = self.ax.text(
            1.005,
            latest_close,
            f"Last {latest_close:.2f}",
            transform=self.ax.get_yaxis_transform(),
            color=self.chart_colors["current_price"],
            fontsize=8,
            va="center",
            ha="left",
            bbox={"boxstyle": "round,pad=0.2", "fc": "black", "ec": self.chart_colors["current_price"], "alpha": 0.6},
        )

        # RSI display added: lower panel in trading-platform style with latest value and slope direction.
        rsi_slope = rsi14.iloc[-1] - rsi14.iloc[-2] if len(rsi14) > 1 else 0.0
        rsi_direction = "Rising" if rsi_slope > 0 else "Falling" if rsi_slope < 0 else "Flat"
        self.ax_rsi.plot(x, rsi14.values, color="mediumpurple", linewidth=1.3, label="RSI(14)")
        self.ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.9, alpha=0.7, label="RSI 70")
        self.ax_rsi.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.8, label="RSI 50")
        self.ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.9, alpha=0.7, label="RSI 30")
        self.ax_rsi.set_ylim(0, 100)
        self.ax_rsi.set_ylabel("RSI")
        self.ax_rsi.set_xlabel("Time")
        self.ax_rsi.grid(True, alpha=0.3)
        self.ax_rsi.legend(loc="upper left", fontsize=8)
        self.ax_rsi.set_title(f"RSI(14): {rsi14.iloc[-1]:.2f} ({rsi_direction})", fontsize=9)

        self.full_x_min = float(x[0])
        self.full_x_max = float(x[-1])
        self.latest_x_right = self.full_x_max
        full_width = max(self.full_x_max - self.full_x_min, 0.0)

        # Right-edge anchoring: when zoomed in, keep latest candle on the right and
        # shift only the left boundary by preserving the current zoom width.
        if self.user_zoom_active and previous_width > 0 and full_width > 0:
            anchored_width = min(previous_width, full_width)
            self.ax.set_xlim(self.latest_x_right - anchored_width, self.latest_x_right)
            self.ax_rsi.set_xlim(self.latest_x_right - anchored_width, self.latest_x_right)
            enforce_right_anchor(self.ax, self.latest_x_right)
            enforce_right_anchor(self.ax_rsi, self.latest_x_right)
        else:
            self.ax.set_xlim(self.full_x_min, self.full_x_max)
            self.ax_rsi.set_xlim(self.full_x_min, self.full_x_max)
            self.user_zoom_active = False

        self.figure.autofmt_xdate()
        # Figure spacing improved: reduce excess margins for a denser trading-panel view.
        self.figure.subplots_adjust(left=0.04, right=0.985, top=0.95, bottom=0.08, hspace=0.04)
        if self.last_hover_y is not None:
            self.render_hover_line(self.last_hover_y)
        # Keep hover tooltip visible across chart refreshes while mouse is still hovering.
        self.refresh_trade_hover_from_last_pointer()
        self.canvas.draw_idle()
        if chart_note:
            self.status_var.set(chart_note)

    def update_recent_from_chart_markers(
        self,
        symbol: str,
        plot_df: pd.DataFrame,
        buy_idx: pd.Index,
        sell_idx: pd.Index,
    ) -> None:
        """
        Keep recent BUY/SHORT tables aligned with plotted chart markers for the selected ticker.
        """
        buy_items: List[Dict[str, str]] = []
        sell_items: List[Dict[str, str]] = []

        for ts in list(buy_idx)[-5:]:
            hist = plot_df.loc[:ts].tail(8)
            if hist.empty:
                continue
            entry = float(hist["Close"].iloc[-1])
            stop = float(hist["Low"].min())
            risk = max(entry - stop, 0.0001)
            target = entry + (1.8 * risk)
            buy_items.append({
                "time": ts.strftime("%Y-%m-%d %H:%M"),
                "symbol": symbol,
                "entry": f"{entry:.2f}",
                "stop": f"{stop:.2f}",
                "target": f"{target:.2f}",
            })

        for ts in list(sell_idx)[-5:]:
            hist = plot_df.loc[:ts].tail(8)
            if hist.empty:
                continue
            entry = float(hist["Close"].iloc[-1])
            stop = float(hist["High"].max())
            risk = max(stop - entry, 0.0001)
            target = entry - (1.8 * risk)
            sell_items.append({
                "time": ts.strftime("%Y-%m-%d %H:%M"),
                "symbol": symbol,
                "entry": f"{entry:.2f}",
                "stop": f"{stop:.2f}",
                "target": f"{target:.2f}",
            })

        self.recent_buy_signals_by_symbol[symbol] = deque(reversed(buy_items), maxlen=5)
        self.recent_short_signals_by_symbol[symbol] = deque(reversed(sell_items), maxlen=5)
        if self.selected_symbol == symbol:
            self.refresh_recent_signal_tables()

    def on_main_xlim_changed(self, _ax) -> None:
        if self.full_x_min is None or self.full_x_max is None:
            return
        x_min, x_max = self.ax.get_xlim()
        full_width = self.full_x_max - self.full_x_min
        current_width = x_max - x_min
        if full_width <= 0:
            self.user_zoom_active = False
            return
        at_home_view = abs(x_min - self.full_x_min) < 1e-6 and abs(x_max - self.full_x_max) < 1e-6
        self.user_zoom_active = (current_width < full_width * 0.995) and not at_home_view

    def on_chart_mouse_move(self, event) -> None:
        if not self.chart_available:
            return
        if event.inaxes != self.ax or event.ydata is None:
            self.clear_trade_hover_label()
            self.last_mouse_px = None
            self.last_mouse_py = None
            return

        self.last_mouse_px = float(event.x) if event.x is not None else None
        self.last_mouse_py = float(event.y) if event.y is not None else None
        self.last_hover_y = float(event.ydata)
        self.render_hover_line(self.last_hover_y)
        self.render_trade_marker_hover(event)
        self.canvas.draw_idle()

    def on_chart_mouse_leave(self, _event) -> None:
        self.last_mouse_px = None
        self.last_mouse_py = None
        self.clear_trade_hover_label()
        if self.chart_available:
            self.canvas.draw_idle()

    def render_hover_line(self, y_value: float) -> None:
        if not self.chart_available:
            return
        if self.price_hover_line is not None:
            try:
                self.price_hover_line.remove()
            except Exception:
                pass
            self.price_hover_line = None
        if self.price_hover_label is not None:
            try:
                self.price_hover_label.remove()
            except Exception:
                pass
            self.price_hover_label = None

        self.price_hover_line = self.ax.axhline(
            y=y_value,
            color="orange",
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
        )
        self.price_hover_label = self.ax.text(
            1.005,
            y_value,
            f"{y_value:.2f}",
            transform=self.ax.get_yaxis_transform(),
            color="orange",
            fontsize=8,
            va="center",
            ha="left",
            bbox={"boxstyle": "round,pad=0.2", "fc": "black", "ec": "orange", "alpha": 0.6},
        )

    def clear_trade_hover_label(self) -> None:
        if self.trade_hover_label is not None:
            try:
                self.trade_hover_label.remove()
            except Exception:
                pass
            self.trade_hover_label = None

    def render_trade_marker_hover(self, event) -> None:
        """
        Display buy/short marker totals when hovering near marker points.
        """
        if event.x is None or event.y is None:
            self.clear_trade_hover_label()
            return
        self._render_trade_marker_hover_at(event.x, event.y)

    def _render_trade_marker_hover_at(self, mouse_x: float, mouse_y: float) -> None:
        nearest = None
        for side, points, total, color, meta in (
            ("BUY", self.buy_marker_points, self.buy_marker_total, self.chart_colors["buy"], self.buy_marker_meta),
            ("SHORT", self.sell_marker_points, self.sell_marker_total, self.chart_colors["sell"], self.sell_marker_meta),
        ):
            if len(points) == 0:
                continue
            screen_points = self.ax.transData.transform(points)
            dists = np.hypot(screen_points[:, 0] - mouse_x, screen_points[:, 1] - mouse_y)
            idx = int(np.argmin(dists))
            dist = float(dists[idx])
            if nearest is None or dist < nearest["dist"]:
                marker_info = meta[idx] if idx < len(meta) else {}
                marker_price = float(marker_info.get("price", points[idx, 1]))
                marker_qty = int(marker_info.get("qty", 1))
                marker_notional = float(marker_info.get("notional", marker_qty * marker_price))
                marker_time = str(marker_info.get("time", "N/A"))
                nearest = {
                    "dist": dist,
                    "x": float(points[idx, 0]),
                    "y": float(points[idx, 1]),
                    "text": (
                        f"{side} total: {marker_qty} @ {marker_price:.2f}\n"
                        f"Notional: ${marker_notional:,.2f}\n"
                        f"Current futures price then: {marker_price:.2f}\n"
                        f"Signal count ({side}): {total}\n"
                        f"Time: {marker_time}"
                    ),
                    "color": color,
                }

        if nearest is None or nearest["dist"] > self.marker_hover_tolerance_px:
            self.clear_trade_hover_label()
            return

        self.clear_trade_hover_label()
        self.trade_hover_label = self.ax.annotate(
            nearest["text"],
            xy=(nearest["x"], nearest["y"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.25", "fc": "black", "ec": nearest["color"], "alpha": 0.8},
            arrowprops={"arrowstyle": "->", "color": nearest["color"], "lw": 0.8},
        )

    def refresh_trade_hover_from_last_pointer(self) -> None:
        if self.last_mouse_px is None or self.last_mouse_py is None:
            return
        self._render_trade_marker_hover_at(self.last_mouse_px, self.last_mouse_py)

    def build_marker_meta(self, df: pd.DataFrame, idx: pd.Index, side: str) -> List[Dict[str, object]]:
        """
        Build hover metadata for each marker point, including estimated size and
        the futures price at the signal time.
        """
        items: List[Dict[str, object]] = []
        for ts in idx:
            if ts not in df.index:
                continue
            price = float(df.loc[ts, "Close"])
            qty = max(1, int(CONFIG.target_notional_per_trade / max(price, 1e-9)))
            items.append({
                "side": side,
                "time": ts.strftime("%Y-%m-%d %H:%M"),
                "price": price,
                "qty": qty,
                "notional": qty * price,
            })
        return items

    def on_chart_scroll(self, event) -> None:
        if not self.chart_available:
            return
        if event.inaxes not in (self.ax, self.ax_rsi):
            return

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        if event.button == "up":
            scale_factor = 0.85
        elif event.button == "down":
            scale_factor = 1.15
        else:
            return

        if self.latest_x_right is None or self.full_x_min is None or self.full_x_max is None:
            return
        x_center = self.latest_x_right
        y_center = event.ydata if event.ydata is not None else (y_min + y_max) / 2

        current_width = max(x_max - x_min, 1e-9)
        full_width = max(self.full_x_max - self.full_x_min, 1e-9)
        new_width = current_width * scale_factor
        if new_width >= full_width:
            new_width = full_width
            self.user_zoom_active = False
        else:
            self.user_zoom_active = True

        new_x_half = new_width / 2
        new_y_half = ((y_max - y_min) * scale_factor) / 2

        # Override X-limits so right edge stays anchored at the newest candle.
        self.ax.set_xlim(x_center - (2 * new_x_half), x_center)
        self.ax_rsi.set_xlim(x_center - (2 * new_x_half), x_center)
        if self.user_zoom_active:
            enforce_right_anchor(self.ax, x_center)
            enforce_right_anchor(self.ax_rsi, x_center)
        if event.inaxes == self.ax:
            self.ax.set_ylim(y_center - new_y_half, y_center + new_y_half)
        else:
            rsi_min, rsi_max = self.ax_rsi.get_ylim()
            rsi_center = event.ydata if event.ydata is not None else (rsi_min + rsi_max) / 2
            rsi_half = ((rsi_max - rsi_min) * scale_factor) / 2
            self.ax_rsi.set_ylim(rsi_center - rsi_half, rsi_center + rsi_half)
        self.canvas.draw_idle()

    def scan_worker(self) -> None:
        while not SHUTDOWN_REQUESTED:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            try:
                rows, latest_frames = run_single_scan(self.csv_path, timestamp)
                self.queue.put((timestamp, rows, latest_frames, None))
            except Exception as exc:
                logging.error("Worker scan failed: %s", exc, exc_info=True)
                self.queue.put((timestamp, [], {}, str(exc)))

            slept = 0
            while slept < CONFIG.scan_interval_seconds and not SHUTDOWN_REQUESTED:
                time.sleep(1)
                slept += 1

    def process_queue(self) -> None:
        try:
            while True:
                timestamp, rows, latest_frames, err = self.queue.get_nowait()
                if err:
                    self.status_var.set(f"{timestamp} | ERROR: {err}")
                    continue

                for sym, frame in latest_frames.items():
                    self.latest_frames[sym] = frame

                for row in rows:
                    iid = row["symbol"]
                    self.record_signal_hit(timestamp, row, latest_frames.get(iid))
                    self.handle_auto_trade_signals(row, timestamp, latest_frames.get(iid))
                    values = (
                        row["symbol"],
                        row["data"],
                        row["last_price"],
                        row["support"],
                        row["resistance"],
                        row["action"],
                        row["trade_mode"],
                        row["entry"],
                        row["estimated_next_buy"],
                        row["estimated_next_short"],
                        row["stop_loss"],
                        row["take_profit"],
                        row["signal"],
                        row["reason"],
                    )

                    if self.tree.exists(iid):
                        self.tree.item(iid, values=values)
                    else:
                        self.tree.insert("", "end", iid=iid, values=values)

                    if not self.selected_symbol:
                        self.selected_symbol = iid
                self.refresh_recent_signal_tables()
                self.update_open_positions_mark_to_market(rows)
                self.status_var.set(
                    f"Last update: {timestamp} | Timeframe: {get_active_scan_timeframe()} | Next scan in ~{CONFIG.scan_interval_seconds}s"
                )
                if self.selected_symbol:
                    self.draw_candles(self.selected_symbol)

        except Empty:
            pass

        if not SHUTDOWN_REQUESTED:
            self.root.after(500, self.process_queue)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    setup_logging()
    validate_config()

    for sig in ("SIGINT", "SIGTERM"):
        try:
            signal.signal(getattr(signal, sig), handle_signal)
        except Exception:
            logging.error("Signal %s is not supported in this environment.", sig)

    try:
        if tk is not None and ttk is not None:
            app = FuturesScannerGUI()
            app.run()
        else:
            run_console_scanner()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Exiting...")
    except Exception as exc:
        logging.error("Fatal error in scanner: %s", exc, exc_info=True)
        print(f"Fatal error: {exc}. Falling back to console mode...")
        run_console_scanner()


if __name__ == "__main__":
    main()
