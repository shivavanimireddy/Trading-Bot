# ============================================================
# G7FX Signal App — Stage 3: Order Flow Confirmation
# ============================================================
# Implements:
#   - Cumulative delta calculator (from tick volume proxy)
#   - CD divergence detector (bullish / bearish / hidden)
#   - Footprint proxy (absorption, exhaustion from OHLCV)
#   - Final signal gate: only passes if Stage 3 confirms
# ============================================================
# NOTE: True footprint requires L2 tick data (bid/ask classified).
# This module uses OHLCV + tick volume as a proxy — directionally
# reliable for CD divergence, approximate for footprint.
# For full accuracy: use CME forex futures via IBKR + Rithmic feed.
# ============================================================

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

from core.stage2_hypothesis import SignalHypothesis

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────

@dataclass
class OrderFlowReading:
    cd_signal: str          # "confirms" | "diverges" | "neutral"
    fp_signal: str          # "absorption" | "exhaustion" | "stacked" | "divergence" | "neutral"
    cd_divergence_type: str # "bullish" | "bearish" | "hidden" | "none"
    stage3_score: int       # 0–30, or -999 if suppressed
    suppress: bool          # True = do not emit signal
    notes: list


# ── Cumulative Delta (Tick Volume Proxy) ──────────────────────

def estimate_delta_from_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate buy/sell volume split from OHLCV data.

    Method: Use candle body direction + close position within range
    to approximate aggressor volume. This is the standard proxy
    used when true L2 bid/ask data is unavailable.

    delta = buy_vol - sell_vol per candle
    """
    df = df.copy()

    # Close position within candle range (0 = at low, 1 = at high)
    close_pos = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9)

    # Bullish candles: more buy aggression; bearish: more sell
    df["buy_vol"]  = df["volume"] * close_pos
    df["sell_vol"] = df["volume"] * (1 - close_pos)
    df["delta"]    = df["buy_vol"] - df["sell_vol"]
    df["cum_delta"] = df["delta"].cumsum()

    return df


def detect_cd_divergence(df: pd.DataFrame,
                          direction: str,
                          window: int = 5) -> str:
    """
    Detect cumulative delta divergence relative to price.
    G7FX Stage 3.1 / 3.2 logic.

    direction: "long" (looking for bullish confirmation) |
               "short" (looking for bearish confirmation)

    Returns: "confirms" | "bullish_div" | "bearish_div" | "hidden_div" | "neutral"
    """
    df = estimate_delta_from_ohlcv(df)
    recent = df.tail(window * 2)

    price_cd = recent[["close", "cum_delta"]].copy()

    # Split into two halves to compare trend direction
    half = len(price_cd) // 2
    first_half  = price_cd.iloc[:half]
    second_half = price_cd.iloc[half:]

    price_change = second_half["close"].mean() - first_half["close"].mean()
    delta_change = second_half["cum_delta"].mean() - first_half["cum_delta"].mean()

    price_up  = price_change > 0
    delta_up  = delta_change > 0

    if direction == "long":
        if not price_up and delta_up:
            # Price falling but delta rising = bullish divergence (buying absorption)
            return "bullish_div"
        elif price_up and delta_up:
            return "confirms"
        elif price_up and not delta_up:
            # Price rising but delta falling = bearish divergence at potential long zone
            return "bearish_div"
        else:
            return "neutral"

    else:  # short
        if price_up and not delta_up:
            # Price rising but delta falling = bearish divergence (selling absorption)
            return "bearish_div"
        elif not price_up and not delta_up:
            return "confirms"
        elif not price_up and delta_up:
            # Price falling but delta rising = bullish divergence at potential short zone
            return "bullish_div"
        else:
            return "neutral"


# ── Footprint Proxy ───────────────────────────────────────────

def detect_absorption(df: pd.DataFrame,
                       zone_price: float,
                       direction: str,
                       pip: float = 0.01,
                       lookback: int = 5) -> str:
    """
    Detect absorption / exhaustion patterns from OHLCV.
    Proxy for true footprint analysis.

    Absorption: High volume candle at zone, small body, price doesn't move.
    Exhaustion: Price extends to new extreme, but volume shrinks.
    Stacked: Multiple consecutive candles all with aligned delta at zone.
    """
    recent = df.tail(lookback).copy()
    recent = estimate_delta_from_ohlcv(recent)

    # Body size relative to range
    recent["body"]       = abs(recent["close"] - recent["open"])
    recent["range"]      = recent["high"] - recent["low"]
    recent["body_ratio"] = recent["body"] / (recent["range"] + 1e-9)

    # Volume z-score (is this candle high volume relative to recent?)
    vol_mean = df["volume"].tail(50).mean()
    vol_std  = df["volume"].tail(50).std() + 1e-9
    recent["vol_z"] = (recent["volume"] - vol_mean) / vol_std

    last = recent.iloc[-1]
    last_body_ratio = last["body_ratio"]
    last_vol_z      = last["vol_z"]

    # ── Absorption: high volume, small body at zone ──
    if last_vol_z > 1.5 and last_body_ratio < 0.35:
        if direction == "long" and abs(last["low"] - zone_price) < pip * 8:
            logger.info("Footprint proxy: ABSORPTION detected at long zone")
            return "absorption"
        elif direction == "short" and abs(last["high"] - zone_price) < pip * 8:
            logger.info("Footprint proxy: ABSORPTION detected at short zone")
            return "absorption"

    # ── Exhaustion: new price extreme with lower volume ──
    if len(recent) >= 3:
        if direction == "short":
            new_high = last["high"] > recent.iloc[-2]["high"]
            vol_shrinking = last["volume"] < recent.iloc[-2]["volume"] * 0.8
            if new_high and vol_shrinking:
                logger.info("Footprint proxy: EXHAUSTION at high")
                return "exhaustion"
        else:
            new_low = last["low"] < recent.iloc[-2]["low"]
            vol_shrinking = last["volume"] < recent.iloc[-2]["volume"] * 0.8
            if new_low and vol_shrinking:
                logger.info("Footprint proxy: EXHAUSTION at low")
                return "exhaustion"

    # ── Stacked imbalances: 3+ candles all showing same delta direction ──
    if direction == "long":
        stacked = all(recent["delta"].tail(3) > 0)
    else:
        stacked = all(recent["delta"].tail(3) < 0)

    if stacked:
        logger.info("Footprint proxy: STACKED IMBALANCES detected")
        return "stacked"

    # ── Delta divergence on candle: green candle, negative delta ──
    last_bullish = last["close"] > last["open"]
    last_neg_delta = last["delta"] < 0
    if last_bullish and last_neg_delta and direction == "short":
        return "divergence"
    if not last_bullish and not last_neg_delta and direction == "long":
        return "divergence"

    return "neutral"


# ── Stage 3 Scoring ───────────────────────────────────────────

def score_stage3(cd_signal: str, fp_signal: str, direction: str) -> tuple:
    """
    Score the Stage 3 order flow reading.
    Returns (score, suppress).

    Suppression (-999) overrides everything — signal is killed.
    """
    suppress = False
    score = 0

    # --- CD signal ---
    if cd_signal in ("bullish_div",) and direction == "long":
        score += 15
    elif cd_signal in ("bearish_div",) and direction == "short":
        score += 15
    elif cd_signal == "confirms":
        score += 10
    elif cd_signal == "bullish_div" and direction == "short":
        suppress = True   # CD confirms buyers at a short zone = danger
    elif cd_signal == "bearish_div" and direction == "long":
        suppress = True   # CD confirms sellers at a long zone = danger

    if suppress:
        return -999, True

    # --- Footprint signal ---
    if fp_signal == "stacked":
        score += 15
    elif fp_signal == "absorption":
        score += 10
    elif fp_signal == "exhaustion":
        score += 10
    elif fp_signal == "divergence":
        suppress = True
        return -999, True

    return min(score, 30), False


# ── Main Stage 3 Evaluator ────────────────────────────────────

def evaluate_stage3(m15_df: pd.DataFrame,
                     h1_df: pd.DataFrame,
                     hypothesis: SignalHypothesis) -> OrderFlowReading:
    """
    Run Stage 3 order flow checks.
    Returns OrderFlowReading with final score and suppress flag.
    """
    direction = "long" if hypothesis.direction == "BUY" else "short"
    notes = []

    # Cumulative delta on M15
    cd_div = detect_cd_divergence(m15_df, direction, window=6)
    notes.append(f"CD signal: {cd_div}")

    # Footprint proxy on M15 at zone
    zone_price = hypothesis.entry_mid
    fp_sig = detect_absorption(m15_df, zone_price, direction)
    notes.append(f"Footprint proxy: {fp_sig}")

    stage3_score, suppress = score_stage3(cd_div, fp_sig, direction)

    # Classify CD divergence type for display
    cd_type = "none"
    if "div" in cd_div:
        cd_type = cd_div.replace("_div", "")
    elif cd_div == "confirms":
        cd_type = "confirming"

    if suppress:
        notes.append("⚠️ Stage 3 SUPPRESSED signal — conflicting order flow")
        logger.warning(f"Stage 3 suppressed signal | CD={cd_div} | FP={fp_sig}")
    else:
        logger.info(f"Stage 3 | CD={cd_div} | FP={fp_sig} | score={stage3_score}")

    return OrderFlowReading(
        cd_signal          = cd_div,
        fp_signal          = fp_sig,
        cd_divergence_type = cd_type,
        stage3_score       = stage3_score,
        suppress           = suppress,
        notes              = notes
    )
