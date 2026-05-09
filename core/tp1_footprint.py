# ============================================================
# G7FX Signal Engine — TP1 Footprint Decision (Gap 4 Fix)
# ============================================================
# At TP1 (VWAP), instead of always closing 50% mechanically,
# read the footprint to decide: hold or exit?
#
# Three scenarios (as identified in the gap analysis):
#
#   Scenario A — sellers still dominant (short trade):
#     Sell volume 3-4x buy volume at VWAP levels
#     → HOLD 50% for TP2, move SL to breakeven
#
#   Scenario B — buyers absorbing (short trade at VWAP):
#     Buy volume 3-4x sell volume — institutions buying
#     → EXIT 100% now, do not hold for TP2
#
#   Scenario C — exhaustion (volume collapsed both sides):
#     Both buy and sell volume thin, delta near zero
#     → EXIT 100%, move is done for the session
# ============================================================

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FootprintReading:
    scenario: str           # "A_hold" | "B_exit" | "C_exit" | "neutral"
    decision: str           # "HOLD_FOR_TP2" | "EXIT_FULL"
    avg_buy_vol: float      # average ask volume at TP1 zone
    avg_sell_vol: float     # average bid volume at TP1 zone
    delta_ratio: float      # sell/buy ratio (>2 = sellers dominant for shorts)
    volume_thin: bool       # True = exhaustion (Scenario C)
    confidence: int         # 0-100 confidence in the reading
    notes: list = field(default_factory=list)


def estimate_candle_delta(candle: pd.Series) -> tuple:
    """
    Estimate buy volume (ask aggressor) and sell volume (bid aggressor)
    from a single OHLCV candle.

    Method: close position within the candle range determines the
    buy/sell split. Closing near the high = more buy aggression.
    Closing near the low = more sell aggression.

    Returns (buy_vol, sell_vol)
    """
    candle_range = candle['high'] - candle['low']
    if candle_range < 1e-9:
        return candle['volume'] * 0.5, candle['volume'] * 0.5

    close_pos = (candle['close'] - candle['low']) / candle_range
    buy_vol   = candle['volume'] * close_pos
    sell_vol  = candle['volume'] * (1 - close_pos)
    return buy_vol, sell_vol


def build_footprint_at_level(m5_df: pd.DataFrame,
                              target_price: float,
                              pip: float = 0.01,
                              zone_pips: float = 15.0) -> list:
    """
    Build a simplified footprint around a target price level.

    Scans M5 candles where price traded within ±zone_pips of
    target_price and returns per-candle buy/sell volume breakdown.

    Returns list of dicts: [{price, buy_vol, sell_vol, delta}, ...]
    """
    zone_range = zone_pips * pip
    rows = []

    for _, candle in m5_df.iterrows():
        # Check if this candle touched the target zone
        touched_zone = (candle['low']  <= target_price + zone_range and
                        candle['high'] >= target_price - zone_range)
        if not touched_zone:
            continue

        buy_vol, sell_vol = estimate_candle_delta(candle)
        rows.append({
            'time':     candle.name,
            'price':    round(float(candle['close']), 3),
            'buy_vol':  round(buy_vol, 1),
            'sell_vol': round(sell_vol, 1),
            'delta':    round(buy_vol - sell_vol, 1),
            'total_vol': candle['volume']
        })

    return rows[-8:] if len(rows) > 8 else rows   # last 8 interactions


def read_footprint_at_tp1(m5_df: pd.DataFrame,
                           tp1_price: float,
                           direction: str,
                           lookback_candles: int = 20) -> FootprintReading:
    """
    Core function: read the footprint at TP1 (VWAP) and return
    hold vs exit decision.

    direction: "long" | "short"
    tp1_price: the VWAP price (TP1 target)
    """
    notes = []

    # Use last N candles around TP1
    recent = m5_df.tail(lookback_candles)
    fp_rows = build_footprint_at_level(recent, tp1_price)

    if not fp_rows:
        # No candles touched the zone — price hasn't reached TP1 yet
        return FootprintReading(
            scenario    = "neutral",
            decision    = "HOLD_FOR_TP2",
            avg_buy_vol = 0, avg_sell_vol = 0,
            delta_ratio = 1.0, volume_thin = False,
            confidence  = 0,
            notes       = ["Price has not yet reached TP1 zone"]
        )

    avg_buy  = np.mean([r['buy_vol']  for r in fp_rows])
    avg_sell = np.mean([r['sell_vol'] for r in fp_rows])
    avg_vol  = np.mean([r['total_vol'] for r in fp_rows])

    # Volume baseline: compare to 20-period average
    baseline_vol = m5_df['volume'].tail(50).mean()
    volume_thin  = avg_vol < baseline_vol * 0.40   # less than 40% of normal

    # Delta ratio for the dominant side
    if direction == "short":
        # For shorts: we want sell_vol > buy_vol at TP1
        delta_ratio = avg_sell / (avg_buy + 1e-9)
        sellers_dominant = delta_ratio > 2.0
        buyers_absorbing = (avg_buy / (avg_sell + 1e-9)) > 2.0
    else:
        # For longs: we want buy_vol > sell_vol at TP1
        delta_ratio      = avg_buy / (avg_sell + 1e-9)
        sellers_dominant = (avg_sell / (avg_buy + 1e-9)) > 2.0
        buyers_absorbing = delta_ratio > 2.0

    notes.append(f"Avg buy vol={avg_buy:.0f} | sell vol={avg_sell:.0f} "
                 f"| ratio={delta_ratio:.2f}")

    # ── Scenario classification ───────────────────────────────

    # Scenario C: exhaustion — volume collapsed
    if volume_thin:
        notes.append("Scenario C: EXHAUSTION — volume collapsed at TP1")
        notes.append("Both sides thin — move is done for this session")
        return FootprintReading(
            scenario    = "C_exit",
            decision    = "EXIT_FULL",
            avg_buy_vol = round(avg_buy, 1),
            avg_sell_vol= round(avg_sell, 1),
            delta_ratio = round(delta_ratio, 2),
            volume_thin = True,
            confidence  = 85,
            notes       = notes
        )

    # Scenario A: direction-aligned volume dominant — hold
    if direction == "short" and sellers_dominant:
        notes.append(f"Scenario A: SELLERS dominant at TP1 "
                     f"(sell {avg_sell:.0f} vs buy {avg_buy:.0f})")
        notes.append("Hold 50% for TP2 — move has continuation fuel")
        return FootprintReading(
            scenario    = "A_hold",
            decision    = "HOLD_FOR_TP2",
            avg_buy_vol = round(avg_buy, 1),
            avg_sell_vol= round(avg_sell, 1),
            delta_ratio = round(delta_ratio, 2),
            volume_thin = False,
            confidence  = min(int(delta_ratio * 30), 95),
            notes       = notes
        )

    if direction == "long" and buyers_absorbing:
        notes.append(f"Scenario A: BUYERS dominant at TP1 "
                     f"(buy {avg_buy:.0f} vs sell {avg_sell:.0f})")
        notes.append("Hold 50% for TP2 — move has continuation fuel")
        return FootprintReading(
            scenario    = "A_hold",
            decision    = "HOLD_FOR_TP2",
            avg_buy_vol = round(avg_buy, 1),
            avg_sell_vol= round(avg_sell, 1),
            delta_ratio = round(delta_ratio, 2),
            volume_thin = False,
            confidence  = min(int(delta_ratio * 30), 95),
            notes       = notes
        )

    # Scenario B: opposing volume absorbing — exit now
    if direction == "short" and buyers_absorbing:
        notes.append(f"Scenario B: BUYERS ABSORBING at TP1 "
                     f"(buy {avg_buy:.0f} vs sell {avg_sell:.0f})")
        notes.append("Institutions buying at VWAP — EXIT 100% now")
        return FootprintReading(
            scenario    = "B_exit",
            decision    = "EXIT_FULL",
            avg_buy_vol = round(avg_buy, 1),
            avg_sell_vol= round(avg_sell, 1),
            delta_ratio = round(delta_ratio, 2),
            volume_thin = False,
            confidence  = 80,
            notes       = notes
        )

    if direction == "long" and sellers_dominant:
        notes.append(f"Scenario B: SELLERS ABSORBING at TP1 "
                     f"(sell {avg_sell:.0f} vs buy {avg_buy:.0f})")
        notes.append("Institutions selling at VWAP — EXIT 100% now")
        return FootprintReading(
            scenario    = "B_exit",
            decision    = "EXIT_FULL",
            avg_buy_vol = round(avg_buy, 1),
            avg_sell_vol= round(avg_sell, 1),
            delta_ratio = round(delta_ratio, 2),
            volume_thin = False,
            confidence  = 80,
            notes       = notes
        )

    # Neutral — no strong signal either way
    notes.append("Neutral footprint at TP1 — no strong bias")
    notes.append("Default: exit 50% at TP1, hold 50% with SL at breakeven")
    return FootprintReading(
        scenario    = "neutral",
        decision    = "HOLD_FOR_TP2",
        avg_buy_vol = round(avg_buy, 1),
        avg_sell_vol= round(avg_sell, 1),
        delta_ratio = round(delta_ratio, 2),
        volume_thin = False,
        confidence  = 40,
        notes       = notes
    )
