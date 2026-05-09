# ============================================================
# G7FX Signal App — Stage 2: Hypothesis Builder
# ============================================================
# Implements:
#   - VWAP + standard deviation bands
#   - Anchored VWAP
#   - Confluence zone detection (static + dynamic value)
#   - M15 rejection candle confirmation
#   - Entry, SL, TP calculation
# ============================================================

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

from config.settings import (
    CONFLUENCE_TOLERANCE_PIPS, MIN_CONFLUENCE_LEVELS,
    MIN_RR_RATIO, VWAP_STD_DEV_1, VWAP_STD_DEV_2
)
from core.stage1_amt import AMTContext, VolumeProfile

logger = logging.getLogger(__name__)

PIP = 0.01  # 1 pip for JPY pairs


# ── Data Structures ──────────────────────────────────────────

@dataclass
class VWAPData:
    vwap: float
    upper_1: float    # +1σ
    lower_1: float    # -1σ
    upper_2: float    # +2σ
    lower_2: float    # -2σ
    anchored_vwap: Optional[float] = None


@dataclass
class ConfluenceZone:
    price: float          # Centre of the zone
    zone_high: float      # Upper bound
    zone_low: float       # Lower bound
    direction: str        # "long" | "short"
    levels: list = field(default_factory=list)   # Which levels form the zone
    level_count: int = 0
    score: int = 0


@dataclass
class SignalHypothesis:
    pair: str
    direction: str          # "BUY" | "SELL"
    entry_low: float
    entry_high: float
    entry_mid: float
    stop_loss: float
    tp1: float              # VWAP target
    tp2: float              # Profile level target
    rr_tp1: float
    rr_tp2: float
    confluence: ConfluenceZone = None
    stage2_score: int = 0
    confirmed: bool = False  # M15 rejection candle confirmed
    notes: list = field(default_factory=list)


# ── VWAP Calculator ───────────────────────────────────────────

def calculate_vwap(df: pd.DataFrame) -> VWAPData:
    """
    Calculate session VWAP with ±1σ and ±2σ bands.
    Uses tick volume as proxy weight.
    """
    df = df.copy()
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"] = df["typical_price"] * df["volume"]

    cumulative_tp_vol = df["tp_vol"].cumsum()
    cumulative_vol    = df["volume"].cumsum()
    df["vwap"]        = cumulative_tp_vol / cumulative_vol

    # Standard deviation bands
    df["deviation"] = df["typical_price"] - df["vwap"]
    df["dev_sq"]    = df["deviation"] ** 2
    df["variance"]  = (df["dev_sq"] * df["volume"]).cumsum() / cumulative_vol

    current_vwap = df["vwap"].iloc[-1]
    std_dev      = np.sqrt(df["variance"].iloc[-1])

    return VWAPData(
        vwap    = round(current_vwap, 3),
        upper_1 = round(current_vwap + VWAP_STD_DEV_1 * std_dev, 3),
        lower_1 = round(current_vwap - VWAP_STD_DEV_1 * std_dev, 3),
        upper_2 = round(current_vwap + VWAP_STD_DEV_2 * std_dev, 3),
        lower_2 = round(current_vwap - VWAP_STD_DEV_2 * std_dev, 3),
    )


def calculate_anchored_vwap(df: pd.DataFrame, anchor_idx: int) -> float:
    """
    Calculate VWAP anchored to a specific candle index (e.g. swing high/low).
    anchor_idx: row position to start VWAP calculation from.
    """
    sub = df.iloc[anchor_idx:].copy()
    if sub.empty:
        return None

    sub["typical_price"] = (sub["high"] + sub["low"] + sub["close"]) / 3
    sub["tp_vol"] = sub["typical_price"] * sub["volume"]

    avwap = sub["tp_vol"].cumsum().iloc[-1] / sub["volume"].cumsum().iloc[-1]
    return round(avwap, 3)


def find_swing_anchor(df: pd.DataFrame, lookback: int = 50) -> int:
    """
    Find the most recent significant swing high or low to anchor VWAP.
    Returns the index position of the anchor candle.
    """
    recent = df.tail(lookback)
    # Find the candle with the most extreme high or low in recent history
    high_idx = recent["high"].idxmax()
    low_idx  = recent["low"].idxmin()

    # Use whichever is more recent
    hi_pos = df.index.get_loc(high_idx)
    lo_pos = df.index.get_loc(low_idx)
    return max(hi_pos, lo_pos)


# ── Confluence Zone Detector ──────────────────────────────────

def find_confluence_zones(price: float,
                           vwap: VWAPData,
                           curr_profile: VolumeProfile,
                           prev_profile: Optional[VolumeProfile],
                           direction: str,
                           tolerance_pips: float = CONFLUENCE_TOLERANCE_PIPS) -> Optional[ConfluenceZone]:
    """
    Find zones where multiple static + dynamic value levels converge.
    G7FX Stage 2 core: hypothesis only forms at confluence.

    direction: "long" (looking for support levels) | "short" (looking for resistance)
    """
    tol = tolerance_pips * PIP

    # --- Collect all relevant levels ---
    candidates = []

    # Dynamic (VWAP-based)
    candidates += [
        ("VWAP",       vwap.vwap),
        ("VWAP +1σ",   vwap.upper_1),
        ("VWAP -1σ",   vwap.lower_1),
        ("VWAP +2σ",   vwap.upper_2),
        ("VWAP -2σ",   vwap.lower_2),
    ]
    if vwap.anchored_vwap:
        candidates.append(("Anchored VWAP", vwap.anchored_vwap))

    # Static (current profile)
    if curr_profile:
        candidates += [
            ("Curr POC", curr_profile.poc),
            ("Curr VAH", curr_profile.vah),
            ("Curr VAL", curr_profile.val),
        ]

    # Static (previous session profile)
    if prev_profile:
        candidates += [
            ("Prev POC", prev_profile.poc),
            ("Prev VAH", prev_profile.vah),
            ("Prev VAL", prev_profile.val),
        ]

    # --- Filter: only levels relevant to direction ---
    # For longs: levels below or at current price (support)
    # For shorts: levels above or at current price (resistance)
    if direction == "long":
        relevant = [(name, lvl) for name, lvl in candidates
                    if price - tol * 5 <= lvl <= price + tol * 2]
    else:
        relevant = [(name, lvl) for name, lvl in candidates
                    if price - tol * 2 <= lvl <= price + tol * 5]

    if not relevant:
        return None

    # --- Group nearby levels into zones ---
    # Sort by price and cluster levels within tolerance
    relevant.sort(key=lambda x: x[1])
    zone_levels = []
    zone_prices = []
    used = set()

    for i, (name_i, price_i) in enumerate(relevant):
        if i in used:
            continue
        cluster_names  = [name_i]
        cluster_prices = [price_i]
        used.add(i)

        for j, (name_j, price_j) in enumerate(relevant):
            if j in used:
                continue
            if abs(price_j - price_i) <= tol:
                cluster_names.append(name_j)
                cluster_prices.append(price_j)
                used.add(j)

        if len(cluster_names) >= MIN_CONFLUENCE_LEVELS:
            zone_levels.append(cluster_names)
            zone_prices.append(cluster_prices)

    if not zone_levels:
        return None

    # Pick the zone closest to current price
    zone_centres = [np.mean(z) for z in zone_prices]
    best_idx = np.argmin([abs(c - price) for c in zone_centres])

    best_prices = zone_prices[best_idx]
    best_names  = zone_levels[best_idx]
    centre      = np.mean(best_prices)
    spread      = max(best_prices) - min(best_prices)

    score = min(len(best_names) * 10, 35)  # Up to 35 points for Stage 2

    zone = ConfluenceZone(
        price      = round(centre, 3),
        zone_high  = round(centre + spread / 2 + tol, 3),
        zone_low   = round(centre - spread / 2 - tol, 3),
        direction  = direction,
        levels     = best_names,
        level_count= len(best_names),
        score      = score
    )

    logger.info(f"Confluence zone found | {direction} | {centre:.3f} | "
                f"levels={best_names} | score={score}")
    return zone


# ── Rejection Candle Checker (Stage 2 / Stage 3 bridge) ──────

def check_rejection_candle(m15_df: pd.DataFrame,
                            zone: ConfluenceZone,
                            direction: str) -> bool:
    """
    Check if the most recent M15 candles show a rejection from the confluence zone.
    G7FX Stage 2.5: confirmation candle before entry.

    A valid rejection candle:
    - Touches the zone (wick into zone)
    - Closes AWAY from the zone (body outside)
    - Has meaningful wick-to-body ratio
    """
    if m15_df is None or len(m15_df) < 2:
        return False

    last = m15_df.iloc[-1]
    prev = m15_df.iloc[-2]

    body_size = abs(last["close"] - last["open"])
    candle_range = last["high"] - last["low"]

    if candle_range < PIP * 3:  # Too small to be meaningful
        return False

    wick_ratio = body_size / (candle_range + 1e-9)

    if direction == "long":
        # Bullish rejection: wick pierces zone from below, closes above zone_low
        touched_zone = last["low"] <= zone.zone_high and last["low"] >= zone.zone_low * 0.999
        closes_away  = last["close"] > zone.zone_low
        bullish_body = last["close"] > last["open"]
        has_wick     = (last["open"] - last["low"]) > body_size * 0.5

        return touched_zone and closes_away and bullish_body and wick_ratio < 0.75

    else:  # short
        # Bearish rejection: wick pierces zone from above, closes below zone_high
        touched_zone = last["high"] >= zone.zone_low and last["high"] <= zone.zone_high * 1.001
        closes_away  = last["close"] < zone.zone_high
        bearish_body = last["close"] < last["open"]
        has_wick     = (last["high"] - last["open"]) > body_size * 0.5

        return touched_zone and closes_away and bearish_body and wick_ratio < 0.75


# ── Entry / SL / TP Builder ───────────────────────────────────

def build_signal_hypothesis(pair: str,
                             direction: str,
                             zone: ConfluenceZone,
                             vwap: VWAPData,
                             curr_profile: VolumeProfile,
                             prev_profile: Optional[VolumeProfile],
                             confirmed: bool,
                             stage2_score: int) -> Optional[SignalHypothesis]:
    """
    Build a complete trade hypothesis with entry, SL, TP1, TP2.
    G7FX Stage 2.5 entry/exit logic.
    """
    notes = []

    if direction == "long":
        entry_low  = zone.zone_low
        entry_high = zone.price + PIP * 3
        entry_mid  = (entry_low + entry_high) / 2
        stop_loss  = zone.zone_low - PIP * 5   # Below zone = invalidation

        # TP1: VWAP (mean reversion)
        tp1 = vwap.vwap
        # TP2: POC or VAH (profile level)
        tp2 = curr_profile.vah if curr_profile else (vwap.vwap + PIP * 50)

        if prev_profile and prev_profile.poc > vwap.vwap:
            tp2 = prev_profile.poc
            notes.append("TP2 = prev session POC")

    else:  # short
        entry_low  = zone.price - PIP * 3
        entry_high = zone.zone_high
        entry_mid  = (entry_low + entry_high) / 2
        stop_loss  = zone.zone_high + PIP * 5  # Above zone = invalidation

        # TP1: VWAP
        tp1 = vwap.vwap
        # TP2: VAL or POC below
        tp2 = curr_profile.val if curr_profile else (vwap.vwap - PIP * 50)

        if prev_profile and prev_profile.poc < vwap.vwap:
            tp2 = prev_profile.poc
            notes.append("TP2 = prev session POC")

    # Validate R:R
    sl_dist  = abs(entry_mid - stop_loss)
    rr_tp1   = abs(tp1 - entry_mid) / (sl_dist + 1e-9)
    rr_tp2   = abs(tp2 - entry_mid) / (sl_dist + 1e-9)

    if rr_tp1 < MIN_RR_RATIO:
        logger.info(f"Signal rejected — R:R {rr_tp1:.2f} below minimum {MIN_RR_RATIO}")
        return None

    if confirmed:
        notes.append("M15 rejection candle confirmed ✓")
        stage2_score = min(stage2_score + 10, 35)

    return SignalHypothesis(
        pair       = pair,
        direction  = direction,
        entry_low  = round(entry_low, 3),
        entry_high = round(entry_high, 3),
        entry_mid  = round(entry_mid, 3),
        stop_loss  = round(stop_loss, 3),
        tp1        = round(tp1, 3),
        tp2        = round(tp2, 3),
        rr_tp1     = round(rr_tp1, 2),
        rr_tp2     = round(rr_tp2, 2),
        confluence = zone,
        stage2_score = stage2_score,
        confirmed  = confirmed,
        notes      = notes
    )


# ── Main Stage 2 Evaluator ────────────────────────────────────

def evaluate_stage2(h1_df: pd.DataFrame,
                     m15_df: pd.DataFrame,
                     amt_ctx: AMTContext,
                     pair: str = "USD_JPY") -> Optional[SignalHypothesis]:
    """
    Run all Stage 2 checks and return a SignalHypothesis if valid.
    """
    if amt_ctx.market_state == "balanced":
        logger.info("Stage 2 skipped — balanced market (no directional hypothesis)")
        return None

    if amt_ctx.bias == "none":
        logger.info("Stage 2 skipped — no directional bias from Stage 1")
        return None

    direction = "long" if amt_ctx.bias == "long" else "short"

    # Calculate VWAP
    vwap = calculate_vwap(h1_df)

    # Anchored VWAP (anchor to most recent swing)
    anchor_idx = find_swing_anchor(h1_df)
    avwap = calculate_anchored_vwap(h1_df, anchor_idx)
    vwap.anchored_vwap = avwap

    # Current price
    current_price = m15_df["close"].iloc[-1] if m15_df is not None else h1_df["close"].iloc[-1]

    # Find confluence zone
    zone = find_confluence_zones(
        price        = current_price,
        vwap         = vwap,
        curr_profile = amt_ctx.profile,
        prev_profile = amt_ctx.prev_profile,
        direction    = direction,
    )

    if zone is None:
        logger.info(f"Stage 2: No confluence zone found at {current_price:.3f}")
        return None

    # Check for rejection candle confirmation
    confirmed = check_rejection_candle(m15_df, zone, direction)

    # Build hypothesis
    hypothesis = build_signal_hypothesis(
        pair         = pair,
        direction    = direction,
        zone         = zone,
        vwap         = vwap,
        curr_profile = amt_ctx.profile,
        prev_profile = amt_ctx.prev_profile,
        confirmed    = confirmed,
        stage2_score = zone.score
    )

    if hypothesis:
        logger.info(f"Stage 2 hypothesis built | {direction} | entry={hypothesis.entry_mid:.3f} "
                    f"| SL={hypothesis.stop_loss:.3f} | TP1={hypothesis.tp1:.3f} "
                    f"| R:R={hypothesis.rr_tp1}")

    return hypothesis
