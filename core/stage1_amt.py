# ============================================================
# G7FX Signal App — Stage 1: AMT Context Engine
# ============================================================
# Implements:
#   - Volume profile builder (POC, VAH, VAL)
#   - Market state detector (balanced vs imbalanced)
#   - Supply/demand dominance detector
#   - Session open type classifier
#   - Value migration tracker (bullish / bearish / none)
#   - Profile shape classifier (D, P, b, skewed)
# ============================================================

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

from config.settings import PROFILE_TICK_SIZE, VALUE_AREA_PERCENT

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────

@dataclass
class VolumeProfile:
    poc: float                    # Point of Control
    vah: float                    # Value Area High
    val: float                    # Value Area Low
    va_volume: float              # Volume inside value area
    total_volume: float           # Total session volume
    shape: str = "unknown"        # D / P / b / skewed
    distribution: dict = field(default_factory=dict)   # price → volume map


@dataclass
class AMTContext:
    market_state: str             # "balanced" | "imbalanced"
    dominance: str                # "buyers" | "sellers" | "balanced"
    open_type: str                # "inside_va" | "above_va" | "below_va" | "open_drive"
    migration: str                # "bullish" | "bearish" | "none"
    profile: VolumeProfile = None
    prev_profile: VolumeProfile = None
    score: int = 0                # Stage 1 contribution (0–35)
    bias: str = "none"            # "long" | "short" | "none"
    notes: list = field(default_factory=list)


# ── Volume Profile Builder ────────────────────────────────────

def build_volume_profile(df: pd.DataFrame,
                          tick_size: float = PROFILE_TICK_SIZE,
                          va_pct: float = VALUE_AREA_PERCENT) -> VolumeProfile:
    """
    Build a volume profile from OHLCV candle data.
    Uses tick volume as a proxy (distributed across price range per candle).
    """
    price_vol = {}

    for _, row in df.iterrows():
        lo = row["low"]
        hi = row["high"]
        vol = row["volume"]
        candle_range = hi - lo

        if candle_range < tick_size:
            # Doji / flat candle — all volume at close
            bucket = _round_to_tick(row["close"], tick_size)
            price_vol[bucket] = price_vol.get(bucket, 0) + vol
            continue

        # Distribute volume proportionally across price buckets
        price = lo
        while price <= hi + tick_size * 0.5:
            bucket = _round_to_tick(price, tick_size)
            # Bell-curve weighting: more volume at mid + close
            dist_from_mid = abs(bucket - (lo + hi) / 2) / (candle_range / 2 + 1e-9)
            weight = max(0.1, 1 - 0.5 * dist_from_mid)
            price_vol[bucket] = price_vol.get(bucket, 0) + vol * weight * tick_size / candle_range
            price += tick_size

    if not price_vol:
        raise ValueError("Empty price_vol — check candle data")

    total_vol = sum(price_vol.values())
    sorted_levels = sorted(price_vol.keys())

    # POC = highest volume price
    poc = max(price_vol, key=price_vol.get)

    # Value Area: start from POC, expand outward until 70% of volume captured
    va_target = total_vol * va_pct
    va_vol = price_vol[poc]
    above_idx = sorted_levels.index(poc) + 1
    below_idx = sorted_levels.index(poc) - 1

    while va_vol < va_target:
        vol_above = price_vol.get(sorted_levels[above_idx], 0) if above_idx < len(sorted_levels) else 0
        vol_below = price_vol.get(sorted_levels[below_idx], 0) if below_idx >= 0 else 0

        if vol_above >= vol_below and above_idx < len(sorted_levels):
            va_vol += vol_above
            above_idx += 1
        elif below_idx >= 0:
            va_vol += vol_below
            below_idx -= 1
        else:
            break

    vah = sorted_levels[min(above_idx, len(sorted_levels) - 1)]
    val = sorted_levels[max(below_idx, 0)]

    profile = VolumeProfile(
        poc=poc, vah=vah, val=val,
        va_volume=va_vol, total_volume=total_vol,
        distribution=price_vol
    )
    profile.shape = classify_profile_shape(profile, df)
    return profile


def _round_to_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 6)


# ── Profile Shape Classifier ──────────────────────────────────

def classify_profile_shape(profile: VolumeProfile, df: pd.DataFrame) -> str:
    """
    Classify profile shape based on G7FX Stage 1.7:
    D = balanced bell curve
    P = buying dominance (value built at top)
    b = selling dominance (value built at bottom)
    skewed = strong imbalance / trend day
    """
    poc = profile.poc
    vah = profile.vah
    val = profile.val
    mid = (vah + val) / 2
    total_range = vah - val

    if total_range < 0.01:
        return "doji"

    poc_position = (poc - val) / total_range  # 0 = at val, 1 = at vah

    # VA width relative to total session range
    session_range = df["high"].max() - df["low"].min()
    va_width_ratio = total_range / (session_range + 1e-9)

    if va_width_ratio < 0.35:
        return "skewed"        # Thin profile = trend day imbalance
    elif poc_position > 0.65:
        return "P"             # POC near top = buying dominance
    elif poc_position < 0.35:
        return "b"             # POC near bottom = selling dominance
    else:
        return "D"             # POC near middle = balanced


# ── Supply / Demand Dominance ─────────────────────────────────

def detect_dominance(df: pd.DataFrame, lookback: int = 20) -> str:
    """
    Detect whether buyers, sellers, or neither dominate.
    Uses candle body direction + volume weighting over recent candles.
    """
    recent = df.tail(lookback).copy()
    recent["body"] = recent["close"] - recent["open"]
    recent["buy_vol"]  = recent.apply(lambda r: r["volume"] if r["body"] > 0 else 0, axis=1)
    recent["sell_vol"] = recent.apply(lambda r: r["volume"] if r["body"] < 0 else 0, axis=1)

    total_buy  = recent["buy_vol"].sum()
    total_sell = recent["sell_vol"].sum()
    total = total_buy + total_sell

    if total == 0:
        return "balanced"

    buy_ratio = total_buy / total

    if buy_ratio > 0.60:
        return "buyers"
    elif buy_ratio < 0.40:
        return "sellers"
    else:
        return "balanced"


# ── Session Open Type ─────────────────────────────────────────

def classify_open_type(open_price: float,
                        prev_profile: VolumeProfile,
                        first_n_candles: pd.DataFrame,
                        drive_threshold_pips: float = 20) -> str:
    """
    Classify today's session open relative to yesterday's value area.
    G7FX Stage 1.6 logic.
    """
    vah = prev_profile.vah
    val = prev_profile.val

    if val <= open_price <= vah:
        base_type = "inside_va"
    elif open_price > vah:
        base_type = "above_va"
    else:
        base_type = "below_va"

    # Check for open drive: first N candles move strongly one way
    if len(first_n_candles) >= 3:
        first_move = first_n_candles["close"].iloc[-1] - first_n_candles["open"].iloc[0]
        if abs(first_move) > drive_threshold_pips * 0.01:  # JPY: 1 pip = 0.01
            return "open_drive"

    return base_type


# ── Value Migration ───────────────────────────────────────────

def detect_value_migration(profiles: list) -> str:
    """
    Compare the last two session profiles to detect value migration.
    G7FX Stage 2.6 / 2.7 logic.

    profiles: list of VolumeProfile, most recent last.
    Returns: "bullish" | "bearish" | "none"
    """
    if len(profiles) < 2:
        return "none"

    prev = profiles[-2]
    curr = profiles[-1]

    overlap = min(prev.vah, curr.vah) - max(prev.val, curr.val)
    prev_range = prev.vah - prev.val

    if prev_range <= 0:
        return "none"

    overlap_ratio = overlap / prev_range

    if overlap_ratio < 0.30:
        if curr.poc > prev.vah:
            return "bullish"
        elif curr.poc < prev.val:
            return "bearish"

    # Even with overlap, track POC direction
    if curr.poc > prev.poc * 1.002:
        return "bullish"
    elif curr.poc < prev.poc * 0.998:
        return "bearish"

    return "none"


# ── Main Stage 1 Evaluator ────────────────────────────────────

def evaluate_stage1(h4_df: pd.DataFrame,
                     h1_df: pd.DataFrame,
                     prev_profile: Optional[VolumeProfile] = None) -> AMTContext:
    """
    Run all Stage 1 checks and return an AMTContext object.
    """
    score = 0
    notes = []

    # Build current session profile from H1 data
    curr_profile = build_volume_profile(h1_df)

    # 1. Market state
    session_range = h1_df["high"].max() - h1_df["low"].min()
    va_range = curr_profile.vah - curr_profile.val
    va_ratio = va_range / (session_range + 1e-9)

    if curr_profile.shape in ("D",) and va_ratio > 0.5:
        market_state = "balanced"
    else:
        market_state = "imbalanced"
        score += 15
        notes.append(f"Imbalanced market (shape: {curr_profile.shape})")

    # 2. Dominance
    dominance = detect_dominance(h4_df)
    if dominance != "balanced":
        score += 10
        notes.append(f"Dominance: {dominance}")

    # 3. Value migration
    profiles = [prev_profile, curr_profile] if prev_profile else [curr_profile]
    migration = detect_value_migration(profiles) if prev_profile else "none"
    if migration != "none":
        score += 10
        notes.append(f"Value migration: {migration}")

    # 4. Open type
    open_price = h1_df["open"].iloc[0]
    first_candles = h1_df.head(4)

    if prev_profile:
        open_type = classify_open_type(open_price, prev_profile, first_candles)
    else:
        open_type = "unknown"

    if open_type in ("above_va", "below_va", "open_drive"):
        score += 5
        notes.append(f"Open type: {open_type}")

    # 5. Determine overall bias
    bias = "none"
    if dominance == "sellers" and migration in ("bearish", "none"):
        bias = "short"
    elif dominance == "buyers" and migration in ("bullish", "none"):
        bias = "long"
    elif migration == "bearish":
        bias = "short"
    elif migration == "bullish":
        bias = "long"

    ctx = AMTContext(
        market_state=market_state,
        dominance=dominance,
        open_type=open_type,
        migration=migration,
        profile=curr_profile,
        prev_profile=prev_profile,
        score=min(score, 35),
        bias=bias,
        notes=notes
    )

    logger.info(f"Stage 1 | state={market_state} | dom={dominance} | "
                f"migration={migration} | open={open_type} | bias={bias} | score={score}")
    return ctx
