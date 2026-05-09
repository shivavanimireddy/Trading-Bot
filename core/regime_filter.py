# ============================================================
# G7FX Signal Engine — Regime Filter (Gap 1 Fix)
# ============================================================
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class RegimeReading:
    adx: float
    vol_ratio: float
    regime: str              # NORMAL | CAUTIOUS | HALF_SIZE | DIRECTION_ONLY | SUPPRESS
    size_multiplier: float
    min_score: int
    notes: list = field(default_factory=list)

def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    df = df.copy()
    df['prev_high']  = df['high'].shift(1)
    df['prev_low']   = df['low'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['prev_close']),
                   abs(df['low']  - df['prev_close']))
    )
    df['dm_plus']  = np.where(
        (df['high'] - df['prev_high']) > (df['prev_low'] - df['low']),
        np.maximum(df['high'] - df['prev_high'], 0), 0)
    df['dm_minus'] = np.where(
        (df['prev_low'] - df['low']) > (df['high'] - df['prev_high']),
        np.maximum(df['prev_low'] - df['low'], 0), 0)
    atr      = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    di_plus  = 100 * df['dm_plus'].ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9)
    di_minus = 100 * df['dm_minus'].ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9)
    dx  = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return round(float(adx.iloc[-1]), 2)

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['prev_close']),
                   abs(df['low']  - df['prev_close']))
    )
    return round(float(df['tr'].ewm(alpha=1/period, adjust=False).mean().iloc[-1]), 5)

def calculate_vol_ratio(df: pd.DataFrame, current_atr: float, lookback: int = 20) -> float:
    atrs, step = [], max(1, len(df) // lookback)
    for i in range(lookback):
        s = df.iloc[i*step : i*step + step + 14]
        if len(s) >= 14:
            atrs.append(calculate_atr(s))
    if not atrs:
        return 1.0
    return round(current_atr / (np.mean(atrs) + 1e-9), 3)

def evaluate_regime(h4_df: pd.DataFrame) -> RegimeReading:
    notes  = []
    adx    = calculate_adx(h4_df)
    c_atr  = calculate_atr(h4_df)
    vr     = calculate_vol_ratio(h4_df, c_atr)
    notes += [f"ADX(14)={adx}", f"VolRatio={vr:.2f}x"]

    if adx > 30 and vr > 1.2:
        r, sz, ms = "SUPPRESS",        0.0,  999
        notes.append("Strong trend + elevated vol — NO TRADE")
    elif adx > 30:
        r, sz, ms = "DIRECTION_ONLY",  0.75,  75
        notes.append("Trending — only trade with migration direction")
    elif vr > 1.4:
        r, sz, ms = "HALF_SIZE",       0.5,   80
        notes.append("Volatility spike — half size, high confidence only")
    elif vr > 1.2 or adx > 20:
        r, sz, ms = "CAUTIOUS",        0.75,  70
        notes.append("Transitional conditions")
    else:
        r, sz, ms = "NORMAL",          1.0,   60
        notes.append("Ideal mean-reversion conditions")

    logger.info(f"Regime={r} | ADX={adx} | VolRatio={vr:.2f} | size={sz}x")
    return RegimeReading(adx=adx, vol_ratio=vr, regime=r,
                         size_multiplier=sz, min_score=ms, notes=notes)
