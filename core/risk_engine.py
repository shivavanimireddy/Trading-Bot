# ============================================================
# G7FX Signal App — Risk Management Engine
# ============================================================
# Implements:
#   - Position sizing (1% risk rule)
#   - Drawdown tracking + circuit breaker
#   - Kill zone checker (London / NY open)
#   - News blackout checker
#   - Final signal assembler (all stages combined)
# ============================================================

import logging
from datetime import datetime, time, timezone
from dataclasses import dataclass, field
from typing import Optional

from config.settings import (
    ACCOUNT_RISK_PERCENT, INTERVENTION_RISK_PERCENT,
    DRAWDOWN_WARN_PCT, DRAWDOWN_PAUSE_PCT, DRAWDOWN_CLOSE_PCT,
    PIP_VALUE_USDJPY, KILL_ZONES, NEWS_BLACKOUT_BEFORE,
    SCORE_EMIT_THRESHOLD, SCORE_HIGH_CONFIDENCE
)
from core.stage1_amt import AMTContext
from core.stage2_hypothesis import SignalHypothesis
from core.stage3_orderflow import OrderFlowReading

logger = logging.getLogger(__name__)

PIP = 0.01  # JPY pip


# ── Data Structures ──────────────────────────────────────────

@dataclass
class FinalSignal:
    """The complete, ready-to-emit signal with all metadata."""
    pair: str
    direction: str             # "BUY" | "SELL"
    entry_zone: tuple          # (low, high)
    entry_mid: float
    stop_loss: float
    tp1: float
    tp2: float
    rr_tp1: float
    rr_tp2: float
    lot_size: float            # Calculated position size
    risk_amount_usd: float
    confidence_score: int      # 0–100
    confidence_label: str      # "HIGH" | "MEDIUM" | "LOW"
    kill_zone: str             # Which kill zone this fires in
    stage1_score: int
    stage2_score: int
    stage3_score: int
    confluence_levels: list
    all_notes: list
    timestamp: str
    status: str                # "READY" | "PENDING_CONFIRMATION" | "SUPPRESSED"


@dataclass
class DrawdownState:
    peak_balance: float
    current_balance: float
    drawdown_pct: float = 0.0
    status: str = "normal"    # "normal" | "warn" | "paused" | "closed"


# ── Kill Zone Checker ─────────────────────────────────────────

def get_active_kill_zone(dt: Optional[datetime] = None) -> Optional[str]:
    """
    Check if current time (EST) is within a kill zone.
    Returns kill zone name or None.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    # Convert to EST (UTC-5, or UTC-4 during DST — simplified here)
    est_hour   = (dt.hour - 4) % 24   # Approximate EST
    est_minute = dt.minute
    current    = est_hour * 60 + est_minute

    for kz in KILL_ZONES:
        start_h, start_m = map(int, kz["start"].split(":"))
        end_h,   end_m   = map(int, kz["end"].split(":"))
        start = start_h * 60 + start_m
        end   = end_h   * 60 + end_m

        if start <= current <= end:
            return kz["name"]

    return None


# ── Position Sizing ───────────────────────────────────────────

def calculate_position_size(account_balance: float,
                              entry_price: float,
                              stop_loss: float,
                              high_volatility: bool = False) -> tuple:
    """
    Calculate position size based on 1% risk rule.
    Returns (lot_size, risk_amount_usd).

    For USD/JPY:
    - 1 standard lot = 100,000 units
    - pip value ≈ $6.30 per pip per mini-lot (10,000 units) at 157
    - 1 pip = 0.01 JPY
    """
    risk_pct = INTERVENTION_RISK_PERCENT if high_volatility else ACCOUNT_RISK_PERCENT
    risk_amount = account_balance * risk_pct

    sl_pips = abs(entry_price - stop_loss) / PIP
    if sl_pips <= 0:
        return 0.0, 0.0

    # pip value in USD for 1 standard lot of USDJPY ≈ 1 / current_price * 100,000
    pip_val_per_lot = 100_000 / entry_price  # in USD, per lot

    lot_size = risk_amount / (sl_pips * pip_val_per_lot)
    lot_size = round(max(0.01, lot_size), 2)   # Minimum 0.01 lot

    actual_risk = lot_size * sl_pips * pip_val_per_lot
    return lot_size, round(actual_risk, 2)


# ── Drawdown Tracker ──────────────────────────────────────────

class DrawdownTracker:
    """
    Tracks portfolio drawdown from peak NAV.
    Implements the G7FX drawdown ladder.
    """

    def __init__(self, initial_balance: float):
        self.peak = initial_balance
        self.current = initial_balance
        self.status = "normal"
        self._history = []

    def update(self, current_balance: float) -> DrawdownState:
        self.current = current_balance
        if current_balance > self.peak:
            self.peak = current_balance

        dd_pct = (self.peak - self.current) / self.peak

        if dd_pct >= DRAWDOWN_CLOSE_PCT:
            self.status = "closed"
            logger.critical(f"DRAWDOWN CIRCUIT BREAKER: {dd_pct:.1%} — CLOSE ALL POSITIONS")
        elif dd_pct >= DRAWDOWN_PAUSE_PCT:
            self.status = "paused"
            logger.warning(f"Drawdown {dd_pct:.1%} — SIGNALS PAUSED")
        elif dd_pct >= DRAWDOWN_WARN_PCT:
            self.status = "warn"
            logger.warning(f"Drawdown {dd_pct:.1%} — reducing position size")
        else:
            self.status = "normal"

        state = DrawdownState(
            peak_balance    = self.peak,
            current_balance = self.current,
            drawdown_pct    = dd_pct,
            status          = self.status
        )
        self._history.append(state)
        return state

    def can_trade(self) -> bool:
        return self.status in ("normal", "warn")

    def is_high_vol(self) -> bool:
        return self.status == "warn"


# ── Final Signal Assembler ────────────────────────────────────

def assemble_final_signal(amt_ctx: AMTContext,
                           hypothesis: SignalHypothesis,
                           of_reading: OrderFlowReading,
                           account_balance: float,
                           dd_tracker: DrawdownTracker,
                           pair: str = "USD_JPY") -> Optional[FinalSignal]:
    """
    Combine all three stages into a final signal.
    Applies kill zone, drawdown, and score filters.
    """
    all_notes = []

    # --- Drawdown gate ---
    if not dd_tracker.can_trade():
        logger.info(f"Signal blocked: drawdown status = {dd_tracker.status}")
        return None

    # --- Stage 3 suppression gate ---
    if of_reading.suppress:
        all_notes.append("Signal suppressed by Stage 3 order flow conflict")
        logger.info("Signal suppressed by Stage 3")
        return None

    # --- Kill zone check ---
    kill_zone = get_active_kill_zone()
    if not kill_zone:
        all_notes.append("Outside kill zone — hypothesis pending")
        status = "PENDING_CONFIRMATION"
    else:
        all_notes.append(f"Active kill zone: {kill_zone}")
        status = "READY"

    # --- Score assembly ---
    s1 = amt_ctx.score
    s2 = hypothesis.stage2_score
    s3 = max(of_reading.stage3_score, 0)
    total = min(s1 + s2 + s3, 100)

    if total < SCORE_EMIT_THRESHOLD:
        logger.info(f"Signal score {total} below threshold {SCORE_EMIT_THRESHOLD}")
        return None

    confidence_label = (
        "HIGH"   if total >= SCORE_HIGH_CONFIDENCE else
        "MEDIUM" if total >= SCORE_EMIT_THRESHOLD  else
        "LOW"
    )

    # --- Position sizing ---
    high_vol = dd_tracker.is_high_vol()
    lot_size, risk_usd = calculate_position_size(
        account_balance = account_balance,
        entry_price     = hypothesis.entry_mid,
        stop_loss       = hypothesis.stop_loss,
        high_volatility = high_vol
    )

    # --- Assemble notes ---
    all_notes += amt_ctx.notes
    all_notes += hypothesis.notes
    all_notes += of_reading.notes

    signal = FinalSignal(
        pair             = pair,
        direction        = hypothesis.direction,
        entry_zone       = (hypothesis.entry_low, hypothesis.entry_high),
        entry_mid        = hypothesis.entry_mid,
        stop_loss        = hypothesis.stop_loss,
        tp1              = hypothesis.tp1,
        tp2              = hypothesis.tp2,
        rr_tp1           = hypothesis.rr_tp1,
        rr_tp2           = hypothesis.rr_tp2,
        lot_size         = lot_size,
        risk_amount_usd  = risk_usd,
        confidence_score = total,
        confidence_label = confidence_label,
        kill_zone        = kill_zone or "None (pending)",
        stage1_score     = s1,
        stage2_score     = s2,
        stage3_score     = s3,
        confluence_levels= hypothesis.confluence.levels if hypothesis.confluence else [],
        all_notes        = all_notes,
        timestamp        = datetime.now(timezone.utc).isoformat(),
        status           = status
    )

    logger.info(
        f"✅ SIGNAL | {pair} {hypothesis.direction} | "
        f"entry={hypothesis.entry_mid:.3f} | SL={hypothesis.stop_loss:.3f} | "
        f"TP1={hypothesis.tp1:.3f} | TP2={hypothesis.tp2:.3f} | "
        f"score={total} ({confidence_label}) | lots={lot_size}"
    )
    return signal


# ── Signal Formatter ──────────────────────────────────────────

def format_signal_alert(signal: FinalSignal) -> str:
    """
    Format a signal into a clean human-readable alert string.
    Used for push notifications, email, and webhook payloads.
    """
    direction_emoji = "🔴 SELL" if signal.direction == "SELL" else "🟢 BUY"
    conf_emoji = "🔥" if signal.confidence_label == "HIGH" else ("⚡" if signal.confidence_label == "MEDIUM" else "📡")

    lines = [
        f"{'='*48}",
        f"  G7FX SIGNAL — {signal.pair.replace('_','/')}",
        f"{'='*48}",
        f"  Direction   : {direction_emoji}",
        f"  Entry zone  : {signal.entry_zone[0]:.3f} – {signal.entry_zone[1]:.3f}",
        f"  Entry mid   : {signal.entry_mid:.3f}",
        f"  Stop loss   : {signal.stop_loss:.3f}",
        f"  TP1 (VWAP) : {signal.tp1:.3f}  (R:R {signal.rr_tp1}:1)",
        f"  TP2 (level) : {signal.tp2:.3f}  (R:R {signal.rr_tp2}:1)",
        f"  Lot size    : {signal.lot_size}",
        f"  Risk (USD)  : ${signal.risk_amount_usd:.2f}",
        f"  Confidence  : {conf_emoji} {signal.confidence_score}/100 ({signal.confidence_label})",
        f"  Kill zone   : {signal.kill_zone}",
        f"  Status      : {signal.status}",
        f"  Scores      : S1={signal.stage1_score} | S2={signal.stage2_score} | S3={signal.stage3_score}",
        f"  Confluence  : {', '.join(signal.confluence_levels)}",
        f"  Time (UTC)  : {signal.timestamp[:19]}",
        f"{'='*48}",
        f"  Notes:",
    ]
    for note in signal.all_notes:
        lines.append(f"    • {note}")
    lines.append(f"{'='*48}")

    return "\n".join(lines)
