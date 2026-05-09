# ============================================================
# G7FX Signal App — Data Feed (OANDA v20 API)
# ============================================================
# Fetches OHLCV candles for H4, H1, M15 timeframes.
# Uses oandapyV20 library: pip install oandapyV20
# ============================================================

import pandas as pd
import logging
from datetime import datetime, timezone

try:
    import oandapyV20
    import oandapyV20.endpoints.instruments as instruments
    import oandapyV20.endpoints.accounts as accounts
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False
    logging.warning("oandapyV20 not installed. Run: pip install oandapyV20")

from config.settings import (
    OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENVIRONMENT,
    INSTRUMENT, CANDLE_COUNT
)

logger = logging.getLogger(__name__)


class OandaFeed:
    """
    Connects to OANDA and fetches OHLCV candle data.
    Supports H4, H1, M15 granularities.
    """

    GRANULARITY_MAP = {
        "M5":  "M5",
        "M15": "M15",
        "H1":  "H1",
        "H4":  "H4",
        "D":   "D",
    }

    def __init__(self):
        if not OANDA_AVAILABLE:
            raise ImportError("Install oandapyV20: pip install oandapyV20")

        env = "live" if OANDA_ENVIRONMENT == "live" else "practice"
        self.client = oandapyV20.API(
            access_token=OANDA_API_KEY,
            environment=env
        )
        self.instrument = INSTRUMENT
        logger.info(f"OandaFeed initialised | {INSTRUMENT} | {OANDA_ENVIRONMENT}")

    def get_candles(self, granularity: str, count: int = CANDLE_COUNT) -> pd.DataFrame:
        """
        Fetch OHLCV candles from OANDA.

        Returns DataFrame with columns:
            time, open, high, low, close, volume
        """
        gran = self.GRANULARITY_MAP.get(granularity)
        if not gran:
            raise ValueError(f"Unsupported granularity: {granularity}")

        params = {
            "granularity": gran,
            "count": count,
            "price": "M"  # Mid prices
        }

        r = instruments.InstrumentsCandles(
            instrument=self.instrument,
            params=params
        )
        self.client.request(r)

        candles = r.response.get("candles", [])
        rows = []
        for c in candles:
            if c.get("complete", False):
                mid = c["mid"]
                rows.append({
                    "time":   pd.to_datetime(c["time"]),
                    "open":   float(mid["o"]),
                    "high":   float(mid["h"]),
                    "low":    float(mid["l"]),
                    "close":  float(mid["c"]),
                    "volume": int(c["volume"]),  # tick volume (proxy)
                })

        df = pd.DataFrame(rows)
        df.set_index("time", inplace=True)
        df.index = df.index.tz_convert("UTC")
        logger.info(f"Fetched {len(df)} {granularity} candles for {self.instrument}")
        return df

    def get_account_balance(self) -> float:
        """Fetch current account NAV."""
        r = accounts.AccountDetails(OANDA_ACCOUNT_ID)
        self.client.request(r)
        nav = float(r.response["account"]["NAV"])
        logger.info(f"Account NAV: {nav}")
        return nav

    def get_current_price(self) -> float:
        """Get current mid price."""
        params = {"granularity": "M1", "count": 1, "price": "M"}
        r = instruments.InstrumentsCandles(instrument=self.instrument, params=params)
        self.client.request(r)
        last = r.response["candles"][-1]["mid"]["c"]
        return float(last)


# ---- Demo / offline mode for testing without OANDA key ----

def get_synthetic_candles(granularity: str, count: int = 200) -> pd.DataFrame:
    """
    Generates synthetic USDJPY candles centred around current
    market conditions (157.20 base) for testing without an API key.
    """
    import numpy as np

    np.random.seed(42)
    base = 157.20
    # Normalise frequency string for pandas >= 2.2
    freq_map = {"4H": "4h", "1H": "1h", "H1": "1h", "H4": "4h",
                "M15": "15min", "15T": "15min",
                "M5": "5min", "5T": "5min",
                "M1": "1min", "1T": "1min"}
    freq = freq_map.get(granularity, granularity.lower())
    dates = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=count, freq=freq)

    # Simulate post-intervention price behaviour
    returns = np.random.normal(0, 0.0008, count)
    # Add a spike at 80% mark to simulate intervention
    spike_idx = int(count * 0.80)
    returns[spike_idx] = -0.025      # -2.5% intervention drop
    returns[spike_idx + 1] = 0.008  # partial recovery

    closes = base * np.exp(np.cumsum(returns))
    opens  = np.roll(closes, 1); opens[0] = base

    highs  = np.maximum(opens, closes) + np.abs(np.random.normal(0, 0.05, count))
    lows   = np.minimum(opens, closes) - np.abs(np.random.normal(0, 0.05, count))
    vols   = np.random.randint(800, 4000, count)

    # Spike volume at intervention
    vols[spike_idx]     = 12000
    vols[spike_idx + 1] = 8000

    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols
    }, index=dates)

    return df.round(3)
