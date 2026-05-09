# ============================================================
# G7FX Signal App — Configuration
# ============================================================

# --- OANDA API ---
OANDA_API_KEY = "YOUR_OANDA_API_KEY"       # Replace with your key
OANDA_ACCOUNT_ID = "YOUR_ACCOUNT_ID"       # Replace with your account ID
OANDA_ENVIRONMENT = "practice"             # "practice" or "live"

# --- Instrument ---
INSTRUMENT = "USD_JPY"

# --- Timeframes ---
TF_CONTEXT   = "H4"    # Stage 1 AMT context
TF_ENTRY     = "H1"    # Stage 2 confluence detection
TF_CONFIRM   = "M15"   # Stage 3 confirmation candle
CANDLE_COUNT = 200      # How many candles to fetch per timeframe

# --- VWAP Settings ---
VWAP_STD_DEV_1 = 1.0   # First standard deviation band
VWAP_STD_DEV_2 = 2.0   # Second standard deviation band

# --- Profile Settings ---
PROFILE_TICK_SIZE   = 0.01   # Price bucket size for JPY pairs (1 pip)
VALUE_AREA_PERCENT  = 0.70   # 70% of volume = value area

# --- Confluence Detection ---
CONFLUENCE_TOLERANCE_PIPS = 10   # How close levels must be to count as confluence (pips)
MIN_CONFLUENCE_LEVELS     = 2    # Minimum number of levels agreeing for a signal
MIN_RR_RATIO              = 1.5  # Minimum risk:reward to emit a signal

# --- Risk Management ---
ACCOUNT_RISK_PERCENT      = 0.01   # 1% risk per trade (normal conditions)
INTERVENTION_RISK_PERCENT = 0.005  # 0.5% risk during high-volatility / intervention weeks
DRAWDOWN_WARN_PCT         = 0.035  # Warn at 3.5% drawdown from peak
DRAWDOWN_PAUSE_PCT        = 0.045  # Pause signals at 4.5%
DRAWDOWN_CLOSE_PCT        = 0.048  # Close all at 4.8%
PIP_VALUE_USDJPY          = 0.063  # USD pip value per 1000 units (approx at 157)

# --- Kill Zones (EST) ---
KILL_ZONES = [
    {"name": "London Open",   "start": "02:00", "end": "05:00"},
    {"name": "NY Open",       "start": "07:00", "end": "10:00"},
    {"name": "London Close",  "start": "10:00", "end": "12:00"},
]

# --- News Blackout (minutes before/after Tier-1 events) ---
NEWS_BLACKOUT_BEFORE = 30
NEWS_BLACKOUT_AFTER  = 15

# --- Alerts ---
ALERT_EMAIL_ENABLED    = False
ALERT_EMAIL_TO         = "trader@example.com"
ALERT_EMAIL_FROM       = "signals@yourdomain.com"
ALERT_SMTP_HOST        = "smtp.gmail.com"
ALERT_SMTP_PORT        = 587
ALERT_SMTP_USER        = ""
ALERT_SMTP_PASS        = ""

ALERT_WEBHOOK_ENABLED  = False
ALERT_WEBHOOK_URL      = ""   # Discord / Slack / custom webhook

# --- Logging ---
LOG_LEVEL = "INFO"
LOG_FILE  = "g7fx_signals.log"

# --- Signal Score Thresholds ---
SCORE_EMIT_THRESHOLD    = 60   # Minimum score to emit a signal
SCORE_HIGH_CONFIDENCE   = 80   # High confidence threshold
