# ============================================================
# G7FX Signal App — Alert Dispatcher
# ============================================================
# Sends signal alerts via:
#   - Console (always)
#   - Email (SMTP)
#   - Webhook (Discord / Slack / custom)
# ============================================================

import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from urllib import request as urlrequest

from config.settings import (
    ALERT_EMAIL_ENABLED, ALERT_EMAIL_TO, ALERT_EMAIL_FROM,
    ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_SMTP_USER, ALERT_SMTP_PASS,
    ALERT_WEBHOOK_ENABLED, ALERT_WEBHOOK_URL
)
from core.risk_engine import FinalSignal, format_signal_alert

logger = logging.getLogger(__name__)


def dispatch_alert(signal: FinalSignal):
    """Send signal through all enabled alert channels."""
    alert_text = format_signal_alert(signal)

    # Always print to console
    print("\n" + alert_text + "\n")

    if ALERT_EMAIL_ENABLED:
        _send_email(signal, alert_text)

    if ALERT_WEBHOOK_ENABLED and ALERT_WEBHOOK_URL:
        _send_webhook(signal, alert_text)


def _send_email(signal: FinalSignal, body: str):
    try:
        msg = MIMEMultipart()
        msg["From"]    = ALERT_EMAIL_FROM
        msg["To"]      = ALERT_EMAIL_TO
        msg["Subject"] = (
            f"G7FX Signal | {signal.pair} {signal.direction} | "
            f"{signal.confidence_label} ({signal.confidence_score}/100)"
        )
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(ALERT_SMTP_HOST, ALERT_SMTP_PORT) as server:
            server.starttls()
            server.login(ALERT_SMTP_USER, ALERT_SMTP_PASS)
            server.sendmail(ALERT_EMAIL_FROM, ALERT_EMAIL_TO, msg.as_string())

        logger.info(f"Email alert sent to {ALERT_EMAIL_TO}")
    except Exception as e:
        logger.error(f"Email alert failed: {e}")


def _send_webhook(signal: FinalSignal, body: str):
    """Send to Discord / Slack / custom webhook."""
    try:
        direction_emoji = "🔴" if signal.direction == "SELL" else "🟢"
        conf_emoji = "🔥" if signal.confidence_label == "HIGH" else "⚡"

        # Discord-compatible payload
        payload = {
            "content": None,
            "embeds": [{
                "title": f"{direction_emoji} {signal.pair.replace('_','/')} {signal.direction}",
                "color": 0xE24B4A if signal.direction == "SELL" else 0x1D9E75,
                "fields": [
                    {"name": "Entry Zone",   "value": f"`{signal.entry_zone[0]:.3f} – {signal.entry_zone[1]:.3f}`", "inline": True},
                    {"name": "Stop Loss",    "value": f"`{signal.stop_loss:.3f}`",  "inline": True},
                    {"name": "TP1 (VWAP)",  "value": f"`{signal.tp1:.3f}` ({signal.rr_tp1}:1)", "inline": True},
                    {"name": "TP2",         "value": f"`{signal.tp2:.3f}` ({signal.rr_tp2}:1)", "inline": True},
                    {"name": "Lot Size",    "value": f"`{signal.lot_size}`", "inline": True},
                    {"name": "Risk",        "value": f"`${signal.risk_amount_usd:.2f}`", "inline": True},
                    {"name": f"{conf_emoji} Confidence", "value": f"`{signal.confidence_score}/100 ({signal.confidence_label})`", "inline": False},
                    {"name": "Confluence",  "value": ", ".join(signal.confluence_levels), "inline": False},
                    {"name": "Kill Zone",   "value": signal.kill_zone, "inline": True},
                    {"name": "Status",      "value": signal.status, "inline": True},
                ],
                "footer": {"text": f"G7FX Signal Engine • {signal.timestamp[:19]} UTC"}
            }]
        }

        data = json.dumps(payload).encode("utf-8")
        req  = urlrequest.Request(
            ALERT_WEBHOOK_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        urlrequest.urlopen(req, timeout=10)
        logger.info("Webhook alert sent")
    except Exception as e:
        logger.error(f"Webhook alert failed: {e}")
