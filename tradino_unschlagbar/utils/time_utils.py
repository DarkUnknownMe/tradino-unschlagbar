"""
⏰ TRADINO UNSCHLAGBAR - Time Utilities
Zeit-Utilities für Trading und Marktdaten

Author: AI Trading Systems
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Union
import time

from utils.logger_pro import setup_logger

logger = setup_logger("TimeUtils")


def utc_now() -> datetime:
    """Aktuelle UTC Zeit"""
    return datetime.now(timezone.utc)


def timestamp_ms() -> int:
    """Aktueller Timestamp in Millisekunden"""
    return int(time.time() * 1000)


def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
    """Timestamp zu DateTime konvertieren"""
    if timestamp > 1e10:  # Millisekunden
        timestamp = timestamp / 1000
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> int:
    """DateTime zu Timestamp konvertieren"""
    return int(dt.timestamp())


def format_duration(seconds: int) -> str:
    """Sekunden zu lesbarer Dauer formatieren"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def market_hours_check() -> bool:
    """Prüfen ob Markt geöffnet (Crypto: 24/7)"""
    return True  # Crypto markets are always open


def get_timeframe_seconds(timeframe: str) -> int:
    """Timeframe zu Sekunden konvertieren"""
    timeframe_map = {
        '1m': 60,
        '3m': 180,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '2h': 7200,
        '4h': 14400,
        '6h': 21600,
        '12h': 43200,
        '1d': 86400,
        '1w': 604800
    }
    return timeframe_map.get(timeframe, 60)
