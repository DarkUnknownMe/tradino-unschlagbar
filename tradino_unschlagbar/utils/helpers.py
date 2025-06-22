"""
🛠️ TRADINO UNSCHLAGBAR - Helper Functions
Allgemeine Helper-Funktionen für das Trading System

Author: AI Trading Systems
"""

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal

from utils.logger_pro import setup_logger

logger = setup_logger("Helpers")


def generate_id(prefix: str = "") -> str:
    """Eindeutige ID generieren"""
    unique_id = str(uuid.uuid4())[:8]
    timestamp = int(datetime.utcnow().timestamp())
    return f"{prefix}{timestamp}_{unique_id}" if prefix else f"{timestamp}_{unique_id}"


def generate_signal_id(symbol: str, strategy: str) -> str:
    """Signal ID generieren"""
    timestamp = int(datetime.utcnow().timestamp())
    return f"SIG_{symbol}_{strategy}_{timestamp}"


def generate_trade_id(symbol: str, side: str) -> str:
    """Trade ID generieren"""
    timestamp = int(datetime.utcnow().timestamp())
    return f"TRD_{symbol}_{side}_{timestamp}"


def hash_data(data: Union[str, Dict, List]) -> str:
    """Daten hashen für Vergleiche"""
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True, default=str)
    else:
        data_str = str(data)
    
    return hashlib.sha256(data_str.encode()).hexdigest()[:12]


def format_currency(amount: Union[float, Decimal], currency: str = "USDT", decimals: int = 2) -> str:
    """Währung formatieren"""
    if isinstance(amount, Decimal):
        amount = float(amount)
    
    return f"{amount:,.{decimals}f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Prozent formatieren"""
    return f"{value:.{decimals}f}%"


def truncate_float(value: float, decimals: int = 8) -> float:
    """Float auf Dezimalstellen kürzen"""
    multiplier = 10 ** decimals
    return int(value * multiplier) / multiplier


def is_valid_symbol(symbol: str) -> bool:
    """Symbol-Format validieren"""
    if not symbol or '/' not in symbol:
        return False
    
    parts = symbol.split('/')
    return len(parts) == 2 and all(part.isalpha() for part in parts)


def normalize_symbol(symbol: str) -> str:
    """Symbol normalisieren"""
    if not symbol:
        return ""
    
    return symbol.upper().replace('-', '/').replace('_', '/')


def extract_base_quote(symbol: str) -> tuple:
    """Base und Quote aus Symbol extrahieren"""
    if '/' in symbol:
        base, quote = symbol.split('/')
        return base.upper(), quote.upper()
    return "", ""


def safe_float_convert(value: Any, default: float = 0.0) -> float:
    """Sichere Float-Konvertierung"""
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int_convert(value: Any, default: int = 0) -> int:
    """Sichere Int-Konvertierung"""
    try:
        if value is None:
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


def merge_dicts(*dicts: Dict) -> Dict[str, Any]:
    """Dictionaries zusammenführen"""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


def filter_dict(data: Dict[str, Any], keys_to_keep: List[str]) -> Dict[str, Any]:
    """Dictionary auf bestimmte Keys filtern"""
    return {k: v for k, v in data.items() if k in keys_to_keep}


def deep_get(data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Tief verschachtelte Dictionary-Werte abrufen"""
    keys = key_path.split('.')
    value = data
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
