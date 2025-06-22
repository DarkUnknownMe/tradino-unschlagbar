"""
🔢 TRADINO UNSCHLAGBAR - Math Utilities
Mathematische Utilities für Trading-Berechnungen

Author: AI Trading Systems
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Union
import numpy as np
from utils.logger_pro import setup_logger

logger = setup_logger("MathUtils")


def safe_divide(a: Union[float, Decimal], b: Union[float, Decimal], default: float = 0.0) -> float:
    """Sichere Division mit Fallback"""
    try:
        if b == 0:
            return default
        return float(a) / float(b)
    except (ZeroDivisionError, TypeError):
        return default


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Prozentuale Änderung berechnen"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def round_to_precision(value: Union[float, Decimal], precision: int = 8) -> Decimal:
    """Wert auf Precision runden"""
    decimal_value = Decimal(str(value))
    return decimal_value.quantize(Decimal('0.' + '0' * precision), rounding=ROUND_HALF_UP)


def calculate_position_size(capital: float, risk_percent: float, entry_price: float, stop_loss: float) -> float:
    """Position Size basierend auf Risiko berechnen"""
    if entry_price <= 0 or stop_loss <= 0 or entry_price == stop_loss:
        return 0.0
    
    risk_amount = capital * (risk_percent / 100)
    price_difference = abs(entry_price - stop_loss)
    risk_per_unit = price_difference / entry_price
    
    if risk_per_unit == 0:
        return 0.0
    
    position_size = risk_amount / (entry_price * risk_per_unit)
    return max(0.0, position_size)


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly Criterion für optimale Position Size"""
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    win_loss_ratio = avg_win / avg_loss
    kelly_percent = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    # Kelly limitieren (max 25% für Sicherheit)
    return max(0.0, min(0.25, kelly_percent))


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Sharpe Ratio berechnen"""
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns)


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """Maximum Drawdown berechnen"""
    if not equity_curve:
        return 0.0
    
    peak = equity_curve[0]
    max_dd = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        else:
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
    
    return max_dd * 100  # Als Prozent zurückgeben
