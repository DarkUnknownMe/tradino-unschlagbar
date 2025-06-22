"""
💼 TRADINO UNSCHLAGBAR - Portfolio Models
Portfolio und Position Management Models

Author: AI Trading Systems
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, validator


class PositionSide(str, Enum):
    """Position Seiten"""
    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    """Position Status"""
    OPEN = "open"
    CLOSED = "closed"
    CLOSING = "closing"


class Position(BaseModel):
    """Trading Position Model"""
    id: str = Field(..., description="Position ID")
    symbol: str = Field(..., description="Trading Pair")
    side: PositionSide = Field(..., description="Position Side")
    size: Decimal = Field(..., description="Position Größe")
    entry_price: Decimal = Field(..., gt=0, description="Einstiegspreis")
    current_price: Optional[Decimal] = Field(None, description="Aktueller Preis")
    unrealized_pnl: Decimal = Field(default=Decimal('0'), description="Unrealisierter P&L")
    unrealized_pnl_percent: float = Field(default=0.0, description="Unrealisierter P&L %")
    leverage: int = Field(default=1, ge=1, le=20, description="Leverage")
    margin_used: Decimal = Field(..., gt=0, description="Verwendete Margin")
    stop_loss: Optional[Decimal] = Field(None, description="Stop Loss")
    take_profit: Optional[Decimal] = Field(None, description="Take Profit")
    status: PositionStatus = Field(default=PositionStatus.OPEN)
    strategy: str = Field(..., description="Verwendete Strategie")
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = Field(None)
    metadata: Dict[str, str] = Field(default_factory=dict)


class Portfolio(BaseModel):
    """Portfolio Model"""
    account_id: str = Field(..., description="Account ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Kapital Information  
    total_balance: Decimal = Field(..., description="Gesamtbalance")
    available_balance: Decimal = Field(..., description="Verfügbare Balance")
    used_margin: Decimal = Field(default=Decimal('0'), description="Verwendete Margin")
    free_margin: Decimal = Field(..., description="Freie Margin")
    
    # P&L Information
    total_pnl: Decimal = Field(default=Decimal('0'), description="Gesamt P&L")
    daily_pnl: Decimal = Field(default=Decimal('0'), description="Tages P&L")
    unrealized_pnl: Decimal = Field(default=Decimal('0'), description="Unrealisierter P&L")
    realized_pnl: Decimal = Field(default=Decimal('0'), description="Realisierter P&L")
    
    # Risiko Metriken
    portfolio_heat: float = Field(default=0.0, ge=0, le=1, description="Portfolio Heat")
    max_drawdown: float = Field(default=0.0, description="Max Drawdown")
    margin_ratio: float = Field(default=0.0, ge=0, le=1, description="Margin Ratio")
    
    # Positionen
    open_positions: List[Position] = Field(default_factory=list)
    position_count: int = Field(default=0, description="Anzahl Positionen")
    
    @validator('free_margin')
    def calculate_free_margin(cls, v, values):
        if 'total_balance' in values and 'used_margin' in values:
            return values['total_balance'] - values['used_margin']
        return v


class PerformanceMetrics(BaseModel):
    """Performance Metriken Model"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    period: str = Field(..., description="Zeitraum (daily/weekly/monthly)")
    
    # Return Metriken
    total_return: float = Field(..., description="Gesamtrendite %")
    annualized_return: float = Field(..., description="Annualisierte Rendite %")
    sharpe_ratio: float = Field(..., description="Sharpe Ratio")
    sortino_ratio: float = Field(..., description="Sortino Ratio")
    
    # Risk Metriken
    max_drawdown: float = Field(..., description="Maximaler Drawdown %")
    volatility: float = Field(..., description="Volatilität %")
    var_95: float = Field(..., description="Value at Risk 95%")
    
    # Trading Metriken
    total_trades: int = Field(..., ge=0, description="Anzahl Trades")
    winning_trades: int = Field(..., ge=0, description="Gewinn Trades")
    losing_trades: int = Field(..., ge=0, description="Verlust Trades")
    win_rate: float = Field(..., ge=0, le=1, description="Gewinnrate")
    profit_factor: float = Field(..., description="Profit Faktor")
    average_win: float = Field(..., description="Durchschnittlicher Gewinn")
    average_loss: float = Field(..., description="Durchschnittlicher Verlust")
    
    @validator('win_rate')
    def calculate_win_rate(cls, v, values):
        if 'total_trades' in values and values['total_trades'] > 0:
            winning = values.get('winning_trades', 0)
            return winning / values['total_trades']
        return v
