"""
🔥 TRADINO UNSCHLAGBAR - Trade Models
Pydantic Models für Trading-Daten mit Validierung

Author: AI Trading Systems
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator


class OrderType(str, Enum):
    """Order Typen"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(str, Enum):
    """Order Seiten"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order Status"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TradeSignal(BaseModel):
    """Trading Signal Model"""
    id: str = Field(..., description="Eindeutige Signal ID")
    symbol: str = Field(..., description="Trading Pair")
    side: OrderSide = Field(..., description="Buy/Sell")
    entry_price: Decimal = Field(..., gt=0, description="Einstiegspreis")
    stop_loss: Optional[Decimal] = Field(None, gt=0, description="Stop Loss")
    take_profit: Optional[Decimal] = Field(None, gt=0, description="Take Profit")
    quantity: Decimal = Field(..., gt=0, description="Handelsvolumen")
    confidence: float = Field(..., ge=0, le=1, description="Signal Confidence")
    strategy: str = Field(..., description="Verwendete Strategie")
    timeframe: str = Field(..., description="Zeitrahmen")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence muss zwischen 0 und 1 liegen')
        return v


class Order(BaseModel):
    """Order Model"""
    id: str = Field(..., description="Order ID")
    exchange_id: Optional[str] = Field(None, description="Exchange Order ID")
    symbol: str = Field(..., description="Trading Pair")
    type: OrderType = Field(..., description="Order Type")
    side: OrderSide = Field(..., description="Order Side")
    amount: Decimal = Field(..., gt=0, description="Order Menge")
    price: Optional[Decimal] = Field(None, description="Order Preis")
    status: OrderStatus = Field(default=OrderStatus.PENDING)
    filled: Decimal = Field(default=Decimal('0'), description="Gefüllte Menge")
    remaining: Decimal = Field(default=Decimal('0'), description="Verbleibende Menge")
    cost: Decimal = Field(default=Decimal('0'), description="Kosten")
    fee: Decimal = Field(default=Decimal('0'), description="Gebühr")
    leverage: int = Field(default=1, ge=1, le=20, description="Leverage")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Trade(BaseModel):
    """Ausgeführter Trade Model"""
    id: str = Field(..., description="Trade ID")
    signal_id: str = Field(..., description="Zugehörige Signal ID")
    symbol: str = Field(..., description="Trading Pair")
    side: OrderSide = Field(..., description="Trade Side")
    entry_price: Decimal = Field(..., gt=0, description="Einstiegspreis")
    exit_price: Optional[Decimal] = Field(None, description="Ausstiegspreis")
    quantity: Decimal = Field(..., gt=0, description="Handelsvolumen")
    pnl: Optional[Decimal] = Field(None, description="Profit/Loss")
    pnl_percentage: Optional[float] = Field(None, description="P&L in %")
    commission: Decimal = Field(default=Decimal('0'), description="Kommission")
    strategy: str = Field(..., description="Verwendete Strategie")
    leverage: int = Field(default=1, ge=1, le=20)
    entry_time: datetime = Field(default_factory=datetime.utcnow)
    exit_time: Optional[datetime] = Field(None)
    duration_minutes: Optional[int] = Field(None, description="Trade Dauer")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('pnl_percentage')
    def calculate_pnl_percentage(cls, v, values):
        if 'entry_price' in values and 'exit_price' in values and values['exit_price']:
            entry = float(values['entry_price'])
            exit_price = float(values['exit_price'])
            side = values.get('side')
            
            if side == OrderSide.BUY:
                return ((exit_price - entry) / entry) * 100
            else:
                return ((entry - exit_price) / entry) * 100
        return v
