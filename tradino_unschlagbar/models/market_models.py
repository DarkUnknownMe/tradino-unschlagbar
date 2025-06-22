"""
📊 TRADINO UNSCHLAGBAR - Market Models
Marktdaten und Kerzendaten Models

Author: AI Trading Systems
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class MarketRegime(str, Enum):
    """Markt-Regime Klassifikation"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


class Candle(BaseModel):
    """Kerzendaten Model"""
    symbol: str = Field(..., description="Trading Pair")
    timeframe: str = Field(..., description="Zeitrahmen")
    timestamp: datetime = Field(..., description="Zeitstempel")
    open: Decimal = Field(..., gt=0, description="Eröffnungspreis")
    high: Decimal = Field(..., gt=0, description="Höchstpreis")
    low: Decimal = Field(..., gt=0, description="Tiefstpreis")
    close: Decimal = Field(..., gt=0, description="Schlusspreis")
    volume: Decimal = Field(..., ge=0, description="Handelsvolumen")
    
    @validator('high')
    def validate_high(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('High muss >= Low sein')
        return v
    
    @validator('close')
    def validate_close(cls, v, values):
        if 'high' in values and 'low' in values:
            if not (values['low'] <= v <= values['high']):
                raise ValueError('Close muss zwischen Low und High liegen')
        return v


class MarketData(BaseModel):
    """Marktdaten Model"""
    symbol: str = Field(..., description="Trading Pair")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    price: Decimal = Field(..., gt=0, description="Aktueller Preis")
    bid: Decimal = Field(..., gt=0, description="Bid Preis")
    ask: Decimal = Field(..., gt=0, description="Ask Preis")
    volume_24h: Decimal = Field(..., ge=0, description="24h Volumen")
    change_24h: Decimal = Field(..., description="24h Änderung")
    change_24h_percent: float = Field(..., description="24h Änderung %")
    high_24h: Decimal = Field(..., gt=0, description="24h Hoch")
    low_24h: Decimal = Field(..., gt=0, description="24h Tief")


class TechnicalIndicators(BaseModel):
    """Technical Analysis Indikatoren"""
    symbol: str = Field(..., description="Trading Pair")
    timeframe: str = Field(..., description="Zeitrahmen")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Trend Indikatoren
    sma_20: Optional[Decimal] = Field(None, description="SMA 20")
    sma_50: Optional[Decimal] = Field(None, description="SMA 50")
    sma_200: Optional[Decimal] = Field(None, description="SMA 200")
    ema_12: Optional[Decimal] = Field(None, description="EMA 12")
    ema_26: Optional[Decimal] = Field(None, description="EMA 26")
    
    # Momentum Indikatoren
    rsi: Optional[float] = Field(None, ge=0, le=100, description="RSI")
    macd: Optional[Decimal] = Field(None, description="MACD")
    macd_signal: Optional[Decimal] = Field(None, description="MACD Signal")
    macd_histogram: Optional[Decimal] = Field(None, description="MACD Histogram")
    
    # Volatilität Indikatoren
    bb_upper: Optional[Decimal] = Field(None, description="Bollinger Upper")
    bb_middle: Optional[Decimal] = Field(None, description="Bollinger Middle")
    bb_lower: Optional[Decimal] = Field(None, description="Bollinger Lower")
    atr: Optional[Decimal] = Field(None, description="ATR")
    
    # Volume Indikatoren
    obv: Optional[Decimal] = Field(None, description="On Balance Volume")
    volume_sma: Optional[Decimal] = Field(None, description="Volume SMA")


class MarketAnalysis(BaseModel):
    """Marktanalyse Results"""
    symbol: str = Field(..., description="Trading Pair")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    regime: MarketRegime = Field(..., description="Markt-Regime")
    trend_strength: float = Field(..., ge=0, le=1, description="Trend Stärke")
    volatility_score: float = Field(..., ge=0, le=1, description="Volatilität Score")
    volume_score: float = Field(..., ge=0, le=1, description="Volumen Score")
    support_levels: List[Decimal] = Field(default_factory=list)
    resistance_levels: List[Decimal] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1, description="Analyse Confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict)
