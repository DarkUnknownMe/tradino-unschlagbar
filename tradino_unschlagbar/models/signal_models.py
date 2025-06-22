"""
🎯 TRADINO UNSCHLAGBAR - Signal Models
AI Trading Signal Models

Author: AI Trading Systems
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator


class SignalType(str, Enum):
    """Signal Typen"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class SignalStrength(str, Enum):
    """Signal Stärke"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class AISignal(BaseModel):
    """AI-generiertes Trading Signal"""
    id: str = Field(..., description="Signal ID")
    symbol: str = Field(..., description="Trading Pair")
    signal_type: SignalType = Field(..., description="Signal Type")
    strength: SignalStrength = Field(..., description="Signal Stärke")
    confidence: float = Field(..., ge=0, le=1, description="AI Confidence")
    predicted_price: Decimal = Field(..., gt=0, description="Vorhergesagter Preis")
    current_price: Decimal = Field(..., gt=0, description="Aktueller Preis")
    entry_price: Optional[Decimal] = Field(None, description="Empfohlener Einstieg")
    stop_loss: Optional[Decimal] = Field(None, description="Stop Loss")
    take_profit: Optional[Decimal] = Field(None, description="Take Profit")
    risk_reward_ratio: Optional[float] = Field(None, description="Risk/Reward Ratio")
    timeframe: str = Field(..., description="Zeitrahmen")
    strategy_source: str = Field(..., description="Quell-Strategie")
    model_version: str = Field(..., description="AI Model Version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    expiry: Optional[datetime] = Field(None, description="Signal Ablauf")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SignalAnalysis(BaseModel):
    """Signal Analyse Ergebnis"""
    signal_id: str = Field(..., description="Signal ID")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Technische Analyse
    technical_score: float = Field(..., ge=0, le=1, description="Technical Score")
    trend_alignment: float = Field(..., ge=-1, le=1, description="Trend Alignment")
    momentum_score: float = Field(..., ge=0, le=1, description="Momentum Score")
    volume_confirmation: float = Field(..., ge=0, le=1, description="Volume Bestätigung")
    
    # Pattern Recognition
    patterns_detected: List[str] = Field(default_factory=list)
    pattern_strength: float = Field(default=0.0, ge=0, le=1)
    
    # Market Context
    market_regime: str = Field(..., description="Markt-Regime")
    volatility_level: float = Field(..., ge=0, le=1, description="Volatilität Level")
    support_resistance: Dict[str, List[Decimal]] = Field(default_factory=dict)
    
    # Final Assessment
    overall_score: float = Field(..., ge=0, le=1, description="Gesamt-Score")
    recommendation: str = Field(..., description="Empfehlung")
    risk_assessment: str = Field(..., description="Risiko-Bewertung")


class SignalPerformance(BaseModel):
    """Signal Performance Tracking"""
    signal_id: str = Field(..., description="Signal ID")
    executed: bool = Field(default=False, description="Ausgeführt")
    execution_price: Optional[Decimal] = Field(None, description="Ausführungspreis")
    execution_time: Optional[datetime] = Field(None, description="Ausführungszeit")
    
    # Outcome Tracking
    outcome: Optional[str] = Field(None, description="Ergebnis (win/loss)")
    pnl: Optional[Decimal] = Field(None, description="P&L")
    pnl_percent: Optional[float] = Field(None, description="P&L %")
    duration_minutes: Optional[int] = Field(None, description="Dauer in Minuten")
    
    # Performance Metriken
    accuracy_score: Optional[float] = Field(None, ge=0, le=1, description="Genauigkeits-Score")
    timing_score: Optional[float] = Field(None, ge=0, le=1, description="Timing-Score")
    
    # Learning Data
    market_conditions: Dict[str, Any] = Field(default_factory=dict)
    feedback_data: Dict[str, Any] = Field(default_factory=dict)
