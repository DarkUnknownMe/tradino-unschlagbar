#!/usr/bin/env python3
"""
🛡️ RISK MANAGEMENT DATA MODELS

Datenstrukturen für das Advanced Risk Management System.
Implementiert Position, Signal, Market Conditions und Scenario Models.

Author: TRADINO Development Team
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

class RiskLevel(Enum):
    """🚨 Risk Level Enumeration"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class SignalType(Enum):
    """📊 Trading Signal Types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

class AssetClass(Enum):
    """💼 Asset Classes"""
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCKS = "stocks"
    COMMODITIES = "commodities"

@dataclass
class Position:
    """💰 Trading Position Model"""
    symbol: str
    asset_class: AssetClass
    size: float  # Position size in base currency
    entry_price: float
    current_price: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: float = 1.0
    
    @property
    def unrealized_pnl(self) -> float:
        """📈 Calculate unrealized P&L"""
        return (self.current_price - self.entry_price) * self.size
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """📊 Calculate unrealized P&L percentage"""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price
    
    @property
    def notional_value(self) -> float:
        """💵 Calculate notional value"""
        return abs(self.size * self.current_price * self.leverage)
    
    @property
    def risk_exposure(self) -> float:
        """🎯 Calculate risk exposure"""
        return self.notional_value * self.leverage

@dataclass
class TradingSignal:
    """📡 Trading Signal Model"""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    strength: float    # Signal strength
    timestamp: datetime
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    expected_return: Optional[float] = None
    risk_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """⚖️ Calculate risk-reward ratio"""
        if not all([self.price_target, self.stop_loss, self.expected_return]):
            return None
        
        # Type guard to ensure values are not None
        if self.stop_loss is None or self.price_target is None or self.expected_return is None:
            return None
        
        if self.stop_loss == self.price_target:
            return None
            
        potential_loss = abs(self.price_target - self.stop_loss)
        if potential_loss == 0:
            return None
            
        return abs(self.expected_return) / potential_loss

@dataclass
class MarketConditions:
    """🌍 Market Conditions Model"""
    timestamp: datetime
    volatility: float  # Market volatility (e.g., VIX)
    trend: str         # 'bullish', 'bearish', 'sideways'
    momentum: float    # Market momentum indicator
    liquidity: float   # Market liquidity score
    correlation_regime: str  # 'low', 'medium', 'high'
    risk_appetite: float     # Risk-on/Risk-off sentiment
    market_stress: float     # Market stress indicator
    
    # Economic indicators
    interest_rates: Optional[float] = None
    inflation: Optional[float] = None
    gdp_growth: Optional[float] = None
    
    # Technical indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    bollinger_position: Optional[float] = None
    
    @property
    def market_regime(self) -> str:
        """🏛️ Determine market regime"""
        if self.volatility > 0.3 and self.market_stress > 0.7:
            return "crisis"
        elif self.volatility > 0.2:
            return "volatile"
        elif self.trend == "bullish" and self.risk_appetite > 0.6:
            return "bull_market"
        elif self.trend == "bearish" and self.risk_appetite < 0.4:
            return "bear_market"
        else:
            return "normal"

@dataclass
class Scenario:
    """🎭 Stress Test Scenario Model"""
    name: str
    description: str
    probability: float  # Probability of occurrence
    impact_duration: timedelta
    
    # Price shocks (percentage changes)
    price_shocks: Dict[str, float] = field(default_factory=dict)
    
    # Volatility changes
    volatility_multiplier: float = 1.0
    
    # Correlation changes
    correlation_shift: float = 0.0
    
    # Liquidity impact
    liquidity_reduction: float = 0.0
    
    # Market conditions during scenario
    forced_market_conditions: Optional[MarketConditions] = None
    
    @classmethod
    def create_market_crash(cls) -> 'Scenario':
        """💥 Create market crash scenario"""
        return cls(
            name="Market Crash",
            description="Severe market downturn with high volatility",
            probability=0.05,
            impact_duration=timedelta(days=30),
            price_shocks={
                "BTC": -0.40,  # 40% drop
                "ETH": -0.45,  # 45% drop
                "market": -0.35  # General market drop
            },
            volatility_multiplier=3.0,
            correlation_shift=0.3,  # Increased correlation
            liquidity_reduction=0.5
        )
    
    @classmethod
    def create_flash_crash(cls) -> 'Scenario':
        """⚡ Create flash crash scenario"""
        return cls(
            name="Flash Crash",
            description="Sudden severe price drop with quick recovery",
            probability=0.10,
            impact_duration=timedelta(hours=2),
            price_shocks={
                "BTC": -0.20,
                "ETH": -0.25,
                "market": -0.15
            },
            volatility_multiplier=5.0,
            liquidity_reduction=0.8
        )
    
    @classmethod
    def create_regulatory_shock(cls) -> 'Scenario':
        """⚖️ Create regulatory shock scenario"""
        return cls(
            name="Regulatory Shock",
            description="Adverse regulatory news causing market panic",
            probability=0.15,
            impact_duration=timedelta(days=7),
            price_shocks={
                "BTC": -0.25,
                "ETH": -0.30,
                "DEFI": -0.40
            },
            volatility_multiplier=2.5,
            correlation_shift=0.2
        )

@dataclass 
class RiskMetrics:
    """📊 Comprehensive Risk Metrics"""
    timestamp: datetime
    
    # Value at Risk metrics
    var_1d_95: float         # 1-day VaR at 95% confidence
    var_1d_99: float         # 1-day VaR at 99% confidence
    var_10d_95: float        # 10-day VaR at 95% confidence
    
    # Expected Shortfall (Conditional VaR)
    es_1d_95: float          # 1-day Expected Shortfall at 95%
    es_1d_99: float          # 1-day Expected Shortfall at 99%
    
    # Portfolio metrics
    portfolio_value: float
    total_exposure: float
    leverage_ratio: float
    concentration_risk: float
    
    # Performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    current_drawdown: float
    
    # Risk-adjusted metrics
    risk_adjusted_return: float
    information_ratio: float
    treynor_ratio: float
    
    # Correlation and diversification
    avg_correlation: float
    diversification_ratio: float
    
    # Risk limits and utilization
    risk_limit_utilization: float  # % of risk limits used
    position_limit_utilization: float
    
    # Stress test results
    stress_test_loss: Dict[str, float] = field(default_factory=dict)
    
    @property
    def overall_risk_score(self) -> float:
        """🎯 Calculate overall risk score (0-100)"""
        # Weighted combination of various risk factors
        var_score = min(abs(self.var_1d_99) / self.portfolio_value * 100, 50)
        drawdown_score = min(abs(self.current_drawdown) * 100, 30)
        leverage_score = min(self.leverage_ratio * 10, 20)
        
        return var_score + drawdown_score + leverage_score
    
    @property
    def risk_level(self) -> RiskLevel:
        """🚨 Determine risk level"""
        score = self.overall_risk_score
        
        if score >= 80:
            return RiskLevel.CRITICAL
        elif score >= 60:
            return RiskLevel.HIGH
        elif score >= 40:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

@dataclass
class RiskAlert:
    """🚨 Risk Alert Model"""
    alert_id: str
    timestamp: datetime
    risk_level: RiskLevel
    alert_type: str
    message: str
    affected_positions: List[str]
    recommended_action: str
    threshold_breached: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """📋 Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'risk_level': self.risk_level.value,
            'alert_type': self.alert_type,
            'message': self.message,
            'affected_positions': self.affected_positions,
            'recommended_action': self.recommended_action,
            'threshold_breached': self.threshold_breached
        }

# Risk Configuration
@dataclass
class RiskLimits:
    """⚙️ Risk Limits Configuration"""
    max_portfolio_var: float = 0.05        # Max 5% daily VaR
    max_position_size: float = 0.10        # Max 10% per position
    max_sector_concentration: float = 0.25  # Max 25% per sector
    max_leverage: float = 3.0              # Max 3x leverage
    max_drawdown: float = 0.15             # Max 15% drawdown
    
    # Correlation limits
    max_correlation: float = 0.8           # Max correlation between positions
    
    # Stress test limits
    max_stress_loss: float = 0.20          # Max 20% loss in stress scenarios
    
    # Alert thresholds
    var_alert_threshold: float = 0.8       # Alert at 80% of VaR limit
    drawdown_alert_threshold: float = 0.8   # Alert at 80% of drawdown limit

def create_default_risk_limits() -> RiskLimits:
    """🛡️ Create default risk limits"""
    return RiskLimits()

def validate_position(position: Position, risk_limits: RiskLimits) -> List[str]:
    """✅ Validate position against risk limits"""
    violations = []
    
    if position.leverage > risk_limits.max_leverage:
        violations.append(f"Leverage {position.leverage:.2f}x exceeds limit {risk_limits.max_leverage:.2f}x")
    
    if position.notional_value < 0:
        violations.append("Negative notional value detected")
    
    return violations 