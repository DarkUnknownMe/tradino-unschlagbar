"""
🛡️ TRADINO RISK MANAGEMENT MODULE

Advanced institutional-grade risk management system for quantitative trading.
Implements VaR, CVaR, stress testing, and real-time risk monitoring.

Author: TRADINO Development Team
"""

try:
    from .advanced_risk_manager import AdvancedRiskManager, VaRCalculator, CorrelationMonitor
    from .risk_models import (
        RiskMetrics, Position, TradingSignal, MarketConditions, Scenario,
        RiskAlert, RiskLimits, RiskLevel, SignalType, AssetClass,
        create_default_risk_limits, validate_position
    )
    
    __all__ = [
        'AdvancedRiskManager',
        'VaRCalculator', 
        'CorrelationMonitor',
        'RiskMetrics',
        'Position',
        'TradingSignal',
        'MarketConditions',
        'Scenario',
        'RiskAlert',
        'RiskLimits', 
        'RiskLevel',
        'SignalType',
        'AssetClass',
        'create_default_risk_limits',
        'validate_position'
    ]
except ImportError as e:
    print(f"⚠️ Warning: Could not import risk management modules: {e}")
    __all__ = [] 