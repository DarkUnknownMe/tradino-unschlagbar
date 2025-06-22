"""
Tradino Unschlagbar - Advanced AI Trading Bot

An intelligent, self-learning trading bot for the Bitget exchange that combines
deep learning, reinforcement learning, and traditional ML techniques with
advanced trading strategies.

Author: AI Trading Bot Developer
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "AI Trading Bot Developer"
__email__ = "developer@tradino.ai"
__license__ = "MIT"

# Core Imports
from .core.trading_engine import TradingEngine
from .core.portfolio_manager import PortfolioManager
from .core.risk_guardian import RiskGuardian

# AI Brain Imports
from .brain.master_ai import MasterAI
from .brain.prediction_engine import PredictionEngine
from .brain.pattern_recognition import PatternRecognition

# Strategy Imports
from .strategies.strategy_selector import StrategySelector

# Connector Imports
from .connectors.bitget_pro import BitgetPro
from .connectors.telegram_commander import TelegramCommander

# Utility Imports
from .utils.config_manager import ConfigManager
from .utils.logger_pro import LoggerPro

__all__ = [
    # Core Classes
    "TradingEngine",
    "PortfolioManager", 
    "RiskGuardian",
    
    # AI Components
    "MasterAI",
    "PredictionEngine",
    "PatternRecognition",
    
    # Strategy Components
    "StrategySelector",
    
    # Connectors
    "BitgetPro",
    "TelegramCommander",
    
    # Utils
    "ConfigManager",
    "LoggerPro",
]

# Package Metadata
PACKAGE_INFO = {
    "name": "tradino-unschlagbar",
    "version": __version__,
    "description": "Advanced AI Trading Bot for Bitget Exchange",
    "author": __author__,
    "license": __license__,
    "supported_exchanges": ["Bitget"],
    "supported_strategies": ["Scalping", "Swing", "Trend Following", "Mean Reversion"],
    "ai_technologies": ["LSTM", "XGBoost", "Random Forest", "Reinforcement Learning"],
} 