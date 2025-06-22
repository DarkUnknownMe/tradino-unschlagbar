"""
📝 TRADINO UNSCHLAGBAR - Professional Logger
Advanced Logging mit Loguru für Production-Ready Systems

Author: AI Trading Systems
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


class TradinoLogger:
    """🔥 Professional Logger für TRADINO UNSCHLAGBAR"""
    
    def __init__(self):
        self.setup_logger()
    
    def setup_logger(self):
        """Logger Setup mit Production-Grade Konfiguration"""
        # Standard Logger entfernen
        logger.remove()
        
        # Console Logger (Colored)
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # File Logger (Detailed)
        log_path = Path("data/logs")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path / "tradino_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="daily",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        # Error Logger (Separate)
        logger.add(
            log_path / "tradino_errors_{time:YYYY-MM-DD}.log",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message} | {extra}",
            rotation="daily",
            retention="90 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        # Trading Logger (Trades only)
        logger.add(
            log_path / "tradino_trades_{time:YYYY-MM-DD}.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | TRADE | {message}",
            filter=lambda record: "TRADE" in record.get("extra", {}),
            rotation="daily",
            retention="365 days",
            compression="zip"
        )


def setup_logger(name: Optional[str] = None) -> object:
    """
    Setup Professional Logger
    
    Args:
        name: Logger Name (optional)
    
    Returns:
        Logger Instance
    """
    # Logger Setup nur einmal
    if not hasattr(setup_logger, '_initialized'):
        TradinoLogger()
        setup_logger._initialized = True
    
    return logger.bind(name=name) if name else logger


# Logger Extensions
def log_trade(message: str, **kwargs):
    """Log Trading Activity"""
    logger.bind(TRADE=True).info(message, **kwargs)


def log_performance(metric: str, value: float, **kwargs):
    """Log Performance Metrics"""
    logger.bind(PERFORMANCE=True).info(f"📊 {metric}: {value}", **kwargs)


def log_ai_decision(model: str, decision: str, confidence: float, **kwargs):
    """Log AI Model Decisions"""
    logger.bind(AI=True).info(f"🧠 {model}: {decision} (Confidence: {confidence:.2%})", **kwargs)


def log_risk_event(event: str, severity: str, **kwargs):
    """Log Risk Management Events"""
    logger.bind(RISK=True).warning(f"🛡️ {severity.upper()}: {event}", **kwargs)


# Custom Log Levels (nur einmal hinzufügen)
try:
    logger.level("SUCCESS", no=25, color="<green>")
    logger.level("TRADE", no=25, color="<blue>")
    logger.level("AI", no=25, color="<magenta>")
    logger.level("RISK", no=35, color="<yellow>")
except ValueError:
    # Levels bereits vorhanden
    pass

# Export Logger
__all__ = ['setup_logger', 'log_trade', 'log_performance', 'log_ai_decision', 'log_risk_event']
