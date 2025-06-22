"""
💼 TRADINO UNSCHLAGBAR - Portfolio Manager
Intelligentes Portfolio Management mit Real-time Tracking

Author: AI Trading Systems
"""

import asyncio
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from models.portfolio_models import Portfolio, Position, PerformanceMetrics
from models.trade_models import Trade
from connectors.bitget_pro import BitgetProConnector
from core.position_tracker import PositionTracker
from utils.logger_pro import setup_logger
from utils.config_manager import ConfigManager
from utils.math_utils import calculate_sharpe_ratio, calculate_max_drawdown

logger = setup_logger("PortfolioManager")


@dataclass
class PortfolioAlert:
    """Portfolio Alert Model"""
    alert_type: str
    message: str
    severity: str  # 'info', 'warning', 'critical'
    timestamp: datetime
    data: Dict[str, Any]


class PortfolioManager:
    """💼 Professional Portfolio Management System"""
    
    def __init__(self, config: ConfigManager, bitget_connector: BitgetProConnector, 
                 position_tracker: PositionTracker):
        self.config = config
        self.bitget = bitget_connector
        self.position_tracker = position_tracker
        
        # Portfolio State
        self.current_portfolio: Optional[Portfolio] = None
        self.portfolio_history: List[Portfolio] = []
        self.performance_metrics: Optional[PerformanceMetrics] = None
        
        # Portfolio Configuration
        self.initial_capital = Decimal(str(config.get('trading.initial_capital', 1000)))
        self.max_drawdown_limit = config.get('trading.max_drawdown_limit', 0.10)
        self.target_return = config.get('trading.target_return', 0.20)  # 20% annual
        
        # Risk Management
        self.portfolio_heat_limit = config.get('trading.portfolio_heat_limit', 0.15)
        self.max_correlation = config.get('trading.max_correlation', 0.50)
        self.max_position_size = config.get('trading.max_position_size', 0.10)  # 10% per position
        
        # Tracking
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []
        self.portfolio_alerts: List[PortfolioAlert] = []
        
        # Update Frequency
        self.last_update: Optional[datetime] = None
        self.update_interval = 30  # 30 Sekunden
        
        # Background Tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def initialize(self) -> bool:
        """🔥 Portfolio Manager initialisieren"""
        try:
            logger.info("💼 Portfolio Manager wird initialisiert...")
            
            # Initial Portfolio laden
            await self.update_portfolio()
            
            if not self.current_portfolio:
                logger.error("❌ Keine Portfolio-Daten verfügbar")
                return False
            
            # Background Monitoring starten
            self._running = True
            self._monitoring_task = asyncio.create_task(self._portfolio_monitoring_loop())
            
            logger.success("✅ Portfolio Manager erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ Portfolio Manager Initialisierung fehlgeschlagen: {e}")
            return False
    
    # ==================== PORTFOLIO MANAGEMENT ====================
    
    async def update_portfolio(self) -> bool:
        """💼 Portfolio aktualisieren"""
        try:
            # Portfolio-Daten von Exchange abrufen
            portfolio = await self.bitget.get_portfolio()
            if not portfolio:
                logger.warning("⚠️ Keine Portfolio-Daten von Exchange")
                return False
            
            # Performance berechnen
            await self._calculate_performance_metrics(portfolio)
            
            # Risk Metrics updaten
            await self._update_risk_metrics(portfolio)
            
            # Portfolio Alerts prüfen
            await self._check_portfolio_alerts(portfolio)
            
            # Portfolio Historie updaten
            if self.current_portfolio:
                self.portfolio_history.append(self.current_portfolio)
                # Historie begrenzen (letzte 1000 Einträge)
                if len(self.portfolio_history) > 1000:
                    self.portfolio_history = self.portfolio_history[-1000:]
            
            self.current_portfolio = portfolio
            self.last_update = datetime.utcnow()
            
            # Equity Curve updaten
            self.equity_curve.append(float(portfolio.total_balance))
            if len(self.equity_curve) > 1000:
                self.equity_curve = self.equity_curve[-1000:]
            
            # Daily Returns berechnen
            await self._calculate_daily_returns()
            
            logger.debug(f"💼 Portfolio aktualisiert - Balance: {portfolio.total_balance}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Portfolio Update: {e}")
            return False
    
    async def _calculate_performance_metrics(self, portfolio: Portfolio):
        """📊 Performance Metriken berechnen"""
        try:
            if len(self.equity_curve) < 2:
                return
            
            # Returns berechnen
            returns = []
            for i in range(1, len(self.equity_curve)):
                ret = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
                returns.append(ret)
            
            if not returns:
                return
            
            # Performance Metriken
            total_return = (float(portfolio.total_balance) - float(self.initial_capital)) / float(self.initial_capital)
            
            # Annualisierte Rendite (vereinfacht)
            periods_per_year = 365 * 24 * 2  # 30-Sekunden-Updates
            periods = len(returns)
            if periods > 0:
                annualized_return = ((1 + total_return) ** (periods_per_year / periods)) - 1
            else:
                annualized_return = 0
            
            # Sharpe Ratio
            sharpe_ratio = calculate_sharpe_ratio(returns) if len(returns) > 1 else 0
            
            # Max Drawdown
            max_drawdown = calculate_max_drawdown(self.equity_curve) / 100  # Convert to decimal
            
            # Win Rate (aus Position Tracker)
            positions_summary = self.position_tracker.get_portfolio_summary()
            
            # Volatilität
            volatility = float(np.std(returns)) * (periods_per_year ** 0.5) if returns else 0
            
            self.performance_metrics = PerformanceMetrics(
                timestamp=datetime.utcnow(),
                period="real_time",
                total_return=total_return * 100,  # Prozent
                annualized_return=annualized_return * 100,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sharpe_ratio * 1.1,  # Vereinfacht
                max_drawdown=max_drawdown,
                volatility=volatility * 100,
                var_95=float(np.percentile(returns, 5)) * 100 if returns else 0,
                total_trades=0,  # Wird von Performance Tracker gefüllt
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                profit_factor=1.0,
                average_win=0,
                average_loss=0
            )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Performance Berechnung: {e}")
    
    async def _update_risk_metrics(self, portfolio: Portfolio):
        """🛡️ Risk Metrics updaten"""
        try:
            # Portfolio Heat berechnen
            total_margin_used = sum(pos.margin_used for pos in portfolio.open_positions)
            portfolio_heat = float(total_margin_used / portfolio.total_balance) if portfolio.total_balance > 0 else 0
            portfolio.portfolio_heat = portfolio_heat
            
            # Margin Ratio
            margin_ratio = float(portfolio.used_margin / portfolio.total_balance) if portfolio.total_balance > 0 else 0
            portfolio.margin_ratio = margin_ratio
            
            # Position Concentration
            largest_position = max([float(pos.margin_used / portfolio.total_balance) for pos in portfolio.open_positions]) if portfolio.open_positions else 0
            
            # Risk Score (0-1, wobei 1 = höchstes Risiko)
            risk_score = (
                portfolio_heat * 0.4 +
                margin_ratio * 0.3 +
                largest_position * 0.3
            )
            
            # Risk Alerts
            if portfolio_heat > self.portfolio_heat_limit:
                await self._create_portfolio_alert(
                    'portfolio_heat_high',
                    f"Portfolio Heat zu hoch: {portfolio_heat:.1%}",
                    'warning',
                    {'portfolio_heat': portfolio_heat, 'limit': self.portfolio_heat_limit}
                )
            
            if largest_position > self.max_position_size:
                await self._create_portfolio_alert(
                    'position_concentration_high',
                    f"Position zu groß: {largest_position:.1%}",
                    'warning',
                    {'position_size': largest_position, 'limit': self.max_position_size}
                )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Risk Metrics Update: {e}")
    
    async def _check_portfolio_alerts(self, portfolio: Portfolio):
        """🚨 Portfolio Alerts prüfen"""
        try:
            # Drawdown Alert
            if self.performance_metrics and self.performance_metrics.max_drawdown > self.max_drawdown_limit:
                await self._create_portfolio_alert(
                    'max_drawdown_exceeded',
                    f"Max Drawdown überschritten: {self.performance_metrics.max_drawdown:.1%}",
                    'critical',
                    {'current_drawdown': self.performance_metrics.max_drawdown, 'limit': self.max_drawdown_limit}
                )
            
            # Balance Alert
            balance_change = 0
            if len(self.equity_curve) >= 2:
                balance_change = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            
            if balance_change < -0.05:  # 5% Verlust
                await self._create_portfolio_alert(
                    'significant_loss',
                    f"Signifikanter Verlust: {balance_change:.1%}",
                    'warning',
                    {'balance_change': balance_change}
                )
            
            # Margin Alert
            if portfolio.margin_ratio > 0.8:  # 80% Margin verwendet
                await self._create_portfolio_alert(
                    'high_margin_usage',
                    f"Hohe Margin-Nutzung: {portfolio.margin_ratio:.1%}",
                    'warning',
                    {'margin_ratio': portfolio.margin_ratio}
                )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Portfolio Alert Check: {e}")
    
    async def _create_portfolio_alert(self, alert_type: str, message: str, severity: str, data: Dict[str, Any]):
        """🚨 Portfolio Alert erstellen"""
        try:
            alert = PortfolioAlert(
                alert_type=alert_type,
                message=message,
                severity=severity,
                timestamp=datetime.utcnow(),
                data=data
            )
            
            self.portfolio_alerts.append(alert)
            
            # Alerts begrenzen
            if len(self.portfolio_alerts) > 100:
                self.portfolio_alerts = self.portfolio_alerts[-100:]
            
            logger.warning(f"🚨 Portfolio Alert: {message}")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Alert-Erstellung: {e}")
    
    async def _calculate_daily_returns(self):
        """📊 Tägliche Returns berechnen"""
        try:
            if len(self.equity_curve) < 2:
                return
            
            # Letzter Return
            latest_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            
            # Vereinfachte tägliche Returns (alle 30 Sekunden ein Update)
            # Für echte tägliche Returns müsste man Zeitstempel berücksichtigen
            self.daily_returns.append(latest_return)
            
            if len(self.daily_returns) > 365:  # Letztes Jahr
                self.daily_returns = self.daily_returns[-365:]
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Daily Returns Berechnung: {e}")
    
    # ==================== MONITORING LOOP ====================
    
    async def _portfolio_monitoring_loop(self):
        """👁️ Portfolio Monitoring Loop"""
        while self._running:
            try:
                # Portfolio Update
                await self.update_portfolio()
                
                # Performance Check
                await self._performance_health_check()
                
                # Rebalancing Check
                await self._check_rebalancing_needs()
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"❌ Fehler im Portfolio Monitoring: {e}")
                await asyncio.sleep(60)  # Längere Pause bei Fehlern
    
    async def _performance_health_check(self):
        """🏥 Performance Health Check"""
        try:
            if not self.performance_metrics:
                return
            
            # Sharpe Ratio Check
            if self.performance_metrics.sharpe_ratio < 0.5:
                logger.warning(f"⚠️ Niedrige Sharpe Ratio: {self.performance_metrics.sharpe_ratio:.2f}")
            
            # Win Rate Check
            if self.performance_metrics.win_rate < 0.4:
                logger.warning(f"⚠️ Niedrige Win Rate: {self.performance_metrics.win_rate:.1%}")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Performance Health Check: {e}")
    
    async def _check_rebalancing_needs(self):
        """⚖️ Rebalancing-Bedarf prüfen"""
        try:
            if not self.current_portfolio or not self.current_portfolio.open_positions:
                return
            
            # Position Size Check
            total_exposure = sum(pos.margin_used for pos in self.current_portfolio.open_positions)
            
            for position in self.current_portfolio.open_positions:
                position_weight = float(position.margin_used / total_exposure) if total_exposure > 0 else 0
                
                if position_weight > self.max_position_size:
                    logger.warning(f"⚠️ Position {position.symbol} zu groß: {position_weight:.1%}")
                    # Hier könnte automatisches Rebalancing implementiert werden
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Rebalancing Check: {e}")
    
    # ==================== PUBLIC METHODS ====================
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """💼 Portfolio-Zusammenfassung abrufen"""
        try:
            if not self.current_portfolio:
                return {}
            
            return {
                'total_balance': float(self.current_portfolio.total_balance),
                'available_balance': float(self.current_portfolio.available_balance),
                'used_margin': float(self.current_portfolio.used_margin),
                'unrealized_pnl': float(self.current_portfolio.unrealized_pnl),
                'daily_pnl': float(self.current_portfolio.daily_pnl),
                'position_count': self.current_portfolio.position_count,
                'portfolio_heat': self.current_portfolio.portfolio_heat,
                'margin_ratio': self.current_portfolio.margin_ratio,
                'last_update': self.last_update.isoformat() if self.last_update else None
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Portfolio Summary: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """📊 Performance-Zusammenfassung abrufen"""
        try:
            if not self.performance_metrics:
                return {}
            
            return {
                'total_return': self.performance_metrics.total_return,
                'annualized_return': self.performance_metrics.annualized_return,
                'sharpe_ratio': self.performance_metrics.sharpe_ratio,
                'max_drawdown': self.performance_metrics.max_drawdown,
                'volatility': self.performance_metrics.volatility,
                'win_rate': self.performance_metrics.win_rate,
                'profit_factor': self.performance_metrics.profit_factor,
                'total_trades': self.performance_metrics.total_trades
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Performance Summary: {e}")
            return {}
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """🛡️ Risk Metrics abrufen"""
        try:
            if not self.current_portfolio:
                return {}
            
            return {
                'portfolio_heat': self.current_portfolio.portfolio_heat,
                'margin_ratio': self.current_portfolio.margin_ratio,
                'max_drawdown': self.performance_metrics.max_drawdown if self.performance_metrics else 0,
                'var_95': self.performance_metrics.var_95 if self.performance_metrics else 0,
                'position_count': self.current_portfolio.position_count,
                'largest_position_weight': self._get_largest_position_weight(),
                'risk_score': self._calculate_overall_risk_score()
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Risk Metrics: {e}")
            return {}
    
    def _get_largest_position_weight(self) -> float:
        """📊 Größte Position-Gewichtung"""
        try:
            if not self.current_portfolio or not self.current_portfolio.open_positions:
                return 0
            
            largest_margin = max(pos.margin_used for pos in self.current_portfolio.open_positions)
            return float(largest_margin / self.current_portfolio.total_balance) if self.current_portfolio.total_balance > 0 else 0
            
        except Exception:
            return 0
    
    def _calculate_overall_risk_score(self) -> float:
        """🎯 Gesamt-Risk-Score berechnen"""
        try:
            if not self.current_portfolio:
                return 0
            
            # Verschiedene Risk-Faktoren
            portfolio_heat = self.current_portfolio.portfolio_heat
            margin_ratio = self.current_portfolio.margin_ratio
            position_concentration = self._get_largest_position_weight()
            drawdown_risk = self.performance_metrics.max_drawdown / 100 if self.performance_metrics else 0
            
            # Gewichteter Risk Score
            risk_score = (
                portfolio_heat * 0.3 +
                margin_ratio * 0.25 +
                position_concentration * 0.25 +
                drawdown_risk * 0.2
            )
            
            return min(1.0, risk_score)
            
        except Exception:
            return 0
    
    def get_recent_alerts(self, hours: int = 24) -> List[PortfolioAlert]:
        """🚨 Aktuelle Alerts abrufen"""
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            return [alert for alert in self.portfolio_alerts if alert.timestamp > cutoff]
        except Exception:
            return []
    
    def get_equity_curve(self, limit: int = 100) -> List[float]:
        """📈 Equity Curve abrufen"""
        try:
            return self.equity_curve[-limit:] if self.equity_curve else []
        except Exception:
            return []
    
    async def shutdown(self):
        """🛑 Portfolio Manager herunterfahren"""
        try:
            self._running = False
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("✅ Portfolio Manager heruntergefahren")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren: {e}")
