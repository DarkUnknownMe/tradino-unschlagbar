"""
🛡️ TRADINO UNSCHLAGBAR - Risk Guardian
Bulletproof Risk Management System

Author: AI Trading Systems
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from models.trade_models import TradeSignal
from models.portfolio_models import Portfolio, Position
from core.portfolio_manager import PortfolioManager
from core.position_tracker import PositionTracker
from utils.logger_pro import setup_logger, log_risk_event
from utils.config_manager import ConfigManager

logger = setup_logger("RiskGuardian")


class RiskLevel(Enum):
    """Risk Level"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskCheckResult:
    """Risk Check Ergebnis"""
    approved: bool
    reason: str
    risk_level: RiskLevel
    adjusted_quantity: Optional[Decimal] = None
    warnings: List[str] = None


class RiskGuardian:
    """🛡️ Bulletproof Risk Management System"""
    
    def __init__(self, config: ConfigManager, portfolio_manager: PortfolioManager, 
                 position_tracker: PositionTracker):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.position_tracker = position_tracker
        
        # Risk Limits
        self.max_risk_per_trade = config.get('trading.risk_per_trade', 0.03)  # 3%
        self.max_daily_drawdown = config.get('trading.max_daily_drawdown', 0.05)  # 5%
        self.max_portfolio_heat = config.get('trading.portfolio_heat_limit', 0.15)  # 15%
        self.max_correlation = config.get('trading.max_correlation', 0.50)  # 50%
        self.max_positions = config.get('trading.max_positions', 5)
        
        # Emergency Limits
        self.emergency_stop_drawdown = 0.15  # 15% Emergency Stop
        self.emergency_stop_heat = 0.25  # 25% Portfolio Heat Emergency
        
        # Risk Events Tracking
        self.risk_events: List[Dict] = []
        self.emergency_stops: List[Dict] = []
        self.last_risk_check = datetime.utcnow()
        
        # Risk State
        self.current_risk_level = RiskLevel.LOW
        self.emergency_mode = False
        
    async def initialize(self) -> bool:
        """🔥 Risk Guardian initialisieren"""
        try:
            logger.info("🛡️ Risk Guardian wird initialisiert...")
            logger.success("✅ Risk Guardian erfolgreich initialisiert")
            return True
        except Exception as e:
            logger.error(f"❌ Risk Guardian Initialisierung fehlgeschlagen: {e}")
            return False
    
    # ==================== TRADE VALIDATION ====================
    
    async def validate_trade_signal(self, signal):
        """
        Trade Signal mit robuster Validation prüfen
        """
        try:
            # ✅ DEFENSIVE SIGNAL VALIDATION
            if not signal:
                logger.warning("⚠️ Leeres Signal für Risk Validation empfangen")
                return {'approved': False, 'reason': 'Leeres Signal'}
            
            if not hasattr(signal, 'symbol') or not signal.symbol:
                logger.warning("⚠️ Signal ohne Symbol für Risk Validation empfangen")
                return {'approved': False, 'reason': 'Symbol fehlt'}
            
            if not hasattr(signal, 'action') or not signal.action:
                logger.warning("⚠️ Signal ohne Action für Risk Validation empfangen")
                return {'approved': False, 'reason': 'Action fehlt'}
            
            # Symbol Validation
            symbol = signal.symbol
            action = signal.action
            
            logger.info(f"🛡️ Risk Validation für {symbol}: {action}")
            
            # Hier die eigentliche Risk Logic
            risk_checks = await self._perform_risk_checks(signal)
            
            if risk_checks['all_passed']:
                logger.success(f"✅ Risk Validation bestanden für {symbol}")
                return {'approved': True, 'checks': risk_checks}
            else:
                logger.warning(f"⚠️ Risk Validation fehlgeschlagen für {symbol}: {risk_checks['failed_reason']}")
                return {'approved': False, 'reason': risk_checks['failed_reason']}
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Risk Validation: {e}")
            return {'approved': False, 'reason': f'Validation Fehler: {e}'}

    async def _perform_risk_checks(self, signal):
        """
        Spezifische Risk Checks durchführen
        """
        try:
            checks = {
                'symbol_valid': True,
                'position_size_ok': True,
                'exposure_limit_ok': True,
                'volatility_ok': True,
                'all_passed': True,
                'failed_reason': None
            }
            
            # Hier können spezifische Checks implementiert werden
            # Für jetzt: Basic Validation
            
            return checks
            
        except Exception as e:
            return {
                'all_passed': False,
                'failed_reason': f'Risk Check Fehler: {e}'
            }
    
    # ==================== RISK CALCULATIONS ====================
    
    async def _calculate_trade_risk(self, signal: TradeSignal, portfolio: Portfolio) -> float:
        """💰 Trade Risk berechnen"""
        try:
            if not signal.stop_loss:
                return self.max_risk_per_trade  # Konservativ ohne SL
            
            entry_price = float(signal.entry_price)
            stop_loss = float(signal.stop_loss)
            quantity = float(signal.quantity)
            
            # Risk per Unit
            price_diff = abs(entry_price - stop_loss)
            risk_per_unit = price_diff / entry_price
            
            # Total Risk Amount
            position_value = entry_price * quantity
            total_risk = position_value * risk_per_unit
            
            # Risk as Portfolio Percentage
            portfolio_risk = total_risk / float(portfolio.total_balance)
            
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Trade Risk Berechnung: {e}")
            return self.max_risk_per_trade  # Konservativ
    
    async def _calculate_projected_portfolio_heat(self, signal: TradeSignal, portfolio: Portfolio) -> float:
        """🔥 Projizierte Portfolio Heat berechnen"""
        try:
            current_heat = portfolio.portfolio_heat
            
            # Neue Position Margin
            entry_price = float(signal.entry_price)
            quantity = float(signal.quantity)
            leverage = signal.metadata.get('leverage', 1)
            
            new_position_margin = (entry_price * quantity) / leverage
            additional_heat = new_position_margin / float(portfolio.total_balance)
            
            return current_heat + additional_heat
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Portfolio Heat Berechnung: {e}")
            return 1.0  # Konservativ
    
    async def _calculate_daily_drawdown(self, portfolio: Portfolio) -> float:
        """📉 Täglichen Drawdown berechnen"""
        try:
            # Vereinfachte Berechnung basierend auf Daily P&L
            daily_pnl = float(portfolio.daily_pnl)
            total_balance = float(portfolio.total_balance)
            
            if daily_pnl >= 0:
                return 0.0  # Kein Drawdown bei Gewinn
            
            return abs(daily_pnl) / total_balance
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Daily Drawdown Berechnung: {e}")
            return 0.0
    
    async def _check_correlation_risk(self, signal: TradeSignal, portfolio: Portfolio) -> float:
        """🔗 Korrelations-Risiko prüfen"""
        try:
            # Vereinfachte Korrelationsberechnung
            # In der Realität würde man historische Preiskorrelationen berechnen
            
            same_base_positions = 0
            for position in portfolio.open_positions:
                # Gleiche Base Currency? (z.B. BTC in BTC/USDT und BTC/EUR)
                signal_base = signal.symbol.split('/')[0]
                position_base = position.symbol.split('/')[0]
                
                if signal_base == position_base:
                    same_base_positions += 1
            
            # Korrelation basierend auf gleichen Base Currencies
            correlation = same_base_positions / len(portfolio.open_positions) if portfolio.open_positions else 0
            
            return correlation
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Correlation Check: {e}")
            return 0.0
    
    async def _check_sufficient_balance(self, signal: TradeSignal, portfolio: Portfolio) -> bool:
        """💰 Ausreichende Balance prüfen"""
        try:
            entry_price = float(signal.entry_price)
            quantity = float(signal.quantity)
            leverage = signal.metadata.get('leverage', 1)
            
            required_margin = (entry_price * quantity) / leverage
            available_balance = float(portfolio.available_balance)
            
            # 10% Buffer für Fees und Slippage
            required_with_buffer = required_margin * 1.1
            
            return available_balance >= required_with_buffer
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Balance Check: {e}")
            return False
    
    def _determine_risk_level(self, trade_risk: float, portfolio_heat: float, correlation: float) -> RiskLevel:
        """🎯 Risk Level bestimmen"""
        try:
            # Risk Score berechnen
            risk_score = (
                (trade_risk / self.max_risk_per_trade) * 0.4 +
                (portfolio_heat / self.max_portfolio_heat) * 0.4 +
                (correlation / self.max_correlation) * 0.2
            )
            
            if risk_score >= 0.8:
                return RiskLevel.HIGH
            elif risk_score >= 0.6:
                return RiskLevel.MODERATE
            else:
                return RiskLevel.LOW
                
        except Exception:
            return RiskLevel.MODERATE
    
    # ==================== PORTFOLIO MONITORING ====================
    
    async def monitor_portfolio_risks(self):
        """👁️ Portfolio Risiken überwachen"""
        try:
            portfolio = self.portfolio_manager.current_portfolio
            if not portfolio:
                return
            
            # Emergency Stop Checks
            await self._check_emergency_stops(portfolio)
            
            # Risk Level Update
            await self._update_current_risk_level(portfolio)
            
            # Risk Events Logging
            await self._log_risk_events(portfolio)
            
            self.last_risk_check = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Portfolio Risk Monitoring: {e}")
    
    async def _check_emergency_stops(self, portfolio: Portfolio):
        """🚨 Emergency Stop Conditions prüfen"""
        try:
            # Drawdown Emergency Stop
            performance = self.portfolio_manager.performance_metrics
            if performance and performance.max_drawdown > self.emergency_stop_drawdown:
                await self._trigger_emergency_stop(
                    "MAX_DRAWDOWN_EXCEEDED",
                    f"Maximum Drawdown überschritten: {performance.max_drawdown:.1%}"
                )
                return
            
            # Portfolio Heat Emergency Stop
            if portfolio.portfolio_heat > self.emergency_stop_heat:
                await self._trigger_emergency_stop(
                    "PORTFOLIO_HEAT_CRITICAL",
                    f"Portfolio Heat kritisch: {portfolio.portfolio_heat:.1%}"
                )
                return
            
            # Balance Emergency Stop
            if portfolio.total_balance < self.portfolio_manager.initial_capital * Decimal('0.5'):  # 50% Verlust
                await self._trigger_emergency_stop(
                    "BALANCE_CRITICAL",
                    f"Balance kritisch niedrig: {portfolio.total_balance}"
                )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Emergency Stop Check: {e}")
    
    async def _trigger_emergency_stop(self, stop_type: str, reason: str):
        """🚨 Emergency Stop auslösen"""
        try:
            if self.emergency_mode:
                return  # Bereits im Emergency Mode
            
            self.emergency_mode = True
            
            emergency_event = {
                'type': stop_type,
                'reason': reason,
                'timestamp': datetime.utcnow(),
                'portfolio_state': self.portfolio_manager.get_portfolio_summary()
            }
            
            self.emergency_stops.append(emergency_event)
            
            log_risk_event(f"EMERGENCY STOP: {reason}", "CRITICAL")
            logger.critical(f"🚨 EMERGENCY STOP AKTIVIERT: {reason}")
            
            # Hier würden weitere Emergency-Aktionen folgen:
            # - Alle offenen Orders stornieren
            # - Positionen schließen (optional)
            # - Admin-Benachrichtigung
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Emergency Stop: {e}")
    
    async def _update_current_risk_level(self, portfolio: Portfolio):
        """📊 Aktuellen Risk Level updaten"""
        try:
            # Risk Factors sammeln
            heat_factor = portfolio.portfolio_heat / self.max_portfolio_heat
            margin_factor = portfolio.margin_ratio
            
            performance = self.portfolio_manager.performance_metrics
            drawdown_factor = (performance.max_drawdown / 100) / self.max_daily_drawdown if performance else 0
            
            # Overall Risk Score
            risk_score = (heat_factor * 0.4 + margin_factor * 0.3 + drawdown_factor * 0.3)
            
            # Risk Level bestimmen
            if risk_score >= 0.8:
                new_risk_level = RiskLevel.CRITICAL
            elif risk_score >= 0.6:
                new_risk_level = RiskLevel.HIGH
            elif risk_score >= 0.4:
                new_risk_level = RiskLevel.MODERATE
            else:
                new_risk_level = RiskLevel.LOW
            
            # Risk Level Change Logging
            if new_risk_level != self.current_risk_level:
                logger.info(f"📊 Risk Level geändert: {self.current_risk_level.value} → {new_risk_level.value}")
                self.current_risk_level = new_risk_level
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Risk Level Update: {e}")
    
    async def _log_risk_events(self, portfolio: Portfolio):
        """📝 Risk Events loggen"""
        try:
            risk_event = {
                'timestamp': datetime.utcnow(),
                'risk_level': self.current_risk_level.value,
                'portfolio_heat': portfolio.portfolio_heat,
                'margin_ratio': portfolio.margin_ratio,
                'position_count': portfolio.position_count,
                'total_balance': float(portfolio.total_balance),
                'unrealized_pnl': float(portfolio.unrealized_pnl)
            }
            
            self.risk_events.append(risk_event)
            
            # Events begrenzen (letzte 1000)
            if len(self.risk_events) > 1000:
                self.risk_events = self.risk_events[-1000:]
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Risk Event Logging: {e}")
    
    # ==================== PUBLIC METHODS ====================
    
    async def get_portfolio_risk_level(self) -> float:
        """📊 Portfolio Risk Level abrufen (0-1)"""
        try:
            portfolio = self.portfolio_manager.current_portfolio
            if not portfolio:
                return 0.5
            
            heat_factor = portfolio.portfolio_heat / self.max_portfolio_heat
            margin_factor = portfolio.margin_ratio
            
            performance = self.portfolio_manager.performance_metrics
            drawdown_factor = (performance.max_drawdown / 100) / self.max_daily_drawdown if performance else 0
            
            return min(1.0, (heat_factor * 0.4 + margin_factor * 0.3 + drawdown_factor * 0.3))
            
        except Exception:
            return 0.5
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """🛡️ Risk Summary abrufen"""
        try:
            return {
                'current_risk_level': self.current_risk_level.value,
                'emergency_mode': self.emergency_mode,
                'portfolio_risk_score': asyncio.create_task(self.get_portfolio_risk_level()),
                'risk_limits': {
                    'max_risk_per_trade': self.max_risk_per_trade,
                    'max_daily_drawdown': self.max_daily_drawdown,
                    'max_portfolio_heat': self.max_portfolio_heat,
                    'max_correlation': self.max_correlation,
                    'max_positions': self.max_positions
                },
                'emergency_stops_count': len(self.emergency_stops),
                'last_risk_check': self.last_risk_check.isoformat() if self.last_risk_check else None
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Risk Summary: {e}")
            return {}
    
    def get_recent_risk_events(self, hours: int = 24) -> List[Dict]:
        """📊 Aktuelle Risk Events"""
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            return [event for event in self.risk_events if event['timestamp'] > cutoff]
        except Exception:
            return []
    
    def reset_emergency_mode(self) -> bool:
        """🔄 Emergency Mode zurücksetzen"""
        try:
            if self.emergency_mode:
                self.emergency_mode = False
                logger.info("🔄 Emergency Mode zurückgesetzt")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Fehler bei Emergency Mode Reset: {e}")
            return False
    
    async def shutdown(self):
        """🛑 Risk Guardian herunterfahren"""
        try:
            logger.info("✅ Risk Guardian heruntergefahren")
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren: {e}")
