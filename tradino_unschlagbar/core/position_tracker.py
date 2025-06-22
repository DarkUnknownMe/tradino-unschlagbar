"""
📊 TRADINO UNSCHLAGBAR - Position Tracker
Real-time Position Tracking und Management

Author: AI Trading Systems
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from models.portfolio_models import Position, PositionStatus, PositionSide
from connectors.bitget_pro import BitgetProConnector
from utils.logger_pro import setup_logger, log_trade
from utils.config_manager import ConfigManager
from utils.helpers import generate_id, safe_float_convert

logger = setup_logger("PositionTracker")


@dataclass
class PositionAlert:
    """Position Alert Model"""
    position_id: str
    alert_type: str  # 'stop_loss', 'take_profit', 'margin_call'
    message: str
    timestamp: datetime
    severity: str  # 'info', 'warning', 'critical'


class PositionTracker:
    """📊 Professional Position Tracking System"""
    
    def __init__(self, config: ConfigManager, bitget_connector: BitgetProConnector):
        self.config = config
        self.bitget = bitget_connector
        
        # Position Tracking
        self.active_positions: Dict[str, Position] = {}
        self.position_history: List[Position] = []
        self.position_callbacks: Dict[str, List[Callable]] = {}
        
        # Alert System
        self.alerts: List[PositionAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        # Risk Monitoring
        self.portfolio_heat = 0.0
        self.total_unrealized_pnl = Decimal('0')
        self.margin_usage = 0.0
        
        # Background Tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def initialize(self) -> bool:
        """🔥 Position Tracker initialisieren"""
        try:
            logger.info("📊 Position Tracker wird initialisiert...")
            
            # Bestehende Positionen laden
            await self._load_existing_positions()
            
            # Background Monitoring starten
            self._running = True
            self._monitoring_task = asyncio.create_task(self._monitor_positions())
            
            logger.success("✅ Position Tracker erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ Position Tracker Initialisierung fehlgeschlagen: {e}")
            return False
    
    async def _load_existing_positions(self):
        """💼 Bestehende Positionen vom Exchange laden"""
        try:
            portfolio = await self.bitget.get_portfolio()
            if portfolio and portfolio.open_positions:
                for position in portfolio.open_positions:
                    self.active_positions[position.id] = position
                    logger.info(f"📊 Position geladen: {position.symbol} {position.side}")
                
                logger.info(f"💼 {len(self.active_positions)} aktive Positionen geladen")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden bestehender Positionen: {e}")
    
    # ==================== POSITION MANAGEMENT ====================
    
    async def add_position(self, position: Position) -> bool:
        """➕ Position hinzufügen"""
        try:
            self.active_positions[position.id] = position
            log_trade(f"➕ Position hinzugefügt: {position.symbol} {position.side} {position.size}")
            
            # Callbacks ausführen
            await self._trigger_callbacks(position.id, 'position_opened', position)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Hinzufügen der Position: {e}")
            return False
    
    async def update_position(self, position_id: str, updated_data: Dict) -> bool:
        """🔄 Position aktualisieren"""
        try:
            if position_id in self.active_positions:
                position = self.active_positions[position_id]
                
                # Position-Daten aktualisieren
                for key, value in updated_data.items():
                    if hasattr(position, key):
                        setattr(position, key, value)
                
                # PnL neu berechnen
                await self._calculate_position_pnl(position)
                
                # Alerts prüfen
                await self._check_position_alerts(position)
                
                # Callbacks ausführen
                await self._trigger_callbacks(position_id, 'position_updated', position)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Aktualisieren der Position: {e}")
            return False
    
    async def close_position(self, position_id: str, exit_price: Decimal) -> bool:
        """❌ Position schließen"""
        try:
            if position_id in self.active_positions:
                position = self.active_positions[position_id]
                
                # Position als geschlossen markieren
                position.status = PositionStatus.CLOSED
                position.closed_at = datetime.utcnow()
                position.current_price = exit_price
                
                # Finales PnL berechnen
                await self._calculate_position_pnl(position)
                
                # Position zu Historie verschieben
                self.position_history.append(position)
                del self.active_positions[position_id]
                
                log_trade(f"❌ Position geschlossen: {position.symbol} PnL: {position.unrealized_pnl}")
                
                # Callbacks ausführen
                await self._trigger_callbacks(position_id, 'position_closed', position)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Schließen der Position: {e}")
            return False
    
    # ==================== POSITION MONITORING ====================
    
    async def _monitor_positions(self):
        """👁️ Kontinuierliche Position-Überwachung"""
        while self._running:
            try:
                # Portfolio-Daten vom Exchange abrufen
                portfolio = await self.bitget.get_portfolio()
                if portfolio:
                    await self._sync_positions_with_exchange(portfolio)
                
                # Risk Metriken aktualisieren
                await self._update_risk_metrics()
                
                # Position Alerts prüfen
                for position in self.active_positions.values():
                    await self._check_position_alerts(position)
                
                # Portfolio Heat überwachen
                await self._monitor_portfolio_heat()
                
                # Kurz warten vor nächster Iteration
                await asyncio.sleep(10)  # 10 Sekunden Intervall
                
            except Exception as e:
                logger.error(f"❌ Fehler beim Position Monitoring: {e}")
                await asyncio.sleep(30)  # Längere Pause bei Fehlern
    
    async def _sync_positions_with_exchange(self, portfolio):
        """🔄 Positionen mit Exchange synchronisieren"""
        try:
            exchange_positions = {pos.symbol: pos for pos in portfolio.open_positions}
            
            # Lokale Positionen mit Exchange-Daten abgleichen
            for position_id, local_position in list(self.active_positions.items()):
                symbol = local_position.symbol
                
                if symbol in exchange_positions:
                    # Position existiert noch - Daten aktualisieren
                    exchange_pos = exchange_positions[symbol]
                    
                    local_position.current_price = exchange_pos.current_price
                    local_position.unrealized_pnl = exchange_pos.unrealized_pnl
                    local_position.unrealized_pnl_percent = exchange_pos.unrealized_pnl_percent
                    local_position.margin_used = exchange_pos.margin_used
                    
                else:
                    # Position wurde extern geschlossen
                    logger.warning(f"⚠️ Position extern geschlossen: {symbol}")
                    await self.close_position(position_id, local_position.current_price or local_position.entry_price)
            
            # Neue Positionen hinzufügen
            for exchange_pos in portfolio.open_positions:
                if not any(pos.symbol == exchange_pos.symbol for pos in self.active_positions.values()):
                    logger.info(f"➕ Neue Position erkannt: {exchange_pos.symbol}")
                    await self.add_position(exchange_pos)
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Position-Synchronisation: {e}")
    
    async def _calculate_position_pnl(self, position: Position):
        """💰 Position PnL berechnen"""
        try:
            if not position.current_price:
                return
            
            entry_price = float(position.entry_price)
            current_price = float(position.current_price)
            size = float(position.size)
            
            if position.side == PositionSide.LONG:
                pnl = (current_price - entry_price) * size
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl = (entry_price - current_price) * size
                pnl_percent = ((entry_price - current_price) / entry_price) * 100
            
            # Leverage berücksichtigen
            pnl *= position.leverage
            pnl_percent *= position.leverage
            
            position.unrealized_pnl = Decimal(str(pnl))
            position.unrealized_pnl_percent = pnl_percent
            
        except Exception as e:
            logger.error(f"❌ Fehler bei PnL-Berechnung: {e}")
    
    async def _check_position_alerts(self, position: Position):
        """🚨 Position Alerts prüfen"""
        try:
            current_price = position.current_price
            if not current_price:
                return
            
            # Stop Loss Alert
            if position.stop_loss:
                if ((position.side == PositionSide.LONG and current_price <= position.stop_loss) or
                    (position.side == PositionSide.SHORT and current_price >= position.stop_loss)):
                    
                    await self._create_alert(
                        position.id, 'stop_loss',
                        f"Stop Loss erreicht: {position.symbol} @ {current_price}",
                        'warning'
                    )
            
            # Take Profit Alert
            if position.take_profit:
                if ((position.side == PositionSide.LONG and current_price >= position.take_profit) or
                    (position.side == PositionSide.SHORT and current_price <= position.take_profit)):
                    
                    await self._create_alert(
                        position.id, 'take_profit',
                        f"Take Profit erreicht: {position.symbol} @ {current_price}",
                        'info'
                    )
            
            # Margin Call Alert
            if position.unrealized_pnl_percent < -50:  # 50% Verlust
                await self._create_alert(
                    position.id, 'margin_call',
                    f"Hoher Verlust: {position.symbol} {position.unrealized_pnl_percent:.1f}%",
                    'critical'
                )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Alert-Prüfung: {e}")
    
    async def _create_alert(self, position_id: str, alert_type: str, message: str, severity: str):
        """🚨 Alert erstellen"""
        alert = PositionAlert(
            position_id=position_id,
            alert_type=alert_type,
            message=message,
            timestamp=datetime.utcnow(),
            severity=severity
        )
        
        self.alerts.append(alert)
        logger.warning(f"🚨 {severity.upper()} Alert: {message}")
        
        # Alert Callbacks ausführen
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"❌ Fehler bei Alert Callback: {e}")
    
    async def _update_risk_metrics(self):
        """📊 Risk Metriken aktualisieren"""
        try:
            if not self.active_positions:
                self.portfolio_heat = 0.0
                self.total_unrealized_pnl = Decimal('0')
                return
            
            # Gesamtes unrealisiertes PnL
            self.total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            
            # Portfolio Heat berechnen (Risiko-Exposure)
            portfolio = await self.bitget.get_portfolio()
            if portfolio:
                total_margin_used = sum(pos.margin_used for pos in self.active_positions.values())
                self.portfolio_heat = float(total_margin_used / portfolio.total_balance) if portfolio.total_balance > 0 else 0
                self.margin_usage = float(portfolio.used_margin / portfolio.total_balance) if portfolio.total_balance > 0 else 0
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Risk Metrics Update: {e}")
    
    async def _monitor_portfolio_heat(self):
        """🔥 Portfolio Heat überwachen"""
        try:
            max_portfolio_heat = self.config.get('trading.portfolio_heat_limit', 0.15)
            
            if self.portfolio_heat > max_portfolio_heat:
                await self._create_alert(
                    'PORTFOLIO', 'portfolio_heat',
                    f"Portfolio Heat zu hoch: {self.portfolio_heat:.1%} > {max_portfolio_heat:.1%}",
                    'critical'
                )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Portfolio Heat Monitoring: {e}")
    
    # ==================== CALLBACK SYSTEM ====================
    
    def add_callback(self, position_id: str, event: str, callback: Callable):
        """Callback für Position-Events registrieren"""
        if position_id not in self.position_callbacks:
            self.position_callbacks[position_id] = []
        self.position_callbacks[position_id].append((event, callback))
    
    def add_alert_callback(self, callback: Callable):
        """Alert Callback registrieren"""
        self.alert_callbacks.append(callback)
    
    async def _trigger_callbacks(self, position_id: str, event: str, position: Position):
        """Callbacks für Position-Event ausführen"""
        try:
            callbacks = self.position_callbacks.get(position_id, [])
            for event_type, callback in callbacks:
                if event_type == event:
                    await callback(position)
        except Exception as e:
            logger.error(f"❌ Fehler bei Callback-Ausführung: {e}")
    
    # ==================== PUBLIC METHODS ====================
    
    def get_active_positions(self) -> List[Position]:
        """📊 Aktive Positionen abrufen"""
        return list(self.active_positions.values())
    
    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """🔍 Position nach Symbol suchen"""
        for position in self.active_positions.values():
            if position.symbol == symbol:
                return position
        return None
    
    def get_portfolio_summary(self) -> Dict:
        """💼 Portfolio-Zusammenfassung"""
        return {
            'active_positions': len(self.active_positions),
            'total_unrealized_pnl': float(self.total_unrealized_pnl),
            'portfolio_heat': self.portfolio_heat,
            'margin_usage': self.margin_usage,
            'recent_alerts': len([alert for alert in self.alerts 
                                if (datetime.utcnow() - alert.timestamp).seconds < 3600])
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[PositionAlert]:
        """🚨 Aktuelle Alerts abrufen"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp > cutoff]
    
    async def shutdown(self):
        """🛑 Position Tracker herunterfahren"""
        try:
            self._running = False
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("✅ Position Tracker heruntergefahren")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren des Position Trackers: {e}")
