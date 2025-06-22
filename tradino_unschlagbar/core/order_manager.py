"""
📋 TRADINO UNSCHLAGBAR - Order Manager
Intelligente Order-Verwaltung mit Auto-Execution

Author: AI Trading Systems
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum

from models.trade_models import Order, OrderType, OrderSide, OrderStatus, TradeSignal
from connectors.bitget_pro import BitgetProConnector
from utils.logger_pro import setup_logger, log_trade
from utils.config_manager import ConfigManager
from utils.helpers import generate_id

logger = setup_logger("OrderManager")


class OrderPriority(Enum):
    """Order Priorität"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class OrderManager:
    """📋 Professional Order Management System"""
    
    def __init__(self, config: ConfigManager, bitget_connector: BitgetProConnector):
        self.config = config
        self.bitget = bitget_connector
        
        # Order Tracking
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.order_callbacks: Dict[str, List[Callable]] = {}
        
        # Auto-Execution Settings
        self.auto_execution = True
        self.max_concurrent_orders = config.get('trading.max_positions', 5)
        self.order_timeout = 300  # 5 Minuten
        
        # Performance Tracking
        self.execution_times: List[float] = []
        self.order_success_rate = 0.0
        
        # Background Tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def initialize(self) -> bool:
        """🔥 Order Manager initialisieren"""
        try:
            logger.info("📋 Order Manager wird initialisiert...")
            
            # Background Monitoring starten
            self._running = True
            self._monitoring_task = asyncio.create_task(self._monitor_orders())
            
            logger.success("✅ Order Manager erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ Order Manager Initialisierung fehlgeschlagen: {e}")
            return False
    
    # ==================== ORDER EXECUTION ====================
    
    async def execute_signal(self, signal: TradeSignal, 
                           priority: OrderPriority = OrderPriority.NORMAL) -> Optional[Order]:
        """🎯 Trading Signal ausführen"""
        try:
            start_time = datetime.utcnow()
            
            logger.info(f"🎯 Signal wird ausgeführt: {signal.symbol} {signal.side.value}")
            
            # Pre-Execution Checks
            if not await self._pre_execution_checks(signal):
                return None
            
            # Leverage automatisch bestimmen
            leverage = await self._calculate_optimal_leverage(signal)
            
            # Order erstellen
            order = await self.bitget.place_order(
                symbol=signal.symbol,
                order_type=OrderType.MARKET,  # Market Orders für schnelle Ausführung
                side=signal.side,
                amount=signal.quantity,
                price=signal.entry_price if signal.side == OrderSide.BUY else None,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                leverage=leverage
            )
            
            if order:
                # Order registrieren
                order.metadata.update({
                    'signal_id': signal.id,
                    'strategy': signal.strategy,
                    'confidence': signal.confidence,
                    'priority': priority.value
                })
                
                self.pending_orders[order.id] = order
                
                # Execution Time tracken
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.execution_times.append(execution_time)
                
                log_trade(f"⚡ Order ausgeführt in {execution_time:.1f}ms: {order.id}")
                
                # Callbacks ausführen
                await self._trigger_callbacks(order.id, 'order_created', order)
                
                return order
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Signal-Ausführung: {e}")
            return None
    
    async def _pre_execution_checks(self, signal: TradeSignal) -> bool:
        """🔍 Pre-Execution Validierung"""
        try:
            # Max Concurrent Orders Check
            if len(self.pending_orders) >= self.max_concurrent_orders:
                logger.warning(f"⚠️ Maximale Anzahl gleichzeitiger Orders erreicht: {len(self.pending_orders)}")
                return False
            
            # Signal Confidence Check
            min_confidence = self.config.get('trading.min_signal_confidence', 0.6)
            if signal.confidence < min_confidence:
                logger.warning(f"⚠️ Signal Confidence zu niedrig: {signal.confidence:.2%}")
                return False
            
            # Portfolio Balance Check
            portfolio = await self.bitget.get_portfolio()
            if not portfolio or portfolio.available_balance < signal.quantity * signal.entry_price:
                logger.warning("⚠️ Unzureichende Balance für Trade")
                return False
            
            # Risk Management Check
            risk_per_trade = self.config.get('trading.risk_per_trade', 0.03)
            max_risk_amount = portfolio.total_balance * Decimal(str(risk_per_trade))
            
            if signal.stop_loss:
                potential_loss = abs(signal.entry_price - signal.stop_loss) * signal.quantity
                if potential_loss > max_risk_amount:
                    logger.warning(f"⚠️ Trade-Risiko zu hoch: {potential_loss} > {max_risk_amount}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Pre-Execution Check: {e}")
            return False
    
    async def _calculate_optimal_leverage(self, signal: TradeSignal) -> int:
        """⚖️ Optimalen Leverage berechnen"""
        try:
            # Basis Leverage aus Konfiguration
            min_leverage = self.config.get('trading.min_leverage', 1)
            max_leverage = self.config.get('trading.max_leverage', 10)
            
            # Volatilität-basierte Anpassung
            # TODO: Implement volatility calculation
            volatility_factor = 1.0  # Placeholder
            
            # Confidence-basierte Anpassung
            confidence_multiplier = 1 + (signal.confidence - 0.5)  # 0.5 - 1.5
            
            # Risk-basierte Anpassung
            if signal.stop_loss:
                risk_distance = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
                risk_multiplier = max(0.5, 1 - risk_distance * 2)  # Weniger Leverage bei größerem Risk
            else:
                risk_multiplier = 0.5  # Konservativ ohne Stop Loss
            
            # Finaler Leverage
            calculated_leverage = int(min_leverage * confidence_multiplier * risk_multiplier)
            optimal_leverage = max(min_leverage, min(max_leverage, calculated_leverage))
            
            logger.info(f"⚖️ Optimaler Leverage für {signal.symbol}: {optimal_leverage}x")
            return optimal_leverage
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Leverage-Berechnung: {e}")
            return 1  # Fallback zu 1x
    
    # ==================== ORDER MONITORING ====================
    
    async def _monitor_orders(self):
        """👁️ Kontinuierliche Order-Überwachung"""
        while self._running:
            try:
                # Pending Orders überprüfen
                orders_to_remove = []
                
                for order_id, order in self.pending_orders.items():
                    # Order Status updaten
                    updated_order = await self.bitget.get_order_status(
                        order.exchange_id, order.symbol
                    )
                    
                    if updated_order:
                        # Status-Änderung verarbeiten
                        if updated_order.status != order.status:
                            await self._handle_status_change(order, updated_order.status)
                        
                        # Order updaten
                        self.pending_orders[order_id] = updated_order
                        
                        # Abgeschlossene Orders entfernen
                        if updated_order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, 
                                                   OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                            orders_to_remove.append(order_id)
                            self.order_history.append(updated_order)
                    
                    # Timeout Check
                    elif (datetime.utcnow() - order.created_at).seconds > self.order_timeout:
                        logger.warning(f"⏰ Order Timeout: {order_id}")
                        await self._handle_order_timeout(order)
                        orders_to_remove.append(order_id)
                
                # Abgeschlossene Orders entfernen
                for order_id in orders_to_remove:
                    self.pending_orders.pop(order_id, None)
                
                # Performance Metriken updaten
                await self._update_performance_metrics()
                
                # Kurz warten vor nächster Iteration
                await asyncio.sleep(5)  # 5 Sekunden Intervall
                
            except Exception as e:
                logger.error(f"❌ Fehler beim Order Monitoring: {e}")
                await asyncio.sleep(10)  # Längere Pause bei Fehlern
    
    async def _handle_status_change(self, order: Order, new_status: OrderStatus):
        """📊 Order Status-Änderung verarbeiten"""
        try:
            old_status = order.status
            order.status = new_status
            order.updated_at = datetime.utcnow()
            
            log_trade(f"📊 Order Status geändert: {order.id} {old_status.value} → {new_status.value}")
            
            # Status-spezifische Aktionen
            if new_status == OrderStatus.FILLED:
                log_trade(f"✅ Order vollständig ausgeführt: {order.id}")
                await self._trigger_callbacks(order.id, 'order_filled', order)
                
            elif new_status == OrderStatus.CANCELED:
                logger.warning(f"🚫 Order storniert: {order.id}")
                await self._trigger_callbacks(order.id, 'order_canceled', order)
                
            elif new_status == OrderStatus.REJECTED:
                logger.error(f"❌ Order abgelehnt: {order.id}")
                await self._trigger_callbacks(order.id, 'order_rejected', order)
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Status-Änderung: {e}")
    
    async def _handle_order_timeout(self, order: Order):
        """⏰ Order Timeout behandeln"""
        try:
            logger.warning(f"⏰ Order {order.id} Timeout - Stornierung versucht")
            
            success = await self.bitget.cancel_order(order.exchange_id, order.symbol)
            if success:
                order.status = OrderStatus.CANCELED
                await self._trigger_callbacks(order.id, 'order_timeout', order)
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Order Timeout: {e}")
    
    async def _update_performance_metrics(self):
        """📈 Performance Metriken aktualisieren"""
        try:
            if self.order_history:
                successful_orders = sum(1 for order in self.order_history 
                                      if order.status == OrderStatus.FILLED)
                self.order_success_rate = successful_orders / len(self.order_history)
            
            # Durchschnittliche Execution Time
            if self.execution_times:
                avg_execution_time = sum(self.execution_times) / len(self.execution_times)
                logger.info(f"📊 Durchschnittliche Execution Time: {avg_execution_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Performance Update: {e}")
    
    # ==================== CALLBACK SYSTEM ====================
    
    def add_callback(self, order_id: str, event: str, callback: Callable):
        """Callback für Order-Events registrieren"""
        if order_id not in self.order_callbacks:
            self.order_callbacks[order_id] = []
        self.order_callbacks[order_id].append((event, callback))
    
    async def _trigger_callbacks(self, order_id: str, event: str, order: Order):
        """Callbacks für Order-Event ausführen"""
        try:
            callbacks = self.order_callbacks.get(order_id, [])
            for event_type, callback in callbacks:
                if event_type == event:
                    await callback(order)
        except Exception as e:
            logger.error(f"❌ Fehler bei Callback-Ausführung: {e}")
    
    # ==================== PUBLIC METHODS ====================
    
    async def cancel_order(self, order_id: str) -> bool:
        """❌ Order manuell stornieren"""
        try:
            if order_id in self.pending_orders:
                order = self.pending_orders[order_id]
                success = await self.bitget.cancel_order(order.exchange_id, order.symbol)
                
                if success:
                    order.status = OrderStatus.CANCELED
                    log_trade(f"🚫 Order manuell storniert: {order_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Order-Stornierung: {e}")
            return False
    
    def get_pending_orders(self) -> List[Order]:
        """📋 Alle pending Orders abrufen"""
        return list(self.pending_orders.values())
    
    def get_order_history(self, limit: int = 100) -> List[Order]:
        """📚 Order-Historie abrufen"""
        return self.order_history[-limit:]
    
    def get_performance_stats(self) -> Dict[str, float]:
        """📊 Performance-Statistiken abrufen"""
        return {
            'success_rate': self.order_success_rate,
            'avg_execution_time_ms': sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0,
            'total_orders': len(self.order_history),
            'pending_orders': len(self.pending_orders)
        }
    
    async def shutdown(self):
        """🛑 Order Manager herunterfahren"""
        try:
            self._running = False
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Alle pending Orders stornieren
            for order_id in list(self.pending_orders.keys()):
                await self.cancel_order(order_id)
            
            logger.info("✅ Order Manager heruntergefahren")
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren des Order Managers: {e}")
