"""
📢 TRADINO UNSCHLAGBAR - Notification Hub
Zentraler Notification Manager für alle Benachrichtigungen

Author: AI Trading Systems
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass

from connectors.telegram_commander import TelegramCommander
from utils.logger_pro import setup_logger
from utils.config_manager import ConfigManager

logger = setup_logger("NotificationHub")


class NotificationPriority(Enum):
    """Notification Priorität"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Notification:
    """Notification Model"""
    id: str
    type: str
    message: str
    priority: NotificationPriority
    timestamp: datetime
    metadata: Dict[str, Any]
    sent: bool = False
    retry_count: int = 0


class NotificationHub:
    """📢 Zentraler Notification Manager"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.telegram: Optional[TelegramCommander] = None
        
        # Notification Queue
        self.notification_queue: List[Notification] = []
        self.sent_notifications: List[Notification] = []
        
        # Rate Limiting
        self.rate_limits = {
            'trade_signals': {'max_per_hour': 50, 'count': 0, 'reset_time': datetime.utcnow()},
            'position_updates': {'max_per_hour': 30, 'count': 0, 'reset_time': datetime.utcnow()},
            'risk_alerts': {'max_per_hour': 20, 'count': 0, 'reset_time': datetime.utcnow()},
            'pnl_updates': {'max_per_hour': 10, 'count': 0, 'reset_time': datetime.utcnow()}
        }
        
        # Statistics
        self.total_sent = 0
        self.failed_sends = 0
        
        # Background Tasks
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def initialize(self) -> bool:
        """🔥 Notification Hub initialisieren"""
        try:
            logger.info("📢 Notification Hub wird initialisiert...")
            
            # Telegram Commander initialisieren
            self.telegram = TelegramCommander(self.config)
            if not await self.telegram.initialize():
                logger.error("❌ Telegram Initialisierung fehlgeschlagen")
                return False
            
            # Background Processor starten
            self._running = True
            self._processor_task = asyncio.create_task(self._process_notifications())
            
            logger.success("✅ Notification Hub erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ Notification Hub Initialisierung fehlgeschlagen: {e}")
            return False
    
    # ==================== NOTIFICATION METHODS ====================
    
    async def send_trade_signal(self, signal_data: Dict, priority: NotificationPriority = NotificationPriority.NORMAL):
        """🎯 Trading Signal Notification"""
        await self._queue_notification(
            'trade_signal', 
            f"Trading Signal: {signal_data.get('symbol')} {signal_data.get('side')}", 
            priority, 
            signal_data
        )
    
    async def send_trade_execution(self, trade_data: Dict, priority: NotificationPriority = NotificationPriority.HIGH):
        """⚡ Trade Execution Notification"""
        await self._queue_notification(
            'trade_execution', 
            f"Trade Executed: {trade_data.get('symbol')}", 
            priority, 
            trade_data
        )
    
    async def send_position_update(self, position_data: Dict, priority: NotificationPriority = NotificationPriority.NORMAL):
        """📊 Position Update Notification"""
        await self._queue_notification(
            'position_update', 
            f"Position Update: {position_data.get('symbol')}", 
            priority, 
            position_data
        )
    
    async def send_risk_alert(self, alert_data: Dict, priority: NotificationPriority = NotificationPriority.CRITICAL):
        """🚨 Risk Alert Notification"""
        await self._queue_notification(
            'risk_alert', 
            f"Risk Alert: {alert_data.get('type')}", 
            priority, 
            alert_data
        )
    
    async def send_daily_report(self, report_data: Dict):
        """📊 Daily Report Notification"""
        await self._queue_notification(
            'daily_report', 
            "Daily Trading Report", 
            NotificationPriority.NORMAL, 
            report_data
        )
    
    async def send_emergency_alert(self, message: str):
        """🚨 Emergency Alert"""
        await self._queue_notification(
            'emergency_alert', 
            f"EMERGENCY: {message}", 
            NotificationPriority.CRITICAL, 
            {'message': message}
        )
    
    # ==================== QUEUE MANAGEMENT ====================
    
    async def _queue_notification(self, notification_type: str, message: str, 
                                priority: NotificationPriority, metadata: Dict):
        """📝 Notification in Queue einreihen"""
        try:
            # Rate Limiting prüfen
            if not self._check_rate_limit(notification_type):
                logger.warning(f"⚠️ Rate Limit erreicht für {notification_type}")
                return
            
            # Notification erstellen
            notification = Notification(
                id=f"{notification_type}_{int(datetime.utcnow().timestamp())}",
                type=notification_type,
                message=message,
                priority=priority,
                timestamp=datetime.utcnow(),
                metadata=metadata
            )
            
            # In Queue einreihen (nach Priorität sortiert)
            self.notification_queue.append(notification)
            self.notification_queue.sort(key=lambda x: x.priority.value, reverse=True)
            
            logger.info(f"📝 Notification eingereiht: {notification_type} (Priority: {priority.name})")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Einreihen der Notification: {e}")
    
    async def _process_notifications(self):
        """🔄 Notification Queue abarbeiten"""
        while self._running:
            try:
                if self.notification_queue:
                    # Höchste Priorität zuerst
                    notification = self.notification_queue.pop(0)
                    
                    # Notification senden
                    success = await self._send_notification(notification)
                    
                    if success:
                        notification.sent = True
                        self.sent_notifications.append(notification)
                        self.total_sent += 1
                    else:
                        # Retry bei Fehlern (max 3x)
                        notification.retry_count += 1
                        if notification.retry_count < 3:
                            self.notification_queue.append(notification)
                        else:
                            logger.error(f"❌ Notification fehlgeschlagen nach 3 Versuchen: {notification.id}")
                            self.failed_sends += 1
                
                # Rate Limiting Reset prüfen
                self._reset_rate_limits()
                
                # Kurz warten
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Fehler beim Verarbeiten der Notifications: {e}")
                await asyncio.sleep(5)
    
    async def _send_notification(self, notification: Notification) -> bool:
        """📤 Einzelne Notification senden"""
        try:
            if not self.telegram:
                return False
            
            # Je nach Type verschiedene Telegram-Methoden aufrufen
            if notification.type == 'trade_signal':
                # Signal-Object aus Metadata rekonstruieren (vereinfacht)
                await self.telegram._send_message(f"🎯 {notification.message}")
                
            elif notification.type == 'trade_execution':
                await self.telegram._send_message(f"⚡ {notification.message}")
                
            elif notification.type == 'position_update':
                await self.telegram._send_message(f"📊 {notification.message}")
                
            elif notification.type == 'risk_alert':
                await self.telegram.send_risk_alert(
                    notification.metadata.get('type', 'unknown'),
                    notification.metadata.get('message', notification.message),
                    notification.metadata.get('severity', 'warning')
                )
                
            elif notification.type == 'daily_report':
                await self.telegram.send_daily_report(notification.metadata)
                
            elif notification.type == 'emergency_alert':
                await self.telegram.send_emergency_alert(notification.metadata.get('message'))
            
            logger.info(f"📤 Notification gesendet: {notification.type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Senden der Notification: {e}")
            return False
    
    # ==================== RATE LIMITING ====================
    
    def _check_rate_limit(self, notification_type: str) -> bool:
        """📊 Rate Limit prüfen"""
        try:
            if notification_type not in self.rate_limits:
                return True  # Keine Limits für unbekannte Types
            
            limit_info = self.rate_limits[notification_type]
            now = datetime.utcnow()
            
            # Reset wenn eine Stunde vergangen ist
            if (now - limit_info['reset_time']).seconds >= 3600:
                limit_info['count'] = 0
                limit_info['reset_time'] = now
            
            # Limit prüfen
            if limit_info['count'] >= limit_info['max_per_hour']:
                return False
            
            # Counter erhöhen
            limit_info['count'] += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Rate Limit Check: {e}")
            return True  # Bei Fehlern durchlassen
    
    def _reset_rate_limits(self):
        """🔄 Rate Limits zurücksetzen wenn nötig"""
        now = datetime.utcnow()
        for limit_info in self.rate_limits.values():
            if (now - limit_info['reset_time']).seconds >= 3600:
                limit_info['count'] = 0
                limit_info['reset_time'] = now
    
    # ==================== PUBLIC METHODS ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """📊 Notification Statistiken"""
        return {
            'total_sent': self.total_sent,
            'failed_sends': self.failed_sends,
            'queue_size': len(self.notification_queue),
            'success_rate': (self.total_sent / (self.total_sent + self.failed_sends)) if (self.total_sent + self.failed_sends) > 0 else 0,
            'rate_limits': self.rate_limits
        }
    
    def register_telegram_callback(self, event: str, callback: Callable):
        """🔧 Telegram Callback registrieren"""
        if self.telegram:
            self.telegram.register_callback(event, callback)
    
    async def shutdown(self):
        """🛑 Notification Hub herunterfahren"""
        try:
            self._running = False
            
            # Verbleidende Notifications senden
            while self.notification_queue:
                notification = self.notification_queue.pop(0)
                await self._send_notification(notification)
            
            # Background Task stoppen
            if self._processor_task:
                self._processor_task.cancel()
                try:
                    await self._processor_task
                except asyncio.CancelledError:
                    pass
            
            # Telegram herunterfahren
            if self.telegram:
                await self.telegram.shutdown()
            
            logger.info("✅ Notification Hub heruntergefahren")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren des Notification Hubs: {e}")
