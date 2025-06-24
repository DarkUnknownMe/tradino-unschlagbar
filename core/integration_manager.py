"""
TRADINO Integration Manager
Zentrale Koordination aller Systemkomponenten
"""

import os
import sys
import asyncio
import logging
import threading
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Füge das Projekt-Root zum Python Path hinzu
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core Imports
from core.bitget_trading_api import BitgetTradingAPI
from core.risk_management_system import RiskManagementSystem
from core.final_live_trading_system import LiveTradingSystem
from core.tp_sl_manager import TPSLManager
from core.tradino_telegram_panel import TradinoTelegramPanel

# Utilities
from tradino_unschlagbar.utils.logger_pro import LoggerPro
from tradino_unschlagbar.utils.config_manager import ConfigManager


class SystemState(Enum):
    """System Status Enumeration"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ComponentHealth:
    """Component Health Status"""
    name: str
    status: str
    last_check: datetime
    error_count: int
    metrics: Dict[str, Any]


class IntegrationManager:
    """Zentrale Koordination aller TRADINO Komponenten"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize Integration Manager"""
        self.logger = LoggerPro().get_logger("IntegrationManager")
        self.config_manager = ConfigManager()
        
        # System State
        self.system_state = SystemState.INITIALIZING
        self.startup_time = datetime.now()
        self.last_health_check = None
        
        # Components
        self.components = {}
        self.component_health = {}
        self.active_threads = {}
        
        # Configuration
        self.config = self._load_configuration(config_path)
        
        # Event handling
        self.event_handlers = {}
        self.shutdown_event = threading.Event()
        
        # Performance metrics
        self.metrics = {
            "trades_executed": 0,
            "signals_processed": 0,
            "errors_handled": 0,
            "uptime_seconds": 0
        }
        
        self.logger.info("🎯 TRADINO Integration Manager initialisiert")
    
    def _load_configuration(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Lade System Konfiguration"""
        try:
            default_config = {
                "trading": {
                    "mode": "live",  # live, demo, backtest
                    "risk_management": True,
                    "tp_sl_enabled": True,
                    "max_concurrent_trades": 5
                },
                "monitoring": {
                    "health_check_interval": 30,
                    "performance_log_interval": 300,
                    "telegram_notifications": True
                },
                "ai": {
                    "prediction_enabled": True,
                    "model_update_interval": 3600,
                    "confidence_threshold": 0.7
                },
                "risk": {
                    "max_portfolio_risk": 10.0,
                    "max_position_risk": 2.0,
                    "stop_loss_percentage": 2.0,
                    "take_profit_percentage": 4.0
                },
                "components": {
                    "bitget_api": {"enabled": True, "timeout": 30},
                    "risk_management": {"enabled": True, "strict_mode": True},
                    "tp_sl_manager": {"enabled": True, "monitoring_interval": 10},
                    "telegram_panel": {"enabled": True, "notification_level": "all"}
                }
            }
            
            if config_path and Path(config_path).exists():
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge configurations
                default_config.update(user_config)
            
            return default_config
            
        except Exception as e:
            self.logger.error(f"❌ Fehler beim Laden der Konfiguration: {e}")
            return {}
    
    async def initialize_system(self) -> bool:
        """Initialisiere alle Systemkomponenten"""
        try:
            self.logger.info("🚀 Starte Systeminitialisierung...")
            self.system_state = SystemState.INITIALIZING
            
            # Initialize components in correct order
            initialization_order = [
                ("config_manager", self._init_config_manager),
                ("bitget_api", self._init_bitget_api),
                ("risk_management", self._init_risk_management),
                ("tp_sl_manager", self._init_tp_sl_manager),
                ("telegram_panel", self._init_telegram_panel),
                ("live_trading", self._init_live_trading)
            ]
            
            for component_name, init_func in initialization_order:
                if not self.config.get("components", {}).get(component_name, {}).get("enabled", True):
                    self.logger.info(f"⏭️ {component_name} deaktiviert, überspringe...")
                    continue
                
                try:
                    self.logger.info(f"🔧 Initialisiere {component_name}...")
                    success = await init_func()
                    
                    if success:
                        self.component_health[component_name] = ComponentHealth(
                            name=component_name,
                            status="healthy",
                            last_check=datetime.now(),
                            error_count=0,
                            metrics={}
                        )
                        self.logger.info(f"✅ {component_name} erfolgreich initialisiert")
                    else:
                        self.logger.error(f"❌ Fehler bei Initialisierung von {component_name}")
                        return False
                        
                except Exception as e:
                    self.logger.error(f"🔥 Kritischer Fehler bei {component_name}: {e}")
                    return False
            
            # Setup event handlers
            self._setup_event_handlers()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.system_state = SystemState.READY
            self.logger.info("✅ System erfolgreich initialisiert!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"🔥 Kritischer Fehler bei Systeminitialisierung: {e}")
            self.system_state = SystemState.ERROR
            return False
    
    async def _init_config_manager(self) -> bool:
        """Initialize Configuration Manager"""
        try:
            self.components["config_manager"] = self.config_manager
            return True
        except Exception as e:
            self.logger.error(f"Config Manager Fehler: {e}")
            return False
    
    async def _init_bitget_api(self) -> bool:
        """Initialize Bitget API"""
        try:
            api_config = self.config.get("components", {}).get("bitget_api", {})
            
            self.components["bitget_api"] = BitgetTradingAPI()
            
            # Test API connection
            server_time = await self.components["bitget_api"].get_server_time()
            self.logger.info(f"🔗 Bitget API verbunden, Server Zeit: {server_time}")
            
            return True
        except Exception as e:
            self.logger.error(f"Bitget API Fehler: {e}")
            return False
    
    async def _init_risk_management(self) -> bool:
        """Initialize Risk Management"""
        try:
            risk_config = self.config.get("risk", {})
            
            self.components["risk_management"] = RiskManagementSystem()
            
            # Configure risk parameters
            if "max_portfolio_risk" in risk_config:
                self.components["risk_management"].max_portfolio_risk = risk_config["max_portfolio_risk"]
            
            return True
        except Exception as e:
            self.logger.error(f"Risk Management Fehler: {e}")
            return False
    
    async def _init_tp_sl_manager(self) -> bool:
        """Initialize TP/SL Manager"""
        try:
            if "bitget_api" not in self.components:
                self.logger.error("TP/SL Manager benötigt Bitget API")
                return False
            
            self.components["tp_sl_manager"] = TPSLManager()
            
            return True
        except Exception as e:
            self.logger.error(f"TP/SL Manager Fehler: {e}")
            return False
    
    async def _init_telegram_panel(self) -> bool:
        """Initialize Telegram Panel"""
        try:
            if not os.getenv("TELEGRAM_BOT_TOKEN"):
                self.logger.warning("⚠️ Telegram Bot Token nicht gefunden, deaktiviere Telegram")
                return True  # Not critical
            
            self.components["telegram_panel"] = TradinoTelegramPanel()
            
            # Send startup notification
            await self.send_notification("🚀 TRADINO System gestartet!", "startup")
            
            return True
        except Exception as e:
            self.logger.error(f"Telegram Panel Fehler: {e}")
            return True  # Not critical for system operation
    
    async def _init_live_trading(self) -> bool:
        """Initialize Live Trading System"""
        try:
            required_components = ["bitget_api", "risk_management"]
            
            for component in required_components:
                if component not in self.components:
                    self.logger.error(f"Live Trading benötigt {component}")
                    return False
            
            # Initialize with components
            self.components["live_trading"] = LiveTradingSystem()
            
            return True
        except Exception as e:
            self.logger.error(f"Live Trading Fehler: {e}")
            return False
    
    def _setup_event_handlers(self):
        """Setup Event Handlers"""
        self.event_handlers = {
            "trade_executed": self._handle_trade_executed,
            "trade_failed": self._handle_trade_failed,
            "tp_hit": self._handle_tp_hit,
            "sl_hit": self._handle_sl_hit,
            "risk_violation": self._handle_risk_violation,
            "system_error": self._handle_system_error
        }
    
    async def _start_background_tasks(self):
        """Starte Background Tasks"""
        # Health monitoring
        health_check_task = asyncio.create_task(self._health_monitor_loop())
        self.active_threads["health_monitor"] = health_check_task
        
        # Performance monitoring
        performance_task = asyncio.create_task(self._performance_monitor_loop())
        self.active_threads["performance_monitor"] = performance_task
        
        # TP/SL monitoring
        if "tp_sl_manager" in self.components:
            tp_sl_task = asyncio.create_task(self._tp_sl_monitor_loop())
            self.active_threads["tp_sl_monitor"] = tp_sl_task
    
    async def _health_monitor_loop(self):
        """Health Monitoring Loop"""
        while not self.shutdown_event.is_set():
            try:
                await self._perform_health_check()
                interval = self.config.get("monitoring", {}).get("health_check_interval", 30)
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Health Monitor Fehler: {e}")
                await asyncio.sleep(10)
    
    async def _performance_monitor_loop(self):
        """Performance Monitoring Loop"""
        while not self.shutdown_event.is_set():
            try:
                await self._update_performance_metrics()
                interval = self.config.get("monitoring", {}).get("performance_log_interval", 300)
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Performance Monitor Fehler: {e}")
                await asyncio.sleep(60)
    
    async def _tp_sl_monitor_loop(self):
        """TP/SL Monitoring Loop"""
        while not self.shutdown_event.is_set():
            try:
                if "tp_sl_manager" in self.components:
                    await self.components["tp_sl_manager"].monitor_all_positions()
                
                interval = self.config.get("components", {}).get("tp_sl_manager", {}).get("monitoring_interval", 10)
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"TP/SL Monitor Fehler: {e}")
                await asyncio.sleep(30)
    
    async def _perform_health_check(self):
        """Führe Health Check durch"""
        self.last_health_check = datetime.now()
        
        for component_name, component in self.components.items():
            try:
                # Basic health check
                if hasattr(component, 'health_check'):
                    is_healthy = await component.health_check()
                else:
                    is_healthy = True  # Assume healthy if no health_check method
                
                if component_name in self.component_health:
                    health = self.component_health[component_name]
                    health.last_check = datetime.now()
                    
                    if is_healthy:
                        health.status = "healthy"
                    else:
                        health.status = "unhealthy"
                        health.error_count += 1
                        
                        # Handle unhealthy component
                        await self._handle_unhealthy_component(component_name, health)
                
            except Exception as e:
                self.logger.error(f"Health Check Fehler für {component_name}: {e}")
                if component_name in self.component_health:
                    self.component_health[component_name].error_count += 1
    
    async def _update_performance_metrics(self):
        """Update Performance Metrics"""
        current_time = datetime.now()
        self.metrics["uptime_seconds"] = (current_time - self.startup_time).total_seconds()
        
        # Log performance metrics
        self.logger.info(f"📊 Performance Metrics: {self.metrics}")
        
        # Send to Telegram if enabled
        if self.config.get("monitoring", {}).get("telegram_notifications", False):
            metrics_msg = f"📊 System Metrics:\n"
            metrics_msg += f"⏱️ Uptime: {self.metrics['uptime_seconds']:.0f}s\n"
            metrics_msg += f"📈 Trades: {self.metrics['trades_executed']}\n"
            metrics_msg += f"🎯 Signals: {self.metrics['signals_processed']}\n"
            metrics_msg += f"❌ Errors: {self.metrics['errors_handled']}"
            
            await self.send_notification(metrics_msg, "metrics")
    
    async def _handle_unhealthy_component(self, component_name: str, health: ComponentHealth):
        """Handle Unhealthy Component"""
        self.logger.warning(f"⚠️ Component {component_name} is unhealthy (Errors: {health.error_count})")
        
        # Attempt recovery based on component type
        if health.error_count >= 3:
            self.logger.error(f"🔥 Component {component_name} has too many errors, attempting restart...")
            
            try:
                # Attempt component restart
                await self._restart_component(component_name)
                health.error_count = 0
                health.status = "restarted"
                
            except Exception as e:
                self.logger.error(f"❌ Restart of {component_name} failed: {e}")
                health.status = "failed"
                
                # Notify about critical component failure
                await self.send_notification(
                    f"🔥 Kritischer Fehler: {component_name} ausgefallen!", 
                    "critical"
                )
    
    async def _restart_component(self, component_name: str):
        """Restart specific component"""
        if component_name == "bitget_api":
            await self._init_bitget_api()
        elif component_name == "risk_management":
            await self._init_risk_management()
        elif component_name == "tp_sl_manager":
            await self._init_tp_sl_manager()
        elif component_name == "telegram_panel":
            await self._init_telegram_panel()
        elif component_name == "live_trading":
            await self._init_live_trading()
    
    # Event Handlers
    async def _handle_trade_executed(self, event_data: Dict[str, Any]):
        """Handle Trade Executed Event"""
        self.metrics["trades_executed"] += 1
        self.logger.info(f"✅ Trade ausgeführt: {event_data}")
        
        await self.send_notification(
            f"✅ Trade ausgeführt: {event_data.get('symbol', 'Unknown')} "
            f"{event_data.get('side', 'Unknown')} {event_data.get('amount', 0)}",
            "trade"
        )
    
    async def _handle_trade_failed(self, event_data: Dict[str, Any]):
        """Handle Trade Failed Event"""
        self.metrics["errors_handled"] += 1
        self.logger.error(f"❌ Trade fehlgeschlagen: {event_data}")
        
        await self.send_notification(
            f"❌ Trade fehlgeschlagen: {event_data.get('error', 'Unknown error')}",
            "error"
        )
    
    async def _handle_tp_hit(self, event_data: Dict[str, Any]):
        """Handle Take Profit Hit Event"""
        self.logger.info(f"🎯 Take Profit erreicht: {event_data}")
        
        await self.send_notification(
            f"🎯 Take Profit erreicht! Gewinn: {event_data.get('profit', 0)}",
            "tp"
        )
    
    async def _handle_sl_hit(self, event_data: Dict[str, Any]):
        """Handle Stop Loss Hit Event"""
        self.logger.info(f"🛑 Stop Loss erreicht: {event_data}")
        
        await self.send_notification(
            f"🛑 Stop Loss erreicht. Verlust: {event_data.get('loss', 0)}",
            "sl"
        )
    
    async def _handle_risk_violation(self, event_data: Dict[str, Any]):
        """Handle Risk Violation Event"""
        self.metrics["errors_handled"] += 1
        self.logger.warning(f"⚠️ Risk Violation: {event_data}")
        
        await self.send_notification(
            f"⚠️ Risk Limit Verletzung: {event_data.get('violation', 'Unknown')}",
            "risk"
        )
    
    async def _handle_system_error(self, event_data: Dict[str, Any]):
        """Handle System Error Event"""
        self.metrics["errors_handled"] += 1
        self.logger.error(f"🔥 System Error: {event_data}")
        
        await self.send_notification(
            f"🔥 System Error: {event_data.get('error', 'Unknown error')}",
            "critical"
        )
    
    async def emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit System Event"""
        if event_type in self.event_handlers:
            try:
                await self.event_handlers[event_type](event_data)
            except Exception as e:
                self.logger.error(f"Event Handler Fehler für {event_type}: {e}")
    
    async def send_notification(self, message: str, notification_type: str = "info"):
        """Send Notification via configured channels"""
        try:
            if "telegram_panel" in self.components:
                # Format message based on type
                if notification_type == "critical":
                    formatted_message = f"🔥 KRITISCH: {message}"
                elif notification_type == "error":
                    formatted_message = f"❌ FEHLER: {message}"
                elif notification_type == "warning":
                    formatted_message = f"⚠️ WARNUNG: {message}"
                else:
                    formatted_message = message
                
                await self.components["telegram_panel"].send_message(formatted_message)
                
        except Exception as e:
            self.logger.error(f"Notification Fehler: {e}")
    
    async def start_trading(self) -> bool:
        """Starte Trading System"""
        try:
            if self.system_state != SystemState.READY:
                self.logger.error("❌ System nicht bereit für Trading")
                return False
            
            self.system_state = SystemState.RUNNING
            self.logger.info("🚀 Trading System gestartet!")
            
            # Start live trading if configured
            if "live_trading" in self.components:
                trading_task = asyncio.create_task(
                    self.components["live_trading"].start_trading()
                )
                self.active_threads["live_trading"] = trading_task
            
            await self.send_notification("🚀 TRADINO Trading gestartet!", "startup")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Trading Start Fehler: {e}")
            return False
    
    async def stop_trading(self, emergency: bool = False):
        """Stoppe Trading System"""
        try:
            self.logger.info("🛑 Stoppe Trading System...")
            
            if emergency:
                self.system_state = SystemState.ERROR
                await self.send_notification("🚨 EMERGENCY STOP aktiviert!", "critical")
            else:
                self.system_state = SystemState.PAUSED
                await self.send_notification("⏸️ Trading pausiert", "info")
            
            # Cancel active trading tasks
            if "live_trading" in self.active_threads:
                self.active_threads["live_trading"].cancel()
            
        except Exception as e:
            self.logger.error(f"Trading Stop Fehler: {e}")
    
    async def shutdown_system(self):
        """Graceful System Shutdown"""
        try:
            self.logger.info("🛑 Starte System Shutdown...")
            self.system_state = SystemState.SHUTDOWN
            
            # Signal shutdown to all background tasks
            self.shutdown_event.set()
            
            # Cancel all active tasks
            for task_name, task in self.active_threads.items():
                self.logger.info(f"🛑 Stoppe {task_name}...")
                task.cancel()
                
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Close connections
            for component_name, component in self.components.items():
                if hasattr(component, 'close'):
                    try:
                        await component.close()
                        self.logger.info(f"✅ {component_name} geschlossen")
                    except Exception as e:
                        self.logger.error(f"Fehler beim Schließen von {component_name}: {e}")
            
            await self.send_notification("👋 TRADINO System heruntergefahren", "info")
            
            self.logger.info("✅ System erfolgreich heruntergefahren")
            
        except Exception as e:
            self.logger.error(f"Shutdown Fehler: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system_state": self.system_state.value,
            "uptime_seconds": (datetime.now() - self.startup_time).total_seconds(),
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "components": {
                name: {
                    "status": health.status,
                    "last_check": health.last_check.isoformat(),
                    "error_count": health.error_count
                }
                for name, health in self.component_health.items()
            },
            "metrics": self.metrics,
            "active_tasks": list(self.active_threads.keys())
        }


# Global instance
integration_manager = None


async def get_integration_manager(config_path: Optional[str] = None) -> IntegrationManager:
    """Get or create global Integration Manager instance"""
    global integration_manager
    
    if integration_manager is None:
        integration_manager = IntegrationManager(config_path)
        await integration_manager.initialize_system()
    
    return integration_manager 