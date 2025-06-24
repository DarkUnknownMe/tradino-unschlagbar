#!/usr/bin/env python3
"""
📊 TRADINO COMPLETE MONITORING SYSTEM - FINAL VERSION
Professional-grade logging and monitoring infrastructure - fully functional
"""

import os
import sys
import json
import time
import logging
import threading
import traceback
import psutil
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import warnings
warnings.filterwarnings('ignore')

class LogLevel(Enum):
    """📝 Enhanced Log Levels for Trading System"""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(Enum):
    """📊 Log Categories for System Organization"""
    TRADE = "trade"
    AI_DECISION = "ai_decision"
    RISK_EVENT = "risk_event"
    SYSTEM_HEALTH = "system_health"
    MARKET_DATA = "market_data"
    TELEGRAM_BOT = "telegram_bot"
    API_CONNECTION = "api_connection"
    PERFORMANCE = "performance"

class AlertSeverity(Enum):
    """🚨 Alert Severity Levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SystemMetrics:
    """💻 System Performance Metrics"""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_threads: int
    open_positions: int
    api_latency: float
    ai_processing_time: float
    last_trade_time: Optional[str]
    network_io: Dict[str, int]

@dataclass
class AlertEvent:
    """🚨 Alert Event Structure"""
    id: str
    timestamp: str
    severity: AlertSeverity
    category: LogCategory
    title: str
    message: str
    data: Dict[str, Any]

class ComprehensiveMonitoring:
    """📊 Complete Monitoring System"""
    
    def __init__(self, log_dir: str = "logs", config: Dict[str, Any] = None):
        self.log_dir = log_dir
        self.config = config or self._get_default_config()
        self.session_id = self._generate_session_id()
        self.start_time = time.time()
        
        # Logging infrastructure
        self.loggers = {}
        self.log_counts = {category.value: 0 for category in LogCategory}
        
        # Alert system
        self.active_alerts = {}
        self.alert_handlers = []
        self.alert_rate_limiter = {}
        
        # Performance tracking
        self.monitoring_threads = {}
        
        # Initialize system
        self._setup_logging()
        self._start_monitoring_threads()
        
        print(f"✅ Comprehensive Monitoring System initialized (Session: {self.session_id})")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """⚙️ Default monitoring configuration"""
        return {
            'log_level': 'INFO',
            'log_rotation_size': 100 * 1024 * 1024,  # 100MB
            'log_backup_count': 10,
            'alert_thresholds': {
                'cpu_usage': 80.0,
                'memory_usage': 85.0,
                'disk_usage': 90.0,
                'api_latency': 5.0,
                'error_rate': 0.05
            },
            'alert_rate_limit': 60,  # seconds
            'monitoring_interval': 10,  # seconds
            'console_output': True
        }
    
    def _generate_session_id(self) -> str:
        """🆔 Generate unique session ID"""
        return f"TMS-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    def _setup_logging(self):
        """📝 Setup structured logging system"""
        os.makedirs(self.log_dir, exist_ok=True)
        
        for category in LogCategory:
            logger = logging.getLogger(f"tradino.{category.value}")
            logger.setLevel(getattr(logging, self.config['log_level']))
            
            # Remove existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # JSON file handler
            log_file = os.path.join(self.log_dir, f"{category.value}.log")
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.config['log_rotation_size'],
                backupCount=self.config['log_backup_count']
            )
            
            # Custom JSON formatter
            class JsonFormatter(logging.Formatter):
                def format(self, record):
                    log_data = {
                        'timestamp': datetime.utcnow().isoformat() + 'Z',
                        'level': record.levelname,
                        'category': category.value,
                        'message': record.getMessage(),
                        'session_id': self.session_id
                    }
                    
                    # Add extra data if available
                    if hasattr(record, 'data'):
                        log_data['data'] = record.data
                    
                    return json.dumps(log_data)
            
            file_handler.setFormatter(JsonFormatter())
            logger.addHandler(file_handler)
            
            # Console handler (optional)
            if self.config.get('console_output', False):
                console_handler = logging.StreamHandler()
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)
            
            logger.propagate = False
            self.loggers[category] = logger
    
    def _start_monitoring_threads(self):
        """🧵 Start background monitoring threads"""
        # System metrics monitoring
        metrics_thread = threading.Thread(
            target=self._monitor_system_metrics,
            daemon=True,
            name="SystemMetricsMonitor"
        )
        metrics_thread.start()
        self.monitoring_threads['system_metrics'] = metrics_thread
        
        # Alert processor
        alert_thread = threading.Thread(
            target=self._process_alerts,
            daemon=True,
            name="AlertProcessor"
        )
        alert_thread.start()
        self.monitoring_threads['alert_processor'] = alert_thread
    
    def _monitor_system_metrics(self):
        """📊 Monitor system performance metrics"""
        while True:
            try:
                metrics = self._collect_system_metrics()
                
                # Log metrics
                self.info(LogCategory.SYSTEM_HEALTH, "System metrics update", {
                    'cpu_usage': metrics.cpu_usage,
                    'memory_usage': metrics.memory_usage,
                    'disk_usage': metrics.disk_usage,
                    'active_threads': metrics.active_threads
                })
                
                # Check thresholds
                self._check_alert_thresholds(metrics)
                
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                self.error(LogCategory.SYSTEM_HEALTH, f"System monitoring error: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """📈 Collect current system metrics"""
        try:
            # CPU and Memory
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            return SystemMetrics(
                timestamp=datetime.utcnow().isoformat() + 'Z',
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                active_threads=threading.active_count(),
                open_positions=0,  # Would be integrated with trading system
                api_latency=0.0,   # Would be measured from actual API calls
                ai_processing_time=0.0,  # Would be measured from AI operations
                last_trade_time=None,    # Would be from last trade
                network_io=network_io
            )
            
        except Exception as e:
            self.error(LogCategory.SYSTEM_HEALTH, f"Failed to collect metrics: {e}")
            return None
    
    def _check_alert_thresholds(self, metrics: SystemMetrics):
        """🚨 Check if metrics exceed alert thresholds"""
        thresholds = self.config['alert_thresholds']
        
        # CPU threshold
        if metrics.cpu_usage > thresholds['cpu_usage']:
            self._create_alert(
                AlertSeverity.HIGH,
                LogCategory.SYSTEM_HEALTH,
                "High CPU Usage",
                f"CPU usage at {metrics.cpu_usage:.1f}%",
                {'cpu_usage': metrics.cpu_usage, 'threshold': thresholds['cpu_usage']}
            )
        
        # Memory threshold
        if metrics.memory_usage > thresholds['memory_usage']:
            self._create_alert(
                AlertSeverity.HIGH,
                LogCategory.SYSTEM_HEALTH,
                "High Memory Usage",
                f"Memory usage at {metrics.memory_usage:.1f}%",
                {'memory_usage': metrics.memory_usage, 'threshold': thresholds['memory_usage']}
            )
        
        # Disk threshold
        if metrics.disk_usage > thresholds['disk_usage']:
            self._create_alert(
                AlertSeverity.CRITICAL,
                LogCategory.SYSTEM_HEALTH,
                "High Disk Usage",
                f"Disk usage at {metrics.disk_usage:.1f}%",
                {'disk_usage': metrics.disk_usage, 'threshold': thresholds['disk_usage']}
            )
    
    def _create_alert(self, severity: AlertSeverity, category: LogCategory, 
                     title: str, message: str, data: Dict[str, Any] = None):
        """🚨 Create and process alert"""
        alert_key = f"{category.value}_{title.lower().replace(' ', '_')}"
        
        # Rate limiting
        now = time.time()
        if alert_key in self.alert_rate_limiter:
            if now - self.alert_rate_limiter[alert_key] < self.config['alert_rate_limit']:
                return  # Skip due to rate limiting
        
        self.alert_rate_limiter[alert_key] = now
        self.active_alerts[alert_key] = now
        
        # Create alert event
        alert = AlertEvent(
            id=f"ALT-{int(now)}",
            timestamp=datetime.utcnow().isoformat() + 'Z',
            severity=severity,
            category=category,
            title=title,
            message=message,
            data=data or {}
        )
        
        # Log alert
        self.warn(category, f"ALERT: {title} - {message}", data)
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.error(LogCategory.SYSTEM_HEALTH, f"Alert handler failed: {e}")
    
    def _process_alerts(self):
        """🔄 Process and manage alerts"""
        while True:
            try:
                # Alert cleanup (remove old alerts)
                now = time.time()
                expired_alerts = []
                
                for alert_key, alert_time in self.active_alerts.items():
                    if now - alert_time > 3600:  # Remove alerts older than 1 hour
                        expired_alerts.append(alert_key)
                
                for alert_key in expired_alerts:
                    del self.active_alerts[alert_key]
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.error(LogCategory.SYSTEM_HEALTH, f"Alert processing error: {e}")
                time.sleep(10)
    
    def _is_system_healthy(self) -> bool:
        """💓 Check overall system health"""
        try:
            metrics = self._collect_system_metrics()
            if not metrics:
                return False
            
            thresholds = self.config['alert_thresholds']
            
            # Check all critical thresholds
            if (metrics.cpu_usage > thresholds['cpu_usage'] or
                metrics.memory_usage > thresholds['memory_usage'] or
                metrics.disk_usage > thresholds['disk_usage']):
                return False
            
            return True
            
        except Exception:
            return False
    
    # Logging methods
    def trace(self, category: LogCategory, message: str, data: Dict[str, Any] = None):
        """🔍 Trace level logging"""
        self._log(LogLevel.TRACE, category, message, data)
    
    def debug(self, category: LogCategory, message: str, data: Dict[str, Any] = None):
        """🐛 Debug level logging"""
        self._log(LogLevel.DEBUG, category, message, data)
    
    def info(self, category: LogCategory, message: str, data: Dict[str, Any] = None):
        """ℹ️ Info level logging"""
        self._log(LogLevel.INFO, category, message, data)
    
    def warn(self, category: LogCategory, message: str, data: Dict[str, Any] = None):
        """⚠️ Warning level logging"""
        self._log(LogLevel.WARN, category, message, data)
    
    def error(self, category: LogCategory, message: str, data: Dict[str, Any] = None):
        """❌ Error level logging"""
        self._log(LogLevel.ERROR, category, message, data)
    
    def critical(self, category: LogCategory, message: str, data: Dict[str, Any] = None):
        """🚨 Critical level logging"""
        self._log(LogLevel.CRITICAL, category, message, data)
        
        # Auto-create critical alert
        self._create_alert(
            AlertSeverity.CRITICAL,
            category,
            "Critical Error",
            message,
            data
        )
    
    def _log(self, level: LogLevel, category: LogCategory, message: str, data: Dict[str, Any] = None):
        """📝 Internal logging method"""
        try:
            logger = self.loggers.get(category)
            if not logger:
                return
            
            # Increment log count
            self.log_counts[category.value] += 1
            
            # Create log record
            log_level = getattr(logging, level.value)
            record = logger.makeRecord(
                logger.name, log_level, "", 0, message, (), None
            )
            
            if data:
                record.data = data
            
            logger.handle(record)
            
        except Exception as e:
            print(f"Logging error: {e}")
    
    # Specialized logging methods
    def log_trade_execution(self, trade_data: Dict[str, Any]):
        """💰 Log trade execution"""
        self.info(LogCategory.TRADE, f"Trade executed: {trade_data.get('action', 'unknown').upper()}", trade_data)
    
    def log_ai_decision(self, ai_data: Dict[str, Any]):
        """🤖 Log AI decision"""
        decision = ai_data.get('decision', 'unknown')
        confidence = ai_data.get('confidence', 0.0)
        self.info(LogCategory.AI_DECISION, f"AI Decision: {decision.upper()} (confidence: {confidence:.2f})", ai_data)
    
    def log_risk_event(self, risk_data: Dict[str, Any]):
        """🛡️ Log risk management event"""
        event_type = risk_data.get('event_type', 'unknown')
        severity = risk_data.get('severity', 'low')
        
        log_level = LogLevel.WARN if severity in ['low', 'medium'] else LogLevel.ERROR
        self._log(log_level, LogCategory.RISK_EVENT, f"Risk Event: {event_type}", risk_data)
    
    def log_api_call(self, api_data: Dict[str, Any]):
        """🌐 Log API call"""
        endpoint = api_data.get('endpoint', 'unknown')
        status = api_data.get('status', 'unknown')
        response_time = api_data.get('response_time_ms', 0)
        
        message = f"API Call: {endpoint} - {status} ({response_time}ms)"
        
        if status == 'success':
            self.info(LogCategory.API_CONNECTION, message, api_data)
        else:
            self.warn(LogCategory.API_CONNECTION, message, api_data)
    
    # System information methods
    def get_performance_summary(self) -> Dict[str, Any]:
        """📊 Get performance summary"""
        uptime_hours = (time.time() - self.start_time) / 3600
        total_logs = sum(self.log_counts.values())
        
        return {
            'session_id': self.session_id,
            'uptime_hours': uptime_hours,
            'total_logs': total_logs,
            'active_alerts': len(self.active_alerts),
            'monitoring_threads': len(self.monitoring_threads),
            'system_healthy': self._is_system_healthy(),
            'log_categories': len(LogCategory)
        }
    
    def get_system_metrics(self) -> Optional[SystemMetrics]:
        """📈 Get current system metrics"""
        return self._collect_system_metrics()
    
    def add_alert_handler(self, handler: Callable[[AlertEvent], None]):
        """🔔 Add alert notification handler"""
        self.alert_handlers.append(handler)
    
    def clear_alerts(self):
        """🧹 Clear all active alerts"""
        self.active_alerts.clear()
        self.info(LogCategory.SYSTEM_HEALTH, "All alerts cleared")

# Global monitoring instance
_monitoring_system = None

def initialize_monitoring_system(log_dir: str = "logs", config: Dict[str, Any] = None) -> ComprehensiveMonitoring:
    """🚀 Initialize global monitoring system"""
    global _monitoring_system
    
    try:
        _monitoring_system = ComprehensiveMonitoring(log_dir, config)
        return _monitoring_system
    except Exception as e:
        print(f"❌ Failed to initialize monitoring system: {e}")
        return None

def get_monitoring_system() -> Optional[ComprehensiveMonitoring]:
    """📊 Get global monitoring system instance"""
    return _monitoring_system

# Convenience functions
def log_trade(trade_data: Dict[str, Any]):
    """💰 Log trade (convenience function)"""
    if _monitoring_system:
        _monitoring_system.log_trade_execution(trade_data)

def log_ai_decision(ai_data: Dict[str, Any]):
    """🤖 Log AI decision (convenience function)"""
    if _monitoring_system:
        _monitoring_system.log_ai_decision(ai_data)

def log_risk_event(risk_data: Dict[str, Any]):
    """🛡️ Log risk event (convenience function)"""
    if _monitoring_system:
        _monitoring_system.log_risk_event(risk_data)

def log_api_call(api_data: Dict[str, Any]):
    """🌐 Log API call (convenience function)"""
    if _monitoring_system:
        _monitoring_system.log_api_call(api_data)

# Performance monitoring decorator
def monitor_performance(category: LogCategory = LogCategory.PERFORMANCE):
    """⚡ Performance monitoring decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # ms
                
                if _monitoring_system:
                    _monitoring_system.info(category, 
                        f"Function {func.__name__} executed successfully",
                        {
                            'function': func.__name__,
                            'execution_time_ms': execution_time,
                            'success': True
                        }
                    )
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000  # ms
                
                if _monitoring_system:
                    _monitoring_system.error(category,
                        f"Function {func.__name__} failed: {e}",
                        {
                            'function': func.__name__,
                            'execution_time_ms': execution_time,
                            'success': False,
                            'error': str(e)
                        }
                    )
                
                raise
        
        return wrapper
    return decorator

if __name__ == "__main__":
    print("📊 TRADINO COMPREHENSIVE MONITORING SYSTEM - FINAL VERSION")
    print("=" * 60)
    
    # Initialize system
    monitoring = initialize_monitoring_system()
    
    if monitoring:
        print("✅ System initialized successfully")
        
        # Test basic functionality
        monitoring.info(LogCategory.SYSTEM_HEALTH, "System started")
        monitoring.debug(LogCategory.AI_DECISION, "Test debug message")
        
        # Test structured logging
        log_trade({
            'trade_id': 'TEST-001',
            'symbol': 'BTC/USDT',
            'action': 'buy',
            'quantity': 0.001,
            'price': 50000.0
        })
        
        log_ai_decision({
            'decision': 'buy',
            'confidence': 0.85,
            'models_used': ['xgboost', 'lightgbm']
        })
        
        print("✅ Test logging completed")
        print(f"📊 Performance: {monitoring.get_performance_summary()}")
        
    else:
        print("❌ Failed to initialize monitoring system") 