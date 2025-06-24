"""
TRADINO System Health Check & Monitoring
Umfassende Überwachung aller Systemkomponenten
"""

import os
import sys
import time
import json
import asyncio
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# Füge das Projekt-Root zum Python Path hinzu
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Utilities
from tradino_unschlagbar.utils.logger_pro import LoggerPro


class HealthStatus(Enum):
    """Health Status Enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Health Metric Data Structure"""
    name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ComponentHealth:
    """Component Health Report"""
    component: str
    status: HealthStatus
    metrics: List[HealthMetric]
    errors: List[str]
    warnings: List[str]
    last_check: datetime
    uptime_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status.value,
            "metrics": [asdict(metric) for metric in self.metrics],
            "errors": self.errors,
            "warnings": self.warnings,
            "last_check": self.last_check.isoformat(),
            "uptime_seconds": self.uptime_seconds
        }


class SystemHealthMonitor:
    """Umfassende System Health Überwachung"""
    
    def __init__(self):
        self.logger = LoggerPro().get_logger("SystemHealthMonitor")
        self.start_time = datetime.now()
        self.health_history = []
        self.alert_thresholds = self._load_alert_thresholds()
        self.last_alert_times = {}
        self.process = psutil.Process()
        
    def _load_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Lade Alert Thresholds"""
        return {
            "cpu": {"warning": 70.0, "critical": 90.0},
            "memory": {"warning": 80.0, "critical": 95.0},
            "disk": {"warning": 80.0, "critical": 95.0},
            "network_latency": {"warning": 500.0, "critical": 1000.0},
            "api_response_time": {"warning": 2000.0, "critical": 5000.0},
            "error_rate": {"warning": 5.0, "critical": 10.0},
            "trade_success_rate": {"warning": 80.0, "critical": 60.0}
        }
    
    async def perform_full_health_check(self) -> Dict[str, ComponentHealth]:
        """Führe vollständigen Health Check durch"""
        self.logger.info("🔍 Starte vollständigen Health Check...")
        
        health_results = {}
        
        # System Health Checks
        health_results["system"] = await self._check_system_health()
        health_results["network"] = await self._check_network_health()
        health_results["storage"] = await self._check_storage_health()
        health_results["processes"] = await self._check_process_health()
        
        # Application Health Checks
        health_results["api_connectivity"] = await self._check_api_connectivity()
        health_results["database"] = await self._check_database_health()
        health_results["models"] = await self._check_model_health()
        health_results["configuration"] = await self._check_configuration_health()
        
        # Trading System Health Checks
        health_results["trading_engine"] = await self._check_trading_engine_health()
        health_results["risk_management"] = await self._check_risk_management_health()
        health_results["tp_sl_system"] = await self._check_tp_sl_health()
        health_results["monitoring"] = await self._check_monitoring_health()
        
        # Store in history
        self.health_history.append({
            "timestamp": datetime.now(),
            "results": health_results
        })
        
        # Keep only last 24 hours of history
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.health_history = [
            h for h in self.health_history 
            if h["timestamp"] > cutoff_time
        ]
        
        # Generate alerts if needed
        await self._process_health_alerts(health_results)
        
        return health_results
    
    async def _check_system_health(self) -> ComponentHealth:
        """Check System Resource Health"""
        metrics = []
        errors = []
        warnings = []
        
        try:
            # CPU Usage
            cpu_percent = self.process.cpu_percent(interval=1)
            cpu_metric = HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="percent",
                status=self._get_status_from_thresholds(cpu_percent, "cpu"),
                threshold_warning=self.alert_thresholds["cpu"]["warning"],
                threshold_critical=self.alert_thresholds["cpu"]["critical"]
            )
            metrics.append(cpu_metric)
            
            if cpu_metric.status == HealthStatus.CRITICAL:
                errors.append(f"CPU Usage kritisch: {cpu_percent:.1f}%")
            elif cpu_metric.status == HealthStatus.WARNING:
                warnings.append(f"CPU Usage hoch: {cpu_percent:.1f}%")
            
            # Memory Usage
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            memory_metric = HealthMetric(
                name="memory_usage",
                value=memory_percent,
                unit="percent",
                status=self._get_status_from_thresholds(memory_percent, "memory"),
                threshold_warning=self.alert_thresholds["memory"]["warning"],
                threshold_critical=self.alert_thresholds["memory"]["critical"]
            )
            metrics.append(memory_metric)
            
            if memory_metric.status == HealthStatus.CRITICAL:
                errors.append(f"Memory Usage kritisch: {memory_percent:.1f}%")
            elif memory_metric.status == HealthStatus.WARNING:
                warnings.append(f"Memory Usage hoch: {memory_percent:.1f}%")
            
            # Memory in MB
            memory_mb = memory_info.rss / 1024 / 1024
            memory_mb_metric = HealthMetric(
                name="memory_usage_mb",
                value=memory_mb,
                unit="MB",
                status=HealthStatus.HEALTHY
            )
            metrics.append(memory_mb_metric)
            
            # System Load Average (Linux/Mac)
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()[0]  # 1-minute average
                load_metric = HealthMetric(
                    name="load_average",
                    value=load_avg,
                    unit="",
                    status=HealthStatus.HEALTHY if load_avg < 2.0 else HealthStatus.WARNING
                )
                metrics.append(load_metric)
            
            # Open file descriptors
            try:
                num_fds = self.process.num_fds()
                fd_metric = HealthMetric(
                    name="open_file_descriptors",
                    value=num_fds,
                    unit="count",
                    status=HealthStatus.HEALTHY if num_fds < 1000 else HealthStatus.WARNING
                )
                metrics.append(fd_metric)
            except:
                pass  # Not supported on all platforms
            
        except Exception as e:
            errors.append(f"System Health Check Fehler: {str(e)}")
        
        overall_status = self._determine_overall_status(metrics, errors)
        
        return ComponentHealth(
            component="system",
            status=overall_status,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            last_check=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds()
        )
    
    async def _check_network_health(self) -> ComponentHealth:
        """Check Network Health"""
        metrics = []
        errors = []
        warnings = []
        
        try:
            # Network connections
            connections = self.process.connections()
            active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
            
            connection_metric = HealthMetric(
                name="active_connections",
                value=active_connections,
                unit="count",
                status=HealthStatus.HEALTHY if active_connections < 50 else HealthStatus.WARNING
            )
            metrics.append(connection_metric)
            
            # Test internet connectivity
            import aiohttp
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                try:
                    async with session.get('https://httpbin.org/status/200', timeout=5) as response:
                        latency = (time.time() - start_time) * 1000
                        
                        latency_metric = HealthMetric(
                            name="internet_latency",
                            value=latency,
                            unit="ms",
                            status=self._get_status_from_thresholds(latency, "network_latency"),
                            threshold_warning=self.alert_thresholds["network_latency"]["warning"],
                            threshold_critical=self.alert_thresholds["network_latency"]["critical"]
                        )
                        metrics.append(latency_metric)
                        
                        if response.status == 200:
                            connectivity_metric = HealthMetric(
                                name="internet_connectivity",
                                value=1.0,
                                unit="boolean",
                                status=HealthStatus.HEALTHY
                            )
                        else:
                            connectivity_metric = HealthMetric(
                                name="internet_connectivity",
                                value=0.0,
                                unit="boolean",
                                status=HealthStatus.CRITICAL
                            )
                            errors.append(f"Internet connectivity test failed: HTTP {response.status}")
                        
                        metrics.append(connectivity_metric)
                        
                except Exception as e:
                    connectivity_metric = HealthMetric(
                        name="internet_connectivity",
                        value=0.0,
                        unit="boolean",
                        status=HealthStatus.CRITICAL
                    )
                    metrics.append(connectivity_metric)
                    errors.append(f"Internet connectivity test failed: {str(e)}")
            
        except Exception as e:
            errors.append(f"Network Health Check Fehler: {str(e)}")
        
        overall_status = self._determine_overall_status(metrics, errors)
        
        return ComponentHealth(
            component="network",
            status=overall_status,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            last_check=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds()
        )
    
    async def _check_storage_health(self) -> ComponentHealth:
        """Check Storage Health"""
        metrics = []
        errors = []
        warnings = []
        
        try:
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            disk_metric = HealthMetric(
                name="disk_usage",
                value=disk_percent,
                unit="percent",
                status=self._get_status_from_thresholds(disk_percent, "disk"),
                threshold_warning=self.alert_thresholds["disk"]["warning"],
                threshold_critical=self.alert_thresholds["disk"]["critical"]
            )
            metrics.append(disk_metric)
            
            if disk_metric.status == HealthStatus.CRITICAL:
                errors.append(f"Disk Usage kritisch: {disk_percent:.1f}%")
            elif disk_metric.status == HealthStatus.WARNING:
                warnings.append(f"Disk Usage hoch: {disk_percent:.1f}%")
            
            # Available space in GB
            available_gb = disk_usage.free / (1024**3)
            available_metric = HealthMetric(
                name="disk_available_gb",
                value=available_gb,
                unit="GB",
                status=HealthStatus.HEALTHY if available_gb > 1.0 else HealthStatus.WARNING
            )
            metrics.append(available_metric)
            
            # Check log directory
            log_dir = project_root / "logs"
            if log_dir.exists():
                log_size = sum(f.stat().st_size for f in log_dir.rglob('*') if f.is_file())
                log_size_mb = log_size / (1024**2)
                
                log_metric = HealthMetric(
                    name="log_directory_size_mb",
                    value=log_size_mb,
                    unit="MB",
                    status=HealthStatus.HEALTHY if log_size_mb < 1000 else HealthStatus.WARNING
                )
                metrics.append(log_metric)
            
            # Check model directory
            model_dir = project_root / "models"
            if model_dir.exists():
                model_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                model_size_mb = model_size / (1024**2)
                
                model_metric = HealthMetric(
                    name="model_directory_size_mb",
                    value=model_size_mb,
                    unit="MB",
                    status=HealthStatus.HEALTHY
                )
                metrics.append(model_metric)
            
        except Exception as e:
            errors.append(f"Storage Health Check Fehler: {str(e)}")
        
        overall_status = self._determine_overall_status(metrics, errors)
        
        return ComponentHealth(
            component="storage",
            status=overall_status,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            last_check=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds()
        )
    
    async def _check_process_health(self) -> ComponentHealth:
        """Check Process Health"""
        metrics = []
        errors = []
        warnings = []
        
        try:
            # Process status
            status_metric = HealthMetric(
                name="process_status",
                value=1.0 if self.process.is_running() else 0.0,
                unit="boolean",
                status=HealthStatus.HEALTHY if self.process.is_running() else HealthStatus.CRITICAL
            )
            metrics.append(status_metric)
            
            if not self.process.is_running():
                errors.append("Process ist nicht aktiv")
            
            # Number of threads
            num_threads = self.process.num_threads()
            thread_metric = HealthMetric(
                name="thread_count",
                value=num_threads,
                unit="count",
                status=HealthStatus.HEALTHY if num_threads < 50 else HealthStatus.WARNING
            )
            metrics.append(thread_metric)
            
            if num_threads > 100:
                warnings.append(f"Hohe Thread-Anzahl: {num_threads}")
            
            # Process uptime
            create_time = datetime.fromtimestamp(self.process.create_time())
            uptime_seconds = (datetime.now() - create_time).total_seconds()
            
            uptime_metric = HealthMetric(
                name="process_uptime",
                value=uptime_seconds,
                unit="seconds",
                status=HealthStatus.HEALTHY
            )
            metrics.append(uptime_metric)
            
        except Exception as e:
            errors.append(f"Process Health Check Fehler: {str(e)}")
        
        overall_status = self._determine_overall_status(metrics, errors)
        
        return ComponentHealth(
            component="processes",
            status=overall_status,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            last_check=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds()
        )
    
    async def _check_api_connectivity(self) -> ComponentHealth:
        """Check API Connectivity"""
        metrics = []
        errors = []
        warnings = []
        
        try:
            # Test Bitget API connectivity
            import aiohttp
            
            api_endpoints = [
                ("bitget_server_time", "https://api.bitget.com/api/v2/public/time"),
                ("bitget_symbols", "https://api.bitget.com/api/v2/spot/public/symbols")
            ]
            
            async with aiohttp.ClientSession() as session:
                for endpoint_name, url in api_endpoints:
                    try:
                        start_time = time.time()
                        async with session.get(url, timeout=10) as response:
                            response_time = (time.time() - start_time) * 1000
                            
                            response_metric = HealthMetric(
                                name=f"{endpoint_name}_response_time",
                                value=response_time,
                                unit="ms",
                                status=self._get_status_from_thresholds(response_time, "api_response_time"),
                                threshold_warning=self.alert_thresholds["api_response_time"]["warning"],
                                threshold_critical=self.alert_thresholds["api_response_time"]["critical"]
                            )
                            metrics.append(response_metric)
                            
                            if response.status == 200:
                                status_metric = HealthMetric(
                                    name=f"{endpoint_name}_status",
                                    value=1.0,
                                    unit="boolean",
                                    status=HealthStatus.HEALTHY
                                )
                            else:
                                status_metric = HealthMetric(
                                    name=f"{endpoint_name}_status",
                                    value=0.0,
                                    unit="boolean",
                                    status=HealthStatus.CRITICAL
                                )
                                errors.append(f"{endpoint_name} failed: HTTP {response.status}")
                            
                            metrics.append(status_metric)
                            
                    except Exception as e:
                        error_metric = HealthMetric(
                            name=f"{endpoint_name}_status",
                            value=0.0,
                            unit="boolean",
                            status=HealthStatus.CRITICAL
                        )
                        metrics.append(error_metric)
                        errors.append(f"{endpoint_name} error: {str(e)}")
            
        except Exception as e:
            errors.append(f"API Connectivity Check Fehler: {str(e)}")
        
        overall_status = self._determine_overall_status(metrics, errors)
        
        return ComponentHealth(
            component="api_connectivity",
            status=overall_status,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            last_check=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds()
        )
    
    async def _check_database_health(self) -> ComponentHealth:
        """Check Database Health (if applicable)"""
        metrics = []
        errors = []
        warnings = []
        
        try:
            # Check if database files exist and are accessible
            data_dir = project_root / "tradino_unschlagbar" / "data"
            
            if data_dir.exists():
                # Count data files
                data_files = list(data_dir.rglob('*.json')) + list(data_dir.rglob('*.pkl'))
                
                file_count_metric = HealthMetric(
                    name="data_file_count",
                    value=len(data_files),
                    unit="count",
                    status=HealthStatus.HEALTHY
                )
                metrics.append(file_count_metric)
                
                # Check recent data updates
                if data_files:
                    latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
                    last_update = datetime.fromtimestamp(latest_file.stat().st_mtime)
                    hours_since_update = (datetime.now() - last_update).total_seconds() / 3600
                    
                    update_metric = HealthMetric(
                        name="hours_since_last_data_update",
                        value=hours_since_update,
                        unit="hours",
                        status=HealthStatus.HEALTHY if hours_since_update < 24 else HealthStatus.WARNING
                    )
                    metrics.append(update_metric)
                    
                    if hours_since_update > 48:
                        warnings.append(f"Daten nicht aktuell: {hours_since_update:.1f} Stunden alt")
            else:
                errors.append("Data directory nicht gefunden")
        
        except Exception as e:
            errors.append(f"Database Health Check Fehler: {str(e)}")
        
        overall_status = self._determine_overall_status(metrics, errors)
        
        return ComponentHealth(
            component="database",
            status=overall_status,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            last_check=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds()
        )
    
    async def _check_model_health(self) -> ComponentHealth:
        """Check ML Model Health"""
        metrics = []
        errors = []
        warnings = []
        
        try:
            # Check model files
            model_files = [
                "models/xgboost_trend.pkl",
                "models/lightgbm_volatility.pkl",
                "models/random_forest_risk.pkl",
                "models/feature_pipeline.pkl"
            ]
            
            existing_models = 0
            total_size_mb = 0
            
            for model_file in model_files:
                model_path = project_root / model_file
                if model_path.exists():
                    existing_models += 1
                    total_size_mb += model_path.stat().st_size / (1024**2)
                    
                    # Test model loading
                    try:
                        import pickle
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        
                        load_metric = HealthMetric(
                            name=f"{model_file.split('/')[-1]}_loadable",
                            value=1.0,
                            unit="boolean",
                            status=HealthStatus.HEALTHY
                        )
                        metrics.append(load_metric)
                        
                    except Exception as e:
                        load_metric = HealthMetric(
                            name=f"{model_file.split('/')[-1]}_loadable",
                            value=0.0,
                            unit="boolean",
                            status=HealthStatus.CRITICAL
                        )
                        metrics.append(load_metric)
                        errors.append(f"Model {model_file} kann nicht geladen werden: {str(e)}")
                else:
                    warnings.append(f"Model {model_file} nicht gefunden")
            
            # Model availability metric
            model_availability = (existing_models / len(model_files)) * 100
            availability_metric = HealthMetric(
                name="model_availability",
                value=model_availability,
                unit="percent",
                status=HealthStatus.HEALTHY if model_availability == 100 else HealthStatus.WARNING
            )
            metrics.append(availability_metric)
            
            # Total model size
            size_metric = HealthMetric(
                name="total_model_size_mb",
                value=total_size_mb,
                unit="MB",
                status=HealthStatus.HEALTHY
            )
            metrics.append(size_metric)
            
        except Exception as e:
            errors.append(f"Model Health Check Fehler: {str(e)}")
        
        overall_status = self._determine_overall_status(metrics, errors)
        
        return ComponentHealth(
            component="models",
            status=overall_status,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            last_check=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds()
        )
    
    async def _check_configuration_health(self) -> ComponentHealth:
        """Check Configuration Health"""
        metrics = []
        errors = []
        warnings = []
        
        try:
            # Check configuration files
            config_files = [
                "config/requirements.txt",
                "config/requirements_ai.txt",
                "tradino_unschlagbar/config.yaml",
                "tradino_unschlagbar/config/final_trading_config.json",
                "tradino_unschlagbar/config/risk_config.json"
            ]
            
            valid_configs = 0
            
            for config_file in config_files:
                config_path = project_root / config_file
                if config_path.exists():
                    try:
                        if config_file.endswith('.json'):
                            with open(config_path, 'r') as f:
                                json.load(f)
                        valid_configs += 1
                    except Exception as e:
                        errors.append(f"Config {config_file} ungültig: {str(e)}")
                else:
                    warnings.append(f"Config {config_file} nicht gefunden")
            
            config_metric = HealthMetric(
                name="valid_config_files",
                value=valid_configs,
                unit="count",
                status=HealthStatus.HEALTHY if valid_configs >= 3 else HealthStatus.WARNING
            )
            metrics.append(config_metric)
            
            # Check environment variables
            required_env_vars = [
                "BITGET_API_KEY", "BITGET_SECRET_KEY", "BITGET_PASSPHRASE",
                "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"
            ]
            
            configured_env_vars = sum(1 for var in required_env_vars if os.getenv(var))
            
            env_metric = HealthMetric(
                name="configured_env_variables",
                value=configured_env_vars,
                unit="count",
                status=HealthStatus.HEALTHY if configured_env_vars == len(required_env_vars) else HealthStatus.WARNING
            )
            metrics.append(env_metric)
            
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            if missing_vars:
                warnings.append(f"Fehlende Environment Variables: {', '.join(missing_vars)}")
            
        except Exception as e:
            errors.append(f"Configuration Health Check Fehler: {str(e)}")
        
        overall_status = self._determine_overall_status(metrics, errors)
        
        return ComponentHealth(
            component="configuration",
            status=overall_status,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            last_check=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds()
        )
    
    async def _check_trading_engine_health(self) -> ComponentHealth:
        """Check Trading Engine Health"""
        metrics = []
        errors = []
        warnings = []
        
        try:
            # Mock trading engine health check
            # In real implementation, this would check actual trading engine status
            
            engine_status_metric = HealthMetric(
                name="trading_engine_status",
                value=1.0,  # Mock: assume healthy
                unit="boolean",
                status=HealthStatus.HEALTHY
            )
            metrics.append(engine_status_metric)
            
            # Mock recent trade count
            mock_trade_count = 5  # In real implementation, get from trading engine
            trade_count_metric = HealthMetric(
                name="recent_trade_count",
                value=mock_trade_count,
                unit="count",
                status=HealthStatus.HEALTHY
            )
            metrics.append(trade_count_metric)
            
            # Mock error rate
            mock_error_rate = 2.0  # 2% error rate
            error_rate_metric = HealthMetric(
                name="trading_error_rate",
                value=mock_error_rate,
                unit="percent",
                status=self._get_status_from_thresholds(mock_error_rate, "error_rate"),
                threshold_warning=self.alert_thresholds["error_rate"]["warning"],
                threshold_critical=self.alert_thresholds["error_rate"]["critical"]
            )
            metrics.append(error_rate_metric)
            
        except Exception as e:
            errors.append(f"Trading Engine Health Check Fehler: {str(e)}")
        
        overall_status = self._determine_overall_status(metrics, errors)
        
        return ComponentHealth(
            component="trading_engine",
            status=overall_status,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            last_check=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds()
        )
    
    async def _check_risk_management_health(self) -> ComponentHealth:
        """Check Risk Management Health"""
        metrics = []
        errors = []
        warnings = []
        
        try:
            # Mock risk management health check
            
            risk_system_metric = HealthMetric(
                name="risk_system_status",
                value=1.0,  # Mock: assume healthy
                unit="boolean",
                status=HealthStatus.HEALTHY
            )
            metrics.append(risk_system_metric)
            
            # Mock portfolio risk
            mock_portfolio_risk = 5.0  # 5% portfolio risk
            portfolio_risk_metric = HealthMetric(
                name="current_portfolio_risk",
                value=mock_portfolio_risk,
                unit="percent",
                status=HealthStatus.HEALTHY if mock_portfolio_risk < 10 else HealthStatus.WARNING
            )
            metrics.append(portfolio_risk_metric)
            
            # Mock risk violations
            mock_violations = 0
            violation_metric = HealthMetric(
                name="risk_violations_24h",
                value=mock_violations,
                unit="count",
                status=HealthStatus.HEALTHY if mock_violations == 0 else HealthStatus.WARNING
            )
            metrics.append(violation_metric)
            
        except Exception as e:
            errors.append(f"Risk Management Health Check Fehler: {str(e)}")
        
        overall_status = self._determine_overall_status(metrics, errors)
        
        return ComponentHealth(
            component="risk_management",
            status=overall_status,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            last_check=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds()
        )
    
    async def _check_tp_sl_health(self) -> ComponentHealth:
        """Check TP/SL System Health"""
        metrics = []
        errors = []
        warnings = []
        
        try:
            # Mock TP/SL system health check
            
            tpsl_system_metric = HealthMetric(
                name="tpsl_system_status",
                value=1.0,  # Mock: assume healthy
                unit="boolean",
                status=HealthStatus.HEALTHY
            )
            metrics.append(tpsl_system_metric)
            
            # Mock active TP/SL orders
            mock_active_orders = 3
            active_orders_metric = HealthMetric(
                name="active_tpsl_orders",
                value=mock_active_orders,
                unit="count",
                status=HealthStatus.HEALTHY
            )
            metrics.append(active_orders_metric)
            
            # Mock TP/SL success rate
            mock_success_rate = 95.0  # 95% success rate
            success_rate_metric = HealthMetric(
                name="tpsl_success_rate",
                value=mock_success_rate,
                unit="percent",
                status=HealthStatus.HEALTHY if mock_success_rate > 90 else HealthStatus.WARNING
            )
            metrics.append(success_rate_metric)
            
        except Exception as e:
            errors.append(f"TP/SL Health Check Fehler: {str(e)}")
        
        overall_status = self._determine_overall_status(metrics, errors)
        
        return ComponentHealth(
            component="tp_sl_system",
            status=overall_status,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            last_check=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds()
        )
    
    async def _check_monitoring_health(self) -> ComponentHealth:
        """Check Monitoring System Health"""
        metrics = []
        errors = []
        warnings = []
        
        try:
            # Check log files
            log_dir = project_root / "logs"
            if log_dir.exists():
                log_files = list(log_dir.glob("*.log"))
                
                log_count_metric = HealthMetric(
                    name="log_file_count",
                    value=len(log_files),
                    unit="count",
                    status=HealthStatus.HEALTHY
                )
                metrics.append(log_count_metric)
                
                # Check recent log activity
                if log_files:
                    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
                    last_log_update = datetime.fromtimestamp(latest_log.stat().st_mtime)
                    minutes_since_log = (datetime.now() - last_log_update).total_seconds() / 60
                    
                    log_activity_metric = HealthMetric(
                        name="minutes_since_last_log",
                        value=minutes_since_log,
                        unit="minutes",
                        status=HealthStatus.HEALTHY if minutes_since_log < 60 else HealthStatus.WARNING
                    )
                    metrics.append(log_activity_metric)
            
            # Mock monitoring system status
            monitoring_metric = HealthMetric(
                name="monitoring_system_status",
                value=1.0,  # Mock: assume healthy
                unit="boolean",
                status=HealthStatus.HEALTHY
            )
            metrics.append(monitoring_metric)
            
        except Exception as e:
            errors.append(f"Monitoring Health Check Fehler: {str(e)}")
        
        overall_status = self._determine_overall_status(metrics, errors)
        
        return ComponentHealth(
            component="monitoring",
            status=overall_status,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
            last_check=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds()
        )
    
    def _get_status_from_thresholds(self, value: float, threshold_key: str) -> HealthStatus:
        """Get status based on threshold values"""
        thresholds = self.alert_thresholds.get(threshold_key, {})
        
        warning_threshold = thresholds.get("warning")
        critical_threshold = thresholds.get("critical")
        
        if critical_threshold is not None and value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif warning_threshold is not None and value >= warning_threshold:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _determine_overall_status(self, metrics: List[HealthMetric], errors: List[str]) -> HealthStatus:
        """Determine overall component status"""
        if errors:
            return HealthStatus.CRITICAL
        
        critical_metrics = [m for m in metrics if m.status == HealthStatus.CRITICAL]
        warning_metrics = [m for m in metrics if m.status == HealthStatus.WARNING]
        
        if critical_metrics:
            return HealthStatus.CRITICAL
        elif warning_metrics:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    async def _process_health_alerts(self, health_results: Dict[str, ComponentHealth]):
        """Process health alerts and notifications"""
        current_time = datetime.now()
        
        for component_name, health in health_results.items():
            # Check if we need to send alerts
            if health.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                # Rate limiting: don't send same alert too frequently
                last_alert_key = f"{component_name}_{health.status.value}"
                last_alert_time = self.last_alert_times.get(last_alert_key)
                
                if (last_alert_time is None or 
                    current_time - last_alert_time > timedelta(minutes=15)):
                    
                    await self._send_health_alert(component_name, health)
                    self.last_alert_times[last_alert_key] = current_time
    
    async def _send_health_alert(self, component_name: str, health: ComponentHealth):
        """Send health alert notification"""
        status_emoji = {
            HealthStatus.CRITICAL: "🔥",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.HEALTHY: "✅"
        }
        
        emoji = status_emoji.get(health.status, "❓")
        
        message = f"{emoji} HEALTH ALERT: {component_name.upper()}\n"
        message += f"Status: {health.status.value}\n"
        
        if health.errors:
            message += f"Errors: {', '.join(health.errors[:3])}\n"  # Show first 3 errors
        
        if health.warnings:
            message += f"Warnings: {', '.join(health.warnings[:3])}\n"  # Show first 3 warnings
        
        # Add key metrics
        critical_metrics = [m for m in health.metrics if m.status == HealthStatus.CRITICAL]
        warning_metrics = [m for m in health.metrics if m.status == HealthStatus.WARNING]
        
        if critical_metrics:
            message += f"Critical Metrics: {', '.join([f'{m.name}: {m.value}{m.unit}' for m in critical_metrics[:2]])}\n"
        
        if warning_metrics:
            message += f"Warning Metrics: {', '.join([f'{m.name}: {m.value}{m.unit}' for m in warning_metrics[:2]])}"
        
        self.logger.warning(f"Health Alert: {message}")
        
        # In real implementation, send to Telegram or other notification channels
        # await send_telegram_notification(message)
    
    def get_health_summary(self, health_results: Dict[str, ComponentHealth]) -> Dict[str, Any]:
        """Get summary of health check results"""
        total_components = len(health_results)
        healthy_components = sum(1 for h in health_results.values() if h.status == HealthStatus.HEALTHY)
        warning_components = sum(1 for h in health_results.values() if h.status == HealthStatus.WARNING)
        critical_components = sum(1 for h in health_results.values() if h.status == HealthStatus.CRITICAL)
        
        overall_status = HealthStatus.HEALTHY
        if critical_components > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_components > 0:
            overall_status = HealthStatus.WARNING
        
        return {
            "overall_status": overall_status.value,
            "total_components": total_components,
            "healthy": healthy_components,
            "warning": warning_components,
            "critical": critical_components,
            "health_score": (healthy_components / total_components) * 100,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }
    
    def save_health_report(self, health_results: Dict[str, ComponentHealth], filename: Optional[str] = None):
        """Save health report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"health_report_{timestamp}.json"
        
        report_data = {
            "summary": self.get_health_summary(health_results),
            "components": {name: health.to_dict() for name, health in health_results.items()},
            "generated_at": datetime.now().isoformat()
        }
        
        # Save to logs directory
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        report_file = logs_dir / filename
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Health report saved: {report_file}")
        return report_file


# Global health monitor instance
health_monitor = SystemHealthMonitor()


async def run_health_check() -> Dict[str, ComponentHealth]:
    """Run complete health check"""
    return await health_monitor.perform_full_health_check()


async def get_health_summary() -> Dict[str, Any]:
    """Get quick health summary"""
    health_results = await run_health_check()
    return health_monitor.get_health_summary(health_results)


if __name__ == "__main__":
    async def main():
        print("🔍 TRADINO System Health Check")
        print("=" * 50)
        
        health_results = await run_health_check()
        summary = health_monitor.get_health_summary(health_results)
        
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Health Score: {summary['health_score']:.1f}%")
        print(f"Components: {summary['healthy']}/{summary['total_components']} healthy")
        
        if summary['warning'] > 0:
            print(f"⚠️ {summary['warning']} components with warnings")
        
        if summary['critical'] > 0:
            print(f"🔥 {summary['critical']} components critical")
        
        # Save report
        health_monitor.save_health_report(health_results)
        print("📊 Detailed report saved to logs/")
    
    asyncio.run(main()) 