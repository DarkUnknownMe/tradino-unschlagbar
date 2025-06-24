#!/usr/bin/env python3
"""
TRADINO 12-Hour Monitoring System
=================================

Überwacht das TRADINO System für 12 Stunden
Erstellt detaillierte Logs und Fehleranalysen
Läuft parallel zum Live-Trading System
"""

import sys
import os
import time
import json
import logging
import psutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Setup
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Monitoring Logger
monitoring_logger = logging.getLogger('monitoring')
monitoring_logger.setLevel(logging.INFO)

# Create detailed log handler
log_file = project_root / 'logs' / f'12h_monitoring_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_file.parent.mkdir(exist_ok=True)

file_handler = logging.FileHandler(log_file)
console_handler = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

monitoring_logger.addHandler(file_handler)
monitoring_logger.addHandler(console_handler)

class TRADINOMonitor:
    """12-Stunden TRADINO System Monitor"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=12)
        self.running = True
        
        # Monitoring data
        self.system_stats = []
        self.trading_stats = []
        self.error_log = []
        self.performance_metrics = []
        
        # Monitoring intervals
        self.system_check_interval = 30  # seconds
        self.trading_check_interval = 60  # seconds
        self.detailed_check_interval = 300  # 5 minutes
        
        monitoring_logger.info("🚀 TRADINO 12-Hour Monitoring System Started")
        monitoring_logger.info(f"📅 Start: {self.start_time}")
        monitoring_logger.info(f"📅 End: {self.end_time}")
        
    def log_system_stats(self):
        """System Resource Monitoring"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network
            network = psutil.net_io_counters()
            
            stats = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_percent': memory.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv
            }
            
            self.system_stats.append(stats)
            
            # Log warnings
            if cpu_percent > 80:
                monitoring_logger.warning(f"⚠️ High CPU usage: {cpu_percent:.1f}%")
            if memory.percent > 85:
                monitoring_logger.warning(f"⚠️ High memory usage: {memory.percent:.1f}%")
            if (disk.used / disk.total) * 100 > 90:
                monitoring_logger.warning(f"⚠️ High disk usage: {(disk.used / disk.total) * 100:.1f}%")
                
        except Exception as e:
            monitoring_logger.error(f"❌ System stats error: {e}")
            self.error_log.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'system_stats_error',
                'error': str(e)
            })
    
    def check_trading_processes(self):
        """Check if trading processes are running"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if any(keyword in cmdline.lower() for keyword in ['tradino', 'trading', 'bitget']):
                        processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': cmdline
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            self.trading_stats.append({
                'timestamp': datetime.now().isoformat(),
                'active_processes': len(processes),
                'processes': processes
            })
            
            if len(processes) == 0:
                monitoring_logger.warning("⚠️ No TRADINO processes detected!")
            else:
                monitoring_logger.info(f"✅ {len(processes)} TRADINO processes running")
                
        except Exception as e:
            monitoring_logger.error(f"❌ Process check error: {e}")
            self.error_log.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'process_check_error',
                'error': str(e)
            })
    
    def monitor_log_files(self):
        """Monitor TRADINO log files for errors"""
        try:
            log_dirs = [
                project_root / 'logs',
                project_root / 'tradino_unschlagbar' / 'logs',
                project_root / 'data' / 'logs'
            ]
            
            error_patterns = ['ERROR', 'CRITICAL', 'EXCEPTION', 'FAILED', 'TIMEOUT']
            
            for log_dir in log_dirs:
                if log_dir.exists():
                    for log_file in log_dir.glob('*.log'):
                        if log_file.stat().st_mtime > time.time() - 300:  # Last 5 minutes
                            try:
                                with open(log_file, 'r') as f:
                                    lines = f.readlines()
                                    recent_lines = lines[-50:]  # Last 50 lines
                                    
                                    for line in recent_lines:
                                        if any(pattern in line.upper() for pattern in error_patterns):
                                            monitoring_logger.warning(f"🚨 Error in {log_file.name}: {line.strip()}")
                                            self.error_log.append({
                                                'timestamp': datetime.now().isoformat(),
                                                'type': 'log_error',
                                                'file': str(log_file),
                                                'line': line.strip()
                                            })
                            except Exception as e:
                                pass  # Skip unreadable files
                                
        except Exception as e:
            monitoring_logger.error(f"❌ Log monitoring error: {e}")
    
    def check_trading_performance(self):
        """Check trading performance metrics"""
        try:
            # Check balance files
            balance_files = list((project_root / 'data').rglob('*balance*.json'))
            
            latest_balance = None
            for balance_file in balance_files:
                if balance_file.stat().st_mtime > time.time() - 3600:  # Last hour
                    try:
                        with open(balance_file, 'r') as f:
                            balance_data = json.load(f)
                            latest_balance = balance_data
                            break
                    except:
                        pass
            
            # Check trade files
            trade_files = list((project_root / 'data').rglob('*trade*.json'))
            recent_trades = 0
            
            for trade_file in trade_files:
                if trade_file.stat().st_mtime > time.time() - 3600:  # Last hour
                    recent_trades += 1
            
            performance = {
                'timestamp': datetime.now().isoformat(),
                'latest_balance': latest_balance,
                'recent_trades': recent_trades,
                'balance_updated': latest_balance is not None
            }
            
            self.performance_metrics.append(performance)
            
            if latest_balance:
                monitoring_logger.info(f"💰 Latest balance check: Available")
            else:
                monitoring_logger.warning("⚠️ No recent balance data")
                
            monitoring_logger.info(f"📊 Recent trades: {recent_trades}")
            
        except Exception as e:
            monitoring_logger.error(f"❌ Performance check error: {e}")
    
    def detailed_system_check(self):
        """Detailed system check every 5 minutes"""
        try:
            monitoring_logger.info("🔍 Performing detailed system check...")
            
            # Check file system
            important_files = [
                'main.py',
                'tradino_unschlagbar/brain/master_ai.py',
                'tradino_unschlagbar/core/trading_engine.py',
                'models/lightgbm_volatility.pkl'
            ]
            
            missing_files = []
            for file_path in important_files:
                if not (project_root / file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                monitoring_logger.error(f"🚨 Missing critical files: {missing_files}")
            else:
                monitoring_logger.info("✅ All critical files present")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            
            if free_gb < 1:
                monitoring_logger.critical(f"🚨 CRITICAL: Low disk space: {free_gb:.1f}GB")
            elif free_gb < 5:
                monitoring_logger.warning(f"⚠️ Low disk space: {free_gb:.1f}GB")
            else:
                monitoring_logger.info(f"💾 Disk space OK: {free_gb:.1f}GB free")
            
            # Memory check
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                monitoring_logger.critical(f"🚨 CRITICAL: Memory usage: {memory.percent:.1f}%")
            
        except Exception as e:
            monitoring_logger.error(f"❌ Detailed check error: {e}")
    
    def generate_hourly_report(self):
        """Generate hourly summary report"""
        try:
            current_time = datetime.now()
            hours_running = (current_time - self.start_time).total_seconds() / 3600
            
            # System stats summary
            if self.system_stats:
                recent_stats = self.system_stats[-12:]  # Last 12 measurements
                avg_cpu = sum(s['cpu_percent'] for s in recent_stats) / len(recent_stats)
                avg_memory = sum(s['memory_percent'] for s in recent_stats) / len(recent_stats)
                
                monitoring_logger.info(f"📊 Hour {hours_running:.1f} Summary:")
                monitoring_logger.info(f"   CPU Average: {avg_cpu:.1f}%")
                monitoring_logger.info(f"   Memory Average: {avg_memory:.1f}%")
                monitoring_logger.info(f"   Errors Logged: {len(self.error_log)}")
                monitoring_logger.info(f"   System Checks: {len(self.system_stats)}")
            
        except Exception as e:
            monitoring_logger.error(f"❌ Report generation error: {e}")
    
    def save_monitoring_data(self):
        """Save all monitoring data to files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save comprehensive data
            monitoring_data = {
                'start_time': self.start_time.isoformat(),
                'current_time': datetime.now().isoformat(),
                'system_stats': self.system_stats,
                'trading_stats': self.trading_stats,
                'performance_metrics': self.performance_metrics,
                'error_log': self.error_log
            }
            
            data_file = project_root / 'data' / f'monitoring_data_{timestamp}.json'
            data_file.parent.mkdir(exist_ok=True)
            
            with open(data_file, 'w') as f:
                json.dump(monitoring_data, f, indent=2)
            
            monitoring_logger.info(f"💾 Monitoring data saved: {data_file}")
            
        except Exception as e:
            monitoring_logger.error(f"❌ Data save error: {e}")
    
    def run_monitoring_loop(self):
        """Main monitoring loop"""
        last_system_check = 0
        last_trading_check = 0
        last_detailed_check = 0
        last_hourly_report = 0
        last_data_save = 0
        
        while self.running and datetime.now() < self.end_time:
            try:
                current_time = time.time()
                
                # System checks (every 30s)
                if current_time - last_system_check >= self.system_check_interval:
                    self.log_system_stats()
                    last_system_check = current_time
                
                # Trading checks (every 60s)
                if current_time - last_trading_check >= self.trading_check_interval:
                    self.check_trading_processes()
                    self.monitor_log_files()
                    last_trading_check = current_time
                
                # Detailed checks (every 5 minutes)
                if current_time - last_detailed_check >= self.detailed_check_interval:
                    self.detailed_system_check()
                    self.check_trading_performance()
                    last_detailed_check = current_time
                
                # Hourly reports (every hour)
                if current_time - last_hourly_report >= 3600:
                    self.generate_hourly_report()
                    last_hourly_report = current_time
                
                # Save data (every 10 minutes)
                if current_time - last_data_save >= 600:
                    self.save_monitoring_data()
                    last_data_save = current_time
                
                time.sleep(10)  # Main loop sleep
                
            except KeyboardInterrupt:
                monitoring_logger.info("🛑 Monitoring stopped by user")
                self.running = False
                break
            except Exception as e:
                monitoring_logger.error(f"❌ Monitoring loop error: {e}")
                time.sleep(30)  # Wait before retrying
        
        # Final save
        self.save_monitoring_data()
        
        # Final report
        total_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        monitoring_logger.info(f"🏁 Monitoring completed after {total_hours:.1f} hours")
        monitoring_logger.info(f"📊 Total system checks: {len(self.system_stats)}")
        monitoring_logger.info(f"🚨 Total errors logged: {len(self.error_log)}")

def main():
    """Start the 12-hour monitoring system"""
    print("🚀 TRADINO 12-Hour Monitoring System")
    print("=" * 50)
    
    monitor = TRADINOMonitor()
    
    try:
        monitor.run_monitoring_loop()
    except Exception as e:
        monitoring_logger.critical(f"🚨 CRITICAL: Monitoring system failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 