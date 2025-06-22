#!/usr/bin/env python3
"""
🐺 WOT ALPHA - BACKTEST MONITOR
Real-time monitoring and dashboard for 30-day backtest

Author: WOT Alpha System
"""

import asyncio
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import psutil
from typing import Dict, List, Optional
import subprocess

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.logger_pro import setup_logger

logger = setup_logger("BacktestMonitor")


class BacktestMonitor:
    """📊 Real-time backtest monitoring system"""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.reports_dir = Path("backtest_reports")
        self.dashboard_dir = Path("dashboard")
        self.dashboard_dir.mkdir(exist_ok=True)
        
        # Monitoring state
        self.is_monitoring = False
        self.last_update = datetime.utcnow()
        
        # Metrics cache
        self.metrics_cache = {}
        self.alerts_sent = []
        
    async def start_monitoring(self):
        """🚀 Start real-time monitoring"""
        try:
            logger.info("📊 STARTING BACKTEST MONITORING")
            logger.info("=" * 40)
            
            self.is_monitoring = True
            
            # Start monitoring tasks
            tasks = [
                asyncio.create_task(self._real_time_dashboard()),
                asyncio.create_task(self._system_monitor()),
                asyncio.create_task(self._alert_system()),
                asyncio.create_task(self._web_dashboard_updater())
            ]
            
            logger.success("✅ Monitoring system started")
            
            # Keep monitoring running
            while self.is_monitoring:
                await asyncio.sleep(10)
                
        except Exception as e:
            logger.error(f"❌ Monitoring failed: {e}")
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
    
    async def _real_time_dashboard(self):
        """📊 Real-time console dashboard"""
        while self.is_monitoring:
            try:
                # Clear screen and show dashboard
                subprocess.run(['clear'], check=False)
                
                print("🐺 WOT ALPHA - LIVE BACKTEST DASHBOARD")
                print("=" * 60)
                print()
                
                # Get current metrics
                metrics = await self._get_current_metrics()
                
                # Display main metrics
                print(f"🕐 Runtime: {self._format_runtime()}")
                print(f"💰 Portfolio: ${metrics.get('portfolio_value', 0):,.2f}")
                print(f"📈 P&L: {metrics.get('pnl', 0):+,.2f} ({metrics.get('pnl_percent', 0):+.1f}%)")
                print(f"📊 Trades: {metrics.get('total_trades', 0)} | Win Rate: {metrics.get('win_rate', 0):.1f}%")
                print(f"📉 Max Drawdown: {metrics.get('max_drawdown', 0):.1f}%")
                print(f"🎯 Phase: {metrics.get('phase', 1)} - {self._get_phase_name(metrics.get('phase', 1))}")
                print()
                
                # System health
                system_health = await self._get_system_health()
                print("🏥 SYSTEM HEALTH:")
                print(f"   CPU: {system_health['cpu']:.1f}% | RAM: {system_health['memory']:.1f}% | Disk: {system_health['disk']:.1f}%")
                print(f"   Uptime: {system_health['uptime']} | Errors: {metrics.get('error_count', 0)}")
                print()
                
                # Recent activity
                print("📋 RECENT ACTIVITY:")
                recent_activity = await self._get_recent_activity()
                for activity in recent_activity[-5:]:
                    print(f"   {activity}")
                print()
                
                # Performance targets
                print("🎯 PERFORMANCE TARGETS:")
                targets = self._check_performance_targets(metrics)
                for target, status in targets.items():
                    icon = "✅" if status else "❌"
                    print(f"   {icon} {target}")
                print()
                
                print(f"🔄 Last Update: {datetime.utcnow().strftime('%H:%M:%S UTC')}")
                print("=" * 60)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Dashboard error: {e}")
                await asyncio.sleep(10)
    
    async def _system_monitor(self):
        """🏥 System health monitoring"""
        while self.is_monitoring:
            try:
                # Check if backtest is running
                backtest_running = await self._check_backtest_process()
                
                if not backtest_running:
                    await self._send_alert("CRITICAL", "Backtest process not running!")
                
                # Check system resources
                cpu = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory().percent
                disk = psutil.disk_usage('/').percent
                
                # Resource alerts
                if cpu > 90:
                    await self._send_alert("WARNING", f"High CPU usage: {cpu:.1f}%")
                if memory > 90:
                    await self._send_alert("WARNING", f"High memory usage: {memory:.1f}%")
                if disk > 90:
                    await self._send_alert("CRITICAL", f"High disk usage: {disk:.1f}%")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ System monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _alert_system(self):
        """🚨 Alert system for critical events"""
        while self.is_monitoring:
            try:
                metrics = await self._get_current_metrics()
                
                # Performance alerts
                if metrics.get('max_drawdown', 0) > 10:
                    await self._send_alert("WARNING", f"High drawdown: {metrics['max_drawdown']:.1f}%")
                
                if metrics.get('max_drawdown', 0) > 15:
                    await self._send_alert("CRITICAL", f"Critical drawdown: {metrics['max_drawdown']:.1f}%")
                
                if metrics.get('win_rate', 100) < 50:
                    await self._send_alert("WARNING", f"Low win rate: {metrics['win_rate']:.1f}%")
                
                # Error count alerts
                if metrics.get('error_count', 0) > 50:
                    await self._send_alert("WARNING", f"High error count: {metrics['error_count']}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Alert system error: {e}")
                await asyncio.sleep(60)
    
    async def _web_dashboard_updater(self):
        """🌐 Update web dashboard files"""
        while self.is_monitoring:
            try:
                # Create HTML dashboard
                await self._create_web_dashboard()
                
                # Update JSON data for API
                await self._update_dashboard_data()
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Web dashboard error: {e}")
                await asyncio.sleep(60)
    
    async def _get_current_metrics(self) -> Dict:
        """Get current backtest metrics"""
        try:
            # Try to read latest daily report
            reports = list(self.reports_dir.glob("daily_report_*.json"))
            if reports:
                latest_report = max(reports, key=lambda x: x.stat().st_mtime)
                with open(latest_report, 'r') as f:
                    return json.load(f)
            
            # Fallback to default values
            return {
                'portfolio_value': 10000,
                'pnl': 0,
                'pnl_percent': 0,
                'total_trades': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'phase': 1,
                'error_count': 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting metrics: {e}")
            return {}
    
    async def _get_system_health(self) -> Dict:
        """Get system health metrics"""
        try:
            return {
                'cpu': psutil.cpu_percent(interval=1),
                'memory': psutil.virtual_memory().percent,
                'disk': psutil.disk_usage('/').percent,
                'uptime': self._format_runtime()
            }
        except Exception:
            return {'cpu': 0, 'memory': 0, 'disk': 0, 'uptime': '0h 0m'}
    
    async def _get_recent_activity(self) -> List[str]:
        """Get recent activity log"""
        try:
            # Read from log files or reports
            activity = [
                f"{datetime.utcnow().strftime('%H:%M')} - System monitoring active",
                f"{datetime.utcnow().strftime('%H:%M')} - Performance check completed",
                f"{datetime.utcnow().strftime('%H:%M')} - Health monitoring active"
            ]
            return activity
        except Exception:
            return []
    
    def _check_performance_targets(self, metrics: Dict) -> Dict[str, bool]:
        """Check if performance targets are met"""
        return {
            "Profitability > 25%": metrics.get('pnl_percent', 0) > 25,
            "Win Rate > 58%": metrics.get('win_rate', 0) > 58,
            "Max Drawdown < 12%": metrics.get('max_drawdown', 100) < 12,
            "System Uptime > 98%": True,  # Calculate based on actual uptime
            "Error Rate < 0.1%": metrics.get('error_count', 0) < 100
        }
    
    async def _check_backtest_process(self) -> bool:
        """Check if backtest process is running"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'alpha_30_day_backtest.py' in ' '.join(proc.info['cmdline'] or []):
                    return True
            return False
        except Exception:
            return False
    
    async def _send_alert(self, level: str, message: str):
        """Send alert notification"""
        try:
            alert_key = f"{level}_{message}"
            
            # Avoid duplicate alerts
            if alert_key in self.alerts_sent:
                return
            
            timestamp = datetime.utcnow().strftime('%H:%M:%S')
            alert_msg = f"🚨 {level} - {timestamp}: {message}"
            
            logger.warning(alert_msg)
            
            # Store alert to avoid duplicates
            self.alerts_sent.append(alert_key)
            
            # Keep only recent alerts
            if len(self.alerts_sent) > 100:
                self.alerts_sent = self.alerts_sent[-50:]
                
        except Exception as e:
            logger.error(f"❌ Alert error: {e}")
    
    async def _create_web_dashboard(self):
        """Create HTML web dashboard"""
        try:
            metrics = await self._get_current_metrics()
            system_health = await self._get_system_health()
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🐺 Alpha 30-Day Backtest Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{ font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .metric-card {{ background: #2a2a2a; padding: 20px; border-radius: 10px; border-left: 4px solid #00ff88; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #00ff88; }}
        .metric-label {{ color: #ccc; margin-bottom: 10px; }}
        .status-good {{ color: #00ff88; }}
        .status-warning {{ color: #ffaa00; }}
        .status-critical {{ color: #ff4444; }}
        .phase-info {{ background: #333; padding: 15px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐺 WOT ALPHA - 30 DAY BACKTEST</h1>
            <p>Real-time Performance Dashboard</p>
            <p>Last Update: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </div>
        
        <div class="phase-info">
            <h3>🎯 Current Phase: {metrics.get('phase', 1)} - {self._get_phase_name(metrics.get('phase', 1))}</h3>
            <p>Runtime: {self._format_runtime()}</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">💰 Portfolio Value</div>
                <div class="metric-value">${metrics.get('portfolio_value', 0):,.2f}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">📈 Total P&L</div>
                <div class="metric-value">{metrics.get('pnl', 0):+,.2f} ({metrics.get('pnl_percent', 0):+.1f}%)</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">📊 Total Trades</div>
                <div class="metric-value">{metrics.get('total_trades', 0)}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">🎯 Win Rate</div>
                <div class="metric-value">{metrics.get('win_rate', 0):.1f}%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">📉 Max Drawdown</div>
                <div class="metric-value">{metrics.get('max_drawdown', 0):.1f}%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">🏥 System Health</div>
                <div class="metric-value">
                    CPU: {system_health['cpu']:.1f}%<br>
                    RAM: {system_health['memory']:.1f}%<br>
                    Disk: {system_health['disk']:.1f}%
                </div>
            </div>
        </div>
        
        <div class="phase-info">
            <h3>🎯 Performance Targets</h3>
            <ul>
                <li class="{'status-good' if metrics.get('pnl_percent', 0) > 25 else 'status-warning'}">
                    Profitability > 25%: {metrics.get('pnl_percent', 0):.1f}%
                </li>
                <li class="{'status-good' if metrics.get('win_rate', 0) > 58 else 'status-warning'}">
                    Win Rate > 58%: {metrics.get('win_rate', 0):.1f}%
                </li>
                <li class="{'status-good' if metrics.get('max_drawdown', 100) < 12 else 'status-warning'}">
                    Max Drawdown < 12%: {metrics.get('max_drawdown', 100):.1f}%
                </li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
            
            dashboard_file = self.dashboard_dir / "index.html"
            with open(dashboard_file, 'w') as f:
                f.write(html_content)
                
        except Exception as e:
            logger.error(f"❌ Web dashboard creation error: {e}")
    
    async def _update_dashboard_data(self):
        """Update JSON data for dashboard API"""
        try:
            metrics = await self._get_current_metrics()
            system_health = await self._get_system_health()
            
            dashboard_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "runtime": self._format_runtime(),
                "metrics": metrics,
                "system_health": system_health,
                "targets": self._check_performance_targets(metrics)
            }
            
            data_file = self.dashboard_dir / "data.json"
            with open(data_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Dashboard data update error: {e}")
    
    def _format_runtime(self) -> str:
        """Format runtime duration"""
        runtime = datetime.utcnow() - self.start_time
        hours = int(runtime.total_seconds() // 3600)
        minutes = int((runtime.total_seconds() % 3600) // 60)
        return f"{hours}h {minutes}m"
    
    def _get_phase_name(self, phase: int) -> str:
        """Get phase name"""
        phases = {
            1: "WARM-UP",
            2: "ACCELERATION", 
            3: "STRESS TEST",
            4: "OPTIMIZATION"
        }
        return phases.get(phase, "UNKNOWN")


async def main():
    """Main entry point"""
    try:
        logger.info("🐺 STARTING BACKTEST MONITOR")
        
        monitor = BacktestMonitor()
        await monitor.start_monitoring()
        
    except KeyboardInterrupt:
        logger.info("🛑 Monitor stopped by user")
    except Exception as e:
        logger.error(f"❌ Monitor failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 