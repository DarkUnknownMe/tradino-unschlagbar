#!/usr/bin/env python3
"""
🐺 WOT ALPHA - 30 DAY CONTINUOUS BACKTEST
The Ultimate Wolf Hunt - 720 Hours of Non-Stop Trading

Author: WOT Alpha System
"""

import asyncio
import sys
import signal
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import psutil
import traceback
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.config_manager import ConfigManager
from utils.logger_pro import setup_logger
from core.trading_engine import TradingEngine
from analytics.performance_tracker import PerformanceTracker

logger = setup_logger("Alpha30DayBacktest")


class Alpha30DayBacktest:
    """🐺 30-Day Continuous Trading Backtest System"""
    
    def __init__(self, capital: float = 10000, duration_days: int = 30):
        self.start_time = datetime.utcnow()
        self.duration_days = duration_days
        self.end_time = self.start_time + timedelta(days=duration_days)
        self.capital = capital
        
        # System Components
        self.config: Optional[ConfigManager] = None
        self.trading_engine: Optional[TradingEngine] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        
        # Backtest State
        self.is_running = False
        self.current_phase = 1
        self.phase_start_time = self.start_time
        
        # Performance Metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.current_portfolio_value = capital
        
        # System Health
        self.uptime_start = datetime.utcnow()
        self.downtime_total = 0.0
        self.error_count = 0
        self.last_health_check = datetime.utcnow()
        
        # Monitoring
        self.reports_dir = Path("backtest_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Phase Definitions
        self.phases = {
            1: {"name": "WARM-UP", "days": 5, "risk_level": 0.02},
            2: {"name": "ACCELERATION", "days": 10, "risk_level": 0.03},
            3: {"name": "STRESS TEST", "days": 10, "risk_level": 0.04},
            4: {"name": "OPTIMIZATION", "days": 5, "risk_level": "optimal"}
        }
        
    async def initialize(self) -> bool:
        """🔧 Initialize backtest system"""
        try:
            logger.info("🐺 ALPHA 30-DAY BACKTEST INITIALIZATION")
            logger.info("=" * 50)
            
            # Load Configuration
            self.config = ConfigManager()
            
            # Override config for backtest
            self.config.set('trading.initial_capital', self.capital)
            self.config.set('system.environment', 'demo')
            self.config.set('exchange.sandbox', True)
            
            logger.info(f"💰 Start Capital: ${self.capital:,.2f} USDT")
            logger.info(f"⏰ Duration: {self.duration_days} days")
            logger.info(f"🎯 Target End Time: {self.end_time}")
            
            # Initialize Trading Engine
            self.trading_engine = TradingEngine(self.config)
            if not await self.trading_engine.initialize():
                raise Exception("Trading Engine initialization failed")
            
            # Initialize Performance Tracker
            self.performance_tracker = PerformanceTracker(self.config)
            await self.performance_tracker.initialize()
            
            # Create initial report
            await self._create_initial_report()
            
            logger.success("✅ Alpha 30-Day Backtest System initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    async def start_backtest(self):
        """🚀 Start the 30-day continuous backtest"""
        try:
            logger.info("🚀 STARTING 30-DAY CONTINUOUS BACKTEST")
            logger.info("=" * 50)
            
            self.is_running = True
            self.uptime_start = datetime.utcnow()
            
            # Start Trading Engine
            if not await self.trading_engine.start():
                raise Exception("Failed to start trading engine")
            
            # Start monitoring tasks
            monitoring_tasks = [
                asyncio.create_task(self._health_monitor()),
                asyncio.create_task(self._performance_monitor()),
                asyncio.create_task(self._phase_manager()),
                asyncio.create_task(self._report_generator()),
                asyncio.create_task(self._emergency_monitor())
            ]
            
            logger.success("🎯 Alpha is hunting! 30-day backtest started")
            logger.info(f"🕐 Start Time: {self.start_time}")
            logger.info(f"🎯 End Time: {self.end_time}")
            
            # Main backtest loop
            while self.is_running and datetime.utcnow() < self.end_time:
                try:
                    # Update metrics
                    await self._update_metrics()
                    
                    # Check system health
                    if not await self._system_health_check():
                        logger.warning("⚠️ System health check failed - attempting recovery")
                        await self._attempt_recovery()
                    
                    # Sleep for monitoring interval
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"❌ Error in main loop: {e}")
                    self.error_count += 1
                    await asyncio.sleep(5)
            
            # Backtest completed
            await self._complete_backtest()
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            await self._emergency_shutdown()
        finally:
            # Cleanup
            for task in monitoring_tasks:
                if not task.done():
                    task.cancel()
    
    async def _health_monitor(self):
        """🏥 Continuous system health monitoring"""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                # System Resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Log health metrics
                if current_time.minute % 5 == 0:  # Every 5 minutes
                    logger.info(f"🏥 Health Check - CPU: {cpu_percent:.1f}% | "
                              f"RAM: {memory.percent:.1f}% | "
                              f"Disk: {disk.percent:.1f}%")
                
                # Alert if resources high
                if cpu_percent > 90:
                    logger.warning(f"⚠️ High CPU usage: {cpu_percent:.1f}%")
                if memory.percent > 90:
                    logger.warning(f"⚠️ High memory usage: {memory.percent:.1f}%")
                if disk.percent > 90:
                    logger.warning(f"⚠️ High disk usage: {disk.percent:.1f}%")
                
                self.last_health_check = current_time
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Health monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _performance_monitor(self):
        """📊 Continuous performance monitoring"""
        while self.is_running:
            try:
                # Get current portfolio value
                if self.trading_engine and self.trading_engine.portfolio_manager:
                    portfolio = await self.trading_engine.portfolio_manager.get_portfolio()
                    if portfolio:
                        self.current_portfolio_value = float(portfolio.total_balance)
                
                # Calculate metrics
                current_pnl = self.current_portfolio_value - self.capital
                current_pnl_percent = (current_pnl / self.capital) * 100
                
                # Update max drawdown
                peak_value = max(self.current_portfolio_value, self.capital)
                current_drawdown = ((peak_value - self.current_portfolio_value) / peak_value) * 100
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
                
                # Log performance every hour
                if datetime.utcnow().minute == 0:
                    elapsed_days = (datetime.utcnow() - self.start_time).days
                    logger.info(f"📊 Day {elapsed_days} Performance - "
                              f"Portfolio: ${self.current_portfolio_value:,.2f} | "
                              f"P&L: {current_pnl:+.2f} ({current_pnl_percent:+.1f}%) | "
                              f"Drawdown: {current_drawdown:.1f}%")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Performance monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _phase_manager(self):
        """🎯 Manage backtest phases"""
        while self.is_running:
            try:
                elapsed_days = (datetime.utcnow() - self.start_time).days
                
                # Determine current phase
                new_phase = 1
                cumulative_days = 0
                for phase_num, phase_info in self.phases.items():
                    cumulative_days += phase_info["days"]
                    if elapsed_days < cumulative_days:
                        new_phase = phase_num
                        break
                    new_phase = phase_num + 1
                
                # Phase transition
                if new_phase != self.current_phase and new_phase <= 4:
                    await self._transition_to_phase(new_phase)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"❌ Phase manager error: {e}")
                await asyncio.sleep(300)
    
    async def _transition_to_phase(self, new_phase: int):
        """🔄 Transition to new backtest phase"""
        try:
            old_phase = self.current_phase
            self.current_phase = new_phase
            self.phase_start_time = datetime.utcnow()
            
            phase_info = self.phases[new_phase]
            
            logger.info(f"🔄 PHASE TRANSITION: {old_phase} → {new_phase}")
            logger.info(f"🎯 New Phase: {phase_info['name']}")
            logger.info(f"🛡️ Risk Level: {phase_info['risk_level']}")
            
            # Update risk settings
            if isinstance(phase_info['risk_level'], float):
                self.config.set('trading.risk_per_trade', phase_info['risk_level'])
            
            # Create phase transition report
            await self._create_phase_report(old_phase, new_phase)
            
        except Exception as e:
            logger.error(f"❌ Phase transition error: {e}")
    
    async def _report_generator(self):
        """📋 Generate periodic reports"""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                # Daily report at midnight UTC
                if current_time.hour == 0 and current_time.minute == 0:
                    await self._create_daily_report()
                
                # Weekly report on Sundays
                if current_time.weekday() == 6 and current_time.hour == 0:
                    await self._create_weekly_report()
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"❌ Report generator error: {e}")
                await asyncio.sleep(300)
    
    async def _emergency_monitor(self):
        """🚨 Emergency monitoring and auto-stop"""
        while self.is_running:
            try:
                # Check for emergency conditions
                current_drawdown = self._calculate_current_drawdown()
                
                # Emergency stop conditions
                if current_drawdown > 15:  # 15% drawdown
                    logger.critical(f"🚨 EMERGENCY STOP: Drawdown {current_drawdown:.1f}% > 15%")
                    await self._emergency_stop()
                    break
                
                if self.error_count > 100:  # Too many errors
                    logger.critical(f"🚨 EMERGENCY STOP: Error count {self.error_count} > 100")
                    await self._emergency_stop()
                    break
                
                # Check system responsiveness
                time_since_health_check = (datetime.utcnow() - self.last_health_check).seconds
                if time_since_health_check > 300:  # 5 minutes
                    logger.critical("🚨 EMERGENCY STOP: System unresponsive")
                    await self._emergency_stop()
                    break
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Emergency monitor error: {e}")
                await asyncio.sleep(30)
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current portfolio drawdown"""
        peak_value = max(self.current_portfolio_value, self.capital)
        return ((peak_value - self.current_portfolio_value) / peak_value) * 100
    
    async def _system_health_check(self) -> bool:
        """Check if system is healthy"""
        try:
            # Check trading engine
            if not self.trading_engine or self.trading_engine.state.value != "running":
                return False
            
            # Check resources
            if psutil.cpu_percent() > 95:
                return False
            if psutil.virtual_memory().percent > 95:
                return False
            if psutil.disk_usage('/').percent > 95:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _attempt_recovery(self):
        """Attempt system recovery"""
        try:
            logger.info("🔄 Attempting system recovery...")
            
            # Restart trading engine if needed
            if self.trading_engine.state.value != "running":
                await self.trading_engine.stop()
                await asyncio.sleep(5)
                await self.trading_engine.start()
            
            logger.info("✅ Recovery attempt completed")
            
        except Exception as e:
            logger.error(f"❌ Recovery failed: {e}")
    
    async def _update_metrics(self):
        """Update performance metrics"""
        try:
            if self.performance_tracker:
                metrics = await self.performance_tracker.get_daily_summary()
                if metrics:
                    self.total_trades = metrics.get('total_trades', 0)
                    self.winning_trades = metrics.get('winning_trades', 0)
                    self.total_pnl = metrics.get('total_pnl', 0.0)
                    
        except Exception as e:
            logger.error(f"❌ Metrics update error: {e}")
    
    async def _create_initial_report(self):
        """Create initial backtest report"""
        report = {
            "backtest_id": f"alpha_30day_{int(self.start_time.timestamp())}",
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_days": self.duration_days,
            "initial_capital": self.capital,
            "phases": self.phases,
            "status": "initialized"
        }
        
        report_file = self.reports_dir / "initial_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 Initial report created: {report_file}")
    
    async def _create_daily_report(self):
        """Create daily performance report"""
        try:
            current_time = datetime.utcnow()
            elapsed_days = (current_time - self.start_time).days
            
            report = {
                "date": current_time.date().isoformat(),
                "day": elapsed_days,
                "phase": self.current_phase,
                "portfolio_value": self.current_portfolio_value,
                "total_pnl": self.current_portfolio_value - self.capital,
                "pnl_percent": ((self.current_portfolio_value - self.capital) / self.capital) * 100,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "win_rate": (self.winning_trades / max(self.total_trades, 1)) * 100,
                "max_drawdown": self.max_drawdown,
                "error_count": self.error_count,
                "uptime_hours": (current_time - self.uptime_start).total_seconds() / 3600
            }
            
            report_file = self.reports_dir / f"daily_report_{current_time.date()}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📋 Daily report created: Day {elapsed_days}")
            
        except Exception as e:
            logger.error(f"❌ Daily report error: {e}")
    
    async def _create_weekly_report(self):
        """Create weekly analysis report"""
        logger.info("📊 Creating weekly analysis report...")
        # Implementation for detailed weekly analysis
    
    async def _create_phase_report(self, old_phase: int, new_phase: int):
        """Create phase transition report"""
        logger.info(f"🔄 Creating phase transition report: {old_phase} → {new_phase}")
        # Implementation for phase transition analysis
    
    async def _emergency_stop(self):
        """Emergency stop procedure"""
        try:
            logger.critical("🚨 EXECUTING EMERGENCY STOP")
            
            self.is_running = False
            
            # Stop trading engine
            if self.trading_engine:
                await self.trading_engine.stop()
            
            # Create emergency report
            await self._create_emergency_report()
            
            logger.critical("🛑 Emergency stop completed")
            
        except Exception as e:
            logger.error(f"❌ Emergency stop error: {e}")
    
    async def _create_emergency_report(self):
        """Create emergency stop report"""
        report = {
            "emergency_stop_time": datetime.utcnow().isoformat(),
            "reason": "Emergency conditions detected",
            "portfolio_value": self.current_portfolio_value,
            "total_pnl": self.current_portfolio_value - self.capital,
            "max_drawdown": self.max_drawdown,
            "error_count": self.error_count,
            "phase": self.current_phase
        }
        
        report_file = self.reports_dir / "emergency_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    async def _complete_backtest(self):
        """Complete the backtest and generate final report"""
        try:
            logger.info("🎯 30-DAY BACKTEST COMPLETED!")
            logger.info("=" * 50)
            
            # Stop trading engine
            if self.trading_engine:
                await self.trading_engine.stop()
            
            # Calculate final metrics
            final_pnl = self.current_portfolio_value - self.capital
            final_pnl_percent = (final_pnl / self.capital) * 100
            total_runtime = (datetime.utcnow() - self.start_time).total_seconds() / 3600
            win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
            
            # Create final report
            final_report = {
                "backtest_completed": datetime.utcnow().isoformat(),
                "duration_hours": total_runtime,
                "initial_capital": self.capital,
                "final_portfolio_value": self.current_portfolio_value,
                "total_pnl": final_pnl,
                "pnl_percent": final_pnl_percent,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "win_rate": win_rate,
                "max_drawdown": self.max_drawdown,
                "error_count": self.error_count,
                "phases_completed": 4
            }
            
            report_file = self.reports_dir / "ALPHA_30_DAY_FINAL_REPORT.json"
            with open(report_file, 'w') as f:
                json.dump(final_report, f, indent=2)
            
            # Log final results
            logger.success("🏆 FINAL RESULTS:")
            logger.success(f"💰 Start Capital: ${self.capital:,.2f}")
            logger.success(f"💼 Final Portfolio: ${self.current_portfolio_value:,.2f}")
            logger.success(f"📈 Total P&L: {final_pnl:+,.2f} ({final_pnl_percent:+.1f}%)")
            logger.success(f"📊 Total Trades: {self.total_trades}")
            logger.success(f"🎯 Win Rate: {win_rate:.1f}%")
            logger.success(f"📉 Max Drawdown: {self.max_drawdown:.1f}%")
            logger.success(f"⏰ Runtime: {total_runtime:.1f} hours")
            
            # Determine success level
            if final_pnl_percent >= 75 and win_rate >= 70 and self.max_drawdown < 5:
                logger.success("👑 LEGENDARY STATUS ACHIEVED!")
            elif final_pnl_percent >= 50 and win_rate >= 65 and self.max_drawdown < 8:
                logger.success("🏆 EXCELLENT PERFORMANCE!")
            elif final_pnl_percent >= 25 and win_rate >= 58 and self.max_drawdown < 12:
                logger.success("✅ MINIMUM REQUIREMENTS MET!")
            else:
                logger.warning("⚠️ Performance below expectations")
            
            self.is_running = False
            
        except Exception as e:
            logger.error(f"❌ Backtest completion error: {e}")
    
    async def _emergency_shutdown(self):
        """Emergency shutdown procedure"""
        logger.critical("🚨 EMERGENCY SHUTDOWN")
        self.is_running = False
        
        if self.trading_engine:
            await self.trading_engine.shutdown()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"🛑 Received signal {signum} - initiating shutdown")
    # Set global flag for graceful shutdown
    global shutdown_requested
    shutdown_requested = True


async def main():
    """Main entry point"""
    global shutdown_requested
    shutdown_requested = False
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Alpha 30-Day Backtest")
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital")
    parser.add_argument("--duration", type=int, default=30, help="Duration in days")
    parser.add_argument("--strategies", default="all", help="Strategies to use")
    parser.add_argument("--risk-level", default="standard", help="Risk level")
    parser.add_argument("--monitoring", default="enabled", help="Enable monitoring")
    parser.add_argument("--alerts", default="enabled", help="Enable alerts")
    
    args = parser.parse_args()
    
    try:
        logger.info("🐺 WOT ALPHA - 30 DAY BACKTEST STARTING")
        logger.info("=" * 50)
        
        # Create backtest instance
        backtest = Alpha30DayBacktest(
            capital=args.capital,
            duration_days=args.duration
        )
        
        # Initialize
        if not await backtest.initialize():
            logger.error("❌ Failed to initialize backtest")
            return 1
        
        # Start backtest
        backtest_task = asyncio.create_task(backtest.start_backtest())
        
        # Wait for completion or shutdown signal
        while not shutdown_requested and not backtest_task.done():
            await asyncio.sleep(1)
        
        if shutdown_requested:
            logger.info("🛑 Shutdown requested - stopping backtest")
            backtest.is_running = False
            await backtest_task
        
        logger.info("✅ Backtest completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    shutdown_requested = False
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 