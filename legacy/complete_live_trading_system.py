#!/usr/bin/env python3
"""
🚀 COMPLETE LIVE TRADING SYSTEM
Vollständiges Live Trading System für TRADINO UNSCHLAGBAR
Integriert AI, Marktdaten, Trading, Risk Management und Performance Monitoring
"""

import os
import sys
import time
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.append('/root/tradino')

# Import all components
try:
    from live_trading_engine import LiveTradingEngine, initialize_live_trading
    from bitget_trading_api import BitgetTradingAPI, initialize_bitget_trading
    from risk_management_system import RiskManagementSystem, initialize_risk_management
    from performance_monitoring_system import PerformanceMonitoringSystem, initialize_performance_monitoring
    from tradino_unschlagbar.connectors.live_market_feed import LiveMarketFeed
    from tradino_unschlagbar.brain.trained_model_integration import TrainedModelIntegration
    COMPONENTS_AVAILABLE = True
    print("✅ All trading components imported successfully")
except ImportError as e:
    print(f"⚠️ Component import failed: {e}")
    COMPONENTS_AVAILABLE = False

class CompleteLiveTradingSystem:
    """🚀 Complete Live Trading System - Das Herzstück von TRADINO UNSCHLAGBAR"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self.get_default_config()
        self.is_running = False
        self.is_initialized = False
        
        # Core components
        self.trading_engine = None
        self.risk_manager = None
        self.performance_monitor = None
        self.market_feed = None
        self.trading_api = None
        self.ai_models = None
        
        # System state
        self.system_health = {
            'all_systems_operational': False,
            'components_status': {},
            'last_health_check': None,
            'errors': [],
            'warnings': []
        }
        
        # Performance tracking
        self.session_stats = {
            'start_time': None,
            'trades_executed': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'uptime': '00:00:00'
        }
        
        print("🚀 Complete Live Trading System initialized")
        
        if COMPONENTS_AVAILABLE:
            self.initialize_all_components()
    
    def get_default_config(self) -> Dict[str, Any]:
        """⚙️ Vollständige System-Konfiguration"""
        
        return {
            # Trading Engine Settings
            'trading': {
                'symbols': ['BTC/USDT', 'ETH/USDT'],
                'signal_interval': 60,
                'min_confidence': 0.65,
                'enable_trading': False,  # Start in simulation
                'max_daily_trades': 15,
                'trading_exchange': 'bitget',
                'market_data_exchange': 'binance'
            },
            
            # Risk Management Settings
            'risk': {
                'max_portfolio_exposure': 0.7,
                'max_single_position': 0.08,
                'daily_loss_limit': 0.04,
                'max_drawdown_limit': 0.12,
                'stop_loss_percent': 0.02,
                'take_profit_percent': 0.04,
                'emergency_stop_loss': 0.08,
                'circuit_breaker_enabled': True
            },
            
            # Performance Monitoring
            'performance': {
                'track_equity_curve': True,
                'save_daily_reports': True,
                'alert_on_drawdown': True,
                'drawdown_alert_threshold': 0.08,
                'performance_alert_threshold': -0.03
            },
            
            # API Configuration
            'api': {
                'bitget_sandbox': True,
                'rate_limit_per_minute': 60,
                'connection_timeout': 30,
                'retry_attempts': 3
            },
            
            # System Settings
            'system': {
                'health_check_interval': 30,
                'auto_restart_on_error': True,
                'save_logs': True,
                'log_level': 'INFO',
                'backup_interval': 3600,  # 1 hour
                'max_log_files': 7
            },
            
            # AI Model Settings
            'ai': {
                'model_update_interval': 300,  # 5 minutes
                'confidence_threshold': 0.6,
                'ensemble_voting': True,
                'feature_importance_tracking': True
            }
        }
    
    def initialize_all_components(self):
        """🔧 Initialisiere alle System-Komponenten"""
        
        print("🔧 Initializing all trading system components...")
        
        try:
            # 1. AI Models
            print("🤖 Initializing AI Models...")
            self.ai_models = TrainedModelIntegration()
            self.system_health['components_status']['ai_models'] = self.ai_models.is_ready
            
            # 2. Risk Management
            print("🛡️ Initializing Risk Management...")
            self.risk_manager = initialize_risk_management(self.config['risk'])
            self.system_health['components_status']['risk_manager'] = True
            
            # 3. Performance Monitoring
            print("📈 Initializing Performance Monitoring...")
            self.performance_monitor = initialize_performance_monitoring(self.config['performance'])
            self.system_health['components_status']['performance_monitor'] = True
            
            # 4. Trading API
            if self.config['trading']['enable_trading']:
                print("🏦 Initializing Trading API...")
                self.trading_api = initialize_bitget_trading(sandbox=self.config['api']['bitget_sandbox'])
                self.system_health['components_status']['trading_api'] = self.trading_api.is_connected if self.trading_api else False
            else:
                print("ℹ️ Trading API disabled - running in simulation mode")
                self.system_health['components_status']['trading_api'] = 'simulation'
            
            # 5. Market Data Feed
            print("📊 Initializing Market Data Feed...")
            self.market_feed = LiveMarketFeed(self.config['trading']['market_data_exchange'])
            self.market_feed.config['symbols'] = self.config['trading']['symbols']
            self.system_health['components_status']['market_feed'] = True
            
            # 6. Trading Engine
            print("🚀 Initializing Trading Engine...")
            trading_config = {
                **self.config['trading'],
                'ai_models': self.ai_models,
                'risk_manager': self.risk_manager,
                'performance_monitor': self.performance_monitor,
                'trading_api': self.trading_api,
                'market_feed': self.market_feed
            }
            
            self.trading_engine = initialize_live_trading(trading_config)
            self.system_health['components_status']['trading_engine'] = True
            
            # Setup component interactions
            self.setup_component_interactions()
            
            self.is_initialized = True
            self.system_health['all_systems_operational'] = all(
                status == True or status == 'simulation' 
                for status in self.system_health['components_status'].values()
            )
            
            print(f"✅ All components initialized successfully!")
            print(f"🎯 System Status: {'OPERATIONAL' if self.system_health['all_systems_operational'] else 'PARTIAL'}")
            
        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
            self.system_health['errors'].append(f"Init error: {e}")
            self.is_initialized = False
    
    def setup_component_interactions(self):
        """🔗 Setup interactions between components"""
        
        # Connect performance monitor to risk manager alerts
        if self.risk_manager and self.performance_monitor:
            self.risk_manager.subscribe_to_alerts(self.on_risk_alert)
        
        # Connect trading engine to performance monitoring
        if self.trading_engine and self.performance_monitor:
            # This would be done via callbacks in the trading engine
            pass
        
        print("🔗 Component interactions configured")
    
    def on_risk_alert(self, alert: Dict[str, Any]):
        """🚨 Handle risk management alerts"""
        
        print(f"🚨 RISK ALERT RECEIVED: {alert}")
        
        # Auto-actions based on alert severity
        if alert.get('severity') == 'critical':
            print("🛑 CRITICAL ALERT - Implementing emergency measures")
            
            if self.trading_engine and self.trading_engine.is_running:
                self.trading_engine.pause_trading()
                print("⏸️ Trading paused due to critical risk alert")
        
        # Log alert
        self.system_health['warnings'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'risk_alert',
            'alert': alert
        })
    
    def start_live_trading(self):
        """🚀 Start the complete live trading system"""
        
        if not self.is_initialized:
            print("❌ Cannot start - system not initialized")
            return False
        
        if self.is_running:
            print("⚠️ System already running")
            return True
        
        print("🚀 STARTING LIVE TRADING SYSTEM")
        print("=" * 60)
        
        try:
            # Start all components
            self.session_stats['start_time'] = datetime.now()
            
            # 1. Start market data feed
            if self.market_feed:
                self.market_feed.start_live_feed()
                print("📊 Market data feed started")
            
            # 2. Start trading engine
            if self.trading_engine:
                self.trading_engine.start_trading()
                print("🚀 Trading engine started")
            
            # 3. Start system monitoring
            self.start_system_monitoring()
            
            self.is_running = True
            
            print("✅ LIVE TRADING SYSTEM STARTED SUCCESSFULLY")
            print(f"📊 Monitoring: {len(self.config['trading']['symbols'])} symbols")
            print(f"💰 Trading Mode: {'LIVE' if self.config['trading']['enable_trading'] else 'SIMULATION'}")
            print(f"🛡️ Risk Management: ACTIVE")
            print(f"📈 Performance Tracking: ACTIVE")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start live trading system: {e}")
            self.system_health['errors'].append(f"Start error: {e}")
            return False
    
    def stop_live_trading(self):
        """🛑 Stop the complete live trading system"""
        
        print("🛑 STOPPING LIVE TRADING SYSTEM")
        
        try:
            # Stop trading engine
            if self.trading_engine:
                self.trading_engine.stop_trading()
            
            # Stop market feed
            if self.market_feed:
                self.market_feed.stop_live_feed()
            
            # Stop system monitoring
            self.stop_system_monitoring()
            
            self.is_running = False
            
            # Generate final report
            self.generate_session_report()
            
            print("✅ Live trading system stopped successfully")
            
        except Exception as e:
            print(f"❌ Error stopping system: {e}")
    
    def start_system_monitoring(self):
        """👁️ Start system health monitoring"""
        
        def monitoring_loop():
            print("👁️ System monitoring started")
            
            while self.is_running:
                try:
                    self.perform_health_check()
                    self.update_session_stats()
                    
                    # Sleep for health check interval
                    time.sleep(self.config['system']['health_check_interval'])
                    
                except Exception as e:
                    print(f"⚠️ Monitoring error: {e}")
                    time.sleep(10)  # Wait before retry
            
            print("👁️ System monitoring stopped")
        
        # Start monitoring in background thread
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
    
    def stop_system_monitoring(self):
        """👁️ Stop system monitoring"""
        # This will be stopped when self.is_running = False
        pass
    
    def perform_health_check(self):
        """🏥 Perform comprehensive system health check"""
        
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'healthy',
            'components': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check AI Models
            if self.ai_models:
                health_status['components']['ai_models'] = 'healthy' if self.ai_models.is_ready else 'unhealthy'
            
            # Check Trading Engine
            if self.trading_engine:
                health_status['components']['trading_engine'] = 'healthy' if self.trading_engine.is_running else 'stopped'
            
            # Check Market Feed
            if self.market_feed:
                health_status['components']['market_feed'] = 'healthy' if self.market_feed.is_running else 'stopped'
            
            # Check Trading API
            if self.trading_api:
                health_status['components']['trading_api'] = 'healthy' if self.trading_api.is_connected else 'disconnected'
            else:
                health_status['components']['trading_api'] = 'simulation'
            
            # Check Risk Manager
            if self.risk_manager:
                health_status['components']['risk_manager'] = 'healthy'
            
            # Check Performance Monitor
            if self.performance_monitor:
                health_status['components']['performance_monitor'] = 'healthy'
            
            # Overall health assessment
            unhealthy_components = [comp for comp, status in health_status['components'].items() 
                                  if status not in ['healthy', 'simulation']]
            
            if unhealthy_components:
                health_status['overall_health'] = 'degraded'
                health_status['warnings'].append(f"Unhealthy components: {unhealthy_components}")
            
            self.system_health['last_health_check'] = health_status
            
            # Print health status periodically
            if datetime.now().minute % 5 == 0:  # Every 5 minutes
                print(f"🏥 System Health: {health_status['overall_health'].upper()} | "
                      f"Components: {len([s for s in health_status['components'].values() if s == 'healthy'])}/{len(health_status['components'])}")
            
        except Exception as e:
            print(f"❌ Health check error: {e}")
    
    def update_session_stats(self):
        """📊 Update session statistics"""
        
        if self.session_stats['start_time']:
            # Calculate uptime
            uptime = datetime.now() - self.session_stats['start_time']
            self.session_stats['uptime'] = str(uptime).split('.')[0]  # Remove microseconds
        
        # Update trade stats from trading engine
        if self.trading_engine:
            engine_stats = self.trading_engine.performance_stats
            self.session_stats['trades_executed'] = engine_stats.get('trades_executed', 0)
            self.session_stats['total_pnl'] = engine_stats.get('total_pnl', 0.0)
        
        # Update drawdown from performance monitor
        if self.performance_monitor:
            self.session_stats['max_drawdown'] = self.performance_monitor.metrics.get('max_drawdown', 0.0)
    
    def get_system_status(self) -> Dict[str, Any]:
        """📊 Get comprehensive system status"""
        
        return {
            'system_info': {
                'running': self.is_running,
                'initialized': self.is_initialized,
                'operational': self.system_health['all_systems_operational'],
                'uptime': self.session_stats['uptime'],
                'trading_mode': 'LIVE' if self.config['trading']['enable_trading'] else 'SIMULATION'
            },
            'components_status': self.system_health['components_status'],
            'session_stats': self.session_stats,
            'trading_stats': {
                'symbols_monitored': len(self.config['trading']['symbols']),
                'trades_today': self.session_stats['trades_executed'],
                'session_pnl': self.session_stats['total_pnl'],
                'max_drawdown': self.session_stats['max_drawdown']
            },
            'last_health_check': self.system_health.get('last_health_check'),
            'recent_errors': self.system_health['errors'][-5:],  # Last 5 errors
            'recent_warnings': self.system_health['warnings'][-5:],  # Last 5 warnings
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_session_report(self):
        """📋 Generate comprehensive session report"""
        
        try:
            report = {
                'session_summary': {
                    'start_time': self.session_stats['start_time'].isoformat() if self.session_stats['start_time'] else None,
                    'end_time': datetime.now().isoformat(),
                    'duration': self.session_stats['uptime'],
                    'total_trades': self.session_stats['trades_executed'],
                    'session_pnl': self.session_stats['total_pnl'],
                    'max_drawdown': self.session_stats['max_drawdown']
                },
                'system_performance': self.get_system_status(),
                'trading_performance': {},
                'risk_metrics': {},
                'errors_and_warnings': {
                    'errors': self.system_health['errors'],
                    'warnings': self.system_health['warnings']
                }
            }
            
            # Add detailed performance data
            if self.performance_monitor:
                report['trading_performance'] = self.performance_monitor.get_performance_summary()
            
            if self.risk_manager:
                report['risk_metrics'] = self.risk_manager.get_risk_dashboard()
            
            # Save report
            os.makedirs('session_reports', exist_ok=True)
            filename = f"session_reports/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"📋 Session report saved: {filename}")
            
            # Print summary
            print("\n" + "="*60)
            print("📋 TRADING SESSION SUMMARY")
            print("="*60)
            print(f"⏰ Duration: {self.session_stats['uptime']}")
            print(f"📈 Trades: {self.session_stats['trades_executed']}")
            print(f"💰 P&L: ${self.session_stats['total_pnl']:.2f}")
            print(f"📉 Max Drawdown: {self.session_stats['max_drawdown']:.1%}")
            print(f"🏥 System Health: {'GOOD' if len(self.system_health['errors']) == 0 else 'ISSUES'}")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error generating session report: {e}")
    
    def emergency_shutdown(self):
        """🆘 Emergency system shutdown"""
        
        print("🆘 EMERGENCY SHUTDOWN INITIATED")
        
        try:
            # Immediately stop trading
            if self.trading_engine:
                self.trading_engine.stop_trading()
            
            # Disconnect APIs
            if self.trading_api:
                print("🔌 Disconnecting trading API")
            
            # Save emergency report
            self.generate_session_report()
            
            self.is_running = False
            
            print("🆘 Emergency shutdown complete")
            
        except Exception as e:
            print(f"❌ Emergency shutdown error: {e}")

# Global system instance
live_trading_system = None

def initialize_complete_system(config: Dict[str, Any] = None) -> CompleteLiveTradingSystem:
    """🚀 Initialize the complete live trading system"""
    
    global live_trading_system
    live_trading_system = CompleteLiveTradingSystem(config)
    return live_trading_system

def get_live_trading_system() -> Optional[CompleteLiveTradingSystem]:
    """🚀 Get the complete live trading system"""
    return live_trading_system

def start_tradino_live_trading():
    """🚀 Start TRADINO live trading with default configuration"""
    
    print("🚀 TRADINO UNSCHLAGBAR LIVE TRADING SYSTEM")
    print("=" * 60)
    
    # Initialize system
    system = initialize_complete_system()
    
    if system.is_initialized:
        # Start trading
        success = system.start_live_trading()
        
        if success:
            print("✅ TRADINO is now LIVE and trading!")
            return system
        else:
            print("❌ Failed to start live trading")
            return None
    else:
        print("❌ System initialization failed")
        return None

# Demo and testing
if __name__ == "__main__":
    print("🚀 COMPLETE LIVE TRADING SYSTEM TEST")
    print("=" * 60)
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Cannot run - components not available")
        print("Please ensure all required files are present:")
        print("- live_trading_engine.py")
        print("- bitget_trading_api.py")  
        print("- risk_management_system.py")
        print("- performance_monitoring_system.py")
        sys.exit(1)
    
    # Test configuration for demo
    demo_config = {
        'trading': {
            'symbols': ['BTC/USDT'],
            'enable_trading': False,  # Simulation mode for demo
            'signal_interval': 30,    # Fast for demo
            'min_confidence': 0.6
        },
        'risk': {
            'max_portfolio_exposure': 0.5,
            'daily_loss_limit': 0.02
        },
        'system': {
            'health_check_interval': 10  # Fast for demo
        }
    }
    
    # Initialize and test
    print("🔧 Initializing system with demo configuration...")
    system = initialize_complete_system(demo_config)
    
    if system.is_initialized:
        print("✅ System initialized successfully!")
        
        # Get status
        status = system.get_system_status()
        print(f"📊 System Status: {status['system_info']}")
        
        # Uncomment to start demo trading
        # print("\n🚀 Starting demo trading for 60 seconds...")
        # system.start_live_trading()
        # time.sleep(60)
        # system.stop_live_trading()
        
    else:
        print("❌ System initialization failed")
    
    print("\n🚀 TRADINO UNSCHLAGBAR Complete Live Trading System ready!")
    print("💡 To start live trading: run start_tradino_live_trading()") 