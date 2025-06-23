#!/usr/bin/env python3
"""
🚀 LIVE TRADING ENGINE
Real-time AI signal processing und trading execution
"""

import os
import sys
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our components
try:
    # Try to import from project structure
    sys.path.append('/root/tradino')
    from tradino_unschlagbar.brain.trained_model_integration import TrainedModelIntegration
    from tradino_unschlagbar.connectors.live_market_feed import LiveMarketFeed
    from bitget_trading_api import BitgetTradingAPI
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Component import failed: {e}")
    COMPONENTS_AVAILABLE = False

class LiveTradingEngine:
    """🚀 Live Trading Engine - Verbindet AI, Marktdaten und Trading"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self.get_default_config()
        self.is_running = False
        self.is_paused = False
        
        # Components
        self.ai_models = None
        self.market_feed = None
        self.trading_api = None
        
        # State tracking
        self.current_signals = {}
        self.last_trades = {}
        self.performance_stats = {
            'trades_executed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'start_time': None,
            'last_signal_time': None
        }
        
        # Signal processing
        self.signal_history = []
        self.trade_log = []
        
        print("🚀 Live Trading Engine initialized")
        
        if COMPONENTS_AVAILABLE:
            self.initialize_components()
    
    def get_default_config(self) -> Dict[str, Any]:
        """⚙️ Get default trading configuration"""
        
        return {
            # Trading parameters
            'symbols': ['BTC/USDT'],
            'signal_interval': 60,  # Process signals every 60 seconds
            'min_confidence': 0.65,  # Minimum confidence for trading
            'max_position_size': 0.05,  # Max 5% per trade
            'enable_trading': False,  # Start in simulation mode
            
            # Risk management
            'max_daily_trades': 10,
            'max_consecutive_losses': 3,
            'stop_loss_percent': 0.02,  # 2% stop loss
            'take_profit_percent': 0.04,  # 4% take profit
            
            # Market data
            'market_update_interval': 30,  # Update market data every 30s
            'data_history_length': 100,  # Keep 100 candles
            
            # API settings
            'exchange': 'binance',  # Market data source
            'trading_exchange': 'bitget',  # Trading execution
            'sandbox_mode': True,
            
            # Logging
            'log_signals': True,
            'log_trades': True,
            'save_performance': True
        }
    
    def initialize_components(self):
        """🔧 Initialize all components"""
        
        try:
            # Initialize AI models
            self.ai_models = TrainedModelIntegration()
            if self.ai_models.is_ready:
                print("✅ AI Models loaded and ready")
            else:
                print("❌ AI Models not ready")
            
            # Initialize market feed
            self.market_feed = LiveMarketFeed(self.config['exchange'])
            self.market_feed.config['symbols'] = self.config['symbols']
            self.market_feed.config['update_interval'] = self.config['market_update_interval']
            
            # Subscribe to market data updates
            self.market_feed.subscribe_to_data(self.on_market_data_update)
            print("✅ Market feed initialized")
            
            # Initialize trading API (if enabled)
            if self.config['enable_trading']:
                self.trading_api = BitgetTradingAPI(sandbox=self.config['sandbox_mode'])
                if self.trading_api and self.trading_api.is_connected:
                    print("✅ Trading API connected")
                else:
                    print("❌ Trading API not connected - running in simulation mode")
                    self.config['enable_trading'] = False
            else:
                print("ℹ️ Trading disabled - running in simulation mode")
            
        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
            COMPONENTS_AVAILABLE = False
    
    def on_market_data_update(self, market_data: Dict, updated_symbols: List[str]):
        """📊 Handle market data updates"""
        
        try:
            for symbol in updated_symbols:
                if symbol in self.config['symbols']:
                    # Process signal for this symbol
                    self.process_signal_for_symbol(symbol, market_data[symbol])
                    
        except Exception as e:
            print(f"❌ Error processing market update: {e}")
    
    def process_signal_for_symbol(self, symbol: str, market_data: Dict):
        """🤖 Process AI signal for a specific symbol"""
        
        if not self.ai_models or not self.ai_models.is_ready:
            return
        
        try:
            # Get OHLCV data
            ohlcv_data = market_data.get('ohlcv')
            if ohlcv_data is None or ohlcv_data.empty:
                return
            
            # Generate AI signal
            signal = self.ai_models.get_trading_signal(
                ohlcv_data, 
                confidence_threshold=self.config['min_confidence']
            )
            
            # Store signal
            signal['symbol'] = symbol
            signal['market_price'] = float(ohlcv_data['Close'].iloc[-1])
            self.current_signals[symbol] = signal
            
            # Log signal
            if self.config['log_signals']:
                self.signal_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'signal': signal
                })
            
            # Execute trade if conditions are met
            if self.should_execute_trade(signal, symbol):
                self.execute_trade(signal, symbol)
            
            self.performance_stats['last_signal_time'] = datetime.now()
            
            # Print signal info
            print(f"🤖 {symbol} Signal: {signal['action']} | "
                  f"Confidence: {signal['confidence']:.2f} | "
                  f"Price: ${signal['market_price']:,.2f}")
            
        except Exception as e:
            print(f"❌ Error processing signal for {symbol}: {e}")
    
    def should_execute_trade(self, signal: Dict[str, Any], symbol: str) -> bool:
        """🤔 Determine if trade should be executed"""
        
        # Check if trading is enabled
        if not self.config['enable_trading']:
            return False
        
        # Check signal action
        if signal['action'] == 'hold':
            return False
        
        # Check confidence threshold
        if signal['confidence'] < self.config['min_confidence']:
            return False
        
        # Check daily trade limit
        today_trades = self.get_todays_trade_count()
        if today_trades >= self.config['max_daily_trades']:
            return False
        
        # Check consecutive losses
        if self.get_consecutive_losses() >= self.config['max_consecutive_losses']:
            return False
        
        # Check if we already have a recent trade for this symbol
        if symbol in self.last_trades:
            last_trade_time = self.last_trades[symbol].get('timestamp')
            if last_trade_time:
                last_trade = datetime.fromisoformat(last_trade_time.replace('Z', '+00:00').replace('+00:00', ''))
                time_since_last = (datetime.now() - last_trade).total_seconds()
                if time_since_last < 300:  # 5 minutes cooldown
                    return False
        
        return True
    
    def execute_trade(self, signal: Dict[str, Any], symbol: str):
        """💰 Execute trade based on signal"""
        
        try:
            if self.trading_api and self.trading_api.is_connected:
                # Real trading
                result = self.trading_api.execute_ai_signal(signal, symbol)
            else:
                # Simulated trading
                result = self.simulate_trade(signal, symbol)
            
            # Log trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'signal': signal,
                'result': result,
                'mode': 'live' if self.config['enable_trading'] else 'simulation'
            }
            
            self.trade_log.append(trade_record)
            self.last_trades[symbol] = trade_record
            
            # Update performance stats
            self.performance_stats['trades_executed'] += 1
            
            if result.get('success'):
                self.performance_stats['successful_trades'] += 1
                print(f"✅ Trade executed: {symbol} {signal['action']} "
                      f"(Confidence: {signal['confidence']:.2f})")
            else:
                self.performance_stats['failed_trades'] += 1
                print(f"❌ Trade failed: {symbol} - {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
            self.performance_stats['failed_trades'] += 1
    
    def simulate_trade(self, signal: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """🎮 Simulate trade execution"""
        
        # Simulate trade with some randomness for realism
        success_probability = signal['confidence']
        simulated_success = np.random.random() < success_probability
        
        if simulated_success:
            # Simulate successful trade
            simulated_pnl = np.random.normal(0.01, 0.005)  # 1% average return with 0.5% std
            
            result = {
                'success': True,
                'action': signal['action'],
                'simulated_pnl': simulated_pnl,
                'simulated_price': signal['market_price'],
                'simulation': True,
                'timestamp': datetime.now().isoformat()
            }
            
            self.performance_stats['total_pnl'] += simulated_pnl
            
        else:
            # Simulate failed trade
            result = {
                'success': False,
                'error': 'Simulated trade failure',
                'simulation': True,
                'timestamp': datetime.now().isoformat()
            }
        
        return result
    
    def get_todays_trade_count(self) -> int:
        """📈 Get number of trades executed today"""
        
        today = datetime.now().date()
        count = 0
        
        for trade in self.trade_log:
            trade_date = datetime.fromisoformat(trade['timestamp']).date()
            if trade_date == today:
                count += 1
        
        return count
    
    def get_consecutive_losses(self) -> int:
        """📉 Get consecutive loss count"""
        
        consecutive_losses = 0
        
        # Check recent trades in reverse order
        for trade in reversed(self.trade_log[-10:]):  # Check last 10 trades
            if trade['result'].get('success'):
                break  # Stop at first success
            else:
                consecutive_losses += 1
        
        return consecutive_losses
    
    def start_trading(self):
        """🚀 Start live trading engine"""
        
        if self.is_running:
            print("⚠️ Trading engine already running")
            return
        
        if not COMPONENTS_AVAILABLE:
            print("❌ Cannot start - components not available")
            return
        
        self.is_running = True
        self.performance_stats['start_time'] = datetime.now()
        
        print("🚀 Starting live trading engine...")
        
        # Start market feed
        if self.market_feed:
            self.market_feed.start_live_feed()
        
        # Start signal processing loop
        def trading_loop():
            print("🔄 Trading loop started")
            
            while self.is_running:
                try:
                    if not self.is_paused:
                        # Process any pending signals
                        self.process_periodic_tasks()
                    
                    # Wait for next iteration
                    time.sleep(self.config['signal_interval'])
                    
                except Exception as e:
                    print(f"❌ Trading loop error: {e}")
                    time.sleep(5)  # Wait before retry
            
            print("🛑 Trading loop stopped")
        
        # Start trading loop in background thread
        trading_thread = threading.Thread(target=trading_loop, daemon=True)
        trading_thread.start()
        
        print(f"✅ Live trading engine started")
        print(f"📊 Monitoring {len(self.config['symbols'])} symbols")
        print(f"🎯 Min confidence: {self.config['min_confidence']}")
        print(f"💰 Trading mode: {'Live' if self.config['enable_trading'] else 'Simulation'}")
    
    def stop_trading(self):
        """🛑 Stop live trading engine"""
        
        self.is_running = False
        
        if self.market_feed:
            self.market_feed.stop_live_feed()
        
        print("🛑 Live trading engine stopped")
        self.print_performance_summary()
    
    def pause_trading(self):
        """⏸️ Pause trading"""
        
        self.is_paused = True
        print("⏸️ Trading paused")
    
    def resume_trading(self):
        """▶️ Resume trading"""
        
        self.is_paused = False
        print("▶️ Trading resumed")
    
    def process_periodic_tasks(self):
        """🔄 Process periodic maintenance tasks"""
        
        try:
            # Clean old signal history
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-500:]
            
            # Clean old trade log
            if len(self.trade_log) > 1000:
                self.trade_log = self.trade_log[-500:]
            
            # Save performance data
            if self.config['save_performance']:
                self.save_performance_data()
                
        except Exception as e:
            print(f"⚠️ Periodic task error: {e}")
    
    def save_performance_data(self):
        """💾 Save performance data to file"""
        
        try:
            performance_data = {
                'stats': self.performance_stats,
                'current_signals': self.current_signals,
                'recent_trades': self.trade_log[-10:],  # Last 10 trades
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            os.makedirs('performance_logs', exist_ok=True)
            
            filename = f"performance_logs/trading_performance_{datetime.now().strftime('%Y%m%d')}.json"
            with open(filename, 'w') as f:
                json.dump(performance_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"⚠️ Error saving performance data: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """📊 Get current trading status"""
        
        return {
            'engine_status': {
                'running': self.is_running,
                'paused': self.is_paused,
                'uptime': self.get_uptime()
            },
            'performance': self.performance_stats,
            'current_signals': self.current_signals,
            'active_symbols': list(self.config['symbols']),
            'trading_mode': 'Live' if self.config['enable_trading'] else 'Simulation',
            'components': {
                'ai_models': self.ai_models.is_ready if self.ai_models else False,
                'market_feed': self.market_feed.is_running if self.market_feed else False,
                'trading_api': self.trading_api.is_connected if self.trading_api else False
            }
        }
    
    def get_uptime(self) -> str:
        """⏰ Get engine uptime"""
        
        if self.performance_stats['start_time']:
            uptime = datetime.now() - self.performance_stats['start_time']
            return str(uptime).split('.')[0]  # Remove microseconds
        return "Not started"
    
    def print_performance_summary(self):
        """📊 Print performance summary"""
        
        print("\n" + "="*60)
        print("📊 TRADING ENGINE PERFORMANCE SUMMARY")
        print("="*60)
        
        stats = self.performance_stats
        
        print(f"⏰ Runtime: {self.get_uptime()}")
        print(f"📈 Total trades: {stats['trades_executed']}")
        print(f"✅ Successful: {stats['successful_trades']}")
        print(f"❌ Failed: {stats['failed_trades']}")
        
        if stats['trades_executed'] > 0:
            success_rate = (stats['successful_trades'] / stats['trades_executed']) * 100
            print(f"🎯 Success rate: {success_rate:.1f}%")
        
        if self.config['enable_trading']:
            print(f"💰 Total P&L: ${stats['total_pnl']:.2f}")
        else:
            print(f"🎮 Simulated P&L: ${stats['total_pnl']:.2f}")
        
        print(f"📊 Symbols monitored: {', '.join(self.config['symbols'])}")
        print(f"🤖 AI Signals processed: {len(self.signal_history)}")
        
        print("\n🚀 Trading engine session complete!")

# Global instance
live_engine = None

def initialize_live_trading(config: Dict[str, Any] = None) -> LiveTradingEngine:
    """🚀 Initialize live trading engine"""
    
    global live_engine
    live_engine = LiveTradingEngine(config)
    return live_engine

def get_trading_engine() -> Optional[LiveTradingEngine]:
    """🚀 Get the global trading engine"""
    return live_engine

def start_live_trading():
    """🚀 Start live trading"""
    if live_engine:
        live_engine.start_trading()

def stop_live_trading():
    """🛑 Stop live trading"""
    if live_engine:
        live_engine.stop_trading()

# Demo
if __name__ == "__main__":
    print("🚀 LIVE TRADING ENGINE TEST")
    print("=" * 50)
    
    # Demo configuration
    demo_config = {
        'symbols': ['BTC/USDT'],
        'enable_trading': False,  # Simulation mode
        'min_confidence': 0.6,
        'signal_interval': 10,  # Fast for demo
        'max_daily_trades': 5
    }
    
    # Initialize and test
    engine = initialize_live_trading(demo_config)
    
    if COMPONENTS_AVAILABLE:
        print("✅ All components available - ready for live trading!")
        
        # Uncomment to start demo trading
        # engine.start_trading()
        # time.sleep(30)  # Run for 30 seconds
        # engine.stop_trading()
    else:
        print("⚠️ Some components missing - check imports and setup")
    
    status = engine.get_current_status()
    print(f"\n📊 Engine Status: {status}")