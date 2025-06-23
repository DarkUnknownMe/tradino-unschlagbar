#!/usr/bin/env python3
"""
🚀 LIVE TRADING ENGINE
Real-time AI signal processing und trading execution
"""

import os
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import components
try:
    sys.path.append('/root/tradino')
    from tradino_unschlagbar.brain.trained_model_integration import TrainedModelIntegration
    from tradino_unschlagbar.connectors.live_market_feed import LiveMarketFeed
    from bitget_trading_api import BitgetTradingAPI
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Component import failed: {e}")
    COMPONENTS_AVAILABLE = False

class LiveTradingEngine:
    """🚀 Live Trading Engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self.get_default_config()
        self.is_running = False
        self.is_paused = False
        
        # Components
        self.ai_models = None
        self.market_feed = None
        self.trading_api = None
        
        # State
        self.current_signals = {}
        self.last_trades = {}
        self.performance_stats = {
            'trades_executed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'start_time': None
        }
        
        self.signal_history = []
        self.trade_log = []
        
        print("🚀 Live Trading Engine initialized")
        
        if COMPONENTS_AVAILABLE:
            self.initialize_components()
    
    def get_default_config(self) -> Dict[str, Any]:
        """⚙️ Default configuration"""
        
        return {
            'symbols': ['BTC/USDT'],
            'signal_interval': 60,
            'min_confidence': 0.65,
            'max_position_size': 0.05,
            'enable_trading': False,
            'max_daily_trades': 10,
            'max_consecutive_losses': 3,
            'market_update_interval': 30,
            'exchange': 'binance',
            'trading_exchange': 'bitget',
            'sandbox_mode': True,
            'log_signals': True,
            'log_trades': True
        }
    
    def initialize_components(self):
        """🔧 Initialize all components"""
        
        try:
            # AI Models
            self.ai_models = TrainedModelIntegration()
            if self.ai_models.is_ready:
                print("✅ AI Models ready")
            
            # Market Feed
            self.market_feed = LiveMarketFeed(self.config['exchange'])
            self.market_feed.config['symbols'] = self.config['symbols']
            self.market_feed.subscribe_to_data(self.on_market_data_update)
            print("✅ Market feed initialized")
            
            # Trading API
            if self.config['enable_trading']:
                self.trading_api = BitgetTradingAPI(sandbox=self.config['sandbox_mode'])
                if self.trading_api and self.trading_api.is_connected:
                    print("✅ Trading API connected")
                else:
                    print("❌ Trading API not connected - simulation mode")
                    self.config['enable_trading'] = False
            
        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
    
    def on_market_data_update(self, market_data: Dict, updated_symbols: List[str]):
        """📊 Handle market data updates"""
        
        try:
            for symbol in updated_symbols:
                if symbol in self.config['symbols']:
                    self.process_signal_for_symbol(symbol, market_data[symbol])
        except Exception as e:
            print(f"❌ Market update error: {e}")
    
    def process_signal_for_symbol(self, symbol: str, market_data: Dict):
        """🤖 Process AI signal"""
        
        if not self.ai_models or not self.ai_models.is_ready:
            return
        
        try:
            ohlcv_data = market_data.get('ohlcv')
            if ohlcv_data is None or ohlcv_data.empty:
                return
            
            # Generate signal
            signal = self.ai_models.get_trading_signal(
                ohlcv_data, 
                confidence_threshold=self.config['min_confidence']
            )
            
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
            
            # Execute trade if conditions met
            if self.should_execute_trade(signal, symbol):
                self.execute_trade(signal, symbol)
            
            print(f"🤖 {symbol}: {signal['action']} | "
                  f"Conf: {signal['confidence']:.2f} | "
                  f"${signal['market_price']:,.2f}")
            
        except Exception as e:
            print(f"❌ Signal processing error for {symbol}: {e}")
    
    def should_execute_trade(self, signal: Dict[str, Any], symbol: str) -> bool:
        """🤔 Should we execute this trade?"""
        
        if not self.config['enable_trading']:
            return False
        
        if signal['action'] == 'hold':
            return False
        
        if signal['confidence'] < self.config['min_confidence']:
            return False
        
        if self.get_todays_trade_count() >= self.config['max_daily_trades']:
            return False
        
        if self.get_consecutive_losses() >= self.config['max_consecutive_losses']:
            return False
        
        # Cooldown check
        if symbol in self.last_trades:
            last_time = self.last_trades[symbol].get('timestamp')
            if last_time:
                last_trade = datetime.fromisoformat(last_time)
                if (datetime.now() - last_trade).total_seconds() < 300:
                    return False
        
        return True
    
    def execute_trade(self, signal: Dict[str, Any], symbol: str):
        """💰 Execute trade"""
        
        try:
            if self.trading_api and self.trading_api.is_connected:
                result = self.trading_api.execute_ai_signal(signal, symbol)
            else:
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
            
            # Update stats
            self.performance_stats['trades_executed'] += 1
            
            if result.get('success'):
                self.performance_stats['successful_trades'] += 1
                print(f"✅ Trade executed: {symbol} {signal['action']}")
            else:
                self.performance_stats['failed_trades'] += 1
                print(f"❌ Trade failed: {symbol}")
            
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
            self.performance_stats['failed_trades'] += 1
    
    def simulate_trade(self, signal: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """🎮 Simulate trade"""
        
        success_probability = signal['confidence']
        simulated_success = np.random.random() < success_probability
        
        if simulated_success:
            simulated_pnl = np.random.normal(0.01, 0.005)
            self.performance_stats['total_pnl'] += simulated_pnl
            
            return {
                'success': True,
                'action': signal['action'],
                'simulated_pnl': simulated_pnl,
                'simulation': True,
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'success': False,
                'error': 'Simulated failure',
                'simulation': True,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_todays_trade_count(self) -> int:
        """📈 Count today's trades"""
        
        today = datetime.now().date()
        return sum(1 for trade in self.trade_log 
                  if datetime.fromisoformat(trade['timestamp']).date() == today)
    
    def get_consecutive_losses(self) -> int:
        """📉 Count consecutive losses"""
        
        consecutive = 0
        for trade in reversed(self.trade_log[-10:]):
            if trade['result'].get('success'):
                break
            consecutive += 1
        return consecutive
    
    def start_trading(self):
        """🚀 Start trading engine"""
        
        if self.is_running:
            print("⚠️ Already running")
            return
        
        if not COMPONENTS_AVAILABLE:
            print("❌ Components not available")
            return
        
        self.is_running = True
        self.performance_stats['start_time'] = datetime.now()
        
        print("🚀 Starting live trading...")
        
        # Start market feed
        if self.market_feed:
            self.market_feed.start_live_feed()
        
        # Trading loop
        def trading_loop():
            print("🔄 Trading loop started")
            
            while self.is_running:
                try:
                    if not self.is_paused:
                        self.process_periodic_tasks()
                    
                    time.sleep(self.config['signal_interval'])
                    
                except Exception as e:
                    print(f"❌ Trading loop error: {e}")
                    time.sleep(5)
            
            print("🛑 Trading loop stopped")
        
        trading_thread = threading.Thread(target=trading_loop, daemon=True)
        trading_thread.start()
        
        print(f"✅ Engine started | Symbols: {len(self.config['symbols'])} | "
              f"Mode: {'Live' if self.config['enable_trading'] else 'Simulation'}")
    
    def stop_trading(self):
        """🛑 Stop trading"""
        
        self.is_running = False
        
        if self.market_feed:
            self.market_feed.stop_live_feed()
        
        print("🛑 Trading engine stopped")
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
        """🔄 Periodic maintenance"""
        
        try:
            # Cleanup old data
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-500:]
            
            if len(self.trade_log) > 1000:
                self.trade_log = self.trade_log[-500:]
                
        except Exception as e:
            print(f"⚠️ Periodic task error: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """📊 Get current status"""
        
        return {
            'engine_status': {
                'running': self.is_running,
                'paused': self.is_paused,
                'uptime': self.get_uptime()
            },
            'performance': self.performance_stats,
            'current_signals': self.current_signals,
            'symbols': self.config['symbols'],
            'mode': 'Live' if self.config['enable_trading'] else 'Simulation',
            'components': {
                'ai_models': self.ai_models.is_ready if self.ai_models else False,
                'market_feed': self.market_feed.is_running if self.market_feed else False,
                'trading_api': self.trading_api.is_connected if self.trading_api else False
            }
        }
    
    def get_uptime(self) -> str:
        """⏰ Get uptime"""
        
        if self.performance_stats['start_time']:
            uptime = datetime.now() - self.performance_stats['start_time']
            return str(uptime).split('.')[0]
        return "Not started"
    
    def print_performance_summary(self):
        """📊 Print performance summary"""
        
        print("\n" + "="*50)
        print("📊 TRADING PERFORMANCE SUMMARY")
        print("="*50)
        
        stats = self.performance_stats
        
        print(f"⏰ Runtime: {self.get_uptime()}")
        print(f"📈 Total trades: {stats['trades_executed']}")
        print(f"✅ Successful: {stats['successful_trades']}")
        print(f"❌ Failed: {stats['failed_trades']}")
        
        if stats['trades_executed'] > 0:
            success_rate = (stats['successful_trades'] / stats['trades_executed']) * 100
            print(f"🎯 Success rate: {success_rate:.1f}%")
        
        print(f"💰 P&L: ${stats['total_pnl']:.2f}")
        print(f"📊 Symbols: {', '.join(self.config['symbols'])}")
        print(f"🤖 Signals: {len(self.signal_history)}")

# Global instance
live_engine = None

def initialize_live_trading(config: Dict[str, Any] = None) -> LiveTradingEngine:
    """🚀 Initialize trading engine"""
    global live_engine
    live_engine = LiveTradingEngine(config)
    return live_engine

def get_trading_engine() -> Optional[LiveTradingEngine]:
    """🚀 Get trading engine"""
    return live_engine

# Demo
if __name__ == "__main__":
    print("🚀 LIVE TRADING ENGINE READY")
    
    demo_config = {
        'symbols': ['BTC/USDT'],
        'enable_trading': False,
        'min_confidence': 0.6,
        'signal_interval': 10
    }
    
    engine = initialize_live_trading(demo_config)
    print(f"📊 Status: {engine.get_current_status()}") 