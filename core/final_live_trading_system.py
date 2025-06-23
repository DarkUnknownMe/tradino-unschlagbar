#!/usr/bin/env python3
"""
🚀 REAL LIVE TRADING SYSTEM - NO SIMULATION!
Vollständiges ECHTES Live Trading System für TRADINO UNSCHLAGBAR
"""

import os
import sys
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from both possible locations
    if os.path.exists('tradino_unschlagbar/.env'):
        load_dotenv('tradino_unschlagbar/.env')
        print("✅ Loaded .env from tradino_unschlagbar/.env")
    elif os.path.exists('.env'):
        load_dotenv('.env')
        print("✅ Loaded .env from current directory")
    else:
        print("⚠️ No .env file found")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables")

# Add project path
sys.path.append('/root/tradino')

# Import REAL components - NO SIMULATION!
try:
    from tradino_unschlagbar.brain.trained_model_integration import TrainedModelIntegration
    from tradino_unschlagbar.connectors.live_market_feed import LiveMarketFeed
    from bitget_trading_api import BitgetTradingAPI
    AI_AVAILABLE = True
    TRADING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    AI_AVAILABLE = False
    TRADING_AVAILABLE = False

class RealLiveTradingSystem:
    """🚀 REAL Live Trading System - NO SIMULATION!"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self.get_default_config()
        self.is_running = False
        self.is_initialized = False
        
        # REAL components - NO SIMULATION!
        self.ai_models = None
        self.market_feed = None
        self.trading_api = None
        
        # System state
        self.session_stats = {
            'start_time': None,
            'signals_generated': 0,
            'trades_executed': 0,
            'total_pnl': 0.0,
            'uptime': '00:00:00',
            'latest_signals': [],
            'real_trades': []
        }
        
        print("🚀 REAL TRADINO Live Trading System - NO SIMULATION!")
        self.initialize_components()
    
    def get_default_config(self) -> Dict[str, Any]:
        """⚙️ Real trading configuration"""
        
        return {
            'symbols': ['BTC/USDT:USDT', 'ETH/USDT:USDT'],  # FUTURES SYMBOLS!
            'signal_interval': 60,  # 1 minute for real trading
            'min_confidence': 0.75,  # Higher confidence for real money
            'enable_trading': True,  # REAL TRADING ENABLED
            'max_daily_trades': 10,  # Reduced for safety
            'position_size': 0.02,  # 2% of portfolio per trade
            'stop_loss': 0.015,  # 1.5% stop loss
            'take_profit': 0.03,  # 3% take profit
            'exchange': 'bitget',
            'use_real_money': True  # REAL MONEY TRADING
        }
    
    def initialize_components(self):
        """🔧 Initialize REAL trading components"""
        
        try:
            # Initialize AI Models
            if AI_AVAILABLE:
                print("🤖 Initializing REAL AI Models...")
                self.ai_models = TrainedModelIntegration()
                
                if self.ai_models.is_ready:
                    print("✅ REAL AI Models loaded successfully")
                else:
                    print("❌ AI Models not ready")
                    return
            else:
                print("❌ AI models not available")
                return
            
            # Initialize REAL Market Feed
            print("📊 Initializing REAL Market Data Feed...")
            self.market_feed = LiveMarketFeed(self.config['exchange'])
            
            if self.market_feed.exchange:
                print("✅ REAL Market Feed connected")
            else:
                print("❌ Market Feed connection failed")
                return
            
            # Initialize REAL Trading API
            print("🏦 Initializing REAL Trading API...")
            
            # Check for API credentials from .env file
            api_key = os.getenv('BITGET_API_KEY')
            secret = os.getenv('BITGET_SECRET_KEY') or os.getenv('BITGET_SECRET')
            passphrase = os.getenv('BITGET_PASSPHRASE')
            sandbox_mode = os.getenv('BITGET_SANDBOX', 'true').lower() == 'true'
            
            print(f"🔍 API Key found: {'Yes' if api_key else 'No'}")
            print(f"🔍 Secret found: {'Yes' if secret else 'No'}")
            print(f"🔍 Passphrase found: {'Yes' if passphrase else 'No'}")
            print(f"🔍 Sandbox mode: {sandbox_mode}")
            
            if api_key and secret and passphrase:
                self.trading_api = BitgetTradingAPI(
                    api_key=api_key,
                    secret=secret,
                    passphrase=passphrase,
                    sandbox=sandbox_mode
                )
                
                if self.trading_api.is_connected:
                    mode = "SANDBOX" if sandbox_mode else "LIVE"
                    print(f"✅ REAL Trading API connected ({mode})")
                    try:
                        balance = self.trading_api.get_total_balance()
                        print(f"💰 Account Balance: ${balance:.2f}")
                    except Exception as e:
                        print(f"⚠️ Could not fetch balance: {e}")
                    self.is_initialized = True
                else:
                    print("❌ Trading API connection failed")
            else:
                print("❌ Trading API credentials incomplete in .env file")
                print("📋 Required in .env file:")
                print("   BITGET_API_KEY=your_key")
                print("   BITGET_SECRET_KEY=your_secret")
                print("   BITGET_PASSPHRASE=your_passphrase")
                
                # Try fallback without credentials
                self.trading_api = BitgetTradingAPI(sandbox=True)
                self.is_initialized = True
                print("⚠️ Running without API credentials - trades will fail")
                
        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
            self.is_initialized = False
    
    def get_real_market_data(self, symbol: str):
        """📊 Get REAL market data - NO SIMULATION!"""
        
        if not self.market_feed:
            raise Exception("Market feed not available")
        
        try:
            # Fetch REAL OHLCV data
            ohlcv_data = self.market_feed.fetch_ohlcv_data(symbol, limit=100)
            
            if ohlcv_data.empty:
                raise Exception(f"No data received for {symbol}")
            
            # Get REAL ticker data
            ticker_data = self.market_feed.fetch_ticker_data(symbol)
            
            print(f"📊 REAL data fetched for {symbol}: {len(ohlcv_data)} candles, "
                  f"Price: ${ticker_data.get('last_price', 0):.2f}")
            
            return ohlcv_data, ticker_data
            
        except Exception as e:
            print(f"❌ Error fetching REAL data for {symbol}: {e}")
            return None, None
    
    def generate_real_trading_signal(self, symbol: str) -> Dict[str, Any]:
        """🤖 Generate REAL trading signal - NO SIMULATION!"""
        
        if not self.ai_models or not self.ai_models.is_ready:
            raise Exception("AI models not ready")
        
        try:
            # Get REAL market data
            ohlcv_data, ticker_data = self.get_real_market_data(symbol)
            
            if ohlcv_data is None:
                return {
                    'symbol': symbol,
                    'action': 'hold',
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'no_data',
                    'error': 'No market data available'
                }
            
            # Get REAL AI signal
            signal = self.ai_models.get_trading_signal(
                ohlcv_data, 
                confidence_threshold=self.config['min_confidence']
            )
            
            # Add real market context
            signal.update({
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'source': 'real_ai_models',
                'market_price': ticker_data.get('last_price', 0),
                'volume_24h': ticker_data.get('volume', 0),
                'change_24h': ticker_data.get('change_24h_percent', 0)
            })
            
            print(f"🤖 REAL AI Signal for {symbol}: {signal['action']} "
                  f"(Confidence: {signal['confidence']:.2f})")
            
            return signal
            
        except Exception as e:
            print(f"❌ Error generating REAL signal for {symbol}: {e}")
            
            return {
                'symbol': symbol,
                'action': 'hold',
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat(),
                'source': 'error_fallback',
                'error': str(e)
            }
    
    def execute_real_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """💰 Execute REAL trade - NO SIMULATION!"""
        
        if not self.trading_api or not self.trading_api.is_connected:
            return {'executed': False, 'reason': 'Trading API not connected'}
        
        try:
            if signal['action'] == 'hold':
                return {'executed': False, 'reason': 'Hold signal'}
            
            if signal['confidence'] < self.config['min_confidence']:
                return {'executed': False, 
                       'reason': f"Low confidence: {signal['confidence']:.2f}"}
            
            if self.session_stats['trades_executed'] >= self.config['max_daily_trades']:
                return {'executed': False, 'reason': 'Daily trade limit reached'}
            
            # Calculate REAL position size
            current_price = signal.get('market_price', 0)
            if current_price <= 0:
                return {'executed': False, 'reason': 'Invalid market price'}
            
            position_size = self.trading_api.calculate_position_size(
                signal['symbol'], 
                current_price, 
                self.config['position_size']
            )
            
            if position_size <= 0:
                return {'executed': False, 'reason': 'Insufficient balance for trade'}
            
            # Execute REAL trade
            side = 'buy' if signal['action'] == 'buy' else 'sell'
            
            print(f"💰 Executing REAL trade: {side} {position_size} {signal['symbol']} "
                  f"at ${current_price:.2f}")
            
            trade_result = self.trading_api.place_market_order(
                symbol=signal['symbol'],
                side=side,
                amount=position_size
            )
            
            if trade_result.get('success'):
                self.session_stats['trades_executed'] += 1
                
                # Store REAL trade record
                real_trade = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': signal['symbol'],
                    'side': side,
                    'amount': position_size,
                    'price': current_price,
                    'value': position_size * current_price,
                    'confidence': signal['confidence'],
                    'order_id': trade_result.get('order', {}).get('id'),
                    'exchange': 'bitget'
                }
                
                self.session_stats['real_trades'].append(real_trade)
                
                print(f"✅ REAL TRADE EXECUTED: {side} ${position_size * current_price:.2f} "
                      f"{signal['symbol']}")
                
                return {'executed': True, 'trade': real_trade, 'result': trade_result}
            
            else:
                error_msg = trade_result.get('error', 'Unknown error')
                print(f"❌ REAL trade failed: {error_msg}")
                
                return {'executed': False, 'reason': error_msg}
            
        except Exception as e:
            error_msg = f"Trade execution error: {e}"
            print(f"❌ {error_msg}")
            
            return {'executed': False, 'reason': error_msg}
    
    def start_real_trading(self):
        """🚀 Start REAL trading - NO SIMULATION!"""
        
        if not self.is_initialized:
            print("❌ System not initialized")
            return
        
        if self.is_running:
            print("⚠️ Trading already running")
            return
        
        self.is_running = True
        self.session_stats['start_time'] = datetime.now()
        
        print("🚀 STARTING REAL LIVE TRADING - NO SIMULATION!")
        print("==================================================")
        print(f"🔄 REAL Trading loop started")
        print(f"📊 Symbols: {', '.join(self.config['symbols'])}")
        print(f"💰 Mode: REAL MONEY TRADING")
        print(f"🏦 Exchange: {self.config['exchange'].upper()}")
        
        if self.trading_api and self.trading_api.is_connected:
            balance = self.trading_api.get_total_balance()
            print(f"💵 Account Balance: ${balance:.2f}")
        
        print("✅ REAL Trading system started!")
        
        def real_trading_loop():
            """🔄 REAL trading execution loop"""
            
            while self.is_running:
                try:
                    loop_start = time.time()
                    
                    # Process each symbol
                    for symbol in self.config['symbols']:
                        if not self.is_running:
                            break
                        
                        # Generate REAL signal
                        signal = self.generate_real_trading_signal(symbol)
                        self.session_stats['signals_generated'] += 1
                        self.session_stats['latest_signals'].append(signal)
                        
                        # Keep only last 10 signals
                        if len(self.session_stats['latest_signals']) > 10:
                            self.session_stats['latest_signals'].pop(0)
                        
                        # Execute REAL trade if signal is strong
                        if signal['action'] != 'hold' and signal['confidence'] >= self.config['min_confidence']:
                            execution_result = self.execute_real_trade(signal)
                            
                            if execution_result['executed']:
                                print(f"✅ REAL TRADE: {signal['symbol']} {signal['action']} | "
                                      f"Conf: {signal['confidence']:.2f} | Status: EXECUTED")
                            else:
                                print(f"⚠️ REAL TRADE: {signal['symbol']} {signal['action']} | "
                                      f"Conf: {signal['confidence']:.2f} | "
                                      f"Reason: {execution_result['reason']}")
                        else:
                            print(f"🤖 {signal['symbol']}: {signal['action']} | "
                                  f"Conf: {signal['confidence']:.2f} | Status: Not executed")
                        
                        # Small delay between symbols
                        time.sleep(2)
                    
                    # Update uptime
                    if self.session_stats['start_time']:
                        uptime = datetime.now() - self.session_stats['start_time']
                        self.session_stats['uptime'] = str(uptime).split('.')[0]
                    
                    # Wait for next cycle
                    loop_duration = time.time() - loop_start
                    sleep_time = max(0, self.config['signal_interval'] - loop_duration)
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                except Exception as e:
                    print(f"❌ Error in REAL trading loop: {e}")
                    time.sleep(10)  # Wait before retrying
        
        # Start REAL trading thread
        trading_thread = threading.Thread(target=real_trading_loop, daemon=True)
        trading_thread.start()
    
    def stop_real_trading(self):
        """🛑 Stop REAL trading"""
        
        print("🛑 Stopping REAL trading system...")
        self.is_running = False
        time.sleep(2)  # Allow current operations to complete
        
        self.print_real_session_summary()
    
    def print_real_session_summary(self):
        """📊 Print REAL trading session summary"""
        
        print("\n==================================================")
        print("📊 REAL TRADING SESSION SUMMARY")
        print("==================================================")
        print(f"⏰ Uptime: {self.session_stats['uptime']}")
        print(f"🤖 Signals: {self.session_stats['signals_generated']}")
        print(f"📈 REAL Trades: {self.session_stats['trades_executed']}")
        
        if self.session_stats['real_trades']:
            total_value = sum(trade['value'] for trade in self.session_stats['real_trades'])
            print(f"💰 Total Trade Value: ${total_value:.2f}")
            
            print(f"📊 Trade Details:")
            for trade in self.session_stats['real_trades'][-5:]:  # Last 5 trades
                print(f"   {trade['timestamp'][:19]} | {trade['symbol']} | "
                      f"{trade['side']} | ${trade['value']:.2f}")
        
        if self.trading_api and self.trading_api.is_connected:
            current_balance = self.trading_api.get_total_balance()
            print(f"💵 Current Balance: ${current_balance:.2f}")
        
        print(f"📊 Symbols: {', '.join(self.config['symbols'])}")
        print("==================================================")
        print("✅ REAL Trading system stopped")
    
    def get_real_status(self) -> Dict[str, Any]:
        """📊 Get REAL system status"""
        
        status = {
            'system': 'REAL Live Trading - NO SIMULATION',
            'initialized': self.is_initialized,
            'running': self.is_running,
            'ai_ready': self.ai_models.is_ready if self.ai_models else False,
            'market_feed_connected': self.market_feed.exchange is not None if self.market_feed else False,
            'trading_api_connected': self.trading_api.is_connected if self.trading_api else False,
            'session_stats': self.session_stats,
            'config': self.config
        }
        
        if self.trading_api and self.trading_api.is_connected:
            status['account_balance'] = self.trading_api.get_total_balance()
        
        return status

def initialize_real_tradino_system(config: Dict[str, Any] = None) -> RealLiveTradingSystem:
    """🚀 Initialize REAL TRADINO system"""
    return RealLiveTradingSystem(config)

def start_real_tradino():
    """🚀 Start REAL TRADINO system"""
    
    print("🚀 TRADINO UNSCHLAGBAR - REAL SYSTEM")
    print("==================================================")
    print("🔧 Initializing REAL TRADINO...")
    
    system = initialize_real_tradino_system()
    
    if system.is_initialized:
        print("✅ REAL TRADINO ready!")
        print(f"📊 AI: {'Ready' if system.ai_models and system.ai_models.is_ready else 'Not Ready'}")
        print(f"📊 Market Feed: {'Connected' if system.market_feed and system.market_feed.exchange else 'Not Connected'}")
        print(f"🏦 Trading API: {'Connected' if system.trading_api and system.trading_api.is_connected else 'Not Connected'}")
        print(f"🎯 Mode: REAL MONEY TRADING")
        
        print(f"\n💡 To start: system.start_real_trading()")
        print(f"💡 To stop: system.stop_real_trading()")
        
        return system
    else:
        print("❌ REAL TRADINO initialization failed")
        return None

def stop_real_tradino():
    """🛑 Stop REAL TRADINO"""
    global system
    if 'system' in globals() and system:
        system.stop_real_trading()

if __name__ == "__main__":
    # Start REAL system
    system = start_real_tradino()
    
    if system:
        print(f"\n🚀 Starting REAL trading demo...")
        system.start_real_trading()
        
        # Run for demonstration
        time.sleep(30)  # 30 seconds of REAL trading
        
        system.stop_real_trading()
        
        print(f"\n🚀 REAL TRADINO ready for deployment!") 