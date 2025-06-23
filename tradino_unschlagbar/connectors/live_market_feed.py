#!/usr/bin/env python3
"""
📊 LIVE MARKET DATA FEED
Real-time market data integration für TRADINO UNSCHLAGBAR
"""

import os
import sys
import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import json
import time
import threading
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LiveMarketFeed:
    """📊 Live Market Data Feed Manager"""
    
    def __init__(self, exchange_name: str = 'binance'):
        self.exchange_name = exchange_name
        self.exchange = None
        self.is_running = False
        self.data_queue = Queue()
        self.subscribers = []
        self.current_data = {}
        self.last_update = {}
        
        # Configuration
        self.config = {
            'symbols': ['BTC/USDT', 'ETH/USDT'],
            'timeframe': '1m',
            'max_history': 500,  # Keep last 500 candles
            'update_interval': 30,  # Update every 30 seconds
            'retry_delay': 5
        }
        
        self.initialize_exchange()
        
        print(f"📊 Live Market Feed initialized with {exchange_name}")
    
    def initialize_exchange(self):
        """🔗 Initialize exchange connection"""
        
        try:
            # Initialize exchange (no API keys needed for public data)
            if self.exchange_name.lower() == 'binance':
                self.exchange = ccxt.binance({
                    'sandbox': False,
                    'rateLimit': 1200,
                    'enableRateLimit': True,
                })
            elif self.exchange_name.lower() == 'bitget':
                self.exchange = ccxt.bitget({
                    'sandbox': False,
                    'rateLimit': 1200,
                    'enableRateLimit': True,
                })
            else:
                self.exchange = ccxt.binance({
                    'sandbox': False,
                    'rateLimit': 1200,
                    'enableRateLimit': True,
                })
            
            # Test connection
            markets = self.exchange.load_markets()
            print(f"✅ Connected to {self.exchange_name}: {len(markets)} markets available")
            
            return True
            
        except Exception as e:
            print(f"❌ Exchange connection failed: {e}")
            self.exchange = None
            return False
    
    def subscribe_to_data(self, callback: Callable):
        """📝 Subscribe to live data updates"""
        
        if callback not in self.subscribers:
            self.subscribers.append(callback)
            print(f"✅ Subscriber added: {len(self.subscribers)} total")
    
    def unsubscribe_from_data(self, callback: Callable):
        """📝 Unsubscribe from data updates"""
        
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            print(f"✅ Subscriber removed: {len(self.subscribers)} total")
    
    def fetch_ohlcv_data(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """📊 Fetch OHLCV data for symbol"""
        
        if not self.exchange:
            raise Exception("Exchange not initialized")
        
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                self.config['timeframe'], 
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            
            # Clean data
            df = df.drop('timestamp', axis=1)
            df = df.reindex(columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol'])
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_ticker_data(self, symbol: str) -> Dict[str, Any]:
        """💹 Fetch real-time ticker data"""
        
        if not self.exchange:
            return {}
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Extract relevant data
            ticker_data = {
                'symbol': symbol,
                'last_price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'change_24h': ticker['change'],
                'change_24h_percent': ticker['percentage'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low'],
                'timestamp': datetime.now().isoformat()
            }
            
            return ticker_data
            
        except Exception as e:
            print(f"❌ Error fetching ticker for {symbol}: {e}")
            return {}
    
    def update_market_data(self):
        """🔄 Update market data for all symbols"""
        
        updated_symbols = []
        
        for symbol in self.config['symbols']:
            try:
                # Fetch OHLCV data
                ohlcv_data = self.fetch_ohlcv_data(symbol, self.config['max_history'])
                
                if not ohlcv_data.empty:
                    self.current_data[symbol] = {
                        'ohlcv': ohlcv_data,
                        'last_update': datetime.now(),
                        'data_points': len(ohlcv_data)
                    }
                    
                    # Fetch ticker data
                    ticker_data = self.fetch_ticker_data(symbol)
                    if ticker_data:
                        self.current_data[symbol]['ticker'] = ticker_data
                    
                    updated_symbols.append(symbol)
                    self.last_update[symbol] = datetime.now()
                    
                # Small delay to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"⚠️ Error updating {symbol}: {e}")
                continue
        
        # Notify subscribers
        if updated_symbols:
            self.notify_subscribers(updated_symbols)
        
        return updated_symbols
    
    def notify_subscribers(self, updated_symbols: List[str]):
        """📢 Notify all subscribers of data updates"""
        
        for callback in self.subscribers:
            try:
                callback(self.current_data, updated_symbols)
            except Exception as e:
                print(f"⚠️ Error notifying subscriber: {e}")
    
    def start_live_feed(self):
        """🚀 Start live data feed"""
        
        if not self.exchange:
            print("❌ Cannot start feed - exchange not initialized")
            return False
        
        if self.is_running:
            print("⚠️ Live feed already running")
            return True
        
        self.is_running = True
        
        def feed_loop():
            print("🚀 Live market feed started")
            
            while self.is_running:
                try:
                    # Update market data
                    updated = self.update_market_data()
                    
                    if updated:
                        print(f"📊 Updated data for: {', '.join(updated)}")
                    
                    # Wait for next update
                    time.sleep(self.config['update_interval'])
                    
                except Exception as e:
                    print(f"❌ Feed error: {e}")
                    time.sleep(self.config['retry_delay'])
                    continue
            
            print("🛑 Live market feed stopped")
        
        # Start feed in background thread
        feed_thread = threading.Thread(target=feed_loop, daemon=True)
        feed_thread.start()
        
        print(f"✅ Live feed started for {len(self.config['symbols'])} symbols")
        return True
    
    def stop_live_feed(self):
        """🛑 Stop live data feed"""
        
        self.is_running = False
        print("🛑 Live feed stop requested")
    
    def get_latest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """📊 Get latest market data for symbol"""
        
        if symbol in self.current_data:
            return self.current_data[symbol]['ohlcv']
        return None
    
    def get_latest_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """💹 Get latest ticker data for symbol"""
        
        if symbol in self.current_data and 'ticker' in self.current_data[symbol]:
            return self.current_data[symbol]['ticker']
        return None
    
    def is_data_fresh(self, symbol: str, max_age_minutes: int = 5) -> bool:
        """⏰ Check if data is fresh enough"""
        
        if symbol not in self.last_update:
            return False
        
        age = datetime.now() - self.last_update[symbol]
        return age.total_seconds() < (max_age_minutes * 60)
    
    def get_market_summary(self) -> Dict[str, Any]:
        """📈 Get market summary"""
        
        summary = {
            'exchange': self.exchange_name,
            'symbols_tracked': len(self.config['symbols']),
            'data_status': {},
            'last_update': None,
            'feed_running': self.is_running
        }
        
        for symbol in self.config['symbols']:
            if symbol in self.current_data:
                data_info = self.current_data[symbol]
                summary['data_status'][symbol] = {
                    'data_points': data_info['data_points'],
                    'last_update': data_info['last_update'].isoformat(),
                    'is_fresh': self.is_data_fresh(symbol)
                }
                
                if 'ticker' in data_info:
                    summary['data_status'][symbol]['last_price'] = data_info['ticker']['last_price']
                    summary['data_status'][symbol]['change_24h'] = data_info['ticker']['change_24h_percent']
        
        if self.last_update:
            latest_update = max(self.last_update.values())
            summary['last_update'] = latest_update.isoformat()
        
        return summary
    
    def test_connection(self) -> Dict[str, Any]:
        """🧪 Test market feed connection"""
        
        test_results = {
            'exchange_connected': False,
            'symbols_accessible': [],
            'sample_data': {},
            'latency_ms': None
        }
        
        try:
            # Test exchange connection
            start_time = time.time()
            markets = self.exchange.load_markets()
            latency = (time.time() - start_time) * 1000
            
            test_results['exchange_connected'] = True
            test_results['latency_ms'] = round(latency, 2)
            
            # Test symbol access
            for symbol in self.config['symbols'][:2]:  # Test first 2 symbols
                try:
                    data = self.fetch_ohlcv_data(symbol, 10)
                    if not data.empty:
                        test_results['symbols_accessible'].append(symbol)
                        test_results['sample_data'][symbol] = {
                            'latest_close': float(data['Close'].iloc[-1]),
                            'data_points': len(data),
                            'timespan': f"{data['Datetime'].iloc[0]} to {data['Datetime'].iloc[-1]}"
                        }
                except Exception as e:
                    print(f"⚠️ Symbol test failed for {symbol}: {e}")
            
            print("✅ Connection test completed")
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            test_results['error'] = str(e)
        
        return test_results

# Global instance for easy access
live_feed = LiveMarketFeed()

def start_market_feed() -> bool:
    """🚀 Start the global market feed"""
    return live_feed.start_live_feed()

def stop_market_feed():
    """🛑 Stop the global market feed"""
    live_feed.stop_live_feed()

def get_live_data(symbol: str) -> Optional[pd.DataFrame]:
    """📊 Get live market data"""
    return live_feed.get_latest_data(symbol)

def subscribe_to_live_data(callback: Callable):
    """📝 Subscribe to live data updates"""
    live_feed.subscribe_to_data(callback)

# Demo and testing
if __name__ == "__main__":
    print("📊 LIVE MARKET FEED TEST")
    print("=" * 50)
    
    # Test connection
    test_results = live_feed.test_connection()
    print(f"Connection Test: {test_results}")
    
    # Test data fetch
    if test_results['exchange_connected']:
        print("\n🔄 Testing data fetch...")
        btc_data = live_feed.fetch_ohlcv_data('BTC/USDT', 10)
        if not btc_data.empty:
            print(f"✅ BTC data: {len(btc_data)} candles")
            print(f"Latest price: ${btc_data['Close'].iloc[-1]:,.2f}")
        
        # Test ticker
        ticker = live_feed.fetch_ticker_data('BTC/USDT')
        if ticker:
            print(f"✅ BTC ticker: ${ticker['last_price']:,.2f} ({ticker['change_24h_percent']:+.2f}%)")
    
    print("\n🚀 Live Market Feed ready for integration!") 