#!/usr/bin/env python3
"""
🏦 BITGET TRADING API INTEGRATION
Real trading execution für TRADINO UNSCHLAGBAR
"""

import os
import sys
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import time
import asyncio
from decimal import Decimal, ROUND_DOWN
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BitgetTradingAPI:
    """🏦 Bitget Trading API Manager"""
    
    def __init__(self, api_key: str = None, secret: str = None, passphrase: str = None, 
                 sandbox: bool = True):
        self.api_key = api_key
        self.secret = secret
        self.passphrase = passphrase
        self.sandbox = sandbox
        self.exchange = None
        self.is_connected = False
        self.account_info = {}
        self.positions = {}
        self.open_orders = {}
        
        # Trading configuration
        self.config = {
            'default_type': 'spot',  # spot, future
            'max_position_size': 0.1,  # Max 10% of account per position
            'min_order_size': 10,  # Minimum $10 order
            'slippage_tolerance': 0.002,  # 0.2% slippage tolerance
            'retry_attempts': 3,
            'retry_delay': 2
        }
        
        if api_key and secret and passphrase:
            self.initialize_api()
        
        print(f"🏦 Bitget Trading API initialized ({'Sandbox' if sandbox else 'Live'})")
    
    def initialize_api(self):
        """🔗 Initialize Bitget API connection"""
        
        try:
            self.exchange = ccxt.bitget({
                'apiKey': self.api_key,
                'secret': self.secret,
                'password': self.passphrase,  # Bitget uses passphrase
                'sandbox': self.sandbox,
                'rateLimit': 1200,
                'enableRateLimit': True,
                'options': {
                    'defaultType': self.config['default_type']
                }
            })
            
            # Test connection
            self.test_connection()
            
            if self.is_connected:
                print("✅ Bitget API connected successfully")
                self.load_account_info()
            
            return self.is_connected
            
        except Exception as e:
            print(f"❌ Bitget API initialization failed: {e}")
            self.exchange = None
            self.is_connected = False
            return False
    
    def test_connection(self) -> bool:
        """🧪 Test API connection"""
        
        if not self.exchange:
            return False
        
        try:
            # Test with account balance request
            balance = self.exchange.fetch_balance()
            self.is_connected = True
            print("✅ API connection test successful")
            return True
            
        except Exception as e:
            print(f"❌ API connection test failed: {e}")
            self.is_connected = False
            return False
    
    def load_account_info(self):
        """💰 Load account information"""
        
        if not self.is_connected:
            return
        
        try:
            # Fetch balance
            balance = self.exchange.fetch_balance()
            
            # Fetch positions (if future trading)
            positions = {}
            try:
                if self.config['default_type'] == 'future':
                    pos_data = self.exchange.fetch_positions()
                    for pos in pos_data:
                        if pos['contracts'] > 0:  # Open position
                            positions[pos['symbol']] = pos
            except:
                pass  # Positions might not be available for spot
            
            self.account_info = {
                'balance': balance,
                'total_balance': balance.get('total', {}),
                'free_balance': balance.get('free', {}),
                'used_balance': balance.get('used', {}),
                'positions': positions,
                'last_update': datetime.now()
            }
            
            print(f"💰 Account info loaded: ${self.get_total_balance():.2f} total balance")
            
        except Exception as e:
            print(f"❌ Error loading account info: {e}")
    
    def get_total_balance(self, currency: str = 'USDT') -> float:
        """💰 Get total account balance"""
        
        if not self.account_info or 'total_balance' not in self.account_info:
            return 0.0
        
        return float(self.account_info['total_balance'].get(currency, 0))
    
    def get_free_balance(self, currency: str = 'USDT') -> float:
        """💵 Get free balance for trading"""
        
        if not self.account_info or 'free_balance' not in self.account_info:
            return 0.0
        
        return float(self.account_info['free_balance'].get(currency, 0))
    
    def calculate_position_size(self, symbol: str, price: float, 
                              percentage: float = None) -> float:
        """📊 Calculate position size based on risk management"""
        
        if percentage is None:
            percentage = self.config['max_position_size']
        
        # Get available balance
        free_balance = self.get_free_balance()
        
        # Calculate position value
        position_value = free_balance * percentage
        
        # Ensure minimum order size
        if position_value < self.config['min_order_size']:
            return 0.0
        
        # Calculate quantity
        quantity = position_value / price
        
        # Round down to avoid insufficient balance
        quantity = float(Decimal(str(quantity)).quantize(
            Decimal('0.00001'), rounding=ROUND_DOWN
        ))
        
        return quantity
    
    def place_market_order(self, symbol: str, side: str, amount: float, 
                          reduce_only: bool = False) -> Dict[str, Any]:
        """📈 Place market order"""
        
        if not self.is_connected:
            return {'error': 'API not connected'}
        
        try:
            # Get current price for validation
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Validate amount
            if amount <= 0:
                return {'error': 'Invalid amount'}
            
            # Check minimum order value
            order_value = amount * current_price
            if order_value < self.config['min_order_size']:
                return {'error': f'Order value too small: ${order_value:.2f}'}
            
            # Prepare order parameters
            params = {}
            if reduce_only:
                params['reduceOnly'] = True
            
            # Place order
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount,
                params=params
            )
            
            print(f"✅ Market order placed: {side} {amount} {symbol} at ~${current_price:.2f}")
            
            # Update account info
            self.load_account_info()
            
            return {
                'success': True,
                'order': order,
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'estimated_price': current_price,
                'estimated_value': order_value,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Market order failed: {e}"
            print(f"❌ {error_msg}")
            
            return {
                'error': error_msg,
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'timestamp': datetime.now().isoformat()
            }
    
    def place_limit_order(self, symbol: str, side: str, amount: float, price: float,
                         time_in_force: str = 'GTC') -> Dict[str, Any]:
        """📊 Place limit order"""
        
        if not self.is_connected:
            return {'error': 'API not connected'}
        
        try:
            # Validate parameters
            if amount <= 0 or price <= 0:
                return {'error': 'Invalid amount or price'}
            
            order_value = amount * price
            if order_value < self.config['min_order_size']:
                return {'error': f'Order value too small: ${order_value:.2f}'}
            
            # Place limit order
            order = self.exchange.create_limit_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                params={'timeInForce': time_in_force}
            )
            
            print(f"✅ Limit order placed: {side} {amount} {symbol} at ${price:.2f}")
            
            return {
                'success': True,
                'order': order,
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'order_value': order_value,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Limit order failed: {e}"
            print(f"❌ {error_msg}")
            
            return {
                'error': error_msg,
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'timestamp': datetime.now().isoformat()
            }
    
    def place_stop_loss_order(self, symbol: str, side: str, amount: float, 
                             stop_price: float, limit_price: float = None) -> Dict[str, Any]:
        """🛡️ Place stop-loss order"""
        
        if not self.is_connected:
            return {'error': 'API not connected'}
        
        try:
            # If no limit price, use stop price
            if limit_price is None:
                limit_price = stop_price * (0.995 if side == 'sell' else 1.005)
            
            # Place stop order
            order = self.exchange.create_order(
                symbol=symbol,
                type='stop',
                side=side,
                amount=amount,
                price=limit_price,
                params={
                    'stopPrice': stop_price,
                    'timeInForce': 'GTC'
                }
            )
            
            print(f"✅ Stop-loss order placed: {side} {amount} {symbol} stop@${stop_price:.2f}")
            
            return {
                'success': True,
                'order': order,
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'stop_price': stop_price,
                'limit_price': limit_price,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Stop-loss order failed: {e}"
            print(f"❌ {error_msg}")
            
            return {
                'error': error_msg,
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'stop_price': stop_price,
                'timestamp': datetime.now().isoformat()
            }
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """❌ Cancel open order"""
        
        if not self.is_connected:
            return {'error': 'API not connected'}
        
        try:
            result = self.exchange.cancel_order(order_id, symbol)
            
            print(f"✅ Order cancelled: {order_id}")
            
            return {
                'success': True,
                'cancelled_order': result,
                'order_id': order_id,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Cancel order failed: {e}"
            print(f"❌ {error_msg}")
            
            return {
                'error': error_msg,
                'order_id': order_id,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """📋 Get open orders"""
        
        if not self.is_connected:
            return []
        
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            
            # Format orders
            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    'id': order['id'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'amount': order['amount'],
                    'price': order.get('price'),
                    'status': order['status'],
                    'timestamp': order['timestamp'],
                    'datetime': order['datetime']
                })
            
            self.open_orders = {o['symbol']: o for o in formatted_orders}
            
            return formatted_orders
            
        except Exception as e:
            print(f"❌ Error fetching open orders: {e}")
            return []
    
    def get_order_history(self, symbol: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """📚 Get order history"""
        
        if not self.is_connected:
            return []
        
        try:
            orders = self.exchange.fetch_orders(symbol, limit=limit)
            
            # Format orders
            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    'id': order['id'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'amount': order['amount'],
                    'filled': order['filled'],
                    'price': order.get('price'),
                    'average': order.get('average'),
                    'status': order['status'],
                    'fee': order.get('fee'),
                    'timestamp': order['timestamp'],
                    'datetime': order['datetime']
                })
            
            return formatted_orders
            
        except Exception as e:
            print(f"❌ Error fetching order history: {e}")
            return []
    
    def get_positions(self, symbol: str = None) -> Dict[str, Any]:
        """📊 Get current positions"""
        
        if not self.is_connected:
            return {}
        
        try:
            if self.config['default_type'] == 'spot':
                # For spot trading, use balance as positions
                balance = self.exchange.fetch_balance()
                positions = {}
                
                for currency, amount in balance['total'].items():
                    if amount > 0 and currency != 'USDT':
                        positions[f"{currency}/USDT"] = {
                            'symbol': f"{currency}/USDT",
                            'side': 'long',
                            'size': amount,
                            'value': amount,  # Would need current price for exact value
                            'currency': currency
                        }
                
                return positions
                
            else:
                # For futures trading
                positions = self.exchange.fetch_positions(symbol)
                return {pos['symbol']: pos for pos in positions if pos['contracts'] > 0}
                
        except Exception as e:
            print(f"❌ Error fetching positions: {e}")
            return {}
    
    def execute_ai_signal(self, signal: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """🤖 Execute AI trading signal"""
        
        if not self.is_connected:
            return {'error': 'API not connected'}
        
        try:
            action = signal.get('action', 'hold')
            confidence = signal.get('confidence', 0)
            position_size = signal.get('position_size', 0)
            
            # Skip if confidence too low or action is hold
            if action == 'hold' or confidence < 0.6:
                return {
                    'action': 'skipped',
                    'reason': f'Low confidence ({confidence:.2f}) or hold signal',
                    'signal': signal
                }
            
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate position size
            quantity = self.calculate_position_size(symbol, current_price, position_size)
            
            if quantity <= 0:
                return {
                    'action': 'skipped',
                    'reason': 'Insufficient balance or position size too small',
                    'signal': signal
                }
            
            # Execute order based on action
            if action == 'buy':
                result = self.place_market_order(symbol, 'buy', quantity)
            elif action == 'sell':
                # First check if we have position to sell
                positions = self.get_positions(symbol)
                if symbol in positions:
                    # Sell existing position
                    pos_size = positions[symbol]['size']
                    sell_quantity = min(quantity, pos_size)
                    result = self.place_market_order(symbol, 'sell', sell_quantity)
                else:
                    return {
                        'action': 'skipped',
                        'reason': 'No position to sell',
                        'signal': signal
                    }
            else:
                return {
                    'error': f'Unknown action: {action}',
                    'signal': signal
                }
            
            return {
                'success': True,
                'executed_action': action,
                'order_result': result,
                'signal_used': signal,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"AI signal execution failed: {e}"
            print(f"❌ {error_msg}")
            
            return {
                'error': error_msg,
                'signal': signal,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """📊 Get trading account summary"""
        
        summary = {
            'connected': self.is_connected,
            'exchange': 'Bitget',
            'trading_mode': 'Sandbox' if self.sandbox else 'Live',
            'account_balance': {},
            'open_positions': {},
            'open_orders_count': 0,
            'last_update': None
        }
        
        if self.is_connected:
            # Account balance
            summary['account_balance'] = {
                'total_usdt': self.get_total_balance(),
                'free_usdt': self.get_free_balance(),
                'currency_breakdown': self.account_info.get('total_balance', {})
            }
            
            # Positions
            positions = self.get_positions()
            summary['open_positions'] = positions
            summary['positions_count'] = len(positions)
            
            # Open orders
            open_orders = self.get_open_orders()
            summary['open_orders_count'] = len(open_orders)
            
            # Last update
            if self.account_info and 'last_update' in self.account_info:
                summary['last_update'] = self.account_info['last_update'].isoformat()
        
        return summary
    
    def demo_trading_test(self) -> Dict[str, Any]:
        """🧪 Demo trading functionality test"""
        
        print("🧪 BITGET TRADING API TEST")
        print("=" * 50)
        
        test_results = {
            'connection_test': False,
            'balance_fetch': False,
            'market_data': False,
            'order_simulation': False
        }
        
        # Test connection
        if self.test_connection():
            test_results['connection_test'] = True
            print("✅ Connection test passed")
        
        # Test balance fetch
        try:
            self.load_account_info()
            balance = self.get_total_balance()
            test_results['balance_fetch'] = True
            test_results['total_balance'] = balance
            print(f"✅ Balance fetch: ${balance:.2f}")
        except Exception as e:
            print(f"❌ Balance fetch failed: {e}")
        
        # Test market data
        try:
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            test_results['market_data'] = True
            test_results['btc_price'] = ticker['last']
            print(f"✅ Market data: BTC ${ticker['last']:,.2f}")
        except Exception as e:
            print(f"❌ Market data failed: {e}")
        
        # Simulate order (without actually placing)
        try:
            btc_price = test_results.get('btc_price', 50000)
            position_size = self.calculate_position_size('BTC/USDT', btc_price, 0.01)
            
            if position_size > 0:
                test_results['order_simulation'] = True
                test_results['simulated_order'] = {
                    'symbol': 'BTC/USDT',
                    'quantity': position_size,
                    'value': position_size * btc_price
                }
                print(f"✅ Order simulation: {position_size:.6f} BTC (${position_size * btc_price:.2f})")
            else:
                print("⚠️ Insufficient balance for order simulation")
                
        except Exception as e:
            print(f"❌ Order simulation failed: {e}")
        
        return test_results

# Configuration management
def load_bitget_config() -> Dict[str, str]:
    """📋 Load Bitget API configuration"""
    
    # Try to load from environment variables first
    config = {
        'api_key': os.getenv('BITGET_API_KEY'),
        'secret': os.getenv('BITGET_SECRET'),
        'passphrase': os.getenv('BITGET_PASSPHRASE'),
        'sandbox': os.getenv('BITGET_SANDBOX', 'true').lower() == 'true'
    }
    
    # Try to load from config file
    config_file = 'tradino_unschlagbar/config/bitget_config.json'
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                for key, value in file_config.items():
                    if value and not config[key]:
                        config[key] = value
        except Exception as e:
            print(f"⚠️ Error loading config file: {e}")
    
    return config

# Global instance
bitget_api = None

def initialize_bitget_trading(api_key: str = None, secret: str = None, 
                             passphrase: str = None, sandbox: bool = True) -> BitgetTradingAPI:
    """🚀 Initialize Bitget trading API"""
    
    global bitget_api
    
    # Load config if not provided
    if not api_key:
        config = load_bitget_config()
        api_key = config['api_key']
        secret = config['secret']
        passphrase = config['passphrase']
        sandbox = config['sandbox']
    
    bitget_api = BitgetTradingAPI(api_key, secret, passphrase, sandbox)
    return bitget_api

def get_bitget_api() -> Optional[BitgetTradingAPI]:
    """🏦 Get the global Bitget API instance"""
    return bitget_api

# Demo
if __name__ == "__main__":
    # Demo without real credentials
    demo_api = BitgetTradingAPI(sandbox=True)
    
    if demo_api.exchange:
        results = demo_api.demo_trading_test()
        print(f"\n📊 Test Results: {results}")
    else:
        print("⚠️ Demo requires API credentials for full testing")
        print("Set BITGET_API_KEY, BITGET_SECRET, BITGET_PASSPHRASE environment variables")