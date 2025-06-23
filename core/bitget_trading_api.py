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
from decimal import Decimal, ROUND_DOWN
import warnings
warnings.filterwarnings('ignore')

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
        
        # Trading configuration
        self.config = {
            'default_type': 'swap',  # FUTURES TRADING!
            'max_position_size': 0.1,
            'min_order_size': 5,     # Lower for futures
            'slippage_tolerance': 0.002,
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
                'password': self.passphrase,
                'sandbox': self.sandbox,
                'rateLimit': 1200,
                'enableRateLimit': True,
                'options': {
                    'defaultType': self.config['default_type'],
                    'createMarketBuyOrderRequiresPrice': False  # For futures trading
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
            # Test with futures balance specifically
            balance = self.exchange.fetch_balance(params={'type': 'swap'})
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
            # Get futures balance specifically
            balance = self.exchange.fetch_balance(params={'type': 'swap'})
            
            self.account_info = {
                'balance': balance,
                'total_balance': balance.get('total', {}),
                'free_balance': balance.get('free', {}),
                'used_balance': balance.get('used', {}),
                'last_update': datetime.now()
            }
            
            print(f"💰 FUTURES Account info loaded: ${self.get_total_balance():.2f} total balance")
            
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
        
        free_balance = self.get_free_balance()
        position_value = free_balance * percentage
        
        if position_value < self.config['min_order_size']:
            return 0.0
        
        quantity = position_value / price
        quantity = float(Decimal(str(quantity)).quantize(
            Decimal('0.00001'), rounding=ROUND_DOWN
        ))
        
        return quantity
    
    def place_market_order(self, symbol: str, side: str, amount: float) -> Dict[str, Any]:
        """📈 Place market order"""
        
        if not self.is_connected:
            return {'error': 'API not connected'}
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            if amount <= 0:
                return {'error': 'Invalid amount'}
            
            order_value = amount * current_price
            if order_value < self.config['min_order_size']:
                return {'error': f'Order value too small: ${order_value:.2f}'}
            
            # Set exchange to futures mode for this order
            self.exchange.options['defaultType'] = 'swap'
            
            # For futures trading, we need to specify the order value in USDT
            if side == 'buy':
                # For buy orders, use the USD value instead of amount
                order_value = amount * current_price
                order = self.exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=order_value,
                    params={'type': 'swap', 'productType': 'UMCBL'}  # Explicitly specify futures
                )
            else:
                # For sell orders, use the amount  
                order = self.exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    params={'type': 'swap', 'productType': 'UMCBL'}  # Explicitly specify futures
                )
            
            # Reset to default after order
            self.exchange.options['defaultType'] = self.config['default_type']
            
            print(f"✅ Market order placed: {side} {amount} {symbol} at ~${current_price:.2f}")
            
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
    
    def execute_ai_signal(self, signal: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """🤖 Execute AI trading signal"""
        
        if not self.is_connected:
            return {'error': 'API not connected'}
        
        try:
            action = signal.get('action', 'hold')
            confidence = signal.get('confidence', 0)
            position_size = signal.get('position_size', 0)
            
            if action == 'hold' or confidence < 0.6:
                return {
                    'action': 'skipped',
                    'reason': f'Low confidence ({confidence:.2f}) or hold signal',
                    'signal': signal
                }
            
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            quantity = self.calculate_position_size(symbol, current_price, position_size)
            
            if quantity <= 0:
                return {
                    'action': 'skipped',
                    'reason': 'Insufficient balance or position size too small',
                    'signal': signal
                }
            
            if action == 'buy':
                result = self.place_market_order(symbol, 'buy', quantity)
            elif action == 'sell':
                result = self.place_market_order(symbol, 'sell', quantity)
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
            'last_update': None
        }
        
        if self.is_connected:
            summary['account_balance'] = {
                'total_usdt': self.get_total_balance(),
                'free_usdt': self.get_free_balance(),
                'currency_breakdown': self.account_info.get('total_balance', {})
            }
            
            if self.account_info and 'last_update' in self.account_info:
                summary['last_update'] = self.account_info['last_update'].isoformat()
        
        return summary
    
    def demo_test(self) -> Dict[str, Any]:
        """🧪 Demo trading functionality test"""
        
        print("🧪 BITGET TRADING API TEST")
        print("=" * 40)
        
        test_results = {
            'connection_test': False,
            'balance_fetch': False,
            'market_data': False
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
        
        return test_results

# Global instance
bitget_api = None

def initialize_bitget_trading(api_key: str = None, secret: str = None, 
                             passphrase: str = None, sandbox: bool = True) -> BitgetTradingAPI:
    """🚀 Initialize Bitget trading API"""
    
    global bitget_api
    bitget_api = BitgetTradingAPI(api_key, secret, passphrase, sandbox)
    return bitget_api

def get_bitget_api() -> Optional[BitgetTradingAPI]:
    """🏦 Get the global Bitget API instance"""
    return bitget_api

# Demo
if __name__ == "__main__":
    print("🏦 Bitget Trading API ready!")
    print("⚠️  Set API credentials for full functionality") 