#!/usr/bin/env python3
"""
🏦 BITGET TRADING API INTEGRATION
Real trading execution für TRADINO UNSCHLAGBAR
Mit automatischem Take Profit / Stop Loss System
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
    """🏦 Bitget Trading API Manager mit TP/SL System"""
    
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
            'retry_delay': 2,
            'default_tp_percent': 0.03,  # 3% Take Profit
            'default_sl_percent': 0.015,  # 1.5% Stop Loss
            'oco_enabled': True,
            'monitoring_enabled': True
        }
        
        # TP/SL Management
        self.active_orders = {}
        self.tp_sl_orders = {}
        self.monitoring_positions = {}
        
        if api_key and secret and passphrase:
            self.initialize_api()
        
        print(f"🏦 Bitget Trading API mit TP/SL System initialized ({'Sandbox' if sandbox else 'Live'})")
    
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
    
    def place_market_order_with_tp_sl(self, symbol: str, side: str, amount: float, 
                                     tp_percent: float = None, sl_percent: float = None,
                                     risk_management_params: Dict = None) -> Dict[str, Any]:
        """📈 Place market order mit automatischem TP/SL"""
        
        if not self.is_connected:
            return {'error': 'API not connected'}
        
        try:
            # Zunächst die Market Order platzieren
            market_order_result = self.place_market_order(symbol, side, amount)
            
            if not market_order_result.get('success'):
                return market_order_result
            
            # Entry Price aus der Order
            entry_price = market_order_result.get('estimated_price')
            order_id = market_order_result.get('order', {}).get('id')
            
            # TP/SL Parameter berechnen
            tp_sl_params = self._calculate_tp_sl_prices(
                entry_price, side, tp_percent, sl_percent, risk_management_params
            )
            
            # TP/SL Orders platzieren
            tp_sl_result = self._place_tp_sl_orders(
                symbol, side, amount, entry_price, tp_sl_params, order_id
            )
            
            # Vollständiges Ergebnis
            complete_result = {
                'success': True,
                'market_order': market_order_result,
                'tp_sl_orders': tp_sl_result,
                'entry_price': entry_price,
                'tp_price': tp_sl_params.get('tp_price'),
                'sl_price': tp_sl_params.get('sl_price'),
                'order_group_id': f"trade_{int(time.time())}",
                'timestamp': datetime.now().isoformat()
            }
            
            # Store für Monitoring
            self.monitoring_positions[order_id] = {
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'entry_price': entry_price,
                'tp_price': tp_sl_params.get('tp_price'),
                'sl_price': tp_sl_params.get('sl_price'),
                'tp_order_id': tp_sl_result.get('tp_order_id'),
                'sl_order_id': tp_sl_result.get('sl_order_id'),
                'status': 'active',
                'created_at': datetime.now()
            }
            
            print(f"✅ Market Order mit TP/SL platziert: {side} {amount} {symbol}")
            print(f"   Entry: ${entry_price:.4f} | TP: ${tp_sl_params.get('tp_price', 0):.4f} | SL: ${tp_sl_params.get('sl_price', 0):.4f}")
            
            return complete_result
            
        except Exception as e:
            error_msg = f"Market order mit TP/SL failed: {e}"
            print(f"❌ {error_msg}")
            return {'error': error_msg, 'timestamp': datetime.now().isoformat()}
    
    def _calculate_tp_sl_prices(self, entry_price: float, side: str, 
                               tp_percent: float = None, sl_percent: float = None,
                               risk_management_params: Dict = None) -> Dict[str, float]:
        """📊 Berechne TP/SL Preise basierend auf Risk Management"""
        
        # Default Werte oder aus Risk Management
        if risk_management_params:
            tp_percent = tp_percent or risk_management_params.get('take_profit_percent', self.config['default_tp_percent'])
            sl_percent = sl_percent or risk_management_params.get('stop_loss_percent', self.config['default_sl_percent'])
        else:
            tp_percent = tp_percent or self.config['default_tp_percent']
            sl_percent = sl_percent or self.config['default_sl_percent']
        
        if side.lower() == 'buy':
            # Long Position
            tp_price = entry_price * (1 + tp_percent)
            sl_price = entry_price * (1 - sl_percent)
        else:
            # Short Position
            tp_price = entry_price * (1 - tp_percent)
            sl_price = entry_price * (1 + sl_percent)
        
        return {
            'tp_price': round(tp_price, 4),
            'sl_price': round(sl_price, 4),
            'tp_percent': tp_percent,
            'sl_percent': sl_percent
        }
    
    def _place_tp_sl_orders(self, symbol: str, side: str, amount: float, 
                           entry_price: float, tp_sl_params: Dict, main_order_id: str) -> Dict[str, Any]:
        """🎯 Platziere TP/SL Orders (OCO wenn verfügbar, sonst separat)"""
        
        result = {
            'method': None,
            'tp_order_id': None,
            'sl_order_id': None,
            'success': False,
            'errors': []
        }
        
        try:
            # Versuche OCO Order (One-Cancels-Other) zuerst
            if self.config['oco_enabled']:
                oco_result = self._place_oco_order(symbol, side, amount, tp_sl_params)
                if oco_result.get('success'):
                    result.update(oco_result)
                    result['method'] = 'OCO'
                    return result
                else:
                    result['errors'].append(f"OCO failed: {oco_result.get('error', 'Unknown')}")
            
            # Fallback: Separate TP/SL Orders
            separate_result = self._place_separate_tp_sl_orders(symbol, side, amount, tp_sl_params)
            result.update(separate_result)
            result['method'] = 'Separate Orders'
            
            return result
            
        except Exception as e:
            result['errors'].append(f"TP/SL placement error: {e}")
            return result
    
    def _place_oco_order(self, symbol: str, side: str, amount: float, tp_sl_params: Dict) -> Dict[str, Any]:
        """🔄 Platziere OCO Order (Bitget spezifisch)"""
        
        try:
            # Bitget OCO Parameter
            opposite_side = 'sell' if side == 'buy' else 'buy'
            
            # OCO Order mit Bitget API
            oco_order = self.exchange.create_order(
                symbol=symbol,
                type='OCO',
                side=opposite_side,  # Exit side
                amount=amount,
                price=tp_sl_params['tp_price'],  # Take Profit Price
                params={
                    'stopPrice': tp_sl_params['sl_price'],  # Stop Loss Price
                    'stopLimitPrice': tp_sl_params['sl_price'] * 0.995,  # Stop Limit (slight buffer)
                    'timeInForce': 'GTC',
                    'type': 'swap',
                    'productType': 'UMCBL'
                }
            )
            
            return {
                'success': True,
                'oco_order_id': oco_order.get('id'),
                'tp_price': tp_sl_params['tp_price'],
                'sl_price': tp_sl_params['sl_price']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _place_separate_tp_sl_orders(self, symbol: str, side: str, amount: float, 
                                   tp_sl_params: Dict) -> Dict[str, Any]:
        """📋 Platziere separate TP und SL Orders"""
        
        result = {
            'success': False,
            'tp_order_id': None,
            'sl_order_id': None,
            'errors': []
        }
        
        opposite_side = 'sell' if side == 'buy' else 'buy'
        
        try:
            # Take Profit Order (Limit Order)
            tp_order = self.exchange.create_limit_order(
                symbol=symbol,
                side=opposite_side,
                amount=amount,
                price=tp_sl_params['tp_price'],
                params={
                    'timeInForce': 'GTC',
                    'type': 'swap',
                    'productType': 'UMCBL'
                }
            )
            
            result['tp_order_id'] = tp_order.get('id')
            print(f"✅ Take Profit Order: ${tp_sl_params['tp_price']:.4f}")
            
        except Exception as e:
            result['errors'].append(f"TP Order failed: {e}")
        
        try:
            # Stop Loss Order (Stop Market Order)
            sl_order = self.exchange.create_order(
                symbol=symbol,
                type='stop_market',
                side=opposite_side,
                amount=amount,
                params={
                    'stopPrice': tp_sl_params['sl_price'],
                    'timeInForce': 'GTC',
                    'type': 'swap',
                    'productType': 'UMCBL'
                }
            )
            
            result['sl_order_id'] = sl_order.get('id')
            print(f"✅ Stop Loss Order: ${tp_sl_params['sl_price']:.4f}")
            
        except Exception as e:
            result['errors'].append(f"SL Order failed: {e}")
        
        # Success wenn mindestens eine Order erfolgreich
        result['success'] = bool(result['tp_order_id'] or result['sl_order_id'])
        
        return result
    
    def monitor_tp_sl_orders(self) -> Dict[str, Any]:
        """👁️ Überwache aktive TP/SL Orders"""
        
        monitoring_result = {
            'active_positions': len(self.monitoring_positions),
            'executed_orders': [],
            'failed_orders': [],
            'status': 'monitoring'
        }
        
        if not self.monitoring_positions:
            return monitoring_result
        
        try:
            for position_id, position_data in list(self.monitoring_positions.items()):
                if position_data['status'] != 'active':
                    continue
                
                symbol = position_data['symbol']
                
                # Check TP Order Status
                if position_data.get('tp_order_id'):
                    tp_status = self._check_order_status(position_data['tp_order_id'], symbol)
                    if tp_status.get('filled'):
                        monitoring_result['executed_orders'].append({
                            'type': 'take_profit',
                            'symbol': symbol,
                            'price': position_data['tp_price'],
                            'order_id': position_data['tp_order_id']
                        })
                        position_data['status'] = 'tp_executed'
                
                # Check SL Order Status
                if position_data.get('sl_order_id'):
                    sl_status = self._check_order_status(position_data['sl_order_id'], symbol)
                    if sl_status.get('filled'):
                        monitoring_result['executed_orders'].append({
                            'type': 'stop_loss',
                            'symbol': symbol,
                            'price': position_data['sl_price'],
                            'order_id': position_data['sl_order_id']
                        })
                        position_data['status'] = 'sl_executed'
                
                # Fallback Monitoring für Orders ohne API Bestätigung
                if not position_data.get('tp_order_id') and not position_data.get('sl_order_id'):
                    manual_result = self._manual_tp_sl_monitoring(position_data)
                    if manual_result.get('triggered'):
                        monitoring_result['executed_orders'].append(manual_result)
                        position_data['status'] = 'manually_executed'
            
            return monitoring_result
            
        except Exception as e:
            monitoring_result['error'] = f"Monitoring error: {e}"
            return monitoring_result
    
    def _check_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """📋 Check Order Status via API"""
        
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            return {
                'filled': order.get('status') == 'closed',
                'status': order.get('status'),
                'filled_amount': order.get('filled', 0),
                'average_price': order.get('average')
            }
        except Exception as e:
            return {'error': f"Order status check failed: {e}"}
    
    def _manual_tp_sl_monitoring(self, position_data: Dict) -> Dict[str, Any]:
        """👀 Manuelles Monitoring als Fallback"""
        
        try:
            symbol = position_data['symbol']
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            side = position_data['side']
            tp_price = position_data.get('tp_price')
            sl_price = position_data.get('sl_price')
            
            # Check TP/SL Trigger Conditions
            if side == 'buy':
                # Long Position
                if tp_price and current_price >= tp_price:
                    return self._execute_manual_tp_sl(position_data, 'take_profit', current_price)
                elif sl_price and current_price <= sl_price:
                    return self._execute_manual_tp_sl(position_data, 'stop_loss', current_price)
            else:
                # Short Position  
                if tp_price and current_price <= tp_price:
                    return self._execute_manual_tp_sl(position_data, 'take_profit', current_price)
                elif sl_price and current_price >= sl_price:
                    return self._execute_manual_tp_sl(position_data, 'stop_loss', current_price)
            
            return {'triggered': False}
            
        except Exception as e:
            return {'error': f"Manual monitoring failed: {e}"}
    
    def _execute_manual_tp_sl(self, position_data: Dict, order_type: str, current_price: float) -> Dict[str, Any]:
        """⚡ Execute manual TP/SL Order"""
        
        try:
            symbol = position_data['symbol']
            side = position_data['side']
            amount = position_data['amount']
            
            # Opposite side für Exit
            exit_side = 'sell' if side == 'buy' else 'buy'
            
            # Execute Market Order für sofortigen Exit
            exit_order = self.place_market_order(symbol, exit_side, amount)
            
            if exit_order.get('success'):
                return {
                    'triggered': True,
                    'type': order_type,
                    'symbol': symbol,
                    'exit_price': current_price,
                    'target_price': position_data.get('tp_price' if order_type == 'take_profit' else 'sl_price'),
                    'exit_order': exit_order,
                    'method': 'manual_execution'
                }
            else:
                return {'error': f"Manual {order_type} execution failed"}
                
        except Exception as e:
            return {'error': f"Manual TP/SL execution error: {e}"}
    
    def get_tp_sl_status(self) -> Dict[str, Any]:
        """📊 Status aller TP/SL Positionen"""
        
        status = {
            'total_positions': len(self.monitoring_positions),
            'active_positions': 0,
            'completed_positions': 0,
            'positions': []
        }
        
        for position_id, position_data in self.monitoring_positions.items():
            position_status = {
                'position_id': position_id,
                'symbol': position_data['symbol'],
                'side': position_data['side'],
                'entry_price': position_data['entry_price'],
                'tp_price': position_data.get('tp_price'),
                'sl_price': position_data.get('sl_price'),
                'status': position_data['status'],
                'created_at': position_data['created_at'].isoformat()
            }
            
            if position_data['status'] == 'active':
                status['active_positions'] += 1
            else:
                status['completed_positions'] += 1
            
            status['positions'].append(position_status)
        
        return status
    
    def cancel_tp_sl_orders(self, position_id: str) -> Dict[str, Any]:
        """❌ Cancel TP/SL Orders für eine Position"""
        
        if position_id not in self.monitoring_positions:
            return {'error': 'Position not found'}
        
        position_data = self.monitoring_positions[position_id]
        result = {'cancelled_orders': [], 'errors': []}
        
        try:
            # Cancel TP Order
            if position_data.get('tp_order_id'):
                try:
                    self.exchange.cancel_order(position_data['tp_order_id'], position_data['symbol'])
                    result['cancelled_orders'].append('take_profit')
                except Exception as e:
                    result['errors'].append(f"TP cancel failed: {e}")
            
            # Cancel SL Order
            if position_data.get('sl_order_id'):
                try:
                    self.exchange.cancel_order(position_data['sl_order_id'], position_data['symbol'])
                    result['cancelled_orders'].append('stop_loss')
                except Exception as e:
                    result['errors'].append(f"SL cancel failed: {e}")
            
            # Update Status
            position_data['status'] = 'cancelled'
            
            return result
            
        except Exception as e:
            return {'error': f"Cancel operation failed: {e}"}

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