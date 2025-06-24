#!/usr/bin/env python3
"""
🎯 TAKE PROFIT / STOP LOSS MANAGER
Dediziertes TP/SL Management System für TRADINO
Mit Telegram Notifications und Risk Management Integration
"""

import os
import sys
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Telegram Bot Integration
try:
    import telegram
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("⚠️ python-telegram-bot not installed. Telegram notifications disabled.")

class TPSLManager:
    """🎯 Advanced Take Profit / Stop Loss Manager"""
    
    def __init__(self, trading_api, risk_manager=None, config: Dict[str, Any] = None):
        self.trading_api = trading_api
        self.risk_manager = risk_manager
        self.config = config or self.get_default_config()
        
        # TP/SL State
        self.active_positions = {}
        self.executed_orders = []
        self.performance_stats = {
            'total_tp_hits': 0,
            'total_sl_hits': 0,
            'total_tp_profit': 0.0,
            'total_sl_loss': 0.0,
            'average_tp_time': 0,
            'average_sl_time': 0,
            'success_rate': 0.0
        }
        
        # Monitoring
        self.is_monitoring = False
        self.monitoring_thread = None
        self.last_monitoring_check = datetime.now()
        
        # Telegram Integration
        self.telegram_bot = None
        if TELEGRAM_AVAILABLE and self.config.get('telegram_token'):
            self.initialize_telegram()
        
        print("🎯 TP/SL Manager initialized")
    
    def get_default_config(self) -> Dict[str, Any]:
        """⚙️ Default TP/SL configuration"""
        
        return {
            'monitoring_interval': 10,  # Sekunden
            'price_tolerance': 0.0005,  # 0.05% tolerance für manual triggering
            'max_monitoring_time': 86400,  # 24 Stunden max
            'telegram_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
            'enable_notifications': True,
            'retry_failed_orders': True,
            'retry_attempts': 3,
            'retry_delay': 5,
            'enable_trailing_stop': False,
            'trailing_stop_percent': 0.005,  # 0.5%
            'emergency_close_enabled': True
        }
    
    def initialize_telegram(self):
        """📱 Initialize Telegram Bot"""
        
        try:
            self.telegram_bot = Bot(token=self.config['telegram_token'])
            # Test connection
            bot_info = self.telegram_bot.get_me()
            print(f"✅ Telegram Bot connected: @{bot_info.username}")
        except Exception as e:
            print(f"❌ Telegram Bot initialization failed: {e}")
            self.telegram_bot = None
    
    def add_position(self, position_data: Dict[str, Any]) -> str:
        """➕ Add new position for TP/SL monitoring"""
        
        position_id = f"pos_{int(time.time())}_{position_data['symbol']}"
        
        enhanced_position = {
            'id': position_id,
            'symbol': position_data['symbol'],
            'side': position_data['side'],
            'amount': position_data['amount'],
            'entry_price': position_data['entry_price'],
            'tp_price': position_data.get('tp_price'),
            'sl_price': position_data.get('sl_price'),
            'tp_order_id': position_data.get('tp_order_id'),
            'sl_order_id': position_data.get('sl_order_id'),
            'status': 'active',
            'created_at': datetime.now(),
            'last_checked': datetime.now(),
            'check_count': 0,
            'price_history': [],
            'trailing_stop_price': position_data.get('sl_price'),  # Initial trailing stop
            'max_favorable_price': position_data['entry_price']
        }
        
        self.active_positions[position_id] = enhanced_position
        
        # Start monitoring if not already running
        if not self.is_monitoring:
            self.start_monitoring()
        
        print(f"✅ Position added to TP/SL monitoring: {position_id}")
        
        # Telegram notification
        self.send_telegram_notification(
            f"🎯 NEW POSITION MONITORED\n\n"
            f"Symbol: {enhanced_position['symbol']}\n"
            f"Side: {enhanced_position['side'].upper()}\n"
            f"Entry: ${enhanced_position['entry_price']:.4f}\n"
            f"TP: ${enhanced_position['tp_price']:.4f}\n"
            f"SL: ${enhanced_position['sl_price']:.4f}"
        )
        
        return position_id
    
    def start_monitoring(self):
        """🔄 Start TP/SL monitoring thread"""
        
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        print("👁️ TP/SL monitoring started")
    
    def stop_monitoring(self):
        """🛑 Stop TP/SL monitoring"""
        
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        print("🛑 TP/SL monitoring stopped")
    
    def _monitoring_loop(self):
        """🔄 Main monitoring loop"""
        
        while self.is_monitoring:
            try:
                current_time = datetime.now()
                
                # Process active positions
                for position_id, position in list(self.active_positions.items()):
                    if position['status'] != 'active':
                        continue
                    
                    # Check if position should be removed (too old)
                    age = (current_time - position['created_at']).total_seconds()
                    if age > self.config['max_monitoring_time']:
                        position['status'] = 'expired'
                        print(f"⏰ Position expired: {position_id}")
                        continue
                    
                    # Check TP/SL conditions
                    self._check_position_tp_sl(position)
                    
                    # Update trailing stop if enabled
                    if self.config['enable_trailing_stop']:
                        self._update_trailing_stop(position)
                    
                    position['last_checked'] = current_time
                    position['check_count'] += 1
                
                # Performance statistics update
                self._update_performance_stats()
                
                self.last_monitoring_check = current_time
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                print(f"❌ TP/SL monitoring error: {e}")
                time.sleep(10)
    
    def _check_position_tp_sl(self, position: Dict[str, Any]):
        """🔍 Check individual position for TP/SL triggers"""
        
        try:
            symbol = position['symbol']
            
            # Get current market price
            ticker = self.trading_api.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Store price history
            position['price_history'].append({
                'price': current_price,
                'timestamp': datetime.now()
            })
            
            # Keep only last 100 price points
            if len(position['price_history']) > 100:
                position['price_history'] = position['price_history'][-100:]
            
            side = position['side']
            tp_price = position.get('tp_price')
            sl_price = position.get('sl_price')
            
            # Check TP/SL trigger conditions
            tp_triggered = False
            sl_triggered = False
            
            if side.lower() == 'buy':
                # Long Position
                if tp_price and current_price >= tp_price * (1 - self.config['price_tolerance']):
                    tp_triggered = True
                elif sl_price and current_price <= sl_price * (1 + self.config['price_tolerance']):
                    sl_triggered = True
            else:
                # Short Position
                if tp_price and current_price <= tp_price * (1 + self.config['price_tolerance']):
                    tp_triggered = True
                elif sl_price and current_price >= sl_price * (1 - self.config['price_tolerance']):
                    sl_triggered = True
            
            # Execute if triggered
            if tp_triggered:
                self._execute_tp_sl(position, 'take_profit', current_price)
            elif sl_triggered:
                self._execute_tp_sl(position, 'stop_loss', current_price)
            
        except Exception as e:
            print(f"❌ Position check error for {position.get('id', 'unknown')}: {e}")
    
    def _execute_tp_sl(self, position: Dict[str, Any], order_type: str, current_price: float):
        """⚡ Execute TP/SL order"""
        
        try:
            symbol = position['symbol']
            side = position['side']
            amount = position['amount']
            position_id = position['id']
            
            # Opposite side for exit
            exit_side = 'sell' if side.lower() == 'buy' else 'buy'
            
            print(f"🎯 Executing {order_type.upper()}: {symbol} @ ${current_price:.4f}")
            
            # Try to cancel existing orders first
            self._cancel_remaining_orders(position)
            
            # Execute market order for immediate exit
            exit_result = self.trading_api.place_market_order(symbol, exit_side, amount)
            
            if exit_result.get('success'):
                # Calculate P&L
                entry_price = position['entry_price']
                if side.lower() == 'buy':
                    pnl = (current_price - entry_price) * amount
                else:
                    pnl = (entry_price - current_price) * amount
                
                # Update position status
                position['status'] = f'{order_type}_executed'
                position['exit_price'] = current_price
                position['exit_time'] = datetime.now()
                position['pnl'] = pnl
                position['exit_order'] = exit_result
                
                # Store executed order
                executed_order = {
                    'position_id': position_id,
                    'symbol': symbol,
                    'type': order_type,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'target_price': position.get('tp_price' if order_type == 'take_profit' else 'sl_price'),
                    'amount': amount,
                    'pnl': pnl,
                    'execution_time': datetime.now(),
                    'exit_order': exit_result
                }
                
                self.executed_orders.append(executed_order)
                
                # Update performance stats
                if order_type == 'take_profit':
                    self.performance_stats['total_tp_hits'] += 1
                    self.performance_stats['total_tp_profit'] += pnl
                else:
                    self.performance_stats['total_sl_hits'] += 1
                    self.performance_stats['total_sl_loss'] += pnl
                
                # Success message
                profit_loss = "PROFIT" if pnl > 0 else "LOSS"
                print(f"✅ {order_type.upper()} EXECUTED: {symbol} | "
                      f"PnL: ${pnl:.2f} ({profit_loss})")
                
                # Telegram notification
                emoji = "✅" if order_type == 'take_profit' else "🛑"
                status = "TAKE PROFIT" if order_type == 'take_profit' else "STOP LOSS"
                
                self.send_telegram_notification(
                    f"{emoji} {status} EXECUTED\n\n"
                    f"Symbol: {symbol}\n"
                    f"Side: {side.upper()}\n"
                    f"Entry: ${entry_price:.4f}\n"
                    f"Exit: ${current_price:.4f}\n"
                    f"Target: ${executed_order['target_price']:.4f}\n"
                    f"Amount: {amount:.6f}\n"
                    f"PnL: ${pnl:.2f} ({profit_loss})\n"
                    f"Time: {executed_order['execution_time'].strftime('%H:%M:%S')}"
                )
                
                return True
                
            else:
                error_msg = exit_result.get('error', 'Unknown error')
                print(f"❌ {order_type.upper()} execution failed: {error_msg}")
                
                # Retry if enabled
                if self.config['retry_failed_orders']:
                    self._retry_tp_sl_execution(position, order_type, current_price)
                
                return False
                
        except Exception as e:
            print(f"❌ TP/SL execution error: {e}")
            return False
    
    def _cancel_remaining_orders(self, position: Dict[str, Any]):
        """❌ Cancel remaining TP/SL orders"""
        
        try:
            symbol = position['symbol']
            
            # Cancel TP order
            if position.get('tp_order_id'):
                try:
                    self.trading_api.exchange.cancel_order(position['tp_order_id'], symbol)
                    print(f"❌ Cancelled TP order: {position['tp_order_id']}")
                except:
                    pass  # Order might already be cancelled
            
            # Cancel SL order
            if position.get('sl_order_id'):
                try:
                    self.trading_api.exchange.cancel_order(position['sl_order_id'], symbol)
                    print(f"❌ Cancelled SL order: {position['sl_order_id']}")
                except:
                    pass  # Order might already be cancelled
                    
        except Exception as e:
            print(f"❌ Order cancellation error: {e}")
    
    def _retry_tp_sl_execution(self, position: Dict[str, Any], order_type: str, price: float):
        """🔄 Retry failed TP/SL execution"""
        
        for attempt in range(self.config['retry_attempts']):
            print(f"🔄 Retrying {order_type} execution (attempt {attempt + 1})")
            
            time.sleep(self.config['retry_delay'])
            
            # Try execution again
            if self._execute_tp_sl(position, order_type, price):
                print(f"✅ {order_type} retry successful")
                return True
        
        print(f"❌ {order_type} retry failed after {self.config['retry_attempts']} attempts")
        
        # Emergency notification
        self.send_telegram_notification(
            f"🚨 URGENT: {order_type.upper()} EXECUTION FAILED\n\n"
            f"Symbol: {position['symbol']}\n"
            f"Manual intervention required!\n"
            f"Current Price: ${price:.4f}"
        )
        
        return False
    
    def _update_trailing_stop(self, position: Dict[str, Any]):
        """📈 Update trailing stop loss"""
        
        if not position.get('sl_price'):
            return
        
        try:
            symbol = position['symbol']
            ticker = self.trading_api.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            side = position['side']
            trailing_percent = self.config['trailing_stop_percent']
            
            # Update max favorable price
            if side.lower() == 'buy':
                # Long position - track highest price
                if current_price > position['max_favorable_price']:
                    position['max_favorable_price'] = current_price
                    
                    # Update trailing stop
                    new_trailing_stop = current_price * (1 - trailing_percent)
                    if new_trailing_stop > position['trailing_stop_price']:
                        position['trailing_stop_price'] = new_trailing_stop
                        position['sl_price'] = new_trailing_stop
                        print(f"📈 Trailing stop updated: {symbol} @ ${new_trailing_stop:.4f}")
            else:
                # Short position - track lowest price
                if current_price < position['max_favorable_price']:
                    position['max_favorable_price'] = current_price
                    
                    # Update trailing stop
                    new_trailing_stop = current_price * (1 + trailing_percent)
                    if new_trailing_stop < position['trailing_stop_price']:
                        position['trailing_stop_price'] = new_trailing_stop
                        position['sl_price'] = new_trailing_stop
                        print(f"📈 Trailing stop updated: {symbol} @ ${new_trailing_stop:.4f}")
                        
        except Exception as e:
            print(f"❌ Trailing stop update error: {e}")
    
    def _update_performance_stats(self):
        """📊 Update performance statistics"""
        
        total_orders = len(self.executed_orders)
        if total_orders == 0:
            return
        
        tp_orders = [o for o in self.executed_orders if o['type'] == 'take_profit']
        sl_orders = [o for o in self.executed_orders if o['type'] == 'stop_loss']
        
        # Success rate (TP vs SL)
        if total_orders > 0:
            self.performance_stats['success_rate'] = len(tp_orders) / total_orders
        
        # Average execution times
        if tp_orders:
            tp_times = [(o['execution_time'] - self.active_positions.get(o['position_id'], {}).get('created_at', o['execution_time'])).total_seconds() for o in tp_orders]
            self.performance_stats['average_tp_time'] = np.mean(tp_times)
        
        if sl_orders:
            sl_times = [(o['execution_time'] - self.active_positions.get(o['position_id'], {}).get('created_at', o['execution_time'])).total_seconds() for o in sl_orders]
            self.performance_stats['average_sl_time'] = np.mean(sl_times)
    
    def send_telegram_notification(self, message: str):
        """📱 Send Telegram notification"""
        
        if not self.config['enable_notifications'] or not self.telegram_bot:
            print(f"📱 Telegram (disabled): {message}")
            return
        
        try:
            chat_id = self.config['telegram_chat_id']
            if chat_id:
                self.telegram_bot.send_message(chat_id=chat_id, text=message)
                print(f"📱 Telegram sent: {message[:50]}...")
            else:
                print(f"📱 No chat_id: {message}")
        except Exception as e:
            print(f"❌ Telegram error: {e}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """📊 Get comprehensive TP/SL status report"""
        
        active_count = len([p for p in self.active_positions.values() if p['status'] == 'active'])
        
        report = {
            'monitoring_status': 'active' if self.is_monitoring else 'stopped',
            'active_positions': active_count,
            'total_positions': len(self.active_positions),
            'executed_orders': len(self.executed_orders),
            'performance_stats': self.performance_stats.copy(),
            'last_check': self.last_monitoring_check.isoformat(),
            'positions': []
        }
        
        # Add position details
        for position_id, position in self.active_positions.items():
            if position['status'] == 'active':
                current_age = (datetime.now() - position['created_at']).total_seconds()
                report['positions'].append({
                    'id': position_id,
                    'symbol': position['symbol'],
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'tp_price': position.get('tp_price'),
                    'sl_price': position.get('sl_price'),
                    'age_seconds': current_age,
                    'check_count': position['check_count']
                })
        
        return report
    
    def emergency_close_all(self) -> Dict[str, Any]:
        """🚨 Emergency close all monitored positions"""
        
        if not self.config['emergency_close_enabled']:
            return {'error': 'Emergency close disabled'}
        
        print("🚨 EMERGENCY CLOSE ALL POSITIONS")
        
        results = {
            'closed_positions': 0,
            'failed_positions': 0,
            'errors': []
        }
        
        for position_id, position in self.active_positions.items():
            if position['status'] != 'active':
                continue
            
            try:
                # Cancel existing orders
                self._cancel_remaining_orders(position)
                
                # Force close position
                symbol = position['symbol']
                side = position['side']
                amount = position['amount']
                exit_side = 'sell' if side.lower() == 'buy' else 'buy'
                
                exit_result = self.trading_api.place_market_order(symbol, exit_side, amount)
                
                if exit_result.get('success'):
                    position['status'] = 'emergency_closed'
                    results['closed_positions'] += 1
                    print(f"🚨 Emergency closed: {symbol}")
                else:
                    results['failed_positions'] += 1
                    results['errors'].append(f"{symbol}: {exit_result.get('error', 'Unknown')}")
                    
            except Exception as e:
                results['failed_positions'] += 1
                results['errors'].append(f"{position_id}: {str(e)}")
        
        # Emergency notification
        self.send_telegram_notification(
            f"🚨 EMERGENCY CLOSE EXECUTED\n\n"
            f"Closed: {results['closed_positions']} positions\n"
            f"Failed: {results['failed_positions']} positions\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        
        return results

# Global instance
tp_sl_manager = None

def initialize_tp_sl_manager(trading_api, config=None):
    """🚀 Initialize global TP/SL Manager"""
    
    global tp_sl_manager
    tp_sl_manager = TPSLManager(trading_api, None, config)
    return tp_sl_manager

def get_tp_sl_manager():
    """🎯 Get global TP/SL Manager instance"""
    return tp_sl_manager

if __name__ == "__main__":
    print("🎯 TP/SL Manager ready for testing!") 