#!/usr/bin/env python3
"""
🤖 ALPHA SMART POSITION MANAGER
===============================

INTELLIGENTES TRADING SYSTEM mit automatischen:
✅ Stop Loss Orders
✅ Take Profit Orders  
✅ Leverage Einstellung
✅ Position Management

🎯 ECHTE KI-GESTEUERTE ORDERS!
"""

import asyncio
import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

try:
    import ccxt
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Install: pip install ccxt python-dotenv")
    sys.exit(1)

# Load environment
load_dotenv('tradino_unschlagbar/.env')

class AlphaSmartPositionManager:
    """🤖 Intelligenter Position Manager mit automatischen Orders"""
    
    def __init__(self):
        self.config = {
            'symbol': 'BTCUSDT',
            'symbol_ccxt': 'BTC/USDT:USDT',
            'position_size_usd': 200,  # $200 Position für bessere Sichtbarkeit
            'leverage': 10,  # 10x Leverage
            'stop_loss_pct': 2.0,  # 2% Stop Loss
            'take_profit_pct': 4.0,  # 4% Take Profit
            'max_position_time': 10,  # 10 minutes max
        }
        
        self.exchange = None
        self.logger = None
        self.setup_logging()
        self._init_exchange()
        
        # Position tracking
        self.order_id = None
        self.stop_loss_order_id = None
        self.take_profit_order_id = None
        self.entry_price = 0
        self.position_size = 0
        self.position_side = None
        self.start_time = None
        
        print("🤖 ALPHA SMART POSITION MANAGER INITIALIZED")
        print("=" * 50)
        print(f"💰 Position Size: ${self.config['position_size_usd']}")
        print(f"🔥 Leverage: {self.config['leverage']}x")
        print(f"🛑 Stop Loss: {self.config['stop_loss_pct']}%")
        print(f"🎯 Take Profit: {self.config['take_profit_pct']}%")
        print("=" * 50)
    
    def _init_exchange(self):
        """Initialize Bitget exchange with correct settings"""
        try:
            self.exchange = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_SECRET_KEY'),
                'password': os.getenv('BITGET_PASSPHRASE'),
                'sandbox': True,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',
                    'defaultSubType': 'linear'
                }
            })
            print("✅ Bitget exchange initialized")
            
        except Exception as e:
            print(f"❌ Exchange initialization failed: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('alpha_smart_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def check_connection(self) -> bool:
        """Check API connection and balance"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            print(f"✅ API Connection successful")
            print(f"💰 Available Balance: ${usdt_balance:,.2f} USDT")
            
            if usdt_balance < self.config['position_size_usd']:
                print(f"⚠️ Insufficient balance for ${self.config['position_size_usd']} position")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Connection check failed: {e}")
            return False
    
    async def get_market_price(self) -> float:
        """Get current market price"""
        try:
            ticker = self.exchange.fetch_ticker(self.config['symbol_ccxt'])
            return ticker['last']
        except Exception as e:
            self.logger.error(f"Failed to get market price: {e}")
            return 0
    
    async def set_leverage_properly(self) -> bool:
        """Properly set leverage using CCXT method"""
        try:
            print(f"🔧 Setting leverage to {self.config['leverage']}x...")
            
            # Use CCXT method - this should work with one-way mode
            self.exchange.set_leverage(self.config['leverage'], self.config['symbol_ccxt'])
            print(f"✅ Leverage set to {self.config['leverage']}x")
            return True
                    
        except Exception as e:
            print(f"❌ Leverage setting failed: {e}")
            return False
    
    async def open_smart_position(self, side: str = 'long') -> bool:
        """Open position and immediately set stop loss and take profit"""
        try:
            print(f"\n🚀 OPENING SMART {side.upper()} POSITION")
            print("=" * 50)
            
            # Get current price
            current_price = await self.get_market_price()
            if not current_price:
                return False
            
            # Calculate position size
            position_size_btc = self.config['position_size_usd'] / current_price
            
            print(f"📊 Market Price: ${current_price:,.2f}")
            print(f"📏 Position Size: {position_size_btc:.6f} BTC")
            print(f"💰 Notional Value: ${self.config['position_size_usd']}")
            
            # Open main position using CCXT (this works)
            main_order = self.exchange.create_market_order(
                self.config['symbol_ccxt'],
                'buy' if side == 'long' else 'sell',
                position_size_btc
            )
            
            if main_order:
                self.order_id = main_order.get('id')
                self.position_size = position_size_btc
                self.position_side = side
                self.start_time = datetime.now()
                self.entry_price = current_price
                
                print(f"✅ MAIN POSITION OPENED!")
                print(f"🆔 Order ID: {self.order_id}")
                
                # Wait for position to settle
                await asyncio.sleep(5)
                
                # Now set stop loss and take profit
                await self.set_automatic_orders(current_price, position_size_btc, side)
                
                return True
            else:
                print("❌ Failed to open main position")
                return False
                
        except Exception as e:
            print(f"❌ Smart position opening failed: {e}")
            return False
    
    async def set_automatic_orders(self, entry_price: float, size: float, side: str):
        """Set automatic stop loss and take profit using CCXT"""
        try:
            print(f"\n🛡️ SETTING AUTOMATIC STOP LOSS & TAKE PROFIT...")
            
            if side == 'long':
                # Long position
                stop_loss_price = entry_price * (1 - self.config['stop_loss_pct'] / 100)
                take_profit_price = entry_price * (1 + self.config['take_profit_pct'] / 100)
                order_side = 'sell'  # Close long with sell
            else:
                # Short position
                stop_loss_price = entry_price * (1 + self.config['stop_loss_pct'] / 100)
                take_profit_price = entry_price * (1 - self.config['take_profit_pct'] / 100)
                order_side = 'buy'   # Close short with buy
            
            print(f"🎯 Entry Price: ${entry_price:,.2f}")
            print(f"🛑 Stop Loss Target: ${stop_loss_price:,.2f}")
            print(f"💰 Take Profit Target: ${take_profit_price:,.2f}")
            
            # Try to place stop loss order
            try:
                sl_order = self.exchange.create_order(
                    self.config['symbol_ccxt'],
                    'stop_market',
                    order_side,
                    size,
                    None,  # No limit price for stop market
                    params={
                        'stopPrice': stop_loss_price,
                        'reduceOnly': True
                    }
                )
                
                if sl_order:
                    self.stop_loss_order_id = sl_order.get('id')
                    print(f"✅ STOP LOSS ORDER PLACED!")
                    print(f"🆔 SL Order ID: {self.stop_loss_order_id}")
                else:
                    print("⚠️ Stop Loss order failed")
                    
            except Exception as e:
                print(f"⚠️ Stop Loss order error: {e}")
            
            # Try to place take profit order
            try:
                tp_order = self.exchange.create_limit_order(
                    self.config['symbol_ccxt'],
                    order_side,
                    size,
                    take_profit_price,
                    params={'reduceOnly': True}
                )
                
                if tp_order:
                    self.take_profit_order_id = tp_order.get('id')
                    print(f"✅ TAKE PROFIT ORDER PLACED!")
                    print(f"🆔 TP Order ID: {self.take_profit_order_id}")
                else:
                    print("⚠️ Take Profit order failed")
                    
            except Exception as e:
                print(f"⚠️ Take Profit order error: {e}")
            
            # Verify orders are active
            await self.verify_automatic_orders()
            
        except Exception as e:
            print(f"❌ Automatic orders setup failed: {e}")
    
    async def verify_automatic_orders(self):
        """Verify that automatic orders are active"""
        try:
            print(f"\n🔍 VERIFYING AUTOMATIC ORDERS...")
            
            # Check position
            positions = self.exchange.fetch_positions([self.config['symbol_ccxt']])
            active_positions = [p for p in positions if abs(p.get('size', 0)) > 0.0001]
            
            if active_positions:
                pos = active_positions[0]
                print(f"✅ POSITION CONFIRMED!")
                print(f"   📊 Size: {pos.get('size', 0):.6f} BTC")
                print(f"   📈 Side: {pos.get('side', 'Unknown')}")
                print(f"   💰 Entry: ${pos.get('entryPrice', 0):,.2f}")
                print(f"   📊 Current PnL: ${pos.get('unrealizedPnl', 0):+.2f}")
            
            # Check open orders
            try:
                open_orders = self.exchange.fetch_open_orders(self.config['symbol_ccxt'])
                print(f"📋 ACTIVE ORDERS: {len(open_orders)}")
                
                for order in open_orders:
                    order_type = order.get('type', 'unknown')
                    side = order.get('side', 'unknown')
                    price = order.get('price', 0)
                    stop_price = order.get('stopPrice', 0)
                    
                    if 'stop' in order_type.lower():
                        print(f"🛑 Stop Loss: {side} @ ${stop_price or price:,.2f}")
                    elif 'limit' in order_type.lower():
                        print(f"💰 Take Profit: {side} @ ${price:,.2f}")
                    else:
                        print(f"📝 Order: {order_type} {side}")
                
                if len(open_orders) >= 1:
                    print("✅ AUTOMATIC ORDERS ARE ACTIVE!")
                else:
                    print("⚠️ No automatic orders found - will monitor manually")
                    
            except Exception as e:
                print(f"⚠️ Could not verify orders: {e}")
                
        except Exception as e:
            print(f"❌ Order verification failed: {e}")
    
    async def monitor_smart_position(self):
        """Monitor position with automatic orders"""
        print(f"\n📊 SMART POSITION MONITORING")
        print("=" * 50)
        print("Time | Price | PnL | Status")
        print("-" * 50)
        
        monitor_count = 0
        
        while self.order_id:
            try:
                monitor_count += 1
                current_price = await self.get_market_price()
                
                # Check if position still exists
                positions = self.exchange.fetch_positions([self.config['symbol_ccxt']])
                active_positions = [p for p in positions if abs(p.get('size', 0)) > 0.0001]
                
                if not active_positions:
                    print(f"\n🎉 POSITION AUTOMATICALLY CLOSED!")
                    print("✅ Stop Loss or Take Profit was triggered!")
                    break
                
                # Get current PnL
                pos = active_positions[0]
                current_pnl = pos.get('unrealizedPnl', 0)
                
                # Display status
                elapsed = int((datetime.now() - self.start_time).total_seconds())
                
                print(f"\r{elapsed:03d}s | ${current_price:8,.2f} | "
                      f"${current_pnl:+7.2f} | "
                      f"{'Auto Orders Active' if self.stop_loss_order_id or self.take_profit_order_id else 'Manual Only'}", end="")
                
                # Time-based exit (backup)
                if self.start_time:
                    elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
                    if elapsed_minutes >= self.config['max_position_time']:
                        print(f"\n⏰ TIME LIMIT REACHED - CLOSING MANUALLY")
                        await self.close_position_manually("Time limit")
                        break
                
                await asyncio.sleep(3)
                
            except Exception as e:
                self.logger.error(f"Error in smart monitoring: {e}")
                await asyncio.sleep(5)
    
    async def close_position_manually(self, reason: str):
        """Manually close position and cancel orders"""
        try:
            print(f"\n🛑 MANUAL POSITION CLOSE")
            print(f"📝 Reason: {reason}")
            
            # Cancel pending orders
            try:
                open_orders = self.exchange.fetch_open_orders(self.config['symbol_ccxt'])
                for order in open_orders:
                    self.exchange.cancel_order(order['id'], self.config['symbol_ccxt'])
                    print(f"❌ Cancelled order: {order['id']}")
            except:
                pass
            
            # Close position
            positions = self.exchange.fetch_positions([self.config['symbol_ccxt']])
            for pos in positions:
                size = abs(pos.get('size', 0))
                if size > 0.0001:
                    side = 'sell' if pos.get('side') == 'long' else 'buy'
                    
                    close_order = self.exchange.create_market_order(
                        self.config['symbol_ccxt'],
                        side,
                        size,
                        params={'reduceOnly': True}
                    )
                    
                    print(f"✅ Manual close order: {close_order.get('id')}")
                    break
            
        except Exception as e:
            print(f"❌ Manual close failed: {e}")
    
    async def run_smart_trading_cycle(self):
        """Run complete smart trading cycle"""
        print("🤖 ALPHA SMART POSITION MANAGER")
        print("=" * 60)
        print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💱 Symbol: {self.config['symbol_ccxt']}")
        print(f"💰 Position Size: ${self.config['position_size_usd']}")
        print(f"🔥 Leverage: {self.config['leverage']}x")
        print(f"🛑 Auto Stop Loss: {self.config['stop_loss_pct']}%")
        print(f"💰 Auto Take Profit: {self.config['take_profit_pct']}%")
        print("🎯 AUTOMATIC ORDERS WILL BE PLACED!")
        print("=" * 60)
        
        # Step 1: Check connection
        if not await self.check_connection():
            print("❌ Connection check failed")
            return
        
        # Step 2: Set leverage
        if not await self.set_leverage_properly():
            print("⚠️ Leverage setting failed - continuing anyway")
        
        # Step 3: Open smart position with auto orders
        if not await self.open_smart_position('long'):
            print("❌ Failed to open smart position")
            return
        
        # Step 4: Monitor smart position
        await self.monitor_smart_position()
        
        print(f"\n🎉 SMART TRADING CYCLE COMPLETED")
        print(f"📅 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

async def main():
    """Main function"""
    try:
        print("🤖 ALPHA SMART POSITION MANAGER")
        print("⚠️  This will open REAL positions with automatic orders!")
        print("⚠️  Stop Loss and Take Profit will be set automatically!")
        print("⚠️  Press Ctrl+C within 5 seconds to cancel...")
        
        await asyncio.sleep(5)
        
        manager = AlphaSmartPositionManager()
        await manager.run_smart_trading_cycle()
        
    except KeyboardInterrupt:
        print("\n🛑 Smart trading cancelled by user")
    except Exception as e:
        print(f"❌ Smart trading failed: {e}")
        logging.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
