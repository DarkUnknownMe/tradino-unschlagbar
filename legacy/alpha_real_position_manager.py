#!/usr/bin/env python3
"""
🚀 ALPHA REAL POSITION MANAGER
==============================
ECHTE Position auf Bitget mit korrekter API
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
    print(f"❌ Missing dependencies: {e}")
    sys.exit(1)

class AlphaRealPositionManager:
    def __init__(self):
        load_dotenv('tradino_unschlagbar/.env')
        
        # Exchange setup
        self.exchange = None
        self._init_exchange()
        
        # Trading configuration
        self.config = {
            'symbol': 'BTCUSDT',  # Bitget format without separators
            'symbol_ccxt': 'BTC/USDT:USDT',  # CCXT format
            'position_size_usd': 100,  # $100 Position für Test
            'leverage': 5,  # 5x Leverage für Sicherheit
            'stop_loss_percent': 2.0,  # 2% Stop Loss
            'take_profit_percent': 4.0,  # 4% Take Profit
            'max_position_time': 15,  # Max 15 minutes
        }
        
        # Position tracking
        self.position_data = {}
        self.entry_price = 0
        self.order_id = None
        self.position_side = None
        self.position_size = 0
        self.start_time = None
        
        # Logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def _init_exchange(self):
        """Initialize Bitget exchange connection"""
        try:
            self.exchange = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_SECRET_KEY'),
                'password': os.getenv('BITGET_PASSPHRASE'),
                'sandbox': True,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',
                    'marginMode': 'isolated'
                }
            })
            print("✅ Bitget exchange initialized")
        except Exception as e:
            print(f"❌ Exchange initialization failed: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Setup detailed logging"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('alpha_real_trading.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
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
    
    async def set_leverage(self):
        """Set leverage for the symbol"""
        try:
            # Use the correct Bitget Mix API
            response = self.exchange.private_mix_post_v2_mix_account_set_leverage({
                'symbol': self.config['symbol'],
                'marginCoin': 'USDT',
                'leverage': str(self.config['leverage']),
                'holdSide': 'long'  # Set for long side
            })
            
            if response.get('code') == '00000':
                print(f"✅ Leverage set to {self.config['leverage']}x for LONG")
            
            # Also set for short side
            response = self.exchange.private_mix_post_v2_mix_account_set_leverage({
                'symbol': self.config['symbol'],
                'marginCoin': 'USDT',
                'leverage': str(self.config['leverage']),
                'holdSide': 'short'  # Set for short side
            })
            
            if response.get('code') == '00000':
                print(f"✅ Leverage set to {self.config['leverage']}x for SHORT")
                
        except Exception as e:
            self.logger.warning(f"Failed to set leverage: {e}")
    
    async def open_real_position(self, side: str = 'long') -> bool:
        """Open REAL position using correct Bitget Mix API"""
        try:
            print(f"\n🚀 OPENING REAL {side.upper()} POSITION")
            print("=" * 50)
            
            # Get current price
            current_price = await self.get_market_price()
            if not current_price:
                return False
            
            # Calculate position size in BTC
            position_size_btc = self.config['position_size_usd'] / current_price
            
            print(f"📊 Market Price: ${current_price:,.2f}")
            print(f"📏 Position Size: {position_size_btc:.6f} BTC")
            print(f"💰 Notional Value: ${self.config['position_size_usd']}")
            print(f"🔥 Leverage: {self.config['leverage']}x")
            
            # Prepare order parameters for Bitget Mix API v2
            order_params = {
                'symbol': self.config['symbol'],
                'productType': 'USDT-FUTURES',
                'marginCoin': 'USDT',
                'marginMode': 'isolated',
                'side': 'buy' if side == 'long' else 'sell',
                'orderType': 'market',
                'size': str(position_size_btc),
                'timeInForceValue': 'normal'
            }
            
            print(f"📝 Order Parameters: {order_params}")
            
            # Place REAL order via Bitget Mix API v2
            print("🔥 PLACING REAL ORDER ON BITGET...")
            response = self.exchange.private_mix_post_v2_mix_order_place_order(order_params)
            
            print(f"📋 API Response: {response}")
            
            if response and response.get('code') == '00000':
                order_data = response.get('data', {})
                order_id = order_data.get('orderId')
                
                if order_id:
                    self.order_id = order_id
                    self.entry_price = current_price
                    self.position_side = side
                    self.position_size = position_size_btc
                    self.start_time = datetime.now()
                    
                    print(f"🎉 REAL POSITION OPENED SUCCESSFULLY!")
                    print(f"🎯 Entry Price: ${self.entry_price:,.2f}")
                    print(f"📏 Position Size: {self.position_size:.6f} BTC")
                    print(f"🆔 Order ID: {order_id}")
                    print(f"⏰ Time: {self.start_time.strftime('%H:%M:%S')}")
                    
                    # Wait a moment for order to fill
                    await asyncio.sleep(3)
                    
                    # Verify position is active
                    await self.verify_position()
                    
                    return True
                else:
                    print(f"❌ No order ID in response")
            else:
                error_msg = response.get('msg', 'Unknown error')
                error_code = response.get('code', 'Unknown code')
                print(f"❌ Order failed: {error_code} - {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Failed to open real position: {e}")
            print(f"❌ Exception: {e}")
            return False
        
        return False
    
    async def verify_position(self):
        """Verify that position is actually open on Bitget"""
        try:
            print(f"\n🔍 VERIFYING REAL POSITION ON BITGET...")
            
            # Fetch positions
            positions = self.exchange.fetch_positions([self.config['symbol_ccxt']])
            active_positions = [p for p in positions if abs(p.get('size', 0)) > 0.0001]
            
            if active_positions:
                pos = active_positions[0]
                print(f"✅ POSITION VERIFIED ON BITGET!")
                print(f"   Symbol: {pos.get('symbol', 'Unknown')}")
                print(f"   Side: {pos.get('side', 'Unknown')}")
                print(f"   Size: {pos.get('size', 0):.6f}")
                print(f"   Notional: ${pos.get('notional', 0):,.2f}")
                print(f"   Entry Price: ${pos.get('entryPrice', 0):,.2f}")
                print(f"   Mark Price: ${pos.get('markPrice', 0):,.2f}")
                print(f"   Unrealized PnL: ${pos.get('unrealizedPnl', 0):+.2f}")
                
                # Update our tracking with real data
                self.entry_price = pos.get('entryPrice', self.entry_price)
                self.position_size = abs(pos.get('size', self.position_size))
                
                return True
            else:
                print(f"⚠️ Position not yet visible (may take a moment)")
                return False
                
        except Exception as e:
            print(f"❌ Position verification failed: {e}")
            return False
    
    async def get_real_position_pnl(self) -> float:
        """Get REAL position P&L from Bitget"""
        try:
            positions = self.exchange.fetch_positions([self.config['symbol_ccxt']])
            
            for pos in positions:
                if abs(pos.get('size', 0)) > 0.0001:
                    unrealized_pnl = pos.get('unrealizedPnl', 0)
                    return unrealized_pnl
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to get real position PnL: {e}")
            return 0
    
    async def close_real_position(self, reason: str = "Manual close") -> bool:
        """Close REAL position on Bitget"""
        try:
            print(f"\n🛑 CLOSING REAL POSITION")
            print("=" * 35)
            print(f"📝 Reason: {reason}")
            
            current_price = await self.get_market_price()
            final_pnl = await self.get_real_position_pnl()
            
            # Prepare close order parameters
            close_params = {
                'symbol': self.config['symbol'],
                'productType': 'USDT-FUTURES',
                'marginCoin': 'USDT',
                'marginMode': 'isolated',
                'side': 'sell' if self.position_side == 'long' else 'buy',
                'orderType': 'market',
                'size': str(self.position_size),
                'timeInForceValue': 'normal'
            }
            
            print(f"📝 Close Parameters: {close_params}")
            
            # Close REAL position
            print("🔥 CLOSING REAL POSITION ON BITGET...")
            response = self.exchange.private_mix_post_v2_mix_order_place_order(close_params)
            
            print(f"📋 Close Response: {response}")
            
            if response and response.get('code') == '00000':
                close_order_id = response.get('data', {}).get('orderId')
                duration = (datetime.now() - self.start_time).total_seconds() / 60
                
                print(f"🎉 REAL POSITION CLOSED SUCCESSFULLY!")
                print(f"💰 Final P&L: ${final_pnl:+.2f}")
                print(f"📊 Exit Price: ${current_price:,.2f}")
                print(f"⏱️ Duration: {duration:.1f} minutes")
                print(f"🆔 Close Order ID: {close_order_id}")
                
                # Log trade summary
                trade_summary = {
                    'entry_time': self.start_time.isoformat(),
                    'exit_time': datetime.now().isoformat(),
                    'symbol': self.config['symbol_ccxt'],
                    'side': self.position_side,
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'size': self.position_size,
                    'pnl': final_pnl,
                    'duration_minutes': duration,
                    'close_reason': reason,
                    'order_id': self.order_id,
                    'close_order_id': close_order_id
                }
                
                self.logger.info(f"REAL Trade completed: {json.dumps(trade_summary, indent=2)}")
                
                # Wait and verify position is closed
                await asyncio.sleep(3)
                await self.verify_position_closed()
                
                return True
            else:
                error_msg = response.get('msg', 'Unknown error')
                print(f"❌ Close failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Failed to close real position: {e}")
            return False
        
        return False
    
    async def verify_position_closed(self):
        """Verify position is actually closed"""
        try:
            positions = self.exchange.fetch_positions([self.config['symbol_ccxt']])
            active_positions = [p for p in positions if abs(p.get('size', 0)) > 0.0001]
            
            if not active_positions:
                print(f"✅ POSITION CONFIRMED CLOSED ON BITGET!")
            else:
                print(f"⚠️ Position may still be closing...")
                
        except Exception as e:
            print(f"❌ Close verification failed: {e}")
    
    async def monitor_real_position(self):
        """Monitor REAL position on Bitget"""
        print(f"\n📊 REAL POSITION MONITORING STARTED")
        print("=" * 40)
        
        monitor_count = 0
        
        while self.order_id:
            try:
                monitor_count += 1
                current_price = await self.get_market_price()
                current_pnl = await self.get_real_position_pnl()
                
                # Display status every 3 iterations (9 seconds)
                if monitor_count % 3 == 0:
                    duration = (datetime.now() - self.start_time).total_seconds() / 60
                    print(f"\n📈 REAL Status Update #{monitor_count//3}")
                    print(f"   💰 Real P&L: ${current_pnl:+.2f}")
                    print(f"   📊 Current Price: ${current_price:,.2f}")
                    print(f"   🎯 Entry Price: ${self.entry_price:,.2f}")
                    print(f"   ⏱️ Duration: {duration:.1f} min")
                    print(f"   📏 Position Size: {self.position_size:.6f} BTC")
                
                # Time-based exit (15 minutes max)
                if self.start_time:
                    elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
                    if elapsed_minutes >= self.config['max_position_time']:
                        await self.close_real_position(f"Time limit reached ({elapsed_minutes:.1f} min)")
                        break
                
                # Simple profit target exit (+$20)
                if current_pnl >= 20:
                    await self.close_real_position(f"Profit target reached: ${current_pnl:+.2f}")
                    break
                
                # Simple loss limit exit (-$30)
                if current_pnl <= -30:
                    await self.close_real_position(f"Loss limit reached: ${current_pnl:+.2f}")
                    break
                
                await asyncio.sleep(3)  # Check every 3 seconds
                
            except Exception as e:
                self.logger.error(f"Error in real position monitoring: {e}")
                await asyncio.sleep(5)
    
    async def run_real_trading_cycle(self):
        """Run complete REAL trading cycle"""
        print("🚀 ALPHA REAL POSITION MANAGER")
        print("=" * 60)
        print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💱 Symbol: {self.config['symbol_ccxt']}")
        print(f"💰 Position Size: ${self.config['position_size_usd']}")
        print(f"🔥 Leverage: {self.config['leverage']}x")
        print(f"⏱️ Max Duration: {self.config['max_position_time']} minutes")
        print("🎯 THIS WILL OPEN A REAL POSITION ON BITGET!")
        print("=" * 60)
        
        # Step 1: Check connection
        if not await self.check_connection():
            print("❌ Connection check failed")
            return
        
        # Step 2: Set leverage
        await self.set_leverage()
        
        # Step 3: Open REAL position
        if not await self.open_real_position('long'):  # Can be 'long' or 'short'
            print("❌ Failed to open real position")
            return
        
        # Step 4: Monitor REAL position
        await self.monitor_real_position()
        
        print(f"\n🎉 REAL TRADING CYCLE COMPLETED")
        print(f"📅 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

async def main():
    """Main function"""
    try:
        print("⚠️  WARNING: This will open a REAL position on Bitget!")
        print("⚠️  Make sure you want to proceed with live trading!")
        print("⚠️  Press Ctrl+C within 5 seconds to cancel...")
        
        await asyncio.sleep(5)
        
        manager = AlphaRealPositionManager()
        await manager.run_real_trading_cycle()
        
    except KeyboardInterrupt:
        print("\n🛑 Real trading cancelled by user")
    except Exception as e:
        print(f"❌ Real trading failed: {e}")
        logging.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 