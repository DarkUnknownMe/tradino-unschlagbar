#!/usr/bin/env python3
"""
🚀 ALPHA LIVE POSITION MANAGER
==============================
Comprehensive Live Trading Script mit vollständigem Position Management
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

class AlphaLivePositionManager:
    def __init__(self):
        load_dotenv('tradino_unschlagbar/.env')
        
        # Exchange setup
        self.exchange = None
        self._init_exchange()
        
        # Trading configuration
        self.config = {
            'symbol': 'BTC/USDT:USDT',
            'position_size_usd': 500,  # $500 Position
            'leverage': 10,
            'stop_loss_percent': 2.0,  # 2% Stop Loss
            'take_profit_percent': 4.0,  # 4% Take Profit
            'max_position_time': 30,  # Max 30 minutes
            'trailing_stop': True,
            'break_even_move': True,
        }
        
        # Position tracking
        self.position_data = {}
        self.entry_price = 0
        self.position_id = None
        self.stop_loss_price = 0
        self.take_profit_price = 0
        self.position_side = None
        self.position_size = 0
        self.start_time = None
        
        # Risk management
        self.max_loss_usd = 50  # Max $50 Loss
        self.current_pnl = 0
        self.peak_pnl = 0  # For trailing stop
        
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
                'options': {'defaultType': 'swap'}
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
                logging.FileHandler('alpha_live_trading.log'),
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
            ticker = self.exchange.fetch_ticker(self.config['symbol'])
            return ticker['last']
        except Exception as e:
            self.logger.error(f"Failed to get market price: {e}")
            return 0
    
    async def set_leverage(self):
        """Set leverage for the symbol"""
        try:
            self.exchange.set_leverage(self.config['leverage'], self.config['symbol'])
            print(f"✅ Leverage set to {self.config['leverage']}x")
        except Exception as e:
            self.logger.warning(f"Failed to set leverage: {e}")
    
    async def open_position(self, side: str = 'long') -> bool:
        """Open a new position"""
        try:
            print(f"\n🚀 OPENING {side.upper()} POSITION")
            print("=" * 40)
            
            # Get current price
            current_price = await self.get_market_price()
            if not current_price:
                return False
            
            # Calculate position size
            position_size = self.config['position_size_usd'] / current_price
            
            # Set leverage
            await self.set_leverage()
            
            print(f"📊 Market Price: ${current_price:,.2f}")
            print(f"📏 Position Size: {position_size:.6f} BTC")
            print(f"💰 Notional Value: ${self.config['position_size_usd']}")
            print(f"🔥 Leverage: {self.config['leverage']}x")
            
            # Place market order
            order = self.exchange.create_market_order(
                symbol=self.config['symbol'],
                side='buy' if side == 'long' else 'sell',
                amount=position_size,
                params={'type': 'swap'}
            )
            
            if order and order.get('id'):
                self.position_id = order['id']
                self.entry_price = current_price
                self.position_side = side
                self.position_size = position_size
                self.start_time = datetime.now()
                
                # Calculate stop loss and take profit
                if side == 'long':
                    self.stop_loss_price = current_price * (1 - self.config['stop_loss_percent'] / 100)
                    self.take_profit_price = current_price * (1 + self.config['take_profit_percent'] / 100)
                else:
                    self.stop_loss_price = current_price * (1 + self.config['stop_loss_percent'] / 100)
                    self.take_profit_price = current_price * (1 - self.config['take_profit_percent'] / 100)
                
                print(f"✅ Position opened successfully!")
                print(f"🎯 Entry Price: ${self.entry_price:,.2f}")
                print(f"🛑 Stop Loss: ${self.stop_loss_price:,.2f}")
                print(f"💎 Take Profit: ${self.take_profit_price:,.2f}")
                print(f"🆔 Position ID: {self.position_id}")
                
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to open position: {e}")
            return False
        
        return False
    
    async def get_position_pnl(self) -> float:
        """Get current position P&L"""
        try:
            positions = self.exchange.fetch_positions([self.config['symbol']])
            
            for pos in positions:
                if abs(pos.get('size', 0)) > 0.0001:
                    unrealized_pnl = pos.get('unrealizedPnl', 0)
                    self.current_pnl = unrealized_pnl
                    
                    # Update peak PnL for trailing stop
                    if unrealized_pnl > self.peak_pnl:
                        self.peak_pnl = unrealized_pnl
                    
                    return unrealized_pnl
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to get position PnL: {e}")
            return 0
    
    async def check_stop_conditions(self) -> tuple[bool, str]:
        """Check if position should be closed"""
        current_price = await self.get_market_price()
        current_pnl = await self.get_position_pnl()
        
        if not current_price:
            return False, "Price fetch failed"
        
        # Time-based exit
        if self.start_time:
            elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
            if elapsed_minutes >= self.config['max_position_time']:
                return True, f"Time limit reached ({elapsed_minutes:.1f} min)"
        
        # Stop Loss
        if self.position_side == 'long' and current_price <= self.stop_loss_price:
            return True, f"Stop Loss hit: ${current_price:,.2f} <= ${self.stop_loss_price:,.2f}"
        elif self.position_side == 'short' and current_price >= self.stop_loss_price:
            return True, f"Stop Loss hit: ${current_price:,.2f} >= ${self.stop_loss_price:,.2f}"
        
        # Take Profit
        if self.position_side == 'long' and current_price >= self.take_profit_price:
            return True, f"Take Profit hit: ${current_price:,.2f} >= ${self.take_profit_price:,.2f}"
        elif self.position_side == 'short' and current_price <= self.take_profit_price:
            return True, f"Take Profit hit: ${current_price:,.2f} <= ${self.take_profit_price:,.2f}"
        
        # Trailing Stop Logic
        if self.config['trailing_stop'] and self.peak_pnl > 20:  # Only if we're in profit > $20
            trailing_threshold = self.peak_pnl * 0.5  # Trail at 50% of peak
            if current_pnl < trailing_threshold:
                return True, f"Trailing stop: PnL ${current_pnl:.2f} < ${trailing_threshold:.2f}"
        
        # Max Loss Protection
        if current_pnl < -self.max_loss_usd:
            return True, f"Max loss protection: PnL ${current_pnl:.2f}"
        
        return False, "All conditions OK"
    
    async def close_position(self, reason: str = "Manual close") -> bool:
        """Close the current position"""
        try:
            print(f"\n🛑 CLOSING POSITION")
            print("=" * 30)
            print(f"📝 Reason: {reason}")
            
            current_price = await self.get_market_price()
            final_pnl = await self.get_position_pnl()
            
            # Close position with market order
            close_order = self.exchange.create_market_order(
                symbol=self.config['symbol'],
                side='sell' if self.position_side == 'long' else 'buy',
                amount=abs(self.position_size),
                params={'type': 'swap', 'reduceOnly': True}
            )
            
            if close_order and close_order.get('id'):
                duration = (datetime.now() - self.start_time).total_seconds() / 60
                
                print(f"✅ Position closed successfully!")
                print(f"💰 Final P&L: ${final_pnl:+.2f}")
                print(f"📊 Exit Price: ${current_price:,.2f}")
                print(f"⏱️ Duration: {duration:.1f} minutes")
                print(f"🆔 Close Order ID: {close_order['id']}")
                
                # Log trade summary
                trade_summary = {
                    'entry_time': self.start_time.isoformat(),
                    'exit_time': datetime.now().isoformat(),
                    'symbol': self.config['symbol'],
                    'side': self.position_side,
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'size': self.position_size,
                    'pnl': final_pnl,
                    'duration_minutes': duration,
                    'close_reason': reason
                }
                
                self.logger.info(f"Trade completed: {json.dumps(trade_summary, indent=2)}")
                
                # Reset position data
                self.reset_position_data()
                
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to close position: {e}")
            return False
        
        return False
    
    def reset_position_data(self):
        """Reset all position tracking data"""
        self.position_id = None
        self.entry_price = 0
        self.position_side = None
        self.position_size = 0
        self.start_time = None
        self.stop_loss_price = 0
        self.take_profit_price = 0
        self.current_pnl = 0
        self.peak_pnl = 0
    
    async def monitor_position(self):
        """Monitor position and manage risk"""
        print(f"\n📊 POSITION MONITORING STARTED")
        print("=" * 35)
        
        monitor_count = 0
        
        while self.position_id:
            try:
                monitor_count += 1
                current_price = await self.get_market_price()
                current_pnl = await self.get_position_pnl()
                
                # Check stop conditions
                should_close, reason = await self.check_stop_conditions()
                
                # Display status every 10 iterations (30 seconds)
                if monitor_count % 10 == 0:
                    duration = (datetime.now() - self.start_time).total_seconds() / 60
                    print(f"\n📈 Status Update #{monitor_count//10}")
                    print(f"   💰 Current P&L: ${current_pnl:+.2f}")
                    print(f"   📊 Current Price: ${current_price:,.2f}")
                    print(f"   🏔️ Peak P&L: ${self.peak_pnl:+.2f}")
                    print(f"   ⏱️ Duration: {duration:.1f} min")
                    print(f"   🛑 SL: ${self.stop_loss_price:,.2f}")
                    print(f"   💎 TP: ${self.take_profit_price:,.2f}")
                
                # Close position if conditions met
                if should_close:
                    await self.close_position(reason)
                    break
                
                # Break even move logic
                if (self.config['break_even_move'] and 
                    current_pnl > 25 and  # If we're up $25+
                    self.stop_loss_price != self.entry_price):
                    
                    self.stop_loss_price = self.entry_price
                    print(f"🎯 Break-even move activated! SL moved to ${self.entry_price:,.2f}")
                
                await asyncio.sleep(3)  # Check every 3 seconds
                
            except Exception as e:
                self.logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(5)
    
    async def run_live_trading_cycle(self):
        """Run complete live trading cycle"""
        print("🚀 ALPHA LIVE POSITION MANAGER")
        print("=" * 50)
        print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💱 Symbol: {self.config['symbol']}")
        print(f"💰 Position Size: ${self.config['position_size_usd']}")
        print(f"🔥 Leverage: {self.config['leverage']}x")
        print(f"🛑 Stop Loss: {self.config['stop_loss_percent']}%")
        print(f"💎 Take Profit: {self.config['take_profit_percent']}%")
        print("=" * 50)
        
        # Step 1: Check connection
        if not await self.check_connection():
            print("❌ Connection check failed")
            return
        
        # Step 2: Open position
        if not await self.open_position('long'):  # Can be 'long' or 'short'
            print("❌ Failed to open position")
            return
        
        # Step 3: Monitor and manage position
        await self.monitor_position()
        
        print(f"\n🎉 TRADING CYCLE COMPLETED")
        print(f"📅 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)

async def main():
    """Main function"""
    try:
        manager = AlphaLivePositionManager()
        await manager.run_live_trading_cycle()
        
    except KeyboardInterrupt:
        print("\n🛑 Trading interrupted by user")
    except Exception as e:
        print(f"❌ Trading failed: {e}")
        logging.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 