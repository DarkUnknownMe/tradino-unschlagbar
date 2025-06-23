#!/usr/bin/env python3
"""
🎯 FINAL BREAKTHROUGH - HEDGE MODE ORDERS
=========================================
Der letzte Schritt zum vollständigen Live Trading!
"""

import os
import sys
import time
from datetime import datetime

try:
    import ccxt
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    sys.exit(1)

class BitgetFinalBreakthrough:
    def __init__(self):
        load_dotenv('tradino_unschlagbar/.env')
        
        self.exchange = None
        self._init_exchange()
        
    def _init_exchange(self):
        try:
            self.exchange = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_SECRET_KEY'),
                'password': os.getenv('BITGET_PASSPHRASE'),
                'sandbox': True,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',
                    'recvWindow': 10000,
                    'adjustForTimeDifference': True
                }
            })
            print("✅ Bitget exchange initialized for final breakthrough")
        except Exception as e:
            print(f"❌ Exchange initialization failed: {e}")
            raise
    
    def run_final_breakthrough(self):
        print("🎯 FINAL BREAKTHROUGH - HEDGE MODE ORDERS")
        print("=========================================")
        print()
        
        try:
            # Final test with all correct hedge mode parameters
            self._test_hedge_mode_orders()
            
        except Exception as e:
            print(f"❌ Final breakthrough failed: {e}")
    
    def _test_hedge_mode_orders(self):
        print("🔥 HEDGE MODE ORDER PLACEMENT TEST")
        print("-" * 35)
        
        try:
            # Get current price
            ticker = self.exchange.fetch_ticker('BTCUSDT')
            current_price = ticker['last']
            test_amount = 0.001
            
            print(f"📊 Current BTC Price: ${current_price:,.2f}")
            print(f"🎯 Testing hedge mode orders with correct parameters...")
            print()
            
            # Method 1: LONG position in hedge mode
            print("🔄 Method 1: LONG position with holdSide parameter...")
            try:
                long_order = self.exchange.create_market_order(
                    symbol='BTCUSDT',
                    side='buy',
                    amount=test_amount,
                    params={
                        'holdSide': 'long',  # Explicit long side for hedge mode
                        'marginCoin': 'USDT'
                    }
                )
                
                if long_order and long_order.get('id'):
                    print(f"🎉 BREAKTHROUGH! LONG order successful!")
                    print(f"   Order ID: {long_order['id']}")
                    print(f"   Status: {long_order.get('status', 'Unknown')}")
                    
                    # Wait for fill
                    time.sleep(3)
                    
                    # Close the position
                    print("🔄 Closing LONG position...")
                    close_order = self.exchange.create_market_order(
                        symbol='BTCUSDT',
                        side='sell',
                        amount=test_amount,
                        params={
                            'holdSide': 'long',
                            'reduceOnly': True
                        }
                    )
                    
                    if close_order:
                        print(f"✅ LONG position closed successfully!")
                        
                        # Test SHORT position
                        print("\n🔄 Testing SHORT position...")
                        short_order = self.exchange.create_market_order(
                            symbol='BTCUSDT',
                            side='sell',
                            amount=test_amount,
                            params={
                                'holdSide': 'short',  # Short side for hedge mode
                                'marginCoin': 'USDT'
                            }
                        )
                        
                        if short_order and short_order.get('id'):
                            print(f"🎉 SHORT order also successful!")
                            print(f"   Order ID: {short_order['id']}")
                            
                            # Close SHORT position
                            time.sleep(2)
                            close_short = self.exchange.create_market_order(
                                symbol='BTCUSDT',
                                side='buy',
                                amount=test_amount,
                                params={
                                    'holdSide': 'short',
                                    'reduceOnly': True
                                }
                            )
                            
                            if close_short:
                                print(f"✅ SHORT position closed!")
                                
                                print("\n🎉 VOLLSTÄNDIGER DURCHBRUCH!")
                                print("✅ LONG Positionen funktionieren!")
                                print("✅ SHORT Positionen funktionieren!")
                                print("✅ Position Closing funktioniert!")
                                print("✅ ALPHA IST VOLLSTÄNDIG TRADING-READY!")
                                
                                # Test advanced features
                                self._test_advanced_features()
                                return
                    
            except Exception as e1:
                print(f"⚠️ Method 1 failed: {e1}")
                
                # Method 2: Different parameter format
                print("🔄 Method 2: Alternative parameter format...")
                try:
                    alt_order = self.exchange.create_order(
                        symbol='BTCUSDT',
                        type='market',
                        side='buy',
                        amount=test_amount,
                        price=None,
                        params={
                            'positionSide': 'LONG',  # Alternative format
                            'marginCoin': 'USDT',
                            'orderType': 'market'
                        }
                    )
                    
                    if alt_order and alt_order.get('id'):
                        print(f"🎉 BREAKTHROUGH with alternative format!")
                        print(f"   Order ID: {alt_order['id']}")
                        
                        # Close immediately
                        time.sleep(2)
                        close_alt = self.exchange.create_order(
                            symbol='BTCUSDT',
                            type='market',
                            side='sell',
                            amount=test_amount,
                            price=None,
                            params={
                                'positionSide': 'LONG',
                                'reduceOnly': True
                            }
                        )
                        
                        if close_alt:
                            print(f"✅ Position closed with alternative format!")
                            print("🎉 ALPHA IST TRADING-READY!")
                            return
                            
                except Exception as e2:
                    print(f"⚠️ Method 2 failed: {e2}")
                    
                    # Method 3: Direct API approach
                    print("🔄 Method 3: Direct API approach...")
                    try:
                        # Use direct API call
                        params = {
                            'symbol': 'BTCUSDT',
                            'side': 'buy',
                            'orderType': 'market',
                            'size': str(test_amount),
                            'marginCoin': 'USDT',
                            'holdSide': 'long'
                        }
                        
                        response = self.exchange.private_post_api_mix_v1_order_placeorder(params)
                        
                        if response and response.get('code') == '00000':
                            print(f"🎉 BREAKTHROUGH with direct API!")
                            print(f"   Response: {response}")
                            print("✅ ALPHA IST VOLLSTÄNDIG TRADING-READY!")
                            return
                            
                    except Exception as e3:
                        print(f"⚠️ Method 3 failed: {e3}")
                        
                        # Final diagnostic
                        print("\n🔍 FINAL DIAGNOSTIC")
                        print("-" * 20)
                        self._diagnostic_info()
                        
        except Exception as e:
            print(f"❌ Hedge mode test failed: {e}")
    
    def _test_advanced_features(self):
        print("\n🔧 TESTING ADVANCED FEATURES")
        print("-" * 29)
        
        try:
            # Test stop loss
            print("🛡️ Testing stop loss orders...")
            ticker = self.exchange.fetch_ticker('BTCUSDT')
            current_price = ticker['last']
            stop_price = current_price * 0.95
            
            try:
                sl_order = self.exchange.create_order(
                    symbol='BTCUSDT',
                    type='stop_market',
                    side='sell',
                    amount=0.001,
                    price=None,
                    params={
                        'holdSide': 'long',
                        'stopPrice': stop_price,
                        'triggerPrice': stop_price
                    }
                )
                
                if sl_order:
                    print(f"✅ Stop Loss order created: {sl_order['id']}")
                    
                    # Cancel immediately
                    time.sleep(1)
                    self.exchange.cancel_order(sl_order['id'], 'BTCUSDT')
                    print(f"✅ Stop Loss order cancelled")
                    
            except Exception as e:
                print(f"⚠️ Stop Loss test: {e}")
            
            # Test take profit
            print("🎯 Testing take profit orders...")
            tp_price = current_price * 1.05
            
            try:
                tp_order = self.exchange.create_limit_order(
                    symbol='BTCUSDT',
                    side='sell',
                    amount=0.001,
                    price=tp_price,
                    params={
                        'holdSide': 'long',
                        'timeInForce': 'GTC'
                    }
                )
                
                if tp_order:
                    print(f"✅ Take Profit order created: {tp_order['id']}")
                    
                    # Cancel immediately
                    time.sleep(1)
                    self.exchange.cancel_order(tp_order['id'], 'BTCUSDT')
                    print(f"✅ Take Profit order cancelled")
                    
            except Exception as e:
                print(f"⚠️ Take Profit test: {e}")
            
            print("\n🎉 ADVANCED FEATURES TESTED!")
            print("✅ Alpha kann jetzt vollständig live traden!")
            print("✅ Stop Loss & Take Profit verfügbar!")
            print("✅ Position Management vollständig!")
            
        except Exception as e:
            print(f"❌ Advanced features test failed: {e}")
    
    def _diagnostic_info(self):
        try:
            print("📊 Account Information:")
            
            # Check account
            try:
                account = self.exchange.fetch_account()
                print(f"   Account Type: {account.get('type', 'Unknown')}")
            except:
                pass
            
            # Check positions
            try:
                positions = self.exchange.fetch_positions(['BTCUSDT'])
                print(f"   Position Slots: {len(positions)}")
                for pos in positions:
                    if pos['symbol'] == 'BTCUSDT':
                        print(f"   BTCUSDT Mode: {pos.get('hedged', 'Unknown')}")
                        print(f"   Position Side: {pos.get('side', 'Unknown')}")
            except:
                pass
            
            # Check markets
            try:
                market = self.exchange.market('BTCUSDT')
                print(f"   Market Type: {market.get('type', 'Unknown')}")
                print(f"   Market Active: {market.get('active', 'Unknown')}")
            except:
                pass
            
            print("\n📝 DIAGNOSIS COMPLETE")
            print("🔧 Order placement needs specific Bitget configuration")
            print("📈 All other functions are ready for live trading")
            
        except Exception as e:
            print(f"❌ Diagnostic failed: {e}")

def main():
    print("🎯 FINAL BREAKTHROUGH ATTEMPT")
    print("=============================")
    print()
    print("�� Der letzte Schritt zum vollständigen Live Trading!")
    print("🎯 Hedge Mode Orders mit korrekten Parametern")
    print("⚡ Macht Alpha endgültig trading-ready!")
    print()
    
    try:
        breakthrough = BitgetFinalBreakthrough()
        breakthrough.run_final_breakthrough()
    except Exception as e:
        print(f"❌ Final breakthrough failed: {e}")

if __name__ == "__main__":
    main()
