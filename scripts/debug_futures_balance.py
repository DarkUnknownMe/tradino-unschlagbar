#!/usr/bin/env python3
"""
🔍 DEBUG FUTURES BALANCE
Analysiert das Balance-Problem bei Bitget Futures
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment
load_dotenv('tradino_unschlagbar/.env')

# Add project path
sys.path.append('/root/tradino')

from bitget_trading_api import BitgetTradingAPI

def debug_balance():
    """🔍 Debug balance information"""
    
    print("🔍 BITGET FUTURES BALANCE DEBUG")
    print("="*50)
    
    # Initialize API
    api_key = os.getenv('BITGET_API_KEY')
    secret = os.getenv('BITGET_SECRET_KEY')
    passphrase = os.getenv('BITGET_PASSPHRASE')
    
    print(f"🔑 API Key: {api_key[:10]}...")
    print(f"🔑 Secret: {secret[:10]}...")
    print(f"🔑 Passphrase: {passphrase[:5]}...")
    
    # Connect to API
    api = BitgetTradingAPI(
        api_key=api_key,
        secret=secret,
        passphrase=passphrase,
        sandbox=True
    )
    
    if not api.is_connected:
        print("❌ API not connected")
        return
    
    print("✅ API connected successfully")
    print("="*50)
    
    try:
        # Get different types of balance info
        print("📊 TESTING DIFFERENT BALANCE CALLS:")
        
        # 1. Standard balance
        try:
            balance_std = api.exchange.fetch_balance()
            print(f"\n1️⃣ STANDARD BALANCE:")
            print(f"   Total USDT: ${balance_std.get('USDT', {}).get('total', 0):.2f}")
            print(f"   Free USDT: ${balance_std.get('USDT', {}).get('free', 0):.2f}")
            print(f"   Used USDT: ${balance_std.get('USDT', {}).get('used', 0):.2f}")
        except Exception as e:
            print(f"1️⃣ Standard balance error: {e}")
        
        # 2. Futures balance
        try:
            balance_futures = api.exchange.fetch_balance({'type': 'swap'})
            print(f"\n2️⃣ FUTURES BALANCE:")
            print(f"   Total USDT: ${balance_futures.get('USDT', {}).get('total', 0):.2f}")
            print(f"   Free USDT: ${balance_futures.get('USDT', {}).get('free', 0):.2f}")
            print(f"   Used USDT: ${balance_futures.get('USDT', {}).get('used', 0):.2f}")
            
            # Show all currencies
            print(f"\n   All currencies in futures:")
            for currency, info in balance_futures.items():
                if isinstance(info, dict) and info.get('total', 0) > 0:
                    print(f"   {currency}: Total=${info.get('total', 0):.2f}, Free=${info.get('free', 0):.2f}")
                    
        except Exception as e:
            print(f"2️⃣ Futures balance error: {e}")
        
        # 3. Test a small trade
        print(f"\n3️⃣ TESTING SMALL TRADE ORDER:")
        try:
            # Test with very small amount
            result = api.place_market_order('BTC/USDT', 'buy', 0.001)  # $100 worth
            if result.get('success'):
                print("✅ Small trade would succeed")
            else:
                print(f"❌ Small trade failed: {result.get('error')}")
        except Exception as e:
            print(f"❌ Trade test error: {e}")
        
        # 4. Check account type and permissions
        print(f"\n4️⃣ ACCOUNT TYPE & PERMISSIONS:")
        try:
            # Get account info
            account_info = api.exchange.fetch_balance({'type': 'swap'})
            print(f"   Account type: {type(account_info)}")
            print(f"   Keys in response: {list(account_info.keys()) if isinstance(account_info, dict) else 'Not dict'}")
            
            # Check if we have trading permissions
            markets = api.exchange.load_markets()
            btc_market = markets.get('BTC/USDT')
            if btc_market:
                print(f"   BTC/USDT market type: {btc_market.get('type')}")
                print(f"   Min trade amount: {btc_market.get('limits', {}).get('amount', {}).get('min', 'Unknown')}")
            
        except Exception as e:
            print(f"4️⃣ Account info error: {e}")
        
        # 5. Raw API call
        print(f"\n5️⃣ RAW API RESPONSE:")
        try:
            raw_response = api.exchange.privateGetMixV1AccountAccounts()
            print(f"   Raw response: {json.dumps(raw_response, indent=2)}")
        except Exception as e:
            print(f"5️⃣ Raw API error: {e}")
            
    except Exception as e:
        print(f"❌ Overall error: {e}")
    
    print("\n" + "="*50)
    print("🔍 Debug complete")

if __name__ == "__main__":
    debug_balance() 