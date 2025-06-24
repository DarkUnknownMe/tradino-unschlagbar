#!/usr/bin/env python3
"""
🧪 ADVANCED RISK MANAGEMENT SYSTEM TEST
Vollständiger Test des überarbeiteten Risk Management Systems
"""

import os
import sys
import time
from datetime import datetime, timedelta

# Add project path
sys.path.append('/root/tradino')
sys.path.append('/root/tradino/core')

def test_basic_functionality():
    """🔬 Test basic functionality"""
    print("🧪 Testing Advanced Risk Management System")
    print("=" * 50)
    
    try:
        from advanced_risk_management import (
            AdvancedRiskManager, 
            RiskSettings, 
            initialize_advanced_risk_manager
        )
        print("✅ Imports successful")
        
        # Initialize risk manager
        risk_manager = initialize_advanced_risk_manager()
        if risk_manager:
            print("✅ Risk Manager initialized")
        else:
            print("❌ Risk Manager initialization failed")
            return False
        
        # Test portfolio update
        risk_manager.update_portfolio_state(
            balance=10000.0,
            positions={'BTC/USDT:USDT': {'value': 2000, 'unrealized_pnl': 50}},
            recent_trades=[{
                'timestamp': datetime.now().isoformat(),
                'pnl': 25,
                'value': 1000
            }]
        )
        print("✅ Portfolio state updated")
        
        # Test trade validation
        test_signal = {
            'action': 'buy',
            'confidence': 0.75,
            'symbol': 'BTC/USDT:USDT'
        }
        
        validation = risk_manager.validate_trade_realtime(
            signal=test_signal,
            symbol='BTC/USDT:USDT',
            current_price=45000.0
        )
        
        print(f"✅ Trade validation: {validation['approved']}")
        print(f"   Risk Score: {validation['risk_score']:.1f}")
        print(f"   Position Size: {validation['position_size']:.6f}")
        print(f"   Stop Loss: ${validation['adjusted_sl']:.2f}")
        print(f"   Take Profit: ${validation['adjusted_tp']:.2f}")
        
        # Test dashboard
        dashboard = risk_manager.get_risk_dashboard()
        print(f"✅ Dashboard generated")
        print(f"   Risk Level: {dashboard['risk_level']}")
        print(f"   Balance: ${dashboard['portfolio_state']['total_balance']:.2f}")
        
        risk_manager.stop_monitoring()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    print("\n" + "=" * 50)
    if success:
        print("🎉 Advanced Risk Management System is working!")
    else:
        print("❌ System has issues that need to be fixed")
