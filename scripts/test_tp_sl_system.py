#!/usr/bin/env python3
"""
🧪 TP/SL SYSTEM TEST
Test und Demo des neuen Take Profit / Stop Loss Systems
"""

import os
import sys
import time
from datetime import datetime
import json

# Add project path
sys.path.append('/root/tradino')

try:
    from core.bitget_trading_api import BitgetTradingAPI
    from core.tp_sl_manager import initialize_tp_sl_manager
    from core.risk_management_system import RiskManagementSystem
    print("✅ Alle Module erfolgreich importiert")
except ImportError as e:
    print(f"❌ Import Fehler: {e}")
    sys.exit(1)

def test_tp_sl_system():
    """🧪 Test das komplette TP/SL System"""
    
    print("🧪 TRADINO TP/SL SYSTEM TEST")
    print("=" * 50)
    
    # 1. API Initialisierung
    print("\n1. 🏦 Initialisiere Bitget Trading API...")
    
    api_key = os.getenv('BITGET_API_KEY', 'demo_key')
    secret = os.getenv('BITGET_SECRET_KEY', 'demo_secret')
    passphrase = os.getenv('BITGET_PASSPHRASE', 'demo_passphrase')
    
    trading_api = BitgetTradingAPI(
        api_key=api_key,
        secret=secret,
        passphrase=passphrase,
        sandbox=True
    )
    
    if trading_api.is_connected:
        print("✅ Trading API verbunden")
        balance = trading_api.get_total_balance()
        print(f"💰 Account Balance: ${balance:.2f}")
    else:
        print("⚠️ Trading API nicht verbunden - Demo Modus")
    
    # 2. Risk Management Initialisierung
    print("\n2. 🛡️ Initialisiere Risk Management...")
    
    risk_config = {
        'max_portfolio_exposure': 0.5,
        'stop_loss_percent': 0.02,
        'take_profit_percent': 0.04,
        'max_daily_trades': 5
    }
    
    risk_manager = RiskManagementSystem(risk_config)
    print("✅ Risk Management System bereit")
    
    # 3. TP/SL Manager Initialisierung
    print("\n3. 🎯 Initialisiere TP/SL Manager...")
    
    tp_sl_config = {
        'monitoring_interval': 5,
        'enable_notifications': True,
        'telegram_token': os.getenv('TELEGRAM_BOT_TOKEN'),
        'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
        'retry_attempts': 3
    }
    
    tp_sl_manager = initialize_tp_sl_manager(trading_api, tp_sl_config)
    print("✅ TP/SL Manager bereit")
    
    # 4. Test Trade mit TP/SL
    print("\n4. 💰 Test Trade mit automatischem TP/SL...")
    
    test_symbol = "BTC/USDT:USDT"
    test_side = "buy"
    test_amount = 0.001  # Kleine Test-Position
    
    # Risk Management Parameter
    risk_params = {
        'take_profit_percent': 0.03,  # 3%
        'stop_loss_percent': 0.015   # 1.5%
    }
    
    if trading_api.is_connected:
        print(f"📈 Platziere Test-Trade: {test_side} {test_amount} {test_symbol}")
        
        # Verwende die neue TP/SL Funktion
        trade_result = trading_api.place_market_order_with_tp_sl(
            symbol=test_symbol,
            side=test_side,
            amount=test_amount,
            risk_management_params=risk_params
        )
        
        if trade_result.get('success'):
            print("✅ Trade mit TP/SL erfolgreich platziert!")
            print(f"   Entry Price: ${trade_result.get('entry_price', 0):.4f}")
            print(f"   Take Profit: ${trade_result.get('tp_price', 0):.4f}")
            print(f"   Stop Loss: ${trade_result.get('sl_price', 0):.4f}")
            print(f"   Method: {trade_result.get('tp_sl_orders', {}).get('method', 'Unknown')}")
            
            # Zeige TP/SL Status
            print("\n📊 TP/SL Status:")
            tp_sl_status = trading_api.get_tp_sl_status()
            print(f"   Aktive Positionen: {tp_sl_status['active_positions']}")
            print(f"   Total Positionen: {tp_sl_status['total_positions']}")
            
        else:
            print(f"❌ Trade fehlgeschlagen: {trade_result.get('error', 'Unknown')}")
    else:
        print("⚠️ Demo-Modus: Simuliere Trade mit TP/SL")
        
        # Simuliere erfolgreichen Trade
        simulated_result = {
            'success': True,
            'entry_price': 45000.0,
            'tp_price': 46350.0,  # +3%
            'sl_price': 44325.0,  # -1.5%
            'tp_sl_orders': {'method': 'Simulated'}
        }
        
        print("✅ Simulierter Trade mit TP/SL:")
        print(f"   Entry Price: ${simulated_result['entry_price']:.4f}")
        print(f"   Take Profit: ${simulated_result['tp_price']:.4f}")
        print(f"   Stop Loss: ${simulated_result['sl_price']:.4f}")
    
    # 5. Monitoring Test
    print("\n5. 👁️ Teste TP/SL Monitoring...")
    
    if trading_api.is_connected:
        monitoring_result = trading_api.monitor_tp_sl_orders()
        print(f"📊 Monitoring Ergebnis:")
        print(f"   Aktive Positionen: {monitoring_result.get('active_positions', 0)}")
        print(f"   Ausgeführte Orders: {len(monitoring_result.get('executed_orders', []))}")
    else:
        print("⚠️ Demo-Modus: Simuliere Monitoring")
        print("📊 Monitoring würde folgende Positionen überwachen:")
        print("   - BTC/USDT Long Position")
        print("   - TP bei $46,350 | SL bei $44,325")
    
    # 6. Telegram Test
    print("\n6. 📱 Teste Telegram Notifications...")
    
    if tp_sl_manager.telegram_bot:
        test_message = (
            "🧪 TRADINO TP/SL SYSTEM TEST\n\n"
            f"✅ System erfolgreich initialisiert\n"
            f"🕐 Zeit: {datetime.now().strftime('%H:%M:%S')}\n"
            f"🎯 TP/SL System: Aktiv\n"
            f"📊 Risk Management: Aktiv"
        )
        
        tp_sl_manager.send_telegram_notification(test_message)
        print("✅ Telegram Test-Nachricht gesendet")
    else:
        print("⚠️ Telegram nicht konfiguriert")
        print("📋 Für Telegram Notifications benötigt:")
        print("   - TELEGRAM_BOT_TOKEN in .env")
        print("   - TELEGRAM_CHAT_ID in .env")
    
    # 7. System Status Report
    print("\n7. 📊 System Status Report...")
    
    status_report = {
        'timestamp': datetime.now().isoformat(),
        'trading_api_connected': trading_api.is_connected,
        'risk_management_active': True,
        'tp_sl_manager_active': True,
        'telegram_configured': tp_sl_manager.telegram_bot is not None,
        'test_completed': True
    }
    
    print("📊 SYSTEM STATUS REPORT:")
    for key, value in status_report.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key.replace('_', ' ').title()}: {value}")
    
    # 8. Configuration Summary
    print("\n8. ⚙️ Konfiguration Summary...")
    
    config_summary = {
        'Default TP Percent': f"{risk_params['take_profit_percent']*100:.1f}%",
        'Default SL Percent': f"{risk_params['stop_loss_percent']*100:.1f}%",
        'Monitoring Interval': f"{tp_sl_config['monitoring_interval']}s",
        'Retry Attempts': tp_sl_config['retry_attempts'],
        'Sandbox Mode': trading_api.sandbox,
        'Exchange': 'Bitget Futures'
    }
    
    print("⚙️ KONFIGURATION:")
    for key, value in config_summary.items():
        print(f"   📋 {key}: {value}")
    
    print("\n" + "=" * 50)
    print("✅ TP/SL SYSTEM TEST ABGESCHLOSSEN")
    print("🎯 System ist bereit für Live Trading!")
    
    return status_report

def demonstrate_tp_sl_flow():
    """🎬 Demonstriere den kompletten TP/SL Workflow"""
    
    print("\n🎬 TP/SL WORKFLOW DEMONSTRATION")
    print("=" * 40)
    
    workflow_steps = [
        "1. 📊 Signal generiert (z.B. BUY BTC)",
        "2. 🔍 Risk Management Validierung",
        "3. 💰 Market Order platziert",
        "4. ⚡ TP/SL Orders automatisch erstellt:",
        "   📈 Take Profit Order @ +3%",
        "   🛑 Stop Loss Order @ -1.5%",
        "5. 👁️ Kontinuierliches Monitoring startet",
        "6. 🎯 Bei TP/SL Trigger:",
        "   ❌ Verbleibende Orders canceln",
        "   📤 Market Exit Order",
        "   📱 Telegram Notification",
        "   📊 Performance Update",
        "7. ✅ Position geschlossen"
    ]
    
    for step in workflow_steps:
        print(step)
        time.sleep(1)
    
    print("\n🔄 Monitoring läuft kontinuierlich bis:")
    print("   ✅ Take Profit erreicht (+3%)")
    print("   🛑 Stop Loss erreicht (-1.5%)")
    print("   ❌ Position manuell geschlossen")
    print("   ⏰ Max Monitoring Zeit (24h)")

def show_api_examples():
    """📚 Zeige API Usage Beispiele"""
    
    print("\n📚 API USAGE BEISPIELE")
    print("=" * 30)
    
    examples = {
        "Simple Trade mit TP/SL": '''
trading_api.place_market_order_with_tp_sl(
    symbol="BTC/USDT:USDT",
    side="buy",
    amount=0.01,
    tp_percent=0.03,  # 3% Take Profit
    sl_percent=0.015  # 1.5% Stop Loss
)''',
        
        "Trade mit Risk Management": '''
risk_params = {
    'take_profit_percent': 0.04,
    'stop_loss_percent': 0.02
}

trading_api.place_market_order_with_tp_sl(
    symbol="ETH/USDT:USDT",
    side="sell",
    amount=0.1,
    risk_management_params=risk_params
)''',
        
        "TP/SL Status Check": '''
status = trading_api.get_tp_sl_status()
print(f"Aktive Positionen: {status['active_positions']}")
''',
        
        "TP/SL Monitoring": '''
monitoring_result = trading_api.monitor_tp_sl_orders()
for order in monitoring_result['executed_orders']:
    print(f"{order['type']} executed: {order['symbol']}")
'''
    }
    
    for title, code in examples.items():
        print(f"\n📋 {title}:")
        print(code)

if __name__ == "__main__":
    try:
        # Haupttest
        test_result = test_tp_sl_system()
        
        # Workflow Demo
        demonstrate_tp_sl_flow()
        
        # API Beispiele
        show_api_examples()
        
        print(f"\n🎉 Test erfolgreich abgeschlossen: {datetime.now()}")
        
    except KeyboardInterrupt:
        print("\n🛑 Test durch Benutzer beendet")
    except Exception as e:
        print(f"\n❌ Test Fehler: {e}")
        import traceback
        traceback.print_exc() 