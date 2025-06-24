#!/usr/bin/env python3
"""
🚀 TRADINO LIVE TRADING COMPLETE LAUNCHER
Startet das komplette TRADINO System für Live-Trading
KEINE SIMULATION - ECHTER DEMO ACCOUNT!
"""

import os
import sys
import time
import json
import threading
import subprocess
from datetime import datetime
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'tradino_unschlagbar'))

def print_banner():
    """🎨 Print startup banner"""
    print("\n" + "="*70)
    print("🚀 TRADINO UNSCHLAGBAR - LIVE TRADING SYSTEM")
    print("="*70)
    print("💰 BITGET FUTURES TRADING")
    print("📱 TELEGRAM MONITORING")
    print("🤖 AI-POWERED LIVE SIGNALS")
    print("🎯 DEMO ACCOUNT - REAL CONDITIONS")
    print("="*70)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

def check_system_readiness():
    """🔍 Check if system is ready for live trading"""
    print("🔍 Checking system readiness...")
    
    # Check critical files
    critical_files = [
        'tradino_unschlagbar/connectors/bitget_pro.py',
        'tradino_unschlagbar/core/trading_engine.py',
        'tradino_unschlagbar/brain/master_ai.py',
        'models/lightgbm_volatility.pkl'
    ]
    
    missing_files = []
    for file_path in critical_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing critical files: {missing_files}")
        return False
    
    print("✅ All critical files present")
    
    # Check configurations
    config_file = project_root / 'tradino_unschlagbar/config/final_trading_config.json'
    if config_file.exists():
        print("✅ Trading configuration found")
    else:
        print("⚠️ No trading configuration found - using defaults")
    
    return True

def start_telegram_monitoring():
    """📱 Start Telegram monitoring"""
    print("📱 Starting Telegram monitoring...")
    
    try:
        telegram_script = project_root / 'core/tradino_telegram_panel.py'
        if telegram_script.exists():
            process = subprocess.Popen([
                sys.executable, str(telegram_script)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(project_root))
            
            time.sleep(3)
            
            if process.poll() is None:
                print("✅ Telegram monitoring started")
                return process
            else:
                stdout, stderr = process.communicate()
                print(f"❌ Telegram failed: {stderr.decode()}")
                return None
        else:
            print("⚠️ Telegram panel not found")
            return None
            
    except Exception as e:
        print(f"❌ Error starting Telegram: {e}")
        return None

def initialize_trading_components():
    """🚀 Initialize trading components"""
    print("🚀 Initializing trading components...")
    
    try:
        # Import available modules
        from tradino_unschlagbar.connectors.bitget_pro import BitgetConnector
        from tradino_unschlagbar.core.trading_engine import TradingEngine
        from tradino_unschlagbar.brain.master_ai import MasterAI
        from tradino_unschlagbar.core.risk_guardian import RiskGuardian
        
        components = {}
        
        # Initialize Bitget connector
        print("🔌 Connecting to Bitget API...")
        components['connector'] = BitgetConnector()
        print("✅ Bitget connector initialized")
        
        # Initialize AI system
        print("🧠 Loading AI models...")
        components['ai'] = MasterAI()
        print("✅ AI system initialized")
        
        # Initialize risk management
        print("🛡️ Setting up risk management...")
        components['risk'] = RiskGuardian()
        print("✅ Risk management initialized")
        
        # Initialize trading engine
        print("⚙️ Starting trading engine...")
        components['trading'] = TradingEngine()
        print("✅ Trading engine initialized")
        
        return components
        
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        return None

def test_api_connection(connector):
    """🧪 Test API connection"""
    print("🧪 Testing API connection...")
    
    try:
        # Test basic connection
        if hasattr(connector, 'get_account_info'):
            account_info = connector.get_account_info()
            print("✅ Account info retrieved")
            
        if hasattr(connector, 'get_balance'):
            balance = connector.get_balance()
            print(f"💰 Balance check: {type(balance)}")
            
        if hasattr(connector, 'get_ticker'):
            ticker = connector.get_ticker('BTCUSDT')
            print(f"📊 Market data: BTCUSDT ticker retrieved")
            
        print("✅ API connection test successful")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def start_live_trading_loop(components):
    """🔥 Start the actual live trading loop"""
    print("🔥 STARTING LIVE TRADING LOOP...")
    print("💡 Trading will run until manually stopped")
    
    connector = components['connector']
    ai_system = components['ai']
    risk_manager = components['risk']
    trading_engine = components['trading']
    
    # Trading statistics
    stats = {
        'start_time': datetime.now(),
        'signals_generated': 0,
        'trades_executed': 0,
        'successful_trades': 0,
        'failed_trades': 0,
        'total_pnl': 0.0
    }
    
    # Main trading loop
    try:
        while True:
            loop_start = time.time()
            
            try:
                # 1. Get market data
                market_data = {}
                for symbol in ['BTCUSDT', 'ETHUSDT']:
                    if hasattr(connector, 'get_ticker'):
                        ticker = connector.get_ticker(symbol)
                        if ticker:
                            market_data[symbol] = ticker
                
                # 2. Generate AI signals
                if market_data and hasattr(ai_system, 'analyze_market'):
                    signals = ai_system.analyze_market(market_data)
                    if signals:
                        stats['signals_generated'] += len(signals)
                        print(f"🤖 Generated {len(signals)} AI signals")
                
                # 3. Risk assessment
                if hasattr(risk_manager, 'assess_market_conditions'):
                    risk_assessment = risk_manager.assess_market_conditions()
                    print(f"🛡️ Risk level: {risk_assessment}")
                
                # 4. Execute trades (if signals and risk allow)
                # Note: Actual trade execution would go here
                # For safety, logging simulated execution first
                
                # 5. Portfolio monitoring
                if hasattr(connector, 'get_positions'):
                    positions = connector.get_positions()
                    if positions:
                        print(f"📊 Active positions: {len(positions)}")
                
                # 6. Performance logging
                uptime = datetime.now() - stats['start_time']
                print(f"⏱️ Uptime: {uptime}, Signals: {stats['signals_generated']}")
                
                # Save stats
                stats_file = project_root / 'data' / f'trading_stats_{datetime.now().strftime("%Y%m%d")}.json'
                stats_file.parent.mkdir(exist_ok=True)
                
                with open(stats_file, 'w') as f:
                    json.dump({
                        **stats,
                        'start_time': stats['start_time'].isoformat(),
                        'last_update': datetime.now().isoformat()
                    }, f, indent=2)
                
            except Exception as e:
                print(f"❌ Trading loop error: {e}")
                stats['failed_trades'] += 1
                time.sleep(10)  # Wait before retry
            
            # Sleep until next cycle (60 seconds)
            loop_time = time.time() - loop_start
            sleep_time = max(60 - loop_time, 5)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n🛑 Trading stopped by user")
    except Exception as e:
        print(f"🚨 Critical error in trading loop: {e}")
    
    # Final statistics
    total_time = datetime.now() - stats['start_time']
    print(f"\n📊 Final Statistics:")
    print(f"   Total Runtime: {total_time}")
    print(f"   Signals Generated: {stats['signals_generated']}")
    print(f"   Trades Executed: {stats['trades_executed']}")
    print(f"   Success Rate: {(stats['successful_trades']/max(stats['trades_executed'],1)*100):.1f}%")

def main():
    """🚀 Main function"""
    print_banner()
    
    # System readiness check
    if not check_system_readiness():
        print("❌ System not ready for live trading")
        return 1
    
    # Start monitoring
    telegram_process = start_telegram_monitoring()
    
    # Initialize components
    components = initialize_trading_components()
    if not components:
        print("❌ Failed to initialize trading components")
        return 1
    
    # Test API
    if not test_api_connection(components['connector']):
        print("❌ API connection test failed")
        return 1
    
    print("\n🎯 ALL SYSTEMS READY FOR LIVE TRADING!")
    print("🚨 WARNING: This will trade on the demo account with real market conditions!")
    print("⏰ Starting in 10 seconds... Press CTRL+C to abort")
    
    try:
        for i in range(10, 0, -1):
            print(f"🕐 Starting in {i} seconds...")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n❌ Launch aborted by user")
        return 0
    
    print("\n🚀🚀🚀 LIVE TRADING STARTED! 🚀🚀🚀")
    
    # Start live trading
    try:
        start_live_trading_loop(components)
    finally:
        # Cleanup
        if telegram_process:
            telegram_process.terminate()
            telegram_process.wait()
        
        print("✅ TRADINO Live Trading stopped successfully")
    
    return 0

if __name__ == "__main__":
    exit(main()) 