#!/usr/bin/env python3
"""
🚀 TRADINO COMPLETE LAUNCHER
Startet Trading System + Telegram Panel
"""

import os
import sys
import time
import threading
import subprocess
from datetime import datetime

# Add project path
sys.path.append('/root/tradino')

def print_banner():
    """🎨 Print startup banner"""
    print("\n" + "="*60)
    print("🚀 TRADINO UNSCHLAGBAR - COMPLETE SYSTEM")
    print("="*60)
    print("💰 FUTURES TRADING SYSTEM")
    print("📱 TELEGRAM CONTROL PANEL")
    print("🤖 AI-POWERED SIGNALS")
    print("="*60)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

def check_env_file():
    """🔍 Check .env file"""
    env_path = 'tradino_unschlagbar/.env'
    if not os.path.exists(env_path):
        print(f"❌ .env file not found at {env_path}")
        return False
    
    # Load and check required variables
    from dotenv import load_dotenv
    load_dotenv(env_path)
    
    required_vars = [
        'BITGET_API_KEY',
        'BITGET_SECRET_KEY',
        'BITGET_PASSPHRASE',
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ All environment variables found")
    return True

def start_telegram_bot():
    """📱 Start Telegram bot in background"""
    print("📱 Starting Telegram Control Panel...")
    
    try:
        # Start telegram bot as subprocess
        process = subprocess.Popen([
            'python3', 'core/tradino_telegram_panel.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(3)  # Give it time to start
        
        if process.poll() is None:  # Still running
            print("✅ Telegram Panel started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Telegram Panel failed to start")
            print(f"Error: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting Telegram Panel: {e}")
        return None

def start_trading_system():
    """🚀 Start trading system"""
    print("🚀 Starting Trading System...")
    
    try:
        from final_live_trading_system import start_real_tradino
        
        # Initialize trading system
        system = start_real_tradino()
        
        if system and system.is_initialized:
            print("✅ Trading System initialized")
            
            # Show system status
            status = system.get_real_status()
            print(f"📊 AI Models: {'✅' if status['ai_ready'] else '❌'}")
            print(f"📊 Market Feed: {'✅' if status['market_feed_connected'] else '❌'}")
            print(f"🏦 Trading API: {'✅' if status['trading_api_connected'] else '❌'}")
            
            if system.trading_api:
                balance = system.trading_api.get_total_balance()
                print(f"💰 FUTURES Balance: ${balance:.2f}")
            
            return system
        else:
            print("❌ Trading System initialization failed")
            return None
            
    except Exception as e:
        print(f"❌ Error starting Trading System: {e}")
        return None

def show_system_info(trading_system):
    """📊 Show system information"""
    print("\n" + "="*60)
    print("📊 SYSTEM STATUS")
    print("="*60)
    
    if trading_system:
        status = trading_system.get_real_status()
        print(f"🤖 AI Models: {'Ready' if status['ai_ready'] else 'Not Ready'}")
        print(f"📊 Market Feed: {'Connected' if status['market_feed_connected'] else 'Disconnected'}")
        print(f"🏦 Trading API: {'Connected' if status['trading_api_connected'] else 'Disconnected'}")
        print(f"🎯 Trading Mode: FUTURES")
        
        if trading_system.trading_api:
            balance = trading_system.trading_api.get_total_balance()
            sandbox_mode = "SANDBOX" if trading_system.trading_api.sandbox else "LIVE"
            print(f"💰 Account Balance: ${balance:.2f} ({sandbox_mode})")
    
    print("📱 Telegram Panel: Running")
    print("="*60)
    
    print("\n💡 CONTROLS:")
    print("📱 Use Telegram: /start to control system")
    print("⌨️  Terminal: Press CTRL+C to stop")
    print("🚀 Auto-start: Trading will begin automatically")
    print("="*60 + "\n")

def main():
    """🚀 Main launcher"""
    
    print_banner()
    
    # Check environment
    if not check_env_file():
        print("❌ Environment check failed. Please check your .env file.")
        return
    
    # Start Telegram bot
    telegram_process = start_telegram_bot()
    if not telegram_process:
        print("⚠️ Continuing without Telegram Panel...")
    
    # Start trading system
    trading_system = start_trading_system()
    if not trading_system:
        print("❌ Trading System failed to start")
        return
    
    # Show system info
    show_system_info(trading_system)
    
    # Auto-start trading
    if trading_system.is_initialized:
        print("🚀 AUTO-STARTING TRADING IN 5 SECONDS...")
        time.sleep(5)
        
        trading_system.start_trading()
        print("✅ TRADINO LIVE TRADING ACTIVE!")
        
        try:
            # Keep running
            while trading_system.is_running:
                time.sleep(60)  # Update every minute
                
                # Show periodic status
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    stats = trading_system.session_stats
                    print(f"📊 Status: {stats['signals_generated']} signals, "
                          f"{stats['trades_executed']} trades, "
                          f"Uptime: {stats['uptime']}")
                
        except KeyboardInterrupt:
            print("\n🛑 Shutdown initiated...")
            
            # Stop trading
            if trading_system.is_running:
                trading_system.stop_real_trading()
            
            # Stop telegram bot
            if telegram_process:
                telegram_process.terminate()
                telegram_process.wait()
            
            print("✅ TRADINO stopped successfully")

if __name__ == "__main__":
    main() 