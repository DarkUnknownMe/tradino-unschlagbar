#!/usr/bin/env python3
"""
🚀 TRADINO - Advanced AI Trading System
Main Entry Point

This is the central entry point for the TRADINO trading system.
All system components are initialized and started from here.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Core imports
try:
    from core.integration_manager import IntegrationManager
    from core.start_tradino_complete import main as start_tradino
    from tradino_unschlagbar.utils.logger_pro import LoggerPro
    from tradino_unschlagbar.utils.config_manager import ConfigManager
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Möglicherweise müssen Dependencies installiert werden:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

def print_banner():
    """Print TRADINO startup banner"""
    banner = """
    ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗ 
    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔═══██╗
       ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║   ██║
       ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
       ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝ 
    
    🤖 Advanced AI Trading System v2.0
    ⚡ Powered by Machine Learning & Risk Management
    """
    print(banner)

def check_environment():
    """Check if environment is properly configured"""
    logger = LoggerPro().get_logger("main")
    
    # Check for required environment variables
    required_vars = [
        "BITGET_API_KEY",
        "BITGET_SECRET_KEY", 
        "BITGET_PASSPHRASE",
        "TELEGRAM_BOT_TOKEN"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"⚠️ Missing environment variables: {missing_vars}")
        logger.info("💡 Create .env file with your API credentials")
        return False
    
    return True

def select_startup_mode():
    """Let user select startup mode"""
    print("\n🚀 TRADINO STARTUP OPTIONEN:")
    print("1. 🔧 System Validation (Empfohlen für ersten Start)")
    print("2. 📊 Integration Manager (Vollständiges System)")
    print("3. 🤖 AI Trading System (Tradino Unschlagbar)")
    print("4. 📈 Paper Trading Mode")
    print("5. 🔴 Live Trading Mode")
    print("6. ❌ Exit")
    
    while True:
        try:
            choice = input("\n👉 Wähle eine Option (1-6): ").strip()
            
            if choice == "1":
                return "validation"
            elif choice == "2":
                return "integration"
            elif choice == "3":
                return "ai_trading"
            elif choice == "4":
                return "paper"
            elif choice == "5":
                return "live"
            elif choice == "6":
                return "exit"
            else:
                print("❌ Ungültige Auswahl. Bitte wähle 1-6.")
        except KeyboardInterrupt:
            print("\n👋 Auf Wiedersehen!")
            return "exit"

async def main():
    """Main entry point"""
    print_banner()
    
    # Basic environment check
    if not check_environment():
        print("\n📋 SETUP ANLEITUNG:")
        print("1. Erstelle .env Datei: cp .env.example .env")
        print("2. Trage deine API Credentials ein")
        print("3. Starte das System erneut")
        return
    
    # Select startup mode
    mode = select_startup_mode()
    
    if mode == "exit":
        print("👋 Auf Wiedersehen!")
        return
    
    try:
        if mode == "validation":
            print("\n🔍 Starte System Validation...")
            os.system("python scripts/system_validation.py")
            
        elif mode == "integration":
            print("\n🔧 Starte Integration Manager...")
            manager = IntegrationManager()
            await manager.start_system()
            
        elif mode == "ai_trading":
            print("\n🤖 Starte AI Trading System...")
            await start_tradino()
            
        elif mode == "paper":
            print("\n📊 Starte Paper Trading Mode...")
            os.environ["TRADING_MODE"] = "paper"
            await start_tradino()
            
        elif mode == "live":
            print("\n🔴 Starte Live Trading Mode...")
            
            # Additional confirmation for live trading
            confirm = input("⚠️ WARNUNG: Live Trading kann zu echten Verlusten führen! Fortfahren? (yes/no): ")
            if confirm.lower() not in ["yes", "y"]:
                print("❌ Live Trading abgebrochen.")
                return
                
            os.environ["TRADING_MODE"] = "live"
            await start_tradino()
            
    except KeyboardInterrupt:
        print("\n🛑 System shutdown durch Benutzer...")
    except Exception as e:
        print(f"\n❌ Fehler beim Systemstart: {e}")
        print("💡 Versuche System Validation: python scripts/system_validation.py")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        sys.exit(1) 