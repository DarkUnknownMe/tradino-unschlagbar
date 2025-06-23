#!/usr/bin/env python3
"""
🚀 TRADINO UNSCHLAGBAR - Main Launcher
Sauberer Haupteinstieg für das komplette Trading System
"""

import os
import sys
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / 'core'))
sys.path.append(str(current_dir / 'tradino_unschlagbar'))

def main():
    """🚀 Hauptstartfunktion"""
    
    print("🚀 TRADINO UNSCHLAGBAR")
    print("=" * 50)
    print("🎯 Wählen Sie eine Option:")
    print("1. 🚀 Complete System (Trading + Telegram)")
    print("2. 💰 Trading Only")
    print("3. 📱 Telegram Panel Only")
    print("4. 🔧 Debug Mode")
    print("5. ❌ Exit")
    print("=" * 50)
    
    choice = input("Ihre Wahl (1-5): ").strip()
    
    if choice == "1":
        print("🚀 Starting Complete System...")
        from core.start_tradino_complete import main as start_complete
        start_complete()
        
    elif choice == "2":
        print("💰 Starting Trading System...")
        from core.final_live_trading_system import start_real_tradino
        system = start_real_tradino()
        if system:
            system.start_real_trading()
            input("Press Enter to stop...")
            system.stop_real_trading()
            
    elif choice == "3":
        print("📱 Starting Telegram Panel...")
        from core.tradino_telegram_panel import main as start_telegram
        start_telegram()
        
    elif choice == "4":
        print("🔧 Starting Debug Mode...")
        from scripts.debug_futures_balance import debug_balance
        debug_balance()
        
    elif choice == "5":
        print("👋 Auf Wiedersehen!")
        return
        
    else:
        print("❌ Ungültige Auswahl!")
        main()

if __name__ == "__main__":
    main() 