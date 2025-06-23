#!/usr/bin/env python3
"""
🚀 TRADINO UNSCHLAGBAR - AI TRADING STARTUP
Hauptstartscript für das AI-erweiterte Trading System
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

try:
    from tradino_unschlagbar.core.ai_integration import EnhancedPositionManager, AITradingSystem
    AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ AI Komponenten nicht verfügbar: {e}")
    AI_AVAILABLE = False

async def start_ai_trading_system():
    """🚀 Starte AI Trading System"""
    print("🚀 TRADINO UNSCHLAGBAR - AI TRADING SYSTEM")
    print("=" * 60)
    print(f"📅 Start Time: {datetime.now()}")
    print("🤖 Starte AI-Enhanced Trading...")
    
    if not AI_AVAILABLE:
        print("❌ AI Komponenten nicht verfügbar")
        print("💡 Führen Sie zuerst aus: python3 create_ai_components.py")
        return
    
    try:
        manager = EnhancedPositionManager(exchange=None)
        
        print("
🎯 Trading-Modus wählen:")
        print("1. 🤖 AI-Enhanced Single Trade")
        print("2. 🎯 AI Demo (ohne Orders)")
        print("3. 📊 AI System Status")
        
        try:
            choice = input("Wählen Sie (1-3): ").strip()
        except KeyboardInterrupt:
            print("
🛑 Abgebrochen")
            return
        
        if choice == "1":
            print("🤖 Starte AI-Enhanced Single Trade...")
            signal = await manager.run_ai_enhanced_trading()
            
            if signal:
                print(f"✅ AI Trade Signal generiert: {signal.signal_type}")
                print("📋 Signal nur angezeigt (Demo)")
        
        elif choice == "2":
            print("🎯 AI Demo Modus...")
            await manager.run_ai_enhanced_trading()
        
        elif choice == "3":
            print("📊 AI System Status...")
            status = manager.ai_system.get_ai_system_status()
            
            print(f"
🤖 AI Komponenten:")
            print(f"   RL Agent: {'✅' if status['ai_components']['rl_agent_available'] else '❌'}")
            print(f"   Multi-Agent: {'✅' if status['ai_components']['multi_agent_available'] else '❌'}")
            
            print(f"
📊 Performance:")
            print(f"   Signals: {status['performance']['total_signals']}")
        
        print("
🎉 Session beendet!")
        
    except KeyboardInterrupt:
        print("
🛑 Trading gestoppt durch Benutzer")
    except Exception as e:
        print(f"
❌ Fehler: {e}")

if __name__ == "__main__":
    asyncio.run(start_ai_trading_system())
