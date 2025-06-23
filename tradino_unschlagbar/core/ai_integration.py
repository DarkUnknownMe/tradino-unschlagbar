#!/usr/bin/env python3
"""
🔗 AI INTEGRATION LAYER
Integriert alle AI-Komponenten in das bestehende Trading-System
"""

import sys
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# AI Komponenten importieren
sys.path.append(os.path.dirname(__file__))

try:
    from brain.models.rl_agent import DQNTradingAgent
    from brain.agents.multi_agent_system import MultiAgentOrchestrator, TradingSignal
    RL_AVAILABLE = True
    MULTI_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ AI Komponenten nicht verfügbar: {e}")
    RL_AVAILABLE = False
    MULTI_AGENT_AVAILABLE = False

class AITradingSystem:
    """🤖 Vollständiges AI Trading System"""
    
    def __init__(self, exchange=None):
        self.exchange = exchange
        
        # AI Komponenten
        self.rl_agent = DQNTradingAgent() if RL_AVAILABLE else None
        self.multi_agent_system = MultiAgentOrchestrator() if MULTI_AGENT_AVAILABLE else None
        
        # Performance Tracking
        self.performance_history = []
        self.active_positions = {}
        
        print("🤖 AI Trading System initialisiert")
        if RL_AVAILABLE:
            print("✅ RL Agent verfügbar")
        if MULTI_AGENT_AVAILABLE:
            print("✅ Multi-Agent System verfügbar")
    
    async def initialize_ai_system(self) -> bool:
        """🤖 Initialisiere AI System"""
        try:
            print("🔧 Initialisiere AI Trading System...")
            
            # Lade trainierte Modelle falls vorhanden
            if self.rl_agent:
                model_path = "tradino_unschlagbar/models/rl_agent.pth"
                if os.path.exists(model_path):
                    self.rl_agent.load_model(model_path)
            
            print("✅ AI System bereit")
            return True
            
        except Exception as e:
            print(f"❌ AI System Initialisierung fehlgeschlagen: {e}")
            return False
    
    async def get_ai_trading_signal(self, symbol: str = "BTC/USDT") -> Optional[TradingSignal]:
        """🧠 Hole AI Trading Signal"""
        try:
            # Demo-Marktdaten für Tests
            demo_market_data = {
                'symbol': symbol,
                'ohlcv': [[i, 50000 + np.random.randn() * 100, 50100, 49900, 50000 + np.random.randn() * 50, 100] 
                         for i in range(60)],
                'timestamp': datetime.now()
            }
            
            # Multi-Agent Signal
            if self.multi_agent_system:
                return await self.multi_agent_system.get_collective_signal(demo_market_data)
            
            # Fallback Demo Signal
            return TradingSignal(
                agent_id="demo_ai",
                signal_type='BUY',
                confidence=0.75,
                reasoning="AI Demo Signal - BUY",
                timestamp=datetime.now(),
                symbol=symbol,
                entry_price=50000,
                position_size=1000
            )
            
        except Exception as e:
            print(f"❌ AI Signal Fehler: {e}")
            return None
    
    def get_ai_system_status(self) -> Dict[str, Any]:
        """📊 AI System Status"""
        try:
            status = {
                'ai_components': {
                    'rl_agent_available': self.rl_agent is not None,
                    'multi_agent_available': self.multi_agent_system is not None,
                    'exchange_connected': self.exchange is not None
                },
                'performance': {
                    'total_signals': len(self.performance_history)
                }
            }
            
            # Multi-Agent Status
            if self.multi_agent_system:
                status['multi_agent_status'] = self.multi_agent_system.get_system_status()
            
            return status
            
        except Exception as e:
            print(f"❌ Status Report Fehler: {e}")
            return {'error': str(e)}

# Enhanced Position Manager
class EnhancedPositionManager:
    """🚀 AI-Enhanced Position Manager"""
    
    def __init__(self, exchange=None):
        self.exchange = exchange
        self.ai_system = AITradingSystem(exchange)
        self.ai_initialized = False
    
    async def initialize_ai(self) -> bool:
        """🤖 Initialisiere AI System"""
        if not self.ai_initialized:
            print("🤖 Initialisiere AI Trading System...")
            success = await self.ai_system.initialize_ai_system()
            self.ai_initialized = success
            return success
        return True
    
    async def get_ai_enhanced_signal(self, symbol: str = 'BTC/USDT'):
        """🧠 Hole AI-verstärktes Signal"""
        if not self.ai_initialized:
            await self.initialize_ai()
        
        try:
            signal = await self.ai_system.get_ai_trading_signal(symbol)
            return signal
        except Exception as e:
            print(f"❌ AI Signal Fehler: {e}")
            return None
    
    async def run_ai_enhanced_trading(self):
        """🚀 AI-Enhanced Trading Cycle"""
        print("🤖 AI-ENHANCED TRADING GESTARTET")
        print("=" * 50)
        
        # AI initialisieren
        await self.initialize_ai()
        
        # Hole AI Signal
        ai_signal = await self.get_ai_enhanced_signal()
        
        if ai_signal and ai_signal.signal_type in ['BUY', 'SELL']:
            print(f"🤖 AI SIGNAL: {ai_signal.signal_type}")
            print(f"🧠 Reasoning: {ai_signal.reasoning}")
            print(f"📊 Confidence: {ai_signal.confidence:.2%}")
            print(f"💰 Entry Price: ${ai_signal.entry_price:.2f}")
            print(f"📏 Position Size: ${ai_signal.position_size}")
            
            print("✅ AI Signal empfangen - bereit für Ausführung!")
            return ai_signal
        else:
            print("📭 Kein AI Signal - keine Aktion")
            return None

# Demo Function
async def demo_ai_integration():
    """🎯 Demo der AI Integration"""
    print("🎯 AI INTEGRATION DEMO")
    print("=" * 40)
    
    # Erstelle Enhanced Position Manager
    manager = EnhancedPositionManager(exchange=None)
    
    # Führe AI-Enhanced Trading aus
    signal = await manager.run_ai_enhanced_trading()
    
    if signal:
        print(f"
🎉 DEMO ERFOLGREICH!")
        print(f"Signal: {signal.signal_type} mit {signal.confidence:.2%} Confidence")
    else:
        print("
📭 Kein Signal in Demo")
    
    # System Status
    status = manager.ai_system.get_ai_system_status()
    print(f"
📊 AI System Status:")
    print(f"🤖 AI Komponenten:")
    print(f"   RL Agent: {'✅' if status['ai_components']['rl_agent_available'] else '❌'}")
    print(f"   Multi-Agent: {'✅' if status['ai_components']['multi_agent_available'] else '❌'}")

    print(f"
📊 Performance:")
    print(f"   Signals: {status['performance']['total_signals']}")

if __name__ == "__main__":
    asyncio.run(demo_ai_integration())
