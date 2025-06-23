#!/usr/bin/env python3
"""
🎭 MULTI-AGENT TRADING SYSTEM
Mehrere spezialisierte Agenten arbeiten zusammen
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio

@dataclass
class TradingSignal:
    agent_id: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime
    symbol: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None

class BaseAgent:
    """🏗️ Basis-Klasse für alle Trading-Agenten"""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.performance_score = 0.5
        self.total_signals = 0
        self.successful_signals = 0
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """🔍 Marktanalyse und Signal-Generierung"""
        raise NotImplementedError("Must be implemented by subclass")
    
    def update_performance(self, success: bool):
        """📊 Performance-Update"""
        self.total_signals += 1
        if success:
            self.successful_signals += 1
        
        if self.total_signals > 0:
            success_rate = self.successful_signals / self.total_signals
            self.performance_score = (self.performance_score * 0.8) + (success_rate * 0.2)

class TrendFollowingAgent(BaseAgent):
    """📈 Trend-Following Agent"""
    
    def __init__(self):
        super().__init__("trend_follower", "TREND_FOLLOWING")
        self.trend_threshold = 0.02
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """📈 Trend-Analyse"""
        try:
            ohlcv = market_data.get('ohlcv', [])
            if len(ohlcv) < 50:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Moving Averages
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            current_price = df['close'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            
            # Trend-Bestimmung
            if len(df) > 24:
                price_change_24h = (current_price - df['close'].iloc[-24]) / df['close'].iloc[-24]
            else:
                price_change_24h = 0
            
            # Bullish Trend
            if (current_price > sma_20 > sma_50 and price_change_24h > self.trend_threshold):
                return TradingSignal(
                    agent_id=self.agent_id,
                    signal_type='BUY',
                    confidence=min(0.9, abs(price_change_24h) * 10),
                    reasoning=f"Bullish Trend: Price {current_price:.2f} > SMA20 {sma_20:.2f}",
                    timestamp=datetime.now(),
                    symbol=market_data.get('symbol', 'BTC/USDT'),
                    entry_price=current_price,
                    position_size=1000
                )
            
            return None
            
        except Exception as e:
            print(f"❌ Trend Agent Error: {e}")
            return None

class MultiAgentOrchestrator:
    """🎭 Multi-Agent System Orchestrator"""
    
    def __init__(self):
        self.agents = [TrendFollowingAgent()]
        self.agent_weights = {agent.agent_id: 1.0 for agent in self.agents}
        self.signal_history = []
        print(f"🎭 Multi-Agent System initialisiert - {len(self.agents)} Agenten")
    
    async def get_collective_signal(self, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """🧠 Kollektive Signal-Generierung"""
        try:
            # Sammle Signale von allen Agenten
            signals = []
            for agent in self.agents:
                signal = await agent.analyze_market(market_data)
                if signal:
                    signals.append(signal)
            
            if not signals:
                return None
            
            # Für Demo: Erstes verfügbares Signal
            best_signal = signals[0]
            self.signal_history.append(best_signal)
            return best_signal
            
        except Exception as e:
            print(f"❌ Multi-Agent Orchestrator Error: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """📊 System Status"""
        return {
            'total_agents': len(self.agents),
            'total_signals': len(self.signal_history),
            'agent_performance': {
                agent.agent_id: {
                    'type': agent.agent_type,
                    'performance_score': agent.performance_score,
                    'total_signals': agent.total_signals
                } for agent in self.agents
            }
        }

# Demo Function
async def demo_multi_agent_system():
    """🎯 Demo des Multi-Agent Systems"""
    print("🎭 Multi-Agent System Demo")
    
    # Erstelle Test-Marktdaten
    test_market_data = {
        'symbol': 'BTC/USDT',
        'ohlcv': [[i, 50000 + i*10, 50100 + i*10, 49900 + i*10, 50000 + i*15, 100] for i in range(60)]
    }
    
    # Erstelle Multi-Agent System
    orchestrator = MultiAgentOrchestrator()
    
    # Hole kollektives Signal
    signal = await orchestrator.get_collective_signal(test_market_data)
    
    if signal:
        print(f"🎯 Signal: {signal.signal_type}")
        print(f"🧠 Reasoning: {signal.reasoning}")
        print(f"📊 Confidence: {signal.confidence:.2%}")
    else:
        print("📭 Kein Signal generiert")
    
    # System Status
    status = orchestrator.get_system_status()
    print(f"📊 System Status: {status}")

if __name__ == "__main__":
    asyncio.run(demo_multi_agent_system())
