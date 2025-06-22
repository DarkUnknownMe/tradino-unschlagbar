import asyncio
import unittest
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.multi_agent_system import (
    TRADINOMultiAgentSystem, AgentSpecialization, 
    TrendSpecialistAgent, VolatilityExpertAgent, 
    SentimentMasterAgent, MomentumTrackerAgent, RiskCommanderAgent
)
# Mock Config Manager
class ConfigManager:
    def __init__(self):
        self.multi_agent_enabled = []
    
    def get(self, key, default=None):
        if key == 'multi_agent_enabled':
            return self.multi_agent_enabled
        return default
from loguru import logger

class TestMultiAgentSystem(unittest.TestCase):
    """
    Test Suite für Multi-Agent System
    """
    
    def setUp(self):
        """
        Test Setup
        """
        self.config = ConfigManager()
        self.config.multi_agent_enabled = [
            AgentSpecialization.TREND_SPECIALIST,
            AgentSpecialization.VOLATILITY_EXPERT,
            AgentSpecialization.SENTIMENT_MASTER,
            AgentSpecialization.MOMENTUM_TRACKER,
            AgentSpecialization.RISK_COMMANDER
        ]
        
        # Mock Trading Engine
        self.mock_trading_engine = MockTradingEngine()
        
        # Multi-Agent System
        self.multi_agent_system = TRADINOMultiAgentSystem(
            config=self.config,
            trading_engine=self.mock_trading_engine
        )
    
    async def test_multi_agent_initialization(self):
        """
        Test Multi-Agent System Initialization
        """
        logger.info("🧪 Testing Multi-Agent System Initialization...")
        
        # Initialize System
        success = await self.multi_agent_system.initialize()
        
        # Validations
        self.assertTrue(success)
        self.assertEqual(len(self.multi_agent_system.agents), 5)  # 5 enabled agents
        self.assertIsNotNone(self.multi_agent_system.coordinator)
        self.assertIsNotNone(self.multi_agent_system.consensus_engine)
        
        logger.success("✅ Multi-Agent System Initialization Test bestanden")
    
    async def test_individual_agent_initialization(self):
        """
        Test Individual Agent Initialization
        """
        logger.info("🧪 Testing Individual Agent Initialization...")
        
        await self.multi_agent_system.initialize()
        
        # Check Agent Types
        agent_types = []
        for agent in self.multi_agent_system.agents.values():
            agent_types.append(type(agent).__name__)
        
        expected_types = [
            'TrendSpecialistAgent',
            'VolatilityExpertAgent', 
            'SentimentMasterAgent',
            'MomentumTrackerAgent',
            'RiskCommanderAgent'
        ]
        
        for expected_type in expected_types:
            self.assertTrue(any(expected_type in agent_type for agent_type in agent_types))
        
        logger.success("✅ Individual Agent Initialization Test bestanden")
    
    async def test_ensemble_signal_generation(self):
        """
        Test Ensemble Signal Generation
        """
        logger.info("🧪 Testing Ensemble Signal Generation...")
        
        await self.multi_agent_system.initialize()
        
        # Mock Market Data
        mock_market_data = {
            'symbol': 'BTC/USDT',
            'close': 50000,
            'high': 51000,
            'low': 49000,
            'volume': 1000000,
            'volatility': 0.025,
            'timestamp': datetime.now()
        }
        
        # Generate Ensemble Signal
        ensemble_signal = await self.multi_agent_system.generate_ensemble_signal(mock_market_data)
        
        # Validations
        self.assertIsNotNone(ensemble_signal)
        self.assertIn('action', ensemble_signal)
        self.assertIn('confidence', ensemble_signal)
        self.assertIn('agent_count', ensemble_signal)
        self.assertIn('source', ensemble_signal)
        
        # Action should be valid
        self.assertIn(ensemble_signal['action'], ['BUY', 'SELL', 'HOLD'])
        
        # Confidence should be between 0 and 1
        self.assertGreaterEqual(ensemble_signal['confidence'], 0.0)
        self.assertLessEqual(ensemble_signal['confidence'], 1.0)
        
        # Should have agent breakdown
        self.assertIn('agent_breakdown', ensemble_signal)
        
        logger.success("✅ Ensemble Signal Generation Test bestanden")
        logger.info(f"Generated Signal: {ensemble_signal['action']} (Confidence: {ensemble_signal['confidence']:.3f})")
    
    async def test_individual_agent_signals(self):
        """
        Test Individual Agent Signal Generation
        """
        logger.info("🧪 Testing Individual Agent Signals...")
        
        await self.multi_agent_system.initialize()
        
        mock_market_data = {
            'symbol': 'BTC/USDT',
            'close': 50000,
            'high': 51000,
            'low': 49000,
            'volume': 1000000,
            'volatility': 0.025,
            'timestamp': datetime.now()
        }
        
        signals_generated = 0
        
        for agent_id, agent in self.multi_agent_system.agents.items():
            try:
                signal = await agent.generate_signal(mock_market_data)
                
                if signal:
                    signals_generated += 1
                    
                    # Validate Signal Structure
                    self.assertEqual(signal.agent_id, agent_id)
                    self.assertIn(signal.action, ['BUY', 'SELL', 'HOLD'])
                    self.assertGreaterEqual(signal.confidence, 0.0)
                    self.assertLessEqual(signal.confidence, 1.0)
                    self.assertIsInstance(signal.reasoning, str)
                    
                    logger.info(f"Agent {agent_id}: {signal.action} (Conf: {signal.confidence:.3f})")
                
            except Exception as e:
                logger.warning(f"Agent {agent_id} Signal Generation Fehler: {e}")
        
        # At least 3 agents should generate signals
        self.assertGreaterEqual(signals_generated, 3)
        
        logger.success(f"✅ Individual Agent Signals Test bestanden ({signals_generated}/5 Signale)")
    
    async def test_consensus_building(self):
        """
        Test Consensus Building
        """
        logger.info("🧪 Testing Consensus Building...")
        
        await self.multi_agent_system.initialize()
        
        # Mock Agent Signals
        from brain.multi_agent_system import AgentSignal
        
        mock_signals = [
            AgentSignal(
                agent_id="agent_1",
                specialization=AgentSpecialization.TREND_SPECIALIST,
                action="BUY",
                confidence=0.8,
                position_size=0.1,
                reasoning="Strong bullish trend",
                technical_factors={},
                risk_assessment={},
                timestamp=datetime.now(),
                priority="high",
                expected_duration="medium",
                market_conditions={}
            ),
            AgentSignal(
                agent_id="agent_2",
                specialization=AgentSpecialization.MOMENTUM_TRACKER,
                action="BUY",
                confidence=0.7,
                position_size=0.12,
                reasoning="Positive momentum",
                technical_factors={},
                risk_assessment={},
                timestamp=datetime.now(),
                priority="medium",
                expected_duration="short",
                market_conditions={}
            ),
            AgentSignal(
                agent_id="agent_3",
                specialization=AgentSpecialization.SENTIMENT_MASTER,
                action="HOLD",
                confidence=0.5,
                position_size=0.0,
                reasoning="Mixed sentiment",
                technical_factors={},
                risk_assessment={},
                timestamp=datetime.now(),
                priority="low",
                expected_duration="medium",
                market_conditions={}
            )
        ]
        
        # Test Consensus Building
        consensus_result = await self.multi_agent_system.consensus_engine.build_consensus(
            mock_signals, {}
        )
        
        # Validations
        self.assertIsNotNone(consensus_result)
        self.assertIn('action', consensus_result)
        self.assertIn('confidence', consensus_result)
        self.assertIn('consensus_strength', consensus_result)
        
        # Should favor BUY (2 out of 3)
        self.assertEqual(consensus_result['action'], 'BUY')
        
        logger.success("✅ Consensus Building Test bestanden")
        logger.info(f"Consensus: {consensus_result['action']} (Strength: {consensus_result['consensus_strength']:.3f})")
    
    async def test_agent_performance_tracking(self):
        """
        Test Agent Performance Tracking
        """
        logger.info("🧪 Testing Agent Performance Tracking...")
        
        await self.multi_agent_system.initialize()
        
        # Generate Signal to trigger performance tracking
        mock_market_data = {
            'symbol': 'BTC/USDT',
            'close': 50000,
            'volume': 1000000,
            'volatility': 0.02
        }
        
        await self.multi_agent_system.generate_ensemble_signal(mock_market_data)
        
        # Check Performance Tracking
        for agent_id, performance in self.multi_agent_system.agent_performance.items():
            self.assertIn('signals_generated', performance)
            self.assertIn('successful_signals', performance)
            self.assertIn('accuracy', performance)
            
        logger.success("✅ Agent Performance Tracking Test bestanden")
    
    def test_system_status(self):
        """
        Test System Status
        """
        logger.info("🧪 Testing System Status...")
        
        status = self.multi_agent_system.get_system_status()
        
        # Validations
        self.assertIn('active_agents', status)
        self.assertIn('enabled_specializations', status)
        self.assertIn('agent_performance', status)
        self.assertIn('system_health', status)
        
        logger.success("✅ System Status Test bestanden")


class TestSpecializedAgents(unittest.TestCase):
    """
    Test Suite für Specialized Agents
    """
    
    def setUp(self):
        self.config = ConfigManager()
        self.mock_trading_engine = MockTradingEngine()
        self.mock_message_bus = MockMessageBus()
        self.mock_shared_memory = MockSharedMemory()
    
    async def test_trend_specialist_agent(self):
        """
        Test Trend Specialist Agent
        """
        logger.info("🧪 Testing Trend Specialist Agent...")
        
        agent = TrendSpecialistAgent(
            self.config, self.mock_trading_engine,
            self.mock_message_bus, self.mock_shared_memory
        )
        
        await agent.initialize()
        
        # Test Signal Generation
        mock_market_data = {
            'symbol': 'BTC/USDT',
            'close': 52000,
            'high': 52500,
            'low': 51500,
            'volume': 1200000,
            'volatility': 0.03
        }
        
        signal = await agent.generate_signal(mock_market_data)
        
        if signal:
            self.assertEqual(signal.specialization, AgentSpecialization.TREND_SPECIALIST)
            self.assertIn(signal.action, ['BUY', 'SELL', 'HOLD'])
            self.assertIn('trend_direction', signal.technical_factors)
            self.assertIn('trend_strength', signal.technical_factors)
        
        logger.success("✅ Trend Specialist Agent Test bestanden")
    
    async def test_volatility_expert_agent(self):
        """
        Test Volatility Expert Agent
        """
        logger.info("🧪 Testing Volatility Expert Agent...")
        
        agent = VolatilityExpertAgent(
            self.config, self.mock_trading_engine,
            self.mock_message_bus, self.mock_shared_memory
        )
        
        await agent.initialize()
        
        # Test with High Volatility
        mock_market_data = {
            'symbol': 'BTC/USDT',
            'close': 50000,
            'high': 53000,
            'low': 47000,
            'volume': 2000000,
            'volatility': 0.06  # High volatility
        }
        
        signal = await agent.generate_signal(mock_market_data)
        
        if signal:
            self.assertEqual(signal.specialization, AgentSpecialization.VOLATILITY_EXPERT)
            self.assertIn('volatility_regime', signal.technical_factors)
            self.assertIn('breakout_strength', signal.technical_factors)
        
        logger.success("✅ Volatility Expert Agent Test bestanden")
    
    async def test_sentiment_master_agent(self):
        """
        Test Sentiment Master Agent
        """
        logger.info("🧪 Testing Sentiment Master Agent...")
        
        agent = SentimentMasterAgent(
            self.config, self.mock_trading_engine,
            self.mock_message_bus, self.mock_shared_memory
        )
        
        await agent.initialize()
        
        mock_market_data = {
            'symbol': 'BTC/USDT',
            'close': 50000,
            'volume': 1000000,
            'volatility': 0.02
        }
        
        signal = await agent.generate_signal(mock_market_data)
        
        if signal:
            self.assertEqual(signal.specialization, AgentSpecialization.SENTIMENT_MASTER)
            self.assertIn('overall_sentiment', signal.technical_factors)
            self.assertIn('sentiment_strength', signal.technical_factors)
        
        logger.success("✅ Sentiment Master Agent Test bestanden")
    
    async def test_momentum_tracker_agent(self):
        """
        Test Momentum Tracker Agent
        """
        logger.info("🧪 Testing Momentum Tracker Agent...")
        
        agent = MomentumTrackerAgent(
            self.config, self.mock_trading_engine,
            self.mock_message_bus, self.mock_shared_memory
        )
        
        await agent.initialize()
        
        mock_market_data = {
            'symbol': 'BTC/USDT',
            'close': 51500,  # 3% up from 50000
            'high': 52000,
            'low': 50500,
            'volume': 1500000,
            'volatility': 0.025
        }
        
        signal = await agent.generate_signal(mock_market_data)
        
        if signal:
            self.assertEqual(signal.specialization, AgentSpecialization.MOMENTUM_TRACKER)
            self.assertIn('momentum_strength', signal.technical_factors)
        
        logger.success("✅ Momentum Tracker Agent Test bestanden")
    
    async def test_risk_commander_agent(self):
        """
        Test Risk Commander Agent
        """
        logger.info("🧪 Testing Risk Commander Agent...")
        
        agent = RiskCommanderAgent(
            self.config, self.mock_trading_engine,
            self.mock_message_bus, self.mock_shared_memory
        )
        
        await agent.initialize()
        
        mock_market_data = {
            'symbol': 'BTC/USDT',
            'close': 50000,
            'volume': 1000000,
            'volatility': 0.08  # High volatility = higher risk
        }
        
        signal = await agent.generate_signal(mock_market_data)
        
        if signal:
            self.assertEqual(signal.specialization, AgentSpecialization.RISK_COMMANDER)
            self.assertIn('portfolio_risk_score', signal.technical_factors)
            self.assertIn('overall_risk_level', signal.technical_factors)
        
        logger.success("✅ Risk Commander Agent Test bestanden")


# Mock Classes
class MockTradingEngine:
    def __init__(self):
        self.market_intelligence = MockMarketIntelligence()
        self.sentiment_analyzer = MockSentimentAnalyzer()
        self.pattern_recognition = MockPatternRecognition()
        self.portfolio_manager = MockPortfolioManager()

class MockMarketIntelligence:
    async def get_technical_indicators(self, symbol):
        return {'rsi': 45, 'macd': 0.1, 'bb_position': 0.3}

class MockSentimentAnalyzer:
    async def analyze_sentiment(self, symbol):
        return {'news_sentiment': 0.1, 'social_sentiment': -0.05}

class MockPatternRecognition:
    async def detect_patterns(self, symbol):
        return {'trend_strength': 0.2, 'breakout_probability': 0.3}

class MockPortfolioManager:
    async def get_portfolio_summary(self):
        return {'current_position': 0.0, 'unrealized_pnl': 0.0}

class MockMessageBus:
    async def start(self):
        return True
    async def publish(self, topic, message):
        pass
    async def subscribe(self, agent_id, topics):
        pass

class MockSharedMemory:
    async def initialize(self):
        return True
    async def update_market_data(self, market_data):
        pass
    async def get_market_data(self, symbol):
        return {}


async def run_multi_agent_tests():
    """
    Führe alle Multi-Agent Tests aus
    """
    logger.info("🧪 Multi-Agent System Test Suite startet...")
    
    tests_passed = 0
    tests_total = 0
    
    # Multi-Agent System Tests
    ma_test_suite = TestMultiAgentSystem()
    ma_test_suite.setUp()
    
    test_methods = [
        ('Multi-Agent Initialization', ma_test_suite.test_multi_agent_initialization),
        ('Individual Agent Initialization', ma_test_suite.test_individual_agent_initialization),
        ('Ensemble Signal Generation', ma_test_suite.test_ensemble_signal_generation),
        ('Individual Agent Signals', ma_test_suite.test_individual_agent_signals),
        ('Consensus Building', ma_test_suite.test_consensus_building),
        ('Agent Performance Tracking', ma_test_suite.test_agent_performance_tracking)
    ]
    
    for test_name, test_method in test_methods:
        tests_total += 1
        try:
            if asyncio.iscoroutinefunction(test_method):
                await test_method()
            else:
                test_method()
            tests_passed += 1
            logger.success(f"✅ {test_name} Test bestanden")
        except Exception as e:
            logger.error(f"❌ {test_name} Test fehlgeschlagen: {e}")
    
    # Specialized Agent Tests
    agent_test_suite = TestSpecializedAgents()
    agent_test_suite.setUp()
    
    agent_test_methods = [
        ('Trend Specialist Agent', agent_test_suite.test_trend_specialist_agent),
        ('Volatility Expert Agent', agent_test_suite.test_volatility_expert_agent),
        ('Sentiment Master Agent', agent_test_suite.test_sentiment_master_agent),
        ('Momentum Tracker Agent', agent_test_suite.test_momentum_tracker_agent),
        ('Risk Commander Agent', agent_test_suite.test_risk_commander_agent)
    ]
    
    for test_name, test_method in agent_test_methods:
        tests_total += 1
        try:
            await test_method()
            tests_passed += 1
            logger.success(f"✅ {test_name} Test bestanden")
        except Exception as e:
            logger.error(f"❌ {test_name} Test fehlgeschlagen: {e}")
    
    # Test Results
    success_rate = (tests_passed / tests_total) * 100
    
    logger.info("=" * 60)
    logger.info("🧪 MULTI-AGENT SYSTEM TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"📊 Tests bestanden: {tests_passed}/{tests_total} ({success_rate:.1f}%)")
    
    if tests_passed == tests_total:
        logger.success("🎉 ALLE MULTI-AGENT TESTS ERFOLGREICH BESTANDEN!")
        return True
    else:
        logger.warning(f"⚠️ {tests_total - tests_passed} Tests fehlgeschlagen")
        return False


if __name__ == '__main__':
    # Führe Tests aus
    asyncio.run(run_multi_agent_tests()) 