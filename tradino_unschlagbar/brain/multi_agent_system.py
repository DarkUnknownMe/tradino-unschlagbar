import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from enum import Enum
import json

# Market Regime Detection Integration
from brain.market_regime_detector import AdvancedMarketRegimeDetector, RegimeBasedStrategySelector

# Neural Architecture Search Integration - NEU!
from brain.neural_architecture_search import TRADINONeuralArchitectureSearch

class AgentSpecialization(Enum):
    """
    Spezialisierungen der verschiedenen AI-Agenten
    """
    TREND_SPECIALIST = "trend_specialist"
    VOLATILITY_EXPERT = "volatility_expert"
    SENTIMENT_MASTER = "sentiment_master"
    ARBITRAGE_HUNTER = "arbitrage_hunter"
    RISK_COMMANDER = "risk_commander"
    MOMENTUM_TRACKER = "momentum_tracker"
    PATTERN_RECOGNIZER = "pattern_recognizer"

@dataclass
class AgentSignal:
    """
    Standardisierte Agent Signal Struktur
    """
    agent_id: str
    specialization: AgentSpecialization
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 - 1.0
    position_size: float  # 0.0 - 1.0
    reasoning: str
    technical_factors: Dict[str, float]
    risk_assessment: Dict[str, float]
    timestamp: datetime
    priority: str  # low, medium, high, critical
    expected_duration: str  # scalp, short, medium, long
    market_conditions: Dict[str, Any]

class TRADINOMultiAgentSystem:
    """
    Hochentwickeltes Multi-Agent System für TRADINO
    Koordiniert spezialisierte AI-Agenten für überlegene Trading-Performance
    """
    
    def __init__(self, config, trading_engine):
        self.config = config
        self.trading_engine = trading_engine
        
        # Agent Registry
        self.agents: Dict[str, SpecializedTradingAgent] = {}
        self.agent_performance: Dict[str, Dict] = {}
        
        # Coordination System
        self.coordinator = AgentCoordinator(config)
        self.consensus_engine = ConsensusEngine(config)
        self.conflict_resolver = ConflictResolver(config)
        
        # Communication System
        self.message_bus = AgentMessageBus()
        self.shared_memory = SharedMarketMemory()
        
        # Performance Tracking
        self.ensemble_performance = EnsemblePerformanceTracker()
        
        # Market Regime Detection - NEU!
        self.regime_detector = AdvancedMarketRegimeDetector(config)
        self.regime_based_selector = None
        
        # Neural Architecture Search - NEU!
        self.nas_enabled = config.get('nas_enabled', True)
        self.nas_engine = None
        self.optimized_models = {}
        
        # Enhanced Configuration
        self.regime_aware_mode = config.get('regime_aware_mode', True)
        
        # Configuration
        enabled_agent_configs = config.get('multi_agent_enabled', [
            AgentSpecialization.TREND_SPECIALIST,
            AgentSpecialization.VOLATILITY_EXPERT,
            AgentSpecialization.SENTIMENT_MASTER,
            AgentSpecialization.MOMENTUM_TRACKER,
            AgentSpecialization.RISK_COMMANDER
        ])
        
        # Convert string configs to enum if needed
        self.enabled_agents = []
        for agent in enabled_agent_configs:
            if isinstance(agent, str):
                # Convert string to enum
                agent_mapping = {
                    'trend_specialist': AgentSpecialization.TREND_SPECIALIST,
                    'volatility_expert': AgentSpecialization.VOLATILITY_EXPERT,
                    'sentiment_master': AgentSpecialization.SENTIMENT_MASTER,
                    'momentum_tracker': AgentSpecialization.MOMENTUM_TRACKER,
                    'risk_commander': AgentSpecialization.RISK_COMMANDER,
                    'pattern_recognizer': AgentSpecialization.PATTERN_RECOGNIZER,
                    'arbitrage_hunter': AgentSpecialization.ARBITRAGE_HUNTER
                }
                self.enabled_agents.append(agent_mapping.get(agent, AgentSpecialization.TREND_SPECIALIST))
            else:
                self.enabled_agents.append(agent)
        
        logger.info("🤖 TRADINO Multi-Agent System mit Regime Detection initialisiert")
    
    async def initialize(self):
        """
        Multi-Agent System mit Regime Detection initialisieren
        """
        try:
            logger.info("🚀 Multi-Agent System wird initialisiert...")
            
            # Spezialisierte Agenten erstellen
            await self._create_specialized_agents()
            
            # Koordinationssystem initialisieren
            await self.coordinator.initialize()
            
            # Shared Memory initialisieren
            await self.shared_memory.initialize()
            
            # Message Bus starten
            await self.message_bus.start()
            
            # Performance Tracking initialisieren
            await self.ensemble_performance.initialize()
            
            # Market Regime Detection initialisieren - NEU!
            if self.regime_aware_mode:
                logger.info("🧠 Market Regime Detection wird initialisiert...")
                await self.regime_detector.initialize()
                self.regime_based_selector = RegimeBasedStrategySelector(self.regime_detector)
                logger.success("✅ Regime-aware Mode aktiviert")
            
            # Neural Architecture Search initialisieren - NEU!
            if self.nas_enabled:
                logger.info("🧠 Neural Architecture Search wird initialisiert...")
                self.nas_engine = TRADINONeuralArchitectureSearch(self.config)
                await self._initialize_nas_optimized_models()
                logger.success("✅ NAS-optimized Models aktiviert")
            
            logger.success(f"✅ Multi-Agent System bereit mit {len(self.agents)} Agenten + Regime Detection + NAS")
            return True
            
        except Exception as e:
            logger.error(f"❌ Multi-Agent System Initialization Fehler: {e}")
            return False
    
    async def _create_specialized_agents(self):
        """
        Spezialisierte Trading Agenten erstellen
        """
        try:
            agent_factories = {
                AgentSpecialization.TREND_SPECIALIST: TrendSpecialistAgent,
                AgentSpecialization.VOLATILITY_EXPERT: VolatilityExpertAgent,
                AgentSpecialization.SENTIMENT_MASTER: SentimentMasterAgent,
                AgentSpecialization.ARBITRAGE_HUNTER: ArbitrageHunterAgent,
                AgentSpecialization.RISK_COMMANDER: RiskCommanderAgent,
                AgentSpecialization.MOMENTUM_TRACKER: MomentumTrackerAgent,
                AgentSpecialization.PATTERN_RECOGNIZER: PatternRecognizerAgent
            }
            
            for specialization in self.enabled_agents:
                if specialization in agent_factories:
                    agent_class = agent_factories[specialization]
                    
                    agent = agent_class(
                        config=self.config,
                        trading_engine=self.trading_engine,
                        message_bus=self.message_bus,
                        shared_memory=self.shared_memory
                    )
                    
                    await agent.initialize()
                    
                    agent_id = f"{specialization.value}_{id(agent)}"
                    self.agents[agent_id] = agent
                    
                    # Performance Tracking initialisieren
                    self.agent_performance[agent_id] = {
                        'signals_generated': 0,
                        'successful_signals': 0,
                        'accuracy': 0.0,
                        'avg_confidence': 0.0,
                        'specialization_score': 0.0,
                        'last_signal': None,
                        'performance_history': deque(maxlen=1000)
                    }
                    
                    logger.success(f"✅ {specialization.value} Agent erstellt: {agent_id}")
            
            logger.info(f"🤖 {len(self.agents)} spezialisierte Agenten bereit")
            
        except Exception as e:
            logger.error(f"❌ Spezialisierte Agenten Creation Fehler: {e}")
    
    async def _initialize_nas_optimized_models(self):
        """
        NAS-optimierte Models für verschiedene Trading Tasks initialisieren
        """
        try:
            # Define trading tasks for optimization
            trading_tasks = [
                'signal_confidence_prediction',
                'market_volatility_forecast',
                'trend_strength_assessment',
                'risk_score_calculation'
            ]
            
            logger.info(f"🔍 Optimiere Neural Networks für {len(trading_tasks)} Trading Tasks...")
            
            for task in trading_tasks:
                try:
                    # Generate task-specific training data
                    features, targets = await self.nas_engine.generate_training_data(task, size=500)
                    
                    # Search optimal architecture
                    optimal_architecture = await self.nas_engine.search_optimal_architecture(
                        task, features, targets
                    )
                    
                    # Store optimized model
                    self.optimized_models[task] = optimal_architecture
                    
                    logger.success(f"✅ NAS optimiert für {task}: Score {optimal_architecture.performance_score:.4f}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ NAS Optimierung für {task} fehlgeschlagen: {e}")
            
            logger.info(f"🧠 NAS Optimierung abgeschlossen: {len(self.optimized_models)} Models optimiert")
            
        except Exception as e:
            logger.error(f"❌ NAS Model Initialization Fehler: {e}")
    
    async def generate_ensemble_signal(self, market_data: Dict) -> Dict:
        """
        NAS-enhanced Ensemble Signal Generation
        """
        try:
            symbol = market_data.get('symbol', 'BTC/USDT')
            logger.info(f"🤖 NAS-enhanced Multi-Agent Analyse für {symbol}...")
            
            # Market Regime Detection
            regime_analysis = None
            if self.regime_aware_mode and self.regime_detector:
                regime_analysis = await self.regime_detector.detect_current_regime(market_data)
                logger.info(f"🧠 Detected Regime: {regime_analysis.current_regime.value} (Conf: {regime_analysis.confidence:.3f})")
            
            # NAS-enhanced Agent Analysis - NEU!
            agent_signals = await self._nas_enhanced_agent_analysis(market_data, regime_analysis)
            
            if not agent_signals:
                logger.warning("⚠️ Keine Agent Signals erhalten")
                return self._create_neutral_signal(symbol)
            
            # Enhanced Market Context mit NAS Predictions
            market_context = await self._analyze_nas_enhanced_market_context(market_data, regime_analysis)
            
            # NAS-enhanced Consensus Building
            consensus_result = await self._nas_enhanced_consensus(agent_signals, market_context, regime_analysis)
            
            # Conflict Resolution
            if consensus_result.get('conflicts'):
                resolved_signal = await self.conflict_resolver.resolve_conflicts(
                    agent_signals, consensus_result
                )
            else:
                resolved_signal = consensus_result
            
            # Final Regime-enhanced Ensemble Signal
            ensemble_signal = await self._create_nas_enhanced_signal(
                resolved_signal, agent_signals, market_context, regime_analysis
            )
            
            # Performance Update
            await self._update_agent_performance(agent_signals, ensemble_signal)
            
            logger.success(f"🎯 NAS-enhanced Ensemble Signal: {ensemble_signal['action']} (Confidence: {ensemble_signal['confidence']:.3f})")
            
            return ensemble_signal
            
        except Exception as e:
            logger.error(f"❌ Regime-aware Ensemble Signal Generation Fehler: {e}")
            return self._create_neutral_signal(market_data.get('symbol', 'BTC/USDT'))
    
    async def _parallel_agent_analysis(self, market_data: Dict) -> List[AgentSignal]:
        """
        Parallel Analysis aller Agenten
        """
        try:
            # Parallel Tasks für alle Agenten
            analysis_tasks = []
            
            for agent_id, agent in self.agents.items():
                task = asyncio.create_task(
                    self._safe_agent_analysis(agent_id, agent, market_data)
                )
                analysis_tasks.append((agent_id, task))
            
            # Warten auf alle Ergebnisse
            agent_signals = []
            
            for agent_id, task in analysis_tasks:
                try:
                    signal = await asyncio.wait_for(task, timeout=5.0)  # 5s Timeout
                    if signal:
                        agent_signals.append(signal)
                        logger.debug(f"📊 {agent_id}: {signal.action} (Conf: {signal.confidence:.3f})")
                    
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ {agent_id} Analysis Timeout")
                except Exception as e:
                    logger.warning(f"⚠️ {agent_id} Analysis Fehler: {e}")
            
            logger.info(f"📊 {len(agent_signals)}/{len(self.agents)} Agenten haben Signale generiert")
            return agent_signals
            
        except Exception as e:
            logger.error(f"❌ Parallel Agent Analysis Fehler: {e}")
            return []
    
    async def _regime_aware_agent_analysis(self, market_data: Dict, regime_analysis) -> List:
        """
        Agent Analysis mit Regime Context
        """
        try:
            # Add regime context to market data
            enhanced_market_data = market_data.copy()
            if regime_analysis:
                enhanced_market_data['regime_context'] = {
                    'current_regime': regime_analysis.current_regime.value,
                    'regime_confidence': regime_analysis.confidence,
                    'optimal_strategies': regime_analysis.optimal_strategies,
                    'risk_level': regime_analysis.risk_level,
                    'expected_volatility': regime_analysis.expected_volatility
                }
            
            # Parallel Agent Analysis wie bisher
            agent_signals = await self._parallel_agent_analysis(enhanced_market_data)
            
            # Regime-based Signal Filtering
            if regime_analysis:
                agent_signals = await self._filter_signals_by_regime(agent_signals, regime_analysis)
            
            return agent_signals
            
        except Exception as e:
            logger.error(f"❌ Regime-aware Agent Analysis Fehler: {e}")
            return await self._parallel_agent_analysis(market_data)
    
    async def _filter_signals_by_regime(self, agent_signals: List, regime_analysis) -> List:
        """
        Filter Agent Signals basierend auf Regime Compatibility
        """
        try:
            filtered_signals = []
            optimal_strategies = regime_analysis.optimal_strategies
            
            for signal in agent_signals:
                # Agent Specialization Mapping zu Strategien
                agent_strategy_mapping = {
                    'trend_specialist': ['trend_following', 'momentum', 'breakout'],
                    'volatility_expert': ['volatility_trading', 'breakout', 'range_trading'],
                    'sentiment_master': ['contrarian', 'momentum', 'value_investing'],
                    'momentum_tracker': ['momentum', 'trend_following', 'breakout'],
                    'risk_commander': ['defensive', 'conservative', 'safe_haven'],
                    'pattern_recognizer': ['technical_analysis', 'breakout', 'reversal']
                }
                
                agent_type = signal.specialization.value
                agent_strategies = agent_strategy_mapping.get(agent_type, [])
                
                # Check compatibility
                compatibility = len(set(agent_strategies) & set(optimal_strategies)) / len(agent_strategies) if agent_strategies else 0.5
                
                # Adjust signal confidence based on regime compatibility
                if compatibility > 0.5:
                    # Agent ist kompatibel mit Regime
                    signal.confidence *= (1 + compatibility * 0.3)  # Boost bis 30%
                    filtered_signals.append(signal)
                elif compatibility > 0.2:
                    # Teilweise kompatibel
                    signal.confidence *= (0.7 + compatibility * 0.3)  # Leichte Reduktion
                    filtered_signals.append(signal)
                else:
                    # Niedrige Kompatibilität - Signal stark reduzieren
                    signal.confidence *= 0.5
                    if signal.confidence > 0.3:  # Nur wenn noch ausreichend Confidence
                        filtered_signals.append(signal)
            
            logger.debug(f"🔍 Regime Filtering: {len(agent_signals)} → {len(filtered_signals)} Signale")
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"❌ Signal Filtering Fehler: {e}")
            return agent_signals
    
    async def _analyze_enhanced_market_context(self, market_data: Dict, regime_analysis) -> Dict:
        """
        Enhanced Market Context mit Regime Information
        """
        try:
            # Bestehender Market Context
            base_context = await self._analyze_market_context(market_data)
            
            # Regime Enhancement
            if regime_analysis:
                base_context.update({
                    'market_regime': regime_analysis.current_regime.value,
                    'regime_confidence': regime_analysis.confidence,
                    'regime_duration': regime_analysis.regime_duration,
                    'transition_probability': regime_analysis.transition_probability,
                    'regime_risk_level': regime_analysis.risk_level,
                    'regime_expected_volatility': regime_analysis.expected_volatility,
                    'regime_optimal_strategies': regime_analysis.optimal_strategies
                })
                
                # Historical Performance Context
                if regime_analysis.historical_performance:
                    base_context['regime_historical_performance'] = regime_analysis.historical_performance
            
            return base_context
            
        except Exception as e:
            logger.error(f"❌ Enhanced Market Context Analysis Fehler: {e}")
            return await self._analyze_market_context(market_data)
    
    async def _regime_aware_consensus(self, agent_signals: List, market_context: Dict, regime_analysis) -> Dict:
        """
        Regime-aware Consensus Building
        """
        try:
            # Base Consensus
            base_consensus = await self.consensus_engine.build_consensus(agent_signals, market_context)
            
            if not regime_analysis:
                return base_consensus
            
            # Regime-based Consensus Enhancement
            regime_weight = regime_analysis.confidence * 0.3  # Max 30% regime influence
            
            # Adjust consensus based on regime optimality
            consensus_action = base_consensus.get('action', 'HOLD')
            
            # Action-Strategy Mapping
            action_strategies = {
                'BUY': ['trend_following', 'momentum', 'breakout', 'value_investing'],
                'SELL': ['short_selling', 'contrarian', 'defensive'],
                'HOLD': ['conservative', 'range_trading', 'defensive']
            }
            
            action_strategy_list = action_strategies.get(consensus_action, [])
            optimal_strategies = regime_analysis.optimal_strategies
            
            # Strategy Compatibility Score
            strategy_compatibility = len(set(action_strategy_list) & set(optimal_strategies)) / len(action_strategy_list) if action_strategy_list else 0.5
            
            # Adjust consensus confidence
            regime_adjustment = 1 + (strategy_compatibility - 0.5) * regime_weight
            adjusted_confidence = base_consensus.get('confidence', 0.5) * regime_adjustment
            
            # Enhanced consensus result
            enhanced_consensus = base_consensus.copy()
            enhanced_consensus.update({
                'confidence': min(1.0, max(0.0, adjusted_confidence)),
                'regime_enhancement': True,
                'strategy_compatibility': strategy_compatibility,
                'regime_adjustment_factor': regime_adjustment
            })
            
            return enhanced_consensus
            
        except Exception as e:
            logger.error(f"❌ Regime-aware Consensus Fehler: {e}")
            return await self.consensus_engine.build_consensus(agent_signals, market_context)
    
    async def _create_regime_enhanced_signal(self, consensus_result: Dict, agent_signals: List,
                                           market_context: Dict, regime_analysis) -> Dict:
        """
        Regime-enhanced Ensemble Signal erstellen
        """
        try:
            # Base Ensemble Signal
            base_signal = await self._create_ensemble_signal(consensus_result, agent_signals, market_context)
            
            if not regime_analysis:
                return base_signal
            
            # Regime Enhancement
            regime_info = {
                'regime_detected': regime_analysis.current_regime.value,
                'regime_confidence': regime_analysis.confidence,
                'regime_duration': regime_analysis.regime_duration,
                'regime_transition_prob': regime_analysis.transition_probability,
                'regime_optimal_strategies': regime_analysis.optimal_strategies,
                'regime_risk_level': regime_analysis.risk_level,
                'regime_expected_volatility': regime_analysis.expected_volatility
            }
            
            # Position Size Adjustment based on Regime
            base_position_size = base_signal.get('position_size', 0.1)
            
            # Regime-based Position Size Multipliers
            regime_position_multipliers = {
                'bull_trending': 1.2,
                'bear_trending': 0.7,
                'sideways_range': 1.0,
                'high_volatility': 0.6,
                'low_volatility': 1.3,
                'bull_correction': 0.8,
                'bear_rally': 0.8,
                'breakout': 1.1,
                'accumulation': 1.1,
                'distribution': 0.9,
                'crisis': 0.4
            }
            
            regime_multiplier = regime_position_multipliers.get(
                regime_analysis.current_regime.value, 1.0
            )
            
            # Confidence-weighted adjustment
            confidence_weight = regime_analysis.confidence
            final_multiplier = 1.0 + (regime_multiplier - 1.0) * confidence_weight
            
            regime_adjusted_position_size = base_position_size * final_multiplier
            regime_adjusted_position_size = min(0.3, max(0.01, regime_adjusted_position_size))
            
            # Enhanced Signal
            enhanced_signal = base_signal.copy()
            enhanced_signal.update({
                'position_size': regime_adjusted_position_size,
                'regime_info': regime_info,
                'regime_enhanced': True,
                'regime_position_multiplier': final_multiplier,
                'source': 'Regime_Enhanced_Multi_Agent_Ensemble'
            })
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"❌ Regime-enhanced Signal Creation Fehler: {e}")
            return await self._create_ensemble_signal(consensus_result, agent_signals, market_context)
    
    async def _safe_agent_analysis(self, agent_id: str, agent: 'SpecializedTradingAgent', 
                                 market_data: Dict) -> Optional[AgentSignal]:
        """
        Sichere Agent Analysis mit Error Handling
        """
        try:
            # Update Shared Memory für Agent
            await self.shared_memory.update_market_data(market_data)
            
            # Agent Signal Generation
            signal = await agent.generate_signal(market_data)
            
            if signal:
                # Signal Validation
                if self._validate_agent_signal(signal):
                    # Performance Tracking
                    self.agent_performance[agent_id]['signals_generated'] += 1
                    self.agent_performance[agent_id]['last_signal'] = signal
                    
                    return signal
                else:
                    logger.warning(f"⚠️ {agent_id} Signal Validation fehlgeschlagen")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ {agent_id} Safe Analysis Fehler: {e}")
            return None
    
    def _validate_agent_signal(self, signal: AgentSignal) -> bool:
        """
        Agent Signal Validation
        """
        try:
            # Basic Validation
            if not signal.action or signal.action not in ['BUY', 'SELL', 'HOLD']:
                return False
            
            if not (0.0 <= signal.confidence <= 1.0):
                return False
            
            if not (0.0 <= signal.position_size <= 1.0):
                return False
            
            if not signal.agent_id or not signal.specialization:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Signal Validation Fehler: {e}")
            return False
    
    async def _analyze_market_context(self, market_data: Dict) -> Dict:
        """
        Umfassende Market Context Analysis
        """
        try:
            context = {
                'volatility_regime': await self._detect_volatility_regime(market_data),
                'trend_strength': await self._calculate_trend_strength(market_data),
                'market_sentiment': await self._assess_market_sentiment(market_data),
                'liquidity_conditions': await self._analyze_liquidity(market_data),
                'time_of_day': self._get_trading_session_info(),
                'recent_news_impact': await self._assess_news_impact(market_data),
                'correlation_environment': await self._analyze_correlations(market_data)
            }
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Market Context Analysis Fehler: {e}")
            return {}
    
    async def _create_ensemble_signal(self, consensus_result: Dict, 
                                    agent_signals: List[AgentSignal],
                                    market_context: Dict) -> Dict:
        """
        Finales Ensemble Signal erstellen
        """
        try:
            # Consensus Informationen extrahieren
            consensus_action = consensus_result.get('action', 'HOLD')
            consensus_confidence = consensus_result.get('confidence', 0.5)
            
            # Agent Weights basierend auf Performance
            agent_weights = await self._calculate_agent_weights(agent_signals)
            
            # Position Size Berechnung
            ensemble_position_size = await self._calculate_ensemble_position_size(
                agent_signals, agent_weights, market_context
            )
            
            # Risk Assessment
            ensemble_risk = await self._assess_ensemble_risk(
                agent_signals, market_context
            )
            
            # Reasoning Aggregation
            reasoning_summary = await self._aggregate_reasoning(agent_signals)
            
            # Final Signal Construction
            ensemble_signal = {
                'action': consensus_action,
                'confidence': min(consensus_confidence, 1.0),
                'position_size': min(ensemble_position_size, 1.0),
                'source': 'Multi_Agent_Ensemble',
                'agent_count': len(agent_signals),
                'consensus_strength': consensus_result.get('consensus_strength', 0.5),
                'risk_level': ensemble_risk.get('risk_level', 'medium'),
                'expected_duration': consensus_result.get('expected_duration', 'medium'),
                'reasoning': reasoning_summary,
                'technical_factors': consensus_result.get('technical_factors', {}),
                'agent_breakdown': {
                    'buy_votes': len([s for s in agent_signals if s.action == 'BUY']),
                    'sell_votes': len([s for s in agent_signals if s.action == 'SELL']),
                    'hold_votes': len([s for s in agent_signals if s.action == 'HOLD']),
                    'avg_confidence': np.mean([s.confidence for s in agent_signals])
                },
                'market_context': market_context,
                'timestamp': datetime.now(),
                'ensemble_id': f"ensemble_{int(datetime.now().timestamp())}"
            }
            
            # Performance Tracking
            await self.ensemble_performance.track_signal(ensemble_signal, agent_signals)
            
            return ensemble_signal
            
        except Exception as e:
            logger.error(f"❌ Ensemble Signal Creation Fehler: {e}")
            return self._create_neutral_signal("UNKNOWN")
    
    def _create_neutral_signal(self, symbol: str) -> Dict:
        """
        Neutrales Signal als Fallback
        """
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'position_size': 0.0,
            'source': 'Multi_Agent_Fallback',
            'agent_count': 0,
            'reasoning': 'No agent signals available',
            'timestamp': datetime.now(),
            'symbol': symbol
        }
    
    async def _calculate_agent_weights(self, agent_signals: List[AgentSignal]) -> Dict[str, float]:
        """
        Dynamische Agent Gewichtung basierend auf Performance
        """
        try:
            weights = {}
            
            for signal in agent_signals:
                agent_id = signal.agent_id
                
                if agent_id in self.agent_performance:
                    perf = self.agent_performance[agent_id]
                    
                    # Performance Metrics
                    accuracy = perf.get('accuracy', 0.5)
                    signal_count = perf.get('signals_generated', 1)
                    specialization_score = perf.get('specialization_score', 0.5)
                    
                    # Confidence Adjustment
                    confidence_factor = signal.confidence
                    
                    # Recency Factor
                    recency_factor = 1.0  # Kann basierend auf letzter Aktivität angepasst werden
                    
                    # Final Weight Calculation
                    base_weight = 1.0 / len(agent_signals)  # Equal base weight
                    performance_multiplier = (accuracy * 0.4 + 
                                           specialization_score * 0.3 + 
                                           confidence_factor * 0.3)
                    
                    final_weight = base_weight * performance_multiplier * recency_factor
                    weights[agent_id] = final_weight
                else:
                    weights[agent_id] = 1.0 / len(agent_signals)  # Default weight
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
            return weights
            
        except Exception as e:
            logger.error(f"❌ Agent Weights Calculation Fehler: {e}")
            return {}
    
    async def _calculate_ensemble_position_size(self, agent_signals: List[AgentSignal],
                                              agent_weights: Dict[str, float],
                                              market_context: Dict) -> float:
        """
        Intelligente Ensemble Position Size Berechnung
        """
        try:
            if not agent_signals:
                return 0.0
            
            # Weighted Position Size
            weighted_positions = []
            
            for signal in agent_signals:
                weight = agent_weights.get(signal.agent_id, 0.0)
                
                # Action zu Position Size Mapping
                action_multiplier = {
                    'BUY': 1.0,
                    'SELL': 1.0,  # Für Short Positions
                    'HOLD': 0.0
                }.get(signal.action, 0.0)
                
                weighted_position = signal.position_size * action_multiplier * weight
                weighted_positions.append(weighted_position)
            
            # Base Position Size
            base_position_size = sum(weighted_positions)
            
            # Market Context Adjustments
            volatility_adjustment = self._get_volatility_adjustment(market_context)
            liquidity_adjustment = self._get_liquidity_adjustment(market_context)
            sentiment_adjustment = self._get_sentiment_adjustment(market_context)
            
            # Final Position Size
            final_position_size = (base_position_size * 
                                 volatility_adjustment * 
                                 liquidity_adjustment * 
                                 sentiment_adjustment)
            
            return max(0.0, min(1.0, final_position_size))
            
        except Exception as e:
            logger.error(f"❌ Ensemble Position Size Calculation Fehler: {e}")
            return 0.1  # Conservative fallback
    
    def _get_volatility_adjustment(self, market_context: Dict) -> float:
        """
        Volatility-basierte Position Size Anpassung
        """
        volatility_regime = market_context.get('volatility_regime', 'normal')
        
        adjustments = {
            'low': 1.2,      # Größere Position bei niedriger Volatilität
            'normal': 1.0,   # Standard Position
            'high': 0.7,     # Kleinere Position bei hoher Volatilität
            'extreme': 0.4   # Sehr kleine Position bei extremer Volatilität
        }
        
        return adjustments.get(volatility_regime, 1.0)
    
    def _get_liquidity_adjustment(self, market_context: Dict) -> float:
        """
        Liquidity-basierte Position Size Anpassung
        """
        liquidity = market_context.get('liquidity_conditions', 'normal')
        
        adjustments = {
            'high': 1.1,     # Leicht größere Position bei hoher Liquidität
            'normal': 1.0,   # Standard Position
            'low': 0.8,      # Kleinere Position bei niedriger Liquidität
            'very_low': 0.5  # Sehr kleine Position bei sehr niedriger Liquidität
        }
        
        return adjustments.get(liquidity, 1.0)
    
    def _get_sentiment_adjustment(self, market_context: Dict) -> float:
        """
        Sentiment-basierte Position Size Anpassung
        """
        sentiment = market_context.get('market_sentiment', 'neutral')
        
        adjustments = {
            'very_bullish': 1.15,
            'bullish': 1.05,
            'neutral': 1.0,
            'bearish': 0.95,
            'very_bearish': 0.85
        }
        
        return adjustments.get(sentiment, 1.0)
    
    # Placeholder-Methoden für Market Context Analysis
    async def _detect_volatility_regime(self, market_data: Dict) -> str:
        """Volatility Regime Detection"""
        try:
            volatility = market_data.get('volatility', 0.02)
            if volatility < 0.01:
                return 'low'
            elif volatility < 0.03:
                return 'normal'
            elif volatility < 0.06:
                return 'high'
            else:
                return 'extreme'
        except:
            return 'normal'
    
    async def _calculate_trend_strength(self, market_data: Dict) -> float:
        """Trend Strength Calculation"""
        return 0.5  # Placeholder
    
    async def _assess_market_sentiment(self, market_data: Dict) -> str:
        """Market Sentiment Assessment"""
        return 'neutral'  # Placeholder
    
    async def _analyze_liquidity(self, market_data: Dict) -> str:
        """Liquidity Analysis"""
        return 'normal'  # Placeholder
    
    def _get_trading_session_info(self) -> Dict:
        """Trading Session Information"""
        return {'session': 'active', 'timezone': 'UTC'}
    
    async def _assess_news_impact(self, market_data: Dict) -> str:
        """News Impact Assessment"""
        return 'neutral'  # Placeholder
    
    async def _analyze_correlations(self, market_data: Dict) -> Dict:
        """Correlation Analysis"""
        return {'correlation_strength': 0.5}  # Placeholder
    
    async def _assess_ensemble_risk(self, agent_signals: List[AgentSignal], 
                                  market_context: Dict) -> Dict:
        """Ensemble Risk Assessment"""
        return {'risk_level': 'medium', 'risk_score': 0.5}
    
    async def _aggregate_reasoning(self, agent_signals: List[AgentSignal]) -> str:
        """Agent Reasoning Aggregation"""
        reasonings = [signal.reasoning for signal in agent_signals if signal.reasoning]
        return f"Ensemble of {len(reasonings)} agent analyses"
    
    async def _update_agent_performance(self, agent_signals: List[AgentSignal], 
                                      ensemble_signal: Dict):
        """Agent Performance Update"""
        # Performance tracking implementation
        for signal in agent_signals:
            if signal.agent_id in self.agent_performance:
                perf = self.agent_performance[signal.agent_id]
                perf['avg_confidence'] = np.mean([perf.get('avg_confidence', 0.5), signal.confidence])
    
    def get_system_status(self) -> Dict:
        """
        Multi-Agent System Status
        """
        return {
            'active_agents': len(self.agents),
            'enabled_specializations': [spec.value for spec in self.enabled_agents],
            'agent_performance': self.agent_performance.copy(),
            'system_health': 'operational',
            'last_ensemble_signal': getattr(self, '_last_ensemble_signal', None)
        }
    
    def get_regime_status(self) -> Dict:
        """
        Regime Detection Status
        """
        if self.regime_detector:
            return self.regime_detector.get_regime_status()
        else:
            return {'regime_aware_mode': False}
    
    def get_enhanced_system_status(self) -> Dict:
        """
        Enhanced System Status mit Regime Information
        """
        base_status = self.get_system_status()
        
        if self.regime_aware_mode and self.regime_detector:
            regime_status = self.get_regime_status()
            base_status.update({
                'regime_aware_mode': True,
                'regime_detector_status': regime_status,
                'current_regime': regime_status.get('current_regime', 'unknown')
            })
        else:
            base_status['regime_aware_mode'] = False
        
        return base_status
    
    async def _nas_enhanced_agent_analysis(self, market_data: Dict, regime_analysis) -> List:
        """
        Agent Analysis mit NAS-enhanced Predictions
        """
        try:
            # Standard agent analysis
            agent_signals = await self._regime_aware_agent_analysis(market_data, regime_analysis)
            
            if not self.nas_enabled or not self.optimized_models:
                return agent_signals
            
            # NAS-enhanced Signal Confidence Adjustment
            for signal in agent_signals:
                try:
                    # Predict signal confidence using NAS model
                    if 'signal_confidence_prediction' in self.optimized_models:
                        confidence_features = await self._extract_confidence_features(signal, market_data)
                        predicted_confidence = await self._predict_with_nas_model(
                            'signal_confidence_prediction', confidence_features
                        )
                        
                        # Blend original and predicted confidence
                        original_confidence = signal.confidence
                        blended_confidence = original_confidence * 0.7 + predicted_confidence * 0.3
                        signal.confidence = min(1.0, max(0.0, blended_confidence))
                        
                        logger.debug(f"🧠 NAS Confidence Adjustment: {original_confidence:.3f} → {signal.confidence:.3f}")
                
                except Exception as e:
                    logger.warning(f"⚠️ NAS Confidence Prediction Fehler: {e}")
            
            return agent_signals
            
        except Exception as e:
            logger.error(f"❌ NAS-enhanced Agent Analysis Fehler: {e}")
            return await self._regime_aware_agent_analysis(market_data, regime_analysis)
    
    async def _analyze_nas_enhanced_market_context(self, market_data: Dict, regime_analysis) -> Dict:
        """
        Market Context mit NAS-enhanced Predictions
        """
        try:
            # Base market context
            base_context = await self._analyze_enhanced_market_context(market_data, regime_analysis)
            
            if not self.nas_enabled or not self.optimized_models:
                return base_context
            
            # NAS-enhanced Predictions
            nas_predictions = {}
            
            # Volatility Forecast
            if 'market_volatility_forecast' in self.optimized_models:
                try:
                    volatility_features = await self._extract_volatility_features(market_data)
                    predicted_volatility = await self._predict_with_nas_model(
                        'market_volatility_forecast', volatility_features
                    )
                    nas_predictions['predicted_volatility'] = predicted_volatility
                except Exception as e:
                    logger.warning(f"⚠️ NAS Volatility Prediction Fehler: {e}")
            
            # Trend Strength Assessment
            if 'trend_strength_assessment' in self.optimized_models:
                try:
                    trend_features = await self._extract_trend_features(market_data)
                    predicted_trend_strength = await self._predict_with_nas_model(
                        'trend_strength_assessment', trend_features
                    )
                    nas_predictions['predicted_trend_strength'] = predicted_trend_strength
                except Exception as e:
                    logger.warning(f"⚠️ NAS Trend Prediction Fehler: {e}")
            
            # Risk Score Calculation
            if 'risk_score_calculation' in self.optimized_models:
                try:
                    risk_features = await self._extract_risk_features(market_data)
                    predicted_risk_score = await self._predict_with_nas_model(
                        'risk_score_calculation', risk_features
                    )
                    nas_predictions['predicted_risk_score'] = predicted_risk_score
                except Exception as e:
                    logger.warning(f"⚠️ NAS Risk Prediction Fehler: {e}")
            
            # Add NAS predictions to context
            base_context['nas_predictions'] = nas_predictions
            
            return base_context
            
        except Exception as e:
            logger.error(f"❌ NAS-enhanced Market Context Fehler: {e}")
            return await self._analyze_enhanced_market_context(market_data, regime_analysis)
    
    async def _nas_enhanced_consensus(self, agent_signals: List, market_context: Dict, regime_analysis) -> Dict:
        """
        NAS-enhanced Consensus Building
        """
        try:
            # Base consensus
            base_consensus = await self._regime_aware_consensus(agent_signals, market_context, regime_analysis)
            
            if not self.nas_enabled or 'nas_predictions' not in market_context:
                return base_consensus
            
            # NAS-based Consensus Enhancement
            nas_predictions = market_context['nas_predictions']
            
            # Confidence adjustment based on NAS predictions
            confidence_adjustments = []
            
            # Volatility adjustment
            if 'predicted_volatility' in nas_predictions:
                volatility = nas_predictions['predicted_volatility']
                if volatility > 0.05:  # High predicted volatility
                    confidence_adjustments.append(0.9)  # Reduce confidence
                elif volatility < 0.015:  # Low predicted volatility
                    confidence_adjustments.append(1.1)  # Increase confidence
                else:
                    confidence_adjustments.append(1.0)  # No change
            
            # Trend strength adjustment
            if 'predicted_trend_strength' in nas_predictions:
                trend_strength = nas_predictions['predicted_trend_strength']
                if trend_strength > 0.7:  # Strong trend predicted
                    confidence_adjustments.append(1.15)  # Increase confidence
                elif trend_strength < 0.3:  # Weak trend predicted
                    confidence_adjustments.append(0.85)  # Reduce confidence
                else:
                    confidence_adjustments.append(1.0)
            
            # Risk score adjustment
            if 'predicted_risk_score' in nas_predictions:
                risk_score = nas_predictions['predicted_risk_score']
                if risk_score > 0.7:  # High risk predicted
                    confidence_adjustments.append(0.8)  # Reduce confidence significantly
                elif risk_score < 0.3:  # Low risk predicted
                    confidence_adjustments.append(1.1)  # Increase confidence
                else:
                    confidence_adjustments.append(1.0)
            
            # Apply adjustments
            if confidence_adjustments:
                avg_adjustment = sum(confidence_adjustments) / len(confidence_adjustments)
                original_confidence = base_consensus.get('confidence', 0.5)
                adjusted_confidence = original_confidence * avg_adjustment
                
                base_consensus['confidence'] = min(1.0, max(0.0, adjusted_confidence))
                base_consensus['nas_enhanced'] = True
                base_consensus['nas_adjustment_factor'] = avg_adjustment
                
                logger.debug(f"🧠 NAS Consensus Enhancement: {original_confidence:.3f} → {base_consensus['confidence']:.3f}")
            
            return base_consensus
            
        except Exception as e:
            logger.error(f"❌ NAS-enhanced Consensus Fehler: {e}")
            return await self._regime_aware_consensus(agent_signals, market_context, regime_analysis)
    
    async def _create_nas_enhanced_signal(self, consensus_result: Dict, agent_signals: List,
                                        market_context: Dict, regime_analysis) -> Dict:
        """
        NAS-enhanced Ensemble Signal erstellen
        """
        try:
            # Base regime-enhanced signal
            base_signal = await self._create_regime_enhanced_signal(
                consensus_result, agent_signals, market_context, regime_analysis
            )
            
            if not self.nas_enabled or 'nas_predictions' not in market_context:
                return base_signal
            
            # NAS Enhancement
            nas_predictions = market_context['nas_predictions']
            
            # Add NAS information to signal
            base_signal['nas_enhanced'] = True
            base_signal['nas_predictions'] = nas_predictions
            
            # NAS-based Position Size Adjustment
            base_position_size = base_signal.get('position_size', 0.1)
            
            # Risk-based position adjustment
            if 'predicted_risk_score' in nas_predictions:
                risk_score = nas_predictions['predicted_risk_score']
                risk_multiplier = 1.0 - (risk_score * 0.5)  # Reduce size based on risk
                base_position_size *= risk_multiplier
            
            # Volatility-based position adjustment
            if 'predicted_volatility' in nas_predictions:
                volatility = nas_predictions['predicted_volatility']
                if volatility > 0.05:  # High volatility
                    base_position_size *= 0.8  # Reduce position size
                elif volatility < 0.015:  # Low volatility
                    base_position_size *= 1.2  # Increase position size
            
            # Trend strength-based adjustment
            if 'predicted_trend_strength' in nas_predictions:
                trend_strength = nas_predictions['predicted_trend_strength']
                if trend_strength > 0.7:  # Strong trend
                    base_position_size *= 1.1  # Slightly increase
            
            base_signal['position_size'] = min(0.25, max(0.01, base_position_size))
            base_signal['source'] = 'NAS_Enhanced_Multi_Agent_Ensemble'
            
            return base_signal
            
        except Exception as e:
            logger.error(f"❌ NAS-enhanced Signal Creation Fehler: {e}")
            return await self._create_regime_enhanced_signal(
                consensus_result, agent_signals, market_context, regime_analysis
            )
    
    async def _predict_with_nas_model(self, task_type: str, features: List[float]) -> float:
        """
        Prediction mit NAS-optimiertem Model
        """
        try:
            architecture = self.optimized_models.get(task_type)
            if not architecture:
                return 0.5  # Default prediction
            
            # Build model from architecture (simplified for demo)
            # In production: Use cached/persisted models
            model = self.nas_engine._build_pytorch_model(architecture)
            model.eval()
            
            # Convert features to tensor
            import torch
            features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                prediction = model(features_tensor)
                return float(prediction.item())
                
        except Exception as e:
            logger.error(f"❌ NAS Model Prediction Fehler für {task_type}: {e}")
            return 0.5
    
    async def _extract_confidence_features(self, signal, market_data: Dict) -> List[float]:
        """
        Features für Signal Confidence Prediction extrahieren
        """
        try:
            features = []
            
            # Signal-related features
            features.append(signal.confidence)
            features.append(len(signal.reasoning) / 100.0 if signal.reasoning else 0)  # Reasoning length
            features.append(1.0 if signal.action == 'BUY' else -1.0 if signal.action == 'SELL' else 0.0)
            
            # Market-related features
            features.append(market_data.get('volatility', 0.02))
            features.append(market_data.get('volume', 1000000) / 1000000)  # Normalized volume
            
            # Agent-specific features
            agent_type_encoding = {
                'trend_specialist': 0.2,
                'volatility_expert': 0.4,
                'sentiment_master': 0.6,
                'momentum_tracker': 0.8,
                'risk_commander': 1.0
            }
            agent_encoding = agent_type_encoding.get(signal.specialization.value, 0.5)
            features.append(agent_encoding)
            
            # Pad to 50 features
            while len(features) < 50:
                features.append(0.0)
            
            return features[:50]
            
        except Exception as e:
            logger.error(f"❌ Confidence Features Extraction Fehler: {e}")
            return [0.0] * 50
    
    async def _extract_volatility_features(self, market_data: Dict) -> List[float]:
        """
        Features für Volatility Prediction extrahieren
        """
        try:
            features = []
            
            # Current volatility
            current_vol = market_data.get('volatility', 0.02)
            features.append(current_vol)
            features.append(current_vol ** 2)  # Squared volatility
            features.append(np.sqrt(current_vol))  # Square root
            
            # Price-related features
            close_price = market_data.get('close', 50000)
            high_price = market_data.get('high', close_price)
            low_price = market_data.get('low', close_price)
            
            daily_range = (high_price - low_price) / close_price
            features.append(daily_range)
            features.append(daily_range ** 2)
            
            # Volume features
            volume = market_data.get('volume', 1000000)
            features.append(volume / 1000000)  # Normalized volume
            features.append(np.log(volume / 1000000 + 1))  # Log volume
            
            # Time-based features (mock)
            import datetime
            hour = datetime.datetime.now().hour
            features.append(hour / 24.0)  # Hour of day
            features.append(np.sin(2 * np.pi * hour / 24))  # Cyclical hour
            features.append(np.cos(2 * np.pi * hour / 24))
            
            # Pad to 50 features
            while len(features) < 50:
                features.append(0.0)
            
            return features[:50]
            
        except Exception as e:
            logger.error(f"❌ Volatility Features Extraction Fehler: {e}")
            return [0.0] * 50
    
    async def _extract_trend_features(self, market_data: Dict) -> List[float]:
        """
        Features für Trend Strength Prediction extrahieren
        """
        try:
            features = []
            
            # Price momentum (mock calculation)
            close_price = market_data.get('close', 50000)
            features.append(close_price / 50000 - 1)  # Normalized price change
            
            # Volume trend
            volume = market_data.get('volume', 1000000)
            features.append(volume / 1000000)
            
            # Volatility trend
            volatility = market_data.get('volatility', 0.02)
            features.append(volatility)
            
            # Mock technical indicators
            features.extend([0.5, 0.3, 0.7, 0.4, 0.6])  # RSI, MACD, etc.
            
            # Pad to 50 features
            while len(features) < 50:
                features.append(0.0)
            
            return features[:50]
            
        except Exception as e:
            logger.error(f"❌ Trend Features Extraction Fehler: {e}")
            return [0.0] * 50
    
    async def _extract_risk_features(self, market_data: Dict) -> List[float]:
        """
        Features für Risk Score Prediction extrahieren
        """
        try:
            features = []
            
            # Volatility-based risk
            volatility = market_data.get('volatility', 0.02)
            features.append(volatility)
            features.append(min(1.0, volatility / 0.1))  # Normalized vol risk
            
            # Price-based risk
            close_price = market_data.get('close', 50000)
            high_price = market_data.get('high', close_price)
            low_price = market_data.get('low', close_price)
            
            price_risk = (high_price - low_price) / close_price
            features.append(price_risk)
            
            # Volume-based risk
            volume = market_data.get('volume', 1000000)
            features.append(volume / 1000000)
            
            # Market structure risk (mock)
            features.extend([0.3, 0.4, 0.2, 0.5, 0.6])
            
            # Pad to 50 features
            while len(features) < 50:
                features.append(0.0)
            
            return features[:50]
            
        except Exception as e:
            logger.error(f"❌ Risk Features Extraction Fehler: {e}")
            return [0.0] * 50
    
    def get_nas_status(self) -> Dict:
        """
        NAS System Status
        """
        if self.nas_engine and self.nas_enabled:
            nas_summary = self.nas_engine.get_architecture_summary()
            return {
                'nas_enabled': True,
                'optimized_models': len(self.optimized_models),
                'model_tasks': list(self.optimized_models.keys()),
                'nas_summary': nas_summary
            }
        else:
            return {'nas_enabled': False}
    
    def get_ultimate_system_status(self) -> Dict:
        """
        Ultimate System Status mit allen Features
        """
        base_status = self.get_enhanced_system_status()
        
        # Add NAS status
        nas_status = self.get_nas_status()
        base_status['nas_system'] = nas_status
        
        # Feature summary
        base_status['features_enabled'] = {
            'multi_agent_ensemble': True,
            'regime_detection': self.regime_aware_mode,
            'neural_architecture_search': self.nas_enabled,
            'reinforcement_learning': True  # From previous upgrade
        }
        
        return base_status


class AgentCoordinator:
    """
    Koordiniert die Zusammenarbeit zwischen Agenten
    """
    
    def __init__(self, config):
        self.config = config
        self.coordination_history = deque(maxlen=1000)
        
    async def initialize(self):
        logger.info("🎯 Agent Coordinator initialisiert")
        return True


class ConsensusEngine:
    """
    Consensus Building zwischen Agenten
    """
    
    def __init__(self, config):
        self.config = config
        self.consensus_threshold = config.get('consensus_threshold', 0.6)
        
    async def build_consensus(self, agent_signals: List[AgentSignal], 
                            market_context: Dict) -> Dict:
        """
        Consensus aus Agent Signals erstellen
        """
        try:
            if not agent_signals:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            # Vote Counting
            actions = [signal.action for signal in agent_signals]
            action_counts = {action: actions.count(action) for action in set(actions)}
            
            # Majority Action
            majority_action = max(action_counts, key=action_counts.get)
            majority_count = action_counts[majority_action]
            
            # Consensus Strength
            consensus_strength = majority_count / len(agent_signals)
            
            # Confidence Aggregation
            action_signals = [s for s in agent_signals if s.action == majority_action]
            avg_confidence = np.mean([s.confidence for s in action_signals])
            
            # Consensus Result
            consensus_result = {
                'action': majority_action,
                'confidence': avg_confidence * consensus_strength,
                'consensus_strength': consensus_strength,
                'conflicts': consensus_strength < self.consensus_threshold,
                'vote_breakdown': action_counts,
                'participating_agents': len(agent_signals)
            }
            
            return consensus_result
            
        except Exception as e:
            logger.error(f"❌ Consensus Building Fehler: {e}")
            return {'action': 'HOLD', 'confidence': 0.0}


class ConflictResolver:
    """
    Löst Konflikte zwischen Agent Signalen
    """
    
    def __init__(self, config):
        self.config = config
        
    async def resolve_conflicts(self, agent_signals: List[AgentSignal], 
                              consensus_result: Dict) -> Dict:
        """
        Konflikt-Resolution zwischen Agenten
        """
        try:
            # Für jetzt: Einfache Gewichtung nach Confidence
            if not agent_signals:
                return consensus_result
            
            # Highest Confidence Signal
            highest_confidence_signal = max(agent_signals, key=lambda s: s.confidence)
            
            # Conflict Resolution basierend auf höchster Confidence
            resolved_result = consensus_result.copy()
            
            if highest_confidence_signal.confidence > 0.8:
                resolved_result['action'] = highest_confidence_signal.action
                resolved_result['confidence'] = highest_confidence_signal.confidence
                resolved_result['resolution_method'] = 'highest_confidence'
            
            return resolved_result
            
        except Exception as e:
            logger.error(f"❌ Conflict Resolution Fehler: {e}")
            return consensus_result


class AgentMessageBus:
    """
    Message Bus für Agent-Kommunikation
    """
    
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.subscribers = {}
        
    async def start(self):
        logger.info("📡 Agent Message Bus gestartet")
        return True
    
    async def publish(self, topic: str, message: Dict):
        await self.message_queue.put({'topic': topic, 'message': message})
    
    async def subscribe(self, agent_id: str, topics: List[str]):
        self.subscribers[agent_id] = topics


class SharedMarketMemory:
    """
    Geteilter Speicher für Market Data zwischen Agenten
    """
    
    def __init__(self):
        self.market_data_cache = {}
        self.indicator_cache = {}
        self.pattern_cache = {}
        
    async def initialize(self):
        logger.info("🧠 Shared Market Memory initialisiert")
        return True
    
    async def update_market_data(self, market_data: Dict):
        symbol = market_data.get('symbol', 'UNKNOWN')
        self.market_data_cache[symbol] = {
            'data': market_data,
            'timestamp': datetime.now()
        }
    
    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        return self.market_data_cache.get(symbol, {}).get('data')


class EnsemblePerformanceTracker:
    """
    Performance Tracking für das Ensemble System
    """
    
    def __init__(self):
        self.ensemble_history = deque(maxlen=10000)
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'best_performing_agent': None
        }
        
    async def initialize(self):
        logger.info("📊 Ensemble Performance Tracker initialisiert")
        return True
    
    async def track_signal(self, ensemble_signal: Dict, agent_signals: List[AgentSignal]):
        """
        Ensemble Signal Performance Tracking
        """
        try:
            self.ensemble_history.append({
                'ensemble_signal': ensemble_signal,
                'agent_signals': agent_signals,
                'timestamp': datetime.now()
            })
            
            self.performance_metrics['total_signals'] += 1
            
        except Exception as e:
            logger.error(f"❌ Ensemble Performance Tracking Fehler: {e}")


# ===== SPEZIALISIERTE TRADING AGENTEN =====

class SpecializedTradingAgent:
    """
    Basis-Klasse für alle spezialisierten Trading Agenten
    """
    
    def __init__(self, config, trading_engine, message_bus, shared_memory):
        self.config = config
        self.trading_engine = trading_engine
        self.message_bus = message_bus
        self.shared_memory = shared_memory
        
        # Agent Properties
        self.agent_id = None
        self.specialization = None
        self.expertise_areas = []
        self.performance_history = deque(maxlen=1000)
        
        # Neural Network Components
        self.decision_network = None
        self.confidence_estimator = None
        
        # Learning Components
        self.experience_buffer = deque(maxlen=5000)
        self.learning_rate = 0.001
        self.training_enabled = True
        
    async def initialize(self):
        """
        Agent Initialization - Override in subclasses
        """
        raise NotImplementedError
    
    async def generate_signal(self, market_data: Dict) -> Optional[AgentSignal]:
        """
        Signal Generation - Override in subclasses
        """
        raise NotImplementedError
    
    def _create_agent_signal(self, action: str, confidence: float, position_size: float,
                           reasoning: str, technical_factors: Dict, 
                           risk_assessment: Dict, market_data: Dict) -> AgentSignal:
        """
        Standardisiertes Agent Signal erstellen
        """
        return AgentSignal(
            agent_id=self.agent_id,
            specialization=self.specialization,
            action=action,
            confidence=confidence,
            position_size=position_size,
            reasoning=reasoning,
            technical_factors=technical_factors,
            risk_assessment=risk_assessment,
            timestamp=datetime.now(),
            priority=self._calculate_priority(confidence, risk_assessment),
            expected_duration=self._estimate_duration(market_data, technical_factors),
            market_conditions=self._assess_market_conditions(market_data)
        )
    
    def _calculate_priority(self, confidence: float, risk_assessment: Dict) -> str:
        """Priority basierend auf Confidence und Risk"""
        if confidence > 0.8 and risk_assessment.get('risk_score', 0.5) < 0.3:
            return 'critical'
        elif confidence > 0.6:
            return 'high'
        elif confidence > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_duration(self, market_data: Dict, technical_factors: Dict) -> str:
        """Duration Estimation basierend auf Market Conditions"""
        volatility = market_data.get('volatility', 0.02)
        if volatility > 0.05:
            return 'scalp'  # High volatility = short-term
        elif volatility > 0.03:
            return 'short'
        elif volatility > 0.015:
            return 'medium'
        else:
            return 'long'
    
    def _assess_market_conditions(self, market_data: Dict) -> Dict:
        """Market Conditions Assessment"""
        return {
            'volatility': market_data.get('volatility', 0.02),
            'volume': market_data.get('volume', 0),
            'trend': market_data.get('trend', 'neutral'),
            'liquidity': 'normal'
        }
    
    def _generate_mock_price_history(self, current_price: float) -> List[float]:
        """Mock Price History für Demo"""
        # Generiere realistische Price History
        history = []
        price = current_price * 0.95  # Start 5% niedriger
        
        for i in range(100):
            # Random Walk mit leichtem Uptrend
            change = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
            price *= (1 + change)
            history.append(price)
        
        return history


class TrendSpecialistAgent(SpecializedTradingAgent):
    """
    🎯 Trend-Following Spezialist
    Experte für Trend-Erkennung und Trend-Following-Strategien
    """
    
    async def initialize(self):
        self.agent_id = f"trend_specialist_{id(self)}"
        self.specialization = AgentSpecialization.TREND_SPECIALIST
        self.expertise_areas = [
            'trend_detection', 'momentum_analysis', 'breakout_patterns',
            'moving_averages', 'trend_strength', 'directional_movement'
        ]
        
        # Spezialisierte Neural Networks
        self.trend_detector = self._build_trend_detection_network()
        self.momentum_analyzer = self._build_momentum_network()
        
        # Trend-spezifische Parameter
        self.trend_confirmation_period = 20
        self.momentum_threshold = 0.02
        self.breakout_strength_threshold = 0.05
        
        logger.success("🎯 Trend Specialist Agent initialisiert")
        return True
    
    def _build_trend_detection_network(self):
        """
        Spezialisiertes Neural Network für Trend Detection
        """
        return nn.Sequential(
            nn.Linear(30, 128),  # 30 trend-specific features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # Strong Up, Neutral, Strong Down
            nn.Softmax(dim=1)
        )
    
    def _build_momentum_network(self):
        """
        Neural Network für Momentum Analysis
        """
        return nn.Sequential(
            nn.Linear(20, 64),  # 20 momentum features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Momentum strength
            nn.Sigmoid()
        )
    
    async def generate_signal(self, market_data: Dict) -> Optional[AgentSignal]:
        """
        Trend-basierte Signal Generation
        """
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')
            logger.debug(f"🎯 Trend Analysis für {symbol}...")
            
            # Trend Analysis Features
            trend_features = await self._extract_trend_features(market_data)
            
            if not trend_features:
                return None
            
            # Trend Direction Detection
            trend_direction = await self._detect_trend_direction(trend_features)
            trend_strength = await self._calculate_trend_strength(trend_features)
            momentum_score = await self._analyze_momentum(trend_features)
            
            # Breakout Analysis
            breakout_probability = await self._analyze_breakout_potential(trend_features)
            
            # Decision Logic
            action, confidence = await self._make_trend_decision(
                trend_direction, trend_strength, momentum_score, breakout_probability
            )
            
            # Position Size basierend auf Trend Strength
            position_size = self._calculate_trend_position_size(
                trend_strength, momentum_score, confidence
            )
            
            # Risk Assessment
            risk_assessment = await self._assess_trend_risk(
                trend_features, market_data
            )
            
            # Technical Factors
            technical_factors = {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'momentum_score': momentum_score,
                'breakout_probability': breakout_probability,
                'ma_alignment': trend_features.get('ma_alignment', 0),
                'adx_strength': trend_features.get('adx', 0),
                'macd_signal': trend_features.get('macd_signal', 'neutral')
            }
            
            # Reasoning
            reasoning = self._generate_trend_reasoning(
                action, trend_direction, trend_strength, momentum_score
            )
            
            signal = self._create_agent_signal(
                action=action,
                confidence=confidence,
                position_size=position_size,
                reasoning=reasoning,
                technical_factors=technical_factors,
                risk_assessment=risk_assessment,
                market_data=market_data
            )
            
            logger.debug(f"🎯 Trend Signal: {action} (Confidence: {confidence:.3f}, Strength: {trend_strength:.3f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Trend Specialist Signal Generation Fehler: {e}")
            return None
    
    async def _extract_trend_features(self, market_data: Dict) -> Dict:
        """
        Trend-spezifische Features extrahieren
        """
        try:
            # Price Data
            close_price = market_data.get('close', 0)
            high_price = market_data.get('high', close_price)
            low_price = market_data.get('low', close_price)
            volume = market_data.get('volume', 0)
            
            # Simulierte historische Daten für Demo
            # In Produktion: Echte historische Daten verwenden
            price_history = self._generate_mock_price_history(close_price)
            
            features = {}
            
            # Moving Averages
            features['sma_5'] = np.mean(price_history[-5:])
            features['sma_20'] = np.mean(price_history[-20:])
            features['sma_50'] = np.mean(price_history[-50:]) if len(price_history) >= 50 else close_price
            features['ema_12'] = self._calculate_ema(price_history, 12)
            features['ema_26'] = self._calculate_ema(price_history, 26)
            
            # MA Alignment
            features['ma_alignment'] = self._calculate_ma_alignment(features)
            
            # Price vs MA
            features['price_vs_sma20'] = (close_price - features['sma_20']) / features['sma_20']
            features['price_vs_sma50'] = (close_price - features['sma_50']) / features['sma_50']
            
            # MACD
            features['macd'] = features['ema_12'] - features['ema_26']
            features['macd_signal'] = self._calculate_ema([features['macd']], 9)
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # ADX (Trend Strength)
            features['adx'] = self._calculate_adx(price_history)
            
            # Directional Movement
            features['plus_di'] = self._calculate_plus_di(price_history)
            features['minus_di'] = self._calculate_minus_di(price_history)
            
            # Higher Highs / Lower Lows
            features['higher_highs'] = self._count_higher_highs(price_history)
            features['lower_lows'] = self._count_lower_lows(price_history)
            
            # Volume Trend
            volume_history = [volume] * 20  # Mock volume history
            features['volume_trend'] = self._calculate_volume_trend(volume_history)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Trend Features Extraction Fehler: {e}")
            return {}
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _calculate_ma_alignment(self, features: Dict) -> float:
        """Moving Average Alignment Score"""
        sma_5 = features.get('sma_5', 0)
        sma_20 = features.get('sma_20', 0)
        sma_50 = features.get('sma_50', 0)
        
        if sma_5 > sma_20 > sma_50:
            return 1.0  # Perfect bullish alignment
        elif sma_5 < sma_20 < sma_50:
            return -1.0  # Perfect bearish alignment
        else:
            return 0.0  # Mixed alignment
    
    def _calculate_adx(self, prices: List[float]) -> float:
        """Average Directional Index (simplified)"""
        if len(prices) < 14:
            return 25.0  # Default ADX
        
        # Simplified ADX calculation
        price_changes = np.diff(prices)
        abs_changes = np.abs(price_changes)
        
        if len(abs_changes) > 0:
            adx = np.mean(abs_changes[-14:]) / np.mean(prices[-14:]) * 100
            return min(100, max(0, adx))
        
        return 25.0
    
    def _calculate_plus_di(self, prices: List[float]) -> float:
        """Plus Directional Indicator (simplified)"""
        if len(prices) < 2:
            return 50.0
        
        up_moves = [max(0, prices[i] - prices[i-1]) for i in range(1, len(prices))]
        return np.mean(up_moves[-14:]) if len(up_moves) >= 14 else 50.0
    
    def _calculate_minus_di(self, prices: List[float]) -> float:
        """Minus Directional Indicator (simplified)"""
        if len(prices) < 2:
            return 50.0
        
        down_moves = [max(0, prices[i-1] - prices[i]) for i in range(1, len(prices))]
        return np.mean(down_moves[-14:]) if len(down_moves) >= 14 else 50.0
    
    def _count_higher_highs(self, prices: List[float]) -> int:
        """Count Higher Highs in recent period"""
        if len(prices) < 10:
            return 0
        
        recent_prices = prices[-10:]
        higher_highs = 0
        
        for i in range(1, len(recent_prices)):
            if recent_prices[i] > recent_prices[i-1]:
                higher_highs += 1
        
        return higher_highs
    
    def _count_lower_lows(self, prices: List[float]) -> int:
        """Count Lower Lows in recent period"""
        if len(prices) < 10:
            return 0
        
        recent_prices = prices[-10:]
        lower_lows = 0
        
        for i in range(1, len(recent_prices)):
            if recent_prices[i] < recent_prices[i-1]:
                lower_lows += 1
        
        return lower_lows
    
    def _calculate_volume_trend(self, volume_history: List[float]) -> float:
        """Volume Trend Analysis"""
        if len(volume_history) < 10:
            return 0.0
        
        recent_volume = np.mean(volume_history[-5:])
        older_volume = np.mean(volume_history[-15:-5])
        
        if older_volume > 0:
            return (recent_volume - older_volume) / older_volume
        
        return 0.0
    
    async def _detect_trend_direction(self, features: Dict) -> str:
        """Trend Direction Detection"""
        try:
            # MA Alignment
            ma_alignment = features.get('ma_alignment', 0)
            
            # MACD Signal
            macd = features.get('macd', 0)
            macd_signal = features.get('macd_signal', 0)
            
            # Price vs MA
            price_vs_sma20 = features.get('price_vs_sma20', 0)
            
            # Higher Highs vs Lower Lows
            higher_highs = features.get('higher_highs', 0)
            lower_lows = features.get('lower_lows', 0)
            
            # Scoring System
            bullish_score = 0
            bearish_score = 0
            
            # MA Alignment Score
            if ma_alignment > 0.5:
                bullish_score += 2
            elif ma_alignment < -0.5:
                bearish_score += 2
            
            # MACD Score
            if macd > macd_signal:
                bullish_score += 1
            else:
                bearish_score += 1
            
            # Price vs MA Score
            if price_vs_sma20 > 0.02:  # 2% above MA
                bullish_score += 1
            elif price_vs_sma20 < -0.02:  # 2% below MA
                bearish_score += 1
            
            # Higher Highs / Lower Lows Score
            if higher_highs > lower_lows:
                bullish_score += 1
            elif lower_lows > higher_highs:
                bearish_score += 1
            
            # Final Decision
            if bullish_score > bearish_score + 1:
                return 'bullish'
            elif bearish_score > bullish_score + 1:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"❌ Trend Direction Detection Fehler: {e}")
            return 'neutral'
    
    async def _calculate_trend_strength(self, features: Dict) -> float:
        """Trend Strength Calculation"""
        try:
            # ADX als primärer Trend Strength Indikator
            adx = features.get('adx', 25)
            
            # MA Alignment
            ma_alignment = abs(features.get('ma_alignment', 0))
            
            # MACD Histogram
            macd_histogram = abs(features.get('macd_histogram', 0))
            
            # Combined Strength Score
            strength_score = (
                (adx / 100) * 0.4 +  # ADX weight
                ma_alignment * 0.3 +  # MA alignment weight
                min(macd_histogram * 10, 1.0) * 0.3  # MACD weight
            )
            
            return min(1.0, max(0.0, strength_score))
            
        except Exception as e:
            logger.error(f"❌ Trend Strength Calculation Fehler: {e}")
            return 0.5
    
    async def _analyze_momentum(self, features: Dict) -> float:
        """Momentum Analysis"""
        try:
            # MACD Momentum
            macd = features.get('macd', 0)
            macd_signal = features.get('macd_signal', 0)
            macd_momentum = macd - macd_signal
            
            # Price Momentum
            price_vs_sma20 = features.get('price_vs_sma20', 0)
            
            # Volume Momentum
            volume_trend = features.get('volume_trend', 0)
            
            # Combined Momentum Score
            momentum_score = (
                np.tanh(macd_momentum * 100) * 0.4 +  # MACD momentum
                np.tanh(price_vs_sma20 * 10) * 0.4 +  # Price momentum
                np.tanh(volume_trend * 5) * 0.2       # Volume momentum
            )
            
            # Normalize to 0-1
            return (momentum_score + 1) / 2
            
        except Exception as e:
            logger.error(f"❌ Momentum Analysis Fehler: {e}")
            return 0.5
    
    async def _analyze_breakout_potential(self, features: Dict) -> float:
        """Breakout Potential Analysis"""
        try:
            # Volume Trend (wichtig für Breakouts)
            volume_trend = features.get('volume_trend', 0)
            
            # ADX (Trend Strength)
            adx = features.get('adx', 25)
            
            # MACD Histogram
            macd_histogram = features.get('macd_histogram', 0)
            
            # Breakout Score
            breakout_score = (
                max(0, volume_trend) * 0.4 +  # Volume increase
                min(adx / 100, 1.0) * 0.3 +   # Trend strength
                abs(macd_histogram) * 10 * 0.3  # MACD momentum
            )
            
            return min(1.0, max(0.0, breakout_score))
            
        except Exception as e:
            logger.error(f"❌ Breakout Analysis Fehler: {e}")
            return 0.3
    
    async def _make_trend_decision(self, trend_direction: str, trend_strength: float,
                                 momentum_score: float, breakout_probability: float) -> Tuple[str, float]:
        """
        Trend-basierte Trading Decision
        """
        try:
            # Minimum Thresholds
            min_trend_strength = 0.3
            min_momentum = 0.4
            min_confidence = 0.5
            
            # Base Confidence
            base_confidence = (trend_strength + momentum_score) / 2
            
            # Trend Direction Decision
            if trend_direction == 'bullish' and trend_strength > min_trend_strength:
                action = 'BUY'
                confidence = base_confidence
                
                # Breakout Bonus
                if breakout_probability > 0.6:
                    confidence *= 1.2
                    
            elif trend_direction == 'bearish' and trend_strength > min_trend_strength:
                action = 'SELL'
                confidence = base_confidence
                
                # Breakout Bonus
                if breakout_probability > 0.6:
                    confidence *= 1.2
                    
            else:
                action = 'HOLD'
                confidence = 0.2  # Low confidence für HOLD
            
            # Momentum Adjustment
            if momentum_score > 0.7:
                confidence *= 1.1
            elif momentum_score < 0.3:
                confidence *= 0.8
            
            # Final Confidence Clipping
            confidence = min(1.0, max(0.0, confidence))
            
            # Minimum Confidence Check
            if confidence < min_confidence and action != 'HOLD':
                action = 'HOLD'
                confidence = 0.2
            
            return action, confidence
            
        except Exception as e:
            logger.error(f"❌ Trend Decision Making Fehler: {e}")
            return 'HOLD', 0.0
    
    def _calculate_trend_position_size(self, trend_strength: float, 
                                     momentum_score: float, confidence: float) -> float:
        """
        Position Size basierend auf Trend Characteristics
        """
        try:
            # Base Position Size
            base_size = 0.1  # 10% base
            
            # Trend Strength Multiplier
            strength_multiplier = 1 + (trend_strength - 0.5)  # 0.5 - 1.5x
            
            # Momentum Multiplier
            momentum_multiplier = 1 + (momentum_score - 0.5) * 0.5  # 0.75 - 1.25x
            
            # Confidence Multiplier
            confidence_multiplier = confidence  # 0 - 1x
            
            # Final Position Size
            position_size = (base_size * 
                           strength_multiplier * 
                           momentum_multiplier * 
                           confidence_multiplier)
            
            return min(0.3, max(0.01, position_size))  # 1% - 30% range
            
        except Exception as e:
            logger.error(f"❌ Trend Position Size Calculation Fehler: {e}")
            return 0.1
    
    async def _assess_trend_risk(self, features: Dict, market_data: Dict) -> Dict:
        """
        Trend-spezifische Risk Assessment
        """
        try:
            # Trend Consistency
            ma_alignment = abs(features.get('ma_alignment', 0))
            trend_consistency = ma_alignment
            
            # Volatility Risk
            volatility = market_data.get('volatility', 0.02)
            volatility_risk = min(1.0, volatility / 0.05)  # Normalize to 5% max
            
            # Counter-Trend Risk
            adx = features.get('adx', 25)
            counter_trend_risk = 1 - (adx / 100)  # Lower ADX = higher risk
            
            # Overall Risk Score
            risk_score = (
                (1 - trend_consistency) * 0.4 +  # Trend inconsistency
                volatility_risk * 0.3 +           # Volatility
                counter_trend_risk * 0.3          # Counter-trend potential
            )
            
            risk_level = 'low' if risk_score < 0.3 else 'medium' if risk_score < 0.7 else 'high'
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'trend_consistency': trend_consistency,
                'volatility_risk': volatility_risk,
                'counter_trend_risk': counter_trend_risk
            }
            
        except Exception as e:
            logger.error(f"❌ Trend Risk Assessment Fehler: {e}")
            return {'risk_score': 0.5, 'risk_level': 'medium'}
    
    def _generate_trend_reasoning(self, action: str, trend_direction: str, 
                                trend_strength: float, momentum_score: float) -> str:
        """
        Human-readable Trend Reasoning
        """
        try:
            strength_desc = 'strong' if trend_strength > 0.7 else 'moderate' if trend_strength > 0.4 else 'weak'
            momentum_desc = 'high' if momentum_score > 0.7 else 'moderate' if momentum_score > 0.4 else 'low'
            
            if action == 'BUY':
                return f"Bullish trend detected with {strength_desc} strength and {momentum_desc} momentum. MA alignment supports upward movement."
            elif action == 'SELL':
                return f"Bearish trend identified with {strength_desc} strength and {momentum_desc} momentum. Technical indicators suggest downward pressure."
            else:
                return f"Mixed trend signals. {trend_direction} direction with {strength_desc} strength insufficient for clear signal."
                
        except Exception as e:
            logger.error(f"❌ Trend Reasoning Generation Fehler: {e}")
            return f"Trend analysis: {action} signal with {trend_strength:.2f} strength"


class VolatilityExpertAgent(SpecializedTradingAgent):
    """
    ⚡ Volatility Trading Spezialist
    Experte für Volatility-basierte Strategien und Breakout-Trading
    """
    
    async def initialize(self):
        self.agent_id = f"volatility_expert_{id(self)}"
        self.specialization = AgentSpecialization.VOLATILITY_EXPERT
        self.expertise_areas = [
            'volatility_analysis', 'breakout_detection', 'bollinger_bands',
            'atr_analysis', 'volatility_regimes', 'squeeze_patterns'
        ]
        
        # Volatility-spezifische Parameter
        self.volatility_lookback = 20
        self.breakout_threshold = 2.0  # Standard deviations
        self.squeeze_threshold = 0.015  # 1.5%
        
        logger.success("⚡ Volatility Expert Agent initialisiert")
        return True
    
    async def generate_signal(self, market_data: Dict) -> Optional[AgentSignal]:
        """
        Volatility-basierte Signal Generation
        """
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')
            logger.debug(f"⚡ Volatility Analysis für {symbol}...")
            
            # Volatility Features
            vol_features = await self._extract_volatility_features(market_data)
            
            if not vol_features:
                return None
            
            # Volatility Analysis
            volatility_regime = await self._detect_volatility_regime(vol_features)
            breakout_signal = await self._detect_breakout(vol_features)
            squeeze_signal = await self._detect_squeeze(vol_features)
            bollinger_signal = await self._analyze_bollinger_bands(vol_features)
            
            # Decision Logic
            action, confidence = await self._make_volatility_decision(
                volatility_regime, breakout_signal, squeeze_signal, bollinger_signal
            )
            
            # Position Size
            position_size = self._calculate_volatility_position_size(
                volatility_regime, confidence, vol_features
            )
            
            # Risk Assessment
            risk_assessment = await self._assess_volatility_risk(vol_features, market_data)
            
            # Technical Factors
            technical_factors = {
                'volatility_regime': volatility_regime,
                'breakout_strength': breakout_signal.get('strength', 0),
                'squeeze_detected': squeeze_signal.get('detected', False),
                'bollinger_position': bollinger_signal.get('position', 0),
                'atr_percentile': vol_features.get('atr_percentile', 50),
                'volatility_trend': vol_features.get('volatility_trend', 'stable')
            }
            
            # Reasoning
            reasoning = self._generate_volatility_reasoning(
                action, volatility_regime, breakout_signal, squeeze_signal
            )
            
            signal = self._create_agent_signal(
                action=action,
                confidence=confidence,
                position_size=position_size,
                reasoning=reasoning,
                technical_factors=technical_factors,
                risk_assessment=risk_assessment,
                market_data=market_data
            )
            
            logger.debug(f"⚡ Volatility Signal: {action} (Confidence: {confidence:.3f}, Regime: {volatility_regime})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Volatility Expert Signal Generation Fehler: {e}")
            return None
    
    async def _extract_volatility_features(self, market_data: Dict) -> Dict:
        """Volatility-spezifische Features extrahieren"""
        try:
            close_price = market_data.get('close', 0)
            high_price = market_data.get('high', close_price)
            low_price = market_data.get('low', close_price)
            volume = market_data.get('volume', 0)
            
            price_history = self._generate_mock_price_history(close_price)
            features = {}
            
            # Historical Volatility
            features['historical_volatility'] = self._calculate_historical_volatility(price_history)
            
            # ATR (Average True Range)
            features['atr'] = self._calculate_atr(price_history)
            features['atr_percentile'] = self._calculate_atr_percentile(features['atr'], price_history)
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(price_history)
            features['bb_upper'] = bb_data['upper']
            features['bb_lower'] = bb_data['lower']
            features['bb_middle'] = bb_data['middle']
            features['bb_width'] = bb_data['width']
            features['bb_position'] = (close_price - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
            
            # Volatility Squeeze
            features['squeeze_detected'] = self._detect_volatility_squeeze(bb_data, features['atr'])
            
            # Price Range Analysis
            features['daily_range'] = (high_price - low_price) / close_price
            features['range_percentile'] = self._calculate_range_percentile(features['daily_range'], price_history)
            
            # Volatility Trend
            features['volatility_trend'] = self._calculate_volatility_trend(price_history)
            
            # Breakout Potential
            features['breakout_potential'] = self._calculate_breakout_potential(price_history, volume)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Volatility Features Extraction Fehler: {e}")
            return {}
    
    def _calculate_historical_volatility(self, prices: List[float]) -> float:
        """Historical Volatility Calculation"""
        if len(prices) < 2:
            return 0.02  # Default 2%
        
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_atr(self, prices: List[float], period: int = 14) -> float:
        """Average True Range"""
        if len(prices) < period + 1:
            return prices[-1] * 0.02 if prices else 0  # 2% of price
        
        true_ranges = []
        for i in range(1, len(prices)):
            high_low = abs(prices[i] - prices[i-1])
            true_ranges.append(high_low)
        
        return np.mean(true_ranges[-period:])
    
    def _calculate_atr_percentile(self, current_atr: float, prices: List[float]) -> float:
        """ATR Percentile (0-100)"""
        if len(prices) < 50:
            return 50.0  # Default median
        
        # Calculate ATR for different periods
        atr_history = []
        for i in range(14, len(prices)):
            period_prices = prices[max(0, i-20):i]
            atr = self._calculate_atr(period_prices)
            atr_history.append(atr)
        
        if not atr_history:
            return 50.0
        
        # Calculate percentile
        percentile = (np.sum(np.array(atr_history) <= current_atr) / len(atr_history)) * 100
        return percentile
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict:
        """Bollinger Bands Calculation"""
        if len(prices) < period:
            current_price = prices[-1] if prices else 50000
            return {
                'upper': current_price * 1.02,
                'lower': current_price * 0.98,
                'middle': current_price,
                'width': current_price * 0.04
            }
        
        # Simple Moving Average
        sma = np.mean(prices[-period:])
        
        # Standard Deviation
        std = np.std(prices[-period:])
        
        # Bollinger Bands
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        band_width = (upper_band - lower_band) / sma
        
        return {
            'upper': upper_band,
            'lower': lower_band,
            'middle': sma,
            'width': band_width
        }
    
    def _detect_volatility_squeeze(self, bb_data: Dict, atr: float) -> bool:
        """Volatility Squeeze Detection"""
        try:
            bb_width = bb_data.get('width', 0.04)
            bb_middle = bb_data.get('middle', 50000)
            
            # Normalize ATR relative to price
            normalized_atr = atr / bb_middle if bb_middle > 0 else 0
            
            # Squeeze detected wenn BB width und ATR beide niedrig sind
            squeeze_threshold = 0.02  # 2%
            
            return bb_width < squeeze_threshold and normalized_atr < squeeze_threshold
            
        except Exception as e:
            logger.error(f"❌ Volatility Squeeze Detection Fehler: {e}")
            return False
    
    def _calculate_range_percentile(self, current_range: float, prices: List[float]) -> float:
        """Daily Range Percentile"""
        if len(prices) < 20:
            return 50.0
        
        # Calculate historical ranges (simplified)
        ranges = []
        for i in range(1, len(prices)):
            # Simulate high/low with price movement
            daily_range = abs(prices[i] - prices[i-1]) / prices[i-1]
            ranges.append(daily_range)
        
        if not ranges:
            return 50.0
        
        percentile = (np.sum(np.array(ranges) <= current_range) / len(ranges)) * 100
        return percentile
    
    def _calculate_volatility_trend(self, prices: List[float]) -> str:
        """Volatility Trend Analysis"""
        if len(prices) < 40:
            return 'stable'
        
        # Recent vs Older Volatility
        recent_vol = self._calculate_historical_volatility(prices[-20:])
        older_vol = self._calculate_historical_volatility(prices[-40:-20])
        
        vol_change = (recent_vol - older_vol) / older_vol if older_vol > 0 else 0
        
        if vol_change > 0.2:  # 20% increase
            return 'increasing'
        elif vol_change < -0.2:  # 20% decrease
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_breakout_potential(self, prices: List[float], volume: float) -> float:
        """Breakout Potential Score"""
        if len(prices) < 20:
            return 0.3
        
        # Price consolidation (low volatility)
        recent_vol = self._calculate_historical_volatility(prices[-10:])
        avg_vol = self._calculate_historical_volatility(prices[-30:])
        
        consolidation_score = 1 - (recent_vol / avg_vol) if avg_vol > 0 else 0
        
        # Volume factor (higher volume = higher breakout potential)
        avg_volume = 1000000  # Mock average volume
        volume_factor = min(2.0, volume / avg_volume) if avg_volume > 0 else 1.0
        
        breakout_potential = consolidation_score * volume_factor * 0.5
        return min(1.0, max(0.0, breakout_potential))
    
    async def _detect_volatility_regime(self, features: Dict) -> str:
        """Volatility Regime Detection"""
        try:
            atr_percentile = features.get('atr_percentile', 50)
            historical_vol = features.get('historical_volatility', 0.02)
            volatility_trend = features.get('volatility_trend', 'stable')
            
            # Regime Classification
            if atr_percentile > 80 and historical_vol > 0.05:
                regime = 'high_volatility'
            elif atr_percentile < 20 and historical_vol < 0.015:
                regime = 'low_volatility'
            elif volatility_trend == 'increasing':
                regime = 'expanding_volatility'
            elif volatility_trend == 'decreasing':
                regime = 'contracting_volatility'
            else:
                regime = 'normal_volatility'
            
            return regime
            
        except Exception as e:
            logger.error(f"❌ Volatility Regime Detection Fehler: {e}")
            return 'normal_volatility'
    
    async def _detect_breakout(self, features: Dict) -> Dict:
        """Breakout Signal Detection"""
        try:
            bb_position = features.get('bb_position', 0.5)
            atr_percentile = features.get('atr_percentile', 50)
            breakout_potential = features.get('breakout_potential', 0.3)
            squeeze_detected = features.get('squeeze_detected', False)
            
            breakout_signal = {
                'detected': False,
                'direction': 'neutral',
                'strength': 0.0
            }
            
            # Breakout Conditions
            if bb_position > 0.95:  # Price above upper BB
                breakout_signal['detected'] = True
                breakout_signal['direction'] = 'bullish'
                breakout_signal['strength'] = bb_position
                
                # Squeeze Breakout Bonus
                if squeeze_detected:
                    breakout_signal['strength'] *= 1.3
                    
            elif bb_position < 0.05:  # Price below lower BB
                breakout_signal['detected'] = True
                breakout_signal['direction'] = 'bearish'
                breakout_signal['strength'] = 1 - bb_position
                
                # Squeeze Breakout Bonus
                if squeeze_detected:
                    breakout_signal['strength'] *= 1.3
            
            # High ATR bestätigt Breakout
            if atr_percentile > 70:
                breakout_signal['strength'] *= 1.2
            
            return breakout_signal
            
        except Exception as e:
            logger.error(f"❌ Breakout Detection Fehler: {e}")
            return {'detected': False, 'direction': 'neutral', 'strength': 0.0}
    
    async def _detect_squeeze(self, features: Dict) -> Dict:
        """Volatility Squeeze Detection"""
        try:
            squeeze_detected = features.get('squeeze_detected', False)
            bb_width = features.get('bb_width', 0.04)
            atr_percentile = features.get('atr_percentile', 50)
            
            squeeze_signal = {
                'detected': squeeze_detected,
                'strength': 0.0,
                'breakout_imminent': False
            }
            
            if squeeze_detected:
                # Squeeze Strength basierend auf BB width und ATR
                squeeze_strength = 1 - (bb_width / 0.04)  # Normalized to typical width
                squeeze_strength = max(0, min(1, squeeze_strength))
                
                squeeze_signal['strength'] = squeeze_strength
                
                # Breakout Imminent wenn sehr enge Squeeze
                if bb_width < 0.015 and atr_percentile < 30:
                    squeeze_signal['breakout_imminent'] = True
            
            return squeeze_signal
            
        except Exception as e:
            logger.error(f"❌ Squeeze Detection Fehler: {e}")
            return {'detected': False, 'strength': 0.0, 'breakout_imminent': False}
    
    async def _analyze_bollinger_bands(self, features: Dict) -> Dict:
        """Bollinger Bands Analysis"""
        try:
            bb_position = features.get('bb_position', 0.5)
            bb_width = features.get('bb_width', 0.04)
            
            bollinger_signal = {
                'position': bb_position,
                'signal': 'neutral',
                'reversal_probability': 0.0
            }
            
            # Bollinger Band Signals
            if bb_position > 0.8:  # Near upper band
                bollinger_signal['signal'] = 'overbought'
                bollinger_signal['reversal_probability'] = (bb_position - 0.8) / 0.2
                
            elif bb_position < 0.2:  # Near lower band
                bollinger_signal['signal'] = 'oversold'
                bollinger_signal['reversal_probability'] = (0.2 - bb_position) / 0.2
            
            # Band Width Analysis
            if bb_width < 0.02:  # Tight bands
                bollinger_signal['band_condition'] = 'tight'
            elif bb_width > 0.06:  # Wide bands
                bollinger_signal['band_condition'] = 'wide'
            else:
                bollinger_signal['band_condition'] = 'normal'
            
            return bollinger_signal
            
        except Exception as e:
            logger.error(f"❌ Bollinger Bands Analysis Fehler: {e}")
            return {'position': 0.5, 'signal': 'neutral', 'reversal_probability': 0.0}
    
    async def _make_volatility_decision(self, volatility_regime: str, breakout_signal: Dict,
                                      squeeze_signal: Dict, bollinger_signal: Dict) -> Tuple[str, float]:
        """Volatility-basierte Trading Decision"""
        try:
            action = 'HOLD'
            confidence = 0.0
            
            # Breakout Trading Logic
            if breakout_signal.get('detected', False):
                breakout_direction = breakout_signal.get('direction', 'neutral')
                breakout_strength = breakout_signal.get('strength', 0)
                
                if breakout_direction == 'bullish' and breakout_strength > 0.7:
                    action = 'BUY'
                    confidence = breakout_strength * 0.8
                    
                elif breakout_direction == 'bearish' and breakout_strength > 0.7:
                    action = 'SELL'
                    confidence = breakout_strength * 0.8
            
            # Squeeze Breakout Logic
            elif squeeze_signal.get('breakout_imminent', False):
                squeeze_strength = squeeze_signal.get('strength', 0)
                
                # Warten auf Directional Breakout (konservativ)
                if volatility_regime == 'expanding_volatility':
                    action = 'BUY'  # Slight bullish bias on expansion
                    confidence = squeeze_strength * 0.6
                else:
                    action = 'HOLD'
                    confidence = 0.3
            
            # Mean Reversion auf Bollinger Bands
            elif bollinger_signal.get('signal') == 'oversold':
                reversal_prob = bollinger_signal.get('reversal_probability', 0)
                if reversal_prob > 0.6 and volatility_regime != 'high_volatility':
                    action = 'BUY'
                    confidence = reversal_prob * 0.7
                    
            elif bollinger_signal.get('signal') == 'overbought':
                reversal_prob = bollinger_signal.get('reversal_probability', 0)
                if reversal_prob > 0.6 and volatility_regime != 'high_volatility':
                    action = 'SELL'
                    confidence = reversal_prob * 0.7
            
            # Volatility Regime Adjustments
            if volatility_regime == 'high_volatility':
                confidence *= 0.8  # Reduce confidence in high vol
            elif volatility_regime == 'low_volatility':
                confidence *= 1.1  # Increase confidence in low vol
            
            # Final Confidence Clipping
            confidence = min(1.0, max(0.0, confidence))
            
            return action, confidence
            
        except Exception as e:
            logger.error(f"❌ Volatility Decision Making Fehler: {e}")
            return 'HOLD', 0.0
    
    def _calculate_volatility_position_size(self, volatility_regime: str, 
                                          confidence: float, features: Dict) -> float:
        """Volatility-adjusted Position Size"""
        try:
            base_size = 0.1  # 10% base
            
            # Volatility Regime Adjustments
            regime_multipliers = {
                'low_volatility': 1.3,      # Larger position in low vol
                'normal_volatility': 1.0,   # Standard position
                'high_volatility': 0.6,     # Smaller position in high vol
                'expanding_volatility': 0.8, # Cautious on expansion
                'contracting_volatility': 1.2  # Larger on contraction
            }
            
            regime_multiplier = regime_multipliers.get(volatility_regime, 1.0)
            
            # Confidence Multiplier
            confidence_multiplier = confidence
            
            # ATR Adjustment
            atr_percentile = features.get('atr_percentile', 50)
            if atr_percentile > 80:  # Very high ATR
                atr_adjustment = 0.7
            elif atr_percentile < 20:  # Very low ATR
                atr_adjustment = 1.2
            else:
                atr_adjustment = 1.0
            
            # Final Position Size
            position_size = (base_size * 
                           regime_multiplier * 
                           confidence_multiplier * 
                           atr_adjustment)
            
            return min(0.25, max(0.01, position_size))  # 1% - 25% range
            
        except Exception as e:
            logger.error(f"❌ Volatility Position Size Calculation Fehler: {e}")
            return 0.1
    
    async def _assess_volatility_risk(self, features: Dict, market_data: Dict) -> Dict:
        """Volatility-spezifische Risk Assessment"""
        try:
            # Volatility Risk
            historical_vol = features.get('historical_volatility', 0.02)
            vol_risk = min(1.0, historical_vol / 0.1)  # Normalize to 10% max
            
            # Regime Risk
            volatility_trend = features.get('volatility_trend', 'stable')
            regime_risk = {
                'increasing': 0.7,  # High risk when vol increasing
                'decreasing': 0.3,  # Low risk when vol decreasing
                'stable': 0.5       # Medium risk when stable
            }.get(volatility_trend, 0.5)
            
            # Breakout Risk
            breakout_potential = features.get('breakout_potential', 0.3)
            breakout_risk = breakout_potential  # Higher potential = higher risk
            
            # Overall Risk Score
            risk_score = (
                vol_risk * 0.4 +
                regime_risk * 0.3 +
                breakout_risk * 0.3
            )
            
            risk_level = 'low' if risk_score < 0.4 else 'medium' if risk_score < 0.7 else 'high'
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'volatility_risk': vol_risk,
                'regime_risk': regime_risk,
                'breakout_risk': breakout_risk
            }
            
        except Exception as e:
            logger.error(f"❌ Volatility Risk Assessment Fehler: {e}")
            return {'risk_score': 0.5, 'risk_level': 'medium'}
    
    def _generate_volatility_reasoning(self, action: str, volatility_regime: str,
                                     breakout_signal: Dict, squeeze_signal: Dict) -> str:
        """Human-readable Volatility Reasoning"""
        try:
            if action == 'BUY':
                if breakout_signal.get('detected') and breakout_signal.get('direction') == 'bullish':
                    return f"Bullish breakout detected in {volatility_regime} environment. Breakout strength: {breakout_signal.get('strength', 0):.2f}"
                elif squeeze_signal.get('breakout_imminent'):
                    return f"Volatility squeeze with imminent breakout potential. Regime: {volatility_regime}"
                else:
                    return f"Oversold condition on Bollinger Bands with mean reversion potential. Volatility: {volatility_regime}"
                    
            elif action == 'SELL':
                if breakout_signal.get('detected') and breakout_signal.get('direction') == 'bearish':
                    return f"Bearish breakout detected in {volatility_regime} environment. Breakout strength: {breakout_signal.get('strength', 0):.2f}"
                else:
                    return f"Overbought condition on Bollinger Bands with mean reversion potential. Volatility: {volatility_regime}"
            else:
                return f"Mixed volatility signals in {volatility_regime} regime. Awaiting clearer directional movement."
                
        except Exception as e:
            logger.error(f"❌ Volatility Reasoning Generation Fehler: {e}")
            return f"Volatility analysis: {action} signal in {volatility_regime} regime"


class SentimentMasterAgent(SpecializedTradingAgent):
    """
    📊 Market Sentiment Master
    Experte für Sentiment-Analyse und Market Psychology
    """
    
    async def initialize(self):
        self.agent_id = f"sentiment_master_{id(self)}"
        self.specialization = AgentSpecialization.SENTIMENT_MASTER
        self.expertise_areas = [
            'news_sentiment', 'social_media_sentiment', 'fear_greed_index',
            'options_sentiment', 'institutional_sentiment', 'market_psychology'
        ]
        
        # Sentiment-spezifische Parameter
        self.sentiment_window = 24  # Hours
        self.fear_greed_threshold = 0.3  # 30%
        self.news_impact_decay = 0.8  # Hourly decay
        
        logger.success("📊 Sentiment Master Agent initialisiert")
        return True
    
    async def generate_signal(self, market_data: Dict) -> Optional[AgentSignal]:
        """
        Sentiment-basierte Signal Generation
        """
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')
            logger.debug(f"📊 Sentiment Analysis für {symbol}...")
            
            # Sentiment Features
            sentiment_features = await self._extract_sentiment_features(market_data)
            
            if not sentiment_features:
                return None
            
            # Sentiment Analysis
            news_sentiment = await self._analyze_news_sentiment(sentiment_features)
            social_sentiment = await self._analyze_social_sentiment(sentiment_features)
            fear_greed = await self._analyze_fear_greed_index(sentiment_features)
            options_sentiment = await self._analyze_options_sentiment(sentiment_features)
            institutional_sentiment = await self._analyze_institutional_sentiment(sentiment_features)
            
            # Overall Sentiment Score
            overall_sentiment = await self._calculate_overall_sentiment(
                news_sentiment, social_sentiment, fear_greed, options_sentiment, institutional_sentiment
            )
            
            # Decision Logic
            action, confidence = await self._make_sentiment_decision(
                overall_sentiment, news_sentiment, fear_greed
            )
            
            # Position Size
            position_size = self._calculate_sentiment_position_size(
                overall_sentiment, confidence, sentiment_features
            )
            
            # Risk Assessment
            risk_assessment = await self._assess_sentiment_risk(sentiment_features, market_data)
            
            # Technical Factors
            technical_factors = {
                'overall_sentiment': overall_sentiment.get('score', 0),
                'news_sentiment': news_sentiment.get('score', 0),
                'social_sentiment': social_sentiment.get('score', 0),
                'fear_greed_index': fear_greed.get('index', 50),
                'options_sentiment': options_sentiment.get('score', 0),
                'institutional_sentiment': institutional_sentiment.get('score', 0),
                'sentiment_momentum': sentiment_features.get('sentiment_momentum', 0)
            }
            
            # Reasoning
            reasoning = self._generate_sentiment_reasoning(
                action, overall_sentiment, news_sentiment, fear_greed
            )
            
            signal = self._create_agent_signal(
                action=action,
                confidence=confidence,
                position_size=position_size,
                reasoning=reasoning,
                technical_factors=technical_factors,
                risk_assessment=risk_assessment,
                market_data=market_data
            )
            
            logger.debug(f"📊 Sentiment Signal: {action} (Confidence: {confidence:.3f}, Overall: {overall_sentiment.get('score', 0):.3f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Sentiment Master Signal Generation Fehler: {e}")
            return None
    
    async def _extract_sentiment_features(self, market_data: Dict) -> Dict:
        """Sentiment-spezifische Features extrahieren"""
        try:
            close_price = market_data.get('close', 0)
            volume = market_data.get('volume', 0)
            
            features = {}
            
            # Mock News Sentiment (in real implementation: News API)
            features['news_sentiment_score'] = self._generate_mock_news_sentiment()
            features['news_volume'] = self._generate_mock_news_volume()
            features['news_impact_score'] = self._calculate_news_impact(features['news_sentiment_score'], features['news_volume'])
            
            # Mock Social Media Sentiment (in real implementation: Twitter/Reddit API)
            features['social_sentiment_score'] = self._generate_mock_social_sentiment()
            features['social_volume'] = self._generate_mock_social_volume()
            features['social_momentum'] = self._calculate_social_momentum(features['social_sentiment_score'])
            
            # Fear & Greed Index (Mock)
            features['fear_greed_index'] = self._generate_mock_fear_greed_index()
            features['fear_greed_trend'] = self._calculate_fear_greed_trend(features['fear_greed_index'])
            
            # Options Sentiment (Mock)
            features['put_call_ratio'] = self._generate_mock_put_call_ratio()
            features['options_volume'] = self._generate_mock_options_volume()
            features['options_sentiment'] = self._calculate_options_sentiment(features['put_call_ratio'])
            
            # Institutional Sentiment (Mock)
            features['institutional_flow'] = self._generate_mock_institutional_flow()
            features['whale_activity'] = self._generate_mock_whale_activity()
            
            # Sentiment Momentum
            features['sentiment_momentum'] = self._calculate_sentiment_momentum(features)
            
            # Market Psychology Indicators
            features['market_psychology'] = self._analyze_market_psychology(features)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Sentiment Features Extraction Fehler: {e}")
            return {}
    
    def _generate_mock_news_sentiment(self) -> float:
        """Mock News Sentiment Score (-1 to 1)"""
        # Simulate news sentiment with some randomness
        base_sentiment = np.random.normal(0, 0.3)  # Neutral with some variation
        return max(-1, min(1, base_sentiment))
    
    def _generate_mock_news_volume(self) -> float:
        """Mock News Volume (0 to 1)"""
        return np.random.uniform(0.2, 0.8)
    
    def _calculate_news_impact(self, sentiment: float, volume: float) -> float:
        """News Impact Score"""
        return abs(sentiment) * volume
    
    def _generate_mock_social_sentiment(self) -> float:
        """Mock Social Media Sentiment Score (-1 to 1)"""
        base_sentiment = np.random.normal(0, 0.4)  # More volatile than news
        return max(-1, min(1, base_sentiment))
    
    def _generate_mock_social_volume(self) -> float:
        """Mock Social Media Volume (0 to 1)"""
        return np.random.uniform(0.3, 0.9)
    
    def _calculate_social_momentum(self, sentiment: float) -> float:
        """Social Sentiment Momentum"""
        # Simulate momentum based on sentiment strength
        return sentiment * np.random.uniform(0.5, 1.5)
    
    def _generate_mock_fear_greed_index(self) -> float:
        """Mock Fear & Greed Index (0 to 100)"""
        return np.random.uniform(20, 80)  # Avoid extreme values
    
    def _calculate_fear_greed_trend(self, current_index: float) -> str:
        """Fear & Greed Trend Analysis"""
        if current_index < 25:
            return 'extreme_fear'
        elif current_index < 45:
            return 'fear'
        elif current_index < 55:
            return 'neutral'
        elif current_index < 75:
            return 'greed'
        else:
            return 'extreme_greed'
    
    def _generate_mock_put_call_ratio(self) -> float:
        """Mock Put/Call Ratio"""
        return np.random.uniform(0.5, 1.5)  # Typical range
    
    def _generate_mock_options_volume(self) -> float:
        """Mock Options Volume"""
        return np.random.uniform(0.4, 0.8)
    
    def _calculate_options_sentiment(self, put_call_ratio: float) -> float:
        """Options Sentiment Score (-1 to 1)"""
        # Lower put/call ratio = more bullish
        if put_call_ratio < 0.7:
            return 0.5  # Bullish
        elif put_call_ratio > 1.3:
            return -0.5  # Bearish
        else:
            return 0  # Neutral
    
    def _generate_mock_institutional_flow(self) -> float:
        """Mock Institutional Flow (-1 to 1)"""
        return np.random.normal(0, 0.3)
    
    def _generate_mock_whale_activity(self) -> float:
        """Mock Whale Activity (0 to 1)"""
        return np.random.uniform(0.1, 0.6)
    
    def _calculate_sentiment_momentum(self, features: Dict) -> float:
        """Overall Sentiment Momentum"""
        news_momentum = features.get('news_impact_score', 0)
        social_momentum = features.get('social_momentum', 0)
        
        # Weighted average
        momentum = (news_momentum * 0.4 + social_momentum * 0.6)
        return max(-1, min(1, momentum))
    
    def _analyze_market_psychology(self, features: Dict) -> str:
        """Market Psychology Analysis"""
        fear_greed = features.get('fear_greed_index', 50)
        social_sentiment = features.get('social_sentiment_score', 0)
        
        if fear_greed < 30 and social_sentiment < -0.3:
            return 'panic'
        elif fear_greed > 70 and social_sentiment > 0.3:
            return 'euphoria'
        elif fear_greed < 40:
            return 'fear'
        elif fear_greed > 60:
            return 'greed'
        else:
            return 'neutral'
    
    async def _analyze_news_sentiment(self, features: Dict) -> Dict:
        """News Sentiment Analysis"""
        try:
            sentiment_score = features.get('news_sentiment_score', 0)
            news_volume = features.get('news_volume', 0)
            impact_score = features.get('news_impact_score', 0)
            
            # Sentiment Classification
            if sentiment_score > 0.3:
                sentiment_class = 'positive'
            elif sentiment_score < -0.3:
                sentiment_class = 'negative'
            else:
                sentiment_class = 'neutral'
            
            # Impact Level
            if impact_score > 0.6:
                impact_level = 'high'
            elif impact_score > 0.3:
                impact_level = 'medium'
            else:
                impact_level = 'low'
            
            return {
                'score': sentiment_score,
                'class': sentiment_class,
                'volume': news_volume,
                'impact_score': impact_score,
                'impact_level': impact_level
            }
            
        except Exception as e:
            logger.error(f"❌ News Sentiment Analysis Fehler: {e}")
            return {'score': 0, 'class': 'neutral', 'impact_level': 'low'}
    
    async def _analyze_social_sentiment(self, features: Dict) -> Dict:
        """Social Media Sentiment Analysis"""
        try:
            sentiment_score = features.get('social_sentiment_score', 0)
            social_volume = features.get('social_volume', 0)
            momentum = features.get('social_momentum', 0)
            
            # Sentiment Classification
            if sentiment_score > 0.2:
                sentiment_class = 'bullish'
            elif sentiment_score < -0.2:
                sentiment_class = 'bearish'
            else:
                sentiment_class = 'neutral'
            
            # Momentum Classification
            if momentum > 0.3:
                momentum_class = 'strong_positive'
            elif momentum < -0.3:
                momentum_class = 'strong_negative'
            else:
                momentum_class = 'weak'
            
            return {
                'score': sentiment_score,
                'class': sentiment_class,
                'volume': social_volume,
                'momentum': momentum,
                'momentum_class': momentum_class
            }
            
        except Exception as e:
            logger.error(f"❌ Social Sentiment Analysis Fehler: {e}")
            return {'score': 0, 'class': 'neutral', 'momentum_class': 'weak'}
    
    async def _analyze_fear_greed_index(self, features: Dict) -> Dict:
        """Fear & Greed Index Analysis"""
        try:
            index = features.get('fear_greed_index', 50)
            trend = features.get('fear_greed_trend', 'neutral')
            
            # Contrarian Signal Strength
            if index < 20:
                contrarian_signal = 'strong_buy'  # Extreme fear = buy opportunity
                signal_strength = 0.8
            elif index < 40:
                contrarian_signal = 'buy'
                signal_strength = 0.6
            elif index > 80:
                contrarian_signal = 'strong_sell'  # Extreme greed = sell signal
                signal_strength = 0.8
            elif index > 60:
                contrarian_signal = 'sell'
                signal_strength = 0.6
            else:
                contrarian_signal = 'neutral'
                signal_strength = 0.2
            
            return {
                'index': index,
                'trend': trend,
                'contrarian_signal': contrarian_signal,
                'signal_strength': signal_strength
            }
            
        except Exception as e:
            logger.error(f"❌ Fear & Greed Analysis Fehler: {e}")
            return {'index': 50, 'trend': 'neutral', 'contrarian_signal': 'neutral', 'signal_strength': 0.2}
    
    async def _analyze_options_sentiment(self, features: Dict) -> Dict:
        """Options Sentiment Analysis"""
        try:
            put_call_ratio = features.get('put_call_ratio', 1.0)
            options_volume = features.get('options_volume', 0.5)
            sentiment_score = features.get('options_sentiment', 0)
            
            # Options Sentiment Classification
            if put_call_ratio < 0.6:
                sentiment_class = 'very_bullish'
            elif put_call_ratio < 0.8:
                sentiment_class = 'bullish'
            elif put_call_ratio > 1.4:
                sentiment_class = 'very_bearish'
            elif put_call_ratio > 1.2:
                sentiment_class = 'bearish'
            else:
                sentiment_class = 'neutral'
            
            return {
                'score': sentiment_score,
                'class': sentiment_class,
                'put_call_ratio': put_call_ratio,
                'volume': options_volume
            }
            
        except Exception as e:
            logger.error(f"❌ Options Sentiment Analysis Fehler: {e}")
            return {'score': 0, 'class': 'neutral', 'put_call_ratio': 1.0}
    
    async def _analyze_institutional_sentiment(self, features: Dict) -> Dict:
        """Institutional Sentiment Analysis"""
        try:
            institutional_flow = features.get('institutional_flow', 0)
            whale_activity = features.get('whale_activity', 0)
            
            # Institutional Sentiment Score
            sentiment_score = institutional_flow * 0.7 + (whale_activity - 0.3) * 0.3
            
            # Classification
            if sentiment_score > 0.3:
                sentiment_class = 'institutional_bullish'
            elif sentiment_score < -0.3:
                sentiment_class = 'institutional_bearish'
            else:
                sentiment_class = 'institutional_neutral'
            
            return {
                'score': sentiment_score,
                'class': sentiment_class,
                'institutional_flow': institutional_flow,
                'whale_activity': whale_activity
            }
            
        except Exception as e:
            logger.error(f"❌ Institutional Sentiment Analysis Fehler: {e}")
            return {'score': 0, 'class': 'institutional_neutral'}
    
    async def _calculate_overall_sentiment(self, news_sentiment: Dict, social_sentiment: Dict,
                                         fear_greed: Dict, options_sentiment: Dict,
                                         institutional_sentiment: Dict) -> Dict:
        """Overall Sentiment Score Calculation"""
        try:
            # Weighted Sentiment Components
            news_weight = 0.25
            social_weight = 0.20
            fear_greed_weight = 0.25
            options_weight = 0.15
            institutional_weight = 0.15
            
            # Convert Fear & Greed to sentiment score
            fear_greed_score = (fear_greed.get('index', 50) - 50) / 50  # Normalize to -1 to 1
            
            # Weighted Average
            overall_score = (
                news_sentiment.get('score', 0) * news_weight +
                social_sentiment.get('score', 0) * social_weight +
                fear_greed_score * fear_greed_weight +
                options_sentiment.get('score', 0) * options_weight +
                institutional_sentiment.get('score', 0) * institutional_weight
            )
            
            # Overall Classification
            if overall_score > 0.3:
                overall_class = 'bullish'
            elif overall_score < -0.3:
                overall_class = 'bearish'
            else:
                overall_class = 'neutral'
            
            # Confidence based on agreement between sources
            sentiment_scores = [
                news_sentiment.get('score', 0),
                social_sentiment.get('score', 0),
                fear_greed_score,
                options_sentiment.get('score', 0),
                institutional_sentiment.get('score', 0)
            ]
            
            # Calculate agreement (low std = high agreement)
            agreement = 1 - (np.std(sentiment_scores) / 2)  # Normalize
            
            return {
                'score': overall_score,
                'class': overall_class,
                'agreement': agreement,
                'components': {
                    'news': news_sentiment.get('score', 0),
                    'social': social_sentiment.get('score', 0),
                    'fear_greed': fear_greed_score,
                    'options': options_sentiment.get('score', 0),
                    'institutional': institutional_sentiment.get('score', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Overall Sentiment Calculation Fehler: {e}")
            return {'score': 0, 'class': 'neutral', 'agreement': 0.5}
    
    async def _make_sentiment_decision(self, overall_sentiment: Dict, 
                                     news_sentiment: Dict, fear_greed: Dict) -> Tuple[str, float]:
        """Sentiment-basierte Trading Decision"""
        try:
            action = 'HOLD'
            confidence = 0.0
            
            overall_score = overall_sentiment.get('score', 0)
            agreement = overall_sentiment.get('agreement', 0.5)
            fear_greed_signal = fear_greed.get('contrarian_signal', 'neutral')
            
            # Strong Sentiment Signals
            if overall_score > 0.4 and agreement > 0.6:
                action = 'BUY'
                confidence = min(0.8, overall_score * agreement)
                
            elif overall_score < -0.4 and agreement > 0.6:
                action = 'SELL'
                confidence = min(0.8, abs(overall_score) * agreement)
            
            # Fear & Greed Contrarian Signals
            elif fear_greed_signal == 'strong_buy' and overall_score > -0.2:
                action = 'BUY'
                confidence = fear_greed.get('signal_strength', 0.5)
                
            elif fear_greed_signal == 'strong_sell' and overall_score < 0.2:
                action = 'SELL'
                confidence = fear_greed.get('signal_strength', 0.5)
            
            # Moderate Sentiment Signals
            elif overall_score > 0.2 and agreement > 0.4:
                action = 'BUY'
                confidence = overall_score * agreement * 0.7
                
            elif overall_score < -0.2 and agreement > 0.4:
                action = 'SELL'
                confidence = abs(overall_score) * agreement * 0.7
            
            # News Impact Boost
            if news_sentiment.get('impact_level') == 'high':
                confidence *= 1.2
            
            # Final Confidence Clipping
            confidence = min(1.0, max(0.0, confidence))
            
            return action, confidence
            
        except Exception as e:
            logger.error(f"❌ Sentiment Decision Making Fehler: {e}")
            return 'HOLD', 0.0
    
    def _calculate_sentiment_position_size(self, overall_sentiment: Dict, 
                                         confidence: float, features: Dict) -> float:
        """Sentiment-adjusted Position Size"""
        try:
            base_size = 0.1  # 10% base
            
            # Sentiment Strength Multiplier
            sentiment_score = abs(overall_sentiment.get('score', 0))
            sentiment_multiplier = 1 + sentiment_score  # 1.0 - 2.0 range
            
            # Agreement Multiplier
            agreement = overall_sentiment.get('agreement', 0.5)
            agreement_multiplier = 0.5 + agreement  # 0.5 - 1.5 range
            
            # News Impact Multiplier
            news_impact = features.get('news_impact_score', 0)
            news_multiplier = 1 + news_impact * 0.5  # Up to 1.5x
            
            # Fear & Greed Extremes
            fear_greed_index = features.get('fear_greed_index', 50)
            if fear_greed_index < 25 or fear_greed_index > 75:
                fear_greed_multiplier = 1.2  # Increase size at extremes
            else:
                fear_greed_multiplier = 1.0
            
            # Final Position Size
            position_size = (base_size * 
                           sentiment_multiplier * 
                           agreement_multiplier * 
                           news_multiplier * 
                           fear_greed_multiplier * 
                           confidence)
            
            return min(0.20, max(0.01, position_size))  # 1% - 20% range
            
        except Exception as e:
            logger.error(f"❌ Sentiment Position Size Calculation Fehler: {e}")
            return 0.1
    
    async def _assess_sentiment_risk(self, features: Dict, market_data: Dict) -> Dict:
        """Sentiment-spezifische Risk Assessment"""
        try:
            # Sentiment Volatility Risk
            sentiment_momentum = features.get('sentiment_momentum', 0)
            volatility_risk = abs(sentiment_momentum)
            
            # News Impact Risk
            news_impact = features.get('news_impact_score', 0)
            news_risk = news_impact  # High impact = high risk
            
            # Market Psychology Risk
            market_psychology = features.get('market_psychology', 'neutral')
            psychology_risk = {
                'panic': 0.9,
                'fear': 0.6,
                'neutral': 0.3,
                'greed': 0.6,
                'euphoria': 0.9
            }.get(market_psychology, 0.3)
            
            # Fear & Greed Extreme Risk
            fear_greed_index = features.get('fear_greed_index', 50)
            if fear_greed_index < 20 or fear_greed_index > 80:
                extreme_risk = 0.8
            else:
                extreme_risk = 0.2
            
            # Overall Risk Score
            risk_score = (
                volatility_risk * 0.3 +
                news_risk * 0.25 +
                psychology_risk * 0.25 +
                extreme_risk * 0.2
            )
            
            risk_level = 'low' if risk_score < 0.4 else 'medium' if risk_score < 0.7 else 'high'
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'volatility_risk': volatility_risk,
                'news_risk': news_risk,
                'psychology_risk': psychology_risk,
                'extreme_risk': extreme_risk
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment Risk Assessment Fehler: {e}")
            return {'risk_score': 0.5, 'risk_level': 'medium'}
    
    def _generate_sentiment_reasoning(self, action: str, overall_sentiment: Dict,
                                    news_sentiment: Dict, fear_greed: Dict) -> str:
        """Human-readable Sentiment Reasoning"""
        try:
            overall_score = overall_sentiment.get('score', 0)
            overall_class = overall_sentiment.get('class', 'neutral')
            agreement = overall_sentiment.get('agreement', 0.5)
            fear_greed_signal = fear_greed.get('contrarian_signal', 'neutral')
            
            if action == 'BUY':
                if fear_greed_signal in ['strong_buy', 'buy']:
                    return f"Contrarian buy signal from extreme fear (F&G: {fear_greed.get('index', 50):.0f}). Market oversold."
                else:
                    return f"Bullish sentiment consensus ({overall_class}) with {agreement:.1%} agreement. Overall score: {overall_score:.2f}"
                    
            elif action == 'SELL':
                if fear_greed_signal in ['strong_sell', 'sell']:
                    return f"Contrarian sell signal from extreme greed (F&G: {fear_greed.get('index', 50):.0f}). Market overbought."
                else:
                    return f"Bearish sentiment consensus ({overall_class}) with {agreement:.1%} agreement. Overall score: {overall_score:.2f}"
            else:
                return f"Mixed sentiment signals ({overall_class}). Low agreement ({agreement:.1%}) suggests waiting for clearer direction."
                
        except Exception as e:
            logger.error(f"❌ Sentiment Reasoning Generation Fehler: {e}")
            return f"Sentiment analysis: {action} signal based on market psychology"


class ArbitrageHunterAgent(SpecializedTradingAgent):
    """Arbitrage Opportunities Hunter"""
    
    async def generate_signal(self, market_data: Dict) -> Optional[AgentSignal]:
        # Placeholder Implementation
        return AgentSignal(
            agent_id=f"arbitrage_hunter_{id(self)}",
            specialization=AgentSpecialization.ARBITRAGE_HUNTER,
            action="HOLD",
            confidence=0.5,
            position_size=0.1,
            reasoning="Arbitrage analysis placeholder",
            technical_factors={},
            risk_assessment={},
            timestamp=datetime.now(),
            priority="medium",
            expected_duration="medium",
            market_conditions={}
        )


class RiskCommanderAgent(SpecializedTradingAgent):
    """
    🛡️ Risk Management Commander
    Experte für Risikomanagement und Portfolio-Schutz
    """
    
    async def initialize(self):
        self.agent_id = f"risk_commander_{id(self)}"
        self.specialization = AgentSpecialization.RISK_COMMANDER
        self.expertise_areas = [
            'portfolio_risk', 'position_sizing', 'correlation_risk',
            'drawdown_control', 'volatility_risk', 'systemic_risk'
        ]
        
        # Risk-spezifische Parameter
        self.max_portfolio_risk = 0.02  # 2% max risk per trade
        self.max_drawdown = 0.10  # 10% max drawdown
        self.correlation_threshold = 0.7  # 70% correlation limit
        self.var_confidence = 0.95  # 95% VaR confidence
        
        logger.success("🛡️ Risk Commander Agent initialisiert")
        return True
    
    async def generate_signal(self, market_data: Dict) -> Optional[AgentSignal]:
        """
        Risk-basierte Signal Generation
        """
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')
            logger.debug(f"🛡️ Risk Analysis für {symbol}...")
            
            # Risk Features
            risk_features = await self._extract_risk_features(market_data)
            
            if not risk_features:
                return None
            
            # Risk Analysis
            portfolio_risk = await self._analyze_portfolio_risk(risk_features)
            position_risk = await self._analyze_position_risk(risk_features)
            market_risk = await self._analyze_market_risk(risk_features)
            systemic_risk = await self._analyze_systemic_risk(risk_features)
            
            # Overall Risk Assessment
            overall_risk = await self._calculate_overall_risk(
                portfolio_risk, position_risk, market_risk, systemic_risk
            )
            
            # Risk-based Decision Logic
            action, confidence = await self._make_risk_decision(
                overall_risk, portfolio_risk, market_risk
            )
            
            # Risk-adjusted Position Size
            position_size = self._calculate_risk_adjusted_position_size(
                overall_risk, confidence, risk_features
            )
            
            # Risk Assessment
            risk_assessment = await self._assess_comprehensive_risk(risk_features, market_data)
            
            # Technical Factors
            technical_factors = {
                'overall_risk_score': overall_risk.get('score', 0.5),
                'portfolio_risk': portfolio_risk.get('score', 0.5),
                'position_risk': position_risk.get('score', 0.5),
                'market_risk': market_risk.get('score', 0.5),
                'var_estimate': risk_features.get('var_estimate', 0),
                'sharpe_ratio': risk_features.get('sharpe_ratio', 0)
            }
            
            # Reasoning
            reasoning = self._generate_risk_reasoning(
                action, overall_risk, portfolio_risk, market_risk
            )
            
            signal = self._create_agent_signal(
                action=action,
                confidence=confidence,
                position_size=position_size,
                reasoning=reasoning,
                technical_factors=technical_factors,
                risk_assessment=risk_assessment,
                market_data=market_data
            )
            
            logger.debug(f"🛡️ Risk Signal: {action} (Confidence: {confidence:.3f}, Risk: {overall_risk.get('level', 'medium')})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Risk Commander Signal Generation Fehler: {e}")
            return None
    
    async def _extract_risk_features(self, market_data: Dict) -> Dict:
        """Risk-spezifische Features extrahieren"""
        try:
            close_price = market_data.get('close', 0)
            volume = market_data.get('volume', 0)
            
            price_history = self._generate_mock_price_history(close_price)
            features = {}
            
            # Volatility Measures
            features['historical_volatility'] = self._calculate_historical_volatility(price_history)
            features['realized_volatility'] = self._calculate_realized_volatility(price_history)
            features['volatility_of_volatility'] = self._calculate_vol_of_vol(price_history)
            
            # Value at Risk (VaR)
            features['var_estimate'] = self._calculate_var(price_history, self.var_confidence)
            features['expected_shortfall'] = self._calculate_expected_shortfall(price_history)
            
            # Drawdown Analysis
            features['max_drawdown'] = self._calculate_max_drawdown(price_history)
            features['current_drawdown'] = self._calculate_current_drawdown(price_history)
            features['drawdown_duration'] = self._calculate_drawdown_duration(price_history)
            
            # Risk-Reward Metrics
            features['sharpe_ratio'] = self._calculate_sharpe_ratio(price_history)
            features['sortino_ratio'] = self._calculate_sortino_ratio(price_history)
            features['calmar_ratio'] = self._calculate_calmar_ratio(price_history, features['max_drawdown'])
            
            # Correlation Risk
            features['market_correlation'] = self._calculate_market_correlation(price_history)
            features['correlation_risk'] = self._assess_correlation_risk(features['market_correlation'])
            
            # Tail Risk
            features['skewness'] = self._calculate_skewness(price_history)
            features['kurtosis'] = self._calculate_kurtosis(price_history)
            features['tail_risk'] = self._calculate_tail_risk(features['skewness'], features['kurtosis'])
            
            # Liquidity Risk
            features['liquidity_risk'] = self._assess_liquidity_risk(volume, close_price)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Risk Features Extraction Fehler: {e}")
            return {}
    
    def _calculate_historical_volatility(self, prices: List[float], window: int = 30) -> float:
        """Historical Volatility"""
        if len(prices) < 2:
            return 0.02
        
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns[-window:]) * np.sqrt(252) if len(returns) >= window else np.std(returns) * np.sqrt(252)
    
    def _calculate_realized_volatility(self, prices: List[float]) -> float:
        """Realized Volatility (Yang-Zhang estimator simplified)"""
        if len(prices) < 2:
            return 0.02
        
        returns = np.diff(prices) / prices[:-1] 
        return np.sqrt(np.sum(returns**2)) * np.sqrt(252)
    
    def _calculate_vol_of_vol(self, prices: List[float], window: int = 20) -> float:
        """Volatility of Volatility"""
        if len(prices) < window * 2:
            return 0.5
        
        vol_series = []
        for i in range(window, len(prices)):
            period_prices = prices[i-window:i]
            vol = self._calculate_historical_volatility(period_prices, window)
            vol_series.append(vol)
        
        return np.std(vol_series) if vol_series else 0.5
    
    def _calculate_var(self, prices: List[float], confidence: float = 0.95) -> float:
        """Value at Risk"""
        if len(prices) < 2:
            return 0.02
        
        returns = np.diff(prices) / prices[:-1]
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_expected_shortfall(self, prices: List[float], confidence: float = 0.95) -> float:
        """Expected Shortfall (Conditional VaR)"""
        if len(prices) < 2:
            return 0.03
        
        returns = np.diff(prices) / prices[:-1]
        var = self._calculate_var(prices, confidence)
        tail_returns = returns[returns <= var]
        
        return np.mean(tail_returns) if len(tail_returns) > 0 else var
    
    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """Maximum Drawdown"""
        if len(prices) < 2:
            return 0.0
        
        peak = prices[0]
        max_dd = 0.0
        
        for price in prices[1:]:
            if price > peak:
                peak = price
            
            drawdown = (peak - price) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_current_drawdown(self, prices: List[float]) -> float:
        """Current Drawdown"""
        if len(prices) < 2:
            return 0.0
        
        peak = max(prices)
        current_price = prices[-1]
        
        return (peak - current_price) / peak if peak > 0 else 0.0
    
    def _calculate_drawdown_duration(self, prices: List[float]) -> int:
        """Drawdown Duration (simplified)"""
        if len(prices) < 2:
            return 0
        
        peak_idx = np.argmax(prices)
        return len(prices) - peak_idx - 1
    
    def _calculate_sharpe_ratio(self, prices: List[float], risk_free_rate: float = 0.02) -> float:
        """Sharpe Ratio"""
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        excess_returns = np.mean(returns) * 252 - risk_free_rate
        volatility = np.std(returns) * np.sqrt(252)
        
        return excess_returns / volatility if volatility > 0 else 0.0
    
    def _calculate_sortino_ratio(self, prices: List[float], risk_free_rate: float = 0.02) -> float:
        """Sortino Ratio"""
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        excess_returns = np.mean(returns) * 252 - risk_free_rate
        
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.01
        
        return excess_returns / downside_deviation if downside_deviation > 0 else 0.0
    
    def _calculate_calmar_ratio(self, prices: List[float], max_drawdown: float) -> float:
        """Calmar Ratio"""
        if len(prices) < 2 or max_drawdown == 0:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        annual_return = np.mean(returns) * 252
        
        return annual_return / max_drawdown
    
    def _calculate_market_correlation(self, prices: List[float]) -> float:
        """Market Correlation (simplified mock)"""
        # In real implementation: correlate with market index
        return np.random.uniform(0.3, 0.8)
    
    def _assess_correlation_risk(self, correlation: float) -> float:
        """Correlation Risk Assessment"""
        if correlation > self.correlation_threshold:
            return 0.8  # High correlation risk
        elif correlation > 0.5:
            return 0.5  # Medium correlation risk
        else:
            return 0.2  # Low correlation risk
    
    def _calculate_skewness(self, prices: List[float]) -> float:
        """Skewness of Returns"""
        if len(prices) < 3:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, prices: List[float]) -> float:
        """Kurtosis of Returns"""
        if len(prices) < 4:
            return 3.0  # Normal distribution kurtosis
        
        returns = np.diff(prices) / prices[:-1]
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 3.0
        
        kurtosis = np.mean(((returns - mean_return) / std_return) ** 4)
        return kurtosis
    
    def _calculate_tail_risk(self, skewness: float, kurtosis: float) -> float:
        """Tail Risk Score"""
        # Negative skewness = left tail risk
        skew_risk = abs(skewness) if skewness < 0 else 0
        
        # Excess kurtosis = fat tails
        excess_kurtosis = max(0, kurtosis - 3)
        
        tail_risk = (skew_risk * 0.6 + excess_kurtosis * 0.4) / 3
        return min(1.0, tail_risk)
    
    def _assess_liquidity_risk(self, volume: float, price: float) -> float:
        """Liquidity Risk Assessment"""
        # Mock liquidity assessment
        avg_volume = 1000000  # Mock average volume
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        if volume_ratio < 0.3:
            return 0.8  # High liquidity risk
        elif volume_ratio < 0.7:
            return 0.5  # Medium liquidity risk
        else:
            return 0.2  # Low liquidity risk
    
    async def _analyze_portfolio_risk(self, features: Dict) -> Dict:
        """Portfolio Risk Analysis"""
        try:
            max_drawdown = features.get('max_drawdown', 0)
            current_drawdown = features.get('current_drawdown', 0)
            correlation_risk = features.get('correlation_risk', 0.5)
            
            # Portfolio Risk Score
            drawdown_risk = max_drawdown / self.max_drawdown if self.max_drawdown > 0 else 0
            current_dd_risk = current_drawdown / (self.max_drawdown * 0.5) if self.max_drawdown > 0 else 0
            
            portfolio_risk_score = (
                drawdown_risk * 0.4 +
                current_dd_risk * 0.3 +
                correlation_risk * 0.3
            )
            
            risk_level = 'low' if portfolio_risk_score < 0.4 else 'medium' if portfolio_risk_score < 0.7 else 'high'
            
            return {
                'score': portfolio_risk_score,
                'level': risk_level,
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown,
                'correlation_risk': correlation_risk
            }
            
        except Exception as e:
            logger.error(f"❌ Portfolio Risk Analysis Fehler: {e}")
            return {'score': 0.5, 'level': 'medium'}
    
    async def _analyze_position_risk(self, features: Dict) -> Dict:
        """Position Risk Analysis"""
        try:
            var_estimate = abs(features.get('var_estimate', 0.02))
            expected_shortfall = abs(features.get('expected_shortfall', 0.03))
            volatility = features.get('historical_volatility', 0.02)
            
            # Position Risk Score
            var_risk = var_estimate / 0.05  # Normalize to 5% daily VaR
            es_risk = expected_shortfall / 0.05  # Normalize to 5% expected shortfall
            vol_risk = volatility / 0.3  # Normalize to 30% annual volatility
            
            position_risk_score = (
                var_risk * 0.4 +
                es_risk * 0.3 +
                vol_risk * 0.3
            )
            
            risk_level = 'low' if position_risk_score < 0.4 else 'medium' if position_risk_score < 0.7 else 'high'
            
            return {
                'score': position_risk_score,
                'level': risk_level,
                'var_estimate': var_estimate,
                'expected_shortfall': expected_shortfall,
                'volatility': volatility
            }
            
        except Exception as e:
            logger.error(f"❌ Position Risk Analysis Fehler: {e}")
            return {'score': 0.5, 'level': 'medium'}
    
    async def _analyze_market_risk(self, features: Dict) -> Dict:
        """Market Risk Analysis"""
        try:
            tail_risk = features.get('tail_risk', 0.3)
            vol_of_vol = features.get('volatility_of_volatility', 0.5)
            liquidity_risk = features.get('liquidity_risk', 0.3)
            
            # Market Risk Score
            market_risk_score = (
                tail_risk * 0.4 +
                vol_of_vol * 0.3 +
                liquidity_risk * 0.3
            )
            
            risk_level = 'low' if market_risk_score < 0.4 else 'medium' if market_risk_score < 0.7 else 'high'
            
            return {
                'score': market_risk_score,
                'level': risk_level,
                'tail_risk': tail_risk,
                'vol_of_vol': vol_of_vol,
                'liquidity_risk': liquidity_risk
            }
            
        except Exception as e:
            logger.error(f"❌ Market Risk Analysis Fehler: {e}")
            return {'score': 0.5, 'level': 'medium'}
    
    async def _analyze_systemic_risk(self, features: Dict) -> Dict:
        """Systemic Risk Analysis"""
        try:
            market_correlation = features.get('market_correlation', 0.5)
            
            # Simplified systemic risk based on market correlation
            # In real implementation: consider macro factors, sector correlation, etc.
            
            systemic_risk_score = market_correlation  # High correlation = high systemic risk
            
            risk_level = 'low' if systemic_risk_score < 0.4 else 'medium' if systemic_risk_score < 0.7 else 'high'
            
            return {
                'score': systemic_risk_score,
                'level': risk_level,
                'market_correlation': market_correlation
            }
            
        except Exception as e:
            logger.error(f"❌ Systemic Risk Analysis Fehler: {e}")
            return {'score': 0.5, 'level': 'medium'}
    
    async def _calculate_overall_risk(self, portfolio_risk: Dict, position_risk: Dict,
                                    market_risk: Dict, systemic_risk: Dict) -> Dict:
        """Overall Risk Score Calculation"""
        try:
            # Risk Component Weights
            portfolio_weight = 0.3
            position_weight = 0.3
            market_weight = 0.25
            systemic_weight = 0.15
            
            # Weighted Overall Risk Score
            overall_score = (
                portfolio_risk.get('score', 0.5) * portfolio_weight +
                position_risk.get('score', 0.5) * position_weight +
                market_risk.get('score', 0.5) * market_weight +
                systemic_risk.get('score', 0.5) * systemic_weight
            )
            
            # Overall Risk Level
            if overall_score < 0.3:
                risk_level = 'low'
            elif overall_score < 0.6:
                risk_level = 'medium'
            elif overall_score < 0.8:
                risk_level = 'high'
            else:
                risk_level = 'extreme'
            
            return {
                'score': overall_score,
                'level': risk_level,
                'components': {
                    'portfolio': portfolio_risk.get('score', 0.5),
                    'position': position_risk.get('score', 0.5),
                    'market': market_risk.get('score', 0.5),
                    'systemic': systemic_risk.get('score', 0.5)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Overall Risk Calculation Fehler: {e}")
            return {'score': 0.5, 'level': 'medium'}
    
    async def _make_risk_decision(self, overall_risk: Dict, portfolio_risk: Dict, 
                                market_risk: Dict) -> Tuple[str, float]:
        """Risk-basierte Trading Decision"""
        try:
            action = 'HOLD'
            confidence = 0.0
            
            risk_score = overall_risk.get('score', 0.5)
            risk_level = overall_risk.get('level', 'medium')
            
            # Risk-based Decision Logic
            if risk_level == 'low' and risk_score < 0.25:
                # Low risk environment - can take larger positions
                action = 'BUY'  # Slight bullish bias in low risk
                confidence = 0.7
                
            elif risk_level == 'medium' and risk_score < 0.5:
                # Medium risk - moderate position
                action = 'HOLD'  # Wait for better opportunities
                confidence = 0.4
                
            elif risk_level == 'high':
                # High risk - reduce exposure
                action = 'SELL'  # Risk-off mode
                confidence = 0.6
                
            elif risk_level == 'extreme':
                # Extreme risk - emergency risk reduction
                action = 'SELL'
                confidence = 0.9
            
            # Portfolio Risk Adjustments
            if portfolio_risk.get('current_drawdown', 0) > self.max_drawdown * 0.7:
                action = 'SELL'  # Force risk reduction
                confidence = min(1.0, confidence * 1.5)
            
            # Market Risk Adjustments
            if market_risk.get('liquidity_risk', 0) > 0.7:
                confidence *= 0.7  # Reduce confidence in illiquid markets
            
            return action, min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"❌ Risk Decision Making Fehler: {e}")
            return 'HOLD', 0.3
    
    def _calculate_risk_adjusted_position_size(self, overall_risk: Dict, 
                                             confidence: float, features: Dict) -> float:
        """Risk-adjusted Position Size using Kelly Criterion principles"""
        try:
            base_size = 0.05  # 5% base (conservative)
            
            # Risk Score Adjustment
            risk_score = overall_risk.get('score', 0.5)
            risk_multiplier = max(0.1, 1 - risk_score)  # Lower risk = larger size
            
            # VaR-based Sizing
            var_estimate = abs(features.get('var_estimate', 0.02))
            var_multiplier = self.max_portfolio_risk / var_estimate if var_estimate > 0 else 1.0
            var_multiplier = min(2.0, var_multiplier)  # Cap at 2x
            
            # Sharpe Ratio Adjustment
            sharpe_ratio = features.get('sharpe_ratio', 0)
            sharpe_multiplier = max(0.5, 1 + sharpe_ratio * 0.2)  # Reward good risk-adjusted returns
            
            # Drawdown Adjustment
            current_drawdown = features.get('current_drawdown', 0)
            drawdown_multiplier = max(0.3, 1 - current_drawdown * 2)  # Reduce size during drawdowns
            
            # Final Position Size
            position_size = (base_size * 
                           risk_multiplier * 
                           var_multiplier * 
                           sharpe_multiplier * 
                           drawdown_multiplier * 
                           confidence)
            
            return min(0.15, max(0.005, position_size))  # 0.5% - 15% range
            
        except Exception as e:
            logger.error(f"❌ Risk-adjusted Position Size Calculation Fehler: {e}")
            return 0.05
    
    async def _assess_comprehensive_risk(self, features: Dict, market_data: Dict) -> Dict:
        """Comprehensive Risk Assessment"""
        try:
            overall_risk_score = (
                features.get('historical_volatility', 0.02) / 0.3 * 0.25 +  # Volatility risk
                abs(features.get('var_estimate', 0.02)) / 0.05 * 0.25 +     # VaR risk
                features.get('max_drawdown', 0) / 0.2 * 0.2 +               # Drawdown risk
                features.get('tail_risk', 0.3) * 0.15 +                     # Tail risk
                features.get('liquidity_risk', 0.3) * 0.15                  # Liquidity risk
            )
            
            risk_level = 'low' if overall_risk_score < 0.4 else 'medium' if overall_risk_score < 0.7 else 'high'
            
            return {
                'risk_score': min(1.0, overall_risk_score),
                'risk_level': risk_level,
                'volatility_risk': features.get('historical_volatility', 0.02) / 0.3,
                'var_risk': abs(features.get('var_estimate', 0.02)) / 0.05,
                'drawdown_risk': features.get('max_drawdown', 0) / 0.2,
                'tail_risk': features.get('tail_risk', 0.3),
                'liquidity_risk': features.get('liquidity_risk', 0.3)
            }
            
        except Exception as e:
            logger.error(f"❌ Comprehensive Risk Assessment Fehler: {e}")
            return {'risk_score': 0.5, 'risk_level': 'medium'}
    
    def _generate_risk_reasoning(self, action: str, overall_risk: Dict,
                               portfolio_risk: Dict, market_risk: Dict) -> str:
        """Human-readable Risk Reasoning"""
        try:
            risk_score = overall_risk.get('score', 0.5)
            risk_level = overall_risk.get('level', 'medium')
            current_dd = portfolio_risk.get('current_drawdown', 0)
            
            if action == 'BUY':
                return f"Low risk environment (Risk Score: {risk_score:.2f}) allows for position building. Current drawdown: {current_dd:.1%}"
                
            elif action == 'SELL':
                if risk_level == 'extreme':
                    return f"EXTREME RISK DETECTED (Score: {risk_score:.2f}). Emergency risk reduction required."
                elif current_dd > self.max_drawdown * 0.7:
                    return f"High drawdown ({current_dd:.1%}) triggers risk reduction. Max allowed: {self.max_drawdown:.1%}"
                else:
                    return f"High risk environment (Score: {risk_score:.2f}) requires defensive positioning."
            else:
                return f"Medium risk environment (Score: {risk_score:.2f}). Maintaining current exposure while monitoring risk levels."
                
        except Exception as e:
            logger.error(f"❌ Risk Reasoning Generation Fehler: {e}")
            return f"Risk management: {action} signal based on comprehensive risk analysis"


class MomentumTrackerAgent(SpecializedTradingAgent):
    """
    🚀 Momentum Tracking Specialist
    Experte für Momentum-basierte Strategien und Oszillatoren
    """
    
    async def initialize(self):
        self.agent_id = f"momentum_tracker_{id(self)}"
        self.specialization = AgentSpecialization.MOMENTUM_TRACKER
        self.expertise_areas = [
            'rsi_analysis', 'macd_signals', 'stochastic_oscillator',
            'momentum_divergence', 'price_momentum', 'volume_momentum'
        ]
        
        # Momentum-spezifische Parameter
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.momentum_lookback = 14
        self.divergence_threshold = 0.1
        
        logger.success("🚀 Momentum Tracker Agent initialisiert")
        return True
    
    async def generate_signal(self, market_data: Dict) -> Optional[AgentSignal]:
        """
        Momentum-basierte Signal Generation
        """
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')
            logger.debug(f"🚀 Momentum Analysis für {symbol}...")
            
            # Momentum Features
            momentum_features = await self._extract_momentum_features(market_data)
            
            if not momentum_features:
                return None
            
            # Momentum Analysis
            rsi_analysis = await self._analyze_rsi(momentum_features)
            macd_analysis = await self._analyze_macd(momentum_features)
            stochastic_analysis = await self._analyze_stochastic(momentum_features)
            momentum_divergence = await self._detect_momentum_divergence(momentum_features)
            
            # Overall Momentum Score
            overall_momentum = await self._calculate_overall_momentum(
                rsi_analysis, macd_analysis, stochastic_analysis, momentum_divergence
            )
            
            # Decision Logic
            action, confidence = await self._make_momentum_decision(
                overall_momentum, rsi_analysis, macd_analysis, momentum_divergence
            )
            
            # Position Size
            position_size = self._calculate_momentum_position_size(
                overall_momentum, confidence, momentum_features
            )
            
            # Risk Assessment
            risk_assessment = await self._assess_momentum_risk(momentum_features, market_data)
            
            # Technical Factors
            technical_factors = {
                'overall_momentum': overall_momentum.get('score', 0),
                'rsi_value': rsi_analysis.get('rsi', 50),
                'macd_signal': macd_analysis.get('signal', 'neutral'),
                'stochastic_k': stochastic_analysis.get('k_percent', 50),
                'divergence_detected': momentum_divergence.get('detected', False),
                'momentum_strength': momentum_features.get('momentum_strength', 0)
            }
            
            # Reasoning
            reasoning = self._generate_momentum_reasoning(
                action, overall_momentum, rsi_analysis, macd_analysis
            )
            
            signal = self._create_agent_signal(
                action=action,
                confidence=confidence,
                position_size=position_size,
                reasoning=reasoning,
                technical_factors=technical_factors,
                risk_assessment=risk_assessment,
                market_data=market_data
            )
            
            logger.debug(f"🚀 Momentum Signal: {action} (Confidence: {confidence:.3f}, Score: {overall_momentum.get('score', 0):.3f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Momentum Tracker Signal Generation Fehler: {e}")
            return None
    
    async def _extract_momentum_features(self, market_data: Dict) -> Dict:
        """Momentum-spezifische Features extrahieren"""
        try:
            close_price = market_data.get('close', 0)
            high_price = market_data.get('high', close_price)
            low_price = market_data.get('low', close_price)
            volume = market_data.get('volume', 0)
            
            price_history = self._generate_mock_price_history(close_price)
            features = {}
            
            # RSI (Relative Strength Index)
            features['rsi'] = self._calculate_rsi(price_history)
            features['rsi_trend'] = self._calculate_rsi_trend(price_history)
            
            # MACD (Moving Average Convergence Divergence)
            macd_data = self._calculate_macd(price_history)
            features['macd_line'] = macd_data['macd']
            features['macd_signal'] = macd_data['signal']
            features['macd_histogram'] = macd_data['histogram']
            
            # Stochastic Oscillator
            stoch_data = self._calculate_stochastic(price_history, high_price, low_price)
            features['stoch_k'] = stoch_data['k_percent']
            features['stoch_d'] = stoch_data['d_percent']
            
            # Price Momentum
            features['price_momentum'] = self._calculate_price_momentum(price_history)
            features['momentum_strength'] = self._calculate_momentum_strength(price_history)
            
            # Volume Momentum
            features['volume_momentum'] = self._calculate_volume_momentum(volume, price_history)
            
            # Rate of Change (ROC)
            features['roc'] = self._calculate_roc(price_history)
            
            # Williams %R
            features['williams_r'] = self._calculate_williams_r(price_history, high_price, low_price)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Momentum Features Extraction Fehler: {e}")
            return {}
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI (Relative Strength Index) Calculation"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50.0
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_rsi_trend(self, prices: List[float]) -> str:
        """RSI Trend Analysis"""
        if len(prices) < 28:
            return 'neutral'
        
        recent_rsi = self._calculate_rsi(prices[-14:])
        older_rsi = self._calculate_rsi(prices[-28:-14])
        
        if recent_rsi > older_rsi + 5:
            return 'rising'
        elif recent_rsi < older_rsi - 5:
            return 'falling'
        else:
            return 'neutral'
    
    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """MACD Calculation"""
        if len(prices) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        # EMA Calculation
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        # MACD Line
        macd_line = ema_fast - ema_slow
        
        # Signal Line (EMA of MACD)
        # Simplified: use last few MACD values
        macd_history = [macd_line] * signal  # Mock MACD history
        signal_line = np.mean(macd_history)
        
        # Histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices) if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_stochastic(self, prices: List[float], current_high: float, current_low: float, 
                            k_period: int = 14, d_period: int = 3) -> Dict:
        """Stochastic Oscillator Calculation"""
        if len(prices) < k_period:
            return {'k_percent': 50, 'd_percent': 50}
        
        # Use price history to simulate highs and lows
        recent_prices = prices[-k_period:]
        highest_high = max(recent_prices)
        lowest_low = min(recent_prices)
        current_close = prices[-1]
        
        # %K Calculation
        if highest_high == lowest_low:
            k_percent = 50
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # %D Calculation (simplified)
        d_percent = k_percent  # In real implementation, this would be SMA of %K
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    def _calculate_price_momentum(self, prices: List[float], period: int = 10) -> float:
        """Price Momentum Calculation"""
        if len(prices) < period + 1:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-period-1]
        
        momentum = (current_price - past_price) / past_price if past_price != 0 else 0
        return momentum
    
    def _calculate_momentum_strength(self, prices: List[float]) -> float:
        """Momentum Strength Score"""
        if len(prices) < 20:
            return 0.3
        
        # Calculate multiple momentum periods
        mom_5 = self._calculate_price_momentum(prices, 5)
        mom_10 = self._calculate_price_momentum(prices, 10)
        mom_20 = self._calculate_price_momentum(prices, 20)
        
        # Weighted average of momentum
        strength = (mom_5 * 0.5 + mom_10 * 0.3 + mom_20 * 0.2)
        
        # Normalize to 0-1 scale
        return min(1.0, max(0.0, abs(strength) * 10))
    
    def _calculate_volume_momentum(self, current_volume: float, prices: List[float]) -> float:
        """Volume Momentum Analysis"""
        if not prices:
            return 0.0
        
        # Mock volume history
        avg_volume = 1000000  # Mock average volume
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Price momentum factor
        price_momentum = self._calculate_price_momentum(prices)
        
        # Volume momentum combines volume and price momentum
        volume_momentum = volume_ratio * abs(price_momentum)
        
        return min(1.0, volume_momentum)
    
    def _calculate_roc(self, prices: List[float], period: int = 12) -> float:
        """Rate of Change Calculation"""
        if len(prices) < period + 1:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-period-1]
        
        roc = ((current_price - past_price) / past_price) * 100 if past_price != 0 else 0
        return roc
    
    def _calculate_williams_r(self, prices: List[float], current_high: float, 
                            current_low: float, period: int = 14) -> float:
        """Williams %R Calculation"""
        if len(prices) < period:
            return -50.0  # Neutral
        
        recent_prices = prices[-period:]
        highest_high = max(recent_prices)
        lowest_low = min(recent_prices)
        current_close = prices[-1]
        
        if highest_high == lowest_low:
            return -50.0
        
        williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
        return williams_r
    
    async def _analyze_rsi(self, features: Dict) -> Dict:
        """RSI Analysis"""
        try:
            rsi = features.get('rsi', 50)
            rsi_trend = features.get('rsi_trend', 'neutral')
            
            # RSI Classification
            if rsi > self.rsi_overbought:
                rsi_signal = 'overbought'
                signal_strength = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            elif rsi < self.rsi_oversold:
                rsi_signal = 'oversold'
                signal_strength = (self.rsi_oversold - rsi) / self.rsi_oversold
            else:
                rsi_signal = 'neutral'
                signal_strength = 0.0
            
            return {
                'rsi': rsi,
                'signal': rsi_signal,
                'strength': signal_strength,
                'trend': rsi_trend
            }
            
        except Exception as e:
            logger.error(f"❌ RSI Analysis Fehler: {e}")
            return {'rsi': 50, 'signal': 'neutral', 'strength': 0.0}
    
    async def _analyze_macd(self, features: Dict) -> Dict:
        """MACD Analysis"""
        try:
            macd_line = features.get('macd_line', 0)
            macd_signal = features.get('macd_signal', 0)
            histogram = features.get('macd_histogram', 0)
            
            # MACD Signal Classification
            if macd_line > macd_signal and histogram > 0:
                signal = 'bullish'
                strength = min(1.0, abs(histogram) / 100)  # Normalize
            elif macd_line < macd_signal and histogram < 0:
                signal = 'bearish'
                strength = min(1.0, abs(histogram) / 100)
            else:
                signal = 'neutral'
                strength = 0.0
            
            # MACD Crossover Detection
            crossover = 'bullish' if macd_line > macd_signal else 'bearish' if macd_line < macd_signal else 'none'
            
            return {
                'macd': macd_line,
                'signal_line': macd_signal,
                'histogram': histogram,
                'signal': signal,
                'strength': strength,
                'crossover': crossover
            }
            
        except Exception as e:
            logger.error(f"❌ MACD Analysis Fehler: {e}")
            return {'signal': 'neutral', 'strength': 0.0, 'crossover': 'none'}
    
    async def _analyze_stochastic(self, features: Dict) -> Dict:
        """Stochastic Oscillator Analysis"""
        try:
            k_percent = features.get('stoch_k', 50)
            d_percent = features.get('stoch_d', 50)
            
            # Stochastic Classification
            if k_percent > 80 and d_percent > 80:
                signal = 'overbought'
                strength = (k_percent - 80) / 20
            elif k_percent < 20 and d_percent < 20:
                signal = 'oversold'
                strength = (20 - k_percent) / 20
            else:
                signal = 'neutral'
                strength = 0.0
            
            # Stochastic Crossover
            if k_percent > d_percent:
                crossover = 'bullish'
            elif k_percent < d_percent:
                crossover = 'bearish'
            else:
                crossover = 'neutral'
            
            return {
                'k_percent': k_percent,
                'd_percent': d_percent,
                'signal': signal,
                'strength': strength,
                'crossover': crossover
            }
            
        except Exception as e:
            logger.error(f"❌ Stochastic Analysis Fehler: {e}")
            return {'signal': 'neutral', 'strength': 0.0, 'crossover': 'neutral'}
    
    async def _detect_momentum_divergence(self, features: Dict) -> Dict:
        """Momentum Divergence Detection"""
        try:
            price_momentum = features.get('price_momentum', 0)
            rsi = features.get('rsi', 50)
            
            # Simplified divergence detection
            # In real implementation, this would compare price highs/lows with RSI highs/lows
            
            divergence_detected = False
            divergence_type = 'none'
            
            # Bullish Divergence: Price makes lower low, RSI makes higher low
            if price_momentum < -0.02 and rsi > 35:  # Price down, RSI not oversold
                divergence_detected = True
                divergence_type = 'bullish'
                
            # Bearish Divergence: Price makes higher high, RSI makes lower high
            elif price_momentum > 0.02 and rsi < 65:  # Price up, RSI not overbought
                divergence_detected = True
                divergence_type = 'bearish'
            
            return {
                'detected': divergence_detected,
                'type': divergence_type,
                'strength': 0.7 if divergence_detected else 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ Momentum Divergence Detection Fehler: {e}")
            return {'detected': False, 'type': 'none', 'strength': 0.0}
    
    async def _calculate_overall_momentum(self, rsi_analysis: Dict, macd_analysis: Dict,
                                        stochastic_analysis: Dict, momentum_divergence: Dict) -> Dict:
        """Overall Momentum Score Calculation"""
        try:
            # Component Weights
            rsi_weight = 0.3
            macd_weight = 0.35
            stoch_weight = 0.25
            divergence_weight = 0.1
            
            # Convert signals to numeric scores
            rsi_score = self._signal_to_score(rsi_analysis.get('signal', 'neutral'))
            macd_score = self._signal_to_score(macd_analysis.get('signal', 'neutral'))
            stoch_score = self._signal_to_score(stochastic_analysis.get('signal', 'neutral'))
            
            # Divergence Score
            div_score = 0.0
            if momentum_divergence.get('detected', False):
                if momentum_divergence.get('type') == 'bullish':
                    div_score = 0.7
                elif momentum_divergence.get('type') == 'bearish':
                    div_score = -0.7
            
            # Weighted Overall Score
            overall_score = (
                rsi_score * rsi_weight +
                macd_score * macd_weight +
                stoch_score * stoch_weight +
                div_score * divergence_weight
            )
            
            # Overall Classification
            if overall_score > 0.3:
                overall_class = 'bullish'
            elif overall_score < -0.3:
                overall_class = 'bearish'
            else:
                overall_class = 'neutral'
            
            # Confidence based on agreement
            scores = [rsi_score, macd_score, stoch_score]
            agreement = 1 - (np.std(scores) / 2) if len(scores) > 1 else 0.5
            
            return {
                'score': overall_score,
                'class': overall_class,
                'agreement': agreement,
                'components': {
                    'rsi': rsi_score,
                    'macd': macd_score,
                    'stochastic': stoch_score,
                    'divergence': div_score
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Overall Momentum Calculation Fehler: {e}")
            return {'score': 0, 'class': 'neutral', 'agreement': 0.5}
    
    def _signal_to_score(self, signal: str) -> float:
        """Convert signal to numeric score"""
        signal_map = {
            'overbought': -0.7,
            'oversold': 0.7,
            'bullish': 0.8,
            'bearish': -0.8,
            'neutral': 0.0
        }
        return signal_map.get(signal, 0.0)
    
    async def _make_momentum_decision(self, overall_momentum: Dict, rsi_analysis: Dict,
                                    macd_analysis: Dict, momentum_divergence: Dict) -> Tuple[str, float]:
        """Momentum-basierte Trading Decision"""
        try:
            action = 'HOLD'
            confidence = 0.0
            
            overall_score = overall_momentum.get('score', 0)
            agreement = overall_momentum.get('agreement', 0.5)
            
            # Strong Momentum Signals
            if overall_score > 0.4 and agreement > 0.6:
                action = 'BUY'
                confidence = min(0.85, overall_score * agreement)
                
            elif overall_score < -0.4 and agreement > 0.6:
                action = 'SELL'
                confidence = min(0.85, abs(overall_score) * agreement)
            
            # Divergence Signals
            elif momentum_divergence.get('detected', False):
                div_type = momentum_divergence.get('type', 'none')
                div_strength = momentum_divergence.get('strength', 0)
                
                if div_type == 'bullish':
                    action = 'BUY'
                    confidence = div_strength * 0.8
                elif div_type == 'bearish':
                    action = 'SELL'
                    confidence = div_strength * 0.8
            
            # Oversold/Overbought Signals
            elif rsi_analysis.get('signal') == 'oversold' and overall_score > -0.2:
                action = 'BUY'
                confidence = rsi_analysis.get('strength', 0) * 0.7
                
            elif rsi_analysis.get('signal') == 'overbought' and overall_score < 0.2:
                action = 'SELL'
                confidence = rsi_analysis.get('strength', 0) * 0.7
            
            # MACD Crossover Boost
            if macd_analysis.get('crossover') == 'bullish' and action == 'BUY':
                confidence *= 1.15
            elif macd_analysis.get('crossover') == 'bearish' and action == 'SELL':
                confidence *= 1.15
            
            # Final Confidence Clipping
            confidence = min(1.0, max(0.0, confidence))
            
            return action, confidence
            
        except Exception as e:
            logger.error(f"❌ Momentum Decision Making Fehler: {e}")
            return 'HOLD', 0.0
    
    def _calculate_momentum_position_size(self, overall_momentum: Dict, 
                                        confidence: float, features: Dict) -> float:
        """Momentum-adjusted Position Size"""
        try:
            base_size = 0.1  # 10% base
            
            # Momentum Strength Multiplier
            momentum_score = abs(overall_momentum.get('score', 0))
            momentum_multiplier = 1 + momentum_score  # 1.0 - 2.0 range
            
            # Agreement Multiplier
            agreement = overall_momentum.get('agreement', 0.5)
            agreement_multiplier = 0.5 + agreement  # 0.5 - 1.5 range
            
            # RSI Extreme Multiplier
            rsi = features.get('rsi', 50)
            if rsi < 25 or rsi > 75:  # Extreme RSI
                rsi_multiplier = 1.3
            else:
                rsi_multiplier = 1.0
            
            # Volume Momentum Multiplier
            volume_momentum = features.get('volume_momentum', 0)
            volume_multiplier = 1 + volume_momentum * 0.3  # Up to 1.3x
            
            # Final Position Size
            position_size = (base_size * 
                           momentum_multiplier * 
                           agreement_multiplier * 
                           rsi_multiplier * 
                           volume_multiplier * 
                           confidence)
            
            return min(0.22, max(0.01, position_size))  # 1% - 22% range
            
        except Exception as e:
            logger.error(f"❌ Momentum Position Size Calculation Fehler: {e}")
            return 0.1
    
    async def _assess_momentum_risk(self, features: Dict, market_data: Dict) -> Dict:
        """Momentum-spezifische Risk Assessment"""
        try:
            # Momentum Strength Risk
            momentum_strength = features.get('momentum_strength', 0)
            strength_risk = momentum_strength  # High momentum = high risk
            
            # RSI Extreme Risk
            rsi = features.get('rsi', 50)
            if rsi > 80 or rsi < 20:
                rsi_risk = 0.8
            elif rsi > 70 or rsi < 30:
                rsi_risk = 0.5
            else:
                rsi_risk = 0.2
            
            # MACD Divergence Risk
            macd_histogram = abs(features.get('macd_histogram', 0))
            macd_risk = min(0.8, macd_histogram / 50)  # Normalize
            
            # Volume Risk
            volume_momentum = features.get('volume_momentum', 0)
            volume_risk = 1 - volume_momentum  # Low volume = high risk
            
            # Overall Risk Score
            risk_score = (
                strength_risk * 0.3 +
                rsi_risk * 0.3 +
                macd_risk * 0.2 +
                volume_risk * 0.2
            )
            
            risk_level = 'low' if risk_score < 0.4 else 'medium' if risk_score < 0.7 else 'high'
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'strength_risk': strength_risk,
                'rsi_risk': rsi_risk,
                'macd_risk': macd_risk,
                'volume_risk': volume_risk
            }
            
        except Exception as e:
            logger.error(f"❌ Momentum Risk Assessment Fehler: {e}")
            return {'risk_score': 0.5, 'risk_level': 'medium'}
    
    def _generate_momentum_reasoning(self, action: str, overall_momentum: Dict,
                                   rsi_analysis: Dict, macd_analysis: Dict) -> str:
        """Human-readable Momentum Reasoning"""
        try:
            overall_score = overall_momentum.get('score', 0)
            overall_class = overall_momentum.get('class', 'neutral')
            rsi = rsi_analysis.get('rsi', 50)
            macd_signal = macd_analysis.get('signal', 'neutral')
            
            if action == 'BUY':
                if rsi_analysis.get('signal') == 'oversold':
                    return f"RSI oversold signal (RSI: {rsi:.1f}) with {overall_class} momentum consensus. Score: {overall_score:.2f}"
                elif macd_signal == 'bullish':
                    return f"MACD bullish crossover with strong momentum. Overall score: {overall_score:.2f}"
                else:
                    return f"Strong bullish momentum consensus ({overall_class}). Multiple indicators aligned."
                    
            elif action == 'SELL':
                if rsi_analysis.get('signal') == 'overbought':
                    return f"RSI overbought signal (RSI: {rsi:.1f}) with {overall_class} momentum consensus. Score: {overall_score:.2f}"
                elif macd_signal == 'bearish':
                    return f"MACD bearish crossover with weak momentum. Overall score: {overall_score:.2f}"
                else:
                    return f"Strong bearish momentum consensus ({overall_class}). Multiple indicators aligned."
            else:
                return f"Mixed momentum signals ({overall_class}). RSI: {rsi:.1f}, MACD: {macd_signal}. Awaiting clearer direction."
                
        except Exception as e:
            logger.error(f"❌ Momentum Reasoning Generation Fehler: {e}")
            return f"Momentum analysis: {action} signal based on oscillator analysis"


class PatternRecognizerAgent(SpecializedTradingAgent):
    """
    🔍 Pattern Recognition Specialist
    Experte für Chart-Pattern und Candlestick-Analyse
    """
    
    async def initialize(self):
        self.agent_id = f"pattern_recognizer_{id(self)}"
        self.specialization = AgentSpecialization.PATTERN_RECOGNIZER
        self.expertise_areas = [
            'chart_patterns', 'candlestick_patterns', 'support_resistance',
            'fibonacci_levels', 'harmonic_patterns', 'wave_analysis'
        ]
        
        # Pattern-spezifische Parameter
        self.pattern_confidence_threshold = 0.6
        self.support_resistance_strength = 3  # Minimum touches
        self.fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.pattern_lookback = 50
        
        logger.success("🔍 Pattern Recognizer Agent initialisiert")
        return True
    
    async def generate_signal(self, market_data: Dict) -> Optional[AgentSignal]:
        """
        Pattern-basierte Signal Generation
        """
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')
            logger.debug(f"🔍 Pattern Analysis für {symbol}...")
            
            # Pattern Features
            pattern_features = await self._extract_pattern_features(market_data)
            
            if not pattern_features:
                return None
            
            # Pattern Analysis
            chart_patterns = await self._detect_chart_patterns(pattern_features)
            candlestick_patterns = await self._detect_candlestick_patterns(pattern_features)
            support_resistance = await self._analyze_support_resistance(pattern_features)
            fibonacci_analysis = await self._analyze_fibonacci_levels(pattern_features)
            
            # Overall Pattern Score
            overall_pattern = await self._calculate_overall_pattern_score(
                chart_patterns, candlestick_patterns, support_resistance, fibonacci_analysis
            )
            
            # Pattern-based Decision Logic
            action, confidence = await self._make_pattern_decision(
                overall_pattern, chart_patterns, support_resistance
            )
            
            # Pattern-adjusted Position Size
            position_size = self._calculate_pattern_position_size(
                overall_pattern, confidence, pattern_features
            )
            
            # Risk Assessment
            risk_assessment = await self._assess_pattern_risk(pattern_features, market_data)
            
            # Technical Factors
            technical_factors = {
                'overall_pattern_score': overall_pattern.get('score', 0),
                'chart_pattern_detected': chart_patterns.get('detected', False),
                'candlestick_signal': candlestick_patterns.get('signal', 'neutral'),
                'support_resistance_level': support_resistance.get('current_level', 'none'),
                'fibonacci_level': fibonacci_analysis.get('current_level', 0),
                'pattern_strength': pattern_features.get('pattern_strength', 0)
            }
            
            # Reasoning
            reasoning = self._generate_pattern_reasoning(
                action, overall_pattern, chart_patterns, support_resistance
            )
            
            signal = self._create_agent_signal(
                action=action,
                confidence=confidence,
                position_size=position_size,
                reasoning=reasoning,
                technical_factors=technical_factors,
                risk_assessment=risk_assessment,
                market_data=market_data
            )
            
            logger.debug(f"🔍 Pattern Signal: {action} (Confidence: {confidence:.3f}, Pattern: {chart_patterns.get('type', 'none')})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Pattern Recognizer Signal Generation Fehler: {e}")
            return None
    
    async def _extract_pattern_features(self, market_data: Dict) -> Dict:
        """Pattern-spezifische Features extrahieren"""
        try:
            close_price = market_data.get('close', 0)
            high_price = market_data.get('high', close_price)
            low_price = market_data.get('low', close_price)
            volume = market_data.get('volume', 0)
            
            price_history = self._generate_mock_price_history(close_price)
            features = {}
            
            # OHLC Data (simulated)
            features['ohlc_data'] = self._generate_mock_ohlc_data(price_history)
            
            # Support and Resistance Levels
            features['support_levels'] = self._find_support_levels(price_history)
            features['resistance_levels'] = self._find_resistance_levels(price_history)
            
            # Price Swings
            features['swing_highs'] = self._find_swing_highs(price_history)
            features['swing_lows'] = self._find_swing_lows(price_history)
            
            # Trend Lines
            features['trend_lines'] = self._calculate_trend_lines(price_history)
            
            # Pattern Strength Indicators
            features['pattern_strength'] = self._calculate_pattern_strength(price_history, volume)
            features['volume_confirmation'] = self._analyze_volume_confirmation(volume, price_history)
            
            # Fibonacci Retracement Levels
            features['fibonacci_levels'] = self._calculate_fibonacci_levels(price_history)
            
            # Recent Price Action
            features['recent_candles'] = self._get_recent_candle_patterns(features['ohlc_data'])
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Pattern Features Extraction Fehler: {e}")
            return {}
    
    def _generate_mock_ohlc_data(self, price_history: List[float]) -> List[Dict]:
        """Generate mock OHLC data from price history"""
        ohlc_data = []
        
        for i, price in enumerate(price_history):
            # Simulate OHLC with some randomness
            volatility = price * 0.01  # 1% volatility
            
            open_price = price_history[i-1] if i > 0 else price
            high_price = price + np.random.uniform(0, volatility)
            low_price = price - np.random.uniform(0, volatility)
            close_price = price
            
            # Ensure OHLC logic
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            ohlc_data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.uniform(500000, 2000000)
            })
        
        return ohlc_data[-20:]  # Last 20 candles
    
    def _find_support_levels(self, prices: List[float]) -> List[float]:
        """Find Support Levels"""
        if len(prices) < 10:
            return []
        
        support_levels = []
        window = 5
        
        for i in range(window, len(prices) - window):
            is_support = True
            current_price = prices[i]
            
            # Check if current price is a local minimum
            for j in range(i - window, i + window + 1):
                if j != i and prices[j] < current_price:
                    is_support = False
                    break
            
            if is_support:
                support_levels.append(current_price)
        
        # Remove duplicates and sort
        support_levels = sorted(list(set([round(level, 2) for level in support_levels])))
        return support_levels[-5:]  # Return last 5 support levels
    
    def _find_resistance_levels(self, prices: List[float]) -> List[float]:
        """Find Resistance Levels"""
        if len(prices) < 10:
            return []
        
        resistance_levels = []
        window = 5
        
        for i in range(window, len(prices) - window):
            is_resistance = True
            current_price = prices[i]
            
            # Check if current price is a local maximum
            for j in range(i - window, i + window + 1):
                if j != i and prices[j] > current_price:
                    is_resistance = False
                    break
            
            if is_resistance:
                resistance_levels.append(current_price)
        
        # Remove duplicates and sort
        resistance_levels = sorted(list(set([round(level, 2) for level in resistance_levels])), reverse=True)
        return resistance_levels[:5]  # Return top 5 resistance levels
    
    def _find_swing_highs(self, prices: List[float]) -> List[int]:
        """Find Swing High indices"""
        swing_highs = []
        window = 3
        
        for i in range(window, len(prices) - window):
            is_swing_high = True
            
            for j in range(i - window, i + window + 1):
                if j != i and prices[j] >= prices[i]:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append(i)
        
        return swing_highs
    
    def _find_swing_lows(self, prices: List[float]) -> List[int]:
        """Find Swing Low indices"""
        swing_lows = []
        window = 3
        
        for i in range(window, len(prices) - window):
            is_swing_low = True
            
            for j in range(i - window, i + window + 1):
                if j != i and prices[j] <= prices[i]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append(i)
        
        return swing_lows
    
    def _calculate_trend_lines(self, prices: List[float]) -> Dict:
        """Calculate Trend Lines"""
        if len(prices) < 10:
            return {'support_line': None, 'resistance_line': None}
        
        # Simplified trend line calculation
        x = np.arange(len(prices))
        
        # Support trend line (connect swing lows)
        swing_lows = self._find_swing_lows(prices)
        if len(swing_lows) >= 2:
            x_lows = np.array(swing_lows[-2:])
            y_lows = np.array([prices[i] for i in swing_lows[-2:]])
            support_slope = (y_lows[1] - y_lows[0]) / (x_lows[1] - x_lows[0]) if x_lows[1] != x_lows[0] else 0
            support_intercept = y_lows[0] - support_slope * x_lows[0]
            support_line = {'slope': support_slope, 'intercept': support_intercept}
        else:
            support_line = None
        
        # Resistance trend line (connect swing highs)
        swing_highs = self._find_swing_highs(prices)
        if len(swing_highs) >= 2:
            x_highs = np.array(swing_highs[-2:])
            y_highs = np.array([prices[i] for i in swing_highs[-2:]])
            resistance_slope = (y_highs[1] - y_highs[0]) / (x_highs[1] - x_highs[0]) if x_highs[1] != x_highs[0] else 0
            resistance_intercept = y_highs[0] - resistance_slope * x_highs[0]
            resistance_line = {'slope': resistance_slope, 'intercept': resistance_intercept}
        else:
            resistance_line = None
        
        return {
            'support_line': support_line,
            'resistance_line': resistance_line
        }
    
    def _calculate_pattern_strength(self, prices: List[float], volume: float) -> float:
        """Calculate Pattern Strength"""
        if len(prices) < 5:
            return 0.3
        
        # Price consistency
        recent_volatility = np.std(prices[-5:]) / np.mean(prices[-5:])
        consistency_score = max(0, 1 - recent_volatility * 10)
        
        # Volume factor
        avg_volume = 1000000  # Mock average volume
        volume_factor = min(2.0, volume / avg_volume) if avg_volume > 0 else 1.0
        
        # Pattern strength
        pattern_strength = (consistency_score * 0.7 + (volume_factor - 1) * 0.3)
        return min(1.0, max(0.0, pattern_strength))
    
    def _analyze_volume_confirmation(self, volume: float, prices: List[float]) -> bool:
        """Analyze Volume Confirmation"""
        if len(prices) < 2:
            return False
        
        # Mock volume analysis
        avg_volume = 1000000
        price_change = abs(prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0
        
        # High volume with significant price change = confirmation
        return volume > avg_volume * 1.5 and price_change > 0.01
    
    def _calculate_fibonacci_levels(self, prices: List[float]) -> Dict:
        """Calculate Fibonacci Retracement Levels"""
        if len(prices) < 10:
            return {'levels': {}, 'swing_high': 0, 'swing_low': 0}
        
        # Find significant swing high and low
        swing_high = max(prices[-20:]) if len(prices) >= 20 else max(prices)
        swing_low = min(prices[-20:]) if len(prices) >= 20 else min(prices)
        
        # Calculate Fibonacci levels
        price_range = swing_high - swing_low
        fib_levels = {}
        
        for level in self.fibonacci_levels:
            fib_levels[f"{level:.1%}"] = swing_high - (price_range * level)
        
        return {
            'levels': fib_levels,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'range': price_range
        }
    
    def _get_recent_candle_patterns(self, ohlc_data: List[Dict]) -> List[str]:
        """Get Recent Candlestick Patterns"""
        if len(ohlc_data) < 3:
            return []
        
        patterns = []
        
        # Analyze last few candles for patterns
        for i in range(1, min(4, len(ohlc_data))):
            candle = ohlc_data[-i]
            prev_candle = ohlc_data[-i-1] if i < len(ohlc_data) else None
            
            pattern = self._identify_candlestick_pattern(candle, prev_candle)
            if pattern:
                patterns.append(pattern)
        
        return patterns
    
    def _identify_candlestick_pattern(self, candle: Dict, prev_candle: Dict = None) -> str:
        """Identify Candlestick Pattern"""
        if not candle:
            return ""
        
        open_price = candle['open']
        high_price = candle['high']
        low_price = candle['low']
        close_price = candle['close']
        
        body_size = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        candle_range = high_price - low_price
        
        # Avoid division by zero
        if candle_range == 0:
            return "doji"
        
        body_ratio = body_size / candle_range
        upper_shadow_ratio = upper_shadow / candle_range
        lower_shadow_ratio = lower_shadow / candle_range
        
        # Doji
        if body_ratio < 0.1:
            return "doji"
        
        # Hammer/Hanging Man
        if lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1 and body_ratio < 0.3:
            return "hammer" if close_price > open_price else "hanging_man"
        
        # Shooting Star/Inverted Hammer
        if upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1 and body_ratio < 0.3:
            return "shooting_star" if close_price < open_price else "inverted_hammer"
        
        # Marubozu
        if body_ratio > 0.9:
            return "bullish_marubozu" if close_price > open_price else "bearish_marubozu"
        
        # Spinning Top
        if body_ratio < 0.3 and upper_shadow_ratio > 0.3 and lower_shadow_ratio > 0.3:
            return "spinning_top"
        
        return ""
    
    async def _detect_chart_patterns(self, features: Dict) -> Dict:
        """Detect Chart Patterns"""
        try:
            support_levels = features.get('support_levels', [])
            resistance_levels = features.get('resistance_levels', [])
            trend_lines = features.get('trend_lines', {})
            swing_highs = features.get('swing_highs', [])
            swing_lows = features.get('swing_lows', [])
            
            pattern_detected = False
            pattern_type = "none"
            pattern_confidence = 0.0
            
            # Triangle Pattern Detection
            if len(support_levels) >= 2 and len(resistance_levels) >= 2:
                support_line = trend_lines.get('support_line')
                resistance_line = trend_lines.get('resistance_line')
                
                if support_line and resistance_line:
                    # Ascending Triangle
                    if abs(resistance_line['slope']) < 0.1 and support_line['slope'] > 0:
                        pattern_detected = True
                        pattern_type = "ascending_triangle"
                        pattern_confidence = 0.7
                    
                    # Descending Triangle
                    elif abs(support_line['slope']) < 0.1 and resistance_line['slope'] < 0:
                        pattern_detected = True
                        pattern_type = "descending_triangle"
                        pattern_confidence = 0.7
                    
                    # Symmetrical Triangle
                    elif support_line['slope'] > 0 and resistance_line['slope'] < 0:
                        pattern_detected = True
                        pattern_type = "symmetrical_triangle"
                        pattern_confidence = 0.6
            
            # Head and Shoulders (simplified)
            if len(swing_highs) >= 3:
                recent_highs = swing_highs[-3:]
                if len(recent_highs) == 3:
                    # Check if middle high is higher than shoulders
                    if (features.get('ohlc_data', [{}])[-1].get('close', 0) > 0 and
                        len(features.get('ohlc_data', [])) > recent_highs[1]):
                        pattern_detected = True
                        pattern_type = "head_and_shoulders"
                        pattern_confidence = 0.6
            
            # Double Top/Bottom
            if len(swing_highs) >= 2:
                last_two_highs = swing_highs[-2:]
                if len(last_two_highs) == 2:
                    pattern_detected = True
                    pattern_type = "double_top"
                    pattern_confidence = 0.5
            
            return {
                'detected': pattern_detected,
                'type': pattern_type,
                'confidence': pattern_confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Chart Pattern Detection Fehler: {e}")
            return {'detected': False, 'type': 'none', 'confidence': 0.0}
    
    async def _detect_candlestick_patterns(self, features: Dict) -> Dict:
        """Detect Candlestick Patterns"""
        try:
            recent_candles = features.get('recent_candles', [])
            
            if not recent_candles:
                return {'signal': 'neutral', 'pattern': 'none', 'strength': 0.0}
            
            # Analyze recent patterns
            bullish_patterns = ['hammer', 'bullish_marubozu', 'inverted_hammer']
            bearish_patterns = ['hanging_man', 'bearish_marubozu', 'shooting_star']
            
            bullish_count = sum(1 for pattern in recent_candles if pattern in bullish_patterns)
            bearish_count = sum(1 for pattern in recent_candles if pattern in bearish_patterns)
            
            if bullish_count > bearish_count:
                signal = 'bullish'
                strength = bullish_count / len(recent_candles)
                pattern = recent_candles[0] if recent_candles[0] in bullish_patterns else 'mixed_bullish'
            elif bearish_count > bullish_count:
                signal = 'bearish'
                strength = bearish_count / len(recent_candles)
                pattern = recent_candles[0] if recent_candles[0] in bearish_patterns else 'mixed_bearish'
            else:
                signal = 'neutral'
                strength = 0.0
                pattern = 'mixed'
            
            return {
                'signal': signal,
                'pattern': pattern,
                'strength': strength,
                'recent_patterns': recent_candles
            }
            
        except Exception as e:
            logger.error(f"❌ Candlestick Pattern Detection Fehler: {e}")
            return {'signal': 'neutral', 'pattern': 'none', 'strength': 0.0}
    
    async def _analyze_support_resistance(self, features: Dict) -> Dict:
        """Analyze Support and Resistance Levels"""
        try:
            support_levels = features.get('support_levels', [])
            resistance_levels = features.get('resistance_levels', [])
            ohlc_data = features.get('ohlc_data', [])
            
            if not ohlc_data:
                return {'current_level': 'none', 'signal': 'neutral', 'strength': 0.0}
            
            current_price = ohlc_data[-1].get('close', 0)
            
            # Find nearest support and resistance
            nearest_support = max([level for level in support_levels if level < current_price], default=0)
            nearest_resistance = min([level for level in resistance_levels if level > current_price], default=float('inf'))
            
            # Determine current level
            support_distance = (current_price - nearest_support) / current_price if nearest_support > 0 else 1
            resistance_distance = (nearest_resistance - current_price) / current_price if nearest_resistance < float('inf') else 1
            
            if support_distance < 0.02:  # Within 2% of support
                current_level = 'near_support'
                signal = 'bullish'  # Bounce expected
                strength = 1 - support_distance / 0.02
            elif resistance_distance < 0.02:  # Within 2% of resistance
                current_level = 'near_resistance'
                signal = 'bearish'  # Rejection expected
                strength = 1 - resistance_distance / 0.02
            else:
                current_level = 'neutral_zone'
                signal = 'neutral'
                strength = 0.0
            
            return {
                'current_level': current_level,
                'signal': signal,
                'strength': strength,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance
            }
            
        except Exception as e:
            logger.error(f"❌ Support/Resistance Analysis Fehler: {e}")
            return {'current_level': 'none', 'signal': 'neutral', 'strength': 0.0}
    
    async def _analyze_fibonacci_levels(self, features: Dict) -> Dict:
        """Analyze Fibonacci Levels"""
        try:
            fib_data = features.get('fibonacci_levels', {})
            ohlc_data = features.get('ohlc_data', [])
            
            if not fib_data.get('levels') or not ohlc_data:
                return {'current_level': 0, 'signal': 'neutral', 'strength': 0.0}
            
            current_price = ohlc_data[-1].get('close', 0)
            fib_levels = fib_data.get('levels', {})
            
            # Find nearest Fibonacci level
            nearest_level = None
            min_distance = float('inf')
            
            for level_name, level_price in fib_levels.items():
                distance = abs(current_price - level_price) / current_price
                if distance < min_distance:
                    min_distance = distance
                    nearest_level = level_name
            
            # Determine signal based on Fibonacci level
            if min_distance < 0.01:  # Within 1% of Fibonacci level
                signal = 'bullish' if nearest_level in ['61.8%', '50.0%', '38.2%'] else 'neutral'
                strength = 1 - min_distance / 0.01
            else:
                signal = 'neutral'
                strength = 0.0
            
            return {
                'current_level': nearest_level or 0,
                'signal': signal,
                'strength': strength,
                'distance': min_distance
            }
            
        except Exception as e:
            logger.error(f"❌ Fibonacci Analysis Fehler: {e}")
            return {'current_level': 0, 'signal': 'neutral', 'strength': 0.0}
    
    async def _calculate_overall_pattern_score(self, chart_patterns: Dict, candlestick_patterns: Dict,
                                             support_resistance: Dict, fibonacci_analysis: Dict) -> Dict:
        """Calculate Overall Pattern Score"""
        try:
            # Component Weights
            chart_weight = 0.35
            candlestick_weight = 0.25
            sr_weight = 0.25
            fibonacci_weight = 0.15
            
            # Convert signals to scores
            chart_score = self._pattern_signal_to_score(chart_patterns.get('type', 'none'))
            candlestick_score = self._signal_to_score(candlestick_patterns.get('signal', 'neutral'))
            sr_score = self._signal_to_score(support_resistance.get('signal', 'neutral'))
            fib_score = self._signal_to_score(fibonacci_analysis.get('signal', 'neutral'))
            
            # Weighted Overall Score
            overall_score = (
                chart_score * chart_weight +
                candlestick_score * candlestick_weight +
                sr_score * sr_weight +
                fib_score * fibonacci_weight
            )
            
            # Overall Classification
            if overall_score > 0.3:
                overall_class = 'bullish'
            elif overall_score < -0.3:
                overall_class = 'bearish'
            else:
                overall_class = 'neutral'
            
            # Confidence based on pattern strength
            strengths = [
                chart_patterns.get('confidence', 0),
                candlestick_patterns.get('strength', 0),
                support_resistance.get('strength', 0),
                fibonacci_analysis.get('strength', 0)
            ]
            
            confidence = np.mean([s for s in strengths if s > 0]) if any(s > 0 for s in strengths) else 0.0
            
            return {
                'score': overall_score,
                'class': overall_class,
                'confidence': confidence,
                'components': {
                    'chart': chart_score,
                    'candlestick': candlestick_score,
                    'support_resistance': sr_score,
                    'fibonacci': fib_score
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Overall Pattern Score Calculation Fehler: {e}")
            return {'score': 0, 'class': 'neutral', 'confidence': 0.0}
    
    def _pattern_signal_to_score(self, pattern_type: str) -> float:
        """Convert pattern type to score"""
        pattern_scores = {
            'ascending_triangle': 0.7,
            'descending_triangle': -0.7,
            'symmetrical_triangle': 0.0,
            'head_and_shoulders': -0.6,
            'inverse_head_and_shoulders': 0.6,
            'double_top': -0.5,
            'double_bottom': 0.5,
            'none': 0.0
        }
        return pattern_scores.get(pattern_type, 0.0)
    
    def _signal_to_score(self, signal: str) -> float:
        """Convert signal to numeric score"""
        signal_map = {
            'bullish': 0.8,
            'bearish': -0.8,
            'neutral': 0.0
        }
        return signal_map.get(signal, 0.0)
    
    async def _make_pattern_decision(self, overall_pattern: Dict, chart_patterns: Dict,
                                   support_resistance: Dict) -> Tuple[str, float]:
        """Pattern-basierte Trading Decision"""
        try:
            # Placeholder Implementation - Pattern-based Decision Logic
            pattern_score = overall_pattern.get('score', 0)
            pattern_confidence = overall_pattern.get('confidence', 0)
            
            if pattern_score > 0.3:
                action = "BUY"
                confidence = min(pattern_confidence * 0.8, 0.9)
            elif pattern_score < -0.3:
                action = "SELL"  
                confidence = min(pattern_confidence * 0.8, 0.9)
            else:
                action = "HOLD"
                confidence = 0.3
                
            return action, confidence
            
        except Exception as e:
            logger.error(f"❌ Pattern Decision Fehler: {e}")
            return "HOLD", 0.3
