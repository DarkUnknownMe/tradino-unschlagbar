"""
🧠 TRADINO UNSCHLAGBAR - Master AI
Zentraler AI Controller - Koordiniert alle AI-Module

Author: AI Trading Systems
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from brain.market_intelligence import MarketIntelligence, MarketSignal
from brain.pattern_recognition import PatternRecognition, DetectedPattern
from brain.sentiment_analyzer import SentimentAnalyzer, SentimentData
from brain.prediction_engine import PredictionEngine, PredictionResult, PredictionHorizon
from models.signal_models import AISignal, SignalType, SignalStrength, SignalAnalysis
from connectors.bitget_pro import BitgetPro
from utils.logger_pro import setup_logger, log_ai_decision
from utils.config_manager import ConfigManager
from utils.helpers import generate_signal_id

logger = setup_logger("MasterAI")


class AIDecisionLevel(Enum):
    """AI-Entscheidungs-Level"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"  
    AGGRESSIVE = "aggressive"


@dataclass
class AIAnalysisResult:
    """AI Analyse Ergebnis"""
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    signal_strength: SignalStrength
    confidence: float
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: float
    analysis_summary: Dict[str, Any]
    contributing_factors: List[str]
    metadata: Dict[str, Any]


class MasterAI:
    """🧠 Master AI Controller - Koordiniert alle AI-Module"""
    
    def __init__(self, config: ConfigManager, bitget_connector: BitgetPro):
        self.config = config
        self.bitget = bitget_connector
        
        # AI Module
        self.market_intelligence: Optional[MarketIntelligence] = None
        self.pattern_recognition: Optional[PatternRecognition] = None
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self.prediction_engine: Optional[PredictionEngine] = None
        
        # AI Configuration
        self.decision_level = AIDecisionLevel(config.get('ai.decision_level', 'balanced'))
        self.min_confidence_threshold = config.get('ai.min_confidence', 0.65)
        self.analysis_timeout = config.get('ai.analysis_timeout', 30)  # Sekunden
        
        # Analysis Weights
        self.analysis_weights = {
            'market_intelligence': 0.30,
            'pattern_recognition': 0.25,
            'sentiment_analysis': 0.20,
            'prediction_engine': 0.25
        }
        
        # Performance Tracking
        self.signals_generated = 0
        self.successful_signals = 0
        self.total_analysis_time = 0
        self.last_analysis: Dict[str, datetime] = {}
        
    async def initialize(self) -> bool:
        """🔥 Master AI initialisieren"""
        try:
            logger.info("🧠 Master AI wird initialisiert...")
            
            # AI Module initialisieren
            self.market_intelligence = MarketIntelligence(self.config, self.bitget)
            await self.market_intelligence.initialize()
            
            self.pattern_recognition = PatternRecognition(self.config, self.bitget)
            await self.pattern_recognition.initialize()
            
            self.sentiment_analyzer = SentimentAnalyzer(self.config)
            await self.sentiment_analyzer.initialize()
            
            self.prediction_engine = PredictionEngine(self.config, self.bitget, self.market_intelligence)
            await self.prediction_engine.initialize()
            
            logger.success("✅ Master AI erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ Master AI Initialisierung fehlgeschlagen: {e}")
            return False
    
    # ==================== MAIN AI ANALYSIS ====================
    
    async def analyze_and_generate_signal(self, symbol: str) -> Optional[AISignal]:
        """🎯 Komplette AI-Analyse und Signal-Generierung"""
        try:
            start_time = datetime.utcnow()
            logger.info(f"🧠 AI-Analyse wird durchgeführt: {symbol}")
            
            # Parallel AI-Analysen
            tasks = [
                self._analyze_market_intelligence(symbol),
                self._analyze_patterns(symbol),
                self._analyze_sentiment(symbol),
                self._analyze_predictions(symbol)
            ]
            
            # Analyses mit Timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.analysis_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ AI-Analyse Timeout für {symbol}")
                return None
            
            # Ergebnisse extrahieren
            market_result, pattern_result, sentiment_result, prediction_result = results
            
            # Fehler-Behandlung
            analyses = {}
            if not isinstance(market_result, Exception) and market_result:
                analyses['market_intelligence'] = market_result
            if not isinstance(pattern_result, Exception) and pattern_result:
                analyses['pattern_recognition'] = pattern_result
            if not isinstance(sentiment_result, Exception) and sentiment_result:
                analyses['sentiment_analysis'] = sentiment_result
            if not isinstance(prediction_result, Exception) and prediction_result:
                analyses['prediction_engine'] = prediction_result
            
            if not analyses:
                logger.warning(f"⚠️ Keine AI-Analyse-Ergebnisse für {symbol}")
                return None
            
            # AI-Signal generieren
            ai_signal = await self._synthesize_ai_signal(symbol, analyses)
            
            # Performance Tracking
            analysis_time = (datetime.utcnow() - start_time).total_seconds()
            self.total_analysis_time += analysis_time
            self.last_analysis[symbol] = datetime.utcnow()
            
            if ai_signal:
                self.signals_generated += 1
                
                log_ai_decision(
                    "MasterAI",
                    f"{symbol} {ai_signal.signal_type.value} {ai_signal.strength.value}",
                    ai_signal.confidence
                )
            
            logger.info(f"🧠 AI-Analyse abgeschlossen in {analysis_time:.2f}s")
            return ai_signal
            
        except Exception as e:
            logger.error(f"❌ Fehler bei AI-Analyse für {symbol}: {e}")
            return None
    
    # ==================== INDIVIDUAL ANALYSES ====================
    
    async def _analyze_market_intelligence(self, symbol: str) -> Optional[Dict[str, Any]]:
        """📊 Market Intelligence Analyse"""
        try:
            if not self.market_intelligence:
                return None
            
            # Market Analysis
            market_analysis = await self.market_intelligence.analyze_market(symbol)
            if not market_analysis:
                return None
            
            # Market Signal
            market_signal = await self.market_intelligence.get_market_signal(symbol)
            
            return {
                'analysis': market_analysis,
                'signal': market_signal,
                'confidence': market_analysis.confidence,
                'trend_strength': market_analysis.trend_strength,
                'volatility': market_analysis.volatility_score,
                'volume': market_analysis.volume_score,
                'regime': market_analysis.regime.value
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Market Intelligence: {e}")
            return None
    
    async def _analyze_patterns(self, symbol: str) -> Optional[Dict[str, Any]]:
        """🔍 Pattern Recognition Analyse"""
        try:
            if not self.pattern_recognition:
                return None
            
            # Pattern Detection
            patterns = await self.pattern_recognition.detect_patterns(symbol)
            if not patterns:
                return None
            
            # Top Patterns nach Confidence
            top_patterns = sorted(patterns, key=lambda x: x.confidence, reverse=True)[:3]
            
            # Pattern Signals aggregieren
            bullish_signals = [p for p in top_patterns if 'bullish' in p.signal.value]
            bearish_signals = [p for p in top_patterns if 'bearish' in p.signal.value]
            
            # Overall Pattern Signal
            bullish_strength = sum(p.confidence for p in bullish_signals)
            bearish_strength = sum(p.confidence for p in bearish_signals)
            
            if bullish_strength > bearish_strength:
                pattern_signal = 'bullish'
                pattern_confidence = bullish_strength / len(bullish_signals) if bullish_signals else 0
            elif bearish_strength > bullish_strength:
                pattern_signal = 'bearish'
                pattern_confidence = bearish_strength / len(bearish_signals) if bearish_signals else 0
            else:
                pattern_signal = 'neutral'
                pattern_confidence = 0.5
            
            return {
                'patterns': top_patterns,
                'signal': pattern_signal,
                'confidence': pattern_confidence,
                'bullish_count': len(bullish_signals),
                'bearish_count': len(bearish_signals),
                'top_pattern': top_patterns[0] if top_patterns else None
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Pattern Recognition: {e}")
            return None
    
    async def _analyze_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """🎭 Sentiment Analyse"""
        try:
            if not self.sentiment_analyzer:
                return None
            
            # Sentiment Analysis
            sentiment_data = await self.sentiment_analyzer.analyze_sentiment(symbol)
            if not sentiment_data:
                return None
            
            # Sentiment Signal
            sentiment_signal = self.sentiment_analyzer.get_sentiment_signal(sentiment_data)
            
            return {
                'sentiment_data': sentiment_data,
                'signal': sentiment_signal,
                'confidence': sentiment_data.confidence,
                'overall_sentiment': sentiment_data.overall_sentiment,
                'fear_greed_index': sentiment_data.fear_greed_index,
                'sentiment_level': sentiment_data.sentiment_level.value
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Sentiment Analysis: {e}")
            return None
    
    async def _analyze_predictions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """🤖 Prediction Engine Analyse"""
        try:
            if not self.prediction_engine:
                return None
            
            # Predictions für verschiedene Horizonte
            predictions = {}
            for horizon in [PredictionHorizon.SHORT_TERM, PredictionHorizon.MEDIUM_TERM]:
                pred = await self.prediction_engine.predict_price(symbol, horizon)
                if pred:
                    predictions[horizon.value] = pred
            
            if not predictions:
                return None
            
            # Hauptprediction (Short-term)
            main_prediction = predictions.get(PredictionHorizon.SHORT_TERM.value)
            if not main_prediction:
                main_prediction = list(predictions.values())[0]
            
            return {
                'predictions': predictions,
                'main_prediction': main_prediction,
                'confidence': main_prediction.confidence,
                'direction': main_prediction.direction,
                'price_change_percent': main_prediction.price_change_percent
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Prediction Engine: {e}")
            return None
    
    # ==================== SIGNAL SYNTHESIS ====================
    
    async def _synthesize_ai_signal(self, symbol: str, analyses: Dict[str, Dict]) -> Optional[AISignal]:
        """🎯 AI-Signal aus allen Analysen synthetisieren"""
        try:
            # Signal Scores sammeln
            signal_scores = {}
            confidence_scores = []
            contributing_factors = []
            
            # Market Intelligence
            if 'market_intelligence' in analyses:
                mi = analyses['market_intelligence']
                if mi['signal']:
                    if mi['signal'].trend_direction.value >= 3:  # Bullish
                        signal_scores['market_intelligence'] = mi['signal'].signal_strength
                    elif mi['signal'].trend_direction.value <= 1:  # Bearish
                        signal_scores['market_intelligence'] = -mi['signal'].signal_strength
                    else:
                        signal_scores['market_intelligence'] = 0
                    
                    confidence_scores.append(mi['confidence'])
                    contributing_factors.append(f"Market {mi['regime']} regime")
            
            # Pattern Recognition
            if 'pattern_recognition' in analyses:
                pr = analyses['pattern_recognition']
                if pr['signal'] == 'bullish':
                    signal_scores['pattern_recognition'] = pr['confidence']
                elif pr['signal'] == 'bearish':
                    signal_scores['pattern_recognition'] = -pr['confidence']
                else:
                    signal_scores['pattern_recognition'] = 0
                
                confidence_scores.append(pr['confidence'])
                if pr['top_pattern']:
                    contributing_factors.append(f"{pr['top_pattern'].name} pattern")
            
            # Sentiment Analysis
            if 'sentiment_analysis' in analyses:
                sa = analyses['sentiment_analysis']
                sentiment_score = sa['overall_sentiment']  # -1 bis 1
                signal_scores['sentiment_analysis'] = sentiment_score
                
                confidence_scores.append(sa['confidence'])
                contributing_factors.append(f"{sa['sentiment_level']} sentiment")
            
            # Prediction Engine
            if 'prediction_engine' in analyses:
                pe = analyses['prediction_engine']
                if pe['direction'] == 'up':
                    signal_scores['prediction_engine'] = pe['confidence']
                elif pe['direction'] == 'down':
                    signal_scores['prediction_engine'] = -pe['confidence']
                else:
                    signal_scores['prediction_engine'] = 0
                
                confidence_scores.append(pe['confidence'])
                contributing_factors.append(f"ML predicts {pe['direction']}")
            
            # Gewichtete Signal-Berechnung
            weighted_signal = 0
            total_weight = 0
            
            for analysis_type, score in signal_scores.items():
                weight = self.analysis_weights.get(analysis_type, 0.25)
                weighted_signal += score * weight
                total_weight += weight
            
            if total_weight == 0:
                return None
            
            weighted_signal /= total_weight
            
            # Signal Type und Strength bestimmen
            signal_type, signal_strength = self._determine_signal_type_strength(weighted_signal)
            
            # Overall Confidence
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            # Confidence Threshold Check
            if overall_confidence < self.min_confidence_threshold:
                logger.info(f"🔍 Signal Confidence zu niedrig: {overall_confidence:.2%} < {self.min_confidence_threshold:.2%}")
                return None
            
            # Entry Price (aktueller Marktpreis)
            market_data = await self.bitget.get_market_data(symbol)
            entry_price = float(market_data.price) if market_data else 50000.0
            
            # Stop Loss und Take Profit basierend auf Signal Strength
            stop_loss, take_profit = self._calculate_sl_tp(entry_price, signal_type, signal_strength)
            
            # AI Signal erstellen
            ai_signal = AISignal(
                id=generate_signal_id(symbol, "AI"),
                symbol=symbol,
                signal_type=signal_type,
                strength=signal_strength,
                confidence=overall_confidence,
                predicted_price=entry_price * (1 + weighted_signal * 0.02),  # 2% max move
                current_price=entry_price,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=self._calculate_risk_reward(entry_price, stop_loss, take_profit) if stop_loss and take_profit else 1.0,
                timeframe="5m",
                strategy_source="MasterAI",
                model_version="v1.0",
                timestamp=datetime.utcnow(),
                metadata={
                    'analysis_count': len(analyses),
                    'contributing_factors': contributing_factors,
                    'signal_scores': signal_scores,
                    'weighted_signal': weighted_signal,
                    'decision_level': self.decision_level.value
                }
            )
            
            return ai_signal
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Signal-Synthese: {e}")
            return None
    
    def _determine_signal_type_strength(self, weighted_signal: float) -> Tuple[SignalType, SignalStrength]:
        """🎯 Signal Type und Strength bestimmen"""
        try:
            abs_signal = abs(weighted_signal)
            
            # Signal Type
            if weighted_signal > 0.1:
                signal_type = SignalType.BUY
            elif weighted_signal < -0.1:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Signal Strength
            if abs_signal >= 0.8:
                strength = SignalStrength.VERY_STRONG
            elif abs_signal >= 0.6:
                strength = SignalStrength.STRONG
            elif abs_signal >= 0.4:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            return signal_type, strength
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Signal Type/Strength Bestimmung: {e}")
            return SignalType.HOLD, SignalStrength.WEAK
    
    def _calculate_sl_tp(self, entry_price: float, signal_type: SignalType, signal_strength: SignalStrength) -> Tuple[Optional[float], Optional[float]]:
        """🛡️ Stop Loss und Take Profit berechnen"""
        try:
            if signal_type == SignalType.HOLD:
                return None, None
            
            # Strength-basierte Faktoren
            strength_factors = {
                SignalStrength.WEAK: (0.01, 0.015),        # 1% SL, 1.5% TP
                SignalStrength.MODERATE: (0.015, 0.03),    # 1.5% SL, 3% TP
                SignalStrength.STRONG: (0.02, 0.05),       # 2% SL, 5% TP
                SignalStrength.VERY_STRONG: (0.025, 0.075) # 2.5% SL, 7.5% TP
            }
            
            sl_factor, tp_factor = strength_factors.get(signal_strength, (0.02, 0.04))
            
            if signal_type == SignalType.BUY:
                stop_loss = entry_price * (1 - sl_factor)
                take_profit = entry_price * (1 + tp_factor)
            else:  # SELL
                stop_loss = entry_price * (1 + sl_factor)
                take_profit = entry_price * (1 - tp_factor)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"❌ Fehler bei SL/TP Berechnung: {e}")
            return None, None
    
    def _calculate_risk_reward(self, entry: float, stop_loss: float, take_profit: float) -> float:
        """⚖️ Risk/Reward Ratio berechnen"""
        try:
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            
            return reward / risk if risk > 0 else 1.0
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Risk/Reward Berechnung: {e}")
            return 1.0
    
    # ==================== PUBLIC METHODS ====================
    
    def get_ai_status(self) -> Dict[str, Any]:
        """📊 AI System Status"""
        return {
            'master_ai_active': True,
            'modules_initialized': {
                'market_intelligence': self.market_intelligence is not None,
                'pattern_recognition': self.pattern_recognition is not None,
                'sentiment_analyzer': self.sentiment_analyzer is not None,
                'prediction_engine': self.prediction_engine is not None
            },
            'signals_generated': self.signals_generated,
            'success_rate': (self.successful_signals / self.signals_generated) if self.signals_generated > 0 else 0,
            'avg_analysis_time': (self.total_analysis_time / self.signals_generated) if self.signals_generated > 0 else 0,
            'decision_level': self.decision_level.value,
            'min_confidence': self.min_confidence_threshold
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """📈 Performance Statistiken aller AI-Module"""
        stats = {}
        
        if self.market_intelligence:
            stats['market_intelligence'] = self.market_intelligence.get_performance_stats()
        if self.pattern_recognition:
            stats['pattern_recognition'] = self.pattern_recognition.get_performance_stats()
        if self.sentiment_analyzer:
            stats['sentiment_analyzer'] = self.sentiment_analyzer.get_performance_stats()
        if self.prediction_engine:
            stats['prediction_engine'] = self.prediction_engine.get_performance_stats()
        
        return stats
    
    async def update_signal_performance(self, signal_id: str, was_successful: bool):
        """📊 Signal Performance updaten"""
        try:
            if was_successful:
                self.successful_signals += 1
            
            logger.info(f"📊 Signal Performance updated: {signal_id} -> {'Success' if was_successful else 'Fail'}")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Signal Performance Update: {e}")
    
    async def shutdown(self):
        """🛑 Master AI herunterfahren"""
        try:
            # Alle AI-Module herunterfahren
            if self.market_intelligence:
                await self.market_intelligence.shutdown()
            if self.pattern_recognition:
                await self.pattern_recognition.shutdown()
            if self.sentiment_analyzer:
                await self.sentiment_analyzer.shutdown()
            if self.prediction_engine:
                await self.prediction_engine.shutdown()
            
            logger.info("✅ Master AI heruntergefahren")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren des Master AI: {e}")
