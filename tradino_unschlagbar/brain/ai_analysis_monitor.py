#!/usr/bin/env python3
"""
🔍 AI ANALYSIS DEBUGGING & MONITORING SYSTEM
Vollständige Transparenz über AI-Entscheidungsprozesse in TRADINO
"""

import os
import sys
import json
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelType(Enum):
    """🤖 AI Model Types"""
    XGBOOST = "xgboost_trend"
    LIGHTGBM = "lightgbm_volatility"
    RANDOM_FOREST = "random_forest_risk"
    RL_AGENT = "rl_agent"
    ENSEMBLE = "ensemble"

class DecisionType(Enum):
    """📊 Decision Types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    NEUTRAL = "neutral"

@dataclass
class ModelPrediction:
    """🔬 Individual Model Prediction"""
    model_type: ModelType
    prediction: float  # Probability or raw prediction
    confidence: float  # 0-1 confidence score
    feature_importance: Dict[str, float]  # Feature importance scores
    processing_time: float  # milliseconds
    accuracy_score: float  # Historical accuracy
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type.value,
            'prediction': float(self.prediction),
            'confidence': float(self.confidence),
            'feature_importance': {k: float(v) for k, v in self.feature_importance.items()},
            'processing_time': float(self.processing_time),
            'accuracy_score': float(self.accuracy_score),
            'timestamp': self.timestamp
        }

@dataclass
class EnsembleAnalysis:
    """🎭 Ensemble Decision Analysis"""
    individual_predictions: List[ModelPrediction]
    ensemble_weights: Dict[str, float]
    final_prediction: float
    final_confidence: float
    decision: DecisionType
    agreement_score: float  # How much models agree (0-1)
    dominant_features: Dict[str, float]  # Most important features
    reasoning: List[str]  # Human-readable reasoning
    risk_assessment: Dict[str, float]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'individual_predictions': [p.to_dict() for p in self.individual_predictions],
            'ensemble_weights': {k: float(v) for k, v in self.ensemble_weights.items()},
            'final_prediction': float(self.final_prediction),
            'final_confidence': float(self.final_confidence),
            'decision': self.decision.value,
            'agreement_score': float(self.agreement_score),
            'dominant_features': {k: float(v) for k, v in self.dominant_features.items()},
            'reasoning': self.reasoning,
            'risk_assessment': {k: float(v) for k, v in self.risk_assessment.items()},
            'timestamp': self.timestamp
        }

@dataclass
class MarketConditions:
    """📈 Market Conditions Context"""
    symbol: str
    price: float
    volume: float
    volatility: float
    trend_strength: float
    support_level: float
    resistance_level: float
    rsi: float
    macd: float
    bollinger_position: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AIAnalysisLogger:
    """📝 AI Analysis Logger Klasse"""
    
    def __init__(self, log_dir: str = "data/ai_analysis"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Analysis Storage
        self.analysis_history: List[EnsembleAnalysis] = []
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self.feature_importance_history: Dict[str, List[float]] = {}
        
        # Configuration
        self.max_history = 1000
        self.log_file = os.path.join(log_dir, f"ai_analysis_{datetime.now().strftime('%Y%m%d')}.json")
        
        # Performance tracking
        self.total_predictions = 0
        self.correct_predictions = 0
        self.model_agreement_history: List[float] = []
        
        print(f"🔍 AI Analysis Logger initialized: {log_dir}")
    
    def log_model_prediction(self, model_type: ModelType, prediction: float, 
                           confidence: float, feature_importance: Dict[str, float],
                           processing_time: float, accuracy_score: float) -> ModelPrediction:
        """📊 Log individual model prediction"""
        
        model_pred = ModelPrediction(
            model_type=model_type,
            prediction=prediction,
            confidence=confidence,
            feature_importance=feature_importance.copy(),
            processing_time=processing_time,
            accuracy_score=accuracy_score,
            timestamp=datetime.now().isoformat()
        )
        
        # Update feature importance history
        for feature, importance in feature_importance.items():
            if feature not in self.feature_importance_history:
                self.feature_importance_history[feature] = []
            self.feature_importance_history[feature].append(importance)
            
            # Keep only recent history
            if len(self.feature_importance_history[feature]) > 100:
                self.feature_importance_history[feature] = self.feature_importance_history[feature][-100:]
        
        # Update model performance
        model_name = model_type.value
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {
                'total_predictions': 0,
                'avg_confidence': 0.0,
                'avg_processing_time': 0.0,
                'accuracy': accuracy_score
            }
        
        perf = self.model_performance[model_name]
        perf['total_predictions'] += 1
        perf['avg_confidence'] = ((perf['avg_confidence'] * (perf['total_predictions'] - 1)) + confidence) / perf['total_predictions']
        perf['avg_processing_time'] = ((perf['avg_processing_time'] * (perf['total_predictions'] - 1)) + processing_time) / perf['total_predictions']
        perf['accuracy'] = accuracy_score
        
        return model_pred
    
    def log_ensemble_analysis(self, individual_predictions: List[ModelPrediction],
                            ensemble_weights: Dict[str, float], final_prediction: float,
                            final_confidence: float, decision: DecisionType,
                            market_conditions: MarketConditions) -> EnsembleAnalysis:
        """🎭 Log ensemble analysis with detailed reasoning"""
        
        # Calculate agreement score
        predictions = [p.prediction for p in individual_predictions]
        agreement_score = 1.0 - np.std(predictions) if len(predictions) > 1 else 1.0
        agreement_score = max(0.0, min(1.0, agreement_score))
        
        # Calculate dominant features
        dominant_features = {}
        for pred in individual_predictions:
            weight = ensemble_weights.get(pred.model_type.value, 1.0)
            for feature, importance in pred.feature_importance.items():
                if feature not in dominant_features:
                    dominant_features[feature] = 0.0
                dominant_features[feature] += importance * weight
        
        # Normalize dominant features
        total_weight = sum(ensemble_weights.values()) if ensemble_weights else 1.0
        for feature in dominant_features:
            dominant_features[feature] /= total_weight
        
        # Sort by importance and keep top 5
        dominant_features = dict(sorted(dominant_features.items(), 
                                      key=lambda x: x[1], reverse=True)[:5])
        
        # Generate reasoning
        reasoning = self._generate_reasoning(individual_predictions, ensemble_weights, 
                                           dominant_features, market_conditions, decision)
        
        # Risk assessment
        risk_assessment = self._calculate_risk_assessment(individual_predictions, 
                                                        market_conditions, final_confidence)
        
        ensemble_analysis = EnsembleAnalysis(
            individual_predictions=individual_predictions,
            ensemble_weights=ensemble_weights.copy(),
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            decision=decision,
            agreement_score=agreement_score,
            dominant_features=dominant_features,
            reasoning=reasoning,
            risk_assessment=risk_assessment,
            timestamp=datetime.now().isoformat()
        )
        
        # Add to history
        self.analysis_history.append(ensemble_analysis)
        if len(self.analysis_history) > self.max_history:
            self.analysis_history = self.analysis_history[-self.max_history:]
        
        # Track agreement
        self.model_agreement_history.append(agreement_score)
        if len(self.model_agreement_history) > 100:
            self.model_agreement_history = self.model_agreement_history[-100:]
        
        # Save to file
        self._save_analysis_to_file(ensemble_analysis)
        
        self.total_predictions += 1
        
        return ensemble_analysis
    
    def _generate_reasoning(self, predictions: List[ModelPrediction], 
                          weights: Dict[str, float], dominant_features: Dict[str, float],
                          market_conditions: MarketConditions, decision: DecisionType) -> List[str]:
        """🧠 Generate human-readable reasoning"""
        
        reasoning = []
        
        # Model consensus analysis
        bullish_models = sum(1 for p in predictions if p.prediction > 0.5)
        bearish_models = len(predictions) - bullish_models
        
        if bullish_models > bearish_models:
            reasoning.append(f"🟢 {bullish_models}/{len(predictions)} Modelle zeigen bullische Signale")
        elif bearish_models > bullish_models:
            reasoning.append(f"🔴 {bearish_models}/{len(predictions)} Modelle zeigen bearische Signale")
        else:
            reasoning.append(f"⚖️ Modelle sind gespalten ({bullish_models}:{bearish_models})")
        
        # Confidence analysis
        avg_confidence = np.mean([p.confidence for p in predictions])
        if avg_confidence > 0.8:
            reasoning.append(f"✅ Hohe Modell-Konfidenz: {avg_confidence:.1%}")
        elif avg_confidence < 0.6:
            reasoning.append(f"⚠️ Niedrige Modell-Konfidenz: {avg_confidence:.1%}")
        
        # Feature importance analysis
        top_feature = max(dominant_features.items(), key=lambda x: x[1])
        reasoning.append(f"📊 Hauptfaktor: {top_feature[0]} ({top_feature[1]:.1%} Gewichtung)")
        
        # Market conditions context
        if market_conditions.volatility > 0.05:
            reasoning.append(f"⚡ Hohe Volatilität: {market_conditions.volatility:.1%}")
        
        if market_conditions.rsi > 70:
            reasoning.append(f"📈 RSI überkauft: {market_conditions.rsi:.1f}")
        elif market_conditions.rsi < 30:
            reasoning.append(f"📉 RSI überverkauft: {market_conditions.rsi:.1f}")
        
        # Decision justification
        if decision == DecisionType.BUY:
            reasoning.append("🚀 Kaufsignal: Positive Marktdynamik erkannt")
        elif decision == DecisionType.SELL:
            reasoning.append("📉 Verkaufssignal: Negative Marktdynamik erkannt")
        else:
            reasoning.append("⏸️ Neutral: Abwarten empfohlen")
        
        return reasoning
    
    def _calculate_risk_assessment(self, predictions: List[ModelPrediction],
                                 market_conditions: MarketConditions, 
                                 final_confidence: float) -> Dict[str, float]:
        """🛡️ Calculate risk assessment"""
        
        # Model disagreement risk
        prediction_std = np.std([p.prediction for p in predictions])
        disagreement_risk = min(1.0, prediction_std * 2)
        
        # Confidence risk
        confidence_risk = 1.0 - final_confidence
        
        # Market volatility risk
        volatility_risk = min(1.0, market_conditions.volatility * 10)
        
        # Volume risk (low volume = higher risk)
        avg_volume = 1000000  # Mock average volume
        volume_risk = max(0.0, min(1.0, 1.0 - (market_conditions.volume / avg_volume)))
        
        # Technical indicator risk
        rsi_risk = 0.0
        if market_conditions.rsi > 80 or market_conditions.rsi < 20:
            rsi_risk = 0.3
        elif market_conditions.rsi > 70 or market_conditions.rsi < 30:
            rsi_risk = 0.1
        
        # Overall risk score
        overall_risk = np.mean([disagreement_risk, confidence_risk, volatility_risk, 
                               volume_risk, rsi_risk])
        
        return {
            'overall_risk': overall_risk,
            'model_disagreement': disagreement_risk,
            'confidence_risk': confidence_risk,
            'volatility_risk': volatility_risk,
            'volume_risk': volume_risk,
            'technical_risk': rsi_risk
        }
    
    def _save_analysis_to_file(self, analysis: EnsembleAnalysis):
        """💾 Save analysis to file"""
        
        try:
            # Load existing data
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {'analyses': []}
            
            # Add new analysis
            data['analyses'].append(analysis.to_dict())
            
            # Keep only recent analyses
            if len(data['analyses']) > self.max_history:
                data['analyses'] = data['analyses'][-self.max_history:]
            
            # Save back to file
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")
    
    def get_recent_analyses(self, count: int = 10) -> List[EnsembleAnalysis]:
        """📋 Get recent analyses"""
        return self.analysis_history[-count:]
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """📊 Get model performance summary"""
        
        summary = {
            'total_predictions': self.total_predictions,
            'avg_agreement': np.mean(self.model_agreement_history) if self.model_agreement_history else 0.0,
            'models': {}
        }
        
        for model_name, perf in self.model_performance.items():
            summary['models'][model_name] = {
                'predictions': perf['total_predictions'],
                'avg_confidence': perf['avg_confidence'],
                'avg_processing_time': perf['avg_processing_time'],
                'accuracy': perf['accuracy']
            }
        
        return summary
    
    def get_feature_importance_trends(self) -> Dict[str, float]:
        """📈 Get feature importance trends"""
        
        trends = {}
        for feature, history in self.feature_importance_history.items():
            if len(history) >= 2:
                recent_avg = np.mean(history[-10:]) if len(history) >= 10 else np.mean(history)
                old_avg = np.mean(history[:-10]) if len(history) >= 20 else np.mean(history[:len(history)//2])
                trends[feature] = recent_avg - old_avg
            else:
                trends[feature] = 0.0
        
        return trends

class AnalysisVisualizer:
    """📊 Analysis Visualizer"""
    
    def __init__(self, logger: AIAnalysisLogger):
        self.logger = logger
    
    def generate_text_report(self, analysis: EnsembleAnalysis, 
                           market_conditions: MarketConditions) -> str:
        """📝 Generate detailed text report"""
        
        report = []
        report.append("🔍 AI ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"📅 Zeit: {analysis.timestamp}")
        report.append(f"💰 Symbol: {market_conditions.symbol}")
        report.append(f"💵 Preis: ${market_conditions.price:.2f}")
        report.append("")
        
        # Decision Summary
        report.append("🎯 ENTSCHEIDUNG")
        report.append("-" * 20)
        report.append(f"Aktion: {analysis.decision.value.upper()}")
        report.append(f"Konfidenz: {analysis.final_confidence:.1%}")
        report.append(f"Vorhersage: {analysis.final_prediction:.3f}")
        report.append(f"Modell-Einigkeit: {analysis.agreement_score:.1%}")
        report.append("")
        
        # Individual Models
        report.append("🤖 MODELL-DETAILS")
        report.append("-" * 20)
        for pred in analysis.individual_predictions:
            weight = analysis.ensemble_weights.get(pred.model_type.value, 0.0)
            report.append(f"{pred.model_type.value.upper()}:")
            report.append(f"  Vorhersage: {pred.prediction:.3f}")
            report.append(f"  Konfidenz: {pred.confidence:.1%}")
            report.append(f"  Gewichtung: {weight:.1%}")
            report.append(f"  Verarbeitungszeit: {pred.processing_time:.1f}ms")
            report.append(f"  Genauigkeit: {pred.accuracy_score:.1%}")
            report.append("")
        
        # Feature Importance
        report.append("📊 WICHTIGSTE FAKTOREN")
        report.append("-" * 25)
        for feature, importance in analysis.dominant_features.items():
            report.append(f"{feature}: {importance:.1%}")
        report.append("")
        
        # Reasoning
        report.append("🧠 BEGRÜNDUNG")
        report.append("-" * 15)
        for reason in analysis.reasoning:
            report.append(f"• {reason}")
        report.append("")
        
        # Risk Assessment
        report.append("🛡️ RISIKO-BEWERTUNG")
        report.append("-" * 20)
        for risk_type, risk_value in analysis.risk_assessment.items():
            risk_level = "NIEDRIG" if risk_value < 0.3 else "MITTEL" if risk_value < 0.7 else "HOCH"
            report.append(f"{risk_type}: {risk_value:.1%} ({risk_level})")
        report.append("")
        
        # Market Context
        report.append("📈 MARKT-KONTEXT")
        report.append("-" * 17)
        report.append(f"Volatilität: {market_conditions.volatility:.1%}")
        report.append(f"Volumen: {market_conditions.volume:,.0f}")
        report.append(f"RSI: {market_conditions.rsi:.1f}")
        report.append(f"MACD: {market_conditions.macd:.4f}")
        report.append(f"Trend-Stärke: {market_conditions.trend_strength:.1%}")
        
        return "\n".join(report)
    
    def generate_json_report(self, analysis: EnsembleAnalysis,
                           market_conditions: MarketConditions) -> Dict[str, Any]:
        """📄 Generate JSON report"""
        
        return {
            'analysis': analysis.to_dict(),
            'market_conditions': market_conditions.to_dict(),
            'summary': {
                'decision': analysis.decision.value,
                'confidence': analysis.final_confidence,
                'agreement': analysis.agreement_score,
                'risk_level': analysis.risk_assessment['overall_risk'],
                'top_feature': max(analysis.dominant_features.items(), 
                                 key=lambda x: x[1])[0] if analysis.dominant_features else 'unknown',
                'models_count': len(analysis.individual_predictions)
            }
        }
    
    def generate_dashboard_summary(self) -> Dict[str, Any]:
        """📊 Generate dashboard summary"""
        
        recent_analyses = self.logger.get_recent_analyses(10)
        performance = self.logger.get_model_performance_summary()
        feature_trends = self.logger.get_feature_importance_trends()
        
        if not recent_analyses:
            return {'error': 'No analyses available'}
        
        # Decision distribution
        decisions = [a.decision.value for a in recent_analyses]
        decision_counts = {decision: decisions.count(decision) for decision in set(decisions)}
        
        # Confidence trends
        confidences = [a.final_confidence for a in recent_analyses]
        avg_confidence = np.mean(confidences)
        confidence_trend = "steigend" if len(confidences) > 1 and confidences[-1] > confidences[0] else "fallend"
        
        # Agreement trends
        agreements = [a.agreement_score for a in recent_analyses]
        avg_agreement = np.mean(agreements)
        
        # Most influential features
        all_features = {}
        for analysis in recent_analyses:
            for feature, importance in analysis.dominant_features.items():
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(importance)
        
        top_features = {feature: np.mean(importances) 
                       for feature, importances in all_features.items()}
        top_features = dict(sorted(top_features.items(), 
                                 key=lambda x: x[1], reverse=True)[:5])
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_analyses': len(recent_analyses),
            'performance': performance,
            'recent_decisions': decision_counts,
            'avg_confidence': avg_confidence,
            'confidence_trend': confidence_trend,
            'avg_agreement': avg_agreement,
            'top_features': top_features,
            'feature_trends': feature_trends,
            'latest_analysis': recent_analyses[-1].to_dict() if recent_analyses else None
        }

class RealTimeAnalysisDisplay:
    """📺 Real-time Analysis Display"""
    
    def __init__(self, logger: AIAnalysisLogger, visualizer: AnalysisVisualizer):
        self.logger = logger
        self.visualizer = visualizer
        self.is_monitoring = False
        self.monitoring_thread = None
        
    def start_monitoring(self, interval: int = 30):
        """🔄 Start real-time monitoring"""
        
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        print(f"🔄 Real-time monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """🛑 Stop monitoring"""
        
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("🛑 Real-time monitoring stopped")
    
    def _monitoring_loop(self, interval: int):
        """🔄 Monitoring loop"""
        
        while self.is_monitoring:
            try:
                self.print_live_dashboard()
                time.sleep(interval)
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(interval)
    
    def print_live_dashboard(self):
        """📊 Print live dashboard"""
        
        try:
            dashboard = self.visualizer.generate_dashboard_summary()
            
            if 'error' in dashboard:
                print("📊 Keine Analysen verfügbar")
                return
            
            print("\n" + "=" * 80)
            print("📊 TRADINO AI LIVE DASHBOARD")
            print("=" * 80)
            print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📈 Analysen total: {dashboard['total_analyses']}")
            print(f"🎯 Durchschnittliche Konfidenz: {dashboard['avg_confidence']:.1%}")
            print(f"🤝 Modell-Einigkeit: {dashboard['avg_agreement']:.1%}")
            print(f"📊 Konfidenz-Trend: {dashboard['confidence_trend']}")
            
            print("\n🤖 MODELL-PERFORMANCE:")
            for model, perf in dashboard['performance']['models'].items():
                print(f"  {model.upper()}:")
                print(f"    Vorhersagen: {perf['predictions']}")
                print(f"    Konfidenz: {perf['avg_confidence']:.1%}")
                print(f"    Genauigkeit: {perf['accuracy']:.1%}")
                print(f"    Ø Zeit: {perf['avg_processing_time']:.1f}ms")
            
            print("\n📊 LETZTE ENTSCHEIDUNGEN:")
            for decision, count in dashboard['recent_decisions'].items():
                print(f"  {decision.upper()}: {count}")
            
            print("\n🔝 TOP FEATURES:")
            for feature, importance in dashboard['top_features'].items():
                print(f"  {feature}: {importance:.1%}")
            
            if dashboard['latest_analysis']:
                latest = dashboard['latest_analysis']
                print(f"\n🔍 LETZTE ANALYSE:")
                print(f"  Entscheidung: {latest['decision'].upper()}")
                print(f"  Konfidenz: {latest['final_confidence']:.1%}")
                print(f"  Einigkeit: {latest['agreement_score']:.1%}")
                print(f"  Zeit: {latest['timestamp'][:19]}")
            
            print("=" * 80)
            
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
    
    def get_last_analyses(self, count: int = 10) -> List[Dict[str, Any]]:
        """📋 Get last analyses for display"""
        
        recent = self.logger.get_recent_analyses(count)
        return [analysis.to_dict() for analysis in recent]
    
    def get_model_agreement_trend(self, period: int = 50) -> Dict[str, Any]:
        """📈 Get model agreement trend"""
        
        if len(self.logger.model_agreement_history) < 2:
            return {'trend': 'insufficient_data', 'current': 0.0, 'change': 0.0}
        
        recent_history = self.logger.model_agreement_history[-period:]
        
        if len(recent_history) < 2:
            return {'trend': 'insufficient_data', 'current': recent_history[-1], 'change': 0.0}
        
        current = recent_history[-1]
        previous = np.mean(recent_history[:-10]) if len(recent_history) > 10 else recent_history[0]
        
        change = current - previous
        trend = 'improving' if change > 0.05 else 'declining' if change < -0.05 else 'stable'
        
        return {
            'trend': trend,
            'current': current,
            'change': change,
            'history': recent_history
        }

# Global instances
ai_logger = None
ai_visualizer = None
ai_display = None

def initialize_ai_monitoring_system(log_dir: str = "data/ai_analysis") -> Tuple[AIAnalysisLogger, AnalysisVisualizer, RealTimeAnalysisDisplay]:
    """🚀 Initialize AI monitoring system"""
    
    global ai_logger, ai_visualizer, ai_display
    
    ai_logger = AIAnalysisLogger(log_dir)
    ai_visualizer = AnalysisVisualizer(ai_logger)
    ai_display = RealTimeAnalysisDisplay(ai_logger, ai_visualizer)
    
    print("🚀 AI Monitoring System initialized")
    return ai_logger, ai_visualizer, ai_display

def get_ai_monitoring_system() -> Tuple[Optional[AIAnalysisLogger], Optional[AnalysisVisualizer], Optional[RealTimeAnalysisDisplay]]:
    """📊 Get AI monitoring system instances"""
    return ai_logger, ai_visualizer, ai_display

if __name__ == "__main__":
    print("🔍 AI ANALYSIS MONITORING SYSTEM TEST")
    print("=" * 50)
    
    # Initialize system
    logger, visualizer, display = initialize_ai_monitoring_system()
    
    # Create mock data for testing
    mock_market_conditions = MarketConditions(
        symbol="BTC/USDT",
        price=45000.0,
        volume=1500000,
        volatility=0.035,
        trend_strength=0.65,
        support_level=44000.0,
        resistance_level=46000.0,
        rsi=62.5,
        macd=0.0015,
        bollinger_position=0.6,
        timestamp=datetime.now().isoformat()
    )
    
    # Create mock predictions
    xgb_pred = logger.log_model_prediction(
        ModelType.XGBOOST, 0.72, 0.85, 
        {'rsi': 0.3, 'macd': 0.25, 'volume_ratio': 0.2, 'trend_strength': 0.15, 'volatility': 0.1},
        15.2, 0.748
    )
    
    lgb_pred = logger.log_model_prediction(
        ModelType.LIGHTGBM, 0.68, 0.78,
        {'volatility': 0.35, 'bollinger_position': 0.25, 'rsi': 0.2, 'volume_ratio': 0.15, 'macd': 0.05},
        12.8, 0.751
    )
    
    rf_pred = logger.log_model_prediction(
        ModelType.RANDOM_FOREST, 0.74, 0.82,
        {'trend_strength': 0.4, 'rsi': 0.3, 'macd': 0.15, 'volatility': 0.1, 'volume_ratio': 0.05},
        18.5, 0.782
    )
    
    # Create ensemble analysis
    ensemble_weights = {'xgboost_trend': 0.35, 'lightgbm_volatility': 0.30, 'random_forest_risk': 0.35}
    final_prediction = 0.71
    final_confidence = 0.82
    decision = DecisionType.BUY
    
    analysis = logger.log_ensemble_analysis(
        [xgb_pred, lgb_pred, rf_pred],
        ensemble_weights,
        final_prediction,
        final_confidence,
        decision,
        mock_market_conditions
    )
    
    # Generate reports
    print("\n📝 TEXT REPORT:")
    print(visualizer.generate_text_report(analysis, mock_market_conditions))
    
    print("\n📊 DASHBOARD:")
    dashboard = visualizer.generate_dashboard_summary()
    for key, value in dashboard.items():
        if key != 'latest_analysis':
            print(f"{key}: {value}")
    
    print("\n🎉 AI Analysis Monitoring System is working!") 