#!/usr/bin/env python3
"""
🧠 TRAINED MODEL INTEGRATION
Integration der trainierten ML Modelle in TRADINO UNSCHLAGBAR
"""

import os
import sys
import pickle
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AI Monitoring System
try:
    from ai_analysis_monitor import (
        initialize_ai_monitoring_system, get_ai_monitoring_system,
        ModelType, DecisionType, MarketConditions
    )
    AI_MONITORING_AVAILABLE = True
except ImportError:
    AI_MONITORING_AVAILABLE = False

class TrainedModelIntegration:
    """🤖 Integration der trainierten ML Modelle"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = {}
        self.feature_pipeline = None
        self.model_info = {}
        self.is_ready = False
        
        # Initialize AI Monitoring System
        if AI_MONITORING_AVAILABLE:
            try:
                self.ai_logger, self.ai_visualizer, self.ai_display = initialize_ai_monitoring_system()
                print("🔍 AI Analysis Monitoring integrated")
            except Exception as e:
                print(f"⚠️ AI Monitoring initialization failed: {e}")
                self.ai_logger = None
                self.ai_visualizer = None
                self.ai_display = None
        else:
            self.ai_logger = None
            self.ai_visualizer = None
            self.ai_display = None
        
        self.load_models()
    
    def load_models(self):
        """📂 Lade alle trainierten Modelle"""
        
        print("🔄 Loading trained models...")
        
        try:
            # Load model info
            info_path = os.path.join(self.models_dir, "model_info.json")
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
                print(f"📊 Model info loaded: {len(self.model_info)} models")
            
            # Load feature pipeline
            pipeline_path = os.path.join(self.models_dir, "feature_pipeline.pkl")
            if os.path.exists(pipeline_path):
                with open(pipeline_path, 'rb') as f:
                    self.feature_pipeline = pickle.load(f)
                print(f"🔧 Feature pipeline loaded: {len(self.feature_pipeline['features'])} features")
            
            # Load individual models
            for model_name, model_info in self.model_info.items():
                if model_info['type'] != 'ensemble':  # Skip ensemble for now
                    model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
                    if os.path.exists(model_path):
                        with open(model_path, 'rb') as f:
                            self.models[model_name] = pickle.load(f)
                        print(f"✅ {model_name} loaded (accuracy: {model_info['metrics']['accuracy']:.3f})")
            
            if len(self.models) > 0 and self.feature_pipeline is not None:
                self.is_ready = True
                print(f"🚀 Model integration ready! {len(self.models)} models loaded")
            else:
                print("❌ Model integration failed - missing components")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.is_ready = False
    
    def prepare_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """🔧 Prepare features from market data"""
        
        if not self.is_ready:
            raise ValueError("Model integration not ready")
        
        df = market_data.copy()
        
        # Create the same features as during training
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Technical indicators (simplified)
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_30'] = df['Close'].rolling(30).mean()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Simple momentum
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Volume features
        df['volume_ma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        
        # Price position
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Get features in the same order as training
        feature_columns = self.feature_pipeline['features']
        
        # Select the latest row with all features
        latest_data = df[feature_columns].iloc[-1:].values
        
        # Apply the same preprocessing as during training
        latest_data = self.feature_pipeline['imputer'].transform(latest_data)
        latest_data = self.feature_pipeline['scaler'].transform(latest_data)
        
        return latest_data
    
    def get_trend_prediction(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """📈 Get trend prediction from XGBoost model"""
        
        if 'xgboost_trend' not in self.models:
            return {'confidence': 0.0, 'direction': 'neutral', 'probability': 0.5}
        
        try:
            start_time = time.time()
            features = self.prepare_features(market_data)
            model = self.models['xgboost_trend']
            
            # Get prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            processing_time = (time.time() - start_time) * 1000  # milliseconds
            
            # Extract feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_names = self.feature_pipeline['features']
                for i, importance in enumerate(model.feature_importances_):
                    if i < len(feature_names):
                        feature_importance[feature_names[i]] = float(importance)
            
            # Log to AI monitoring system
            if self.ai_logger:
                self.ai_logger.log_model_prediction(
                    ModelType.XGBOOST,
                    float(probability[1]),  # Probability of bullish
                    float(max(probability)),  # Confidence
                    feature_importance,
                    processing_time,
                    self.model_info['xgboost_trend']['metrics']['accuracy']
                )
            
            result = {
                'confidence': float(max(probability)),
                'direction': 'bullish' if prediction == 1 else 'bearish',
                'probability': float(probability[1]),  # Probability of bullish
                'model': 'xgboost_trend',
                'accuracy': self.model_info['xgboost_trend']['metrics']['accuracy'],
                'feature_importance': feature_importance,
                'processing_time': processing_time
            }
            
            return result
            
        except Exception as e:
            print(f"⚠️ Trend prediction error: {e}")
            return {'confidence': 0.0, 'direction': 'neutral', 'probability': 0.5}
    
    def get_volatility_prediction(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """📊 Get volatility prediction from LightGBM model"""
        
        if 'lightgbm_volatility' not in self.models:
            return {'confidence': 0.0, 'direction': 'neutral', 'probability': 0.5}
        
        try:
            start_time = time.time()
            features = self.prepare_features(market_data)
            model = self.models['lightgbm_volatility']
            
            # Get prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            processing_time = (time.time() - start_time) * 1000  # milliseconds
            
            # Extract feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_names = self.feature_pipeline['features']
                for i, importance in enumerate(model.feature_importances_):
                    if i < len(feature_names):
                        feature_importance[feature_names[i]] = float(importance)
            
            # Log to AI monitoring system
            if self.ai_logger:
                self.ai_logger.log_model_prediction(
                    ModelType.LIGHTGBM,
                    float(probability[1]),  # Probability of bullish
                    float(max(probability)),  # Confidence
                    feature_importance,
                    processing_time,
                    self.model_info['lightgbm_volatility']['metrics']['accuracy']
                )
            
            result = {
                'confidence': float(max(probability)),
                'direction': 'bullish' if prediction == 1 else 'bearish',
                'probability': float(probability[1]),
                'model': 'lightgbm_volatility',
                'accuracy': self.model_info['lightgbm_volatility']['metrics']['accuracy'],
                'feature_importance': feature_importance,
                'processing_time': processing_time
            }
            
            return result
            
        except Exception as e:
            print(f"⚠️ Volatility prediction error: {e}")
            return {'confidence': 0.0, 'direction': 'neutral', 'probability': 0.5}
    
    def get_risk_assessment(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """🛡️ Get risk assessment from Random Forest model"""
        
        if 'random_forest_risk' not in self.models:
            return {'risk_level': 0.5, 'confidence': 0.0}
        
        try:
            start_time = time.time()
            features = self.prepare_features(market_data)
            model = self.models['random_forest_risk']
            
            # Get prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            processing_time = (time.time() - start_time) * 1000  # milliseconds
            
            # Extract feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_names = self.feature_pipeline['features']
                for i, importance in enumerate(model.feature_importances_):
                    if i < len(feature_names):
                        feature_importance[feature_names[i]] = float(importance)
            
            # Risk level: 0 = low risk (bearish), 1 = high risk (bullish)
            risk_level = float(probability[1])  # Probability of bullish (higher risk)
            
            # Log to AI monitoring system
            if self.ai_logger:
                self.ai_logger.log_model_prediction(
                    ModelType.RANDOM_FOREST,
                    float(probability[1]),  # Probability of bullish
                    float(max(probability)),  # Confidence
                    feature_importance,
                    processing_time,
                    self.model_info['random_forest_risk']['metrics']['accuracy']
                )
            
            result = {
                'risk_level': risk_level,
                'confidence': float(max(probability)),
                'direction': 'bullish' if prediction == 1 else 'bearish',
                'model': 'random_forest_risk',
                'accuracy': self.model_info['random_forest_risk']['metrics']['accuracy'],
                'feature_importance': feature_importance,
                'processing_time': processing_time
            }
            
            return result
            
        except Exception as e:
            print(f"⚠️ Risk assessment error: {e}")
            return {'risk_level': 0.5, 'confidence': 0.0}
    
    def get_ensemble_prediction(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """🎭 Get ensemble prediction from all models"""
        
        if not self.is_ready:
            return {
                'direction': 'neutral',
                'confidence': 0.0,
                'probability': 0.5,
                'models_active': 0
            }
        
        try:
            features = self.prepare_features(market_data)
            
            predictions = []
            probabilities = []
            model_results = {}
            
            # Get predictions from all models
            for model_name, model in self.models.items():
                try:
                    pred = model.predict(features)[0]
                    prob = model.predict_proba(features)[0][1]  # Probability of bullish
                    
                    predictions.append(pred)
                    probabilities.append(prob)
                    
                    model_results[model_name] = {
                        'prediction': int(pred),
                        'probability': float(prob),
                        'accuracy': self.model_info[model_name]['metrics']['accuracy']
                    }
                    
                except Exception as e:
                    print(f"⚠️ Error with {model_name}: {e}")
                    continue
            
            if len(predictions) == 0:
                return {
                    'direction': 'neutral',
                    'confidence': 0.0,
                    'probability': 0.5,
                    'models_active': 0
                }
            
            # Ensemble prediction: average probability
            ensemble_probability = np.mean(probabilities)
            ensemble_prediction = 1 if ensemble_probability > 0.5 else 0
            
            # Confidence: how much models agree
            confidence = 1.0 - np.std(probabilities)  # Lower std = higher confidence
            
            result = {
                'direction': 'bullish' if ensemble_prediction == 1 else 'bearish',
                'confidence': float(max(0.0, min(1.0, confidence))),
                'probability': float(ensemble_probability),
                'models_active': len(predictions),
                'model_results': model_results,
                'ensemble_strength': float(abs(ensemble_probability - 0.5) * 2)  # 0-1 scale
            }
            
            return result
            
        except Exception as e:
            print(f"⚠️ Ensemble prediction error: {e}")
            return {
                'direction': 'neutral',
                'confidence': 0.0,
                'probability': 0.5,
                'models_active': 0
            }
    
    def get_trading_signal(self, market_data: pd.DataFrame, 
                          confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """🎯 Get final trading signal with confidence filtering"""
        
        # Get ensemble prediction
        ensemble = self.get_ensemble_prediction(market_data)
        
        # Apply confidence threshold
        if ensemble['confidence'] < confidence_threshold:
            signal_strength = 'weak'
            action = 'hold'
        else:
            signal_strength = 'strong' if ensemble['confidence'] > 0.8 else 'medium'
            action = 'buy' if ensemble['direction'] == 'bullish' else 'sell'
        
        # Position sizing based on confidence
        position_size = min(1.0, ensemble['confidence'] * ensemble['ensemble_strength'])
        
        # Create comprehensive market conditions for AI monitoring
        if self.ai_logger and len(market_data) > 0:
            try:
                latest_data = market_data.iloc[-1]
                
                # Calculate additional market indicators
                volatility = market_data['Close'].pct_change().rolling(20).std().iloc[-1] if len(market_data) >= 20 else 0.02
                trend_strength = abs(market_data['Close'].pct_change(5).iloc[-1]) if len(market_data) >= 5 else 0.01
                
                # Create market conditions object
                market_conditions = MarketConditions(
                    symbol="BTC/USDT",  # Default, could be parameterized
                    price=float(latest_data['Close']),
                    volume=float(latest_data['Volume']),
                    volatility=float(volatility) if not np.isnan(volatility) else 0.02,
                    trend_strength=float(trend_strength) if not np.isnan(trend_strength) else 0.01,
                    support_level=float(market_data['Low'].rolling(20).min().iloc[-1]) if len(market_data) >= 20 else float(latest_data['Low']),
                    resistance_level=float(market_data['High'].rolling(20).max().iloc[-1]) if len(market_data) >= 20 else float(latest_data['High']),
                    rsi=self._calculate_rsi(market_data),
                    macd=self._calculate_macd(market_data),
                    bollinger_position=self._calculate_bollinger_position(market_data),
                    timestamp=datetime.now().isoformat()
                )
                
                # Collect individual model predictions for ensemble analysis
                individual_predictions = []
                
                # Get detailed predictions from each model
                trend_pred = self.get_trend_prediction(market_data)
                if trend_pred.get('feature_importance'):
                    from ai_analysis_monitor import ModelPrediction
                    individual_predictions.append(ModelPrediction(
                        model_type=ModelType.XGBOOST,
                        prediction=trend_pred['probability'],
                        confidence=trend_pred['confidence'],
                        feature_importance=trend_pred['feature_importance'],
                        processing_time=trend_pred.get('processing_time', 0.0),
                        accuracy_score=trend_pred['accuracy'],
                        timestamp=datetime.now().isoformat()
                    ))
                
                vol_pred = self.get_volatility_prediction(market_data)
                if vol_pred.get('feature_importance'):
                    individual_predictions.append(ModelPrediction(
                        model_type=ModelType.LIGHTGBM,
                        prediction=vol_pred['probability'],
                        confidence=vol_pred['confidence'],
                        feature_importance=vol_pred['feature_importance'],
                        processing_time=vol_pred.get('processing_time', 0.0),
                        accuracy_score=vol_pred['accuracy'],
                        timestamp=datetime.now().isoformat()
                    ))
                
                risk_pred = self.get_risk_assessment(market_data)
                if risk_pred.get('feature_importance'):
                    individual_predictions.append(ModelPrediction(
                        model_type=ModelType.RANDOM_FOREST,
                        prediction=risk_pred.get('probability', risk_pred['risk_level']),
                        confidence=risk_pred['confidence'],
                        feature_importance=risk_pred['feature_importance'],
                        processing_time=risk_pred.get('processing_time', 0.0),
                        accuracy_score=risk_pred['accuracy'],
                        timestamp=datetime.now().isoformat()
                    ))
                
                # Determine ensemble weights
                ensemble_weights = {
                    'xgboost_trend': 0.35,
                    'lightgbm_volatility': 0.30,
                    'random_forest_risk': 0.35
                }
                
                # Map action to decision type
                decision_map = {
                    'buy': DecisionType.BUY,
                    'sell': DecisionType.SELL,
                    'hold': DecisionType.HOLD
                }
                decision = decision_map.get(action, DecisionType.NEUTRAL)
                
                # Log complete ensemble analysis
                if individual_predictions:
                    self.ai_logger.log_ensemble_analysis(
                        individual_predictions=individual_predictions,
                        ensemble_weights=ensemble_weights,
                        final_prediction=ensemble['probability'],
                        final_confidence=ensemble['confidence'],
                        decision=decision,
                        market_conditions=market_conditions
                    )
                    
            except Exception as e:
                print(f"⚠️ AI monitoring logging error: {e}")
        
        result = {
            'action': action,
            'direction': ensemble['direction'],
            'confidence': ensemble['confidence'],
            'signal_strength': signal_strength,
            'position_size': float(position_size),
            'models_used': ensemble['models_active'],
            'probability': ensemble['probability'],
            'timestamp': datetime.now().isoformat(),
            'model_details': ensemble.get('model_results', {}),
            'ensemble_strength': ensemble.get('ensemble_strength', 0.0)
        }
        
        return result
    
    def _calculate_rsi(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """📊 Calculate RSI"""
        try:
            if len(market_data) < period + 1:
                return 50.0  # Neutral RSI
            
            delta = market_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_macd(self, market_data: pd.DataFrame, fast: int = 12, slow: int = 26) -> float:
        """📊 Calculate MACD"""
        try:
            if len(market_data) < slow:
                return 0.0
            
            exp1 = market_data['Close'].ewm(span=fast).mean()
            exp2 = market_data['Close'].ewm(span=slow).mean()
            macd = exp1 - exp2
            
            return float(macd.iloc[-1]) if not np.isnan(macd.iloc[-1]) else 0.0
        except:
            return 0.0
    
    def _calculate_bollinger_position(self, market_data: pd.DataFrame, period: int = 20) -> float:
        """📊 Calculate Bollinger Band Position"""
        try:
            if len(market_data) < period:
                return 0.5  # Middle position
            
            sma = market_data['Close'].rolling(window=period).mean()
            std = market_data['Close'].rolling(window=period).std()
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            current_price = market_data['Close'].iloc[-1]
            bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            
            return float(bb_position) if not np.isnan(bb_position) else 0.5
        except:
            return 0.5
    
    def get_ai_analysis_report(self) -> Optional[str]:
        """📊 Get latest AI analysis report"""
        if not self.ai_visualizer:
            return None
        
        try:
            recent_analyses = self.ai_logger.get_recent_analyses(1)
            if not recent_analyses:
                return "📊 Keine AI-Analysen verfügbar"
            
            latest_analysis = recent_analyses[0]
            
            # Mock market conditions for report
            mock_conditions = MarketConditions(
                symbol="BTC/USDT",
                price=45000.0,
                volume=1000000,
                volatility=0.03,
                trend_strength=0.5,
                support_level=44000.0,
                resistance_level=46000.0,
                rsi=50.0,
                macd=0.0,
                bollinger_position=0.5,
                timestamp=datetime.now().isoformat()
            )
            
            return self.ai_visualizer.generate_text_report(latest_analysis, mock_conditions)
        except Exception as e:
            return f"❌ Fehler beim Generieren des Reports: {e}"
    
    def get_ai_dashboard_data(self) -> Optional[Dict[str, Any]]:
        """📊 Get AI dashboard data"""
        if not self.ai_visualizer:
            return None
        
        try:
            return self.ai_visualizer.generate_dashboard_summary()
        except Exception as e:
            print(f"❌ Dashboard data error: {e}")
            return None
    
    def start_ai_monitoring(self, interval: int = 60):
        """🔄 Start AI monitoring display"""
        if self.ai_display:
            self.ai_display.start_monitoring(interval)
            print(f"🔄 AI monitoring started (interval: {interval}s)")
        else:
            print("⚠️ AI monitoring not available")
    
    def stop_ai_monitoring(self):
        """🛑 Stop AI monitoring"""
        if self.ai_display:
            self.ai_display.stop_monitoring()
            print("🛑 AI monitoring stopped")
    
    def export_ai_analysis_for_telegram(self) -> Optional[Dict[str, Any]]:
        """📱 Export AI analysis data for Telegram bot"""
        if not self.ai_logger:
            return None
        
        try:
            dashboard = self.get_ai_dashboard_data()
            recent_analyses = self.ai_logger.get_recent_analyses(3)
            
            if not dashboard or not recent_analyses:
                return None
            
            # Format for Telegram
            telegram_data = {
                'summary': {
                    'total_analyses': dashboard.get('total_analyses', 0),
                    'avg_confidence': dashboard.get('avg_confidence', 0.0),
                    'avg_agreement': dashboard.get('avg_agreement', 0.0),
                    'confidence_trend': dashboard.get('confidence_trend', 'unknown')
                },
                'latest_decision': {
                    'action': recent_analyses[0].decision.value if recent_analyses else 'unknown',
                    'confidence': recent_analyses[0].final_confidence if recent_analyses else 0.0,
                    'agreement': recent_analyses[0].agreement_score if recent_analyses else 0.0,
                    'timestamp': recent_analyses[0].timestamp if recent_analyses else '',
                    'top_feature': max(recent_analyses[0].dominant_features.items(), 
                                     key=lambda x: x[1])[0] if recent_analyses and recent_analyses[0].dominant_features else 'unknown'
                },
                'model_performance': dashboard.get('performance', {}).get('models', {}),
                'top_features': dashboard.get('top_features', {}),
                'recent_decisions': dashboard.get('recent_decisions', {})
            }
            
            return telegram_data
            
        except Exception as e:
            print(f"❌ Telegram export error: {e}")
            return None
    
    def get_model_status(self) -> Dict[str, Any]:
        """📊 Get status of all models"""
        
        status = {
            'is_ready': self.is_ready,
            'models_loaded': len(self.models),
            'models_available': list(self.models.keys()),
            'feature_count': len(self.feature_pipeline['features']) if self.feature_pipeline else 0,
            'model_accuracies': {}
        }
        
        for model_name, info in self.model_info.items():
            if 'metrics' in info:
                status['model_accuracies'][model_name] = info['metrics']['accuracy']
        
        return status
    
    def test_prediction(self, test_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """🧪 Test prediction with sample data"""
        
        if test_data is None:
            # Create sample test data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
            test_data = pd.DataFrame({
                'Datetime': dates,
                'Open': 50000 + np.random.randn(100) * 1000,
                'High': 50100 + np.random.randn(100) * 1000,
                'Low': 49900 + np.random.randn(100) * 1000,
                'Close': 50000 + np.random.randn(100) * 1000,
                'Volume': 1000000 + np.random.randn(100) * 100000
            })
        
        try:
            signal = self.get_trading_signal(test_data)
            
            print("🧪 MODEL TEST RESULTS")
            print("=" * 40)
            print(f"Action: {signal['action']}")
            print(f"Direction: {signal['direction']}")
            print(f"Confidence: {signal['confidence']:.3f}")
            print(f"Signal Strength: {signal['signal_strength']}")
            print(f"Position Size: {signal['position_size']:.3f}")
            print(f"Models Active: {signal['models_used']}")
            
            return signal
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return {'error': str(e)}

# Global instance for easy access
trained_models = TrainedModelIntegration()

def get_ai_trading_signal(market_data: pd.DataFrame) -> Dict[str, Any]:
    """🤖 Main function to get AI trading signal"""
    return trained_models.get_trading_signal(market_data)

def get_ai_model_status() -> Dict[str, Any]:
    """📊 Get AI model status"""
    return trained_models.get_model_status()

# Demo
if __name__ == "__main__":
    print("🤖 TRADINO AI MODEL INTEGRATION TEST")
    print("=" * 50)
    
    # Test the integration
    result = trained_models.test_prediction()
    
    # Show model status
    status = trained_models.get_model_status()
    print(f"\n📊 Model Status: {status}") 