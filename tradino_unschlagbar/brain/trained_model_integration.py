#!/usr/bin/env python3
"""
🧠 TRAINED MODEL INTEGRATION
Integration der trainierten ML Modelle in TRADINO UNSCHLAGBAR
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainedModelIntegration:
    """🤖 Integration der trainierten ML Modelle"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = {}
        self.feature_pipeline = None
        self.model_info = {}
        self.is_ready = False
        
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
            features = self.prepare_features(market_data)
            model = self.models['xgboost_trend']
            
            # Get prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            result = {
                'confidence': float(max(probability)),
                'direction': 'bullish' if prediction == 1 else 'bearish',
                'probability': float(probability[1]),  # Probability of bullish
                'model': 'xgboost_trend',
                'accuracy': self.model_info['xgboost_trend']['metrics']['accuracy']
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
            features = self.prepare_features(market_data)
            model = self.models['lightgbm_volatility']
            
            # Get prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            result = {
                'confidence': float(max(probability)),
                'direction': 'bullish' if prediction == 1 else 'bearish',
                'probability': float(probability[1]),
                'model': 'lightgbm_volatility',
                'accuracy': self.model_info['lightgbm_volatility']['metrics']['accuracy']
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
            features = self.prepare_features(market_data)
            model = self.models['random_forest_risk']
            
            # Get prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            # Risk level: 0 = low risk (bearish), 1 = high risk (bullish)
            risk_level = float(probability[1])  # Probability of bullish (higher risk)
            
            result = {
                'risk_level': risk_level,
                'confidence': float(max(probability)),
                'direction': 'bullish' if prediction == 1 else 'bearish',
                'model': 'random_forest_risk',
                'accuracy': self.model_info['random_forest_risk']['metrics']['accuracy']
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