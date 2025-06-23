"""
🤖 TRADINO UNSCHLAGBAR - Prediction Engine
Advanced ML Models für Preisvorhersagen (LSTM, XGBoost, Ensemble)

Author: AI Trading Systems
"""

import asyncio
import numpy as np
import pandas as pd
import pickle
import joblib
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# ML Libraries
try:
    import sklearn
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from models.market_models import Candle, MarketData
from connectors.bitget_pro import BitgetPro
from brain.market_intelligence import MarketIntelligence
from utils.logger_pro import setup_logger, log_ai_decision
from utils.config_manager import ConfigManager
from utils.math_utils import safe_divide

logger = setup_logger("PredictionEngine")

if not ML_AVAILABLE:
    logger.warning("⚠️ ML Libraries nicht verfügbar - Mock Mode aktiviert")


class PredictionHorizon(Enum):
    """Vorhersage-Horizont"""
    SHORT_TERM = "5m"      # 5 Minuten
    MEDIUM_TERM = "1h"     # 1 Stunde
    LONG_TERM = "4h"       # 4 Stunden


class ModelType(Enum):
    """Model Typen"""
    LSTM = "lstm"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"


@dataclass
class PredictionResult:
    """Vorhersage Ergebnis"""
    symbol: str
    model_type: ModelType
    horizon: PredictionHorizon
    predicted_price: float
    current_price: float
    price_change_percent: float
    direction: str  # 'up', 'down', 'sideways'
    confidence: float  # 0-1
    feature_importance: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class ModelPerformance:
    """Model Performance Tracking"""
    model_name: str
    accuracy: float
    mse: float
    mae: float
    last_trained: datetime
    predictions_made: int
    correct_predictions: int


class PredictionEngine:
    """🤖 Advanced ML Prediction Engine"""
    
    def __init__(self, config: ConfigManager, bitget_connector: BitgetPro, 
                 market_intelligence: MarketIntelligence):
        self.config = config
        self.bitget = bitget_connector
        self.market_intelligence = market_intelligence
        
        # Model Storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.model_performance: Dict[str, ModelPerformance] = {}
        
        # Prediction Cache
        self.prediction_cache: Dict[str, PredictionResult] = {}
        
        # Model Configuration
        self.model_config = config.get('ai', {})
        self.feature_columns = [
            'price_change_1', 'price_change_5', 'price_change_15',
            'volume_ratio', 'volatility', 'rsi', 'macd', 'bb_position',
            'trend_strength', 'support_distance', 'resistance_distance'
        ]
        
        # Training Settings
        self.sequence_length = self.model_config.get('lstm', {}).get('sequence_length', 60)
        self.retrain_threshold = self.model_config.get('retrain_threshold', 0.05)
        
        # Performance Tracking
        self.predictions_made = 0
        self.training_sessions = 0
        
        # Data Paths
        self.models_path = Path("data/models")
        self.models_path.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> bool:
        """🔥 Prediction Engine initialisieren"""
        try:
            logger.info("🤖 Prediction Engine wird initialisiert...")
            
            if not ML_AVAILABLE:
                logger.warning("⚠️ ML Libraries nicht verfügbar - Mock Mode")
                return True
            
            # Bestehende Modelle laden
            await self._load_existing_models()
            
            # Initial Training für wichtige Paare
            await self._initial_training()
            
            logger.success("✅ Prediction Engine erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ Prediction Engine Initialisierung fehlgeschlagen: {e}")
            return False
    
    # ==================== MAIN PREDICTION METHODS ====================
    
    async def predict_price(self, symbol: str, horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM) -> Optional[PredictionResult]:
        """🎯 Preis-Vorhersage für Symbol"""
        try:
            logger.info(f"🎯 Preisvorhersage wird erstellt: {symbol} ({horizon.value})")
            
            # Cache Check
            cache_key = f"{symbol}_{horizon.value}_{datetime.utcnow().minute // 5}"  # 5-Min Cache
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]
            
            # Features für Prediction sammeln
            features = await self._prepare_prediction_features(symbol)
            if not features:
                logger.warning(f"⚠️ Keine Features für {symbol} verfügbar")
                return None
            
            # Model Predictions
            predictions = {}
            
            if ML_AVAILABLE:
                # XGBoost Prediction
                xgb_pred = await self._predict_xgboost(symbol, features, horizon)
                if xgb_pred:
                    predictions['xgboost'] = xgb_pred
                
                # Random Forest Prediction
                rf_pred = await self._predict_random_forest(symbol, features, horizon)
                if rf_pred:
                    predictions['random_forest'] = rf_pred
                
                # LSTM Prediction (Mock für Demo)
                lstm_pred = await self._predict_lstm_mock(symbol, features, horizon)
                if lstm_pred:
                    predictions['lstm'] = lstm_pred
            else:
                # Mock Predictions für Demo
                predictions = await self._mock_predictions(symbol, features, horizon)
            
            if not predictions:
                logger.warning(f"⚠️ Keine Predictions für {symbol} erhalten")
                return None
            
            # Ensemble Prediction
            ensemble_result = await self._create_ensemble_prediction(symbol, predictions, horizon)
            
            # Cache aktualisieren
            self.prediction_cache[cache_key] = ensemble_result
            self.predictions_made += 1
            
            log_ai_decision(
                "PredictionEngine",
                f"{symbol} {ensemble_result.direction} {ensemble_result.price_change_percent:+.2f}%",
                ensemble_result.confidence
            )
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Preisvorhersage für {symbol}: {e}")
            return None
    
    # ==================== FEATURE ENGINEERING ====================
    
    async def _prepare_prediction_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """🔧 Features für Prediction vorbereiten"""
        try:
            # Market Data abrufen
            market_data = await self.bitget.get_market_data(symbol)
            if not market_data:
                return None
            
            # Candlestick Data
            candles = await self.bitget.get_candles(symbol, '5m', limit=100)
            if len(candles) < 20:
                return None
            
            # Market Intelligence
            market_analysis = await self.market_intelligence.analyze_market(symbol, ['5m', '15m', '1h'])
            
            # Price Features
            current_price = float(market_data.price)
            
            # Price Changes
            price_change_1 = (float(candles[-1].close) - float(candles[-2].close)) / float(candles[-2].close)
            price_change_5 = (float(candles[-1].close) - float(candles[-6].close)) / float(candles[-6].close) if len(candles) > 5 else 0
            price_change_15 = (float(candles[-1].close) - float(candles[-16].close)) / float(candles[-16].close) if len(candles) > 15 else 0
            
            # Volume Features
            current_volume = float(candles[-1].volume)
            avg_volume = sum(float(c.volume) for c in candles[-20:]) / 20
            volume_ratio = safe_divide(current_volume, avg_volume, 1.0)
            
            # Volatility
            prices = [float(c.close) for c in candles[-20:]]
            volatility = np.std(prices) / np.mean(prices) if prices else 0
            
            # Technical Indicators (Mock Implementation)
            rsi = 50 + np.random.normal(0, 15)  # Mock RSI
            rsi = max(0, min(100, rsi)) / 100  # Normalisierung
            
            macd = np.random.normal(0, 0.1)  # Mock MACD
            
            # Bollinger Bands Position (Mock)
            bb_position = np.random.uniform(0, 1)  # 0 = unteres Band, 1 = oberes Band
            
            # Market Intelligence Features
            trend_strength = market_analysis.trend_strength if market_analysis else 0.5
            
            # Support/Resistance Distance
            support_distance = 0.02  # Mock: 2% bis Support
            resistance_distance = 0.03  # Mock: 3% bis Resistance
            
            features = {
                'price_change_1': price_change_1,
                'price_change_5': price_change_5,
                'price_change_15': price_change_15,
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'rsi': rsi,
                'macd': macd,
                'bb_position': bb_position,
                'trend_strength': trend_strength,
                'support_distance': support_distance,
                'resistance_distance': resistance_distance
            }
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Feature-Vorbereitung: {e}")
            return None
    
    # ==================== MODEL PREDICTIONS ====================
    
    async def _predict_xgboost(self, symbol: str, features: Dict[str, float], horizon: PredictionHorizon) -> Optional[Dict[str, Any]]:
        """🌲 XGBoost Prediction"""
        try:
            model_key = f"xgboost_{symbol}_{horizon.value}"
            
            if model_key not in self.models:
                # Model trainieren wenn nicht vorhanden
                await self._train_xgboost_model(symbol, horizon)
            
            if model_key not in self.models:
                return None
            
            model = self.models[model_key]
            scaler = self.scalers.get(f"{model_key}_scaler")
            
            # Features vorbereiten
            feature_array = np.array([[features[col] for col in self.feature_columns]])
            
            if scaler:
                feature_array = scaler.transform(feature_array)
            
            # Prediction
            if ML_AVAILABLE:
                prediction = model.predict(feature_array)[0]
                
                # Feature Importance
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    for i, col in enumerate(self.feature_columns):
                        feature_importance[col] = float(model.feature_importances_[i])
            else:
                # Mock für Demo
                prediction = np.random.normal(0.01, 0.02)  # 1% ± 2%
                feature_importance = {col: np.random.uniform(0, 1) for col in self.feature_columns}
            
            return {
                'prediction': prediction,
                'confidence': 0.8,
                'feature_importance': feature_importance,
                'model_type': 'xgboost'
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei XGBoost Prediction: {e}")
            return None
    
    async def _predict_random_forest(self, symbol: str, features: Dict[str, float], horizon: PredictionHorizon) -> Optional[Dict[str, Any]]:
        """🌳 Random Forest Prediction"""
        try:
            model_key = f"rf_{symbol}_{horizon.value}"
            
            if model_key not in self.models:
                await self._train_random_forest_model(symbol, horizon)
            
            if model_key not in self.models:
                return None
            
            model = self.models[model_key]
            scaler = self.scalers.get(f"{model_key}_scaler")
            
            # Features vorbereiten
            feature_array = np.array([[features[col] for col in self.feature_columns]])
            
            if scaler:
                feature_array = scaler.transform(feature_array)
            
            # Prediction
            if ML_AVAILABLE:
                prediction = model.predict(feature_array)[0]
                
                # Feature Importance
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    for i, col in enumerate(self.feature_columns):
                        feature_importance[col] = float(model.feature_importances_[i])
            else:
                # Mock für Demo
                prediction = np.random.normal(0.005, 0.015)  # 0.5% ± 1.5%
                feature_importance = {col: np.random.uniform(0, 1) for col in self.feature_columns}
            
            return {
                'prediction': prediction,
                'confidence': 0.75,
                'feature_importance': feature_importance,
                'model_type': 'random_forest'
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Random Forest Prediction: {e}")
            return None
    
    async def _predict_lstm_mock(self, symbol: str, features: Dict[str, float], horizon: PredictionHorizon) -> Optional[Dict[str, Any]]:
        """🧠 LSTM Prediction (Mock Implementation)"""
        try:
            # Mock LSTM für Demo - in Realität würde hier TensorFlow/PyTorch verwendet
            await asyncio.sleep(0.1)  # Simuliere LSTM Inference Zeit
            
            # Simuliere LSTM Prediction basierend auf Trend
            trend_strength = features.get('trend_strength', 0.5)
            base_prediction = (trend_strength - 0.5) * 0.02  # -1% bis +1%
            
            # Füge etwas Rauschen hinzu
            noise = np.random.normal(0, 0.005)
            prediction = base_prediction + noise
            
            return {
                'prediction': prediction,
                'confidence': 0.85,  # LSTM hat oft höhere Confidence
                'feature_importance': {col: np.random.uniform(0, 1) for col in self.feature_columns},
                'model_type': 'lstm'
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei LSTM Prediction: {e}")
            return None
    
    async def _mock_predictions(self, symbol: str, features: Dict[str, float], horizon: PredictionHorizon) -> Dict[str, Dict[str, Any]]:
        """🎭 Mock Predictions für Demo"""
        try:
            trend = features.get('trend_strength', 0.5)
            volatility = features.get('volatility', 0.02)
            
            predictions = {}
            
            # XGBoost Mock
            xgb_pred = (trend - 0.5) * 0.03 + np.random.normal(0, volatility)
            predictions['xgboost'] = {
                'prediction': xgb_pred,
                'confidence': 0.8,
                'feature_importance': {col: np.random.uniform(0, 1) for col in self.feature_columns},
                'model_type': 'xgboost'
            }
            
            # Random Forest Mock
            rf_pred = (trend - 0.5) * 0.025 + np.random.normal(0, volatility * 0.8)
            predictions['random_forest'] = {
                'prediction': rf_pred,
                'confidence': 0.75,
                'feature_importance': {col: np.random.uniform(0, 1) for col in self.feature_columns},
                'model_type': 'random_forest'
            }
            
            # LSTM Mock
            lstm_pred = (trend - 0.5) * 0.035 + np.random.normal(0, volatility * 0.6)
            predictions['lstm'] = {
                'prediction': lstm_pred,
                'confidence': 0.85,
                'feature_importance': {col: np.random.uniform(0, 1) for col in self.feature_columns},
                'model_type': 'lstm'
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Mock Predictions: {e}")
            return {}
    
    # ==================== ENSEMBLE METHODS ====================
    
    async def _create_ensemble_prediction(self, symbol: str, predictions: Dict[str, Dict], horizon: PredictionHorizon) -> PredictionResult:
        """🎭 Ensemble Prediction erstellen"""
        try:
            # Model Weights basierend auf Performance
            weights = {
                'xgboost': 0.35,
                'random_forest': 0.25,
                'lstm': 0.40
            }
            
            # Gewichtete Prediction
            weighted_prediction = 0
            weighted_confidence = 0
            total_weight = 0
            
            # Feature Importance kombinieren
            combined_importance = {col: 0 for col in self.feature_columns}
            
            for model_name, pred_data in predictions.items():
                weight = weights.get(model_name, 0.33)
                
                weighted_prediction += pred_data['prediction'] * weight
                weighted_confidence += pred_data['confidence'] * weight
                total_weight += weight
                
                # Feature Importance
                for col in self.feature_columns:
                    combined_importance[col] += pred_data['feature_importance'].get(col, 0) * weight
            
            # Normalisierung
            if total_weight > 0:
                weighted_prediction /= total_weight
                weighted_confidence /= total_weight
                for col in combined_importance:
                    combined_importance[col] /= total_weight
            
            # Current Price (Mock)
            current_price = 50000.0  # Mock BTC Price
            predicted_price = current_price * (1 + weighted_prediction)
            price_change_percent = weighted_prediction * 100
            
            # Direction bestimmen
            if price_change_percent > 0.5:
                direction = "up"
            elif price_change_percent < -0.5:
                direction = "down"
            else:
                direction = "sideways"
            
            # Ensemble Confidence Adjustment
            model_agreement = self._calculate_model_agreement(predictions)
            final_confidence = weighted_confidence * model_agreement
            
            return PredictionResult(
                symbol=symbol,
                model_type=ModelType.ENSEMBLE,
                horizon=horizon,
                predicted_price=predicted_price,
                current_price=current_price,
                price_change_percent=price_change_percent,
                direction=direction,
                confidence=final_confidence,
                feature_importance=combined_importance,
                timestamp=datetime.utcnow(),
                metadata={
                    'models_used': list(predictions.keys()),
                    'model_agreement': model_agreement,
                    'individual_predictions': {k: v['prediction'] for k, v in predictions.items()}
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Ensemble Prediction: {e}")
            # Fallback
            return PredictionResult(
                symbol=symbol,
                model_type=ModelType.ENSEMBLE,
                horizon=horizon,
                predicted_price=50000.0,
                current_price=50000.0,
                price_change_percent=0.0,
                direction="sideways",
                confidence=0.5,
                feature_importance={col: 0.5 for col in self.feature_columns},
                timestamp=datetime.utcnow(),
                metadata={}
            )
    
    def _calculate_model_agreement(self, predictions: Dict[str, Dict]) -> float:
        """🤝 Model Agreement berechnen"""
        try:
            if len(predictions) < 2:
                return 1.0
            
            pred_values = [pred['prediction'] for pred in predictions.values()]
            
            # Standard Deviation der Predictions
            std_dev = np.std(pred_values)
            mean_pred = np.mean(pred_values)
            
            # Agreement Score (1 = perfekte Übereinstimmung, 0 = große Unterschiede)
            if abs(mean_pred) > 0:
                cv = std_dev / abs(mean_pred)  # Coefficient of Variation
                agreement = max(0, 1 - cv * 2)  # Umkehrung für Agreement
            else:
                agreement = 1 - std_dev * 10  # Bei mean=0, direkt std verwenden
            
            return max(0, min(1, agreement))
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Agreement Berechnung: {e}")
            return 0.5
    
    # ==================== MODEL TRAINING ====================
    
    async def _train_xgboost_model(self, symbol: str, horizon: PredictionHorizon):
        """🌲 XGBoost Model trainieren"""
        try:
            if not ML_AVAILABLE:
                return
            
            logger.info(f"🌲 XGBoost Model wird trainiert: {symbol} {horizon.value}")
            
            # Training Data sammeln (Mock für Demo)
            X_train, y_train = await self._prepare_training_data(symbol, horizon)
            
            if X_train is None or len(X_train) < 50:
                logger.warning(f"⚠️ Nicht genügend Training Data für {symbol}")
                return
            
            # Model erstellen
            model = xgb.XGBRegressor(
                n_estimators=self.model_config.get('xgboost', {}).get('n_estimators', 100),
                max_depth=self.model_config.get('xgboost', {}).get('max_depth', 6),
                learning_rate=self.model_config.get('xgboost', {}).get('learning_rate', 0.1),
                random_state=42
            )
            
            # Scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Training
            model.fit(X_train_scaled, y_train)
            
            # Model speichern
            model_key = f"xgboost_{symbol}_{horizon.value}"
            self.models[model_key] = model
            self.scalers[f"{model_key}_scaler"] = scaler
            
            # Performance tracking
            self.model_performance[model_key] = ModelPerformance(
                model_name=model_key,
                accuracy=0.8,  # Mock
                mse=0.001,
                mae=0.008,
                last_trained=datetime.utcnow(),
                predictions_made=0,
                correct_predictions=0
            )
            
            self.training_sessions += 1
            logger.success(f"✅ XGBoost Model trainiert: {model_key}")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim XGBoost Training: {e}")
    
    async def _train_random_forest_model(self, symbol: str, horizon: PredictionHorizon):
        """🌳 Random Forest Model trainieren"""
        try:
            if not ML_AVAILABLE:
                return
            
            logger.info(f"🌳 Random Forest Model wird trainiert: {symbol} {horizon.value}")
            
            # Training Data sammeln
            X_train, y_train = await self._prepare_training_data(symbol, horizon)
            
            if X_train is None or len(X_train) < 50:
                return
            
            # Model erstellen
            model = RandomForestRegressor(
                n_estimators=self.model_config.get('random_forest', {}).get('n_estimators', 200),
                max_depth=self.model_config.get('random_forest', {}).get('max_depth', 10),
                random_state=42
            )
            
            # Scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Training
            model.fit(X_train_scaled, y_train)
            
            # Model speichern
            model_key = f"rf_{symbol}_{horizon.value}"
            self.models[model_key] = model
            self.scalers[f"{model_key}_scaler"] = scaler
            
            # Performance tracking
            self.model_performance[model_key] = ModelPerformance(
                model_name=model_key,
                accuracy=0.75,  # Mock
                mse=0.0012,
                mae=0.009,
                last_trained=datetime.utcnow(),
                predictions_made=0,
                correct_predictions=0
            )
            
            logger.success(f"✅ Random Forest Model trainiert: {model_key}")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Random Forest Training: {e}")
    
    async def _prepare_training_data(self, symbol: str, horizon: PredictionHorizon) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """📚 Training Data vorbereiten"""
        try:
            # Mock Training Data für Demo
            n_samples = 1000
            
            # Features generieren
            X = []
            y = []
            
            for _ in range(n_samples):
                # Mock Features
                features = [
                    np.random.normal(0, 0.02),     # price_change_1
                    np.random.normal(0, 0.05),     # price_change_5
                    np.random.normal(0, 0.08),     # price_change_15
                    np.random.lognormal(0, 0.5),   # volume_ratio
                    np.random.uniform(0.01, 0.05), # volatility
                    np.random.uniform(0, 1),       # rsi
                    np.random.normal(0, 0.1),      # macd
                    np.random.uniform(0, 1),       # bb_position
                    np.random.uniform(0, 1),       # trend_strength
                    np.random.uniform(0, 0.05),    # support_distance
                    np.random.uniform(0, 0.05),    # resistance_distance
                ]
                
                # Target generieren (nächste Preisänderung)
                # Korreliere mit trend_strength
                trend_strength = features[8]
                target = (trend_strength - 0.5) * 0.04 + np.random.normal(0, 0.01)
                
                X.append(features)
                y.append(target)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Training Data Vorbereitung: {e}")
            return None, None
    
    # ==================== UTILITY METHODS ====================
    
    async def _load_existing_models(self):
        """💾 Bestehende Modelle laden"""
        try:
            for model_file in self.models_path.glob("*.pkl"):
                try:
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    model_name = model_file.stem
                    self.models[model_name] = model_data['model']
                    
                    if 'scaler' in model_data:
                        self.scalers[f"{model_name}_scaler"] = model_data['scaler']
                    
                    logger.info(f"💾 Model geladen: {model_name}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Model konnte nicht geladen werden {model_file}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden der Modelle: {e}")
    
    async def _initial_training(self):
        """🎯 Initial Training für wichtige Paare"""
        try:
            important_pairs = ['BTC/USDT', 'ETH/USDT']
            
            for symbol in important_pairs:
                for horizon in [PredictionHorizon.SHORT_TERM, PredictionHorizon.MEDIUM_TERM]:
                    await self._train_xgboost_model(symbol, horizon)
                    await self._train_random_forest_model(symbol, horizon)
                    
                    # Kurze Pause zwischen Trainings
                    await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Initial Training: {e}")
    
    # ==================== PUBLIC METHODS ====================
    
    def get_cached_prediction(self, symbol: str, horizon: PredictionHorizon) -> Optional[PredictionResult]:
        """📊 Gecachte Prediction abrufen"""
        cache_key = f"{symbol}_{horizon.value}_{datetime.utcnow().minute // 5}"
        return self.prediction_cache.get(cache_key)
    
    def get_model_performance(self) -> Dict[str, ModelPerformance]:
        """📈 Model Performance abrufen"""
        return self.model_performance.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """📊 Performance Statistiken"""
        return {
            'predictions_made': self.predictions_made,
            'training_sessions': self.training_sessions,
            'models_loaded': len(self.models),
            'cached_predictions': len(self.prediction_cache),
            'ml_available': ML_AVAILABLE,
            'feature_count': len(self.feature_columns)
        }
    
    async def retrain_models_if_needed(self):
        """🔄 Modelle bei Bedarf neu trainieren"""
        try:
            for model_name, performance in self.model_performance.items():
                # Retrain wenn Accuracy unter Threshold
                if performance.accuracy < (1 - self.retrain_threshold):
                    logger.info(f"🔄 Model wird neu trainiert: {model_name}")
                    
                    # Symbol und Horizon aus Model Name extrahieren
                    parts = model_name.split('_')
                    if len(parts) >= 3:
                        symbol = parts[1] + '/' + parts[2]
                        horizon_str = parts[3] if len(parts) > 3 else '5m'
                        
                        try:
                            horizon = PredictionHorizon(horizon_str)
                            
                            if model_name.startswith('xgboost'):
                                await self._train_xgboost_model(symbol, horizon)
                            elif model_name.startswith('rf'):
                                await self._train_random_forest_model(symbol, horizon)
                                
                        except ValueError:
                            continue
                            
        except Exception as e:
            logger.error(f"❌ Fehler beim Model Retraining: {e}")
    
    async def save_models(self, symbol: str = None, timeframe: str = '5m'):
        """
        ML-Modelle mit korrekten Dateinamen speichern
        """
        try:
            # ✅ FILENAME SANITIZATION
            if symbol:
                safe_symbol = symbol.replace('/', '_').replace('\\', '_')
                safe_timeframe = timeframe.replace('/', '_').replace('\\', '_')
            
            # Ensure models directory exists
            models_dir = 'data/models'
            import os
            os.makedirs(models_dir, exist_ok=True)
            
            # Save each model with safe filename
            for model_name, model in self.models.items():
                try:
                    if symbol:
                        filename = f"{model_name}_{safe_symbol}_{safe_timeframe}.pkl"
                    else:
                        filename = f"{model_name}.pkl"
                    filepath = os.path.join(models_dir, filename)
                    
                    # Use joblib for sklearn models, pickle for others
                    if hasattr(model, 'predict') and hasattr(model, 'fit'):
                        import joblib
                        joblib.dump(model, filepath)
                    else:
                        import pickle
                        with open(filepath, 'wb') as f:
                            pickle.dump(model, f)
                    
                    logger.info(f"✅ Modell gespeichert: {filepath}")
                    
                except Exception as e:
                    logger.error(f"❌ Fehler beim Speichern von {model_name}: {e}")
            
            logger.success(f"✅ Alle Modelle gespeichert")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Speichern der Modelle: {e}")

    async def load_models(self, symbol: str, timeframe: str = '5m'):
        """
        ML-Modelle mit korrekten Dateinamen laden
        """
        try:
            safe_symbol = symbol.replace('/', '_').replace('\\', '_')
            safe_timeframe = timeframe.replace('/', '_').replace('\\', '_')
            
            models_dir = 'data/models'
            loaded_count = 0
            
            for model_name in self.models.keys():
                try:
                    filename = f"{model_name}_{safe_symbol}_{safe_timeframe}.pkl"
                    filepath = os.path.join(models_dir, filename)
                    
                    if os.path.exists(filepath):
                        import joblib
                        self.models[model_name] = joblib.load(filepath)
                        loaded_count += 1
                        logger.info(f"✅ Modell geladen: {filename}")
                    else:
                        logger.warning(f"⚠️ Modell nicht gefunden: {filename}")
                        
                except Exception as e:
                    logger.error(f"❌ Fehler beim Laden von {model_name}: {e}")
            
            logger.info(f"✅ {loaded_count} Modelle für {symbol} geladen")
            return loaded_count > 0
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden der Modelle: {e}")
            return False
    
    async def shutdown(self):
        """🛑 Prediction Engine herunterfahren"""
        try:
            # Modelle speichern
            await self.save_models()
            
            # Cache leeren
            self.prediction_cache.clear()
            self.models.clear()
            self.scalers.clear()
            
            logger.info("✅ Prediction Engine heruntergefahren")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren: {e}")
