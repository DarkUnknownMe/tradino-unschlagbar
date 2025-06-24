#!/usr/bin/env python3
"""
🧠 ADAPTIVE AI PIPELINE
Enhanced AI-Pipeline mit modernsten ML-Techniken für TRADINO UNSCHLAGBAR

FEATURES:
- Online Learning Implementation mit River
- Model Drift Detection mit Adaptive Windows
- Feature Selection Optimization mit SHAP
- Ensemble Meta-Learning
- MLflow Experiment Tracking
- Optuna Hyperparameter Optimization
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Online Learning with River
try:
    from river import ensemble, tree, linear_model, metrics, preprocessing
    from river.drift import ADWIN
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

# Feature Importance with SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Hyperparameter Optimization with Optuna
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# MLflow Experiment Tracking
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing TRADINO components
try:
    from ai_analysis_monitor import ModelType, DecisionType, MarketConditions
    from trained_model_integration import TrainedModelIntegration
    TRADINO_COMPONENTS_AVAILABLE = True
except ImportError:
    TRADINO_COMPONENTS_AVAILABLE = False

@dataclass
class ModelPerformanceMetrics:
    """📊 Model Performance Tracking"""
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    training_time: float
    prediction_time: float
    drift_detected: bool
    last_update: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DriftDetectionResult:
    """📈 Drift Detection Results"""
    model_name: str
    drift_detected: bool
    drift_magnitude: float
    detection_method: str
    timestamp: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DriftDetector:
    """🔍 Model Drift Detection System"""
    
    def __init__(self, window_size: int = 1000, delta: float = 0.002):
        """
        Initialize Drift Detector
        
        Args:
            window_size: Size of sliding window for drift detection
            delta: Sensitivity parameter for ADWIN algorithm
        """
        self.window_size = window_size
        self.delta = delta
        self.drift_detectors: Dict[str, Any] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
        if RIVER_AVAILABLE:
            print("🔍 Drift Detector initialized with ADWIN algorithm")
        else:
            print("⚠️ River not available - using statistical drift detection")
    
    def initialize_detector(self, model_name: str) -> None:
        """Initialize drift detector for specific model"""
        if RIVER_AVAILABLE:
            self.drift_detectors[model_name] = ADWIN(delta=self.delta)
        else:
            self.drift_detectors[model_name] = {
                'window': [],
                'baseline_mean': None,
                'baseline_std': None
            }
        
        self.performance_history[model_name] = []
        print(f"🔍 Drift detector initialized for {model_name}")
    
    def detect_concept_drift(self, model_name: str, performance_score: float) -> DriftDetectionResult:
        """
        Detect concept drift in model performance
        
        Args:
            model_name: Name of the model
            performance_score: Current performance score (0-1)
            
        Returns:
            DriftDetectionResult with detection information
        """
        if model_name not in self.drift_detectors:
            self.initialize_detector(model_name)
        
        # Add to performance history
        self.performance_history[model_name].append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history[model_name]) > self.window_size:
            self.performance_history[model_name] = self.performance_history[model_name][-self.window_size:]
        
        drift_detected = False
        drift_magnitude = 0.0
        confidence = 0.0
        detection_method = "statistical"
        
        if RIVER_AVAILABLE:
            # Use ADWIN algorithm
            detector = self.drift_detectors[model_name]
            detector.update(performance_score)
            drift_detected = detector.drift_detected
            drift_magnitude = abs(performance_score - np.mean(self.performance_history[model_name][-10:]))
            confidence = 0.95 if drift_detected else 0.0
            detection_method = "ADWIN"
        
        else:
            # Statistical drift detection
            if len(self.performance_history[model_name]) >= 50:
                recent_scores = self.performance_history[model_name][-20:]
                historical_scores = self.performance_history[model_name][-50:-20]
                
                if len(historical_scores) > 0:
                    recent_mean = np.mean(recent_scores)
                    historical_mean = np.mean(historical_scores)
                    historical_std = np.std(historical_scores)
                    
                    # Calculate z-score
                    if historical_std > 0:
                        z_score = abs(recent_mean - historical_mean) / historical_std
                        drift_detected = z_score > 2.0  # 95% confidence
                        drift_magnitude = z_score
                        confidence = min(z_score / 3.0, 1.0)
        
        result = DriftDetectionResult(
            model_name=model_name,
            drift_detected=drift_detected,
            drift_magnitude=drift_magnitude,
            detection_method=detection_method,
            timestamp=datetime.now().isoformat(),
            confidence=confidence
        )
        
        if drift_detected:
            print(f"⚠️ Concept drift detected in {model_name}! Magnitude: {drift_magnitude:.3f}")
        
        return result

class AdaptiveFeatureSelector:
    """🎯 Adaptive Feature Selection with SHAP"""
    
    def __init__(self, max_features: int = 20, selection_threshold: float = 0.01):
        """
        Initialize Adaptive Feature Selector
        
        Args:
            max_features: Maximum number of features to select
            selection_threshold: Minimum SHAP value threshold for feature selection
        """
        self.max_features = max_features
        self.selection_threshold = selection_threshold
        self.feature_importance_history: Dict[str, List[float]] = {}
        self.selected_features: List[str] = []
        self.shap_explainer = None
        
        if SHAP_AVAILABLE:
            print("🎯 Adaptive Feature Selector initialized with SHAP")
        else:
            print("⚠️ SHAP not available - using feature importance from models")
    
    def analyze_feature_importance(self, model: Any, X: np.ndarray, 
                                 feature_names: List[str]) -> Dict[str, float]:
        """
        Analyze feature importance using SHAP values
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        importance_scores = {}
        
        if SHAP_AVAILABLE:
            try:
                # Create SHAP explainer based on model type
                if hasattr(model, 'predict_proba'):
                    # For tree-based models
                    if hasattr(model, 'feature_importances_'):
                        self.shap_explainer = shap.TreeExplainer(model)
                    else:
                        # For other models, use KernelExplainer
                        self.shap_explainer = shap.KernelExplainer(
                            model.predict_proba, 
                            X[:100]  # Use subset for speed
                        )
                
                # Calculate SHAP values
                shap_values = self.shap_explainer.shap_values(X[:100])
                
                # If binary classification, use positive class SHAP values
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Calculate mean absolute SHAP values for each feature
                mean_shap_values = np.mean(np.abs(shap_values), axis=0)
                
                for i, feature in enumerate(feature_names):
                    if i < len(mean_shap_values):
                        importance_scores[feature] = float(mean_shap_values[i])
                
                print(f"✅ SHAP analysis completed for {len(feature_names)} features")
                
            except Exception as e:
                print(f"⚠️ SHAP analysis failed: {e}")
                # Fallback to model feature importance
                importance_scores = self._get_model_feature_importance(model, feature_names)
        
        else:
            # Use model's built-in feature importance
            importance_scores = self._get_model_feature_importance(model, feature_names)
        
        return importance_scores
    
    def _get_model_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model's built-in methods"""
        importance_scores = {}
        
        if hasattr(model, 'feature_importances_'):
            for i, feature in enumerate(feature_names):
                if i < len(model.feature_importances_):
                    importance_scores[feature] = float(model.feature_importances_[i])
        
        return importance_scores
    
    def optimize_feature_selection(self, importance_scores: Dict[str, float]) -> List[str]:
        """
        Optimize feature selection based on importance scores
        
        Args:
            importance_scores: Dictionary of feature importance scores
            
        Returns:
            List of selected feature names
        """
        # Update feature importance history
        for feature, score in importance_scores.items():
            if feature not in self.feature_importance_history:
                self.feature_importance_history[feature] = []
            self.feature_importance_history[feature].append(score)
            
            # Keep only recent history (last 100 updates)
            if len(self.feature_importance_history[feature]) > 100:
                self.feature_importance_history[feature] = self.feature_importance_history[feature][-100:]
        
        # Calculate rolling average importance
        avg_importance = {}
        for feature, scores in self.feature_importance_history.items():
            if len(scores) > 0:
                avg_importance[feature] = np.mean(scores)
        
        # Filter features above threshold
        filtered_features = {
            feature: score for feature, score in avg_importance.items() 
            if score >= self.selection_threshold
        }
        
        # Sort by importance and select top features
        sorted_features = sorted(filtered_features.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [feature for feature, _ in sorted_features[:self.max_features]]
        
        print(f"🎯 Selected {len(self.selected_features)} features from {len(importance_scores)} available")
        
        return self.selected_features

class MetaLearner:
    """🎭 Ensemble Meta-Learning System"""
    
    def __init__(self, meta_model_type: str = "xgboost"):
        """
        Initialize Meta-Learner
        
        Args:
            meta_model_type: Type of meta-model ("xgboost", "lightgbm", "random_forest")
        """
        self.meta_model_type = meta_model_type
        self.meta_model = None
        self.base_models: Dict[str, Any] = {}
        self.meta_training_data: List[Dict[str, Any]] = []
        self.ensemble_weights: Dict[str, float] = {}
        
        print(f"🎭 Meta-Learner initialized with {meta_model_type}")
    
    def add_base_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        """Add base model to ensemble"""
        self.base_models[name] = model
        self.ensemble_weights[name] = weight
        print(f"✅ Base model '{name}' added with weight {weight}")
    
    def collect_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Collect meta-features from base models
        
        Args:
            X: Input features
            
        Returns:
            Meta-features array
        """
        meta_features = []
        
        for name, model in self.base_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    # Get probability predictions
                    probas = model.predict_proba(X)
                    if probas.shape[1] == 2:  # Binary classification
                        meta_features.append(probas[:, 1])  # Probability of positive class
                    else:
                        meta_features.extend(probas.T)  # All class probabilities
                else:
                    # Get raw predictions
                    predictions = model.predict(X)
                    meta_features.append(predictions)
                    
            except Exception as e:
                print(f"⚠️ Error collecting meta-features from {name}: {e}")
                # Add zeros as fallback
                meta_features.append(np.zeros(X.shape[0]))
        
        if len(meta_features) == 0:
            return np.zeros((X.shape[0], 1))
        
        return np.column_stack(meta_features)
    
    def train_meta_model(self, X_meta: np.ndarray, y: np.ndarray) -> None:
        """
        Train meta-model on meta-features
        
        Args:
            X_meta: Meta-features from base models
            y: Target labels
        """
        if self.meta_model_type == "xgboost" and SKLEARN_AVAILABLE:
            self.meta_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        elif self.meta_model_type == "lightgbm" and SKLEARN_AVAILABLE:
            self.meta_model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1
            )
        elif self.meta_model_type == "random_forest" and SKLEARN_AVAILABLE:
            self.meta_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported meta-model type: {self.meta_model_type}")
        
        # Train meta-model
        self.meta_model.fit(X_meta, y)
        print(f"✅ Meta-model trained: {self.meta_model_type}")
    
    def predict_ensemble(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Make ensemble prediction using meta-learner
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, confidence)
        """
        if self.meta_model is None:
            # Fallback to weighted average
            return self._weighted_average_prediction(X)
        
        # Get meta-features
        X_meta = self.collect_meta_features(X)
        
        # Meta-model prediction
        predictions = self.meta_model.predict(X)
        probabilities = self.meta_model.predict_proba(X)
        
        # Calculate confidence as max probability
        confidence = np.max(probabilities, axis=1) if len(probabilities.shape) > 1 else np.max(probabilities)
        
        return predictions, float(confidence)
    
    def _weighted_average_prediction(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fallback weighted average prediction"""
        weighted_predictions = []
        total_weight = sum(self.ensemble_weights.values())
        
        for name, model in self.base_models.items():
            weight = self.ensemble_weights.get(name, 1.0) / total_weight
            try:
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X)
                    predictions = probas[:, 1] if probas.shape[1] == 2 else np.argmax(probas, axis=1)
                else:
                    predictions = model.predict(X)
                
                weighted_predictions.append(predictions * weight)
            except Exception as e:
                print(f"⚠️ Error in weighted prediction from {name}: {e}")
        
        if len(weighted_predictions) == 0:
            return np.zeros(X.shape[0]), 0.0
        
        final_predictions = np.sum(weighted_predictions, axis=0)
        confidence = 0.5  # Default confidence for weighted average
        
        return final_predictions, confidence 

class AdaptiveModelPipeline:
    """🧠 Main Adaptive AI Pipeline Class"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Adaptive Model Pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Core Components
        self.drift_detector = DriftDetector(
            window_size=self.config.get('drift_window_size', 1000),
            delta=self.config.get('drift_delta', 0.002)
        )
        self.feature_selector = AdaptiveFeatureSelector(
            max_features=self.config.get('max_features', 20),
            selection_threshold=self.config.get('feature_threshold', 0.01)
        )
        self.meta_learner = MetaLearner(
            meta_model_type=self.config.get('meta_model', 'xgboost')
        )
        
        # Online Learning Models
        self.online_learners: Dict[str, Any] = {}
        
        # Traditional Models (Backward Compatibility)
        self.traditional_models: Dict[str, Any] = {}
        
        # Performance Tracking
        self.model_performance: Dict[str, ModelPerformanceMetrics] = {}
        
        # MLflow Integration
        self.mlflow_experiment_name = self.config.get('mlflow_experiment', 'tradino_adaptive_ai')
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(self.mlflow_experiment_name)
            print("📊 MLflow experiment tracking enabled")
        
        # Integration with existing TRADINO components
        self.legacy_integration = None
        if TRADINO_COMPONENTS_AVAILABLE:
            try:
                self.legacy_integration = TrainedModelIntegration()
                print("🔗 Legacy model integration enabled")
            except Exception as e:
                print(f"⚠️ Legacy integration failed: {e}")
        
        print("🧠 Adaptive Model Pipeline initialized")
    
    def initialize_online_learners(self) -> None:
        """Initialize online learning models"""
        if not RIVER_AVAILABLE:
            print("⚠️ River not available - online learning disabled")
            return
        
        # Initialize different online learners with correct River API
        try:
            self.online_learners = {
                'hoeffding_tree': tree.HoeffdingTreeClassifier(
                    max_depth=5,
                    grace_period=200,
                    split_confidence=0.0001
                ),
                'logistic_regression': linear_model.LogisticRegression(),
                'passive_aggressive': linear_model.PAClassifier()
            }
            
            # Try to add adaptive random forest if available
            try:
                from river.ensemble import AdaptiveRandomForestClassifier
                self.online_learners['adaptive_random_forest'] = AdaptiveRandomForestClassifier(
                    n_models=10,
                    seed=42
                )
            except ImportError:
                # Fallback: use standard random forest from ensemble
                try:
                    self.online_learners['random_forest'] = ensemble.SRPClassifier(
                        n_models=10,
                        seed=42
                    )
                except (ImportError, AttributeError):
                    pass
            
        except Exception as e:
            print(f"⚠️ Error initializing some online learners: {e}")
            # Minimal fallback
            self.online_learners = {
                'hoeffding_tree': tree.HoeffdingTreeClassifier(),
                'logistic_regression': linear_model.LogisticRegression()
            }
        
        # Initialize drift detectors for each online learner
        for name in self.online_learners.keys():
            self.drift_detector.initialize_detector(f"online_{name}")
        
        print(f"🌊 {len(self.online_learners)} online learners initialized")
    
    def update_models_online(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Update models with new data using online learning
        
        Args:
            new_data: New market data
            
        Returns:
            Update results and performance metrics
        """
        if not RIVER_AVAILABLE or len(self.online_learners) == 0:
            return self._update_traditional_models(new_data)
        
        update_results = {}
        
        # Prepare features
        X, y, feature_names = self._prepare_online_features(new_data)
        
        # Update each online learner
        for name, model in self.online_learners.items():
            try:
                start_time = time.time()
                
                # Update model with new data point by point
                for i in range(len(X)):
                    x_dict = {feature_names[j]: X[i][j] for j in range(len(feature_names))}
                    model.learn_one(x_dict, y[i])
                
                training_time = time.time() - start_time
                
                # Make predictions to evaluate performance
                predictions = []
                for i in range(len(X)):
                    x_dict = {feature_names[j]: X[i][j] for j in range(len(feature_names))}
                    pred = model.predict_one(x_dict)
                    predictions.append(pred if pred is not None else 0)
                
                # Calculate performance metrics
                accuracy = accuracy_score(y, predictions) if len(predictions) > 0 else 0.0
                
                # Drift detection
                drift_result = self.drift_detector.detect_concept_drift(f"online_{name}", accuracy)
                
                # Update performance tracking
                self.model_performance[f"online_{name}"] = ModelPerformanceMetrics(
                    accuracy=accuracy,
                    f1_score=f1_score(y, predictions, average='weighted') if len(predictions) > 0 else 0.0,
                    precision=0.0,  # Not calculated for online learning
                    recall=0.0,     # Not calculated for online learning
                    training_time=training_time,
                    prediction_time=0.0,
                    drift_detected=drift_result.drift_detected,
                    last_update=datetime.now().isoformat()
                )
                
                update_results[name] = {
                    'samples_processed': len(X),
                    'accuracy': accuracy,
                    'training_time': training_time,
                    'drift_detected': drift_result.drift_detected
                }
                
                print(f"🌊 Online model '{name}' updated: {len(X)} samples, accuracy: {accuracy:.3f}")
                
            except Exception as e:
                print(f"❌ Error updating online model '{name}': {e}")
                update_results[name] = {'error': str(e)}
        
        return update_results
    
    def _prepare_online_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for online learning"""
        df = data.copy()
        
        # Create basic features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['returns'].rolling(10).std()
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(10).mean()
        df['price_momentum'] = df['Close'] / df['Close'].shift(5) - 1
        
        # Simple target: next period return direction
        df['target'] = (df['returns'].shift(-1) > 0).astype(int)
        
        # Select feature columns
        feature_columns = ['returns', 'log_returns', 'volatility', 'volume_ratio', 'price_momentum']
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Clean data
        df = df.dropna()
        
        if len(df) == 0:
            return np.array([]), np.array([]), available_features
        
        X = df[available_features].values
        y = df['target'].values
        
        return X, y, available_features
    
    def _update_traditional_models(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback method for updating traditional models"""
        return {'message': 'Online learning not available - traditional models maintained'}
    
    def detect_concept_drift(self, recent_performance: List[float]) -> Dict[str, DriftDetectionResult]:
        """
        Detect concept drift across all models
        
        Args:
            recent_performance: List of recent performance scores
            
        Returns:
            Dictionary of drift detection results for each model
        """
        drift_results = {}
        
        # Check drift for all tracked models
        for model_name in self.model_performance.keys():
            if model_name in recent_performance:
                performance_score = recent_performance[model_name]
            else:
                # Use latest performance if available
                performance_score = self.model_performance[model_name].accuracy
            
            drift_result = self.drift_detector.detect_concept_drift(model_name, performance_score)
            drift_results[model_name] = drift_result
            
            # If drift detected, trigger model retraining
            if drift_result.drift_detected:
                self._handle_drift_detection(model_name, drift_result)
        
        return drift_results
    
    def _handle_drift_detection(self, model_name: str, drift_result: DriftDetectionResult) -> None:
        """Handle detected concept drift"""
        print(f"🚨 Handling concept drift in {model_name}")
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE:
            with mlflow.start_run():
                mlflow.log_metric("drift_detected", 1)
                mlflow.log_metric("drift_magnitude", drift_result.drift_magnitude)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("detection_method", drift_result.detection_method)
        
        # Reset online learner if applicable
        if model_name.startswith("online_") and RIVER_AVAILABLE:
            learner_name = model_name.replace("online_", "")
            if learner_name in self.online_learners:
                # Reinitialize the online learner with correct River API
                try:
                    if learner_name == 'hoeffding_tree':
                        self.online_learners[learner_name] = tree.HoeffdingTreeClassifier(
                            max_depth=5, grace_period=200, split_confidence=0.0001
                        )
                    elif learner_name == 'logistic_regression':
                        self.online_learners[learner_name] = linear_model.LogisticRegression()
                    elif learner_name == 'passive_aggressive':
                        self.online_learners[learner_name] = linear_model.PAClassifier()
                    elif learner_name == 'adaptive_random_forest':
                        from river.ensemble import AdaptiveRandomForestClassifier
                        self.online_learners[learner_name] = AdaptiveRandomForestClassifier(
                            n_models=10, seed=42
                        )
                    elif learner_name == 'random_forest':
                        self.online_learners[learner_name] = ensemble.SRPClassifier(
                            n_models=10, seed=42
                        )
                    
                    print(f"🔄 Online learner '{learner_name}' reinitialized due to drift")
                    
                except Exception as e:
                    print(f"⚠️ Failed to reinitialize '{learner_name}': {e}")
    
    def optimize_feature_selection(self, market_data: pd.DataFrame) -> List[str]:
        """
        Optimize feature selection based on current market data
        
        Args:
            market_data: Current market data
            
        Returns:
            List of optimized feature names
        """
        # Use legacy integration if available
        if self.legacy_integration and self.legacy_integration.is_ready:
            try:
                # Get feature importance from existing models
                importance_scores = {}
                
                # Analyze each model's feature importance
                for model_name, model in self.legacy_integration.models.items():
                    model_importance = self.feature_selector.analyze_feature_importance(
                        model, 
                        self.legacy_integration.prepare_features(market_data),
                        self.legacy_integration.feature_pipeline['features']
                    )
                    
                    # Merge importance scores
                    for feature, score in model_importance.items():
                        if feature not in importance_scores:
                            importance_scores[feature] = 0.0
                        importance_scores[feature] += score
                
                # Normalize by number of models
                num_models = len(self.legacy_integration.models)
                if num_models > 0:
                    for feature in importance_scores:
                        importance_scores[feature] /= num_models
                
                # Optimize selection
                selected_features = self.feature_selector.optimize_feature_selection(importance_scores)
                
                return selected_features
                
            except Exception as e:
                print(f"⚠️ Feature selection optimization failed: {e}")
        
        # Fallback: return default features
        return ['returns', 'log_returns', 'volatility', 'volume_ratio', 'price_momentum']
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                model_type: str = "xgboost") -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to optimize
            
        Returns:
            Best hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not available - using default hyperparameters")
            return self._get_default_hyperparameters(model_type)
        
        def objective(trial):
            if model_type == "xgboost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                }
                model = xgb.XGBClassifier(**params)
            
            elif model_type == "lightgbm":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42,
                    'verbosity': -1
                }
                model = lgb.LGBMClassifier(**params)
            
            elif model_type == "random_forest":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Cross-validation score
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.get('optuna_trials', 50))
        
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"🎯 Hyperparameter optimization completed for {model_type}")
        print(f"   Best score: {best_score:.3f}")
        print(f"   Best params: {best_params}")
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE:
            with mlflow.start_run():
                mlflow.log_params(best_params)
                mlflow.log_metric("best_cv_score", best_score)
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("optimization_trials", len(study.trials))
        
        return best_params
    
    def _get_default_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for fallback"""
        defaults = {
            "xgboost": {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            "lightgbm": {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbosity': -1
            },
            "random_forest": {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        }
        
        return defaults.get(model_type, {})
    
    def get_ensemble_prediction(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get ensemble prediction combining all models
        
        Args:
            market_data: Current market data
            
        Returns:
            Ensemble prediction results
        """
        predictions = {}
        confidences = {}
        
        # Get predictions from legacy models (backward compatibility)
        if self.legacy_integration and self.legacy_integration.is_ready:
            try:
                legacy_signal = self.legacy_integration.get_trading_signal(market_data)
                predictions['legacy_ensemble'] = legacy_signal.get('probability', 0.5)
                confidences['legacy_ensemble'] = legacy_signal.get('confidence', 0.0)
                
                # Individual legacy models
                for model_name in ['trend', 'volatility', 'risk']:
                    method_name = f"get_{model_name}_prediction"
                    if hasattr(self.legacy_integration, method_name):
                        pred_result = getattr(self.legacy_integration, method_name)(market_data)
                        predictions[f"legacy_{model_name}"] = pred_result.get('probability', 0.5)
                        confidences[f"legacy_{model_name}"] = pred_result.get('confidence', 0.0)
                
            except Exception as e:
                print(f"⚠️ Legacy prediction error: {e}")
        
        # Get predictions from online learners
        if RIVER_AVAILABLE and len(self.online_learners) > 0:
            try:
                X, _, feature_names = self._prepare_online_features(market_data)
                if len(X) > 0:
                    latest_features = X[-1]  # Use most recent features
                    
                    for name, model in self.online_learners.items():
                        try:
                            x_dict = {feature_names[j]: latest_features[j] for j in range(len(feature_names))}
                            
                            # Get prediction
                            pred = model.predict_one(x_dict)
                            if pred is not None:
                                predictions[f"online_{name}"] = float(pred)
                                
                                # Get prediction probability if available
                                if hasattr(model, 'predict_proba_one'):
                                    proba = model.predict_proba_one(x_dict)
                                    if proba:
                                        confidences[f"online_{name}"] = max(proba.values()) if isinstance(proba, dict) else 0.5
                                    else:
                                        confidences[f"online_{name}"] = 0.5
                                else:
                                    confidences[f"online_{name}"] = 0.5
                        
                        except Exception as e:
                            print(f"⚠️ Online prediction error for {name}: {e}")
            
            except Exception as e:
                print(f"⚠️ Online learning prediction error: {e}")
        
        # Combine predictions using meta-learner or weighted average
        if len(predictions) > 0:
            # Simple weighted average for now
            weights = {name: confidences.get(name, 0.5) for name in predictions.keys()}
            total_weight = sum(weights.values()) if sum(weights.values()) > 0 else 1.0
            
            weighted_prediction = sum(pred * weights.get(name, 0.5) / total_weight 
                                    for name, pred in predictions.items())
            
            avg_confidence = sum(confidences.values()) / len(confidences) if confidences else 0.0
            
            # Determine action
            if weighted_prediction > 0.55:
                action = "buy"
            elif weighted_prediction < 0.45:
                action = "sell"
            else:
                action = "hold"
            
            ensemble_result = {
                'action': action,
                'probability': float(weighted_prediction),
                'confidence': float(avg_confidence),
                'individual_predictions': predictions,
                'individual_confidences': confidences,
                'model_count': len(predictions),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                with mlflow.start_run():
                    mlflow.log_metric("ensemble_probability", weighted_prediction)
                    mlflow.log_metric("ensemble_confidence", avg_confidence)
                    mlflow.log_metric("active_models", len(predictions))
                    mlflow.log_param("action", action)
            
            return ensemble_result
        
        else:
            # No predictions available
            return {
                'action': 'hold',
                'probability': 0.5,
                'confidence': 0.0,
                'individual_predictions': {},
                'individual_confidences': {},
                'model_count': 0,
                'timestamp': datetime.now().isoformat(),
                'message': 'No models available for prediction'
            }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive model status"""
        status = {
            'adaptive_pipeline': {
                'online_learners': len(self.online_learners),
                'drift_detectors': len(self.drift_detector.drift_detectors),
                'feature_selector_active': self.feature_selector is not None,
                'meta_learner_trained': self.meta_learner.meta_model is not None
            },
            'legacy_integration': {
                'available': self.legacy_integration is not None,
                'ready': self.legacy_integration.is_ready if self.legacy_integration else False,
                'models_loaded': len(self.legacy_integration.models) if self.legacy_integration else 0
            },
            'libraries': {
                'river_available': RIVER_AVAILABLE,
                'shap_available': SHAP_AVAILABLE,
                'optuna_available': OPTUNA_AVAILABLE,
                'mlflow_available': MLFLOW_AVAILABLE,
                'sklearn_available': SKLEARN_AVAILABLE
            },
            'performance': {
                'tracked_models': len(self.model_performance),
                'models_with_drift': sum(1 for p in self.model_performance.values() if p.drift_detected)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return status


# Convenience Functions for Integration

def initialize_adaptive_pipeline(config: Optional[Dict[str, Any]] = None) -> AdaptiveModelPipeline:
    """
    Initialize the Adaptive AI Pipeline
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized AdaptiveModelPipeline instance
    """
    pipeline = AdaptiveModelPipeline(config)
    pipeline.initialize_online_learners()
    
    print("🚀 Adaptive AI Pipeline fully initialized")
    return pipeline

def get_ai_prediction(market_data: pd.DataFrame, 
                     pipeline: Optional[AdaptiveModelPipeline] = None) -> Dict[str, Any]:
    """
    Get AI prediction from the adaptive pipeline
    
    Args:
        market_data: Current market data
        pipeline: Optional pipeline instance (will create new if None)
        
    Returns:
        Prediction results
    """
    if pipeline is None:
        pipeline = initialize_adaptive_pipeline()
    
    return pipeline.get_ensemble_prediction(market_data)

def update_ai_models(new_data: pd.DataFrame, 
                    pipeline: Optional[AdaptiveModelPipeline] = None) -> Dict[str, Any]:
    """
    Update AI models with new data
    
    Args:
        new_data: New market data
        pipeline: Optional pipeline instance
        
    Returns:
        Update results
    """
    if pipeline is None:
        pipeline = initialize_adaptive_pipeline()
    
    return pipeline.update_models_online(new_data)


if __name__ == "__main__":
    # Demo execution
    print("🧠 ADAPTIVE AI PIPELINE DEMO")
    print("=" * 50)
    
    # Initialize pipeline
    config = {
        'max_features': 15,
        'drift_window_size': 500,
        'optuna_trials': 30,
        'mlflow_experiment': 'tradino_adaptive_demo'
    }
    
    pipeline = initialize_adaptive_pipeline(config)
    
    # Print status
    status = pipeline.get_model_status()
    print(f"📊 Pipeline Status:")
    print(f"   Online Learners: {status['adaptive_pipeline']['online_learners']}")
    print(f"   Legacy Integration: {status['legacy_integration']['available']}")
    print(f"   Libraries Available: {sum(status['libraries'].values())}/5")
    
    print("\n✅ Adaptive AI Pipeline ready for trading!") 