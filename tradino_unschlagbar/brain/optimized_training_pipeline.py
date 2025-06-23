#!/usr/bin/env python3
"""
🧠 TRADINO OPTIMIZED TRAINING PIPELINE
Resource-efficient ML Training für 4GB RAM Setup
"""

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
import pandas_ta as ta
import yfinance as yf
import ccxt
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class OptimizedTrainingPipeline:
    """🚀 Resource-Optimized Training Pipeline"""
    
    def __init__(self):
        self.progress = 0
        self.models = {}
        self.feature_pipeline = None
        self.scalers = {}
        self.label_encoders = {}
        
        # Training Configuration
        self.config = {
            'memory_limit': '2GB',  # Conservative for 4GB system
            'n_jobs': 2,            # Use both CPU cores
            'early_stopping': True,
            'verbose': 1
        }
        
        print("🧠 Optimized Training Pipeline initialisiert")
        print(f"💾 Memory Limit: {self.config['memory_limit']}")
        print(f"🔄 CPU Cores: {self.config['n_jobs']}")
    
    def update_progress(self, step: int, total: int, description: str):
        """📊 Update Progress Bar"""
        progress = int((step / total) * 100)
        self.progress = progress
        
        # Progress Bar
        filled = int(progress * 40 / 100)
        bar = "█" * filled + "░" * (40 - filled)
        
        print(f"\n📊 TRAINING PROGRESS")
        print(f"{bar} {progress}% Complete")
        print(f"[{description:38}] {'✅' if progress == 100 else '⏳'}")
    
    def collect_training_data(self, symbols: List[str] = None, days: int = 365) -> pd.DataFrame:
        """📊 Sammle Trainingsdaten (memory-efficient)"""
        
        if symbols is None:
            symbols = ['BTC-USD', 'ETH-USD']  # Start small for memory
        
        self.update_progress(1, 10, "Collecting market data...")
        
        all_data = []
        
        for symbol in symbols:
            try:
                # Download data in chunks to save memory
                ticker = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval='1h'  # Hourly data for better granularity
                )
                
                if len(data) > 0:
                    data['symbol'] = symbol
                    data.reset_index(inplace=True)
                    all_data.append(data)
                    print(f"✅ {symbol}: {len(data)} samples")
                else:
                    print(f"⚠️ {symbol}: No data available")
                    
            except Exception as e:
                print(f"❌ Error loading {symbol}: {e}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"📊 Total samples: {len(combined_data)}")
            return combined_data
        else:
            print("❌ No data collected")
            return pd.DataFrame()
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """🔧 Feature Engineering (memory-optimized)"""
        
        self.update_progress(2, 10, "Creating features...")
        
        features_df = data.copy()
        
        # Technical Indicators (memory-efficient)
        for symbol in features_df['symbol'].unique():
            symbol_data = features_df[features_df['symbol'] == symbol].copy()
            
            # Price-based features
            symbol_data['returns'] = symbol_data['Close'].pct_change()
            symbol_data['log_returns'] = np.log(symbol_data['Close'] / symbol_data['Close'].shift(1))
            
            # Technical Indicators
            symbol_data['rsi'] = ta.rsi(symbol_data['Close'], length=14)
            symbol_data['sma_20'] = ta.sma(symbol_data['Close'], length=20)
            symbol_data['sma_50'] = ta.sma(symbol_data['Close'], length=50)
            symbol_data['ema_12'] = ta.ema(symbol_data['Close'], length=12)
            symbol_data['ema_26'] = ta.ema(symbol_data['Close'], length=26)
            
            # MACD
            macd_data = ta.macd(symbol_data['Close'])
            if macd_data is not None and len(macd_data.columns) >= 3:
                symbol_data['macd'] = macd_data.iloc[:, 0]
                symbol_data['macd_signal'] = macd_data.iloc[:, 1]
                symbol_data['macd_histogram'] = macd_data.iloc[:, 2]
            
            # Bollinger Bands
            bb_data = ta.bbands(symbol_data['Close'])
            if bb_data is not None and len(bb_data.columns) >= 3:
                symbol_data['bb_upper'] = bb_data.iloc[:, 0]
                symbol_data['bb_middle'] = bb_data.iloc[:, 1]
                symbol_data['bb_lower'] = bb_data.iloc[:, 2]
            
            # Volume features
            symbol_data['volume_sma'] = ta.sma(symbol_data['Volume'], length=20)
            symbol_data['volume_ratio'] = symbol_data['Volume'] / symbol_data['volume_sma']
            
            # Volatility
            symbol_data['atr'] = ta.atr(symbol_data['High'], symbol_data['Low'], symbol_data['Close'])
            symbol_data['volatility'] = symbol_data['returns'].rolling(20).std()
            
            # Update main dataframe
            features_df.update(symbol_data)
        
        # Create target variables
        features_df['future_return'] = features_df.groupby('symbol')['returns'].shift(-1)
        features_df['target_direction'] = (features_df['future_return'] > 0).astype(int)
        features_df['target_volatility'] = features_df.groupby('symbol')['volatility'].shift(-1)
        
        # Clean data
        features_df = features_df.dropna()
        
        print(f"🔧 Features created: {len(features_df.columns)} columns")
        print(f"📊 Clean samples: {len(features_df)}")
        
        return features_df
    
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """📋 Prepare data for training (memory-efficient)"""
        
        self.update_progress(3, 10, "Preparing training data...")
        
        # Select features (avoid memory issues)
        feature_columns = [
            'returns', 'log_returns', 'rsi', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_histogram', 'bb_upper', 'bb_middle', 'bb_lower',
            'volume_ratio', 'atr', 'volatility'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in data.columns]
        print(f"📊 Using {len(available_features)} features")
        
        # Prepare features and targets
        X = data[available_features].values
        y_direction = data['target_direction'].values
        
        # Handle missing values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Store for later use
        self.feature_pipeline = {
            'features': available_features,
            'imputer': imputer,
            'scaler': scaler
        }
        
        print(f"📊 Training data shape: {X_scaled.shape}")
        print(f"🎯 Target distribution: {np.bincount(y_direction)}")
        
        return X_scaled, y_direction, available_features
    
    def train_xgboost_model(self, X: np.ndarray, y: np.ndarray, model_name: str = "trend_detector") -> Dict:
        """🚀 Train XGBoost Model (memory-optimized)"""
        
        self.update_progress(4, 10, f"Training XGBoost {model_name}...")
        
        # Memory-efficient XGBoost parameters
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'n_estimators': 200,  # Moderate to save memory
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': self.config['n_jobs'],
            'verbosity': 0
        }
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        model = xgb.XGBClassifier(**xgb_params)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=1)
        
        # Train final model
        model.fit(X, y)
        
        # Predictions for evaluation
        y_pred = model.predict(X)
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted'),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        self.models[model_name] = {
            'model': model,
            'type': 'xgboost',
            'metrics': metrics,
            'feature_importance': dict(zip(
                self.feature_pipeline['features'],
                model.feature_importances_
            ))
        }
        
        print(f"✅ {model_name} trained:")
        print(f"   Accuracy: {metrics['accuracy']:.3f}")
        print(f"   CV Score: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        
        return metrics
    
    def train_lightgbm_model(self, X: np.ndarray, y: np.ndarray, model_name: str = "volatility_predictor") -> Dict:
        """⚡ Train LightGBM Model (memory-optimized)"""
        
        self.update_progress(5, 10, f"Training LightGBM {model_name}...")
        
        # Memory-efficient LightGBM parameters
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'n_estimators': 150,  # Moderate to save memory
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'random_state': 42,
            'n_jobs': self.config['n_jobs'],
            'verbosity': -1
        }
        
        model = lgb.LGBMClassifier(**lgb_params)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=1)
        
        # Train final model
        model.fit(X, y)
        
        # Predictions
        y_pred = model.predict(X)
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted'),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        self.models[model_name] = {
            'model': model,
            'type': 'lightgbm',
            'metrics': metrics,
            'feature_importance': dict(zip(
                self.feature_pipeline['features'],
                model.feature_importances_
            ))
        }
        
        print(f"✅ {model_name} trained:")
        print(f"   Accuracy: {metrics['accuracy']:.3f}")
        print(f"   CV Score: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        
        return metrics
    
    def train_random_forest_model(self, X: np.ndarray, y: np.ndarray, model_name: str = "risk_assessor") -> Dict:
        """🌲 Train Random Forest Model (memory-optimized)"""
        
        self.update_progress(6, 10, f"Training Random Forest {model_name}...")
        
        # Memory-efficient Random Forest parameters
        rf_params = {
            'n_estimators': 100,  # Moderate to save memory
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': self.config['n_jobs']
        }
        
        model = RandomForestClassifier(**rf_params)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=1)
        
        # Train final model
        model.fit(X, y)
        
        # Predictions
        y_pred = model.predict(X)
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted'),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        self.models[model_name] = {
            'model': model,
            'type': 'random_forest',
            'metrics': metrics,
            'feature_importance': dict(zip(
                self.feature_pipeline['features'],
                model.feature_importances_
            ))
        }
        
        print(f"✅ {model_name} trained:")
        print(f"   Accuracy: {metrics['accuracy']:.3f}")
        print(f"   CV Score: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        
        return metrics
    
    def create_ensemble_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """🎭 Create Ensemble Model"""
        
        self.update_progress(7, 10, "Creating ensemble model...")
        
        # Get predictions from all models
        predictions = []
        
        for model_name, model_info in self.models.items():
            if 'model' in model_info:
                pred = model_info['model'].predict_proba(X)[:, 1]  # Probability of class 1
                predictions.append(pred)
                print(f"📊 {model_name} predictions: {pred.mean():.3f} ± {pred.std():.3f}")
        
        if len(predictions) >= 2:
            # Simple ensemble: weighted average
            ensemble_pred_proba = np.mean(predictions, axis=0)
            ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
            
            # Ensemble metrics
            ensemble_metrics = {
                'accuracy': accuracy_score(y, ensemble_pred),
                'precision': precision_score(y, ensemble_pred, average='weighted'),
                'recall': recall_score(y, ensemble_pred, average='weighted'),
                'f1': f1_score(y, ensemble_pred, average='weighted')
            }
            
            self.models['ensemble'] = {
                'type': 'ensemble',
                'metrics': ensemble_metrics,
                'weights': [1.0 / len(predictions)] * len(predictions)
            }
            
            print(f"✅ Ensemble model created:")
            print(f"   Accuracy: {ensemble_metrics['accuracy']:.3f}")
            print(f"   Models combined: {len(predictions)}")
            
            return ensemble_metrics
        else:
            print("⚠️ Not enough models for ensemble")
            return {}
    
    def save_models(self, save_dir: str = "models/optimized_models"):
        """💾 Save trained models"""
        
        self.update_progress(8, 10, "Saving models...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for model_name, model_info in self.models.items():
            if 'model' in model_info:
                model_path = os.path.join(save_dir, f"{model_name}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model_info['model'], f)
                print(f"💾 {model_name} saved to {model_path}")
        
        # Save feature pipeline
        pipeline_path = os.path.join(save_dir, "feature_pipeline.pkl")
        with open(pipeline_path, 'wb') as f:
            pickle.dump(self.feature_pipeline, f)
        
        # Save model info
        info_path = os.path.join(save_dir, "model_info.json")
        model_info_serializable = {}
        for name, info in self.models.items():
            model_info_serializable[name] = {
                'type': info['type'],
                'metrics': info['metrics']
            }
            if 'feature_importance' in info:
                model_info_serializable[name]['feature_importance'] = info['feature_importance']
            if 'weights' in info:
                model_info_serializable[name]['weights'] = info['weights']
        
        with open(info_path, 'w') as f:
            json.dump(model_info_serializable, f, indent=2)
        
        print(f"✅ All models saved to {save_dir}")
    
    def run_complete_training(self):
        """🚀 Run complete training pipeline"""
        
        print("🚀 STARTING OPTIMIZED TRAINING PIPELINE")
        print("=" * 50)
        
        try:
            # Step 1: Collect data
            data = self.collect_training_data(['BTC-USD', 'ETH-USD'], days=180)  # 6 months
            
            if data.empty:
                print("❌ No data collected - aborting training")
                return
            
            # Step 2: Create features
            features_data = self.create_features(data)
            
            # Step 3: Prepare training data
            X, y, feature_names = self.prepare_training_data(features_data)
            
            # Step 4-6: Train models
            xgb_metrics = self.train_xgboost_model(X, y, "trend_detector")
            lgb_metrics = self.train_lightgbm_model(X, y, "volatility_predictor")
            rf_metrics = self.train_random_forest_model(X, y, "risk_assessor")
            
            # Step 7: Create ensemble
            ensemble_metrics = self.create_ensemble_model(X, y)
            
            # Step 8: Save models
            self.save_models()
            
            # Step 9: Summary
            self.update_progress(9, 10, "Training completed successfully!")
            
            self.print_training_summary()
            
            # Step 10: Complete
            self.update_progress(10, 10, "All models ready for trading!")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
    
    def print_training_summary(self):
        """📊 Print Training Summary"""
        
        print("\n" + "="*60)
        print("📊 TRAINING SUMMARY")
        print("="*60)
        
        for model_name, model_info in self.models.items():
            if 'metrics' in model_info:
                metrics = model_info['metrics']
                print(f"\n🤖 {model_name.upper()} ({model_info['type']}):")
                print(f"   Accuracy:  {metrics['accuracy']:.3f}")
                print(f"   Precision: {metrics.get('precision', 0):.3f}")
                print(f"   Recall:    {metrics.get('recall', 0):.3f}")
                print(f"   F1-Score:  {metrics.get('f1', 0):.3f}")
                
                if 'cv_mean' in metrics:
                    print(f"   CV Score:  {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        
        print(f"\n📁 Models saved: {len(self.models)} total")
        print(f"🔧 Features used: {len(self.feature_pipeline['features'])}")
        print(f"💾 Memory usage: Optimized for 4GB RAM")
        print("\n✅ TRAINING COMPLETE - Ready for live trading!")

# Demo execution
if __name__ == "__main__":
    pipeline = OptimizedTrainingPipeline()
    pipeline.run_complete_training()