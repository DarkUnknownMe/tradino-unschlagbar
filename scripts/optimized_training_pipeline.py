#!/usr/bin/env python3
"""
🧠 TRADINO OPTIMIZED TRAINING PIPELINE
Resource-efficient ML Training für 4GB RAM Setup
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.impute import SimpleImputer
    import pandas_ta as ta
    import yfinance as yf
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML libraries not available: {e}")
    ML_AVAILABLE = False

import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

class OptimizedTrainingPipeline:
    """🚀 Resource-Optimized Training Pipeline"""
    
    def __init__(self):
        self.progress = 0
        self.models = {}
        self.feature_pipeline = None
        
        print("🧠 Optimized Training Pipeline initialisiert")
        
        if not ML_AVAILABLE:
            print("❌ Required ML libraries missing!")
            return
    
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
    
    def collect_demo_data(self) -> pd.DataFrame:
        """📊 Create demo training data"""
        
        self.update_progress(1, 10, "Creating demo training data...")
        
        # Create synthetic market data for demo
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1H')
        
        data = []
        for i, date in enumerate(dates):
            # Simulate price movement
            base_price = 50000 + 1000 * np.sin(i * 0.01) + 5000 * np.random.randn()
            
            high = base_price + abs(np.random.normal(0, 100))
            low = base_price - abs(np.random.normal(0, 100))
            close = base_price + np.random.normal(0, 50)
            volume = abs(np.random.normal(1000000, 200000))
            
            data.append({
                'Datetime': date,
                'Open': base_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume,
                'symbol': 'BTC-USD'
            })
        
        demo_df = pd.DataFrame(data)
        print(f"📊 Demo data created: {len(demo_df)} samples")
        return demo_df
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """🔧 Feature Engineering"""
        
        self.update_progress(2, 10, "Creating features...")
        
        df = data.copy()
        
        # Basic features
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
        
        # Target: future return direction
        df['future_return'] = df['returns'].shift(-1)
        df['target'] = (df['future_return'] > 0).astype(int)
        
        # Clean data
        df = df.dropna()
        
        print(f"🔧 Features created: {len(df.columns)} columns")
        print(f"📊 Clean samples: {len(df)}")
        
        return df
    
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """📋 Prepare data for training"""
        
        self.update_progress(3, 10, "Preparing training data...")
        
        feature_columns = [
            'returns', 'log_returns', 'sma_10', 'sma_30', 'volatility',
            'momentum_5', 'momentum_10', 'volume_ratio', 'price_position'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in data.columns]
        
        X = data[available_features].values
        y = data['target'].values
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Store pipeline
        self.feature_pipeline = {
            'features': available_features,
            'imputer': imputer,
            'scaler': scaler
        }
        
        print(f"📊 Training data shape: {X_scaled.shape}")
        print(f"🎯 Target distribution: {np.bincount(y)}")
        
        return X_scaled, y, available_features
    
    def train_xgboost_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """🚀 Train XGBoost Model"""
        
        self.update_progress(4, 10, "Training XGBoost...")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=2
        )
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
        
        # Train final model
        model.fit(X, y)
        y_pred = model.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        self.models['xgboost_trend'] = {
            'model': model,
            'type': 'xgboost',
            'metrics': metrics
        }
        
        print(f"✅ XGBoost trained: {metrics['accuracy']:.3f} accuracy")
        return metrics
    
    def train_lightgbm_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """⚡ Train LightGBM Model"""
        
        self.update_progress(5, 10, "Training LightGBM...")
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=2,
            verbosity=-1
        )
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
        
        # Train final model
        model.fit(X, y)
        y_pred = model.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        self.models['lightgbm_volatility'] = {
            'model': model,
            'type': 'lightgbm',
            'metrics': metrics
        }
        
        print(f"✅ LightGBM trained: {metrics['accuracy']:.3f} accuracy")
        return metrics
    
    def train_random_forest_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """🌲 Train Random Forest Model"""
        
        self.update_progress(6, 10, "Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            n_jobs=2
        )
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
        
        # Train final model
        model.fit(X, y)
        y_pred = model.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        self.models['random_forest_risk'] = {
            'model': model,
            'type': 'random_forest',
            'metrics': metrics
        }
        
        print(f"✅ Random Forest trained: {metrics['accuracy']:.3f} accuracy")
        return metrics
    
    def create_ensemble_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """🎭 Create Ensemble Model"""
        
        self.update_progress(7, 10, "Creating ensemble...")
        
        if len(self.models) < 2:
            print("⚠️ Not enough models for ensemble")
            return {}
        
        # Get predictions from all models
        predictions = []
        for model_name, model_info in self.models.items():
            pred_proba = model_info['model'].predict_proba(X)[:, 1]
            predictions.append(pred_proba)
        
        # Simple ensemble: average
        ensemble_pred_proba = np.mean(predictions, axis=0)
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y, ensemble_pred)
        }
        
        self.models['ensemble'] = {
            'type': 'ensemble',
            'metrics': metrics,
            'num_models': len(predictions)
        }
        
        print(f"✅ Ensemble created: {metrics['accuracy']:.3f} accuracy")
        return metrics
    
    def save_models(self):
        """💾 Save models"""
        
        self.update_progress(8, 10, "Saving models...")
        
        os.makedirs('models', exist_ok=True)
        
        # Save individual models
        for name, info in self.models.items():
            if 'model' in info:
                with open(f'models/{name}.pkl', 'wb') as f:
                    pickle.dump(info['model'], f)
        
        # Save feature pipeline
        with open('models/feature_pipeline.pkl', 'wb') as f:
            pickle.dump(self.feature_pipeline, f)
        
        # Save model info
        model_info = {}
        for name, info in self.models.items():
            model_info[name] = {
                'type': info['type'],
                'metrics': info['metrics']
            }
        
        with open('models/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("✅ Models saved successfully")
    
    def run_complete_training(self):
        """🚀 Run complete training pipeline"""
        
        if not ML_AVAILABLE:
            print("❌ Cannot run training - missing dependencies")
            return
        
        print("🚀 STARTING OPTIMIZED TRAINING")
        print("=" * 40)
        
        try:
            # Collect data
            data = self.collect_demo_data()
            
            # Create features
            features_data = self.create_features(data)
            
            # Prepare training data
            X, y, feature_names = self.prepare_training_data(features_data)
            
            # Train models
            self.train_xgboost_model(X, y)
            self.train_lightgbm_model(X, y)
            self.train_random_forest_model(X, y)
            
            # Create ensemble
            self.create_ensemble_model(X, y)
            
            # Save models
            self.save_models()
            
            # Complete
            self.update_progress(10, 10, "Training completed!")
            
            self.print_summary()
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
    
    def print_summary(self):
        """📊 Print summary"""
        
        print("\n" + "="*50)
        print("📊 TRAINING SUMMARY")
        print("="*50)
        
        for name, info in self.models.items():
            if 'metrics' in info:
                acc = info['metrics']['accuracy']
                print(f"🤖 {name}: {acc:.3f} accuracy")
        
        print(f"\n✅ Training complete!")
        print(f"📁 Models saved: {len(self.models)}")
        print(f"🔧 Features: {len(self.feature_pipeline['features'])}")
        print("🚀 Ready for trading!")

if __name__ == "__main__":
    pipeline = OptimizedTrainingPipeline()
    pipeline.run_complete_training() 