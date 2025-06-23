#!/usr/bin/env python3
"""
📈 MARKET REGIME DETECTION - HIDDEN MARKOV MODELS
Erkennung von Marktphasen für optimale Trading-Strategien
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """📊 Weltklasse Market Regime Detection"""
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = StandardScaler()
        self.regime_names = {
            0: "🐻 Bear Market",
            1: "📈 Bull Market", 
            2: "📊 Sideways Market",
            3: "⚡ High Volatility"
        }
        self.regime_characteristics = {}
        
    def prepare_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """🔧 Feature Engineering für Regime Detection"""
        
        features = pd.DataFrame(index=market_data.index)
        
        # Preisbasierte Features
        features['returns'] = market_data['close'].pct_change()
        features['log_returns'] = np.log(market_data['close'] / market_data['close'].shift(1))
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Momentum Features
        features['momentum_5'] = market_data['close'] / market_data['close'].shift(5) - 1
        features['momentum_20'] = market_data['close'] / market_data['close'].shift(20) - 1
        
        # Volume Features
        features['volume_ratio'] = market_data['volume'] / market_data['volume'].rolling(20).mean()
        features['price_volume'] = features['returns'] * features['volume_ratio']
        
        # Technical Indicators
        features['rsi'] = self._calculate_rsi(market_data['close'])
        features['bollinger_position'] = self._calculate_bollinger_position(market_data['close'])
        
        # Trend Features
        features['sma_ratio'] = market_data['close'] / market_data['close'].rolling(50).mean()
        features['ema_ratio'] = market_data['close'] / market_data['close'].ewm(span=20).mean()
        
        # Higher Order Moments
        features['skewness'] = features['returns'].rolling(20).skew()
        features['kurtosis'] = features['returns'].rolling(20).kurt()
        
        # Regime Transition Features
        features['volatility_regime'] = (features['volatility'] > features['volatility'].rolling(50).quantile(0.8)).astype(int)
        features['return_regime'] = (features['returns'] > features['returns'].rolling(50).quantile(0.6)).astype(int)
        
        # Dropna und return
        features = features.dropna()
        
        return features.values
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """📈 RSI Berechnung"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """📊 Bollinger Band Position"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return (prices - lower) / (upper - lower)
    
    def fit(self, market_data: pd.DataFrame) -> 'MarketRegimeDetector':
        """🎯 Trainiere HMM für Regime Detection"""
        
        print("🔧 Trainiere Market Regime Detection Model...")
        
        # Feature Preparation
        features = self.prepare_features(market_data)
        
        # Normalization
        features_scaled = self.scaler.fit_transform(features)
        
        # HMM Model Training
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42,
            verbose=False
        )
        
        self.model.fit(features_scaled)
        
        # Regime Charakteristiken berechnen
        self._analyze_regime_characteristics(features_scaled)
        
        print(f"✅ HMM Model trainiert - {self.n_regimes} Regimes erkannt")
        return self
    
    def _analyze_regime_characteristics(self, features: np.ndarray):
        """📊 Analysiere Regime-Charakteristiken"""
        
        # Regime Predictions
        regimes = self.model.predict(features)
        
        for regime in range(self.n_regimes):
            regime_mask = regimes == regime
            regime_features = features[regime_mask]
            
            if len(regime_features) > 0:
                self.regime_characteristics[regime] = {
                    'mean_returns': np.mean(regime_features[:, 1]),  # log_returns
                    'volatility': np.std(regime_features[:, 1]),
                    'avg_volume_ratio': np.mean(regime_features[:, 5]),
                    'frequency': np.sum(regime_mask) / len(regimes),
                    'duration': self._calculate_avg_duration(regimes, regime)
                }
    
    def _calculate_avg_duration(self, regimes: np.ndarray, regime: int) -> float:
        """⏱️ Berechne durchschnittliche Regime-Dauer"""
        durations = []
        current_duration = 0
        
        for r in regimes:
            if r == regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        return np.mean(durations) if durations else 0
    
    def predict_regime(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """🎯 Vorhersage aktuelles Market Regime"""
        
        if self.model is None:
            raise ValueError("Model muss zuerst trainiert werden!")
        
        # Prepare Features
        features = self.prepare_features(market_data)
        features_scaled = self.scaler.transform(features[-1:])  # Nur letzter Datenpunkt
        
        # Prediction
        regime = self.model.predict(features_scaled)[0]
        regime_probs = self.model.predict_proba(features_scaled)[0]
        
        # Regime Transition Wahrscheinlichkeit
        transition_matrix = self.model.transmat_
        
        result = {
            'current_regime': regime,
            'regime_name': self.regime_names.get(regime, f"Regime {regime}"),
            'confidence': regime_probs[regime],
            'regime_probabilities': dict(enumerate(regime_probs)),
            'transition_probabilities': dict(enumerate(transition_matrix[regime])),
            'characteristics': self.regime_characteristics.get(regime, {}),
            'trading_recommendation': self._get_trading_recommendation(regime)
        }
        
        return result
    
    def _get_trading_recommendation(self, regime: int) -> Dict[str, Any]:
        """💡 Trading-Empfehlung basierend auf Regime"""
        
        recommendations = {
            0: {  # Bear Market
                'strategy': 'Defensive',
                'position_size': 0.5,
                'instruments': ['Short Positions', 'Hedging', 'Cash'],
                'risk_level': 'High'
            },
            1: {  # Bull Market
                'strategy': 'Aggressive',
                'position_size': 1.0,
                'instruments': ['Long Positions', 'Growth Assets'],
                'risk_level': 'Medium'
            },
            2: {  # Sideways Market
                'strategy': 'Range Trading',
                'position_size': 0.7,
                'instruments': ['Mean Reversion', 'Arbitrage'],
                'risk_level': 'Low'
            },
            3: {  # High Volatility
                'strategy': 'Momentum',
                'position_size': 0.3,
                'instruments': ['Volatility Trading', 'Options'],
                'risk_level': 'Very High'
            }
        }
        
        return recommendations.get(regime, recommendations[2])
    
    def get_regime_analysis(self) -> Dict[str, Any]:
        """📊 Vollständige Regime-Analyse"""
        
        analysis = {
            'model_info': {
                'n_regimes': self.n_regimes,
                'model_type': 'Hidden Markov Model',
                'covariance_type': 'full'
            },
            'regime_characteristics': self.regime_characteristics,
            'regime_names': self.regime_names
        }
        
        return analysis

    # Legacy compatibility method
    def detect_regime(self, market_data: pd.DataFrame) -> str:
        """Legacy method für Backwards Compatibility"""
        try:
            regime_result = self.predict_regime(market_data)
            return regime_result['regime_name']
        except:
            return "📊 Sideways Market"

# Verwendungsbeispiel
if __name__ == "__main__":
    # Simuliere Marktdaten
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Simulierte OHLCV-Daten
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(n_days) * 100)
    volumes = np.random.uniform(1000, 10000, n_days)
    
    market_data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': volumes
    })
    
    # Market Regime Detection
    detector = MarketRegimeDetector(n_regimes=4)
    detector.fit(market_data)
    
    # Aktuelle Regime-Vorhersage
    current_regime = detector.predict_regime(market_data)
    
    print("🎯 CURRENT MARKET REGIME ANALYSIS")
    print("=" * 50)
    print(f"Regime: {current_regime['regime_name']}")
    print(f"Confidence: {current_regime['confidence']:.2%}")
    print(f"Strategy: {current_regime['trading_recommendation']['strategy']}")
    print(f"Risk Level: {current_regime['trading_recommendation']['risk_level']}")
