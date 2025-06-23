import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
from collections import deque
import asyncio
from dataclasses import dataclass
from enum import Enum

# Scientific Computing Libraries
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from hmmlearn import hmm
    ADVANCED_ML_AVAILABLE = True
    logger.success("✅ Advanced ML libraries verfügbar")
except ImportError:
    logger.warning("⚠️ Advanced ML libraries nicht verfügbar - verwende Fallback Implementation")
    ADVANCED_ML_AVAILABLE = False

class MarketRegime(Enum):
    """
    Definierte Marktregime
    """
    BULL_TRENDING = "bull_trending"           # Starker Aufwärtstrend
    BEAR_TRENDING = "bear_trending"           # Starker Abwärtstrend
    SIDEWAYS_RANGE = "sideways_range"         # Seitwärtsbewegung
    HIGH_VOLATILITY = "high_volatility"       # Hohe Volatilität
    LOW_VOLATILITY = "low_volatility"         # Niedrige Volatilität
    BULL_CORRECTION = "bull_correction"       # Korrektur im Bullenmarkt
    BEAR_RALLY = "bear_rally"                 # Rally im Bärenmarkt
    ACCUMULATION = "accumulation"             # Akkumulationsphase
    DISTRIBUTION = "distribution"             # Distributionsphase
    BREAKOUT = "breakout"                     # Breakout-Phase
    CRISIS = "crisis"                         # Krisenphase

@dataclass
class RegimeAnalysis:
    """
    Marktregime-Analyse Ergebnis
    """
    current_regime: MarketRegime
    confidence: float
    regime_probabilities: Dict[MarketRegime, float]
    regime_duration: int  # Tage im aktuellen Regime
    transition_probability: float
    optimal_strategies: List[str]
    risk_level: str
    expected_volatility: float
    regime_features: Dict[str, float]
    historical_performance: Dict[str, float]

class AdvancedMarketRegimeDetector:
    """
    Hochentwickelter Market Regime Detector
    Verwendet Hidden Markov Models + Gaussian Mixture Models für präzise Regime-Erkennung
    """
    
    def __init__(self, config):
        self.config = config
        
        # Model Configuration
        self.n_regimes = 8  # Anzahl der Hidden States
        self.n_components = 5  # GMM Components
        self.lookback_period = 252  # 1 Jahr für Training
        self.min_regime_duration = 5  # Minimum Tage für Regime-Wechsel
        
        # Models
        self.hmm_model = None
        self.gmm_model = None
        self.scaler = StandardScaler() if ADVANCED_ML_AVAILABLE else None
        self.pca = PCA(n_components=0.95) if ADVANCED_ML_AVAILABLE else None  # 95% Varianz
        
        # State
        self.is_trained = False
        self.current_regime = MarketRegime.SIDEWAYS_RANGE
        self.regime_history = deque(maxlen=1000)
        self.feature_history = deque(maxlen=self.lookback_period)
        
        # Regime Mappings
        self.regime_mappings = self._initialize_regime_mappings()
        self.strategy_mappings = self._initialize_strategy_mappings()
        
        # Performance Tracking
        self.regime_performance = {}
        self.transition_matrix = np.ones((len(MarketRegime), len(MarketRegime))) / len(MarketRegime)
        
        logger.info("🧠 Advanced Market Regime Detector initialisiert")
    
    def _initialize_regime_mappings(self) -> Dict:
        """
        Regime Mapping Initialisierung
        """
        return {
            0: MarketRegime.BULL_TRENDING,
            1: MarketRegime.BEAR_TRENDING,
            2: MarketRegime.SIDEWAYS_RANGE,
            3: MarketRegime.HIGH_VOLATILITY,
            4: MarketRegime.LOW_VOLATILITY,
            5: MarketRegime.BULL_CORRECTION,
            6: MarketRegime.BEAR_RALLY,
            7: MarketRegime.BREAKOUT
        }
    
    def _initialize_strategy_mappings(self) -> Dict[MarketRegime, List[str]]:
        """
        Optimale Strategien für jedes Regime
        """
        return {
            MarketRegime.BULL_TRENDING: ['trend_following', 'momentum', 'breakout'],
            MarketRegime.BEAR_TRENDING: ['short_selling', 'mean_reversion', 'defensive'],
            MarketRegime.SIDEWAYS_RANGE: ['mean_reversion', 'range_trading', 'arbitrage'],
            MarketRegime.HIGH_VOLATILITY: ['volatility_trading', 'straddle', 'breakout'],
            MarketRegime.LOW_VOLATILITY: ['carry_trade', 'momentum', 'trend_following'],
            MarketRegime.BULL_CORRECTION: ['buy_dips', 'value_investing', 'defensive'],
            MarketRegime.BEAR_RALLY: ['short_covering', 'contrarian', 'momentum'],
            MarketRegime.ACCUMULATION: ['value_investing', 'position_building', 'patient_capital'],
            MarketRegime.DISTRIBUTION: ['profit_taking', 'short_selling', 'defensive'],
            MarketRegime.BREAKOUT: ['momentum', 'trend_following', 'breakout'],
            MarketRegime.CRISIS: ['defensive', 'safe_haven', 'cash']
        }
    
    async def initialize(self):
        """
        Regime Detector initialisieren
        """
        try:
            if ADVANCED_ML_AVAILABLE:
                # HMM Model
                self.hmm_model = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    n_iter=100,
                    random_state=42
                )
                
                # GMM Model  
                self.gmm_model = GaussianMixture(
                    n_components=self.n_components,
                    covariance_type='full',
                    random_state=42
                )
                
                logger.success("✅ Advanced ML Models initialisiert")
            else:
                logger.info("📊 Fallback Models initialisiert")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Regime Detector Initialization Fehler: {e}")
            return False
    
    async def detect_current_regime(self, market_data: Dict) -> RegimeAnalysis:
        """
        Aktuelles Marktregime erkennen
        """
        try:
            # Market Features extrahieren
            features = await self._extract_regime_features(market_data)
            
            if not features:
                return self._create_fallback_analysis()
            
            # Feature History aktualisieren
            self.feature_history.append(features)
            
            # Model Training falls erforderlich
            if not self.is_trained and len(self.feature_history) >= 50:
                await self._train_models()
            
            # Regime Detection
            if self.is_trained and ADVANCED_ML_AVAILABLE:
                regime_analysis = await self._advanced_regime_detection(features)
            else:
                regime_analysis = await self._fallback_regime_detection(features)
            
            # Regime History aktualisieren
            self.regime_history.append({
                'regime': regime_analysis.current_regime,
                'confidence': regime_analysis.confidence,
                'timestamp': datetime.now()
            })
            
            # Update current regime
            self.current_regime = regime_analysis.current_regime
            
            logger.debug(f"🧠 Detected Regime: {regime_analysis.current_regime.value} (Confidence: {regime_analysis.confidence:.3f})")
            
            return regime_analysis
            
        except Exception as e:
            logger.error(f"❌ Regime Detection Fehler: {e}")
            return self._create_fallback_analysis()
    
    async def _extract_regime_features(self, market_data: Dict) -> Optional[Dict[str, float]]:
        """
        Regime-relevante Features extrahieren
        """
        try:
            # Basic Market Data
            close_price = market_data.get('close', 0)
            high_price = market_data.get('high', close_price)
            low_price = market_data.get('low', close_price)
            volume = market_data.get('volume', 0)
            
            # Mock Historical Data für Demo
            price_history = self._generate_price_history(close_price, 100)
            volume_history = [volume * (1 + np.random.normal(0, 0.2)) for _ in range(100)]
            
            features = {}
            
            # 1. Trend Features
            features['price_momentum_5'] = (close_price - price_history[-6]) / price_history[-6] if len(price_history) > 5 else 0
            features['price_momentum_20'] = (close_price - price_history[-21]) / price_history[-21] if len(price_history) > 20 else 0
            features['price_momentum_50'] = (close_price - price_history[-51]) / price_history[-51] if len(price_history) > 50 else 0
            
            # Moving Averages
            features['sma_5'] = np.mean(price_history[-5:]) if len(price_history) >= 5 else close_price
            features['sma_20'] = np.mean(price_history[-20:]) if len(price_history) >= 20 else close_price
            features['sma_50'] = np.mean(price_history[-50:]) if len(price_history) >= 50 else close_price
            
            # MA Relationships
            features['price_vs_sma20'] = (close_price - features['sma_20']) / features['sma_20']
            features['sma5_vs_sma20'] = (features['sma_5'] - features['sma_20']) / features['sma_20']
            features['sma20_vs_sma50'] = (features['sma_20'] - features['sma_50']) / features['sma_50']
            
            # 2. Volatility Features
            returns = np.diff(price_history) / price_history[:-1] if len(price_history) > 1 else [0]
            features['realized_volatility'] = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
            features['volatility_5d'] = np.std(returns[-5:]) if len(returns) >= 5 else 0.02
            features['volatility_20d'] = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
            
            # Volatility Ratios
            features['vol_ratio_5_20'] = (features['volatility_5d'] / features['volatility_20d']) if features['volatility_20d'] > 0 else 1.0
            
            # Range Features
            features['daily_range'] = (high_price - low_price) / close_price
            features['true_range'] = max(
                high_price - low_price,
                abs(high_price - price_history[-2]) if len(price_history) > 1 else 0,
                abs(low_price - price_history[-2]) if len(price_history) > 1 else 0
            ) / close_price
            
            # 3. Volume Features
            features['volume_sma_20'] = np.mean(volume_history[-20:]) if len(volume_history) >= 20 else volume
            features['volume_ratio'] = volume / features['volume_sma_20'] if features['volume_sma_20'] > 0 else 1.0
            features['volume_trend'] = np.corrcoef(range(len(volume_history[-20:])), volume_history[-20:])[0,1] if len(volume_history) >= 20 else 0
            
            # Price-Volume Relationship
            if len(price_history) >= 20 and len(volume_history) >= 20:
                price_changes = np.diff(price_history[-20:])
                volume_changes = volume_history[-19:]  # Align with price changes
                if len(price_changes) == len(volume_changes) and len(price_changes) > 0:
                    features['price_volume_correlation'] = np.corrcoef(price_changes, volume_changes)[0,1]
                    if np.isnan(features['price_volume_correlation']):
                        features['price_volume_correlation'] = 0
                else:
                    features['price_volume_correlation'] = 0
            else:
                features['price_volume_correlation'] = 0
            
            # 4. Technical Indicators
            features['rsi'] = self._calculate_rsi(price_history)
            features['macd'], features['macd_signal'] = self._calculate_macd(price_history)
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Bollinger Bands
            bb_middle = features['sma_20']
            bb_std = np.std(price_history[-20:]) if len(price_history) >= 20 else price_history[-1] * 0.02
            features['bb_position'] = (close_price - bb_middle) / (2 * bb_std) if bb_std > 0 else 0
            features['bb_width'] = (4 * bb_std) / bb_middle if bb_middle > 0 else 0.04
            
            # Clean Features (remove NaN/Inf)
            for key, value in features.items():
                if np.isnan(value) or np.isinf(value):
                    features[key] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Feature Extraction Fehler: {e}")
            return None
    
    def _generate_price_history(self, current_price: float, length: int) -> List[float]:
        """
        Mock Price History für Demo
        """
        history = []
        price = current_price * 0.9  # Start 10% lower
        
        for i in range(length):
            # Regime-aware Random Walk
            if i < length * 0.3:  # Bull phase
                change = np.random.normal(0.002, 0.015)  # 0.2% mean, 1.5% std
            elif i < length * 0.6:  # Sideways phase  
                change = np.random.normal(0.0, 0.01)    # 0% mean, 1% std
            else:  # Volatile phase
                change = np.random.normal(0.001, 0.025)  # 0.1% mean, 2.5% std
                
            price *= (1 + change)
            history.append(price)
        
        return history
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI Calculation"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: List[float]) -> Tuple[float, float]:
        """MACD Calculation"""
        if len(prices) < 26:
            return 0.0, 0.0
        
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd_line = ema_12 - ema_26
        macd_signal = macd_line * 0.9  # Simplified signal line
        
        return macd_line, macd_signal
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """EMA Calculation"""
        if len(prices) < period:
            return np.mean(prices) if prices else 0
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    async def _train_models(self):
        """
        HMM und GMM Models trainieren
        """
        try:
            if not ADVANCED_ML_AVAILABLE or len(self.feature_history) < 50:
                logger.info("📊 Fallback Training verwendet")
                self.is_trained = True
                return
            
            logger.info("🎓 Training Advanced Regime Detection Models...")
            
            # Feature Matrix erstellen
            feature_matrix = []
            for features in list(self.feature_history):
                feature_vector = list(features.values())
                feature_matrix.append(feature_vector)
            
            feature_matrix = np.array(feature_matrix)
            
            # Feature Scaling
            scaled_features = self.scaler.fit_transform(feature_matrix)
            
            # PCA für Dimensionality Reduction
            reduced_features = self.pca.fit_transform(scaled_features)
            
            # GMM Training
            self.gmm_model.fit(reduced_features)
            
            # HMM Training
            self.hmm_model.fit(reduced_features)
            
            self.is_trained = True
            logger.success(f"✅ Models trainiert mit {len(feature_matrix)} Samples")
            
        except Exception as e:
            logger.error(f"❌ Model Training Fehler: {e}")
            self.is_trained = True  # Fallback zu Regel-basierter Detection
    
    async def _advanced_regime_detection(self, features: Dict[str, float]) -> RegimeAnalysis:
        """
        Advanced Regime Detection mit HMM + GMM
        """
        try:
            # Feature Vector erstellen
            feature_vector = np.array([list(features.values())])
            
            # Feature Scaling
            scaled_features = self.scaler.transform(feature_vector)
            
            # PCA Transform
            reduced_features = self.pca.transform(scaled_features)
            
            # HMM Regime Prediction
            hmm_state = self.hmm_model.predict(reduced_features)[0]
            hmm_probabilities = self.hmm_model.predict_proba(reduced_features)[0]
            
            # Regime Mapping
            primary_regime = self.regime_mappings.get(hmm_state, MarketRegime.SIDEWAYS_RANGE)
            
            # Confidence basierend auf HMM Probability
            regime_confidence = np.max(hmm_probabilities)
            
            # Regime Probabilities für alle Regime
            regime_probs = {}
            for i, regime in self.regime_mappings.items():
                if i < len(hmm_probabilities):
                    regime_probs[regime] = hmm_probabilities[i]
                else:
                    regime_probs[regime] = 0.0
            
            # Fill missing regimes
            for regime in MarketRegime:
                if regime not in regime_probs:
                    regime_probs[regime] = 0.0
            
            # Enhanced Analysis
            regime_analysis = await self._create_enhanced_regime_analysis(
                primary_regime, regime_confidence, regime_probs, features
            )
            
            return regime_analysis
            
        except Exception as e:
            logger.error(f"❌ Advanced Regime Detection Fehler: {e}")
            return await self._fallback_regime_detection(features)
    
    async def _fallback_regime_detection(self, features: Dict[str, float]) -> RegimeAnalysis:
        """
        Fallback Regime Detection mit Regel-basierter Logik
        """
        try:
            # Rule-based Regime Detection
            momentum_5 = features.get('price_momentum_5', 0)
            momentum_20 = features.get('price_momentum_20', 0)
            volatility = features.get('realized_volatility', 0.02)
            volume_ratio = features.get('volume_ratio', 1.0)
            
            # Regime Classification Logic
            regime_scores = {}
            
            # Bull Trending
            if momentum_5 > 0.02 and momentum_20 > 0.05:
                regime_scores[MarketRegime.BULL_TRENDING] = 0.8 + min(0.2, momentum_20 * 4)
            else:
                regime_scores[MarketRegime.BULL_TRENDING] = max(0, momentum_20 * 5)
            
            # Bear Trending  
            if momentum_5 < -0.02 and momentum_20 < -0.05:
                regime_scores[MarketRegime.BEAR_TRENDING] = 0.8 + min(0.2, abs(momentum_20) * 4)
            else:
                regime_scores[MarketRegime.BEAR_TRENDING] = max(0, abs(momentum_20) * 5) if momentum_20 < 0 else 0
            
            # Sideways Range
            if abs(momentum_20) < 0.02 and volatility < 0.03:
                regime_scores[MarketRegime.SIDEWAYS_RANGE] = 0.7 + (0.3 * (1 - abs(momentum_20) * 25))
            else:
                regime_scores[MarketRegime.SIDEWAYS_RANGE] = max(0, 0.5 - abs(momentum_20) * 10)
            
            # High Volatility
            if volatility > 0.04:
                regime_scores[MarketRegime.HIGH_VOLATILITY] = min(1.0, volatility * 15)
            else:
                regime_scores[MarketRegime.HIGH_VOLATILITY] = volatility * 10
            
            # Low Volatility
            if volatility < 0.015:
                regime_scores[MarketRegime.LOW_VOLATILITY] = 1.0 - (volatility * 40)
            else:
                regime_scores[MarketRegime.LOW_VOLATILITY] = max(0, 0.8 - (volatility * 20))
            
            # Fill remaining regimes with low scores
            for regime in MarketRegime:
                if regime not in regime_scores:
                    regime_scores[regime] = 0.1
            
            # Normalize probabilities
            total_score = sum(regime_scores.values())
            regime_probabilities = {k: v/total_score for k, v in regime_scores.items()}
            
            # Select regime with highest probability
            current_regime = max(regime_probabilities, key=regime_probabilities.get)
            confidence = regime_probabilities[current_regime]
            
            # Enhanced Analysis
            regime_analysis = await self._create_enhanced_regime_analysis(
                current_regime, confidence, regime_probabilities, features
            )
            
            return regime_analysis
            
        except Exception as e:
            logger.error(f"❌ Fallback Regime Detection Fehler: {e}")
            return self._create_fallback_analysis()
    
    async def _create_enhanced_regime_analysis(self, current_regime: MarketRegime, 
                                             confidence: float, 
                                             regime_probabilities: Dict[MarketRegime, float],
                                             features: Dict[str, float]) -> RegimeAnalysis:
        """
        Erweiterte Regime-Analyse erstellen
        """
        try:
            # Optimal Strategies
            optimal_strategies = self.strategy_mappings.get(current_regime, ['conservative'])
            
            # Risk Level Assessment
            volatility = features.get('realized_volatility', 0.02)
            risk_level = 'medium'
            if volatility > 0.05:
                risk_level = 'high'
            elif volatility < 0.015:
                risk_level = 'low'
            
            # Expected Volatility
            expected_volatility = volatility * 1.1  # Slight increase
            
            # Historical Performance (mock)
            historical_performance = {'avg_return': 0.05, 'volatility': 0.15, 'sharpe_ratio': 0.8}
            
            return RegimeAnalysis(
                current_regime=current_regime,
                confidence=confidence,
                regime_probabilities=regime_probabilities,
                regime_duration=1,
                transition_probability=0.3,
                optimal_strategies=optimal_strategies,
                risk_level=risk_level,
                expected_volatility=expected_volatility,
                regime_features=features,
                historical_performance=historical_performance
            )
            
        except Exception as e:
            logger.error(f"❌ Enhanced Regime Analysis Fehler: {e}")
            return self._create_fallback_analysis()
    
    def _create_fallback_analysis(self) -> RegimeAnalysis:
        """
        Fallback Regime Analysis
        """
        return RegimeAnalysis(
            current_regime=MarketRegime.SIDEWAYS_RANGE,
            confidence=0.5,
            regime_probabilities={regime: 1.0/len(MarketRegime) for regime in MarketRegime},
            regime_duration=1,
            transition_probability=0.3,
            optimal_strategies=['conservative', 'balanced'],
            risk_level='medium',
            expected_volatility=0.025,
            regime_features={},
            historical_performance={'avg_return': 0.03, 'volatility': 0.15}
        )
    
    def get_regime_status(self) -> Dict:
        """
        Aktueller Status des Regime Detectors
        """
        return {
            'is_trained': self.is_trained,
            'current_regime': self.current_regime.value if self.current_regime else 'unknown',
            'regime_history_length': len(self.regime_history),
            'feature_history_length': len(self.feature_history),
            'advanced_ml_available': ADVANCED_ML_AVAILABLE,
            'model_types': 'HMM+GMM' if ADVANCED_ML_AVAILABLE else 'Rule-based',
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


# Integration Classes for Trading System
class RegimeBasedStrategySelector:
    """
    Strategy Selector basierend auf Market Regime
    """
    
    def __init__(self, regime_detector: AdvancedMarketRegimeDetector):
        self.regime_detector = regime_detector
        self.strategy_performance = {}
        
    async def select_optimal_strategy(self, market_data: Dict) -> Dict[str, Any]:
        """
        Optimale Strategie basierend on aktuellem Regime auswählen
        """
        try:
            # Current Regime Detection
            regime_analysis = await self.regime_detector.detect_current_regime(market_data)
            
            # Get Optimal Strategies
            optimal_strategies = regime_analysis.optimal_strategies
            
            # Strategy Weighting basierend auf Regime Confidence
            strategy_weights = {}
            base_weight = 1.0 / len(optimal_strategies)
            
            for strategy in optimal_strategies:
                # Weight by regime confidence and historical performance
                historical_perf = regime_analysis.historical_performance.get('sharpe_ratio', 0.5)
                strategy_weights[strategy] = base_weight * regime_analysis.confidence * historical_perf
            
            # Normalize weights
            total_weight = sum(strategy_weights.values())
            if total_weight > 0:
                strategy_weights = {k: v/total_weight for k, v in strategy_weights.items()}
            
            return {
                'regime': regime_analysis.current_regime.value,
                'regime_confidence': regime_analysis.confidence,
                'optimal_strategies': optimal_strategies,
                'strategy_weights': strategy_weights,
                'risk_level': regime_analysis.risk_level,
                'expected_volatility': regime_analysis.expected_volatility,
                'transition_probability': regime_analysis.transition_probability
            }
            
        except Exception as e:
            logger.error(f"❌ Regime-based Strategy Selection Fehler: {e}")
            return {
                'regime': 'unknown',
                'regime_confidence': 0.5,
                'optimal_strategies': ['conservative'],
                'strategy_weights': {'conservative': 1.0},
                'risk_level': 'medium'
            }


class RegimeAwareRiskManager:
    """
    Risk Manager mit Regime-Awareness
    """
    
    def __init__(self, regime_detector: AdvancedMarketRegimeDetector):
        self.regime_detector = regime_detector
        
    async def calculate_regime_adjusted_position_size(self, base_position_size: float, 
                                                    market_data: Dict) -> float:
        """
        Position Size basierend auf aktuellem Regime anpassen
        """
        try:
            regime_analysis = await self.regime_detector.detect_current_regime(market_data)
            
            # Risk Multipliers by Regime
            risk_multipliers = {
                MarketRegime.BULL_TRENDING: 1.2,      # Increase size in bull market
                MarketRegime.BEAR_TRENDING: 0.6,      # Reduce size in bear market
                MarketRegime.SIDEWAYS_RANGE: 1.0,     # Normal size in range
                MarketRegime.HIGH_VOLATILITY: 0.5,    # Reduce size in high vol
                MarketRegime.LOW_VOLATILITY: 1.3,     # Increase size in low vol
                MarketRegime.BULL_CORRECTION: 0.8,    # Cautious in correction
                MarketRegime.BEAR_RALLY: 0.7,         # Cautious in bear rally
                MarketRegime.BREAKOUT: 1.1,           # Slightly larger on breakout
                MarketRegime.CRISIS: 0.3              # Very small in crisis
            }
            
            multiplier = risk_multipliers.get(regime_analysis.current_regime, 1.0)
            
            # Adjust by regime confidence
            confidence_adjustment = 0.5 + (regime_analysis.confidence * 0.5)  # 0.5x to 1.0x
            
            # Final position size
            adjusted_size = base_position_size * multiplier * confidence_adjustment
            
            # Safety limits
            max_size = 0.2  # 20% max position
            min_size = 0.01  # 1% min position
            
            return max(min_size, min(max_size, adjusted_size))
            
        except Exception as e:
            logger.error(f"❌ Regime-adjusted Position Size Fehler: {e}")
            return base_position_size * 0.8  # Conservative fallback
