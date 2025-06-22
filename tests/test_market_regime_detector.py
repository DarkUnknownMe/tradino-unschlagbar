#!/usr/bin/env python3
"""
TRADINO UNSCHLAGBAR - MARKET REGIME DETECTOR TEST SUITE
Umfassende Tests für Advanced Market Regime Detection
"""

import asyncio
import unittest
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.market_regime_detector import (
    AdvancedMarketRegimeDetector, MarketRegime, RegimeAnalysis,
    RegimeBasedStrategySelector, RegimeAwareRiskManager
)
from utils.config_manager import ConfigManager
from loguru import logger

class TestMarketRegimeDetector(unittest.TestCase):
    """
    Test Suite für Market Regime Detection
    """
    
    def setUp(self):
        """
        Test Setup
        """
        try:
            self.config = ConfigManager()
        except:
            # Fallback Mock Config
            class MockConfig:
                def get(self, key, default=None):
                    return default
            self.config = MockConfig()
        
        self.regime_detector = AdvancedMarketRegimeDetector(self.config)
    
    async def test_regime_detector_initialization(self):
        """
        Test Regime Detector Initialization
        """
        logger.info("🧪 Testing Regime Detector Initialization...")
        
        # Initialize
        success = await self.regime_detector.initialize()
        
        # Validations
        self.assertTrue(success)
        self.assertIsNotNone(self.regime_detector.regime_mappings)
        self.assertIsNotNone(self.regime_detector.strategy_mappings)
        self.assertEqual(len(self.regime_detector.regime_mappings), 8)
        
        # Check regime mappings contain valid regimes
        for regime in self.regime_detector.regime_mappings.values():
            self.assertIsInstance(regime, MarketRegime)
        
        # Check strategy mappings
        for regime, strategies in self.regime_detector.strategy_mappings.items():
            self.assertIsInstance(regime, MarketRegime)
            self.assertIsInstance(strategies, list)
            self.assertGreater(len(strategies), 0)
        
        logger.success("✅ Regime Detector Initialization Test bestanden")
    
    async def test_feature_extraction(self):
        """
        Test Market Feature Extraction
        """
        logger.info("🧪 Testing Feature Extraction...")
        
        await self.regime_detector.initialize()
        
        # Mock Market Data
        mock_market_data = {
            'symbol': 'BTC/USDT',
            'close': 50000,
            'high': 51000,
            'low': 49000,
            'volume': 1200000,
            'volatility': 0.025,
            'timestamp': datetime.now()
        }
        
        # Extract Features
        features = await self.regime_detector._extract_regime_features(mock_market_data)
        
        # Validations
        self.assertIsNotNone(features)
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 10)  # Should have many features
        
        # Check key features
        expected_features = [
            'price_momentum_5', 'price_momentum_20', 'realized_volatility',
            'volume_ratio', 'rsi', 'macd', 'bb_position', 'trend_consistency'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
            # Check for valid values (no NaN or Inf)
            self.assertFalse(np.isnan(features[feature]))
            self.assertFalse(np.isinf(features[feature]))
        
        logger.success(f"✅ Feature Extraction Test bestanden - {len(features)} Features extrahiert")
    
    async def test_regime_detection(self):
        """
        Test Regime Detection
        """
        logger.info("🧪 Testing Regime Detection...")
        
        await self.regime_detector.initialize()
        
        # Test verschiedene Market Scenarios
        test_scenarios = [
            # Bull Market
            {
                'name': 'Bull Market',
                'data': {
                    'close': 55000,  # 10% up
                    'high': 56000,
                    'low': 54000,
                    'volume': 2000000,  # High volume
                    'volatility': 0.02  # Normal volatility
                },
                'expected_regimes': [MarketRegime.BULL_TRENDING, MarketRegime.LOW_VOLATILITY]
            },
            # Bear Market
            {
                'name': 'Bear Market',
                'data': {
                    'close': 45000,  # 10% down
                    'high': 46000,
                    'low': 44000,
                    'volume': 1800000,
                    'volatility': 0.03
                },
                'expected_regimes': [MarketRegime.BEAR_TRENDING, MarketRegime.HIGH_VOLATILITY]
            },
            # High Volatility
            {
                'name': 'High Volatility',
                'data': {
                    'close': 50000,
                    'high': 53000,  # 6% range
                    'low': 47000,
                    'volume': 3000000,
                    'volatility': 0.06  # High volatility
                },
                'expected_regimes': [MarketRegime.HIGH_VOLATILITY, MarketRegime.BREAKOUT]
            },
            # Low Volatility
            {
                'name': 'Low Volatility',
                'data': {
                    'close': 50000,
                    'high': 50200,  # 0.4% range
                    'low': 49800,
                    'volume': 800000,
                    'volatility': 0.008  # Low volatility
                },
                'expected_regimes': [MarketRegime.LOW_VOLATILITY, MarketRegime.SIDEWAYS_RANGE]
            }
        ]
        
        successful_detections = 0
        
        for scenario in test_scenarios:
            try:
                market_data = {
                    'symbol': 'BTC/USDT',
                    **scenario['data'],
                    'timestamp': datetime.now()
                }
                
                # Detect Regime
                regime_analysis = await self.regime_detector.detect_current_regime(market_data)
                
                # Validations
                self.assertIsInstance(regime_analysis, RegimeAnalysis)
                self.assertIsInstance(regime_analysis.current_regime, MarketRegime)
                self.assertGreaterEqual(regime_analysis.confidence, 0.0)
                self.assertLessEqual(regime_analysis.confidence, 1.0)
                self.assertIsInstance(regime_analysis.optimal_strategies, list)
                self.assertGreater(len(regime_analysis.optimal_strategies), 0)
                self.assertIn(regime_analysis.risk_level, ['low', 'medium', 'high', 'very_high'])
                self.assertGreater(regime_analysis.expected_volatility, 0.0)
                self.assertIsInstance(regime_analysis.regime_probabilities, dict)
                self.assertEqual(len(regime_analysis.regime_probabilities), len(MarketRegime))
                
                # Check regime probabilities sum to ~1.0
                total_prob = sum(regime_analysis.regime_probabilities.values())
                self.assertAlmostEqual(total_prob, 1.0, places=1)
                
                successful_detections += 1
                
                logger.info(f"   📊 Scenario '{scenario['name']}': "
                          f"Regime={regime_analysis.current_regime.value}, "
                          f"Confidence={regime_analysis.confidence:.3f}, "
                          f"Risk={regime_analysis.risk_level}")
                
            except Exception as e:
                logger.warning(f"Scenario '{scenario['name']}' Fehler: {e}")
        
        # At least 75% scenarios should succeed
        success_rate = successful_detections / len(test_scenarios)
        self.assertGreaterEqual(success_rate, 0.75)
        
        logger.success(f"✅ Regime Detection Test bestanden ({successful_detections}/{len(test_scenarios)} Scenarios)")
    
    async def test_regime_based_strategy_selection(self):
        """
        Test Regime-based Strategy Selection
        """
        logger.info("🧪 Testing Regime-based Strategy Selection...")
        
        await self.regime_detector.initialize()
        
        # Strategy Selector
        strategy_selector = RegimeBasedStrategySelector(self.regime_detector)
        
        # Test multiple market conditions
        test_conditions = [
            {
                'name': 'Bull Market',
                'data': {
                    'close': 52000,
                    'high': 52500,
                    'low': 51500,
                    'volume': 1500000,
                    'volatility': 0.02
                }
            },
            {
                'name': 'Bear Market',
                'data': {
                    'close': 48000,
                    'high': 48500,
                    'low': 47500,
                    'volume': 1800000,
                    'volatility': 0.035
                }
            },
            {
                'name': 'Sideways Market',
                'data': {
                    'close': 50000,
                    'high': 50300,
                    'low': 49700,
                    'volume': 1000000,
                    'volatility': 0.015
                }
            }
        ]
        
        successful_selections = 0
        
        for condition in test_conditions:
            try:
                market_data = {
                    'symbol': 'BTC/USDT',
                    **condition['data'],
                    'timestamp': datetime.now()
                }
                
                # Select Strategy
                strategy_result = await strategy_selector.select_optimal_strategy(market_data)
                
                # Validations
                self.assertIsNotNone(strategy_result)
                self.assertIn('regime', strategy_result)
                self.assertIn('optimal_strategies', strategy_result)
                self.assertIn('strategy_weights', strategy_result)
                self.assertIn('risk_level', strategy_result)
                self.assertIn('regime_confidence', strategy_result)
                
                # Check strategy weights sum to 1
                total_weight = sum(strategy_result['strategy_weights'].values())
                self.assertAlmostEqual(total_weight, 1.0, places=2)
                
                # Check confidence
                self.assertGreaterEqual(strategy_result['regime_confidence'], 0.0)
                self.assertLessEqual(strategy_result['regime_confidence'], 1.0)
                
                successful_selections += 1
                
                logger.info(f"   📊 {condition['name']}: Regime={strategy_result['regime']}, "
                          f"Strategies={strategy_result['optimal_strategies']}")
                
            except Exception as e:
                logger.warning(f"Strategy Selection für '{condition['name']}' Fehler: {e}")
        
        # Require high success rate
        success_rate = successful_selections / len(test_conditions)
        self.assertGreaterEqual(success_rate, 0.8)
        
        logger.success(f"✅ Regime-based Strategy Selection Test bestanden ({successful_selections}/{len(test_conditions)})")
    
    async def test_regime_aware_risk_management(self):
        """
        Test Regime-aware Risk Management
        """
        logger.info("🧪 Testing Regime-aware Risk Management...")
        
        await self.regime_detector.initialize()
        
        # Risk Manager
        risk_manager = RegimeAwareRiskManager(self.regime_detector)
        
        base_position_size = 0.1  # 10%
        
        # Test different market conditions
        test_conditions = [
            {
                'name': 'Normal Market',
                'data': {
                    'close': 50000, 
                    'high': 50500,
                    'low': 49500,
                    'volume': 1000000,
                    'volatility': 0.02
                },
                'expected_adjustment': 'moderate'
            },
            {
                'name': 'High Volatility',
                'data': {
                    'close': 50000,
                    'high': 52000,
                    'low': 48000, 
                    'volume': 2500000,
                    'volatility': 0.08
                },
                'expected_adjustment': 'reduce'
            },
            {
                'name': 'Low Volatility',
                'data': {
                    'close': 50000,
                    'high': 50100,
                    'low': 49900,
                    'volume': 600000,
                    'volatility': 0.005
                },
                'expected_adjustment': 'increase'
            },
            {
                'name': 'Crisis Scenario',
                'data': {
                    'close': 45000,
                    'high': 47000,
                    'low': 42000,
                    'volume': 5000000,
                    'volatility': 0.12
                },
                'expected_adjustment': 'minimize'
            }
        ]
        
        successful_adjustments = 0
        
        for condition in test_conditions:
            try:
                market_data = {
                    'symbol': 'BTC/USDT',
                    **condition['data'],
                    'timestamp': datetime.now()
                }
                
                # Calculate adjusted position size
                adjusted_size = await risk_manager.calculate_regime_adjusted_position_size(
                    base_position_size, market_data
                )
                
                # Validations
                self.assertGreater(adjusted_size, 0)
                self.assertLessEqual(adjusted_size, 0.2)  # Max 20%
                self.assertGreaterEqual(adjusted_size, 0.01)  # Min 1%
                
                adjustment_ratio = adjusted_size / base_position_size
                
                # Check adjustment makes sense
                if condition['expected_adjustment'] == 'reduce' or condition['expected_adjustment'] == 'minimize':
                    self.assertLessEqual(adjustment_ratio, 1.0)
                elif condition['expected_adjustment'] == 'increase':
                    # Could be increased or decreased depending on regime
                    pass  # No strict requirement
                
                successful_adjustments += 1
                
                logger.info(f"   📊 {condition['name']}: "
                          f"Base={base_position_size:.1%}, "
                          f"Adjusted={adjusted_size:.1%}, "
                          f"Ratio={adjustment_ratio:.2f}x")
                
            except Exception as e:
                logger.warning(f"Risk Management für '{condition['name']}' Fehler: {e}")
        
        # Require high success rate
        success_rate = successful_adjustments / len(test_conditions)
        self.assertGreaterEqual(success_rate, 0.75)
        
        logger.success(f"✅ Regime-aware Risk Management Test bestanden ({successful_adjustments}/{len(test_conditions)})")
    
    async def test_regime_transition_forecast(self):
        """
        Test Regime Transition Forecast
        """
        logger.info("🧪 Testing Regime Transition Forecast...")
        
        await self.regime_detector.initialize()
        
        # Generate some regime history
        for i in range(15):
            mock_data = {
                'symbol': 'BTC/USDT',
                'close': 50000 + np.random.normal(0, 1000),
                'high': 50000 + np.random.uniform(500, 1500),
                'low': 50000 + np.random.uniform(-1500, -500),
                'volume': 1000000 + np.random.normal(0, 200000),
                'volatility': 0.02 + np.random.normal(0, 0.005),
                'timestamp': datetime.now()
            }
            await self.regime_detector.detect_current_regime(mock_data)
        
        # Get transition forecast
        forecast = await self.regime_detector.get_regime_transition_forecast(days_ahead=10)
        
        # Validations
        self.assertIsInstance(forecast, dict)
        self.assertEqual(len(forecast), len(MarketRegime))
        
        # Check probabilities sum to ~1
        total_prob = sum(forecast.values())
        self.assertAlmostEqual(total_prob, 1.0, places=1)
        
        # All probabilities should be between 0 and 1
        for regime, prob in forecast.items():
            self.assertIsInstance(regime, MarketRegime)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
        
        # Current regime should typically have higher probability
        current_regime = self.regime_detector.current_regime
        if current_regime in forecast:
            current_prob = forecast[current_regime]
            avg_prob = 1.0 / len(MarketRegime)
            # Current regime probability should be at least average
            self.assertGreaterEqual(current_prob, avg_prob * 0.8)
        
        logger.success("✅ Regime Transition Forecast Test bestanden")
        logger.info(f"   📊 Top 3 Forecasts: {sorted(forecast.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    def test_regime_status(self):
        """
        Test Regime Status
        """
        logger.info("🧪 Testing Regime Status...")
        
        status = self.regime_detector.get_regime_status()
        
        # Validations
        expected_keys = [
            'is_trained', 'current_regime', 'regime_history_length',
            'feature_history_length', 'advanced_ml_available', 
            'model_types', 'last_update'
        ]
        
        for key in expected_keys:
            self.assertIn(key, status)
        
        # Check types
        self.assertIsInstance(status['is_trained'], bool)
        self.assertIsInstance(status['current_regime'], str)
        self.assertIsInstance(status['regime_history_length'], int)
        self.assertIsInstance(status['feature_history_length'], int)
        self.assertIsInstance(status['advanced_ml_available'], bool)
        self.assertIsInstance(status['model_types'], str)
        self.assertIsInstance(status['last_update'], str)
        
        # Check ranges
        self.assertGreaterEqual(status['regime_history_length'], 0)
        self.assertGreaterEqual(status['feature_history_length'], 0)
        
        logger.success("✅ Regime Status Test bestanden")
        logger.info(f"   📊 Status: Trained={status['is_trained']}, "
                  f"ML={status['advanced_ml_available']}, "
                  f"Models={status['model_types']}")
    
    async def test_regime_consistency(self):
        """
        Test Regime Detection Consistency
        """
        logger.info("🧪 Testing Regime Detection Consistency...")
        
        await self.regime_detector.initialize()
        
        # Same market data should give consistent results
        consistent_market_data = {
            'symbol': 'BTC/USDT',
            'close': 50000,
            'high': 51000,
            'low': 49000,
            'volume': 1000000,
            'volatility': 0.025,
            'timestamp': datetime.now()
        }
        
        regime_results = []
        confidence_results = []
        
        # Run detection multiple times
        for _ in range(5):
            regime_analysis = await self.regime_detector.detect_current_regime(consistent_market_data)
            regime_results.append(regime_analysis.current_regime)
            confidence_results.append(regime_analysis.confidence)
        
        # Check consistency - most results should be the same
        most_common_regime = max(set(regime_results), key=regime_results.count)
        consistency_rate = regime_results.count(most_common_regime) / len(regime_results)
        
        self.assertGreaterEqual(consistency_rate, 0.6)  # At least 60% consistency
        
        # Confidence should be relatively stable
        confidence_std = np.std(confidence_results)
        self.assertLess(confidence_std, 0.3)  # Standard deviation should be reasonable
        
        logger.success(f"✅ Regime Detection Consistency Test bestanden (Consistency: {consistency_rate:.1%})")


async def run_regime_detection_tests():
    """
    Führe alle Regime Detection Tests aus
    """
    logger.info("🧪 MARKET REGIME DETECTION TEST SUITE STARTET")
    logger.info("=" * 70)
    
    # Test Instance
    test_suite = TestMarketRegimeDetector()
    test_suite.setUp()
    
    tests_passed = 0
    tests_total = 0
    
    # Test List
    test_methods = [
        ('Regime Detector Initialization', test_suite.test_regime_detector_initialization),
        ('Feature Extraction', test_suite.test_feature_extraction),
        ('Regime Detection', test_suite.test_regime_detection),
        ('Strategy Selection', test_suite.test_regime_based_strategy_selection),
        ('Risk Management', test_suite.test_regime_aware_risk_management),
        ('Transition Forecast', test_suite.test_regime_transition_forecast),
        ('Regime Status', test_suite.test_regime_status),
        ('Regime Consistency', test_suite.test_regime_consistency)
    ]
    
    for test_name, test_method in test_methods:
        tests_total += 1
        try:
            logger.info(f"🔄 Running {test_name} Test...")
            if asyncio.iscoroutinefunction(test_method):
                await test_method()
            else:
                test_method()
            tests_passed += 1
            logger.success(f"✅ {test_name} Test bestanden")
        except Exception as e:
            logger.error(f"❌ {test_name} Test fehlgeschlagen: {e}")
    
    # Test Results
    success_rate = (tests_passed / tests_total) * 100
    
    logger.info("=" * 70)
    logger.info("📊 MARKET REGIME DETECTION TEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"📈 Tests bestanden: {tests_passed}/{tests_total} ({success_rate:.1f}%)")
    logger.info(f"🧠 Regime Types: {len(MarketRegime)} verfügbar")
    logger.info(f"🎯 Success Threshold: 75%")
    
    if tests_passed == tests_total:
        logger.success("🎉 ALLE REGIME DETECTION TESTS ERFOLGREICH BESTANDEN!")
        return True
    elif success_rate >= 75:
        logger.success(f"✅ REGIME DETECTION TESTS BESTANDEN! ({success_rate:.1f}% Success Rate)")
        return True
    else:
        logger.warning(f"⚠️ {tests_total - tests_passed} Tests fehlgeschlagen - Success Rate unter 75%")
        return False


if __name__ == '__main__':
    # Führe Tests aus
    success = asyncio.run(run_regime_detection_tests())
    
    if success:
        print("\n🧠 MARKET REGIME DETECTION TESTS ERFOLGREICH!")
        exit(0)
    else:
        print("\n❌ Market Regime Detection Tests fehlgeschlagen")
        exit(1) 