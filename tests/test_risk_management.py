"""
Unit Tests für Risk Management System
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.risk_management_system import RiskManagementSystem


class TestRiskManagementSystem(unittest.TestCase):
    """Test cases for Risk Management System"""
    
    def setUp(self):
        """Setup test environment"""
        self.risk_manager = RiskManagementSystem()
        self.test_portfolio_value = 10000.0
        self.test_symbol = "BTCUSDT"
        self.test_price = 45000.0
    
    def test_risk_manager_initialization(self):
        """Test risk manager initialization"""
        self.assertIsInstance(self.risk_manager, RiskManagementSystem)
        self.assertGreater(self.risk_manager.max_portfolio_risk, 0)
        self.assertGreater(self.risk_manager.max_position_risk, 0)
    
    def test_calculate_position_size(self):
        """Test position size calculation"""
        risk_percentage = 2.0  # 2% risk
        entry_price = 45000.0
        stop_loss = 44000.0  # 1000 point stop loss
        
        position_size = self.risk_manager.calculate_position_size(
            self.test_portfolio_value, risk_percentage, entry_price, stop_loss
        )
        
        # Expected calculation:
        # Risk amount = 10000 * 0.02 = 200
        # Risk per unit = 45000 - 44000 = 1000
        # Position size = 200 / 1000 = 0.2
        expected_size = 0.2
        
        self.assertAlmostEqual(position_size, expected_size, places=4)
        self.assertGreater(position_size, 0)
    
    def test_calculate_position_size_sell(self):
        """Test position size calculation for sell orders"""
        risk_percentage = 2.0  # 2% risk
        entry_price = 45000.0
        stop_loss = 46000.0  # 1000 point stop loss (above entry for sell)
        
        position_size = self.risk_manager.calculate_position_size(
            self.test_portfolio_value, risk_percentage, entry_price, stop_loss
        )
        
        # Expected calculation:
        # Risk amount = 10000 * 0.02 = 200
        # Risk per unit = abs(45000 - 46000) = 1000
        # Position size = 200 / 1000 = 0.2
        expected_size = 0.2
        
        self.assertAlmostEqual(position_size, expected_size, places=4)
        self.assertGreater(position_size, 0)
    
    def test_calculate_position_size_edge_cases(self):
        """Test position size calculation edge cases"""
        # Zero risk percentage
        with self.assertRaises(ValueError):
            self.risk_manager.calculate_position_size(
                self.test_portfolio_value, 0.0, 45000.0, 44000.0
            )
        
        # Entry price equals stop loss
        with self.assertRaises(ValueError):
            self.risk_manager.calculate_position_size(
                self.test_portfolio_value, 2.0, 45000.0, 45000.0
            )
        
        # Negative portfolio value
        with self.assertRaises(ValueError):
            self.risk_manager.calculate_position_size(
                -1000.0, 2.0, 45000.0, 44000.0
            )
    
    def test_validate_trade_risk(self):
        """Test trade risk validation"""
        # Valid trade
        trade = {
            "symbol": self.test_symbol,
            "side": "buy",
            "amount": 0.1,
            "price": self.test_price,
            "stop_loss": 44000.0
        }
        
        is_valid = self.risk_manager.validate_trade_risk(trade, self.test_portfolio_value)
        self.assertTrue(is_valid)
    
    def test_validate_trade_risk_excessive(self):
        """Test trade risk validation with excessive risk"""
        # Trade with too much risk
        trade = {
            "symbol": self.test_symbol,
            "side": "buy",
            "amount": 5.0,  # Very large position
            "price": self.test_price,
            "stop_loss": 44000.0
        }
        
        is_valid = self.risk_manager.validate_trade_risk(trade, self.test_portfolio_value)
        self.assertFalse(is_valid)
    
    def test_calculate_portfolio_risk(self):
        """Test portfolio risk calculation"""
        positions = [
            {
                "symbol": "BTCUSDT",
                "amount": 0.1,
                "entry_price": 45000.0,
                "current_price": 44000.0,
                "stop_loss": 43000.0
            },
            {
                "symbol": "ETHUSDT", 
                "amount": 1.0,
                "entry_price": 3000.0,
                "current_price": 2900.0,
                "stop_loss": 2800.0
            }
        ]
        
        portfolio_risk = self.risk_manager.calculate_portfolio_risk(
            positions, self.test_portfolio_value
        )
        
        self.assertIsInstance(portfolio_risk, float)
        self.assertGreaterEqual(portfolio_risk, 0)
        self.assertLessEqual(portfolio_risk, 100)
    
    def test_check_correlation_risk(self):
        """Test correlation risk checking"""
        existing_positions = [
            {"symbol": "BTCUSDT", "amount": 0.1},
            {"symbol": "ETHUSDT", "amount": 1.0}
        ]
        
        # Test adding correlated position (another crypto)
        new_trade = {
            "symbol": "ADAUSDT",
            "amount": 10.0
        }
        
        correlation_risk = self.risk_manager.check_correlation_risk(
            existing_positions, new_trade
        )
        
        self.assertIsInstance(correlation_risk, float)
        self.assertGreaterEqual(correlation_risk, 0)
        self.assertLessEqual(correlation_risk, 1)
    
    def test_calculate_var(self):
        """Test Value at Risk calculation"""
        returns = [-0.02, 0.01, -0.015, 0.025, -0.01, 0.03, -0.005]
        confidence_level = 0.95
        
        var = self.risk_manager.calculate_var(returns, confidence_level)
        
        self.assertIsInstance(var, float)
        self.assertLess(var, 0)  # VaR should be negative (loss)
    
    def test_update_risk_limits(self):
        """Test risk limits updating"""
        new_limits = {
            "max_portfolio_risk": 15.0,
            "max_position_risk": 3.0,
            "max_daily_trades": 20
        }
        
        self.risk_manager.update_risk_limits(new_limits)
        
        self.assertEqual(self.risk_manager.max_portfolio_risk, 15.0)
        self.assertEqual(self.risk_manager.max_position_risk, 3.0)
        self.assertEqual(self.risk_manager.max_daily_trades, 20)
    
    def test_daily_trade_limit(self):
        """Test daily trade limit enforcement"""
        # Mock current date trades
        today = datetime.now().date()
        
        # Add trades for today
        for i in range(10):
            self.risk_manager.trade_history.append({
                "timestamp": datetime.now(),
                "symbol": f"TEST{i}USDT",
                "amount": 0.1
            })
        
        # Set low daily limit
        self.risk_manager.max_daily_trades = 5
        
        # Check if daily limit is exceeded
        can_trade = self.risk_manager.can_place_new_trade()
        self.assertFalse(can_trade)
    
    def test_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        portfolio_values = [10000, 9500, 9000, 9200, 8800, 9100, 9500, 10200]
        
        max_drawdown = self.risk_manager.calculate_max_drawdown(portfolio_values)
        
        # Maximum drawdown should be (10000 - 8800) / 10000 = 12%
        expected_drawdown = 12.0
        
        self.assertAlmostEqual(max_drawdown, expected_drawdown, places=1)
    
    def test_risk_metrics(self):
        """Test risk metrics calculation"""
        returns = [0.01, -0.02, 0.015, -0.01, 0.025, -0.005]
        
        metrics = self.risk_manager.calculate_risk_metrics(returns)
        
        self.assertIn("volatility", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("var_95", metrics)
        self.assertIn("max_drawdown", metrics)
        
        self.assertIsInstance(metrics["volatility"], float)
        self.assertIsInstance(metrics["sharpe_ratio"], float)
        self.assertGreaterEqual(metrics["volatility"], 0)


class TestRiskManagementIntegration(unittest.TestCase):
    """Integration tests for Risk Management System"""
    
    def setUp(self):
        """Setup integration test environment"""
        self.risk_manager = RiskManagementSystem()
        self.portfolio_tracker = Mock()
        self.notification_service = Mock()
    
    def test_real_time_risk_monitoring(self):
        """Test real-time risk monitoring"""
        # Mock portfolio data
        portfolio_data = {
            "total_value": 10000.0,
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "amount": 0.1,
                    "entry_price": 45000.0,
                    "current_price": 44000.0,
                    "unrealized_pnl": -100.0
                }
            ]
        }
        
        # Test risk monitoring
        risk_status = self.risk_manager.monitor_real_time_risk(portfolio_data)
        
        self.assertIn("status", risk_status)
        self.assertIn("portfolio_risk", risk_status)
        self.assertIn("recommendations", risk_status)
    
    def test_emergency_stop_conditions(self):
        """Test emergency stop conditions"""
        # Simulate severe portfolio loss
        portfolio_data = {
            "total_value": 7000.0,  # 30% loss from initial 10000
            "daily_pnl": -3000.0
        }
        
        should_stop = self.risk_manager.check_emergency_stop_conditions(portfolio_data)
        
        # Should trigger emergency stop for large losses
        self.assertTrue(should_stop)
    
    def test_position_sizing_with_volatility(self):
        """Test position sizing adjusted for volatility"""
        # High volatility asset
        asset_volatility = 0.05  # 5% daily volatility
        base_position_size = 0.1
        
        adjusted_size = self.risk_manager.adjust_position_for_volatility(
            base_position_size, asset_volatility
        )
        
        # Should reduce position size for high volatility
        self.assertLess(adjusted_size, base_position_size)
        self.assertGreater(adjusted_size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2) 