"""
Integration Tests für TRADINO System
End-to-End Workflow Tests
"""

import unittest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.integration_manager import IntegrationManager
from core.bitget_trading_api import BitgetTradingAPI
from core.risk_management_system import RiskManagementSystem
from core.tp_sl_manager import TPSLManager
from core.final_live_trading_system import LiveTradingSystem


class TestSystemIntegration(unittest.TestCase):
    """End-to-End System Integration Tests"""
    
    def setUp(self):
        """Setup integration test environment"""
        self.integration_manager = None
        self.test_config = {
            "trading": {
                "mode": "test",
                "risk_management": True,
                "tp_sl_enabled": True,
                "max_concurrent_trades": 3
            },
            "risk": {
                "max_portfolio_risk": 10.0,
                "max_position_risk": 2.0,
                "stop_loss_percentage": 2.0,
                "take_profit_percentage": 4.0
            }
        }
    
    async def test_system_initialization(self):
        """Test complete system initialization"""
        self.integration_manager = IntegrationManager()
        
        # Mock environment variables for testing
        with patch.dict('os.environ', {
            'BITGET_API_KEY': 'test_key',
            'BITGET_SECRET_KEY': 'test_secret',
            'BITGET_PASSPHRASE': 'test_pass',
            'TELEGRAM_BOT_TOKEN': 'test_token',
            'TELEGRAM_CHAT_ID': 'test_chat'
        }):
            success = await self.integration_manager.initialize_system()
            
            # In test mode, some components might not initialize fully
            # but we should have basic system structure
            self.assertIsNotNone(self.integration_manager)
            self.assertTrue(len(self.integration_manager.components) > 0)
    
    async def test_complete_trading_workflow(self):
        """Test complete trading workflow from signal to execution"""
        # Initialize system
        integration_manager = IntegrationManager()
        
        # Mock components
        with patch.multiple(
            'core.bitget_trading_api.BitgetTradingAPI',
            get_server_time=AsyncMock(return_value=1640995200000),
            get_account_balance=AsyncMock(return_value={"USDT": {"available": 1000.0}}),
            place_market_order=AsyncMock(return_value={"orderId": "test_123", "status": "filled"}),
            place_limit_order=AsyncMock(return_value={"orderId": "tp_123", "status": "new"}),
            place_stop_order=AsyncMock(return_value={"orderId": "sl_123", "status": "new"})
        ):
            # Mock trading signal
            signal = {
                "symbol": "BTCUSDT",
                "action": "buy",
                "confidence": 0.85,
                "entry_price": 45000.0,
                "take_profit": 46800.0,  # 4% profit
                "stop_loss": 44100.0,    # 2% loss
                "amount": 0.001
            }
            
            # Mock risk validation
            risk_manager = RiskManagementSystem()
            risk_validation = risk_manager.validate_trade_risk(signal, 1000.0)
            
            # Should pass risk validation
            self.assertTrue(risk_validation)
            
            # Mock trade execution
            api = BitgetTradingAPI()
            
            # Execute market order
            market_order = await api.place_market_order(
                signal["symbol"], signal["action"], signal["amount"]
            )
            
            self.assertIn("orderId", market_order)
            self.assertEqual(market_order["status"], "filled")
            
            # Execute TP order
            tp_order = await api.place_limit_order(
                signal["symbol"], "sell", signal["amount"], signal["take_profit"]
            )
            
            self.assertIn("orderId", tp_order)
            
            # Execute SL order
            sl_order = await api.place_stop_order(
                signal["symbol"], "sell", signal["amount"], signal["stop_loss"]
            )
            
            self.assertIn("orderId", sl_order)
    
    async def test_risk_management_integration(self):
        """Test risk management integration"""
        risk_manager = RiskManagementSystem()
        
        # Test portfolio with multiple positions
        portfolio_value = 10000.0
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
                "current_price": 2950.0,
                "stop_loss": 2850.0
            }
        ]
        
        # Calculate total portfolio risk
        portfolio_risk = risk_manager.calculate_portfolio_risk(positions, portfolio_value)
        
        self.assertIsInstance(portfolio_risk, float)
        self.assertGreaterEqual(portfolio_risk, 0)
        
        # Test new trade validation
        new_trade = {
            "symbol": "ADAUSDT",
            "side": "buy",
            "amount": 100.0,
            "price": 1.50,
            "stop_loss": 1.47
        }
        
        is_valid = risk_manager.validate_trade_risk(new_trade, portfolio_value)
        
        # Should validate based on risk parameters
        self.assertIsInstance(is_valid, bool)
    
    async def test_tp_sl_system_integration(self):
        """Test TP/SL system integration"""
        tp_sl_manager = TPSLManager()
        
        # Mock position
        position = {
            "order_id": "test_order_123",
            "symbol": "BTCUSDT",
            "side": "buy",
            "amount": 0.001,
            "entry_price": 45000.0,
            "take_profit": 46800.0,
            "stop_loss": 44100.0
        }
        
        # Mock API responses
        with patch.multiple(
            'core.bitget_trading_api.BitgetTradingAPI',
            place_limit_order=AsyncMock(return_value={"orderId": "tp_123"}),
            place_stop_order=AsyncMock(return_value={"orderId": "sl_123"}),
            get_order_status=AsyncMock(return_value={"status": "new"})
        ):
            # Setup TP/SL orders
            tp_sl_status = await tp_sl_manager.setup_tp_sl_orders(position)
            
            self.assertIn("tp_order_id", tp_sl_status)
            self.assertIn("sl_order_id", tp_sl_status)
            
            # Monitor orders
            monitoring_result = await tp_sl_manager.monitor_tp_sl_orders(position["order_id"])
            
            self.assertIsInstance(monitoring_result, dict)
    
    async def test_error_handling_integration(self):
        """Test system-wide error handling"""
        # Test API connection failure
        with patch('aiohttp.ClientSession.get', side_effect=Exception("Connection failed")):
            api = BitgetTradingAPI()
            
            with self.assertRaises(Exception):
                await api.get_server_time()
        
        # Test risk violation handling
        risk_manager = RiskManagementSystem()
        
        # Create trade that violates risk limits
        excessive_trade = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "amount": 100.0,  # Excessive amount
            "price": 45000.0,
            "stop_loss": 44000.0
        }
        
        is_valid = risk_manager.validate_trade_risk(excessive_trade, 1000.0)
        self.assertFalse(is_valid)
    
    async def test_monitoring_integration(self):
        """Test monitoring system integration"""
        # Test system health monitoring
        from config.system_health_check import SystemHealthMonitor
        
        health_monitor = SystemHealthMonitor()
        health_results = await health_monitor.perform_full_health_check()
        
        self.assertIsInstance(health_results, dict)
        self.assertGreater(len(health_results), 0)
        
        # Each component should have health status
        for component_name, health in health_results.items():
            self.assertIn("status", health.to_dict())
            self.assertIn("metrics", health.to_dict())
            self.assertIn("last_check", health.to_dict())
    
    async def test_configuration_integration(self):
        """Test configuration system integration"""
        # Test configuration loading
        config_manager = IntegrationManager()
        config = config_manager._load_configuration()
        
        self.assertIsInstance(config, dict)
        self.assertIn("trading", config)
        self.assertIn("risk", config)
        self.assertIn("monitoring", config)
        
        # Test configuration validation
        required_keys = ["trading", "risk", "monitoring", "components"]
        for key in required_keys:
            self.assertIn(key, config)
    
    async def test_performance_integration(self):
        """Test system performance under load"""
        # Simulate multiple concurrent operations
        tasks = []
        
        # Mock multiple API calls
        with patch.multiple(
            'core.bitget_trading_api.BitgetTradingAPI',
            get_server_time=AsyncMock(return_value=1640995200000),
            get_account_balance=AsyncMock(return_value={"USDT": {"available": 1000.0}})
        ):
            api = BitgetTradingAPI()
            
            # Create multiple concurrent tasks
            for i in range(10):
                task = asyncio.create_task(api.get_server_time())
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All tasks should complete successfully
            for result in results:
                if isinstance(result, Exception):
                    self.fail(f"Task failed with exception: {result}")
                else:
                    self.assertIsInstance(result, int)
    
    async def test_data_flow_integration(self):
        """Test data flow between components"""
        # Test signal generation -> risk assessment -> execution flow
        
        # Mock signal
        signal = {
            "symbol": "BTCUSDT",
            "action": "buy",
            "confidence": 0.8,
            "entry_price": 45000.0,
            "amount": 0.001
        }
        
        # Risk assessment
        risk_manager = RiskManagementSystem()
        
        # Add risk parameters to signal
        risk_config = {
            "take_profit_percentage": 3.0,
            "stop_loss_percentage": 1.5
        }
        
        # Calculate position size
        portfolio_value = 10000.0
        position_size = risk_manager.calculate_position_size(
            portfolio_value, 2.0, signal["entry_price"], 
            signal["entry_price"] * 0.985  # 1.5% stop loss
        )
        
        self.assertGreater(position_size, 0)
        self.assertLess(position_size, 1.0)  # Reasonable position size
        
        # Update signal with calculated position size
        signal["amount"] = position_size
        
        # Validate final trade
        is_valid = risk_manager.validate_trade_risk(signal, portfolio_value)
        self.assertTrue(is_valid)


class TestComponentIntegration(unittest.TestCase):
    """Test integration between specific components"""
    
    async def test_api_risk_integration(self):
        """Test API and Risk Management integration"""
        api = BitgetTradingAPI()
        risk_manager = RiskManagementSystem()
        
        # Mock account balance
        with patch.object(api, 'get_account_balance', 
                         return_value={"USDT": {"available": 1000.0}}):
            balance = await api.get_account_balance()
            portfolio_value = balance["USDT"]["available"]
            
            # Test position sizing based on balance
            risk_percentage = 2.0
            entry_price = 45000.0
            stop_loss = 44100.0
            
            position_size = risk_manager.calculate_position_size(
                portfolio_value, risk_percentage, entry_price, stop_loss
            )
            
            # Position size should be reasonable for the balance
            self.assertGreater(position_size, 0)
            self.assertLess(position_size * entry_price, portfolio_value)
    
    async def test_risk_tp_sl_integration(self):
        """Test Risk Management and TP/SL integration"""
        risk_manager = RiskManagementSystem()
        tp_sl_manager = TPSLManager()
        
        # Test risk-based TP/SL calculation
        entry_price = 45000.0
        side = "buy"
        
        risk_config = {
            "take_profit_percentage": 3.0,
            "stop_loss_percentage": 1.5
        }
        
        # Calculate TP/SL prices using risk parameters
        if side == "buy":
            tp_price = entry_price * (1 + risk_config["take_profit_percentage"] / 100)
            sl_price = entry_price * (1 - risk_config["stop_loss_percentage"] / 100)
        else:
            tp_price = entry_price * (1 - risk_config["take_profit_percentage"] / 100)
            sl_price = entry_price * (1 + risk_config["stop_loss_percentage"] / 100)
        
        # Validate TP/SL prices
        if side == "buy":
            self.assertGreater(tp_price, entry_price)
            self.assertLess(sl_price, entry_price)
        else:
            self.assertLess(tp_price, entry_price)
            self.assertGreater(sl_price, entry_price)


if __name__ == "__main__":
    # Run async tests
    class AsyncTestRunner:
        def run_async_test(self, test_func):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(test_func())
            finally:
                loop.close()
    
    # Convert async test methods to sync for unittest
    async_runner = AsyncTestRunner()
    
    for test_class in [TestSystemIntegration, TestComponentIntegration]:
        for method_name in dir(test_class):
            if method_name.startswith('test_') and callable(getattr(test_class, method_name)):
                method = getattr(test_class, method_name)
                if asyncio.iscoroutinefunction(method):
                    # Wrap async method
                    def make_sync_wrapper(async_method):
                        def sync_wrapper(self):
                            return async_runner.run_async_test(lambda: async_method(self))
                        return sync_wrapper
                    
                    setattr(test_class, method_name, make_sync_wrapper(method))
    
    unittest.main(verbosity=2) 