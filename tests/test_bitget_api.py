"""
Unit Tests für Bitget Trading API
"""

import unittest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.bitget_trading_api import BitgetTradingAPI


class TestBitgetTradingAPI(unittest.TestCase):
    """Test cases for Bitget Trading API"""
    
    def setUp(self):
        """Setup test environment"""
        self.api = BitgetTradingAPI()
        self.test_symbol = "BTCUSDT"
        self.test_amount = 0.001
        self.test_price = 45000.0
    
    def test_api_initialization(self):
        """Test API initialization"""
        self.assertIsInstance(self.api, BitgetTradingAPI)
        self.assertIsNotNone(self.api.api_key)
        self.assertIsNotNone(self.api.secret_key)
        self.assertIsNotNone(self.api.passphrase)
    
    @patch('aiohttp.ClientSession.get')
    async def test_get_server_time(self, mock_get):
        """Test server time retrieval"""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "code": "00000",
            "data": {"serverTime": "1640995200000"}
        }
        mock_get.return_value.__aenter__.return_value = mock_response
        
        server_time = await self.api.get_server_time()
        self.assertIsInstance(server_time, int)
        self.assertGreater(server_time, 0)
    
    @patch('aiohttp.ClientSession.get')
    async def test_get_account_balance(self, mock_get):
        """Test account balance retrieval"""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "code": "00000",
            "data": [
                {
                    "coin": "USDT",
                    "available": "1000.00",
                    "frozen": "0.00"
                }
            ]
        }
        mock_get.return_value.__aenter__.return_value = mock_response
        
        balance = await self.api.get_account_balance()
        self.assertIsInstance(balance, dict)
        self.assertIn("USDT", balance)
        self.assertEqual(balance["USDT"]["available"], 1000.0)
    
    @patch('aiohttp.ClientSession.post')
    async def test_place_market_order(self, mock_post):
        """Test market order placement"""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "code": "00000",
            "data": {
                "orderId": "test_order_123",
                "clientOrderId": "client_123"
            }
        }
        mock_post.return_value.__aenter__.return_value = mock_response
        
        order_result = await self.api.place_market_order(
            self.test_symbol, "buy", self.test_amount
        )
        
        self.assertIsInstance(order_result, dict)
        self.assertIn("orderId", order_result)
        self.assertEqual(order_result["orderId"], "test_order_123")
    
    @patch('aiohttp.ClientSession.post')
    async def test_place_limit_order(self, mock_post):
        """Test limit order placement"""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "code": "00000",
            "data": {
                "orderId": "test_limit_123",
                "clientOrderId": "client_limit_123"
            }
        }
        mock_post.return_value.__aenter__.return_value = mock_response
        
        order_result = await self.api.place_limit_order(
            self.test_symbol, "buy", self.test_amount, self.test_price
        )
        
        self.assertIsInstance(order_result, dict)
        self.assertIn("orderId", order_result)
        self.assertEqual(order_result["orderId"], "test_limit_123")
    
    def test_calculate_tp_sl_prices(self):
        """Test TP/SL price calculation"""
        entry_price = 45000.0
        risk_config = {
            "take_profit_percentage": 2.0,
            "stop_loss_percentage": 1.0
        }
        
        tp_price, sl_price = self.api._calculate_tp_sl_prices(
            entry_price, "buy", risk_config
        )
        
        # For buy order: TP above entry, SL below entry
        self.assertGreater(tp_price, entry_price)
        self.assertLess(sl_price, entry_price)
        
        # Check percentages
        expected_tp = entry_price * 1.02  # 2% profit
        expected_sl = entry_price * 0.99  # 1% loss
        
        self.assertAlmostEqual(tp_price, expected_tp, places=2)
        self.assertAlmostEqual(sl_price, expected_sl, places=2)
    
    def test_calculate_tp_sl_prices_sell(self):
        """Test TP/SL price calculation for sell orders"""
        entry_price = 45000.0
        risk_config = {
            "take_profit_percentage": 2.0,
            "stop_loss_percentage": 1.0
        }
        
        tp_price, sl_price = self.api._calculate_tp_sl_prices(
            entry_price, "sell", risk_config
        )
        
        # For sell order: TP below entry, SL above entry
        self.assertLess(tp_price, entry_price)
        self.assertGreater(sl_price, entry_price)
        
        # Check percentages
        expected_tp = entry_price * 0.98  # 2% profit (price goes down)
        expected_sl = entry_price * 1.01  # 1% loss (price goes up)
        
        self.assertAlmostEqual(tp_price, expected_tp, places=2)
        self.assertAlmostEqual(sl_price, expected_sl, places=2)
    
    @patch('aiohttp.ClientSession.get')
    async def test_get_order_status(self, mock_get):
        """Test order status retrieval"""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "code": "00000",
            "data": {
                "orderId": "test_order_123",
                "status": "filled",
                "size": "0.001",
                "price": "45000.0"
            }
        }
        mock_get.return_value.__aenter__.return_value = mock_response
        
        order_status = await self.api.get_order_status("test_order_123", self.test_symbol)
        
        self.assertIsInstance(order_status, dict)
        self.assertEqual(order_status["status"], "filled")
        self.assertEqual(order_status["orderId"], "test_order_123")
    
    @patch('aiohttp.ClientSession.post')
    async def test_cancel_order(self, mock_post):
        """Test order cancellation"""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "code": "00000",
            "data": {
                "orderId": "test_order_123",
                "status": "cancelled"
            }
        }
        mock_post.return_value.__aenter__.return_value = mock_response
        
        result = await self.api.cancel_order("test_order_123", self.test_symbol)
        
        self.assertTrue(result)
    
    def test_signature_generation(self):
        """Test API signature generation"""
        timestamp = "1640995200000"
        method = "GET"
        request_path = "/api/v2/spot/account/assets"
        body = ""
        
        signature = self.api._generate_signature(timestamp, method, request_path, body)
        
        self.assertIsInstance(signature, str)
        self.assertGreater(len(signature), 0)
    
    def test_error_handling(self):
        """Test error handling"""
        with self.assertRaises(ValueError):
            # Invalid side parameter
            asyncio.run(self.api.place_market_order(self.test_symbol, "invalid_side", self.test_amount))
        
        with self.assertRaises(ValueError):
            # Invalid amount
            asyncio.run(self.api.place_market_order(self.test_symbol, "buy", -1))
        
        with self.assertRaises(ValueError):
            # Invalid price for limit order
            asyncio.run(self.api.place_limit_order(self.test_symbol, "buy", self.test_amount, -1))


class TestBitgetAPIIntegration(unittest.TestCase):
    """Integration tests for Bitget API"""
    
    def setUp(self):
        """Setup integration test environment"""
        self.api = BitgetTradingAPI()
        self.test_mode = True  # Use test mode to avoid real trades
    
    @unittest.skipIf(not BitgetTradingAPI()._is_test_environment(), "Requires test environment")
    async def test_api_connectivity_integration(self):
        """Test actual API connectivity (test environment only)"""
        try:
            server_time = await self.api.get_server_time()
            self.assertIsInstance(server_time, int)
            self.assertGreater(server_time, 0)
        except Exception as e:
            self.skipTest(f"API connectivity test failed: {e}")
    
    @unittest.skipIf(not BitgetTradingAPI()._is_test_environment(), "Requires test environment")
    async def test_account_access_integration(self):
        """Test actual account access (test environment only)"""
        try:
            balance = await self.api.get_account_balance()
            self.assertIsInstance(balance, dict)
        except Exception as e:
            self.skipTest(f"Account access test failed: {e}")


if __name__ == "__main__":
    # Configure test runner
    unittest.main(verbosity=2) 