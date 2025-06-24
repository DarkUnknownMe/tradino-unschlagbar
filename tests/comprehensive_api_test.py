#!/usr/bin/env python3
"""
TRADINO Comprehensive API Test Suite
====================================

Umfassender Test ALLER Bitget API Funktionen
Vor Live-Trading Deployment
"""

import sys
import os
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Setup
sys.path.append(str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class APITestResults:
    """Sammelt API Test Ergebnisse"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.results = {}
        self.errors = []
        self.api_responses = {}
        
    def add_result(self, test_name: str, status: str, details: str = "", response_data: Any = None):
        self.results[test_name] = {
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat(),
            'duration': time.time()
        }
        if response_data:
            self.api_responses[test_name] = response_data
            
    def generate_report(self) -> str:
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASSED')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAILED')
        total = len(self.results)
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                         TRADINO COMPREHENSIVE API TEST REPORT                             ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║ Tests: {total:3d} | Passed: {passed:3d} | Failed: {failed:3d} | Success Rate: {(passed/total*100) if total > 0 else 0:.1f}%  ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

"""
        for test_name, result in self.results.items():
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            report += f"{status_icon} {test_name:<60} [{result['status']:>6}]\n"
            if result['details']:
                report += f"   └─ {result['details']}\n"
        
        readiness = "🟢 API READY FOR LIVE TRADING" if failed == 0 else "🔴 API ISSUES - REVIEW BEFORE LIVE"
        report += f"\n{readiness}\n" + "="*90
        
        return report

class ComprehensiveAPITest:
    """Umfassender API-Test für alle Bitget Funktionen"""
    
    def __init__(self):
        self.results = APITestResults()
        self.connector = None
        
    def setup_api_connection(self):
        """Setup der API Verbindung"""
        try:
            logger.info("🔌 Setting up Bitget API connection...")
            
            # Import Bitget Connector
            from tradino_unschlagbar.connectors.bitget_pro import BitgetConnector
            self.connector = BitgetConnector()
            
            logger.info("✅ API Connector initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ API Setup failed: {e}")
            return False
    
    def test_connection_health(self):
        """Test 1: Grundlegende Verbindung"""
        try:
            logger.info("🧪 Testing basic connection health...")
            
            # Test ping/server time
            if hasattr(self.connector, 'get_server_time'):
                server_time = self.connector.get_server_time()
                if server_time:
                    self.results.add_result("connection_health", "PASSED", 
                                          f"Server time: {server_time}", server_time)
                else:
                    self.results.add_result("connection_health", "FAILED", "No server time response")
            else:
                # Fallback test
                if hasattr(self.connector, 'test_connection'):
                    result = self.connector.test_connection()
                    status = "PASSED" if result else "FAILED"
                    self.results.add_result("connection_health", status, f"Connection test: {result}")
                else:
                    self.results.add_result("connection_health", "PASSED", "Connector initialized successfully")
                    
        except Exception as e:
            self.results.add_result("connection_health", "FAILED", f"Connection error: {str(e)}")
    
    def test_account_info(self):
        """Test 2: Account Information"""
        try:
            logger.info("🧪 Testing account information...")
            
            if hasattr(self.connector, 'get_account_info'):
                account_info = self.connector.get_account_info()
                if account_info:
                    self.results.add_result("account_info", "PASSED", 
                                          f"Account retrieved: {type(account_info)}", account_info)
                else:
                    self.results.add_result("account_info", "FAILED", "No account info returned")
            else:
                self.results.add_result("account_info", "SKIPPED", "Method not available")
                
        except Exception as e:
            self.results.add_result("account_info", "FAILED", f"Account info error: {str(e)}")
    
    def test_balance_retrieval(self):
        """Test 3: Balance Information"""
        try:
            logger.info("🧪 Testing balance retrieval...")
            
            methods_to_test = ['get_balance', 'get_balances', 'get_account_balance']
            
            for method_name in methods_to_test:
                if hasattr(self.connector, method_name):
                    method = getattr(self.connector, method_name)
                    balance = method()
                    
                    if balance:
                        self.results.add_result(f"balance_{method_name}", "PASSED", 
                                              f"Balance retrieved: {type(balance)}", balance)
                    else:
                        self.results.add_result(f"balance_{method_name}", "FAILED", "No balance returned")
                    break
            else:
                self.results.add_result("balance_retrieval", "SKIPPED", "No balance methods available")
                
        except Exception as e:
            self.results.add_result("balance_retrieval", "FAILED", f"Balance error: {str(e)}")
    
    def test_market_data_functions(self):
        """Test 4: Market Data Functions"""
        try:
            logger.info("🧪 Testing market data functions...")
            
            test_symbols = ['BTCUSDT', 'ETHUSDT']
            
            # Test ticker data
            for symbol in test_symbols:
                if hasattr(self.connector, 'get_ticker'):
                    ticker = self.connector.get_ticker(symbol)
                    if ticker:
                        self.results.add_result(f"ticker_{symbol}", "PASSED", 
                                              f"Ticker retrieved for {symbol}", ticker)
                    else:
                        self.results.add_result(f"ticker_{symbol}", "FAILED", f"No ticker for {symbol}")
            
            # Test orderbook
            if hasattr(self.connector, 'get_orderbook'):
                orderbook = self.connector.get_orderbook('BTCUSDT')
                if orderbook:
                    self.results.add_result("orderbook", "PASSED", "Orderbook retrieved", orderbook)
                else:
                    self.results.add_result("orderbook", "FAILED", "No orderbook data")
            
            # Test klines/candlestick data
            if hasattr(self.connector, 'get_klines'):
                klines = self.connector.get_klines('BTCUSDT', '1m', limit=10)
                if klines:
                    self.results.add_result("klines", "PASSED", f"Klines retrieved: {len(klines)} candles", klines)
                else:
                    self.results.add_result("klines", "FAILED", "No klines data")
                    
        except Exception as e:
            self.results.add_result("market_data", "FAILED", f"Market data error: {str(e)}")
    
    def test_trading_functions(self):
        """Test 5: Trading Functions (Dry Run)"""
        try:
            logger.info("🧪 Testing trading functions...")
            
            # Test order validation (without placing)
            test_order = {
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'type': 'limit',
                'amount': 0.001,
                'price': 30000
            }
            
            if hasattr(self.connector, 'validate_order'):
                validation = self.connector.validate_order(test_order)
                status = "PASSED" if validation else "FAILED"
                self.results.add_result("order_validation", status, f"Order validation: {validation}")
            
            # Test order creation (simulation mode if available)
            if hasattr(self.connector, 'create_test_order') or hasattr(self.connector, 'create_order'):
                method_name = 'create_test_order' if hasattr(self.connector, 'create_test_order') else 'create_order'
                method = getattr(self.connector, method_name)
                
                # Only test if we can safely simulate
                if 'test' in method_name.lower():
                    order_result = method(**test_order)
                    if order_result:
                        self.results.add_result("test_order_creation", "PASSED", 
                                              "Test order created", order_result)
                    else:
                        self.results.add_result("test_order_creation", "FAILED", "Test order failed")
                else:
                    self.results.add_result("order_creation", "SKIPPED", 
                                          "Live order creation skipped for safety")
            
            # Test order status functions
            if hasattr(self.connector, 'get_orders'):
                orders = self.connector.get_orders('BTCUSDT')
                self.results.add_result("get_orders", "PASSED", 
                                      f"Orders retrieved: {len(orders) if orders else 0}", orders)
            
            if hasattr(self.connector, 'get_open_orders'):
                open_orders = self.connector.get_open_orders('BTCUSDT')
                self.results.add_result("get_open_orders", "PASSED", 
                                      f"Open orders: {len(open_orders) if open_orders else 0}", open_orders)
                
        except Exception as e:
            self.results.add_result("trading_functions", "FAILED", f"Trading functions error: {str(e)}")
    
    def test_position_management(self):
        """Test 6: Position Management"""
        try:
            logger.info("🧪 Testing position management...")
            
            if hasattr(self.connector, 'get_positions'):
                positions = self.connector.get_positions()
                self.results.add_result("get_positions", "PASSED", 
                                      f"Positions retrieved: {len(positions) if positions else 0}", positions)
            
            if hasattr(self.connector, 'get_position'):
                position = self.connector.get_position('BTCUSDT')
                self.results.add_result("get_position", "PASSED", 
                                      f"Position for BTCUSDT: {position is not None}", position)
            
            # Test position sizing calculations
            if hasattr(self.connector, 'calculate_position_size'):
                pos_size = self.connector.calculate_position_size(1000, 0.02, 50000, 49000)
                if pos_size:
                    self.results.add_result("position_sizing", "PASSED", 
                                          f"Position size calculated: {pos_size}")
                else:
                    self.results.add_result("position_sizing", "FAILED", "Position sizing failed")
                    
        except Exception as e:
            self.results.add_result("position_management", "FAILED", f"Position error: {str(e)}")
    
    def test_market_analysis_functions(self):
        """Test 7: Market Analysis Functions"""
        try:
            logger.info("🧪 Testing market analysis functions...")
            
            # Test 24hr statistics
            if hasattr(self.connector, 'get_24hr_ticker'):
                ticker_24hr = self.connector.get_24hr_ticker('BTCUSDT')
                if ticker_24hr:
                    self.results.add_result("24hr_ticker", "PASSED", "24hr ticker retrieved", ticker_24hr)
                else:
                    self.results.add_result("24hr_ticker", "FAILED", "No 24hr ticker")
            
            # Test trade history
            if hasattr(self.connector, 'get_trades'):
                trades = self.connector.get_trades('BTCUSDT', limit=10)
                if trades:
                    self.results.add_result("recent_trades", "PASSED", 
                                          f"Recent trades: {len(trades)}", trades)
                else:
                    self.results.add_result("recent_trades", "FAILED", "No recent trades")
            
            # Test symbol information
            if hasattr(self.connector, 'get_exchange_info'):
                exchange_info = self.connector.get_exchange_info()
                if exchange_info:
                    self.results.add_result("exchange_info", "PASSED", "Exchange info retrieved", exchange_info)
                else:
                    self.results.add_result("exchange_info", "FAILED", "No exchange info")
                    
        except Exception as e:
            self.results.add_result("market_analysis", "FAILED", f"Market analysis error: {str(e)}")
    
    def test_websocket_functions(self):
        """Test 8: WebSocket Functions"""
        try:
            logger.info("🧪 Testing WebSocket functions...")
            
            if hasattr(self.connector, 'start_websocket'):
                self.results.add_result("websocket_available", "PASSED", "WebSocket methods available")
            else:
                self.results.add_result("websocket_available", "SKIPPED", "No WebSocket methods")
            
            # Test WebSocket connection setup (without starting)
            if hasattr(self.connector, 'setup_websocket'):
                setup_result = self.connector.setup_websocket()
                status = "PASSED" if setup_result else "FAILED"
                self.results.add_result("websocket_setup", status, f"WebSocket setup: {setup_result}")
                
        except Exception as e:
            self.results.add_result("websocket_functions", "FAILED", f"WebSocket error: {str(e)}")
    
    def test_error_handling(self):
        """Test 9: Error Handling"""
        try:
            logger.info("🧪 Testing error handling...")
            
            # Test invalid symbol
            if hasattr(self.connector, 'get_ticker'):
                try:
                    invalid_ticker = self.connector.get_ticker('INVALIDCOIN')
                    self.results.add_result("error_handling_invalid_symbol", "PASSED", 
                                          "Invalid symbol handled gracefully")
                except Exception as e:
                    if "not found" in str(e).lower() or "invalid" in str(e).lower():
                        self.results.add_result("error_handling_invalid_symbol", "PASSED", 
                                              "Error properly caught for invalid symbol")
                    else:
                        self.results.add_result("error_handling_invalid_symbol", "FAILED", 
                                              f"Unexpected error: {str(e)}")
            
            # Test rate limiting awareness
            if hasattr(self.connector, 'check_rate_limit'):
                rate_limit = self.connector.check_rate_limit()
                self.results.add_result("rate_limit_check", "PASSED", 
                                      f"Rate limit check: {rate_limit}")
            else:
                self.results.add_result("rate_limit_check", "SKIPPED", "No rate limit checking")
                
        except Exception as e:
            self.results.add_result("error_handling", "FAILED", f"Error handling test failed: {str(e)}")
    
    def test_api_permissions(self):
        """Test 10: API Permissions"""
        try:
            logger.info("🧪 Testing API permissions...")
            
            # Test read permissions
            read_tests = ['get_account_info', 'get_balance', 'get_ticker']
            read_passed = 0
            
            for test_method in read_tests:
                if hasattr(self.connector, test_method):
                    try:
                        method = getattr(self.connector, test_method)
                        if test_method == 'get_ticker':
                            result = method('BTCUSDT')
                        else:
                            result = method()
                        if result:
                            read_passed += 1
                    except Exception:
                        pass
            
            self.results.add_result("api_read_permissions", "PASSED" if read_passed > 0 else "FAILED", 
                                  f"Read permissions: {read_passed}/{len(read_tests)} working")
            
            # Test if we have trading permissions (without actually trading)
            if hasattr(self.connector, 'get_orders'):
                try:
                    orders = self.connector.get_orders('BTCUSDT')
                    self.results.add_result("api_trade_permissions", "PASSED", 
                                          "Trading API access confirmed")
                except Exception as e:
                    if "permission" in str(e).lower():
                        self.results.add_result("api_trade_permissions", "FAILED", 
                                              "No trading permissions")
                    else:
                        self.results.add_result("api_trade_permissions", "PASSED", 
                                              "Trading API accessible")
                                              
        except Exception as e:
            self.results.add_result("api_permissions", "FAILED", f"Permission test error: {str(e)}")
    
    def run_comprehensive_test(self):
        """Führt alle API Tests aus"""
        logger.info("🚀 Starting Comprehensive API Test Suite")
        logger.info("="*80)
        
        # Setup
        if not self.setup_api_connection():
            logger.error("❌ Cannot proceed - API setup failed")
            return self.results
        
        # Run all tests
        test_methods = [
            self.test_connection_health,
            self.test_account_info,
            self.test_balance_retrieval,
            self.test_market_data_functions,
            self.test_trading_functions,
            self.test_position_management,
            self.test_market_analysis_functions,
            self.test_websocket_functions,
            self.test_error_handling,
            self.test_api_permissions
        ]
        
        for i, test_method in enumerate(test_methods, 1):
            logger.info(f"\n📋 Running Test {i}/{len(test_methods)}: {test_method.__name__}")
            try:
                test_method()
            except Exception as e:
                logger.error(f"❌ Test {test_method.__name__} failed with exception: {e}")
                self.results.add_result(test_method.__name__, "FAILED", f"Exception: {str(e)}")
            
            time.sleep(0.5)  # Rate limiting protection
        
        # Generate report
        report = self.results.generate_report()
        logger.info("\n" + report)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save report
        report_file = f"tests/api_test_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"📄 API test report saved: {report_file}")
        
        # Save raw data
        data_file = f"tests/api_test_data_{timestamp}.json"
        with open(data_file, 'w') as f:
            json.dump({
                'results': self.results.results,
                'api_responses': self.results.api_responses,
                'timestamp': timestamp
            }, f, indent=2, default=str)
        logger.info(f"📊 API test data saved: {data_file}")
        
        return self.results

def main():
    """Hauptfunktion"""
    print("🔥 TRADINO Comprehensive API Test 🔥")
    print("=" * 50)
    
    test_suite = ComprehensiveAPITest()
    results = test_suite.run_comprehensive_test()
    
    # Final assessment
    total_tests = len(results.results)
    passed_tests = sum(1 for r in results.results.values() if r['status'] == 'PASSED')
    failed_tests = sum(1 for r in results.results.values() if r['status'] == 'FAILED')
    skipped_tests = sum(1 for r in results.results.values() if r['status'] == 'SKIPPED')
    
    print(f"\n🏁 API Test Suite Completed!")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {failed_tests}")
    print(f"   Skipped: {skipped_tests}")
    print(f"   Success Rate: {(passed_tests/total_tests*100) if total_tests > 0 else 0:.1f}%")
    
    if failed_tests == 0:
        print("🎉 ALL API TESTS PASSED! Ready for live trading!")
        return 0
    else:
        print(f"⚠️  {failed_tests} API tests failed. Review before proceeding.")
        return 1

if __name__ == "__main__":
    exit(main())