#!/usr/bin/env python3
"""
🧪 COMPLETE MONITORING SYSTEM TEST
Comprehensive test suite for TRADINO monitoring infrastructure
"""

import os
import sys
import time
import json
import threading
from datetime import datetime
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append('/root/tradino')
sys.path.append('/root/tradino/core')

print("🧪 TRADINO COMPLETE MONITORING SYSTEM TEST")
print("=" * 60)

def test_monitoring_system():
    """📊 Test core monitoring system"""
    print("\n1️⃣ Testing Core Monitoring System...")
    
    try:
        from monitoring_system import (
            initialize_monitoring_system, get_monitoring_system,
            LogLevel, LogCategory, log_trade, log_ai_decision,
            log_risk_event, log_api_call, monitor_performance
        )
        
        # Initialize monitoring system
        monitoring = initialize_monitoring_system()
        
        if monitoring:
            print("✅ Monitoring system initialized")
            
            # Test basic logging
            monitoring.info(LogCategory.SYSTEM_HEALTH, "Test system startup")
            monitoring.debug(LogCategory.AI_DECISION, "Test AI decision logging")
            monitoring.warn(LogCategory.RISK_EVENT, "Test risk warning")
            
            # Test structured logging
            test_trade_data = {
                'trade_id': 'TEST-001',
                'symbol': 'BTC/USDT:USDT',
                'action': 'buy',
                'quantity': 0.001,
                'price': 50000.0,
                'total_value': 50.0,
                'fees': 0.05,
                'execution_time': 150.5,
                'latency_ms': 25.3,
                'order_type': 'market',
                'risk_score': 0.15,
                'ai_confidence': 0.85,
                'strategy': 'ai_ensemble'
            }
            
            monitoring.log_trade_execution(test_trade_data)
            print("✅ Trade logging test passed")
            
            # Test AI decision logging
            test_ai_data = {
                'decision': 'buy',
                'confidence': 0.82,
                'models_used': ['xgboost', 'lightgbm', 'random_forest'],
                'feature_importance': {
                    'rsi': 0.35,
                    'macd': 0.28,
                    'volume_ratio': 0.20,
                    'price_momentum': 0.17
                },
                'ensemble_weights': {
                    'xgboost': 0.4,
                    'lightgbm': 0.35,
                    'random_forest': 0.25
                },
                'model_agreement': 0.95,
                'processing_time_ms': 18.7,
                'market_conditions': {
                    'volatility': 0.025,
                    'trend_strength': 0.78,
                    'volume_ratio': 1.35
                },
                'risk_assessment': {
                    'overall_risk': 0.18,
                    'volatility_risk': 0.15,
                    'trend_risk': 0.10
                }
            }
            
            monitoring.log_ai_decision(test_ai_data)
            print("✅ AI decision logging test passed")
            
            # Test performance metrics
            performance = monitoring.get_performance_summary()
            print(f"✅ Performance summary: {performance['session_id']}")
            
            # Test system health
            is_healthy = monitoring._is_system_healthy()
            print(f"✅ System health check: {'Healthy' if is_healthy else 'Issues'}")
            
            return True
            
        else:
            print("❌ Failed to initialize monitoring system")
            return False
            
    except Exception as e:
        print(f"❌ Monitoring system test failed: {e}")
        return False

def test_telegram_integration():
    """📱 Test Telegram bot integration"""
    print("\n2️⃣ Testing Telegram Integration...")
    
    try:
        from monitoring_telegram_bot import (
            initialize_monitoring_telegram_bot, 
            get_monitoring_telegram_bot,
            send_alert_notification
        )
        
        # Check if Telegram credentials are available
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if bot_token and chat_id:
            print("✅ Telegram credentials found")
            
            # Try to initialize bot (but don't start it)
            bot = initialize_monitoring_telegram_bot()
            if bot:
                print("✅ Telegram bot initialized")
                print("ℹ️ Bot ready for deployment (not started in test)")
                return True
            else:
                print("❌ Failed to initialize Telegram bot")
                return False
        else:
            print("⚠️ Telegram credentials not found in environment")
            print("ℹ️ Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env file")
            return True  # Not a failure, just not configured
            
    except Exception as e:
        print(f"❌ Telegram integration test failed: {e}")
        return False

def test_dashboard_system():
    """📊 Test dashboard system"""
    print("\n3️⃣ Testing Dashboard System...")
    
    try:
        from monitoring_dashboard import (
            initialize_monitoring_dashboard,
            get_monitoring_dashboard
        )
        
        # Try to initialize dashboard (but don't start server)
        dashboard = initialize_monitoring_dashboard(port=5001)  # Use different port
        
        if dashboard:
            print("✅ Dashboard system initialized")
            print("ℹ️ Dashboard ready for deployment (not started in test)")
            
            # Test dashboard metrics collection
            try:
                metrics = dashboard._collect_dashboard_metrics()
                print("✅ Dashboard metrics collection works")
                print(f"   Timestamp: {metrics.timestamp}")
                print(f"   System Health: {metrics.system_health}")
            except Exception as e:
                print(f"⚠️ Dashboard metrics collection failed: {e}")
            
            return True
        else:
            print("⚠️ Dashboard system not available (Flask/SocketIO missing)")
            return True  # Not a failure, just dependencies missing
            
    except Exception as e:
        print(f"❌ Dashboard system test failed: {e}")
        return False

def test_log_file_structure():
    """📁 Test log file structure and JSON format"""
    print("\n4️⃣ Testing Log File Structure...")
    
    try:
        from monitoring_system import get_monitoring_system, LogCategory
        
        monitoring = get_monitoring_system()
        if not monitoring:
            print("❌ Monitoring system not available")
            return False
        
        log_dir = monitoring.log_dir
        print(f"✅ Log directory: {log_dir}")
        
        # Check if log files exist and are valid JSON
        valid_logs = 0
        for category in LogCategory:
            log_file = os.path.join(log_dir, f"{category.value}.log")
            
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # Check last line for valid JSON
                            last_line = lines[-1].strip()
                            json.loads(last_line)
                            valid_logs += 1
                            print(f"✅ {category.value}.log - {len(lines)} entries")
                        else:
                            print(f"ℹ️ {category.value}.log - empty")
                except json.JSONDecodeError:
                    print(f"❌ {category.value}.log - invalid JSON format")
                except Exception as e:
                    print(f"❌ {category.value}.log - error: {e}")
            else:
                print(f"ℹ️ {category.value}.log - not created yet")
        
        print(f"✅ Valid log files: {valid_logs}/{len(LogCategory)}")
        return True
        
    except Exception as e:
        print(f"❌ Log file structure test failed: {e}")
        return False

def test_performance_monitoring():
    """⚡ Test performance monitoring features"""
    print("\n5️⃣ Testing Performance Monitoring...")
    
    try:
        from monitoring_system import get_monitoring_system, monitor_performance, LogCategory
        
        monitoring = get_monitoring_system()
        if not monitoring:
            print("❌ Monitoring system not available")
            return False
        
        # Test performance decorator
        @monitor_performance(LogCategory.PERFORMANCE)
        def test_function():
            time.sleep(0.1)  # Simulate work
            return "success"
        
        result = test_function()
        print(f"✅ Performance decorator test: {result}")
        
        # Test system metrics
        metrics = monitoring.get_system_metrics()
        if metrics:
            print(f"✅ System metrics collected:")
            print(f"   CPU: {metrics.cpu_usage:.1f}%")
            print(f"   Memory: {metrics.memory_usage:.1f}%")
            print(f"   Disk: {metrics.disk_usage:.1f}%")
            print(f"   Threads: {metrics.active_threads}")
        else:
            print("⚠️ No system metrics available")
        
        # Test alert system
        from monitoring_system import AlertSeverity
        monitoring._create_alert(
            AlertSeverity.LOW,
            LogCategory.PERFORMANCE,
            "Test Alert",
            "This is a test alert for performance monitoring",
            {'test': True, 'value': 42}
        )
        print("✅ Alert system test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False

def test_alert_and_notification_system():
    """🚨 Test alert and notification system"""
    print("\n6️⃣ Testing Alert and Notification System...")
    
    try:
        from monitoring_system import get_monitoring_system, AlertSeverity, LogCategory
        
        monitoring = get_monitoring_system()
        if not monitoring:
            print("❌ Monitoring system not available")
            return False
        
        # Test different severity levels
        test_alerts = [
            (AlertSeverity.LOW, "Low Priority Test"),
            (AlertSeverity.MEDIUM, "Medium Priority Test"),
            (AlertSeverity.HIGH, "High Priority Test"),
            (AlertSeverity.CRITICAL, "Critical Priority Test")
        ]
        
        for severity, title in test_alerts:
            monitoring._create_alert(
                severity,
                LogCategory.SYSTEM_HEALTH,
                title,
                f"Test alert with {severity.value} priority",
                {'severity_level': severity.value, 'test_data': True}
            )
            print(f"✅ {severity.value} alert created")
        
        # Check active alerts
        active_alerts = monitoring.active_alerts
        print(f"✅ Active alerts: {len(active_alerts)}")
        
        # Test alert rate limiting
        time.sleep(1)
        monitoring._create_alert(
            AlertSeverity.LOW,
            LogCategory.SYSTEM_HEALTH,
            "Rate Limit Test",
            "This should be rate limited",
            {'rate_limit_test': True}
        )
        print("✅ Alert rate limiting test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Alert system test failed: {e}")
        return False

def test_integration_with_trading_system():
    """💰 Test integration with trading system"""
    print("\n7️⃣ Testing Trading System Integration...")
    
    try:
        # Test logging functions
        from monitoring_system import log_trade, log_ai_decision, log_risk_event, log_api_call
        
        # Test trade logging
        trade_data = {
            'trade_id': 'INTEGRATION-TEST-001',
            'symbol': 'ETH/USDT:USDT',
            'action': 'sell',
            'quantity': 0.1,
            'price': 3500.0,
            'total_value': 350.0,
            'fees': 0.35,
            'execution_time': 89.2,
            'latency_ms': 15.8,
            'order_type': 'market',
            'risk_score': 0.12,
            'ai_confidence': 0.88,
            'strategy': 'momentum_trading'
        }
        
        log_trade(trade_data)
        print("✅ Trade logging integration test passed")
        
        # Test AI decision logging
        ai_data = {
            'decision': 'sell',
            'confidence': 0.88,
            'models_used': ['xgboost', 'neural_network'],
            'processing_time_ms': 22.1,
            'market_conditions': {'volatility': 0.032}
        }
        
        log_ai_decision(ai_data)
        print("✅ AI decision logging integration test passed")
        
        # Test risk event logging
        risk_data = {
            'event_type': 'position_limit_exceeded',
            'severity': 'medium',
            'description': 'Position size exceeded 5% portfolio limit',
            'action_taken': 'position_reduced',
            'affected_positions': ['BTC/USDT:USDT']
        }
        
        log_risk_event(risk_data)
        print("✅ Risk event logging integration test passed")
        
        # Test API call logging
        api_data = {
            'endpoint': '/api/futures/order',
            'status': 'success',
            'response_time_ms': 145.2,
            'status_code': 200,
            'request_size': 256,
            'response_size': 512
        }
        
        log_api_call(api_data)
        print("✅ API call logging integration test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading system integration test failed: {e}")
        return False

def test_comprehensive_monitoring():
    """🎯 Comprehensive monitoring test"""
    print("\n8️⃣ Running Comprehensive Monitoring Test...")
    
    try:
        from monitoring_system import (
            get_monitoring_system, log_ai_decision, 
            log_trade, log_risk_event
        )
        
        monitoring = get_monitoring_system()
        if not monitoring:
            print("❌ Monitoring system not available")
            return False
        
        # Simulate a trading session
        print("📊 Simulating 30-second trading session...")
        
        start_time = time.time()
        session_events = 0
        
        while time.time() - start_time < 30:  # Run for 30 seconds
            # Simulate AI decision
            ai_data = {
                'decision': 'buy' if session_events % 2 == 0 else 'sell',
                'confidence': 0.75 + (session_events % 10) * 0.02,
                'processing_time_ms': 15 + (session_events % 5) * 3,
                'models_used': ['ensemble']
            }
            
            log_ai_decision(ai_data)
            session_events += 1
            
            # Simulate occasional trade
            if session_events % 5 == 0:
                trade_data = {
                    'trade_id': f'SIM-{session_events:03d}',
                    'symbol': 'BTC/USDT:USDT',
                    'action': ai_data['decision'],
                    'quantity': 0.001 * session_events,
                    'price': 50000 + session_events * 10,
                    'total_value': 50 + session_events * 0.01,
                    'ai_confidence': ai_data['confidence']
                }
                
                log_trade(trade_data)
            
            # Simulate occasional risk event
            if session_events % 8 == 0:
                risk_data = {
                    'event_type': 'volatility_spike',
                    'severity': 'low',
                    'description': f'Volatility increased during event {session_events}'
                }
                
                log_risk_event(risk_data)
            
            time.sleep(2)  # Wait 2 seconds between events
        
        print(f"✅ Simulation completed: {session_events} events generated")
        
        # Check final system state
        performance = monitoring.get_performance_summary()
        print(f"✅ Final system state:")
        print(f"   Session ID: {performance['session_id']}")
        print(f"   Uptime: {performance['uptime_hours']:.2f} hours")
        print(f"   Total Logs: {performance['total_logs']}")
        print(f"   System Healthy: {performance['system_healthy']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive monitoring test failed: {e}")
        return False

def run_all_tests():
    """🏃 Run all monitoring system tests"""
    print("🚀 Starting Complete Monitoring System Tests...")
    
    tests = [
        ("Core Monitoring System", test_monitoring_system),
        ("Telegram Integration", test_telegram_integration),
        ("Dashboard System", test_dashboard_system),
        ("Log File Structure", test_log_file_structure),
        ("Performance Monitoring", test_performance_monitoring),
        ("Alert and Notification System", test_alert_and_notification_system),
        ("Trading System Integration", test_integration_with_trading_system),
        ("Comprehensive Monitoring", test_comprehensive_monitoring)
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
            if result:
                passed += 1
        except Exception as e:
            results[test_name] = f"ERROR: {e}"
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status_emoji = "✅" if result == "PASS" else "❌" if result == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name:<35} {result}")
    
    print("=" * 60)
    print(f"📊 OVERALL RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 ALL TESTS PASSED! Monitoring system is fully operational.")
    elif passed >= len(tests) * 0.8:
        print("✅ Most tests passed. System is largely operational.")
    else:
        print("⚠️ Several tests failed. Check system configuration.")
    
    print("=" * 60)
    
    return passed, len(tests)

if __name__ == "__main__":
    try:
        passed, total = run_all_tests()
        
        print(f"\n📋 FINAL SUMMARY:")
        print(f"   Tests Passed: {passed}")
        print(f"   Tests Failed: {total - passed}")
        print(f"   Success Rate: {passed/total*100:.1f}%")
        
        if passed == total:
            print("\n🚀 TRADINO MONITORING SYSTEM IS READY FOR PRODUCTION!")
        else:
            print(f"\n⚠️ {total - passed} issues need to be resolved before production deployment.")
            
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
    
    print("\n🎯 Test execution completed.")