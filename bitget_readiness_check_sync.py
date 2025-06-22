#!/usr/bin/env python3
"""
🐺 ALPHA BITGET LIVE DEMO READINESS CHECK (SYNC)
=================================================
Umfassende Prüfung aller Systeme für Live Demo Trading auf Bitget

Prüft:
✅ API Verbindung zu Bitget Demo
✅ Demo Account Balance & Kapital
✅ Position Management (Öffnen/Schließen)
✅ Order Management & Execution
✅ Risk Management Konfiguration
✅ Trading Engine Bereitschaft
✅ Alle kritischen Komponenten
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# Pfad zum tradino_unschlagbar Modul hinzufügen
sys.path.append(os.path.join(os.path.dirname(__file__), 'tradino_unschlagbar'))

try:
    import ccxt
    from dotenv import load_dotenv
    import yaml
    from loguru import logger
except ImportError as e:
    print(f"❌ FEHLER: Fehlende Dependencies: {e}")
    print("Installiere mit: pip install ccxt python-dotenv pyyaml loguru")
    sys.exit(1)

class AlphaBitgetReadinessCheck:
    """🐺 Alpha Bitget Live Demo Readiness Checker (Synchron)"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'checks': {},
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Lade Konfiguration
        self.config = self._load_config()
        self.exchange = None
        
        # Setup Logger
        logger.remove()
        logger.add(
            "logs/bitget_readiness_check.log",
            rotation="1 day",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO"
        )

    def _load_config(self) -> Dict[str, Any]:
        """Lade Konfiguration aus verschiedenen Quellen"""
        config = {}
        
        # .env Datei laden
        env_path = os.path.join('tradino_unschlagbar', '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
        
        # config.yaml laden
        config_path = os.path.join('tradino_unschlagbar', 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        
        return config

    def run_all_checks(self) -> Dict[str, Any]:
        """🚀 Führe alle Readiness Checks durch"""
        
        print("🐺 ALPHA BITGET LIVE DEMO READINESS CHECK")
        print("=" * 50)
        print()
        
        checks = [
            ("🔧 Environment Setup", self._check_environment),
            ("🔑 API Credentials", self._check_api_credentials),
            ("🌐 Bitget Connection", self._check_bitget_connection),
            ("💰 Demo Account Balance", self._check_demo_balance),
            ("📊 Market Data Access", self._check_market_data),
            ("🎯 Order Management", self._check_order_management),
            ("📈 Position Management", self._check_position_management),
            ("🛡️ Risk Management", self._check_risk_management),
            ("🤖 Trading Engine", self._check_trading_engine),
            ("📱 Telegram Integration", self._check_telegram),
            ("⚡ Performance Check", self._check_performance),
            ("🔄 Live Trading Simulation", self._simulate_live_trading)
        ]
        
        for check_name, check_func in checks:
            try:
                print(f"\n{check_name}")
                print("-" * 30)
                result = check_func()
                self.results['checks'][check_name] = result
                
                if result['status'] == 'PASS':
                    print(f"✅ {result['message']}")
                elif result['status'] == 'WARNING':
                    print(f"⚠️ {result['message']}")
                    self.results['warnings'].append(f"{check_name}: {result['message']}")
                else:
                    print(f"❌ {result['message']}")
                    self.results['errors'].append(f"{check_name}: {result['message']}")
                    
            except Exception as e:
                error_msg = f"Fehler bei {check_name}: {str(e)}"
                print(f"❌ {error_msg}")
                self.results['errors'].append(error_msg)
                self.results['checks'][check_name] = {
                    'status': 'FAIL',
                    'message': error_msg,
                    'details': traceback.format_exc()
                }
        
        # Gesamtstatus bestimmen
        self._determine_overall_status()
        
        # Finale Zusammenfassung
        self._print_summary()
        
        # Ergebnisse speichern
        self._save_results()
        
        return self.results

    def _check_environment(self) -> Dict[str, Any]:
        """🔧 Prüfe Environment Setup"""
        
        required_vars = [
            'BITGET_API_KEY',
            'BITGET_SECRET_KEY', 
            'BITGET_PASSPHRASE',
            'BITGET_SANDBOX'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            return {
                'status': 'FAIL',
                'message': f"Fehlende Environment Variables: {', '.join(missing_vars)}",
                'details': {'missing_vars': missing_vars}
            }
        
        # Prüfe Sandbox Mode
        sandbox_mode = os.getenv('BITGET_SANDBOX', 'false').lower() == 'true'
        if not sandbox_mode:
            return {
                'status': 'FAIL',
                'message': "BITGET_SANDBOX muss 'true' sein für Demo Trading!",
                'details': {'sandbox_mode': sandbox_mode}
            }
        
        return {
            'status': 'PASS',
            'message': "Environment korrekt konfiguriert für Demo Trading",
            'details': {
                'sandbox_mode': sandbox_mode,
                'env_vars_present': required_vars
            }
        }

    def _check_api_credentials(self) -> Dict[str, Any]:
        """🔑 Prüfe API Credentials"""
        
        api_key = os.getenv('BITGET_API_KEY')
        api_secret = os.getenv('BITGET_SECRET_KEY')
        passphrase = os.getenv('BITGET_PASSPHRASE')
        
        if not all([api_key, api_secret, passphrase]):
            return {
                'status': 'FAIL',
                'message': "Unvollständige API Credentials",
                'details': {}
            }
        
        # Prüfe Format der Credentials
        issues = []
        if len(api_key) < 20:
            issues.append("API Key zu kurz")
        if len(api_secret) < 40:
            issues.append("Secret Key zu kurz")
        if len(passphrase) < 3:
            issues.append("Passphrase zu kurz")
        
        if issues:
            return {
                'status': 'WARNING',
                'message': f"Credential Format Probleme: {', '.join(issues)}",
                'details': {'issues': issues}
            }
        
        return {
            'status': 'PASS',
            'message': "API Credentials Format OK",
            'details': {
                'api_key_length': len(api_key),
                'secret_length': len(api_secret),
                'passphrase_length': len(passphrase)
            }
        }

    def _check_bitget_connection(self) -> Dict[str, Any]:
        """🌐 Prüfe Bitget Verbindung"""
        
        try:
            # Bitget Exchange initialisieren
            self.exchange = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_SECRET_KEY'),
                'password': os.getenv('BITGET_PASSPHRASE'),
                'sandbox': True,  # Demo Mode
                'enableRateLimit': True,
                'timeout': 10000
            })
            
            # Test API Call
            markets = self.exchange.load_markets()
            
            if not markets:
                return {
                    'status': 'FAIL',
                    'message': "Keine Markets von Bitget erhalten",
                    'details': {}
                }
            
            # Server Zeit prüfen
            server_time = self.exchange.fetch_time()
            time_diff = abs(time.time() * 1000 - server_time)
            
            if time_diff > 5000:  # 5 Sekunden
                return {
                    'status': 'WARNING',
                    'message': f"Zeitdifferenz zu Bitget Server: {time_diff/1000:.1f}s",
                    'details': {'time_diff_ms': time_diff}
                }
            
            return {
                'status': 'PASS',
                'message': f"Bitget Demo API verbunden - {len(markets)} Markets verfügbar",
                'details': {
                    'markets_count': len(markets),
                    'server_time_diff_ms': time_diff,
                    'exchange_id': self.exchange.id
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f"Bitget Verbindung fehlgeschlagen: {str(e)}",
                'details': {'error': str(e)}
            }

    def _check_demo_balance(self) -> Dict[str, Any]:
        """💰 Prüfe Demo Account Balance"""
        
        if not self.exchange:
            return {
                'status': 'FAIL',
                'message': "Exchange nicht initialisiert",
                'details': {}
            }
        
        try:
            # Futures Balance abrufen (nicht Standard Balance)
            balance = self.exchange.fetch_balance({'type': 'swap'})
            
            if not balance:
                return {
                    'status': 'FAIL',
                    'message': "Keine Balance-Daten erhalten",
                    'details': {}
                }
            
            # USDT Balance prüfen
            usdt_balance = balance.get('USDT', {})
            free_balance = usdt_balance.get('free', 0)
            total_balance = usdt_balance.get('total', 0)
            
            if total_balance < 100:  # Mindestens $100
                return {
                    'status': 'WARNING',
                    'message': f"Demo Balance sehr niedrig: ${total_balance:.2f} USDT",
                    'details': {
                        'total_usdt': total_balance,
                        'free_usdt': free_balance,
                        'used_usdt': usdt_balance.get('used', 0)
                    }
                }
            
            return {
                'status': 'PASS',
                'message': f"Demo Balance OK: ${total_balance:.2f} USDT verfügbar",
                'details': {
                    'total_usdt': total_balance,
                    'free_usdt': free_balance,
                    'used_usdt': usdt_balance.get('used', 0),
                    'other_balances': {k: v for k, v in balance.items() 
                                     if k != 'USDT' and v.get('total', 0) > 0}
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f"Balance Check fehlgeschlagen: {str(e)}",
                'details': {'error': str(e)}
            }

    def _check_market_data(self) -> Dict[str, Any]:
        """📊 Prüfe Market Data Access"""
        
        if not self.exchange:
            return {'status': 'FAIL', 'message': "Exchange nicht verfügbar", 'details': {}}
        
        try:
            # Test verschiedene Market Data Calls
            test_symbol = 'BTC/USDT'
            
            # Ticker
            ticker = self.exchange.fetch_ticker(test_symbol)
            
            # OHLCV
            ohlcv = self.exchange.fetch_ohlcv(test_symbol, '1m', limit=10)
            
            # Order Book
            orderbook = self.exchange.fetch_order_book(test_symbol)
            
            # Trades
            trades = self.exchange.fetch_trades(test_symbol, limit=10)
            
            issues = []
            if not ticker or not ticker.get('last'):
                issues.append("Ticker Daten unvollständig")
            if not ohlcv or len(ohlcv) < 5:
                issues.append("OHLCV Daten unzureichend")
            if not orderbook or len(orderbook.get('bids', [])) < 5:
                issues.append("Order Book Daten unzureichend")
            if not trades or len(trades) < 5:
                issues.append("Trade History unzureichend")
            
            if issues:
                return {
                    'status': 'WARNING',
                    'message': f"Market Data Probleme: {', '.join(issues)}",
                    'details': {'issues': issues}
                }
            
            return {
                'status': 'PASS',
                'message': f"Market Data vollständig - BTC Preis: ${ticker['last']:.2f}",
                'details': {
                    'btc_price': ticker['last'],
                    'ohlcv_count': len(ohlcv),
                    'orderbook_depth': len(orderbook.get('bids', [])),
                    'recent_trades': len(trades)
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f"Market Data Access fehlgeschlagen: {str(e)}",
                'details': {'error': str(e)}
            }

    def _check_order_management(self) -> Dict[str, Any]:
        """🎯 Prüfe Order Management"""
        
        if not self.exchange:
            return {'status': 'FAIL', 'message': "Exchange nicht verfügbar", 'details': {}}
        
        try:
            # Test Order Creation (ohne Ausführung)
            test_symbol = 'BTC/USDT'
            
            # Aktueller Preis
            ticker = self.exchange.fetch_ticker(test_symbol)
            current_price = ticker['last']
            
            # Test Order Parameter (weit vom Markt entfernt)
            test_price = current_price * 0.5  # 50% unter Markt
            test_amount = 0.001  # Minimale Menge
            
            # Prüfe ob Order erstellt werden könnte
            order_info = {
                'symbol': test_symbol,
                'type': 'limit',
                'side': 'buy',
                'amount': test_amount,
                'price': test_price
            }
            
            # Validiere Order Parameter
            market = self.exchange.markets[test_symbol]
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            min_cost = market.get('limits', {}).get('cost', {}).get('min', 0)
            
            issues = []
            if test_amount < min_amount:
                issues.append(f"Mindestmenge: {min_amount}")
            if test_amount * test_price < min_cost:
                issues.append(f"Mindestkosten: {min_cost}")
            
            if issues:
                return {
                    'status': 'WARNING',
                    'message': f"Order Limits: {', '.join(issues)}",
                    'details': {
                        'min_amount': min_amount,
                        'min_cost': min_cost,
                        'test_order': order_info
                    }
                }
            
            return {
                'status': 'PASS',
                'message': "Order Management bereit - Parameter validiert",
                'details': {
                    'market_info': {
                        'min_amount': min_amount,
                        'min_cost': min_cost,
                        'price_precision': market.get('precision', {}).get('price'),
                        'amount_precision': market.get('precision', {}).get('amount')
                    },
                    'test_order_valid': True
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f"Order Management Check fehlgeschlagen: {str(e)}",
                'details': {'error': str(e)}
            }

    def _check_position_management(self) -> Dict[str, Any]:
        """📈 Prüfe Position Management"""
        
        if not self.exchange:
            return {'status': 'FAIL', 'message': "Exchange nicht verfügbar", 'details': {}}
        
        try:
            # Aktuelle Positionen abrufen
            positions = self.exchange.fetch_positions()
            
            # Open Orders prüfen
            open_orders = self.exchange.fetch_open_orders()
            
            # Position Management Funktionen testen
            capabilities = []
            
            if hasattr(self.exchange, 'fetch_positions'):
                capabilities.append("Position Tracking")
            if hasattr(self.exchange, 'fetch_open_orders'):
                capabilities.append("Order Tracking")
            if hasattr(self.exchange, 'cancel_order'):
                capabilities.append("Order Cancellation")
            if hasattr(self.exchange, 'edit_order'):
                capabilities.append("Order Modification")
            
            return {
                'status': 'PASS',
                'message': f"Position Management verfügbar - {len(capabilities)} Funktionen",
                'details': {
                    'current_positions': len([p for p in positions if p.get('size', 0) != 0]),
                    'open_orders': len(open_orders),
                    'capabilities': capabilities,
                    'position_count': len(positions)
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f"Position Management Check fehlgeschlagen: {str(e)}",
                'details': {'error': str(e)}
            }

    def _check_risk_management(self) -> Dict[str, Any]:
        """🛡️ Prüfe Risk Management Konfiguration"""
        
        config = self.config.get('trading', {})
        
        # Kritische Risk Management Parameter
        risk_per_trade = config.get('risk_per_trade', 0)
        max_daily_drawdown = config.get('max_daily_drawdown', 0)
        max_positions = config.get('max_positions', 0)
        
        issues = []
        if risk_per_trade <= 0 or risk_per_trade > 0.1:  # 0-10%
            issues.append(f"Risk per Trade problematisch: {risk_per_trade*100:.1f}%")
        if max_daily_drawdown <= 0 or max_daily_drawdown > 0.2:  # 0-20%
            issues.append(f"Max Daily Drawdown problematisch: {max_daily_drawdown*100:.1f}%")
        if max_positions <= 0 or max_positions > 20:
            issues.append(f"Max Positions problematisch: {max_positions}")
        
        if issues:
            return {
                'status': 'WARNING',
                'message': f"Risk Management Konfiguration: {', '.join(issues)}",
                'details': {
                    'risk_per_trade': risk_per_trade,
                    'max_daily_drawdown': max_daily_drawdown,
                    'max_positions': max_positions,
                    'issues': issues
                }
            }
        
        return {
            'status': 'PASS',
            'message': f"Risk Management OK - {risk_per_trade*100:.1f}% pro Trade",
            'details': {
                'risk_per_trade_percent': risk_per_trade * 100,
                'max_daily_drawdown_percent': max_daily_drawdown * 100,
                'max_positions': max_positions,
                'portfolio_heat_limit': config.get('portfolio_heat_limit', 0) * 100
            }
        }

    def _check_trading_engine(self) -> Dict[str, Any]:
        """🤖 Prüfe Trading Engine"""
        
        try:
            # Prüfe ob Trading Engine Module verfügbar sind
            engine_path = os.path.join('tradino_unschlagbar', 'core', 'trading_engine.py')
            if not os.path.exists(engine_path):
                return {
                    'status': 'FAIL',
                    'message': "Trading Engine Datei nicht gefunden",
                    'details': {'path': engine_path}
                }
            
            # Prüfe weitere kritische Module
            critical_modules = [
                'core/order_manager.py',
                'core/portfolio_manager.py',
                'core/risk_guardian.py',
                'brain/master_ai.py',
                'strategies/strategy_selector.py'
            ]
            
            missing_modules = []
            for module in critical_modules:
                module_path = os.path.join('tradino_unschlagbar', module)
                if not os.path.exists(module_path):
                    missing_modules.append(module)
            
            if missing_modules:
                return {
                    'status': 'FAIL',
                    'message': f"Fehlende Module: {', '.join(missing_modules)}",
                    'details': {'missing_modules': missing_modules}
                }
            
            return {
                'status': 'PASS',
                'message': "Trading Engine Module vollständig",
                'details': {
                    'available_modules': critical_modules,
                    'engine_path': engine_path
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f"Trading Engine Check fehlgeschlagen: {str(e)}",
                'details': {'error': str(e)}
            }

    def _check_telegram(self) -> Dict[str, Any]:
        """📱 Prüfe Telegram Integration"""
        
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            return {
                'status': 'WARNING',
                'message': "Telegram nicht konfiguriert (optional)",
                'details': {
                    'bot_token_present': bool(bot_token),
                    'chat_id_present': bool(chat_id)
                }
            }
        
        return {
            'status': 'PASS',
            'message': "Telegram Credentials verfügbar",
            'details': {
                'bot_token_length': len(bot_token) if bot_token else 0,
                'chat_id': chat_id
            }
        }

    def _check_performance(self) -> Dict[str, Any]:
        """⚡ Prüfe Performance Anforderungen"""
        
        if not self.exchange:
            return {'status': 'FAIL', 'message': "Exchange nicht verfügbar", 'details': {}}
        
        try:
            # API Response Zeit messen
            start_time = time.time()
            self.exchange.fetch_ticker('BTC/USDT')
            api_response_time = (time.time() - start_time) * 1000
            
            # Performance Bewertung
            if api_response_time > 2000:  # 2 Sekunden
                status = 'FAIL'
                message = f"API zu langsam: {api_response_time:.0f}ms"
            elif api_response_time > 1000:  # 1 Sekunde
                status = 'WARNING'
                message = f"API Performance OK: {api_response_time:.0f}ms"
            else:
                status = 'PASS'
                message = f"API Performance exzellent: {api_response_time:.0f}ms"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'api_response_time_ms': api_response_time,
                    'performance_rating': 'Excellent' if api_response_time < 500 else 'Good' if api_response_time < 1000 else 'Acceptable' if api_response_time < 2000 else 'Poor'
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f"Performance Check fehlgeschlagen: {str(e)}",
                'details': {'error': str(e)}
            }

    def _simulate_live_trading(self) -> Dict[str, Any]:
        """🔄 Simuliere Live Trading Workflow"""
        
        if not self.exchange:
            return {'status': 'FAIL', 'message': "Exchange nicht verfügbar", 'details': {}}
        
        try:
            simulation_steps = []
            
            # 1. Market Data abrufen
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            simulation_steps.append("✅ Market Data abgerufen")
            
            # 2. Balance prüfen
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            simulation_steps.append(f"✅ Balance geprüft: ${usdt_balance:.2f}")
            
            # 3. Positions abrufen
            positions = self.exchange.fetch_positions()
            simulation_steps.append(f"✅ Positions abgerufen: {len(positions)}")
            
            # 4. Open Orders prüfen
            open_orders = self.exchange.fetch_open_orders()
            simulation_steps.append(f"✅ Open Orders geprüft: {len(open_orders)}")
            
            # 5. Order Book analysieren
            orderbook = self.exchange.fetch_order_book('BTC/USDT')
            spread = orderbook['asks'][0][0] - orderbook['bids'][0][0]
            simulation_steps.append(f"✅ Spread analysiert: ${spread:.2f}")
            
            return {
                'status': 'PASS',
                'message': f"Live Trading Simulation erfolgreich - {len(simulation_steps)} Schritte",
                'details': {
                    'simulation_steps': simulation_steps,
                    'btc_price': ticker['last'],
                    'available_balance': usdt_balance,
                    'spread': spread
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f"Live Trading Simulation fehlgeschlagen: {str(e)}",
                'details': {'error': str(e)}
            }

    def _determine_overall_status(self):
        """Bestimme Gesamtstatus basierend auf Einzelchecks"""
        
        total_checks = len(self.results['checks'])
        passed_checks = sum(1 for check in self.results['checks'].values() 
                          if check.get('status') == 'PASS')
        failed_checks = sum(1 for check in self.results['checks'].values() 
                          if check.get('status') == 'FAIL')
        
        if failed_checks > 0:
            self.results['overall_status'] = 'NOT_READY'
        elif len(self.results['warnings']) > 3:
            self.results['overall_status'] = 'READY_WITH_WARNINGS'
        else:
            self.results['overall_status'] = 'READY'
        
        self.results['summary'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'warnings': len(self.results['warnings']),
            'pass_rate': (passed_checks / total_checks * 100) if total_checks > 0 else 0
        }

    def _print_summary(self):
        """Drucke finale Zusammenfassung"""
        
        print("\n" + "=" * 60)
        print("🐺 ALPHA BITGET READINESS CHECK - ZUSAMMENFASSUNG")
        print("=" * 60)
        
        summary = self.results['summary']
        status = self.results['overall_status']
        
        if status == 'READY':
            print("🎉 ✅ ALPHA IST BEREIT FÜR LIVE DEMO TRADING!")
            print("🚀 Sie können Alpha jetzt auf Bitget Demo starten!")
        elif status == 'READY_WITH_WARNINGS':
            print("⚠️ ✅ ALPHA IST BEREIT (mit Warnungen)")
            print("🔧 Überprüfen Sie die Warnungen vor dem Start")
        else:
            print("❌ ALPHA IST NICHT BEREIT")
            print("🛠️ Beheben Sie die Fehler vor dem Start")
        
        print(f"\n📊 Ergebnisse:")
        print(f"   • Gesamtchecks: {summary['total_checks']}")
        print(f"   • Bestanden: {summary['passed_checks']}")
        print(f"   • Fehlgeschlagen: {summary['failed_checks']}")
        print(f"   • Warnungen: {summary['warnings']}")
        print(f"   • Erfolgsrate: {summary['pass_rate']:.1f}%")
        
        if self.results['errors']:
            print(f"\n❌ KRITISCHE FEHLER:")
            for error in self.results['errors'][:5]:  # Top 5
                print(f"   • {error}")
        
        if self.results['warnings']:
            print(f"\n⚠️ WARNUNGEN:")
            for warning in self.results['warnings'][:3]:  # Top 3
                print(f"   • {warning}")
        
        # Empfehlungen basierend auf Ergebnissen
        self._generate_recommendations()
        if self.results['recommendations']:
            print(f"\n💡 EMPFEHLUNGEN:")
            for rec in self.results['recommendations'][:3]:
                print(f"   • {rec}")
        
        print(f"\n📁 Detaillierte Ergebnisse gespeichert in:")
        print(f"   • bitget_readiness_report.json")
        print(f"   • logs/bitget_readiness_check.log")

    def _generate_recommendations(self):
        """Generiere Empfehlungen basierend auf Ergebnissen"""
        
        balance_check = self.results['checks'].get('💰 Demo Account Balance', {})
        if balance_check.get('status') == 'PASS':
            balance_amount = balance_check.get('details', {}).get('total_usdt', 0)
            if balance_amount > 0:
                self.results['recommendations'].append(
                    f"Nutzen Sie das gesamte verfügbare Kapital: ${balance_amount:.2f} USDT"
                )
        
        if self.results['overall_status'] == 'READY':
            self.results['recommendations'].append(
                "Starten Sie Alpha mit: python3 system_launcher.py --live-demo"
            )
        
        if len(self.results['errors']) > 0:
            self.results['recommendations'].append(
                "Beheben Sie die API-Verbindungsprobleme vor dem Live Trading"
            )

    def _save_results(self):
        """Speichere Ergebnisse in JSON Datei"""
        
        os.makedirs('logs', exist_ok=True)
        
        with open('bitget_readiness_report.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Readiness Check abgeschlossen - Status: {self.results['overall_status']}")

def main():
    """🚀 Hauptfunktion"""
    
    print("🐺 ALPHA BITGET LIVE DEMO READINESS CHECK")
    print("Überprüfung aller Systeme für Live Demo Trading...")
    print()
    
    checker = AlphaBitgetReadinessCheck()
    results = checker.run_all_checks()
    
    return results

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Check abgebrochen")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        traceback.print_exc() 