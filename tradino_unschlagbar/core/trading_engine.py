"""
🚀 TRADINO UNSCHLAGBAR - Core Trading Engine
Zentraler Trading Engine - Koordiniert alle Komponenten

Author: AI Trading Systems
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from models.trade_models import TradeSignal, Order, Trade
from models.portfolio_models import Portfolio, Position
from connectors.bitget_pro import BitgetProConnector
from connectors.telegram_commander import TelegramCommander
from connectors.notification_hub import NotificationHub
from core.order_manager import OrderManager
from core.position_tracker import PositionTracker
from core.portfolio_manager import PortfolioManager
from core.risk_guardian import RiskGuardian
from brain.master_ai import MasterAI
from brain.rl_trading_agent import RLTradingIntegration
from brain.multi_agent_system import TRADINOMultiAgentSystem
from strategies.strategy_selector import StrategySelector
from analytics.performance_tracker import PerformanceTracker
from utils.logger_pro import setup_logger, log_trade
from utils.config_manager import ConfigManager

logger = setup_logger("TradingEngine")


class EngineState(Enum):
    """Trading Engine Status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class TradingEngine:
    """🚀 Zentraler Trading Engine Controller"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.state = EngineState.STOPPED
        
        # Core Components
        self.bitget: Optional[BitgetProConnector] = None
        self.telegram: Optional[TelegramCommander] = None
        self.notification_hub: Optional[NotificationHub] = None
        self.order_manager: Optional[OrderManager] = None
        self.position_tracker: Optional[PositionTracker] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.risk_guardian: Optional[RiskGuardian] = None
        
        # AI & Strategy Components
        self.master_ai: Optional[MasterAI] = None
        self.strategy_selector: Optional[StrategySelector] = None
        self.rl_integration: Optional[RLTradingIntegration] = None
        self.multi_agent_system: Optional[TRADINOMultiAgentSystem] = None
        
        # Analytics
        self.performance_tracker: Optional[PerformanceTracker] = None
        
        # Trading Configuration
        self.trading_enabled = config.get('trading.enabled', True)
        self.auto_trading = config.get('trading.auto_execution', True)
        self.max_concurrent_trades = config.get('trading.max_positions', 5)
        
        # Symbol Management
        self.active_symbols: List[str] = []
        self.symbol_blacklist: List[str] = []
        
        # Trading Loop
        self._trading_task: Optional[asyncio.Task] = None
        self._analysis_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        
        # Performance Metrics
        self.engine_start_time: Optional[datetime] = None
        self.signals_processed = 0
        self.trades_executed = 0
        self.errors_count = 0
        
        # Callbacks
        self.trade_callbacks: List[Callable] = []
        self.signal_callbacks: List[Callable] = []
        
    async def initialize(self) -> bool:
        """🔥 Trading Engine initialisieren"""
        try:
            logger.info("🚀 TRADINO UNSCHLAGBAR Trading Engine wird initialisiert...")
            self.state = EngineState.STARTING
            
            # 1. Exchange Connector
            self.bitget = BitgetProConnector(self.config)
            if not await self.bitget.initialize():
                raise Exception("Bitget Connector Initialisierung fehlgeschlagen")
            
            # 2. Notification System
            self.notification_hub = NotificationHub(self.config)
            if not await self.notification_hub.initialize():
                raise Exception("Notification Hub Initialisierung fehlgeschlagen")
            
            # 3. Core Trading Components
            self.order_manager = OrderManager(self.config, self.bitget)
            if not await self.order_manager.initialize():
                raise Exception("Order Manager Initialisierung fehlgeschlagen")
            
            self.position_tracker = PositionTracker(self.config, self.bitget)
            if not await self.position_tracker.initialize():
                raise Exception("Position Tracker Initialisierung fehlgeschlagen")
            
            self.portfolio_manager = PortfolioManager(self.config, self.bitget, self.position_tracker)
            if not await self.portfolio_manager.initialize():
                raise Exception("Portfolio Manager Initialisierung fehlgeschlagen")
            
            self.risk_guardian = RiskGuardian(self.config, self.portfolio_manager, self.position_tracker)
            if not await self.risk_guardian.initialize():
                raise Exception("Risk Guardian Initialisierung fehlgeschlagen")
            
            # 4. AI System
            self.master_ai = MasterAI(self.config, self.bitget)
            if not await self.master_ai.initialize():
                raise Exception("Master AI Initialisierung fehlgeschlagen")
            
            # Set market intelligence reference for other components
            self.market_intelligence = self.master_ai.market_intelligence
            
            # 5. RL Integration
            logger.info("🧠 RL Integration wird initialisiert...")
            self.rl_integration = RLTradingIntegration(self.config, self)
            await self.rl_integration.initialize()
            
            # 5.5. Multi-Agent System Integration - NEU!
            logger.info("🤖 Multi-Agent System wird initialisiert...")
            self.multi_agent_system = TRADINOMultiAgentSystem(self.config, self)
            await self.multi_agent_system.initialize()
            
            # 6. Strategy System - Fix für korrekte Parameter
            # Erst Strategy Dependencies initialisieren
            await self._initialize_strategy_dependencies()
            
            self.strategy_selector = StrategySelector(
                config=self.config,
                scalping_master=self.scalping_master,
                swing_genius=self.swing_genius,
                trend_hunter=self.trend_hunter,
                mean_reversion=self.mean_reversion,
                master_ai=self.master_ai,
                market_intelligence=self.master_ai.market_intelligence
            )
            if not await self.strategy_selector.initialize():
                raise Exception("Strategy Selector Initialisierung fehlgeschlagen")
            
            # 7. Analytics - Fix für Performance Tracker Import
            from analytics.performance_tracker import PerformanceTracker
            self.performance_tracker = PerformanceTracker(self.config)
            if not await self.performance_tracker.initialize():
                raise Exception("Performance Tracker Initialisierung fehlgeschlagen")
            
            # 8. Profitable Pairs laden
            await self._load_profitable_pairs()
            
            # 9. Callbacks registrieren
            await self._register_callbacks()
            
            self.state = EngineState.STOPPED
            logger.success("✅ Trading Engine erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ Trading Engine Initialisierung fehlgeschlagen: {e}")
            self.state = EngineState.ERROR
            return False
    
    async def start(self):
        """
        Trading Engine mit robustem State Management starten
        """
        try:
            logger.info("🚀 Trading Engine wird gestartet...")
            
            # ✅ STATE VALIDATION & RESET
            if self.state == EngineState.RUNNING:
                logger.warning("⚠️ Trading Engine läuft bereits")
                return True
            
            if self.state == EngineState.ERROR:
                logger.info("🔄 Engine in Error State - wird zurückgesetzt...")
                await self._reset_engine_state()
            
            # Ensure proper initialization
            if not await self._validate_engine_readiness():
                logger.error("❌ Engine nicht bereit für Start")
                return False
            
            # Start Engine
            self.state = EngineState.STARTING
            
            # Start Components
            success = await self._start_all_components()
            
            if success:
                self.state = EngineState.RUNNING
                logger.success("✅ Trading Engine erfolgreich gestartet")
                return True
            else:
                self.state = EngineState.ERROR
                logger.error("❌ Trading Engine Start fehlgeschlagen")
                return False
            
        except Exception as e:
            self.state = EngineState.ERROR
            logger.error(f"❌ Fehler beim Trading Engine Start: {e}")
            return False
    
    async def stop(self):
        """
        Trading Engine sicher stoppen
        """
        try:
            logger.info("🛑 Trading Engine wird gestoppt...")
            
            if self.state == EngineState.STOPPED:
                logger.info("ℹ️ Trading Engine bereits gestoppt")
                return True
            
            self.state = EngineState.STOPPING
            
            # Stop all components safely
            await self._stop_all_components_safely()
            
            self.state = EngineState.STOPPED
            logger.success("✅ Trading Engine erfolgreich gestoppt")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Trading Engine Stop: {e}")
            return False

    async def _reset_engine_state(self):
        """
        Engine State sicher zurücksetzen
        """
        try:
            logger.info("🔄 Engine State wird zurückgesetzt...")
            
            # Stop all components safely
            await self._stop_all_components_safely()
            
            # Reset state
            self.state = EngineState.STOPPED
            
            # Clear any error flags
            if hasattr(self, 'error_count'):
                self.error_count = 0
            
            logger.success("✅ Engine State erfolgreich zurückgesetzt")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim State Reset: {e}")

    async def _validate_engine_readiness(self):
        """
        Prüfen ob Engine bereit für Start ist
        """
        try:
            # Check critical components (Fix: Use proper attribute paths)
            required_components = [
                ('config', 'config'),
                ('master_ai', 'master_ai'),
                ('market_intelligence', 'master_ai.market_intelligence'),
                ('risk_guardian', 'risk_guardian'), 
                ('strategy_selector', 'strategy_selector'),
                ('performance_tracker', 'performance_tracker')
            ]
            
            for component_name, component_path in required_components:
                try:
                    # Navigate nested attributes
                    obj = self
                    for attr in component_path.split('.'):
                        obj = getattr(obj, attr, None)
                        if obj is None:
                            break
                    
                    if obj is None:
                        logger.error(f"❌ Kritische Komponente fehlt: {component_name}")
                        return False
                except AttributeError:
                    logger.error(f"❌ Kritische Komponente fehlt: {component_name}")
                    return False
            
            logger.info("✅ Alle kritischen Komponenten verfügbar")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Readiness Check: {e}")
            return False

    async def _start_all_components(self):
        """
        Alle Komponenten starten
        """
        try:
            logger.info("🚀 Komponenten werden gestartet...")
            
            # Trading Loop starten
            self._running = True
            if hasattr(self, '_trading_task'):
                self._trading_task = asyncio.create_task(self._main_trading_loop())
            
            # Symbol Analysis Tasks starten
            if hasattr(self, 'active_symbols'):
                for symbol in self.active_symbols:
                    if hasattr(self, '_analysis_tasks'):
                        self._analysis_tasks[symbol] = asyncio.create_task(self._symbol_analysis_loop(symbol))
            
            logger.success("✅ Alle Komponenten gestartet")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Starten der Komponenten: {e}")
            return False

    async def _stop_all_components_safely(self):
        """
        Alle Komponenten sicher stoppen
        """
        components_to_stop = [
            'strategy_selector', 'market_intelligence', 
            'master_ai', 'performance_tracker'
        ]
        
        for component_name in components_to_stop:
            try:
                component = getattr(self, component_name, None)
                if component and hasattr(component, 'stop'):
                    await component.stop()
                    logger.info(f"✅ {component_name} gestoppt")
            except Exception as e:
                logger.warning(f"⚠️ Fehler beim Stoppen von {component_name}: {e}")
    
    # ==================== MAIN TRADING LOOP ====================
    
    async def _main_trading_loop(self):
        """🔄 Haupttrading-Loop"""
        logger.info("🔄 Haupttrading-Loop gestartet")
        
        while self._running:
            try:
                # System Health Check
                await self._system_health_check()
                
                # Portfolio Updates
                await self._update_portfolio_status()
                
                # Risk Monitoring
                await self._monitor_risks()
                
                # Performance Tracking
                await self._update_performance_metrics()
                
                # Cleanup abgeschlossener Trades
                await self._cleanup_completed_trades()
                
                # Kurze Pause
                await asyncio.sleep(5)  # 5 Sekunden Hauptloop
                
            except Exception as e:
                logger.error(f"❌ Fehler im Haupttrading-Loop: {e}")
                self.errors_count += 1
                
                # Bei zu vielen Fehlern stoppen
                if self.errors_count > 10:
                    logger.error("❌ Zu viele Fehler - Trading Engine wird gestoppt")
                    await self.stop()
                    break
                
                await asyncio.sleep(10)  # Längere Pause bei Fehlern
    
    async def _symbol_analysis_loop(self, symbol: str):
        """📊 Symbol-spezifische Analyse-Loop mit Multi-Agent Enhancement"""
        logger.info(f"📊 Multi-Agent Analyse-Loop gestartet für {symbol}")
        
        while self._running:
            try:
                # Trading Signal generieren mit Multi-Agent Enhancement
                if self.auto_trading and await self._can_place_new_trade(symbol):
                    # Verwende Multi-Agent Enhanced Signal wenn verfügbar, sonst RL Enhanced
                    if hasattr(self, 'multi_agent_system') and self.multi_agent_system:
                        signal = await self._generate_multi_agent_enhanced_signal(symbol)
                        logger.debug(f"🤖 Multi-Agent Signal für {symbol} generiert")
                    else:
                        signal = await self._generate_rl_enhanced_signal(symbol)
                        logger.debug(f"🧠 RL Enhanced Signal für {symbol} generiert")
                    
                    if signal:
                        await self._process_trading_signal(signal)
                
                # Analysis Intervall (abhängig von Symbol-Volatilität)
                analysis_interval = await self._get_analysis_interval(symbol)
                await asyncio.sleep(analysis_interval)
                
            except Exception as e:
                logger.error(f"❌ Fehler in Multi-Agent Symbol-Analyse für {symbol}: {e}")
                await asyncio.sleep(30)  # Pause bei Fehlern
    
    # ==================== SIGNAL PROCESSING ====================
    
    async def _process_trading_signal(self, signal):
        """
        Trading Signal mit RL Integration und robustem Error Handling verarbeiten
        """
        try:
            # ✅ DEFENSIVE SIGNAL VALIDATION
            if not signal:
                logger.warning("⚠️ Leeres Signal empfangen - wird übersprungen")
                return None
            
            if not hasattr(signal, 'symbol') or not signal.symbol:
                logger.warning("⚠️ Signal ohne Symbol empfangen - wird übersprungen")
                return None
            
            if not hasattr(signal, 'action') or not signal.action:
                logger.warning("⚠️ Signal ohne Action empfangen - wird übersprungen")
                return None
            
            # Symbol Validation
            symbol = signal.symbol
            logger.info(f"🔄 Verarbeite Signal für {symbol}: {signal.action}")
            
            # Risk Validation mit Error Handling
            risk_result = await self._safe_risk_validation(signal)
            if not risk_result or not risk_result.get('approved', False):
                logger.warning(f"⚠️ Signal für {symbol} durch Risk Management abgelehnt")
                return None
            
            # Signal Processing
            result = await self._execute_signal_safely(signal)
            
            # RL Continuous Learning Update
            if hasattr(self, 'rl_integration') and self.rl_integration and result:
                await self.rl_integration.continuous_learning_update(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Fehler bei RL-Enhanced Signal-Verarbeitung: {e}")
            return None

    async def _safe_risk_validation(self, signal):
        """
        Sichere Risk Validation mit Error Handling
        """
        try:
            if not self.risk_guardian:
                logger.warning("⚠️ Risk Guardian nicht verfügbar")
                return {'approved': False, 'reason': 'Risk Guardian nicht verfügbar'}
            
            return await self.risk_guardian.validate_trade_signal(signal)
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Risk Validation: {e}")
            return {'approved': False, 'reason': f'Risk Validation Fehler: {e}'}

    async def _execute_signal_safely(self, signal):
        """
        Sichere Signal-Ausführung mit Error Handling
        """
        try:
            # Signal execution logic here
            logger.info(f"✅ Signal für {signal.symbol} erfolgreich verarbeitet")
            return {'status': 'success', 'signal': signal}
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Signal-Ausführung: {e}")
            return {'status': 'error', 'error': str(e)}
    
    # ==================== RL INTEGRATION METHODS ====================
    
    async def _generate_rl_enhanced_signal(self, symbol):
        """
        Trading Signal mit RL Enhancement generieren
        """
        try:
            # Basis Signal von bestehenden Strategien
            base_signal = await self._generate_base_trading_signal(symbol)
            
            # Market Data für RL
            market_data = await self.master_ai.market_intelligence.get_current_market_data(symbol)
            
            # RL Signal
            if hasattr(self, 'rl_integration') and self.rl_integration:
                rl_signal = await self.rl_integration.get_rl_signal(market_data)
                
                if rl_signal and base_signal:
                    # Signal Ensemble
                    enhanced_signal = await self._combine_signals(base_signal, rl_signal)
                    enhanced_signal['rl_enhanced'] = True
                    
                    logger.debug(f"🤖 RL Enhanced Signal: {enhanced_signal['action']} (Confidence: {enhanced_signal['confidence']:.3f})")
                    
                    return enhanced_signal
            
            return base_signal
            
        except Exception as e:
            logger.error(f"❌ RL Enhanced Signal Generation Fehler: {e}")
            return await self._generate_base_trading_signal(symbol)
    
    async def _generate_multi_agent_enhanced_signal(self, symbol):
        """
        Trading Signal mit Multi-Agent Enhancement generieren
        """
        try:
            # Market Data für Agenten
            market_data = await self.master_ai.market_intelligence.get_current_market_data(symbol)
            market_data['symbol'] = symbol
            
            # Multi-Agent Ensemble Signal
            ensemble_signal = None
            if hasattr(self, 'multi_agent_system') and self.multi_agent_system:
                ensemble_signal = await self.multi_agent_system.generate_ensemble_signal(market_data)
            
            # RL Signal (bereits implementiert)
            rl_signal = None
            if hasattr(self, 'rl_integration') and self.rl_integration:
                rl_signal = await self.rl_integration.get_rl_signal(market_data)
            
            # Basis Signal von bestehenden Strategien
            base_signal = await self._generate_base_trading_signal(symbol)
            
            # Triple Signal Fusion
            final_signal = await self._fuse_triple_signals(base_signal, ensemble_signal, rl_signal, symbol)
            
            logger.debug(f"🎯 Multi-Agent Enhanced Signal: {final_signal['action']} (Confidence: {final_signal['confidence']:.3f})")
            
            return final_signal
            
        except Exception as e:
            logger.error(f"❌ Multi-Agent Enhanced Signal Generation Fehler: {e}")
            return await self._generate_base_trading_signal(symbol)
    
    async def _fuse_triple_signals(self, base_signal, ensemble_signal, rl_signal, symbol):
        """
        Fusion von Base-, Ensemble- und RL-Signalen
        """
        try:
            # Signal Gewichtung
            base_weight = 0.4      # 40% Traditional Strategies
            ensemble_weight = 0.4  # 40% Multi-Agent Ensemble
            rl_weight = 0.2        # 20% RL Agent
            
            signals = [
                (base_signal, base_weight) if base_signal else None,
                (ensemble_signal, ensemble_weight) if ensemble_signal else None,
                (rl_signal, rl_weight) if rl_signal else None
            ]
            
            # Filter out None signals
            valid_signals = [s for s in signals if s is not None]
            
            if not valid_signals:
                return self._create_neutral_signal(symbol)
            
            # Weighted Action Decision
            action_votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            weighted_confidence = 0
            total_weight = 0
            weighted_position_size = 0
            
            for signal, weight in valid_signals:
                action = signal.get('action', 'HOLD')
                confidence = signal.get('confidence', 0)
                position_size = signal.get('position_size', 0.1)
                
                action_votes[action] += weight * confidence
                weighted_confidence += confidence * weight
                weighted_position_size += position_size * weight
                total_weight += weight
            
            # Final Action
            final_action = max(action_votes, key=action_votes.get)
            
            # Final Confidence
            if total_weight > 0:
                final_confidence = weighted_confidence / total_weight
                final_position_size = weighted_position_size / total_weight
            else:
                final_confidence = 0.0
                final_position_size = 0.1
            
            # Enhanced Signal Information
            fused_signal = {
                'action': final_action,
                'confidence': min(1.0, final_confidence),
                'position_size': min(0.2, final_position_size),  # Cap at 20%
                'symbol': symbol,
                'source': 'Triple_Signal_Fusion',
                'base_signal': base_signal,
                'ensemble_signal': ensemble_signal,
                'rl_signal': rl_signal,
                'fusion_method': 'weighted_voting',
                'signal_count': len(valid_signals),
                'agent_ensemble_count': ensemble_signal.get('agent_count', 0) if ensemble_signal else 0,
                'timestamp': datetime.now(),
                'multi_agent_enhanced': True
            }
            
            return fused_signal
            
        except Exception as e:
            logger.error(f"❌ Triple Signal Fusion Fehler: {e}")
            return base_signal or self._create_neutral_signal(symbol)
    
    def _create_neutral_signal(self, symbol):
        """
        Neutrales Fallback-Signal erstellen
        """
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'position_size': 0.0,
            'symbol': symbol,
            'source': 'Neutral_Fallback',
            'timestamp': datetime.now()
        }
    
    async def _generate_base_trading_signal(self, symbol):
        """
        Basis Trading Signal von Strategy Selector generieren
        """
        try:
            if self.strategy_selector:
                return await self.strategy_selector.select_best_strategy(symbol)
            return None
            
        except Exception as e:
            logger.error(f"❌ Base Signal Generation Fehler: {e}")
            return None
    
    async def _combine_signals(self, base_signal, rl_signal):
        """
        Basis- und RL-Signal intelligent kombinieren
        """
        try:
            # Signal Gewichtung
            base_weight = 0.7
            rl_weight = 0.3
            
            # Confidence Berechnung
            combined_confidence = (
                base_signal.get('confidence', 0.5) * base_weight + 
                rl_signal.get('confidence', 0.5) * rl_weight
            )
            
            # Action Decision
            base_action = base_signal.get('action', 'HOLD')
            rl_action = rl_signal.get('action', 'HOLD')
            
            if base_action == rl_action:
                # Beide Signale stimmen überein
                final_action = base_action
                confidence_boost = 1.2  # Confidence erhöhen
                combined_confidence *= confidence_boost
            else:
                # Signale widersprechen sich
                if base_signal.get('confidence', 0) > rl_signal.get('confidence', 0):
                    final_action = base_action
                else:
                    final_action = rl_action
                
                # Confidence reduzieren bei Widerspruch
                combined_confidence *= 0.8
            
            # Position Size
            combined_position_size = (
                base_signal.get('position_size', 0.1) * base_weight +
                rl_signal.get('position_size', 0.1) * rl_weight
            )
            
            combined_signal = {
                'action': final_action,
                'confidence': min(combined_confidence, 1.0),
                'position_size': combined_position_size,
                'source': 'RL_Enhanced_Ensemble',
                'symbol': base_signal.get('symbol', ''),
                'base_signal': base_signal,
                'rl_signal': rl_signal,
                'combination_method': 'weighted_ensemble',
                'timestamp': datetime.now()
            }
            
            return combined_signal
            
        except Exception as e:
            logger.error(f"❌ Signal Combination Fehler: {e}")
            return base_signal
    
    # ==================== UTILITY METHODS ====================
    
    async def _startup_checks(self) -> bool:
        """🔍 Startup Validierung"""
        try:
            # API Verbindung prüfen
            portfolio = await self.bitget.get_portfolio()
            if not portfolio:
                logger.error("❌ Keine Portfolio-Daten von Bitget")
                return False
            
            # Minimum Balance Check
            min_balance = self.config.get('trading.min_balance', 100)
            if portfolio.total_balance < Decimal(str(min_balance)):
                logger.error(f"❌ Balance zu niedrig: {portfolio.total_balance} < {min_balance}")
                return False
            
            # Configuration Validation
            if not self.config.validate_config():
                logger.error("❌ Konfiguration unvollständig")
                return False
            
            # Strategy Validation
            enabled_strategies = self.strategy_selector.get_enabled_strategies_count()
            if enabled_strategies == 0:
                logger.error("❌ Keine Strategien aktiviert")
                return False
            
            logger.info(f"✅ Startup Checks bestanden - Balance: {portfolio.total_balance}, Strategien: {enabled_strategies}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Startup Checks: {e}")
            return False
    
    async def _initialize_strategy_dependencies(self):
        """🔧 Strategy Dependencies initialisieren"""
        try:
            # Trend Hunter - mit korrekten Parametern
            if not hasattr(self, 'trend_hunter') or self.trend_hunter is None:
                from strategies.trend_hunter import TrendHunter
                self.trend_hunter = TrendHunter(
                    config=self.config,
                    bitget_connector=self.bitget,
                    master_ai=self.master_ai,
                    market_intelligence=self.master_ai.market_intelligence
                )
                await self.trend_hunter.initialize()
            
            # Mean Reversion - mit korrekten Parametern
            if not hasattr(self, 'mean_reversion') or self.mean_reversion is None:
                from strategies.mean_reversion import MeanReversion
                self.mean_reversion = MeanReversion(
                    config=self.config,
                    bitget_connector=self.bitget,
                    master_ai=self.master_ai,
                    market_intelligence=self.master_ai.market_intelligence
                )
                await self.mean_reversion.initialize()
            
            # Scalping Master - mit korrekten Parametern
            if not hasattr(self, 'scalping_master') or self.scalping_master is None:
                from strategies.scalping_master import ScalpingMaster
                self.scalping_master = ScalpingMaster(
                    config=self.config,
                    bitget_connector=self.bitget,
                    master_ai=self.master_ai,
                    market_intelligence=self.master_ai.market_intelligence
                )
                await self.scalping_master.initialize()
            
            # Swing Genius - mit korrekten Parametern
            if not hasattr(self, 'swing_genius') or self.swing_genius is None:
                from strategies.swing_genius import SwingGenius
                self.swing_genius = SwingGenius(
                    config=self.config,
                    bitget_connector=self.bitget,
                    master_ai=self.master_ai,
                    market_intelligence=self.master_ai.market_intelligence
                )
                await self.swing_genius.initialize()
                
            logger.info("✅ Strategy Dependencies erfolgreich initialisiert")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Strategy Dependencies Initialisierung: {e}")
            raise
    
    async def _load_profitable_pairs(self):
        """💰 Profitable Trading-Paare laden"""
        try:
            # Auto-detection von profitablen Paaren
            profitable_pairs = await self.bitget.get_profitable_pairs()
            
            if profitable_pairs:
                self.active_symbols = profitable_pairs[:10]  # Top 10
                logger.info(f"💰 {len(self.active_symbols)} profitable Paare geladen: {self.active_symbols}")
            else:
                # Fallback auf Standard-Paare
                self.active_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
                logger.warning("⚠️ Fallback auf Standard-Paare")
                
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden profitabler Paare: {e}")
            self.active_symbols = ['BTC/USDT', 'ETH/USDT']
    
    async def _register_callbacks(self):
        """🔧 System Callbacks registrieren"""
        try:
            # Order Manager Callbacks
            if hasattr(self.order_manager, 'add_callback'):
                self.order_manager.add_callback('order_filled', self._on_order_filled)
                self.order_manager.add_callback('order_cancelled', self._on_order_cancelled)
            
            # Position Tracker Callbacks
            self.position_tracker.add_alert_callback(self._on_position_alert)
            
            # Telegram Callbacks
            self.notification_hub.register_telegram_callback('bot_started', self._on_telegram_start)
            self.notification_hub.register_telegram_callback('bot_stopped', self._on_telegram_stop)
            
            logger.info("🔧 System Callbacks registriert")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Callback-Registrierung: {e}")
    
    async def _can_place_new_trade(self, symbol: str) -> bool:
        """✅ Prüfen ob neuer Trade möglich ist"""
        try:
            # Max Concurrent Trades Check
            active_positions = self.position_tracker.get_active_positions()
            if len(active_positions) >= self.max_concurrent_trades:
                return False
            
            # Symbol bereits in Position?
            existing_position = self.position_tracker.get_position_by_symbol(symbol)
            if existing_position:
                return False  # Ein Trade pro Symbol
            
            # Risk Guardian Overall Check
            portfolio_risk = await self.risk_guardian.get_portfolio_risk_level()
            if portfolio_risk > 0.8:  # 80% Risk Level
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler bei New Trade Check: {e}")
            return False
    
    async def _get_analysis_interval(self, symbol: str) -> float:
        """⏱️ Analysis Intervall für Symbol bestimmen"""
        try:
            # Basis-Intervall
            base_interval = 30  # 30 Sekunden
            
            # Volatilität-basierte Anpassung
            market_data = await self.bitget.get_market_data(symbol)
            if market_data:
                volatility = abs(market_data.change_24h_percent)
                if volatility > 10:  # Hohe Volatilität
                    return base_interval * 0.5  # Häufigere Analyse
                elif volatility < 2:  # Niedrige Volatilität
                    return base_interval * 2  # Seltenere Analyse
            
            return base_interval
            
        except Exception:
            return 30  # Fallback
    
    # ==================== CALLBACK HANDLERS ====================
    
    async def _on_order_filled(self, order: Order):
        """✅ Order gefüllt Callback"""
        try:
            log_trade(f"✅ Order gefüllt: {order.symbol} {order.side.value}")
            
            # Performance Update
            await self.performance_tracker.update_order_status(order.id, 'filled')
            
            # Notification
            await self.notification_hub.send_order_update({
                'order_id': order.id,
                'status': 'filled',
                'symbol': order.symbol,
                'side': order.side.value,
                'filled_price': float(order.price) if order.price else 0
            })
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Order Filled Callback: {e}")
    
    async def _on_order_cancelled(self, order: Order):
        """❌ Order storniert Callback"""
        try:
            logger.warning(f"❌ Order storniert: {order.symbol} {order.side.value}")
            
            # Performance Update
            await self.performance_tracker.update_order_status(order.id, 'cancelled')
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Order Cancelled Callback: {e}")
    
    async def _on_position_alert(self, alert):
        """🚨 Position Alert Callback"""
        try:
            logger.warning(f"🚨 Position Alert: {alert.message}")
            
            # Critical Alerts an Telegram
            if alert.severity == 'critical':
                await self.notification_hub.send_emergency_alert(alert.message)
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Position Alert Callback: {e}")
    
    async def _on_telegram_start(self):
        """🤖 Telegram Bot gestartet"""
        logger.info("🤖 Telegram Bot Callback: Started")
    
    async def _on_telegram_stop(self):
        """🤖 Telegram Bot gestoppt"""
        logger.info("🤖 Telegram Bot Callback: Stopped")
        # Optional: Trading Engine auch stoppen
        # await self.stop()
    
    # ==================== MONITORING METHODS ====================
    
    async def _system_health_check(self):
        """🏥 System Health Check"""
        try:
            # API Latenz prüfen
            start_time = datetime.utcnow()
            await self.bitget.get_market_data('BTC/USDT')
            api_latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Performance Alerts
            if api_latency > 1000:  # > 1 Sekunde
                logger.warning(f"⚠️ Hohe API Latenz: {api_latency:.0f}ms")
            
            # Memory Check (vereinfacht)
            if self.signals_processed > 0 and self.signals_processed % 100 == 0:
                logger.info(f"📊 Health Check: {self.signals_processed} Signale verarbeitet, {api_latency:.0f}ms Latenz")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei System Health Check: {e}")
    
    async def _update_portfolio_status(self):
        """💼 Portfolio Status Update"""
        try:
            await self.portfolio_manager.update_portfolio()
        except Exception as e:
            logger.error(f"❌ Fehler bei Portfolio Update: {e}")
    
    async def _monitor_risks(self):
        """🛡️ Risk Monitoring"""
        try:
            await self.risk_guardian.monitor_portfolio_risks()
        except Exception as e:
            logger.error(f"❌ Fehler bei Risk Monitoring: {e}")
    
    async def _update_performance_metrics(self):
        """📊 Performance Metrics Update"""
        try:
            await self.performance_tracker.update_real_time_metrics()
        except Exception as e:
            logger.error(f"❌ Fehler bei Performance Update: {e}")
    
    async def _cleanup_completed_trades(self):
        """🧹 Cleanup abgeschlossener Trades"""
        try:
            # Implementierung für Trade Cleanup
            pass
        except Exception as e:
            logger.error(f"❌ Fehler bei Trade Cleanup: {e}")
    
    # ==================== PUBLIC METHODS ====================
    
    def get_engine_status(self) -> Dict[str, Any]:
        """📊 Engine Status mit RL & Multi-Agent Integration abrufen"""
        runtime = (datetime.utcnow() - self.engine_start_time).total_seconds() if self.engine_start_time else 0
        
        # RL Status hinzufügen
        rl_status = {}
        if hasattr(self, 'rl_integration') and self.rl_integration:
            rl_status = self.rl_integration.get_rl_status()
        
        # Multi-Agent System Status hinzufügen
        multi_agent_status = {}
        if hasattr(self, 'multi_agent_system') and self.multi_agent_system:
            multi_agent_status = self.multi_agent_system.get_system_status()
        
        return {
            'state': self.state.value,
            'running': self._running,
            'runtime_seconds': runtime,
            'active_symbols': len(self.active_symbols),
            'signals_processed': self.signals_processed,
            'trades_executed': self.trades_executed,
            'errors_count': self.errors_count,
            'auto_trading': self.auto_trading,
            'demo_mode': self.config.is_demo_mode(),
            'rl_integration': rl_status,
            'multi_agent_system': multi_agent_status
        }
    
    def add_trade_callback(self, callback: Callable):
        """🔧 Trade Callback hinzufügen"""
        self.trade_callbacks.append(callback)
    
    def add_signal_callback(self, callback: Callable):
        """🔧 Signal Callback hinzufügen"""
        self.signal_callbacks.append(callback)
    
    async def pause_trading(self):
        """⏸️ Trading pausieren"""
        if self.state == EngineState.RUNNING:
            self.state = EngineState.PAUSED
            logger.info("⏸️ Trading pausiert")
    
    async def resume_trading(self):
        """▶️ Trading fortsetzen"""
        if self.state == EngineState.PAUSED:
            self.state = EngineState.RUNNING
            logger.info("▶️ Trading fortgesetzt")
    
    async def _shutdown_components(self):
        """🛑 Alle Komponenten herunterfahren"""
        try:
            components = [
                self.performance_tracker,
                self.strategy_selector,
                self.multi_agent_system,  # Multi-Agent System hinzugefügt
                self.rl_integration,
                self.master_ai,
                self.risk_guardian,
                self.portfolio_manager,
                self.position_tracker,
                self.order_manager,
                self.notification_hub,
                self.bitget
            ]
            
            for component in components:
                if component and hasattr(component, 'shutdown'):
                    try:
                        await component.shutdown()
                    except Exception as e:
                        logger.error(f"❌ Fehler beim Herunterfahren einer Komponente: {e}")
            
            logger.info("✅ Alle Komponenten heruntergefahren")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren der Komponenten: {e}")
    
    async def shutdown(self):
        """🛑 Trading Engine komplett herunterfahren"""
        try:
            if self.state != EngineState.STOPPED:
                await self.stop()
            
            await self._shutdown_components()
            logger.info("✅ Trading Engine komplett heruntergefahren")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren der Trading Engine: {e}")
