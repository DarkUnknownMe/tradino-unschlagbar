"""
🧪 TRADINO UNSCHLAGBAR - Strategy Selector
Intelligente Strategie-Auswahl basierend auf Marktbedingungen

Author: AI Trading Systems
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

from models.trade_models import TradeSignal
from models.market_models import MarketRegime
from strategies.scalping_master import ScalpingMaster
from strategies.swing_genius import SwingGenius
from strategies.trend_hunter import TrendHunter
from strategies.mean_reversion import MeanReversion
from brain.master_ai import MasterAI
from brain.market_intelligence import MarketIntelligence
from utils.logger_pro import setup_logger, log_ai_decision
from utils.config_manager import ConfigManager

logger = setup_logger("StrategySelector")


class StrategyType(Enum):
    """Strategie Typen"""
    SCALPING = "scalping_master"
    SWING = "swing_genius"
    TREND = "trend_hunter"
    MEAN_REVERSION = "mean_reversion"


class StrategySelector:
    """🧪 Intelligente Strategie-Auswahl"""
    
    def __init__(self, config: ConfigManager, scalping_master: ScalpingMaster, 
                 swing_genius: SwingGenius, trend_hunter: TrendHunter, 
                 mean_reversion: MeanReversion, master_ai: MasterAI, 
                 market_intelligence: MarketIntelligence):
        self.config = config
        self.scalping_master = scalping_master
        self.swing_genius = swing_genius
        self.trend_hunter = trend_hunter
        self.mean_reversion = mean_reversion
        self.master_ai = master_ai
        self.market_intelligence = market_intelligence
        
        # Strategy Performance Tracking
        self.strategy_performance: Dict[str, Dict] = {
            'scalping_master': {'signals': 0, 'wins': 0, 'losses': 0, 'last_used': None},
            'swing_genius': {'signals': 0, 'wins': 0, 'losses': 0, 'last_used': None},
            'trend_hunter': {'signals': 0, 'wins': 0, 'losses': 0, 'last_used': None},
            'mean_reversion': {'signals': 0, 'wins': 0, 'losses': 0, 'last_used': None}
        }
        
        # Market Condition Mappings
        self.strategy_preferences = {
            MarketRegime.HIGH_VOLATILITY: ['scalping_master', 'swing_genius'],
            MarketRegime.LOW_VOLATILITY: ['mean_reversion', 'scalping_master'],
            MarketRegime.TRENDING_UP: ['trend_hunter', 'swing_genius'],
            MarketRegime.TRENDING_DOWN: ['trend_hunter', 'swing_genius'],
            MarketRegime.RANGE_BOUND: ['mean_reversion', 'scalping_master'],
            MarketRegime.BREAKOUT: ['swing_genius', 'trend_hunter'],
            MarketRegime.REVERSAL: ['swing_genius', 'mean_reversion']
        }
        
        # Selection Statistics
        self.selections_made = 0
        self.last_selection_time = datetime.utcnow()
        
    async def initialize(self) -> bool:
        """🔥 Strategy Selector initialisieren"""
        try:
            logger.info("🧪 Strategy Selector wird initialisiert...")
            logger.success("✅ Strategy Selector erfolgreich initialisiert")
            return True
        except Exception as e:
            logger.error(f"❌ Strategy Selector Initialisierung fehlgeschlagen: {e}")
            return False
    
    async def select_best_strategy(self, symbol: str) -> Optional[TradeSignal]:
        """🎯 Beste Strategie für aktuelles Market Regime auswählen"""
        try:
            logger.info(f"🧪 Strategie-Auswahl wird durchgeführt: {symbol}")
            
            # Market Analysis für Regime-Bestimmung
            market_analysis = await self.market_intelligence.analyze_market(symbol)
            if not market_analysis:
                logger.warning(f"⚠️ Keine Market Analysis für {symbol}")
                return None
            
            # Preferred Strategies basierend auf Market Regime
            preferred_strategies = self.strategy_preferences.get(
                market_analysis.regime, 
                ['scalping_master', 'swing_genius']
            )
            
            # Strategy Signals parallel abrufen
            strategy_signals = await self._get_all_strategy_signals(symbol)
            
            if not strategy_signals:
                logger.info(f"📊 Keine Strategie-Signale für {symbol}")
                return None
            
            # Beste Strategie auswählen
            best_signal = await self._select_optimal_signal(
                strategy_signals, preferred_strategies, market_analysis
            )
            
            if best_signal:
                strategy_name = best_signal.strategy
                self.strategy_performance[strategy_name]['signals'] += 1
                self.strategy_performance[strategy_name]['last_used'] = datetime.utcnow()
                self.selections_made += 1
                
                log_ai_decision(
                    "StrategySelector",
                    f"{symbol} {strategy_name} selected",
                    best_signal.confidence
                )
                
                logger.info(f"🎯 Strategie ausgewählt: {strategy_name} (Confidence: {best_signal.confidence:.2%})")
            
            return best_signal
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Strategie-Auswahl für {symbol}: {e}")
            return None
    
    async def _get_all_strategy_signals(self, symbol: str) -> Dict[str, TradeSignal]:
        """📊 Alle verfügbaren Strategie-Signale abrufen"""
        try:
            signals = {}
            tasks = []
            
            # Alle 4 Strategien
            if self.scalping_master.enabled:
                tasks.append(('scalping_master', self.scalping_master.generate_trade_signal(symbol)))
            
            if self.swing_genius.enabled:
                tasks.append(('swing_genius', self.swing_genius.generate_trade_signal(symbol)))
            
            if self.trend_hunter.enabled:
                tasks.append(('trend_hunter', self.trend_hunter.generate_trade_signal(symbol)))
            
            if self.mean_reversion.enabled:
                tasks.append(('mean_reversion', self.mean_reversion.generate_trade_signal(symbol)))
            
            # Alle Tasks parallel ausführen
            if tasks:
                results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
                
                for i, (strategy_name, result) in enumerate([(task[0], results[i]) for task in tasks]):
                    if not isinstance(result, Exception) and result:
                        signals[strategy_name] = result
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Abrufen der Strategie-Signale: {e}")
            return {}
    
    async def _select_optimal_signal(self, strategy_signals: Dict[str, TradeSignal], 
                                   preferred_strategies: List[str], market_analysis) -> Optional[TradeSignal]:
        """🏆 Optimales Signal basierend auf Kriterien auswählen"""
        try:
            if not strategy_signals:
                return None
            
            # Signal Scoring
            signal_scores = []
            
            for strategy_name, signal in strategy_signals.items():
                # Base Score = Signal Confidence
                base_score = signal.confidence
                
                # Market Regime Preference Bonus
                preference_bonus = 0
                if strategy_name in preferred_strategies:
                    preference_index = preferred_strategies.index(strategy_name)
                    preference_bonus = (len(preferred_strategies) - preference_index) * 0.1
                
                # Strategy Performance Bonus
                performance_bonus = self._calculate_performance_bonus(strategy_name)
                
                # Risk/Reward Bonus
                rr_bonus = self._calculate_risk_reward_bonus(signal)
                
                # Market Condition Alignment
                alignment_bonus = self._calculate_market_alignment_bonus(signal, market_analysis)
                
                # Final Score
                final_score = base_score + preference_bonus + performance_bonus + rr_bonus + alignment_bonus
                
                signal_scores.append({
                    'strategy': strategy_name,
                    'signal': signal,
                    'score': final_score,
                    'breakdown': {
                        'base_confidence': base_score,
                        'preference_bonus': preference_bonus,
                        'performance_bonus': performance_bonus,
                        'risk_reward_bonus': rr_bonus,
                        'alignment_bonus': alignment_bonus
                    }
                })
            
            # Bestes Signal auswählen
            if signal_scores:
                best_signal_data = max(signal_scores, key=lambda x: x['score'])
                
                logger.info(f"🏆 Signal Scoring: {best_signal_data['strategy']} "
                          f"(Score: {best_signal_data['score']:.2f})")
                
                return best_signal_data['signal']
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Fehler bei optimaler Signal-Auswahl: {e}")
            return None
    
    def _calculate_performance_bonus(self, strategy_name: str) -> float:
        """📈 Performance-Bonus für Strategie berechnen"""
        try:
            perf = self.strategy_performance[strategy_name]
            total_trades = perf['wins'] + perf['losses']
            
            if total_trades == 0:
                return 0.0  # Kein Bonus für ungetestete Strategien
            
            win_rate = perf['wins'] / total_trades
            
            # Bonus basierend auf Win Rate
            if win_rate >= 0.7:
                return 0.15  # Sehr gut
            elif win_rate >= 0.6:
                return 0.10  # Gut
            elif win_rate >= 0.5:
                return 0.05  # Durchschnitt
            else:
                return -0.05  # Unter Durchschnitt
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Performance Bonus: {e}")
            return 0.0
    
    def _calculate_risk_reward_bonus(self, signal: TradeSignal) -> float:
        """⚖️ Risk/Reward Bonus berechnen"""
        try:
            if not signal.stop_loss or not signal.take_profit:
                return 0.0
            
            entry = float(signal.entry_price)
            sl = float(signal.stop_loss)
            tp = float(signal.take_profit)
            
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            
            if risk == 0:
                return 0.0
            
            rr_ratio = reward / risk
            
            # Bonus für gute Risk/Reward Ratios
            if rr_ratio >= 3.0:
                return 0.10
            elif rr_ratio >= 2.0:
                return 0.05
            elif rr_ratio >= 1.5:
                return 0.02
            else:
                return -0.02  # Penalty für schlechte RR
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Risk/Reward Bonus: {e}")
            return 0.0
    
    def _calculate_market_alignment_bonus(self, signal: TradeSignal, market_analysis) -> float:
        """🎯 Market Alignment Bonus berechnen"""
        try:
            # Trend Strength Alignment
            trend_strength = market_analysis.trend_strength
            
            # Volume Confirmation
            volume_score = market_analysis.volume_score
            
            # Volatility Suitability (je nach Strategie)
            volatility = market_analysis.volatility_score
            
            alignment_score = 0
            
            # Trend Alignment (für alle Strategien gut)
            alignment_score += trend_strength * 0.05
            
            # Volume Alignment
            if volume_score >= 0.6:
                alignment_score += 0.03
            
            # Volatility Alignment (strategie-spezifisch)
            if signal.strategy == 'scalping_master':
                # Scalping bevorzugt moderate Volatilität
                if 0.3 <= volatility <= 0.7:
                    alignment_score += 0.05
            elif signal.strategy == 'swing_genius':
                # Swing Trading bevorzugt höhere Volatilität
                if volatility >= 0.5:
                    alignment_score += 0.05
            
            return alignment_score
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Market Alignment Bonus: {e}")
            return 0.0
    
    # ==================== PERFORMANCE TRACKING ====================
    
    async def update_strategy_performance(self, signal_id: str, strategy_name: str, was_successful: bool):
        """📊 Strategie-Performance updaten"""
        try:
            if strategy_name in self.strategy_performance:
                if was_successful:
                    self.strategy_performance[strategy_name]['wins'] += 1
                else:
                    self.strategy_performance[strategy_name]['losses'] += 1
                
                # Alle Strategien informieren
                if strategy_name == 'scalping_master':
                    await self.scalping_master.update_performance(signal_id, was_successful)
                elif strategy_name == 'swing_genius':
                    await self.swing_genius.update_performance(signal_id, was_successful)
                elif strategy_name == 'trend_hunter':
                    await self.trend_hunter.update_performance(signal_id, was_successful)
                elif strategy_name == 'mean_reversion':
                    await self.mean_reversion.update_performance(signal_id, was_successful)
                
                logger.info(f"📊 Strategie-Performance updated: {strategy_name} -> {'Win' if was_successful else 'Loss'}")
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Strategy Performance Update: {e}")
    
    def get_strategy_rankings(self) -> List[Dict[str, Any]]:
        """🏆 Strategie-Rankings basierend auf Performance"""
        try:
            rankings = []
            
            for strategy_name, perf in self.strategy_performance.items():
                total_trades = perf['wins'] + perf['losses']
                win_rate = perf['wins'] / total_trades if total_trades > 0 else 0
                
                rankings.append({
                    'strategy': strategy_name,
                    'win_rate': win_rate,
                    'total_trades': total_trades,
                    'wins': perf['wins'],
                    'losses': perf['losses'],
                    'last_used': perf['last_used'],
                    'enabled': self._is_strategy_enabled(strategy_name)
                })
            
            # Nach Win Rate sortieren
            rankings.sort(key=lambda x: x['win_rate'], reverse=True)
            
            return rankings
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Strategy Rankings: {e}")
            return []
    
    def _is_strategy_enabled(self, strategy_name: str) -> bool:
        """✅ Prüfen ob Strategie aktiviert ist"""
        try:
            if strategy_name == 'scalping_master':
                return self.scalping_master.enabled
            elif strategy_name == 'swing_genius':
                return self.swing_genius.enabled
            elif strategy_name == 'trend_hunter':
                return self.trend_hunter.enabled
            elif strategy_name == 'mean_reversion':
                return self.mean_reversion.enabled
            else:
                return False
        except Exception:
            return False
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """📊 Selection Statistiken"""
        try:
            # Strategy Usage Statistics
            usage_stats = {}
            total_signals = sum(perf['signals'] for perf in self.strategy_performance.values())
            
            for strategy_name, perf in self.strategy_performance.items():
                usage_percentage = (perf['signals'] / total_signals * 100) if total_signals > 0 else 0
                usage_stats[strategy_name] = {
                    'signals': perf['signals'],
                    'usage_percentage': usage_percentage,
                    'last_used': perf['last_used']
                }
            
            return {
                'selections_made': self.selections_made,
                'last_selection_time': self.last_selection_time,
                'strategy_usage': usage_stats,
                'total_signals_generated': total_signals
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Selection Stats: {e}")
            return {}
    
    def get_market_regime_preferences(self) -> Dict[str, List[str]]:
        """📊 Market Regime Preferences abrufen"""
        return {regime.value: strategies for regime, strategies in self.strategy_preferences.items()}
    
    def get_enabled_strategies_count(self) -> int:
        """📊 Anzahl aktivierter Strategien"""
        try:
            enabled_count = 0
            if hasattr(self, 'scalping_master') and self.scalping_master.enabled:
                enabled_count += 1
            if hasattr(self, 'swing_genius') and self.swing_genius.enabled:
                enabled_count += 1
            if hasattr(self, 'trend_hunter') and self.trend_hunter.enabled:
                enabled_count += 1
            if hasattr(self, 'mean_reversion') and self.mean_reversion.enabled:
                enabled_count += 1
            return enabled_count
        except Exception as e:
            logger.error(f"❌ Fehler bei Enabled Strategies Count: {e}")
            return 0
    
    async def shutdown(self):
        """🛑 Strategy Selector herunterfahren"""
        try:
            logger.info("✅ Strategy Selector heruntergefahren")
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren: {e}")
