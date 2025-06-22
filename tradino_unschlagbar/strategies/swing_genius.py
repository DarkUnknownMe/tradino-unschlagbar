"""
📈 TRADINO UNSCHLAGBAR - Swing Genius Strategy
Intelligente Swing-Trading-Strategie für 1h-4h Timeframes
Target: 55-60% Win Rate, 1:2.5 Risk/Reward, 5-15 Trades/Woche

Author: AI Trading Systems
"""

import asyncio
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from models.trade_models import TradeSignal, OrderSide
from models.market_models import MarketRegime
from brain.master_ai import MasterAI
from brain.market_intelligence import MarketIntelligence
from connectors.bitget_pro import BitgetProConnector
from utils.logger_pro import setup_logger, log_trade
from utils.config_manager import ConfigManager
from utils.math_utils import calculate_position_size
from utils.helpers import generate_signal_id

logger = setup_logger("SwingGenius")


class SwingGenius:
    """📈 Intelligente Swing-Trading-Strategie"""
    
    def __init__(self, config: ConfigManager, bitget_connector: BitgetProConnector, 
                 master_ai: MasterAI, market_intelligence: MarketIntelligence):
        self.config = config
        self.bitget = bitget_connector
        self.master_ai = master_ai
        self.market_intelligence = market_intelligence
        
        # Strategy Configuration
        self.strategy_config = config.get('strategies.swing_genius', {})
        self.enabled = self.strategy_config.get('enabled', True)
        self.timeframes = self.strategy_config.get('timeframes', ['1h', '2h', '4h'])
        self.target_win_rate = self.strategy_config.get('target_win_rate', 0.58)
        self.risk_reward_ratio = self.strategy_config.get('risk_reward', 2.5)
        self.max_trades_per_week = self.strategy_config.get('max_trades_per_week', 15)
        
        # Swing Parameters
        self.min_trend_strength = 0.6  # 60% Minimum Trend-Stärke
        self.min_volume_confirmation = 1.2  # 120% Volumen-Bestätigung
        self.ideal_market_regimes = [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.BREAKOUT]
        
        # Technical Thresholds
        self.rsi_trend_min = 45
        self.rsi_trend_max = 55
        self.macd_histogram_threshold = 0.001
        
        # Performance Tracking
        self.signals_generated = 0
        self.trades_this_week = 0
        self.weekly_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        self.win_count = 0
        self.loss_count = 0
        
        # Analysis Cache
        self.swing_analysis_cache: Dict[str, Dict] = {}
        
    async def initialize(self) -> bool:
        """🔥 Swing Genius initialisieren"""
        try:
            logger.info("📈 Swing Genius wird initialisiert...")
            
            if not self.enabled:
                logger.info("⚠️ Swing Genius ist deaktiviert")
                return True
            
            logger.success(f"✅ Swing Genius initialisiert (Target Win Rate: {self.target_win_rate:.1%})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Swing Genius Initialisierung fehlgeschlagen: {e}")
            return False
    
    # ==================== MAIN STRATEGY LOGIC ====================
    
    async def analyze_swing_opportunity(self, symbol: str) -> Optional[Dict[str, Any]]:
        """🎯 Swing-Trading-Gelegenheit analysieren"""
        try:
            # Weekly Trade Limit Check
            await self._check_weekly_reset()
            if self.trades_this_week >= self.max_trades_per_week:
                logger.info(f"📊 Wöchentliches Trade-Limit erreicht: {self.trades_this_week}/{self.max_trades_per_week}")
                return None
            
            logger.info(f"📈 Swing-Analyse wird durchgeführt: {symbol}")
            
            # Market Intelligence abrufen
            market_analysis = await self.market_intelligence.analyze_market(symbol, self.timeframes)
            if not market_analysis:
                logger.info(f"📊 Keine Market Analysis für {symbol}")
                return None
            
            # Market Regime Check
            if market_analysis.regime not in self.ideal_market_regimes:
                logger.info(f"📊 Market Regime nicht geeignet für Swing: {market_analysis.regime.value}")
                return None
            
            # Trend Strength Check
            if market_analysis.trend_strength < self.min_trend_strength:
                logger.info(f"📊 Trend zu schwach für Swing: {market_analysis.trend_strength:.2%}")
                return None
            
            # Multi-Timeframe Swing Analysis
            swing_signals = []
            for timeframe in self.timeframes:
                signal = await self._analyze_swing_timeframe(symbol, timeframe, market_analysis)
                if signal:
                    swing_signals.append(signal)
            
            if not swing_signals:
                return None
            
            # Beste Signal auswählen
            best_signal = max(swing_signals, key=lambda x: x['confidence'])
            
            # AI-Enhanced Analysis
            ai_signal = await self.master_ai.analyze_and_generate_signal(symbol)
            if ai_signal and ai_signal.confidence > 0.7:
                # AI-Signal mit Swing-Signal kombinieren
                best_signal = await self._combine_with_ai_signal(best_signal, ai_signal)
            
            # Final Validation
            if best_signal['confidence'] >= 0.7:  # 70% Minimum für Swing
                self.signals_generated += 1
                log_trade(f"📈 Swing Signal generiert: {symbol} {best_signal['side'].value} (Confidence: {best_signal['confidence']:.2%})")
                return best_signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Swing-Analyse für {symbol}: {e}")
            return None
    
    async def _analyze_swing_timeframe(self, symbol: str, timeframe: str, market_analysis) -> Optional[Dict[str, Any]]:
        """📊 Timeframe-spezifische Swing-Analyse"""
        try:
            # Candles abrufen
            candles = await self.bitget.get_candles(symbol, timeframe, limit=100)
            if len(candles) < 50:
                return None
            
            # Technical Analysis
            technical_analysis = await self._analyze_swing_technicals(candles, timeframe)
            if not technical_analysis:
                return None
            
            # Trend Confirmation
            trend_confirmation = await self._analyze_trend_confirmation(candles, market_analysis)
            
            # Volume Analysis
            volume_analysis = await self._analyze_swing_volume(candles)
            
            # Support/Resistance Analysis
            sr_analysis = await self._analyze_support_resistance(candles, symbol)
            
            # Signal Score berechnen
            signal_score = self._calculate_swing_score(
                technical_analysis, trend_confirmation, volume_analysis, sr_analysis, timeframe
            )
            
            if signal_score['score'] < 0.65:  # Minimum Score für Swing
                return None
            
            # Entry, SL, TP berechnen
            current_price = float(candles[-1].close)
            signal_side = signal_score['side']
            
            entry_price, stop_loss, take_profit = self._calculate_swing_levels(
                current_price, signal_side, market_analysis, sr_analysis, timeframe
            )
            
            return {
                'symbol': symbol,
                'side': signal_side,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': signal_score['confidence'],
                'swing_score': signal_score['score'],
                'timeframe': timeframe,
                'analysis': {
                    'technical': technical_analysis,
                    'trend': trend_confirmation,
                    'volume': volume_analysis,
                    'support_resistance': sr_analysis
                },
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Swing Timeframe Analysis: {e}")
            return None
    
    async def _analyze_swing_technicals(self, candles: List, timeframe: str) -> Optional[Dict[str, Any]]:
        """📊 Technical Analysis für Swing Trading"""
        try:
            closes = [float(c.close) for c in candles]
            highs = [float(c.high) for c in candles]
            lows = [float(c.low) for c in candles]
            
            signals = []
            
            # EMA Golden/Death Cross (21 vs 50)
            ema_21 = self._calculate_ema(closes, 21)
            ema_50 = self._calculate_ema(closes, 50)
            
            if len(ema_21) >= 2 and len(ema_50) >= 2:
                # Golden Cross
                if ema_21[-1] > ema_50[-1] and ema_21[-2] <= ema_50[-2]:
                    signals.append({'type': 'golden_cross', 'side': OrderSide.BUY, 'strength': 0.8})
                # Death Cross
                elif ema_21[-1] < ema_50[-1] and ema_21[-2] >= ema_50[-2]:
                    signals.append({'type': 'death_cross', 'side': OrderSide.SELL, 'strength': 0.8})
            
            # MACD Divergence
            macd_line, macd_signal, macd_hist = self._calculate_macd(closes)
            if len(macd_hist) >= 5:
                # Bullish MACD Cross
                if (macd_line[-1] > macd_signal[-1] and macd_line[-2] <= macd_signal[-2] 
                    and macd_hist[-1] > self.macd_histogram_threshold):
                    signals.append({'type': 'macd_bullish', 'side': OrderSide.BUY, 'strength': 0.7})
                # Bearish MACD Cross
                elif (macd_line[-1] < macd_signal[-1] and macd_line[-2] >= macd_signal[-2] 
                      and macd_hist[-1] < -self.macd_histogram_threshold):
                    signals.append({'type': 'macd_bearish', 'side': OrderSide.SELL, 'strength': 0.7})
            
            # RSI Trend Alignment
            rsi = self._calculate_rsi(closes, 14)
            if rsi:
                current_rsi = rsi[-1]
                # RSI in Trend-Zone (nicht extreme Bereiche)
                if self.rsi_trend_min <= current_rsi <= self.rsi_trend_max:
                    signals.append({'type': 'rsi_trend_zone', 'strength': 0.5})
            
            # ADX Trend Strength
            adx = self._calculate_adx(highs, lows, closes, 14)
            if adx and adx[-1] > 25:  # Starker Trend
                signals.append({'type': 'strong_trend', 'strength': min(1.0, adx[-1] / 50)})
            
            if not signals:
                return None
            
            # Stärkstes Signal auswählen
            best_signal = max(signals, key=lambda x: x['strength'])
            
            return {
                'primary_signal': best_signal,
                'all_signals': signals,
                'signal_count': len(signals),
                'rsi': current_rsi if rsi else 50,
                'adx': adx[-1] if adx else 0,
                'ema_trend': 'bullish' if len(ema_21) >= 1 and len(ema_50) >= 1 and ema_21[-1] > ema_50[-1] else 'bearish'
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Swing Technical Analysis: {e}")
            return None
    
    async def _analyze_trend_confirmation(self, candles: List, market_analysis) -> Dict[str, Any]:
        """📈 Trend-Bestätigung analysieren"""
        try:
            closes = [float(c.close) for c in candles[-20:]]  # Letzte 20 Kerzen
            
            # Linear Regression Trend
            x = np.arange(len(closes))
            z = np.polyfit(x, closes, 1)
            slope = z[0]
            
            # R² für Trend-Qualität
            p = np.poly1d(z)
            predicted = p(x)
            ss_res = np.sum((closes - predicted) ** 2)
            ss_tot = np.sum((closes - np.mean(closes)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Trend Direction
            avg_price = np.mean(closes)
            slope_percent = (slope / avg_price) * 100 if avg_price > 0 else 0
            
            # Higher Highs / Lower Lows Pattern
            recent_highs = [float(c.high) for c in candles[-10:]]
            recent_lows = [float(c.low) for c in candles[-10:]]
            
            higher_highs = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
            lower_lows = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1])
            
            # Market Intelligence Alignment
            mi_trend_strength = market_analysis.trend_strength if market_analysis else 0
            mi_regime = market_analysis.regime.value if market_analysis else 'neutral'
            
            # Trend Confirmation Score
            trend_factors = [
                r_squared,  # Trend-Qualität
                min(1.0, abs(slope_percent) / 2),  # Trend-Stärke
                mi_trend_strength,  # Market Intelligence
                higher_highs / 9 if slope_percent > 0 else lower_lows / 9  # Pattern Confirmation
            ]
            
            trend_confirmation = sum(trend_factors) / len(trend_factors)
            
            return {
                'trend_confirmation': trend_confirmation,
                'slope_percent': slope_percent,
                'r_squared': r_squared,
                'higher_highs_ratio': higher_highs / 9,
                'lower_lows_ratio': lower_lows / 9,
                'mi_alignment': mi_trend_strength,
                'trend_direction': 'bullish' if slope_percent > 0.1 else 'bearish' if slope_percent < -0.1 else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Trend Confirmation: {e}")
            return {'trend_confirmation': 0, 'trend_direction': 'neutral'}
    
    async def _analyze_swing_volume(self, candles: List) -> Dict[str, Any]:
        """📊 Volume Analysis für Swing Trading"""
        try:
            volumes = [float(c.volume) for c in candles]
            
            if len(volumes) < 20:
                return {'volume_confirmation': 0.5}
            
            # Volume Trend
            recent_volume = volumes[-5:]
            prev_volume = volumes[-10:-5]
            
            volume_trend = (sum(recent_volume) - sum(prev_volume)) / sum(prev_volume) if sum(prev_volume) > 0 else 0
            
            # Volume Spike Detection
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # On-Balance Volume Trend
            obv = []
            obv_value = 0
            
            for i in range(1, len(candles)):
                if float(candles[i].close) > float(candles[i-1].close):
                    obv_value += volumes[i]
                elif float(candles[i].close) < float(candles[i-1].close):
                    obv_value -= volumes[i]
                obv.append(obv_value)
            
            # OBV Trend
            if len(obv) >= 10:
                obv_recent = obv[-5:]
                obv_prev = obv[-10:-5]
                obv_trend = (sum(obv_recent) - sum(obv_prev)) / abs(sum(obv_prev)) if sum(obv_prev) != 0 else 0
            else:
                obv_trend = 0
            
            # Volume Confirmation Score
            volume_factors = [
                min(1.0, volume_ratio / 1.5),  # Volume vs Average
                min(1.0, abs(volume_trend)),   # Volume Trend
                min(1.0, abs(obv_trend) * 2)   # OBV Trend
            ]
            
            volume_confirmation = sum(volume_factors) / len(volume_factors)
            
            return {
                'volume_confirmation': volume_confirmation,
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'obv_trend': obv_trend,
                'avg_volume': avg_volume
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Volume Analysis: {e}")
            return {'volume_confirmation': 0.5}
    
    async def _analyze_support_resistance(self, candles: List, symbol: str) -> Dict[str, Any]:
        """📊 Support/Resistance Analysis"""
        try:
            highs = [float(c.high) for c in candles]
            lows = [float(c.low) for c in candles]
            closes = [float(c.close) for c in candles]
            
            current_price = closes[-1]
            
            # Pivot Points (letzte 50 Kerzen)
            window = min(50, len(candles))
            recent_highs = highs[-window:]
            recent_lows = lows[-window:]
            
            # Resistance Levels (lokale Hochs)
            resistance_levels = []
            for i in range(5, len(recent_highs) - 5):
                if (recent_highs[i] == max(recent_highs[i-5:i+6]) and 
                    recent_highs[i] > current_price * 1.005):  # Mindestens 0.5% über aktuellem Preis
                    resistance_levels.append(recent_highs[i])
            
            # Support Levels (lokale Tiefs)
            support_levels = []
            for i in range(5, len(recent_lows) - 5):
                if (recent_lows[i] == min(recent_lows[i-5:i+6]) and 
                    recent_lows[i] < current_price * 0.995):  # Mindestens 0.5% unter aktuellem Preis
                    support_levels.append(recent_lows[i])
            
            # Nächste Key Levels
            next_resistance = min(resistance_levels) if resistance_levels else current_price * 1.05
            next_support = max(support_levels) if support_levels else current_price * 0.95
            
            # Distance to Key Levels
            resistance_distance = (next_resistance - current_price) / current_price
            support_distance = (current_price - next_support) / current_price
            
            return {
                'resistance_levels': sorted(resistance_levels)[:3],  # Top 3
                'support_levels': sorted(support_levels, reverse=True)[:3],  # Top 3
                'next_resistance': next_resistance,
                'next_support': next_support,
                'resistance_distance': resistance_distance,
                'support_distance': support_distance,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Support/Resistance Analysis: {e}")
            return {
                'resistance_levels': [],
                'support_levels': [],
                'resistance_distance': 0.05,
                'support_distance': 0.05
            }
    
    def _calculate_swing_score(self, technical_analysis: Dict, trend_confirmation: Dict, 
                             volume_analysis: Dict, sr_analysis: Dict, timeframe: str) -> Dict[str, Any]:
        """🎯 Swing Trading Score berechnen"""
        try:
            scores = []
            sides = []
            
            # Technical Analysis Score
            if technical_analysis and 'primary_signal' in technical_analysis:
                primary_signal = technical_analysis['primary_signal']
                scores.append(primary_signal['strength'] * 0.35)
                
                if 'side' in primary_signal:
                    sides.append(primary_signal['side'])
            
            # Trend Confirmation Score
            trend_conf = trend_confirmation.get('trend_confirmation', 0)
            if trend_conf > 0.6:
                scores.append(trend_conf * 0.30)
                
                if trend_confirmation.get('trend_direction') == 'bullish':
                    sides.append(OrderSide.BUY)
                elif trend_confirmation.get('trend_direction') == 'bearish':
                    sides.append(OrderSide.SELL)
            
            # Volume Confirmation Score
            volume_conf = volume_analysis.get('volume_confirmation', 0)
            if volume_conf >= self.min_volume_confirmation / 2:  # 60% des Minimums
                scores.append(volume_conf * 0.25)
            
            # Support/Resistance Score
            resistance_dist = sr_analysis.get('resistance_distance', 0.05)
            support_dist = sr_analysis.get('support_distance', 0.05)
            
            # Bessere Position wenn weit von S/R entfernt
            sr_score = min(resistance_dist, support_dist) * 10  # Normalisierung
            if sr_score > 0.3:
                scores.append(sr_score * 0.10)
            
            if not scores:
                return {'score': 0, 'confidence': 0, 'side': OrderSide.BUY}
            
            # Base Score
            base_score = sum(scores)
            
            # Timeframe Bonus (4h bevorzugt für Swing)
            tf_bonus = {'1h': 0.0, '2h': 0.05, '4h': 0.10}.get(timeframe, 0.0)
            final_score = min(1.0, base_score + tf_bonus)
            
            # Dominant Side bestimmen
            if sides:
                buy_votes = sum(1 for side in sides if side == OrderSide.BUY)
                sell_votes = sum(1 for side in sides if side == OrderSide.SELL)
                
                if buy_votes > sell_votes:
                    dominant_side = OrderSide.BUY
                elif sell_votes > buy_votes:
                    dominant_side = OrderSide.SELL
                else:
                    # Bei Gleichstand nehme Trend Direction
                    if trend_confirmation.get('trend_direction') == 'bullish':
                        dominant_side = OrderSide.BUY
                    elif trend_confirmation.get('trend_direction') == 'bearish':
                        dominant_side = OrderSide.SELL
                    else:
                        dominant_side = OrderSide.BUY
            else:
                dominant_side = OrderSide.BUY
            
            # Confidence basierend auf Signal-Übereinstimmung
            signal_agreement = max(buy_votes, sell_votes) / len(sides) if sides else 0.5
            confidence = final_score * (0.7 + signal_agreement * 0.3)
            
            return {
                'score': final_score,
                'confidence': confidence,
                'side': dominant_side,
                'signal_count': len(scores),
                'agreement': signal_agreement
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Swing Score Berechnung: {e}")
            return {'score': 0, 'confidence': 0, 'side': OrderSide.BUY}
    
    def _calculate_swing_levels(self, current_price: float, side: OrderSide, market_analysis, 
                              sr_analysis: Dict, timeframe: str) -> Tuple[float, float, float]:
        """💰 Swing Trading Levels berechnen"""
        try:
            # Timeframe-spezifische Faktoren
            tf_factors = {
                '1h': {'sl_factor': 0.03, 'tp_factor': 0.075},   # 3% SL, 7.5% TP (1:2.5)
                '2h': {'sl_factor': 0.04, 'tp_factor': 0.10},    # 4% SL, 10% TP (1:2.5)
                '4h': {'sl_factor': 0.05, 'tp_factor': 0.125}    # 5% SL, 12.5% TP (1:2.5)
            }
            
            factors = tf_factors.get(timeframe, tf_factors['2h'])
            
            # Volatilität-Anpassung
            volatility = market_analysis.volatility_score if market_analysis else 0.03
            volatility_multiplier = 1 + (volatility - 0.03) * 2
            volatility_multiplier = max(0.7, min(1.5, volatility_multiplier))
            
            sl_distance = factors['sl_factor'] * volatility_multiplier
            tp_distance = factors['tp_factor'] * volatility_multiplier
            
            # Support/Resistance Adjustment
            if side == OrderSide.BUY:
                # Stop Loss knapp unter nächstem Support
                next_support = sr_analysis.get('next_support', current_price * (1 - sl_distance))
                calculated_sl = current_price * (1 - sl_distance)
                
                # Wähle den näheren Stop Loss (aber nicht zu nah)
                if next_support < current_price and (current_price - next_support) / current_price <= 0.08:
                    stop_loss = next_support * 0.995  # 0.5% Puffer unter Support
                else:
                    stop_loss = calculated_sl
                
                # Take Profit knapp unter nächster Resistance
                next_resistance = sr_analysis.get('next_resistance', current_price * (1 + tp_distance))
                calculated_tp = current_price * (1 + tp_distance)
                
                if next_resistance > current_price and (next_resistance - current_price) / current_price >= 0.04:
                    take_profit = min(calculated_tp, next_resistance * 0.995)  # 0.5% vor Resistance
                else:
                    take_profit = calculated_tp
                
                entry_price = current_price * 1.001  # 0.1% über Markt
                
            else:  # SELL
                # Stop Loss knapp über nächster Resistance
                next_resistance = sr_analysis.get('next_resistance', current_price * (1 + sl_distance))
                calculated_sl = current_price * (1 + sl_distance)
                
                if next_resistance > current_price and (next_resistance - current_price) / current_price <= 0.08:
                    stop_loss = next_resistance * 1.005  # 0.5% Puffer über Resistance
                else:
                    stop_loss = calculated_sl
                
                # Take Profit knapp über nächstem Support
                next_support = sr_analysis.get('next_support', current_price * (1 - tp_distance))
                calculated_tp = current_price * (1 - tp_distance)
                
                if next_support < current_price and (current_price - next_support) / current_price >= 0.04:
                    take_profit = max(calculated_tp, next_support * 1.005)  # 0.5% nach Support
                else:
                    take_profit = calculated_tp
                
                entry_price = current_price * 0.999  # 0.1% unter Markt
            
            return entry_price, stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Swing Levels Berechnung: {e}")
            # Fallback
            if side == OrderSide.BUY:
                return current_price * 1.001, current_price * 0.96, current_price * 1.08
            else:
                return current_price * 0.999, current_price * 1.04, current_price * 0.92
    
    async def _combine_with_ai_signal(self, swing_signal: Dict, ai_signal) -> Dict[str, Any]:
        """🤖 Swing-Signal mit AI-Signal kombinieren"""
        try:
            # AI-Signal Gewichtung
            ai_weight = 0.3  # 30% AI, 70% Swing
            swing_weight = 0.7
            
            # Confidence kombinieren
            combined_confidence = (swing_signal['confidence'] * swing_weight + 
                                 ai_signal.confidence * ai_weight)
            
            # Side Agreement Check
            if swing_signal['side'] == OrderSide.BUY and ai_signal.signal_type.value == 'buy':
                # Agreement Bonus
                combined_confidence *= 1.1
            elif swing_signal['side'] == OrderSide.SELL and ai_signal.signal_type.value == 'sell':
                # Agreement Bonus
                combined_confidence *= 1.1
            else:
                # Disagreement Penalty
                combined_confidence *= 0.9
            
            # AI-Enhanced Levels
            if ai_signal.stop_loss and ai_signal.take_profit:
                # Bessere SL/TP aus AI verwenden wenn verfügbar
                ai_sl = float(ai_signal.stop_loss)
                ai_tp = float(ai_signal.take_profit)
                
                current_sl = swing_signal['stop_loss']
                current_tp = swing_signal['take_profit']
                
                # Konservativere SL wählen
                if swing_signal['side'] == OrderSide.BUY:
                    better_sl = max(current_sl, ai_sl)  # Höhere SL für BUY
                    better_tp = min(current_tp, ai_tp)  # Konservativere TP
                else:
                    better_sl = min(current_sl, ai_sl)  # Niedrigere SL für SELL
                    better_tp = max(current_tp, ai_tp)  # Konservativere TP
                
                swing_signal['stop_loss'] = better_sl
                swing_signal['take_profit'] = better_tp
            
            swing_signal['confidence'] = min(1.0, combined_confidence)
            swing_signal['ai_enhanced'] = True
            swing_signal['ai_confidence'] = ai_signal.confidence
            
            return swing_signal
            
        except Exception as e:
            logger.error(f"❌ Fehler bei AI-Signal Kombination: {e}")
            return swing_signal
    
    # ==================== UTILITY METHODS ====================
    
    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """📈 EMA berechnen"""
        try:
            if len(prices) < period:
                return []
            
            multiplier = 2 / (period + 1)
            ema_values = [sum(prices[:period]) / period]
            
            for price in prices[period:]:
                ema = (price * multiplier) + (ema_values[-1] * (1 - multiplier))
                ema_values.append(ema)
            
            return ema_values
        except Exception:
            return []
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """📊 RSI berechnen"""
        try:
            if len(prices) < period + 1:
                return []
            
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]
            
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period
            
            rsi_values = []
            
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                rsi_values.append(rsi)
            
            return rsi_values
        except Exception:
            return []
    
    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """📊 MACD berechnen"""
        try:
            if len(prices) < slow:
                return [], [], []
            
            ema_fast = self._calculate_ema(prices, fast)
            ema_slow = self._calculate_ema(prices, slow)
            
            # MACD Line
            macd_line = []
            min_len = min(len(ema_fast), len(ema_slow))
            for i in range(min_len):
                macd_line.append(ema_fast[i] - ema_slow[i])
            
            # Signal Line
            signal_line = self._calculate_ema(macd_line, signal)
            
            # Histogram
            histogram = []
            min_len = min(len(macd_line), len(signal_line))
            for i in range(min_len):
                histogram.append(macd_line[i] - signal_line[i])
            
            return macd_line, signal_line, histogram
        except Exception:
            return [], [], []
    
    def _calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
        """📊 ADX berechnen (vereinfacht)"""
        try:
            if len(closes) < period + 1:
                return []
            
            # Vereinfachte ADX-Berechnung
            adx_values = []
            
            for i in range(period, len(closes)):
                price_range = highs[i] - lows[i]
                price_changes = [abs(closes[j] - closes[j-1]) for j in range(i-period+1, i+1)]
                avg_change = sum(price_changes) / period
                
                # Vereinfachter ADX-Wert
                adx = min(100, (avg_change / price_range) * 100) if price_range > 0 else 0
                adx_values.append(adx)
            
            return adx_values
        except Exception:
            return []
    
    async def _check_weekly_reset(self):
        """📅 Wöchentliche Statistiken zurücksetzen"""
        try:
            now = datetime.utcnow()
            # Montag = 0, Sonntag = 6
            if now.weekday() == 0 and now.date() > self.weekly_reset_time.date():
                self.trades_this_week = 0
                self.weekly_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
                logger.info("📅 Wöchentliche Swing-Statistiken zurückgesetzt")
        except Exception as e:
            logger.error(f"❌ Fehler bei Weekly Reset: {e}")
    
    # ==================== PUBLIC METHODS ====================
    
    async def generate_trade_signal(self, symbol: str) -> Optional[TradeSignal]:
        """🎯 Trading Signal für das Hauptsystem generieren"""
        try:
            if not self.enabled:
                return None
            
            # Swing Signal analysieren
            swing_signal = await self.analyze_swing_opportunity(symbol)
            if not swing_signal:
                return None
            
            # Position Size berechnen
            portfolio = await self.bitget.get_portfolio()
            if not portfolio:
                return None
            
            risk_amount = float(portfolio.total_balance) * 0.02  # 2% Risiko für Swing
            entry_price = swing_signal['entry_price']
            stop_loss = swing_signal['stop_loss']
            
            position_size = calculate_position_size(
                float(portfolio.total_balance),
                2.0,  # 2% Risiko
                entry_price,
                stop_loss
            )
            
            # TradeSignal erstellen
            trade_signal = TradeSignal(
                id=generate_signal_id(symbol, "SWING"),
                symbol=symbol,
                side=swing_signal['side'],
                entry_price=Decimal(str(swing_signal['entry_price'])),
                stop_loss=Decimal(str(swing_signal['stop_loss'])),
                take_profit=Decimal(str(swing_signal['take_profit'])),
                quantity=Decimal(str(position_size)),
                confidence=swing_signal['confidence'],
                strategy="swing_genius",
                timeframe=swing_signal['timeframe'],
                timestamp=datetime.utcnow(),
                metadata={
                    'swing_score': swing_signal['swing_score'],
                    'analysis': swing_signal['analysis'],
                    'ai_enhanced': swing_signal.get('ai_enhanced', False),
                    'risk_reward_ratio': abs(swing_signal['take_profit'] - swing_signal['entry_price']) / abs(swing_signal['entry_price'] - swing_signal['stop_loss'])
                }
            )
            
            self.trades_this_week += 1
            return trade_signal
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Swing Trade Signal Generation: {e}")
            return None
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """📊 Strategie-Statistiken abrufen"""
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        
        return {
            'strategy_name': 'swing_genius',
            'enabled': self.enabled,
            'signals_generated': self.signals_generated,
            'trades_this_week': self.trades_this_week,
            'max_trades_per_week': self.max_trades_per_week,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'target_win_rate': self.target_win_rate,
            'timeframes': self.timeframes,
            'cache_size': len(self.swing_analysis_cache)
        }
    
    async def update_performance(self, signal_id: str, was_successful: bool):
        """📊 Performance Update"""
        try:
            if was_successful:
                self.win_count += 1
            else:
                self.loss_count += 1
            
            logger.info(f"📊 Swing Performance updated: {signal_id} -> {'Win' if was_successful else 'Loss'}")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Performance Update: {e}")
    
    async def shutdown(self):
        """🛑 Swing Genius herunterfahren"""
        try:
            self.swing_analysis_cache.clear()
            logger.info("✅ Swing Genius heruntergefahren")
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren: {e}")
