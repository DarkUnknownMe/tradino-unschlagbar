"""
🔥 TRADINO UNSCHLAGBAR - Scalping Master Strategy
Ultra-schnelle Scalping-Strategie für 1-5min Timeframes
Target: 65-70% Win Rate, 1:1.2 Risk/Reward, 20-50 Trades/Tag

Author: AI Trading Systems
"""

import asyncio
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from models.trade_models import TradeSignal, OrderSide
from models.market_models import Candle, MarketData, MarketRegime
from brain.master_ai import MasterAI
from brain.market_intelligence import MarketIntelligence
from connectors.bitget_pro import BitgetProConnector
from utils.logger_pro import setup_logger, log_trade
from utils.config_manager import ConfigManager
from utils.math_utils import calculate_position_size
from utils.helpers import generate_signal_id

logger = setup_logger("ScalpingMaster")


@dataclass
class ScalpingSignal:
    """Scalping-spezifisches Signal"""
    symbol: str
    side: OrderSide
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    confidence: float
    scalping_score: float
    market_conditions: Dict[str, Any]
    timestamp: datetime


class ScalpingMaster:
    """🔥 Ultra-schnelle Scalping-Strategie"""
    
    def __init__(self, config: ConfigManager, bitget_connector: BitgetProConnector, 
                 master_ai: MasterAI, market_intelligence: MarketIntelligence):
        self.config = config
        self.bitget = bitget_connector
        self.master_ai = master_ai
        self.market_intelligence = market_intelligence
        
        # Strategy Configuration
        self.strategy_config = config.get('strategies.scalping_master', {})
        self.enabled = self.strategy_config.get('enabled', True)
        self.timeframes = self.strategy_config.get('timeframes', ['1m', '3m', '5m'])
        self.target_win_rate = self.strategy_config.get('target_win_rate', 0.67)
        self.risk_reward_ratio = self.strategy_config.get('risk_reward', 1.2)
        self.max_trades_per_day = self.strategy_config.get('max_trades_per_day', 50)
        
        # Scalping Parameters
        self.min_volatility = 0.02  # 2% Minimum Volatilität
        self.max_volatility = 0.08  # 8% Maximum Volatilität
        self.min_volume_ratio = 1.5  # 150% des Average Volume
        self.max_spread_percent = 0.05  # 0.05% Max Spread
        self.min_price_movement = 0.003  # 0.3% Minimum erwartete Bewegung
        
        # Technical Indicators Thresholds
        self.rsi_oversold = 25
        self.rsi_overbought = 75
        self.rsi_neutral_min = 40
        self.rsi_neutral_max = 60
        
        # Performance Tracking
        self.signals_generated = 0
        self.trades_today = 0
        self.daily_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        self.win_count = 0
        self.loss_count = 0
        
        # Market Condition Cache
        self.market_conditions_cache: Dict[str, Dict] = {}
        
    async def initialize(self) -> bool:
        """🔥 Scalping Master initialisieren"""
        try:
            logger.info("🔥 Scalping Master wird initialisiert...")
            
            if not self.enabled:
                logger.info("⚠️ Scalping Master ist deaktiviert")
                return True
            
            logger.success(f"✅ Scalping Master initialisiert (Target Win Rate: {self.target_win_rate:.1%})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Scalping Master Initialisierung fehlgeschlagen: {e}")
            return False
    
    # ==================== MAIN STRATEGY LOGIC ====================
    
    async def analyze_scalping_opportunity(self, symbol: str) -> Optional[ScalpingSignal]:
        """🎯 Scalping-Gelegenheit analysieren"""
        try:
            # Daily Trade Limit Check
            await self._check_daily_reset()
            if self.trades_today >= self.max_trades_per_day:
                logger.info(f"📊 Tägliches Trade-Limit erreicht: {self.trades_today}/{self.max_trades_per_day}")
                return None
            
            logger.info(f"🔥 Scalping-Analyse wird durchgeführt: {symbol}")
            
            # Market Conditions prüfen
            market_conditions = await self._analyze_market_conditions(symbol)
            if not market_conditions['suitable_for_scalping']:
                logger.info(f"📊 Marktbedingungen nicht geeignet für Scalping: {symbol}")
                return None
            
            # Multi-Timeframe Analysis
            signals = []
            for timeframe in self.timeframes:
                signal = await self._analyze_timeframe_scalping(symbol, timeframe, market_conditions)
                if signal:
                    signals.append(signal)
            
            if not signals:
                return None
            
            # Beste Signal auswählen
            best_signal = max(signals, key=lambda x: x.scalping_score)
            
            # Minimum Confidence Check
            min_confidence = 0.65  # 65% Minimum für Scalping
            if best_signal.confidence < min_confidence:
                logger.info(f"🔍 Signal Confidence zu niedrig: {best_signal.confidence:.2%}")
                return None
            
            # Final Signal Validation
            if await self._validate_scalping_signal(best_signal, market_conditions):
                self.signals_generated += 1
                log_trade(f"🔥 Scalping Signal generiert: {symbol} {best_signal.side.value} (Score: {best_signal.scalping_score:.2f})")
                return best_signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Scalping-Analyse für {symbol}: {e}")
            return None
    
    async def _analyze_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """📊 Marktbedingungen für Scalping analysieren"""
        try:
            # Cache Check
            cache_key = f"{symbol}_{datetime.utcnow().minute // 2}"  # 2-Min Cache
            if cache_key in self.market_conditions_cache:
                return self.market_conditions_cache[cache_key]
            
            # Market Data abrufen
            market_data = await self.bitget.get_market_data(symbol)
            candles = await self.bitget.get_candles(symbol, '1m', limit=60)
            
            if not market_data or len(candles) < 30:
                return {'suitable_for_scalping': False}
            
            # Volatilität berechnen
            prices = [float(c.close) for c in candles[-20:]]
            volatility = np.std(prices) / np.mean(prices) if prices else 0
            
            # Volume Analysis
            current_volume = float(candles[-1].volume)
            avg_volume = sum(float(c.volume) for c in candles[-20:]) / 20
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Spread Analysis (Mock - in Realität über Order Book)
            spread = (float(market_data.ask) - float(market_data.bid)) / float(market_data.price)
            
            # Trend Stability (für Scalping brauchen wir kurze, schnelle Moves)
            price_changes = [(float(candles[i].close) - float(candles[i-1].close)) / float(candles[i-1].close) 
                           for i in range(1, min(10, len(candles)))]
            trend_consistency = 1 - np.std(price_changes) if price_changes else 0
            
            # Market Intelligence
            market_analysis = self.market_intelligence.get_cached_analysis(symbol)
            market_regime = market_analysis.regime if market_analysis else MarketRegime.RANGE_BOUND
            
            # Scalping Suitability Check
            suitable_conditions = (
                self.min_volatility <= volatility <= self.max_volatility and
                volume_ratio >= self.min_volume_ratio and
                spread <= self.max_spread_percent and
                market_regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.RANGE_BOUND, MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]
            )
            
            conditions = {
                'suitable_for_scalping': suitable_conditions,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'spread': spread,
                'trend_consistency': trend_consistency,
                'market_regime': market_regime.value,
                'current_price': float(market_data.price),
                'timestamp': datetime.utcnow()
            }
            
            # Cache aktualisieren
            self.market_conditions_cache[cache_key] = conditions
            
            return conditions
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Market Conditions Analysis: {e}")
            return {'suitable_for_scalping': False}
    
    async def _analyze_timeframe_scalping(self, symbol: str, timeframe: str, market_conditions: Dict) -> Optional[ScalpingSignal]:
        """📈 Timeframe-spezifische Scalping-Analyse"""
        try:
            # Candles für Timeframe abrufen
            candles = await self.bitget.get_candles(symbol, timeframe, limit=50)
            if len(candles) < 20:
                return None
            
            # Technical Analysis
            technical_signals = await self._analyze_technical_scalping(candles, timeframe)
            if not technical_signals:
                return None
            
            # Price Action Analysis
            price_action_signals = await self._analyze_price_action_scalping(candles)
            
            # Micro-Trend Analysis
            micro_trend = await self._analyze_micro_trend(candles)
            
            # Signal kombinieren
            combined_score = self._calculate_scalping_score(
                technical_signals, price_action_signals, micro_trend, market_conditions
            )
            
            if combined_score['score'] < 0.6:  # Minimum Score für Scalping
                return None
            
            # Entry, SL, TP berechnen
            current_price = float(candles[-1].close)
            signal_side = combined_score['side']
            
            entry_price, stop_loss, take_profit = self._calculate_scalping_levels(
                current_price, signal_side, market_conditions['volatility'], timeframe
            )
            
            return ScalpingSignal(
                symbol=symbol,
                side=signal_side,
                entry_price=Decimal(str(entry_price)),
                stop_loss=Decimal(str(stop_loss)),
                take_profit=Decimal(str(take_profit)),
                confidence=combined_score['confidence'],
                scalping_score=combined_score['score'],
                market_conditions=market_conditions,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Timeframe Scalping Analysis: {e}")
            return None
    
    async def _analyze_technical_scalping(self, candles: List[Candle], timeframe: str) -> Optional[Dict[str, Any]]:
        """📊 Technical Analysis für Scalping"""
        try:
            prices = [float(c.close) for c in candles]
            highs = [float(c.high) for c in candles]
            lows = [float(c.low) for c in candles]
            volumes = [float(c.volume) for c in candles]
            
            # RSI Calculation (14-period)
            rsi = self._calculate_rsi(prices, 14)
            current_rsi = rsi[-1] if rsi else 50
            
            # EMA Crossover (5 vs 13)
            ema_5 = self._calculate_ema(prices, 5)
            ema_13 = self._calculate_ema(prices, 13)
            
            ema_crossover = None
            if len(ema_5) >= 2 and len(ema_13) >= 2:
                if ema_5[-1] > ema_13[-1] and ema_5[-2] <= ema_13[-2]:
                    ema_crossover = 'bullish'
                elif ema_5[-1] < ema_13[-1] and ema_5[-2] >= ema_13[-2]:
                    ema_crossover = 'bearish'
            
            # Bollinger Bands Squeeze
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices, 20, 2)
            if bb_upper and bb_lower:
                bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
                current_price = prices[-1]
                bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            else:
                bb_width = 0
                bb_position = 0.5
            
            # Volume Spike Detection
            avg_volume = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else volumes[-1]
            volume_spike = volumes[-1] / avg_volume if avg_volume > 0 else 1
            
            # Signal Generation
            signals = []
            
            # RSI Signals
            if current_rsi <= self.rsi_oversold and volume_spike >= 1.3:
                signals.append({'type': 'rsi_oversold', 'side': OrderSide.BUY, 'strength': 0.8})
            elif current_rsi >= self.rsi_overbought and volume_spike >= 1.3:
                signals.append({'type': 'rsi_overbought', 'side': OrderSide.SELL, 'strength': 0.8})
            
            # EMA Crossover Signals
            if ema_crossover == 'bullish' and current_rsi < 70:
                signals.append({'type': 'ema_bull_cross', 'side': OrderSide.BUY, 'strength': 0.7})
            elif ema_crossover == 'bearish' and current_rsi > 30:
                signals.append({'type': 'ema_bear_cross', 'side': OrderSide.SELL, 'strength': 0.7})
            
            # Bollinger Bands Signals
            if bb_position <= 0.1 and bb_width > 0.02:  # Near lower band + normal width
                signals.append({'type': 'bb_lower_bounce', 'side': OrderSide.BUY, 'strength': 0.6})
            elif bb_position >= 0.9 and bb_width > 0.02:  # Near upper band + normal width
                signals.append({'type': 'bb_upper_bounce', 'side': OrderSide.SELL, 'strength': 0.6})
            
            if not signals:
                return None
            
            # Stärkstes Signal auswählen
            best_signal = max(signals, key=lambda x: x['strength'])
            
            return {
                'signal_type': best_signal['type'],
                'side': best_signal['side'],
                'strength': best_signal['strength'],
                'rsi': current_rsi,
                'ema_crossover': ema_crossover,
                'bb_position': bb_position,
                'volume_spike': volume_spike,
                'supporting_signals': len(signals)
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Technical Scalping Analysis: {e}")
            return None
    
    async def _analyze_price_action_scalping(self, candles: List[Candle]) -> Dict[str, Any]:
        """📈 Price Action Analysis für Scalping"""
        try:
            if len(candles) < 5:
                return {'score': 0, 'signals': []}
            
            signals = []
            
            # Letzte 5 Kerzen analysieren
            recent_candles = candles[-5:]
            
            # Hammer/Doji Detection
            for i, candle in enumerate(recent_candles[-3:], -3):  # Letzte 3 Kerzen
                open_price = float(candle.open)
                close_price = float(candle.close)
                high_price = float(candle.high)
                low_price = float(candle.low)
                
                body_size = abs(close_price - open_price)
                total_range = high_price - low_price
                
                if total_range > 0:
                    body_ratio = body_size / total_range
                    
                    # Doji Pattern (kleine Body)
                    if body_ratio <= 0.1:
                        signals.append({'type': 'doji', 'strength': 0.5, 'index': i})
                    
                    # Hammer Pattern
                    lower_shadow = min(open_price, close_price) - low_price
                    if lower_shadow >= body_size * 2 and body_ratio <= 0.3:
                        signals.append({'type': 'hammer', 'side': OrderSide.BUY, 'strength': 0.7, 'index': i})
                    
                    # Shooting Star Pattern
                    upper_shadow = high_price - max(open_price, close_price)
                    if upper_shadow >= body_size * 2 and body_ratio <= 0.3:
                        signals.append({'type': 'shooting_star', 'side': OrderSide.SELL, 'strength': 0.7, 'index': i})
            
            # Engulfing Patterns
            if len(recent_candles) >= 2:
                prev_candle = recent_candles[-2]
                curr_candle = recent_candles[-1]
                
                prev_body = abs(float(prev_candle.close) - float(prev_candle.open))
                curr_body = abs(float(curr_candle.close) - float(curr_candle.open))
                
                # Bullish Engulfing
                if (float(prev_candle.close) < float(prev_candle.open) and  # Prev bearish
                    float(curr_candle.close) > float(curr_candle.open) and  # Curr bullish
                    float(curr_candle.open) < float(prev_candle.close) and  # Opens below prev close
                    float(curr_candle.close) > float(prev_candle.open) and  # Closes above prev open
                    curr_body > prev_body * 1.1):  # Bigger body
                    signals.append({'type': 'bullish_engulfing', 'side': OrderSide.BUY, 'strength': 0.8})
                
                # Bearish Engulfing
                elif (float(prev_candle.close) > float(prev_candle.open) and  # Prev bullish
                      float(curr_candle.close) < float(curr_candle.open) and  # Curr bearish
                      float(curr_candle.open) > float(prev_candle.close) and  # Opens above prev close
                      float(curr_candle.close) < float(prev_candle.open) and  # Closes below prev open
                      curr_body > prev_body * 1.1):  # Bigger body
                    signals.append({'type': 'bearish_engulfing', 'side': OrderSide.SELL, 'strength': 0.8})
            
            # Support/Resistance Breaks
            prices = [float(c.close) for c in candles[-20:]]
            current_price = prices[-1]
            
            # Recent High/Low
            recent_high = max(prices[-10:]) if len(prices) >= 10 else current_price
            recent_low = min(prices[-10:]) if len(prices) >= 10 else current_price
            
            # Breakout Detection
            if current_price > recent_high * 1.002:  # 0.2% über recent high
                signals.append({'type': 'resistance_break', 'side': OrderSide.BUY, 'strength': 0.6})
            elif current_price < recent_low * 0.998:  # 0.2% unter recent low
                signals.append({'type': 'support_break', 'side': OrderSide.SELL, 'strength': 0.6})
            
            # Overall Score basierend auf Signal-Qualität und -Anzahl
            if signals:
                signal_scores = [s.get('strength', 0.5) for s in signals]
                avg_strength = sum(signal_scores) / len(signal_scores)
                signal_count_bonus = min(0.2, len(signals) * 0.05)  # Bonus für multiple Signale
                overall_score = min(1.0, avg_strength + signal_count_bonus)
            else:
                overall_score = 0
            
            return {
                'score': overall_score,
                'signals': signals,
                'signal_count': len(signals)
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Price Action Analysis: {e}")
            return {'score': 0, 'signals': []}
    
    async def _analyze_micro_trend(self, candles: List[Candle]) -> Dict[str, Any]:
        """📊 Micro-Trend Analysis für ultra-kurze Bewegungen"""
        try:
            if len(candles) < 10:
                return {'trend': 'neutral', 'strength': 0}
            
            # Letzte 10 Kerzen für Micro-Trend
            recent_closes = [float(c.close) for c in candles[-10:]]
            
            # Linear Regression für Trend
            x = np.arange(len(recent_closes))
            z = np.polyfit(x, recent_closes, 1)
            slope = z[0]
            
            # Trend Stärke basierend auf R²
            p = np.poly1d(z)
            predicted = p(x)
            ss_res = np.sum((recent_closes - predicted) ** 2)
            ss_tot = np.sum((recent_closes - np.mean(recent_closes)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Trend bestimmen
            avg_price = np.mean(recent_closes)
            slope_percent = (slope / avg_price) * 100 if avg_price > 0 else 0
            
            if slope_percent > 0.1 and r_squared > 0.7:
                trend = 'bullish'
                strength = min(1.0, r_squared * abs(slope_percent) * 2)
            elif slope_percent < -0.1 and r_squared > 0.7:
                trend = 'bearish'
                strength = min(1.0, r_squared * abs(slope_percent) * 2)
            else:
                trend = 'neutral'
                strength = 0
            
            # Momentum Check (letzte 3 vs vorherige 3 Kerzen)
            recent_3 = recent_closes[-3:]
            prev_3 = recent_closes[-6:-3]
            
            momentum = (np.mean(recent_3) - np.mean(prev_3)) / np.mean(prev_3) if prev_3 else 0
            
            return {
                'trend': trend,
                'strength': strength,
                'slope_percent': slope_percent,
                'r_squared': r_squared,
                'momentum': momentum
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Micro-Trend Analysis: {e}")
            return {'trend': 'neutral', 'strength': 0}
    
    def _calculate_scalping_score(self, technical_signals: Dict, price_action: Dict, 
                                micro_trend: Dict, market_conditions: Dict) -> Dict[str, Any]:
        """🎯 Gesamten Scalping Score berechnen"""
        try:
            scores = []
            sides = []
            
            # Technical Signals Score
            if technical_signals:
                scores.append(technical_signals['strength'] * 0.4)
                sides.append(technical_signals['side'])
            
            # Price Action Score
            if price_action['score'] > 0:
                scores.append(price_action['score'] * 0.35)
                
                # Side aus Price Action Signals extrahieren
                pa_signals = [s for s in price_action['signals'] if 'side' in s]
                if pa_signals:
                    strongest_pa = max(pa_signals, key=lambda x: x['strength'])
                    sides.append(strongest_pa['side'])
            
            # Micro-Trend Score
            if micro_trend['strength'] > 0:
                scores.append(micro_trend['strength'] * 0.25)
                
                if micro_trend['trend'] == 'bullish':
                    sides.append(OrderSide.BUY)
                elif micro_trend['trend'] == 'bearish':
                    sides.append(OrderSide.SELL)
            
            if not scores:
                return {'score': 0, 'confidence': 0, 'side': OrderSide.BUY}
            
            # Overall Score
            base_score = sum(scores)
            
            # Market Conditions Bonus/Penalty
            volatility_bonus = 0
            if 0.02 <= market_conditions['volatility'] <= 0.05:
                volatility_bonus = 0.1  # Ideale Volatilität für Scalping
            elif market_conditions['volatility'] > 0.08:
                volatility_bonus = -0.2  # Zu hohe Volatilität
            
            volume_bonus = min(0.1, (market_conditions['volume_ratio'] - 1.5) * 0.1)
            
            final_score = min(1.0, base_score + volatility_bonus + volume_bonus)
            
            # Dominant Side bestimmen
            if sides:
                buy_votes = sum(1 for side in sides if side == OrderSide.BUY)
                sell_votes = sum(1 for side in sides if side == OrderSide.SELL)
                
                if buy_votes > sell_votes:
                    dominant_side = OrderSide.BUY
                elif sell_votes > buy_votes:
                    dominant_side = OrderSide.SELL
                else:
                    dominant_side = OrderSide.BUY  # Default
            else:
                dominant_side = OrderSide.BUY
            
            # Confidence basierend auf Signal-Übereinstimmung
            signal_agreement = max(buy_votes, sell_votes) / len(sides) if sides else 0.5
            confidence = final_score * signal_agreement
            
            return {
                'score': final_score,
                'confidence': confidence,
                'side': dominant_side,
                'signal_count': len(scores),
                'agreement': signal_agreement
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Scalping Score Berechnung: {e}")
            return {'score': 0, 'confidence': 0, 'side': OrderSide.BUY}
    
    def _calculate_scalping_levels(self, current_price: float, side: OrderSide, 
                                 volatility: float, timeframe: str) -> Tuple[float, float, float]:
        """💰 Entry, Stop Loss und Take Profit für Scalping berechnen"""
        try:
            # Timeframe-spezifische Faktoren
            tf_factors = {
                '1m': {'sl_factor': 0.008, 'tp_factor': 0.010},  # 0.8% SL, 1.0% TP
                '3m': {'sl_factor': 0.012, 'tp_factor': 0.015},  # 1.2% SL, 1.5% TP
                '5m': {'sl_factor': 0.015, 'tp_factor': 0.020}   # 1.5% SL, 2.0% TP
            }
            
            factors = tf_factors.get(timeframe, tf_factors['3m'])
            
            # Volatilität-Anpassung
            volatility_multiplier = 1 + (volatility - 0.02) * 2  # Base 2% Volatilität
            volatility_multiplier = max(0.5, min(2.0, volatility_multiplier))
            
            sl_distance = factors['sl_factor'] * volatility_multiplier
            tp_distance = factors['tp_factor'] * volatility_multiplier
            
            # Entry Price (minimal adjusted für bessere Fills)
            if side == OrderSide.BUY:
                entry_price = current_price * 1.0002  # 0.02% über Markt
                stop_loss = entry_price * (1 - sl_distance)
                take_profit = entry_price * (1 + tp_distance)
            else:  # SELL
                entry_price = current_price * 0.9998  # 0.02% unter Markt
                stop_loss = entry_price * (1 + sl_distance)
                take_profit = entry_price * (1 - tp_distance)
            
            return entry_price, stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Scalping Levels Berechnung: {e}")
            return current_price, current_price * 0.99, current_price * 1.01
    
    # ==================== UTILITY METHODS ====================
    
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
            
        except Exception as e:
            logger.error(f"❌ Fehler bei RSI Berechnung: {e}")
            return []
    
    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """📈 EMA berechnen"""
        try:
            if len(prices) < period:
                return []
            
            multiplier = 2 / (period + 1)
            ema_values = [sum(prices[:period]) / period]  # Erste EMA ist SMA
            
            for price in prices[period:]:
                ema = (price * multiplier) + (ema_values[-1] * (1 - multiplier))
                ema_values.append(ema)
            
            return ema_values
            
        except Exception as e:
            logger.error(f"❌ Fehler bei EMA Berechnung: {e}")
            return []
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: int = 2) -> Tuple[List[float], List[float], List[float]]:
        """📊 Bollinger Bands berechnen"""
        try:
            if len(prices) < period:
                return [], [], []
            
            sma_values = []
            upper_bands = []
            lower_bands = []
            
            for i in range(period - 1, len(prices)):
                window = prices[i - period + 1:i + 1]
                sma = sum(window) / period
                std = np.std(window)
                
                sma_values.append(sma)
                upper_bands.append(sma + (std * std_dev))
                lower_bands.append(sma - (std * std_dev))
            
            return upper_bands, sma_values, lower_bands
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Bollinger Bands Berechnung: {e}")
            return [], [], []
    
    async def _validate_scalping_signal(self, signal: ScalpingSignal, market_conditions: Dict) -> bool:
        """✅ Finales Signal-Validation"""
        try:
            # Risk/Reward Check
            entry = float(signal.entry_price)
            sl = float(signal.stop_loss)
            tp = float(signal.take_profit)
            
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            risk_reward = reward / risk if risk > 0 else 0
            
            # Minimum Risk/Reward für Scalping
            if risk_reward < 1.0:  # Mindestens 1:1
                logger.info(f"📊 Risk/Reward zu niedrig: {risk_reward:.2f}")
                return False
            
            # Maximum Risk Check (2% des Entry Price)
            max_risk_percent = 0.02
            risk_percent = risk / entry
            if risk_percent > max_risk_percent:
                logger.info(f"📊 Risiko zu hoch: {risk_percent:.2%}")
                return False
            
            # Market Conditions Final Check
            if not market_conditions['suitable_for_scalping']:
                return False
            
            # Spread Check (nicht zu hoch für Scalping)
            if market_conditions['spread'] > self.max_spread_percent:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Signal Validation: {e}")
            return False
    
    async def _check_daily_reset(self):
        """📅 Tägliche Statistiken zurücksetzen"""
        try:
            now = datetime.utcnow()
            if now.date() > self.daily_reset_time.date():
                self.trades_today = 0
                self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
                logger.info("📅 Tägliche Scalping-Statistiken zurückgesetzt")
        except Exception as e:
            logger.error(f"❌ Fehler bei Daily Reset: {e}")
    
    # ==================== PUBLIC METHODS ====================
    
    async def generate_trade_signal(self, symbol: str) -> Optional[TradeSignal]:
        """🎯 Trading Signal für das Hauptsystem generieren"""
        try:
            if not self.enabled:
                return None
            
            # Scalping Signal analysieren
            scalping_signal = await self.analyze_scalping_opportunity(symbol)
            if not scalping_signal:
                return None
            
            # Position Size berechnen
            portfolio = await self.bitget.get_portfolio()
            if not portfolio:
                return None
            
            risk_amount = float(portfolio.total_balance) * 0.01  # 1% Risiko für Scalping
            entry_price = float(scalping_signal.entry_price)
            stop_loss = float(scalping_signal.stop_loss)
            
            position_size = calculate_position_size(
                float(portfolio.total_balance),
                1.0,  # 1% Risiko
                entry_price,
                stop_loss
            )
            
            # TradeSignal erstellen
            trade_signal = TradeSignal(
                id=generate_signal_id(symbol, "SCALPING"),
                symbol=symbol,
                side=scalping_signal.side,
                entry_price=scalping_signal.entry_price,
                stop_loss=scalping_signal.stop_loss,
                take_profit=scalping_signal.take_profit,
                quantity=Decimal(str(position_size)),
                confidence=scalping_signal.confidence,
                strategy="scalping_master",
                timeframe="1m",
                timestamp=datetime.utcnow(),
                metadata={
                    'scalping_score': scalping_signal.scalping_score,
                    'market_conditions': scalping_signal.market_conditions,
                    'risk_reward_ratio': abs(float(scalping_signal.take_profit) - float(scalping_signal.entry_price)) / abs(float(scalping_signal.entry_price) - float(scalping_signal.stop_loss))
                }
            )
            
            self.trades_today += 1
            return trade_signal
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Trade Signal Generation: {e}")
            return None
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """📊 Strategie-Statistiken abrufen"""
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        
        return {
            'strategy_name': 'scalping_master',
            'enabled': self.enabled,
            'signals_generated': self.signals_generated,
            'trades_today': self.trades_today,
            'max_trades_per_day': self.max_trades_per_day,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'target_win_rate': self.target_win_rate,
            'timeframes': self.timeframes,
            'cache_size': len(self.market_conditions_cache)
        }
    
    async def update_performance(self, signal_id: str, was_successful: bool):
        """📊 Performance Update"""
        try:
            if was_successful:
                self.win_count += 1
            else:
                self.loss_count += 1
            
            logger.info(f"📊 Scalping Performance updated: {signal_id} -> {'Win' if was_successful else 'Loss'}")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Performance Update: {e}")
    
    async def shutdown(self):
        """🛑 Scalping Master herunterfahren"""
        try:
            self.market_conditions_cache.clear()
            logger.info("✅ Scalping Master heruntergefahren")
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren: {e}")
