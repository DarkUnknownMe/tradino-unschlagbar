"""
🎯 TRADINO UNSCHLAGBAR - Trend Hunter Strategy
Langfristige Trend-Following-Strategie für 4h-1D Timeframes
Target: 50-55% Win Rate, 1:3.0 Risk/Reward, 10-20 Trades/Monat

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

logger = setup_logger("TrendHunter")


class TrendHunter:
    """🎯 Langfristige Trend-Following-Strategie"""
    
    def __init__(self, config: ConfigManager, bitget_connector: BitgetProConnector, 
                 master_ai: MasterAI, market_intelligence: MarketIntelligence):
        self.config = config
        self.bitget = bitget_connector
        self.master_ai = master_ai
        self.market_intelligence = market_intelligence
        
        # Strategy Configuration
        self.strategy_config = config.get('strategies.trend_hunter', {})
        self.enabled = self.strategy_config.get('enabled', True)
        self.timeframes = self.strategy_config.get('timeframes', ['4h', '6h', '1d'])
        self.target_win_rate = self.strategy_config.get('target_win_rate', 0.53)
        self.risk_reward_ratio = self.strategy_config.get('risk_reward', 3.0)
        self.max_trades_per_month = self.strategy_config.get('max_trades_per_month', 20)
        
        # Trend Parameters
        self.min_trend_strength = 0.7  # 70% Minimum Trend-Stärke
        self.min_trend_duration = 5  # Mindestens 5 Perioden Trend
        self.ideal_market_regimes = [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.BREAKOUT]
        self.min_adx_strength = 30  # Mindestens ADX 30 für starken Trend
        
        # Technical Thresholds
        self.ema_periods = [21, 50, 100, 200]  # Multiple EMAs für Trend-Bestätigung
        self.volume_threshold = 1.3  # 130% Volume-Bestätigung
        
        # Performance Tracking
        self.signals_generated = 0
        self.trades_this_month = 0
        self.monthly_reset_time = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        self.win_count = 0
        self.loss_count = 0
        
        # Trend Analysis Cache
        self.trend_analysis_cache: Dict[str, Dict] = {}
        
    async def initialize(self) -> bool:
        """🔥 Trend Hunter initialisieren"""
        try:
            logger.info("🎯 Trend Hunter wird initialisiert...")
            
            if not self.enabled:
                logger.info("⚠️ Trend Hunter ist deaktiviert")
                return True
            
            logger.success(f"✅ Trend Hunter initialisiert (Target Win Rate: {self.target_win_rate:.1%})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Trend Hunter Initialisierung fehlgeschlagen: {e}")
            return False
    
    # ==================== MAIN STRATEGY LOGIC ====================
    
    async def analyze_trend_opportunity(self, symbol: str) -> Optional[Dict[str, Any]]:
        """🎯 Trend-Following-Gelegenheit analysieren"""
        try:
            # Monthly Trade Limit Check
            await self._check_monthly_reset()
            if self.trades_this_month >= self.max_trades_per_month:
                logger.info(f"📊 Monatliches Trade-Limit erreicht: {self.trades_this_month}/{self.max_trades_per_month}")
                return None
            
            logger.info(f"🎯 Trend-Analyse wird durchgeführt: {symbol}")
            
            # Market Intelligence für Trend-Bestätigung
            market_analysis = await self.market_intelligence.analyze_market(symbol, self.timeframes)
            if not market_analysis:
                logger.info(f"📊 Keine Market Analysis für {symbol}")
                return None
            
            # Market Regime Check
            if market_analysis.regime not in self.ideal_market_regimes:
                logger.info(f"📊 Market Regime nicht geeignet für Trend Following: {market_analysis.regime.value}")
                return None
            
            # Trend Strength Check
            if market_analysis.trend_strength < self.min_trend_strength:
                logger.info(f"📊 Trend zu schwach: {market_analysis.trend_strength:.2%}")
                return None
            
            # Multi-Timeframe Trend Analysis
            trend_signals = []
            for timeframe in self.timeframes:
                signal = await self._analyze_trend_timeframe(symbol, timeframe, market_analysis)
                if signal:
                    trend_signals.append(signal)
            
            if not trend_signals:
                return None
            
            # Beste Signal auswählen (längster Timeframe bevorzugt)
            best_signal = max(trend_signals, key=lambda x: (x['timeframe_weight'], x['confidence']))
            
            # Trend Continuation vs Trend Start unterscheiden
            trend_type = await self._classify_trend_type(symbol, best_signal)
            best_signal['trend_type'] = trend_type
            
            # Final Validation
            if best_signal['confidence'] >= 0.7:  # 70% Minimum für Trend Following
                self.signals_generated += 1
                log_trade(f"🎯 Trend Signal generiert: {symbol} {best_signal['side'].value} {trend_type} (Confidence: {best_signal['confidence']:.2%})")
                return best_signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Trend-Analyse für {symbol}: {e}")
            return None
    
    async def _analyze_trend_timeframe(self, symbol: str, timeframe: str, market_analysis) -> Optional[Dict[str, Any]]:
        """📊 Timeframe-spezifische Trend-Analyse"""
        try:
            # Candles abrufen (mehr für Trend-Analyse)
            candles = await self.bitget.get_candles(symbol, timeframe, limit=200)
            if len(candles) < 100:
                return None
            
            # Multi-EMA Trend Analysis
            ema_analysis = await self._analyze_multi_ema_trend(candles)
            if not ema_analysis['trend_confirmed']:
                return None
            
            # ADX Trend Strength
            adx_analysis = await self._analyze_adx_trend_strength(candles)
            if adx_analysis['adx'] < self.min_adx_strength:
                return None
            
            # Trend Duration & Quality
            trend_quality = await self._analyze_trend_quality(candles, timeframe)
            
            # Volume Trend Confirmation
            volume_confirmation = await self._analyze_trend_volume(candles)
            
            # Momentum Divergence Check
            momentum_analysis = await self._analyze_trend_momentum(candles)
            
            # Trend Signal Score
            trend_score = self._calculate_trend_score(
                ema_analysis, adx_analysis, trend_quality, volume_confirmation, momentum_analysis, timeframe
            )
            
            if trend_score['score'] < 0.65:
                return None
            
            # Entry, SL, TP für Trend Following
            current_price = float(candles[-1].close)
            signal_side = trend_score['side']
            
            entry_price, stop_loss, take_profit = self._calculate_trend_levels(
                current_price, signal_side, ema_analysis, trend_quality, timeframe
            )
            
            # Timeframe Weight (längere Timeframes haben höheres Gewicht)
            timeframe_weights = {'4h': 1.0, '6h': 1.2, '1d': 1.5}
            timeframe_weight = timeframe_weights.get(timeframe, 1.0)
            
            return {
                'symbol': symbol,
                'side': signal_side,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': trend_score['confidence'],
                'trend_score': trend_score['score'],
                'timeframe': timeframe,
                'timeframe_weight': timeframe_weight,
                'analysis': {
                    'ema': ema_analysis,
                    'adx': adx_analysis,
                    'quality': trend_quality,
                    'volume': volume_confirmation,
                    'momentum': momentum_analysis
                },
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Trend Timeframe Analysis: {e}")
            return None
    
    async def _analyze_multi_ema_trend(self, candles: List) -> Dict[str, Any]:
        """📈 Multi-EMA Trend Analysis"""
        try:
            closes = [float(c.close) for c in candles]
            
            # EMAs berechnen
            emas = {}
            for period in self.ema_periods:
                ema_values = self._calculate_ema(closes, period)
                if ema_values:
                    emas[period] = ema_values[-1]
            
            if len(emas) < 3:
                return {'trend_confirmed': False}
            
            current_price = closes[-1]
            
            # EMA Alignment Check (alle EMAs in richtiger Reihenfolge)
            ema_values = [emas[period] for period in sorted(self.ema_periods) if period in emas]
            
            # Bullish: Preis > EMA21 > EMA50 > EMA100 > EMA200
            bullish_alignment = all(ema_values[i] > ema_values[i+1] for i in range(len(ema_values)-1))
            bullish_alignment = bullish_alignment and current_price > emas[21]
            
            # Bearish: Preis < EMA21 < EMA50 < EMA100 < EMA200
            bearish_alignment = all(ema_values[i] < ema_values[i+1] for i in range(len(ema_values)-1))
            bearish_alignment = bearish_alignment and current_price < emas[21]
            
            trend_direction = None
            trend_strength = 0
            
            if bullish_alignment:
                trend_direction = 'bullish'
                # Stärke basierend auf Abstand zwischen EMAs
                ema_spread = (emas[21] - emas[200]) / emas[200] if 200 in emas else 0.05
                trend_strength = min(1.0, ema_spread * 10)
            elif bearish_alignment:
                trend_direction = 'bearish'
                ema_spread = (emas[200] - emas[21]) / emas[200] if 200 in emas else 0.05
                trend_strength = min(1.0, ema_spread * 10)
            
            # EMA Slope (Trend-Momentum)
            ema_slope = 0
            if 21 in emas and len(closes) >= 10:
                ema_21_prev = self._calculate_ema(closes[:-5], 21)
                if ema_21_prev:
                    ema_slope = (emas[21] - ema_21_prev[-1]) / ema_21_prev[-1]
            
            return {
                'trend_confirmed': bullish_alignment or bearish_alignment,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'ema_slope': ema_slope,
                'emas': emas,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Multi-EMA Analysis: {e}")
            return {'trend_confirmed': False}
    
    async def _analyze_adx_trend_strength(self, candles: List) -> Dict[str, Any]:
        """📊 ADX Trend Strength Analysis"""
        try:
            highs = [float(c.high) for c in candles]
            lows = [float(c.low) for c in candles]
            closes = [float(c.close) for c in candles]
            
            # ADX berechnen (vereinfacht)
            adx_values = self._calculate_adx(highs, lows, closes, 14)
            
            if not adx_values:
                return {'adx': 0, 'trend_strength': 'weak'}
            
            current_adx = adx_values[-1]
            
            # ADX Trend (steigend/fallend)
            adx_trend = 'rising' if len(adx_values) >= 5 and current_adx > adx_values[-5] else 'falling'
            
            # Trend Strength Klassifikation
            if current_adx >= 50:
                strength = 'very_strong'
            elif current_adx >= 30:
                strength = 'strong'
            elif current_adx >= 20:
                strength = 'moderate'
            else:
                strength = 'weak'
            
            return {
                'adx': current_adx,
                'adx_trend': adx_trend,
                'trend_strength': strength,
                'adx_values': adx_values[-5:] if len(adx_values) >= 5 else adx_values
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei ADX Analysis: {e}")
            return {'adx': 0, 'trend_strength': 'weak'}
    
    async def _analyze_trend_quality(self, candles: List, timeframe: str) -> Dict[str, Any]:
        """🔍 Trend-Qualität analysieren"""
        try:
            closes = [float(c.close) for c in candles]
            
            # Trend Duration (Anzahl aufeinanderfolgender Perioden in gleiche Richtung)
            trend_duration = 0
            if len(closes) >= 10:
                recent_closes = closes[-10:]
                
                # Bullish Duration
                bullish_count = 0
                for i in range(1, len(recent_closes)):
                    if recent_closes[i] > recent_closes[i-1]:
                        bullish_count += 1
                    else:
                        break
                
                # Bearish Duration
                bearish_count = 0
                for i in range(1, len(recent_closes)):
                    if recent_closes[i] < recent_closes[i-1]:
                        bearish_count += 1
                    else:
                        break
                
                trend_duration = max(bullish_count, bearish_count)
            
            # Trend Consistency (R² der Linear Regression)
            if len(closes) >= 20:
                x = np.arange(len(closes[-20:]))
                z = np.polyfit(x, closes[-20:], 1)
                p = np.poly1d(z)
                predicted = p(x)
                
                ss_res = np.sum((closes[-20:] - predicted) ** 2)
                ss_tot = np.sum((closes[-20:] - np.mean(closes[-20:])) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                slope = z[0]
                slope_percent = (slope / np.mean(closes[-20:])) * 100
            else:
                r_squared = 0
                slope_percent = 0
            
            # Higher Highs / Lower Lows Pattern
            highs = [float(c.high) for c in candles[-15:]]
            lows = [float(c.low) for c in candles[-15:]]
            
            higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
            lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
            
            pattern_strength = max(higher_highs, lower_lows) / (len(highs) - 1) if len(highs) > 1 else 0
            
            return {
                'trend_duration': trend_duration,
                'r_squared': r_squared,
                'slope_percent': slope_percent,
                'pattern_strength': pattern_strength,
                'higher_highs': higher_highs,
                'lower_lows': lower_lows,
                'quality_score': (r_squared + pattern_strength + min(1.0, trend_duration / 10)) / 3
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Trend Quality Analysis: {e}")
            return {'quality_score': 0}
    
    async def _analyze_trend_volume(self, candles: List) -> Dict[str, Any]:
        """📊 Volume Trend Confirmation"""
        try:
            volumes = [float(c.volume) for c in candles]
            closes = [float(c.close) for c in candles]
            
            if len(volumes) < 20:
                return {'volume_confirmation': 0.5}
            
            # Volume Trend (letzte 10 vs vorherige 10)
            recent_volume = volumes[-10:]
            prev_volume = volumes[-20:-10]
            
            volume_trend = (sum(recent_volume) - sum(prev_volume)) / sum(prev_volume) if sum(prev_volume) > 0 else 0
            
            # Volume-Price Trend Confirmation
            price_direction = 1 if closes[-1] > closes[-10] else -1
            volume_price_alignment = 1 if (volume_trend > 0 and price_direction > 0) or (volume_trend < 0 and price_direction < 0) else 0
            
            # Volume MA Ratio
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Volume Confirmation Score
            volume_confirmation = (
                min(1.0, abs(volume_trend) * 2) * 0.4 +
                volume_price_alignment * 0.3 +
                min(1.0, volume_ratio / 1.5) * 0.3
            )
            
            return {
                'volume_confirmation': volume_confirmation,
                'volume_trend': volume_trend,
                'volume_ratio': volume_ratio,
                'volume_price_alignment': volume_price_alignment
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Volume Analysis: {e}")
            return {'volume_confirmation': 0.5}
    
    async def _analyze_trend_momentum(self, candles: List) -> Dict[str, Any]:
        """⚡ Momentum & Divergence Analysis"""
        try:
            closes = [float(c.close) for c in candles]
            
            # RSI für Momentum
            rsi = self._calculate_rsi(closes, 14)
            current_rsi = rsi[-1] if rsi else 50
            
            # MACD für Trend-Momentum
            macd_line, macd_signal, macd_hist = self._calculate_macd(closes)
            
            momentum_signals = []
            
            # RSI Trend Alignment (nicht in extremen Bereichen)
            if 30 < current_rsi < 70:
                momentum_signals.append('rsi_neutral')
            elif current_rsi >= 70:
                momentum_signals.append('rsi_overbought')
            elif current_rsi <= 30:
                momentum_signals.append('rsi_oversold')
            
            # MACD Trend Confirmation
            if len(macd_line) >= 2 and len(macd_signal) >= 2:
                if macd_line[-1] > macd_signal[-1] and macd_line[-1] > 0:
                    momentum_signals.append('macd_bullish')
                elif macd_line[-1] < macd_signal[-1] and macd_line[-1] < 0:
                    momentum_signals.append('macd_bearish')
            
            # Momentum Score
            momentum_score = 0.7 if 'rsi_neutral' in momentum_signals else 0.5
            
            if 'macd_bullish' in momentum_signals or 'macd_bearish' in momentum_signals:
                momentum_score += 0.2
            
            return {
                'momentum_score': momentum_score,
                'rsi': current_rsi,
                'momentum_signals': momentum_signals,
                'macd_trend': 'bullish' if len(macd_line) >= 1 and macd_line[-1] > 0 else 'bearish'
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Momentum Analysis: {e}")
            return {'momentum_score': 0.5}
    
    def _calculate_trend_score(self, ema_analysis: Dict, adx_analysis: Dict, trend_quality: Dict, 
                             volume_confirmation: Dict, momentum_analysis: Dict, timeframe: str) -> Dict[str, Any]:
        """🎯 Trend Following Score berechnen"""
        try:
            scores = []
            
            # EMA Trend Score (40% Gewichtung)
            if ema_analysis['trend_confirmed']:
                ema_score = ema_analysis['trend_strength'] * 0.4
                scores.append(ema_score)
                trend_side = OrderSide.BUY if ema_analysis['trend_direction'] == 'bullish' else OrderSide.SELL
            else:
                return {'score': 0, 'confidence': 0, 'side': OrderSide.BUY}
            
            # ADX Score (25% Gewichtung)
            adx_score = min(1.0, adx_analysis['adx'] / 50) * 0.25
            scores.append(adx_score)
            
            # Trend Quality Score (20% Gewichtung)
            quality_score = trend_quality['quality_score'] * 0.20
            scores.append(quality_score)
            
            # Volume Confirmation (10% Gewichtung)
            volume_score = volume_confirmation['volume_confirmation'] * 0.10
            scores.append(volume_score)
            
            # Momentum Score (5% Gewichtung)
            momentum_score = momentum_analysis['momentum_score'] * 0.05
            scores.append(momentum_score)
            
            # Base Score
            base_score = sum(scores)
            
            # Timeframe Bonus (längere Timeframes bevorzugt)
            tf_bonus = {'4h': 0.0, '6h': 0.05, '1d': 0.10}.get(timeframe, 0.0)
            final_score = min(1.0, base_score + tf_bonus)
            
            # Confidence basierend auf Signal-Stärke
            confidence = final_score * 0.9  # Konservativ für Trend Following
            
            return {
                'score': final_score,
                'confidence': confidence,
                'side': trend_side,
                'components': {
                    'ema_score': ema_score,
                    'adx_score': adx_score,
                    'quality_score': quality_score,
                    'volume_score': volume_score,
                    'momentum_score': momentum_score
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Trend Score Berechnung: {e}")
            return {'score': 0, 'confidence': 0, 'side': OrderSide.BUY}
    
    def _calculate_trend_levels(self, current_price: float, side: OrderSide, ema_analysis: Dict, 
                              trend_quality: Dict, timeframe: str) -> Tuple[float, float, float]:
        """💰 Trend Following Levels berechnen"""
        try:
            # Timeframe-spezifische Faktoren für 1:3 Risk/Reward
            tf_factors = {
                '4h': {'sl_factor': 0.04, 'tp_factor': 0.12},   # 4% SL, 12% TP (1:3)
                '6h': {'sl_factor': 0.05, 'tp_factor': 0.15},   # 5% SL, 15% TP (1:3)
                '1d': {'sl_factor': 0.06, 'tp_factor': 0.18}    # 6% SL, 18% TP (1:3)
            }
            
            factors = tf_factors.get(timeframe, tf_factors['4h'])
            
            # EMA-basierte Stop Loss (näher EMA als Schutz)
            if side == OrderSide.BUY:
                # Stop Loss unter EMA 21 oder EMA 50
                ema_21 = ema_analysis['emas'].get(21, current_price * 0.98)
                ema_50 = ema_analysis['emas'].get(50, current_price * 0.96)
                
                # Wähle nähere EMA für SL, aber mindestens die kalkulierte Distanz
                calculated_sl = current_price * (1 - factors['sl_factor'])
                ema_based_sl = min(ema_21, ema_50) * 0.995  # 0.5% unter EMA
                
                stop_loss = max(calculated_sl, ema_based_sl)  # Konservativere SL
                take_profit = current_price * (1 + factors['tp_factor'])
                entry_price = current_price * 1.002  # 0.2% über Markt
                
            else:  # SELL
                # Stop Loss über EMA 21 oder EMA 50
                ema_21 = ema_analysis['emas'].get(21, current_price * 1.02)
                ema_50 = ema_analysis['emas'].get(50, current_price * 1.04)
                
                calculated_sl = current_price * (1 + factors['sl_factor'])
                ema_based_sl = max(ema_21, ema_50) * 1.005  # 0.5% über EMA
                
                stop_loss = min(calculated_sl, ema_based_sl)  # Konservativere SL
                take_profit = current_price * (1 - factors['tp_factor'])
                entry_price = current_price * 0.998  # 0.2% unter Markt
            
            return entry_price, stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Trend Levels Berechnung: {e}")
            # Fallback
            if side == OrderSide.BUY:
                return current_price * 1.002, current_price * 0.95, current_price * 1.15
            else:
                return current_price * 0.998, current_price * 1.05, current_price * 0.85
    
    async def _classify_trend_type(self, symbol: str, signal: Dict) -> str:
        """🔍 Trend Type klassifizieren (Start vs Continuation)"""
        try:
            # Vereinfachte Klassifikation basierend auf Trend Duration
            trend_duration = signal['analysis']['quality']['trend_duration']
            
            if trend_duration <= 3:
                return 'trend_start'
            elif trend_duration <= 7:
                return 'trend_continuation'
            else:
                return 'trend_maturity'
                
        except Exception:
            return 'trend_continuation'
    
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
            
            adx_values = []
            
            for i in range(period, len(closes)):
                # Vereinfachte ADX-Berechnung
                tr_values = []
                for j in range(i - period + 1, i + 1):
                    if j > 0:
                        tr = max(
                            highs[j] - lows[j],
                            abs(highs[j] - closes[j-1]),
                            abs(lows[j] - closes[j-1])
                        )
                        tr_values.append(tr)
                
                if tr_values:
                    atr = sum(tr_values) / len(tr_values)
                    price_range = highs[i] - lows[i]
                    adx = min(100, (atr / price_range) * 100) if price_range > 0 else 0
                    adx_values.append(adx)
            
            return adx_values
        except Exception:
            return []
    
    async def _check_monthly_reset(self):
        """📅 Monatliche Statistiken zurücksetzen"""
        try:
            now = datetime.utcnow()
            if now.month != self.monthly_reset_time.month or now.year != self.monthly_reset_time.year:
                self.trades_this_month = 0
                self.monthly_reset_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                logger.info("📅 Monatliche Trend-Statistiken zurückgesetzt")
        except Exception as e:
            logger.error(f"❌ Fehler bei Monthly Reset: {e}")
    
    # ==================== PUBLIC METHODS ====================
    
    async def generate_trade_signal(self, symbol: str) -> Optional[TradeSignal]:
        """🎯 Trading Signal für das Hauptsystem generieren"""
        try:
            if not self.enabled:
                return None
            
            # Trend Signal analysieren
            trend_signal = await self.analyze_trend_opportunity(symbol)
            if not trend_signal:
                return None
            
            # Position Size berechnen
            portfolio = await self.bitget.get_portfolio()
            if not portfolio:
                return None
            
            risk_amount = float(portfolio.total_balance) * 0.025  # 2.5% Risiko für Trend Following
            entry_price = trend_signal['entry_price']
            stop_loss = trend_signal['stop_loss']
            
            position_size = calculate_position_size(
                float(portfolio.total_balance),
                2.5,  # 2.5% Risiko
                entry_price,
                stop_loss
            )
            
            # TradeSignal erstellen
            trade_signal = TradeSignal(
                id=generate_signal_id(symbol, "TREND"),
                symbol=symbol,
                side=trend_signal['side'],
                entry_price=Decimal(str(trend_signal['entry_price'])),
                stop_loss=Decimal(str(trend_signal['stop_loss'])),
                take_profit=Decimal(str(trend_signal['take_profit'])),
                quantity=Decimal(str(position_size)),
                confidence=trend_signal['confidence'],
                strategy="trend_hunter",
                timeframe=trend_signal['timeframe'],
                timestamp=datetime.utcnow(),
                metadata={
                    'trend_score': trend_signal['trend_score'],
                    'trend_type': trend_signal['trend_type'],
                    'analysis': trend_signal['analysis'],
                    'timeframe_weight': trend_signal['timeframe_weight'],
                    'risk_reward_ratio': abs(trend_signal['take_profit'] - trend_signal['entry_price']) / abs(trend_signal['entry_price'] - trend_signal['stop_loss'])
                }
            )
            
            self.trades_this_month += 1
            return trade_signal
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Trend Trade Signal Generation: {e}")
            return None
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """📊 Strategie-Statistiken abrufen"""
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        
        return {
            'strategy_name': 'trend_hunter',
            'enabled': self.enabled,
            'signals_generated': self.signals_generated,
            'trades_this_month': self.trades_this_month,
            'max_trades_per_month': self.max_trades_per_month,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'target_win_rate': self.target_win_rate,
            'timeframes': self.timeframes,
            'cache_size': len(self.trend_analysis_cache)
        }
    
    async def update_performance(self, signal_id: str, was_successful: bool):
        """📊 Performance Update"""
        try:
            if was_successful:
                self.win_count += 1
            else:
                self.loss_count += 1
            
            logger.info(f"📊 Trend Performance updated: {signal_id} -> {'Win' if was_successful else 'Loss'}")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Performance Update: {e}")
    
    async def shutdown(self):
        """🛑 Trend Hunter herunterfahren"""
        try:
            self.trend_analysis_cache.clear()
            logger.info("✅ Trend Hunter heruntergefahren")
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren: {e}")
