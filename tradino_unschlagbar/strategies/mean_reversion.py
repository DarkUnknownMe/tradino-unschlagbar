"""
⚖️ TRADINO UNSCHLAGBAR - Mean Reversion Strategy
Range-bound Mean Reversion Strategie für 15min-1h Timeframes
Target: 70-75% Win Rate, 1:1.5 Risk/Reward, 5-15 Trades/Tag

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

logger = setup_logger("MeanReversion")


class MeanReversion:
    """⚖️ Mean Reversion Trading Strategie"""
    
    def __init__(self, config: ConfigManager, bitget_connector: BitgetProConnector, 
                 master_ai: MasterAI, market_intelligence: MarketIntelligence):
        self.config = config
        self.bitget = bitget_connector
        self.master_ai = master_ai
        self.market_intelligence = market_intelligence
        
        # Strategy Configuration
        self.strategy_config = config.get('strategies.mean_reversion', {})
        self.enabled = self.strategy_config.get('enabled', True)
        self.timeframes = self.strategy_config.get('timeframes', ['15m', '30m', '1h'])
        self.target_win_rate = self.strategy_config.get('target_win_rate', 0.73)
        self.risk_reward_ratio = self.strategy_config.get('risk_reward', 1.5)
        self.max_trades_per_day = self.strategy_config.get('max_trades_per_day', 15)
        
        # Mean Reversion Parameters
        self.ideal_market_regimes = [MarketRegime.RANGE_BOUND, MarketRegime.LOW_VOLATILITY]
        self.max_trend_strength = 0.4  # Maximal 40% Trend-Stärke (für Range-bound)
        self.min_range_duration = 10  # Mindestens 10 Perioden in Range
        
        # Technical Thresholds
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.bollinger_oversold = 0.2  # 20% Position in BB
        self.bollinger_overbought = 0.8  # 80% Position in BB
        self.mean_reversion_strength = 2.0  # 2 Standardabweichungen
        
        # Performance Tracking
        self.signals_generated = 0
        self.trades_today = 0
        self.daily_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        self.win_count = 0
        self.loss_count = 0
        
        # Range Detection Cache
        self.range_analysis_cache: Dict[str, Dict] = {}
        
    async def initialize(self) -> bool:
        """🔥 Mean Reversion initialisieren"""
        try:
            logger.info("⚖️ Mean Reversion wird initialisiert...")
            
            if not self.enabled:
                logger.info("⚠️ Mean Reversion ist deaktiviert")
                return True
            
            logger.success(f"✅ Mean Reversion initialisiert (Target Win Rate: {self.target_win_rate:.1%})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Mean Reversion Initialisierung fehlgeschlagen: {e}")
            return False
    
    # ==================== MAIN STRATEGY LOGIC ====================
    
    async def analyze_mean_reversion_opportunity(self, symbol: str) -> Optional[Dict[str, Any]]:
        """⚖️ Mean Reversion Gelegenheit analysieren"""
        try:
            # Daily Trade Limit Check
            await self._check_daily_reset()
            if self.trades_today >= self.max_trades_per_day:
                logger.info(f"📊 Tägliches Trade-Limit erreicht: {self.trades_today}/{self.max_trades_per_day}")
                return None
            
            logger.info(f"⚖️ Mean Reversion Analyse wird durchgeführt: {symbol}")
            
            # Market Intelligence für Range-bound Check
            market_analysis = await self.market_intelligence.analyze_market(symbol, self.timeframes)
            if not market_analysis:
                logger.info(f"📊 Keine Market Analysis für {symbol}")
                return None
            
            # Market Regime Check (Range-bound bevorzugt)
            if market_analysis.regime not in self.ideal_market_regimes:
                logger.info(f"📊 Market Regime nicht geeignet für Mean Reversion: {market_analysis.regime.value}")
                return None
            
            # Trend Strength Check (schwacher Trend bevorzugt)
            if market_analysis.trend_strength > self.max_trend_strength:
                logger.info(f"📊 Trend zu stark für Mean Reversion: {market_analysis.trend_strength:.2%}")
                return None
            
            # Multi-Timeframe Range Analysis
            reversion_signals = []
            for timeframe in self.timeframes:
                signal = await self._analyze_reversion_timeframe(symbol, timeframe, market_analysis)
                if signal:
                    reversion_signals.append(signal)
            
            if not reversion_signals:
                return None
            
            # Beste Signal auswählen (höchste Confidence)
            best_signal = max(reversion_signals, key=lambda x: x['confidence'])
            
            # Final Validation
            if best_signal['confidence'] >= 0.75:  # 75% Minimum für Mean Reversion
                self.signals_generated += 1
                log_trade(f"⚖️ Mean Reversion Signal generiert: {symbol} {best_signal['side'].value} (Confidence: {best_signal['confidence']:.2%})")
                return best_signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Mean Reversion Analyse für {symbol}: {e}")
            return None
    
    async def _analyze_reversion_timeframe(self, symbol: str, timeframe: str, market_analysis) -> Optional[Dict[str, Any]]:
        """📊 Timeframe-spezifische Mean Reversion Analyse"""
        try:
            # Candles abrufen
            candles = await self.bitget.get_candles(symbol, timeframe, limit=100)
            if len(candles) < 50:
                return None
            
            # Range Detection
            range_analysis = await self._detect_trading_range(candles, timeframe)
            if not range_analysis['in_range']:
                return None
            
            # Mean Reversion Signals
            reversion_signals = await self._analyze_reversion_signals(candles, range_analysis)
            if not reversion_signals['signals']:
                return None
            
            # Bollinger Bands Mean Reversion
            bb_analysis = await self._analyze_bollinger_reversion(candles)
            
            # RSI Mean Reversion
            rsi_analysis = await self._analyze_rsi_reversion(candles)
            
            # Statistical Mean Reversion
            statistical_analysis = await self._analyze_statistical_reversion(candles, range_analysis)
            
            # Mean Reversion Score
            reversion_score = self._calculate_reversion_score(
                reversion_signals, bb_analysis, rsi_analysis, statistical_analysis, timeframe
            )
            
            if reversion_score['score'] < 0.7:  # Hoher Threshold für Mean Reversion
                return None
            
            # Entry, SL, TP für Mean Reversion
            current_price = float(candles[-1].close)
            signal_side = reversion_score['side']
            
            entry_price, stop_loss, take_profit = self._calculate_reversion_levels(
                current_price, signal_side, range_analysis, bb_analysis, timeframe
            )
            
            return {
                'symbol': symbol,
                'side': signal_side,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': reversion_score['confidence'],
                'reversion_score': reversion_score['score'],
                'timeframe': timeframe,
                'analysis': {
                    'range': range_analysis,
                    'bollinger': bb_analysis,
                    'rsi': rsi_analysis,
                    'statistical': statistical_analysis
                },
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Mean Reversion Timeframe Analysis: {e}")
            return None
    
    async def _detect_trading_range(self, candles: List, timeframe: str) -> Dict[str, Any]:
        """📊 Trading Range Detection"""
        try:
            highs = [float(c.high) for c in candles]
            lows = [float(c.low) for c in candles]
            closes = [float(c.close) for c in candles]
            
            # Support und Resistance Levels finden
            lookback = min(50, len(candles))
            recent_highs = highs[-lookback:]
            recent_lows = lows[-lookback:]
            
            # Resistance (höchste Hochs)
            resistance_level = max(recent_highs)
            resistance_touches = sum(1 for h in recent_highs if abs(h - resistance_level) / resistance_level <= 0.01)
            
            # Support (niedrigste Tiefs)
            support_level = min(recent_lows)
            support_touches = sum(1 for l in recent_lows if abs(l - support_level) / support_level <= 0.01)
            
            # Range Characteristics
            range_size = (resistance_level - support_level) / support_level
            current_price = closes[-1]
            range_position = (current_price - support_level) / (resistance_level - support_level)
            
            # Range Duration (wie lange schon in Range)
            range_duration = 0
            for i in range(len(closes) - 1, 0, -1):
                if support_level <= closes[i] <= resistance_level:
                    range_duration += 1
                else:
                    break
            
            # Range Quality Score
            min_touches = 3  # Mindestens 3 Touches für valide Range
            range_quality = (
                min(1.0, resistance_touches / min_touches) * 0.3 +
                min(1.0, support_touches / min_touches) * 0.3 +
                min(1.0, range_duration / self.min_range_duration) * 0.4
            )
            
            in_range = (range_quality >= 0.7 and 
                       range_size >= 0.03 and  # Mindestens 3% Range
                       range_duration >= self.min_range_duration)
            
            return {
                'in_range': in_range,
                'support_level': support_level,
                'resistance_level': resistance_level,
                'range_size': range_size,
                'range_position': range_position,
                'range_duration': range_duration,
                'range_quality': range_quality,
                'support_touches': support_touches,
                'resistance_touches': resistance_touches
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Range Detection: {e}")
            return {'in_range': False}
    
    async def _analyze_reversion_signals(self, candles: List, range_analysis: Dict) -> Dict[str, Any]:
        """📊 Mean Reversion Signale analysieren"""
        try:
            closes = [float(c.close) for c in candles]
            current_price = closes[-1]
            
            signals = []
            
            # Range Position Signals
            range_pos = range_analysis['range_position']
            support = range_analysis['support_level']
            resistance = range_analysis['resistance_level']
            
            # Near Support (Buy Signal)
            if range_pos <= 0.25:  # Unteren 25% der Range
                distance_to_support = (current_price - support) / support
                if distance_to_support <= 0.02:  # Maximal 2% über Support
                    signals.append({
                        'type': 'support_bounce',
                        'side': OrderSide.BUY,
                        'strength': 1 - distance_to_support * 25  # Je näher, desto stärker
                    })
            
            # Near Resistance (Sell Signal)
            elif range_pos >= 0.75:  # Oberen 25% der Range
                distance_to_resistance = (resistance - current_price) / resistance
                if distance_to_resistance <= 0.02:  # Maximal 2% unter Resistance
                    signals.append({
                        'type': 'resistance_rejection',
                        'side': OrderSide.SELL,
                        'strength': 1 - distance_to_resistance * 25
                    })
            
            # Range Center Reversion
            center = (support + resistance) / 2
            distance_from_center = abs(current_price - center) / center
            
            if distance_from_center >= 0.03:  # Mindestens 3% von Center entfernt
                if current_price > center:
                    signals.append({
                        'type': 'center_reversion_sell',
                        'side': OrderSide.SELL,
                        'strength': min(1.0, distance_from_center * 15)
                    })
                else:
                    signals.append({
                        'type': 'center_reversion_buy',
                        'side': OrderSide.BUY,
                        'strength': min(1.0, distance_from_center * 15)
                    })
            
            return {
                'signals': signals,
                'range_position': range_pos,
                'distance_from_center': distance_from_center
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Reversion Signals: {e}")
            return {'signals': []}
    
    async def _analyze_bollinger_reversion(self, candles: List) -> Dict[str, Any]:
        """📊 Bollinger Bands Mean Reversion"""
        try:
            closes = [float(c.close) for c in candles]
            
            # Bollinger Bands berechnen
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes, 20, 2)
            
            if not bb_upper or not bb_lower:
                return {'bb_signal': None, 'bb_position': 0.5}
            
            current_price = closes[-1]
            bb_position = ((current_price - bb_lower[-1]) / 
                          (bb_upper[-1] - bb_lower[-1]))
            
            bb_signal = None
            bb_strength = 0
            
            # Bollinger Bands Oversold (Buy)
            if bb_position <= self.bollinger_oversold:
                bb_signal = OrderSide.BUY
                bb_strength = (self.bollinger_oversold - bb_position) / self.bollinger_oversold
            
            # Bollinger Bands Overbought (Sell)
            elif bb_position >= self.bollinger_overbought:
                bb_signal = OrderSide.SELL
                bb_strength = (bb_position - self.bollinger_overbought) / (1 - self.bollinger_overbought)
            
            # BB Squeeze Detection (niedrige Volatilität)
            bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
            recent_widths = [(bb_upper[i] - bb_lower[i]) / bb_middle[i] 
                           for i in range(max(0, len(bb_upper) - 10), len(bb_upper))]
            avg_width = sum(recent_widths) / len(recent_widths) if recent_widths else bb_width
            
            squeeze = bb_width < avg_width * 0.8  # 20% enger als Durchschnitt
            
            return {
                'bb_signal': bb_signal,
                'bb_strength': bb_strength,
                'bb_position': bb_position,
                'bb_squeeze': squeeze,
                'bb_width': bb_width
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Bollinger Analysis: {e}")
            return {'bb_signal': None, 'bb_position': 0.5}
    
    async def _analyze_rsi_reversion(self, candles: List) -> Dict[str, Any]:
        """📊 RSI Mean Reversion"""
        try:
            closes = [float(c.close) for c in candles]
            
            # RSI berechnen
            rsi = self._calculate_rsi(closes, 14)
            if not rsi:
                return {'rsi_signal': None, 'rsi_value': 50}
            
            current_rsi = rsi[-1]
            
            rsi_signal = None
            rsi_strength = 0
            
            # RSI Oversold (Buy)
            if current_rsi <= self.rsi_oversold:
                rsi_signal = OrderSide.BUY
                rsi_strength = (self.rsi_oversold - current_rsi) / self.rsi_oversold
            
            # RSI Overbought (Sell)
            elif current_rsi >= self.rsi_overbought:
                rsi_signal = OrderSide.SELL
                rsi_strength = (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            
            # RSI Divergence Detection (vereinfacht)
            rsi_trend = 'neutral'
            if len(rsi) >= 10:
                recent_rsi = rsi[-5:]
                prev_rsi = rsi[-10:-5]
                
                if sum(recent_rsi) > sum(prev_rsi):
                    rsi_trend = 'rising'
                elif sum(recent_rsi) < sum(prev_rsi):
                    rsi_trend = 'falling'
            
            return {
                'rsi_signal': rsi_signal,
                'rsi_strength': rsi_strength,
                'rsi_value': current_rsi,
                'rsi_trend': rsi_trend
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei RSI Analysis: {e}")
            return {'rsi_signal': None, 'rsi_value': 50}
    
    async def _analyze_statistical_reversion(self, candles: List, range_analysis: Dict) -> Dict[str, Any]:
        """📊 Statistische Mean Reversion"""
        try:
            closes = [float(c.close) for c in candles]
            
            # Moving Average (Range Center als Mean)
            ma_period = 20
            if len(closes) < ma_period:
                return {'stat_signal': None}
            
            moving_average = sum(closes[-ma_period:]) / ma_period
            current_price = closes[-1]
            
            # Standard Deviation
            squared_diffs = [(price - moving_average) ** 2 for price in closes[-ma_period:]]
            std_dev = (sum(squared_diffs) / ma_period) ** 0.5
            
            # Z-Score (Standardabweichungen vom Mean)
            z_score = (current_price - moving_average) / std_dev if std_dev > 0 else 0
            
            stat_signal = None
            stat_strength = 0
            
            # Mean Reversion basierend auf Z-Score
            if z_score <= -self.mean_reversion_strength:  # 2 Std Dev unter Mean
                stat_signal = OrderSide.BUY
                stat_strength = min(1.0, abs(z_score) / 3)  # Max bei 3 Std Dev
            elif z_score >= self.mean_reversion_strength:  # 2 Std Dev über Mean
                stat_signal = OrderSide.SELL
                stat_strength = min(1.0, abs(z_score) / 3)
            
            return {
                'stat_signal': stat_signal,
                'stat_strength': stat_strength,
                'z_score': z_score,
                'moving_average': moving_average,
                'std_dev': std_dev
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Statistical Analysis: {e}")
            return {'stat_signal': None}
    
    def _calculate_reversion_score(self, reversion_signals: Dict, bb_analysis: Dict, 
                                 rsi_analysis: Dict, statistical_analysis: Dict, timeframe: str) -> Dict[str, Any]:
        """⚖️ Mean Reversion Score berechnen"""
        try:
            scores = []
            sides = []
            
            # Range Position Signals (40% Gewichtung)
            if reversion_signals['signals']:
                best_signal = max(reversion_signals['signals'], key=lambda x: x['strength'])
                scores.append(best_signal['strength'] * 0.4)
                sides.append(best_signal['side'])
            
            # Bollinger Bands (25% Gewichtung)
            if bb_analysis['bb_signal']:
                scores.append(bb_analysis['bb_strength'] * 0.25)
                sides.append(bb_analysis['bb_signal'])
            
            # RSI (20% Gewichtung)
            if rsi_analysis['rsi_signal']:
                scores.append(rsi_analysis['rsi_strength'] * 0.20)
                sides.append(rsi_analysis['rsi_signal'])
            
            # Statistical (15% Gewichtung)
            if statistical_analysis['stat_signal']:
                scores.append(statistical_analysis['stat_strength'] * 0.15)
                sides.append(statistical_analysis['stat_signal'])
            
            if not scores:
                return {'score': 0, 'confidence': 0, 'side': OrderSide.BUY}
            
            # Base Score
            base_score = sum(scores)
            
            # Timeframe Bonus (kürzere Timeframes bevorzugt für Mean Reversion)
            tf_bonus = {'15m': 0.10, '30m': 0.05, '1h': 0.0}.get(timeframe, 0.0)
            final_score = min(1.0, base_score + tf_bonus)
            
            # Dominant Side
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
            
            # Confidence (hoch für Mean Reversion bei guter Signal-Übereinstimmung)
            signal_agreement = max(buy_votes, sell_votes) / len(sides) if sides else 0.5
            confidence = final_score * (0.8 + signal_agreement * 0.2)
            
            return {
                'score': final_score,
                'confidence': confidence,
                'side': dominant_side,
                'signal_count': len(scores),
                'agreement': signal_agreement
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Reversion Score Berechnung: {e}")
            return {'score': 0, 'confidence': 0, 'side': OrderSide.BUY}
    
    def _calculate_reversion_levels(self, current_price: float, side: OrderSide, 
                                  range_analysis: Dict, bb_analysis: Dict, timeframe: str) -> Tuple[float, float, float]:
        """💰 Mean Reversion Levels berechnen"""
        try:
            support = range_analysis['support_level']
            resistance = range_analysis['resistance_level']
            range_center = (support + resistance) / 2
            
            # Timeframe-spezifische Faktoren für 1:1.5 Risk/Reward
            tf_factors = {
                '15m': {'sl_factor': 0.015, 'tp_factor': 0.0225},  # 1.5% SL, 2.25% TP
                '30m': {'sl_factor': 0.020, 'tp_factor': 0.030},   # 2% SL, 3% TP
                '1h': {'sl_factor': 0.025, 'tp_factor': 0.0375}    # 2.5% SL, 3.75% TP
            }
            
            factors = tf_factors.get(timeframe, tf_factors['30m'])
            
            if side == OrderSide.BUY:
                # Entry leicht unter aktuellem Preis für besseren Fill
                entry_price = current_price * 0.9995
                
                # Stop Loss unter Support oder basierend auf Faktor
                support_based_sl = support * 0.995  # 0.5% unter Support
                calculated_sl = current_price * (1 - factors['sl_factor'])
                stop_loss = max(support_based_sl, calculated_sl)  # Konservativere SL
                
                # Take Profit zum Range Center oder basierend auf Faktor
                center_based_tp = min(range_center, resistance * 0.98)  # Vor Resistance stoppen
                calculated_tp = current_price * (1 + factors['tp_factor'])
                take_profit = min(center_based_tp, calculated_tp)
                
            else:  # SELL
                entry_price = current_price * 1.0005
                
                # Stop Loss über Resistance
                resistance_based_sl = resistance * 1.005  # 0.5% über Resistance
                calculated_sl = current_price * (1 + factors['sl_factor'])
                stop_loss = min(resistance_based_sl, calculated_sl)
                
                # Take Profit zum Range Center
                center_based_tp = max(range_center, support * 1.02)  # Nach Support stoppen
                calculated_tp = current_price * (1 - factors['tp_factor'])
                take_profit = max(center_based_tp, calculated_tp)
            
            return entry_price, stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Reversion Levels Berechnung: {e}")
            # Fallback
            if side == OrderSide.BUY:
                return current_price * 0.9995, current_price * 0.98, current_price * 1.025
            else:
                return current_price * 1.0005, current_price * 1.02, current_price * 0.975
    
    # ==================== UTILITY METHODS ====================
    
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
        except Exception:
            return [], [], []
    
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
    
    async def _check_daily_reset(self):
        """📅 Tägliche Statistiken zurücksetzen"""
        try:
            now = datetime.utcnow()
            if now.date() > self.daily_reset_time.date():
                self.trades_today = 0
                self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
                logger.info("📅 Tägliche Mean Reversion Statistiken zurückgesetzt")
        except Exception as e:
            logger.error(f"❌ Fehler bei Daily Reset: {e}")
    
    # ==================== PUBLIC METHODS ====================
    
    async def generate_trade_signal(self, symbol: str) -> Optional[TradeSignal]:
        """⚖️ Trading Signal für das Hauptsystem generieren"""
        try:
            if not self.enabled:
                return None
            
            # Mean Reversion Signal analysieren
            reversion_signal = await self.analyze_mean_reversion_opportunity(symbol)
            if not reversion_signal:
                return None
            
            # Position Size berechnen
            portfolio = await self.bitget.get_portfolio()
            if not portfolio:
                return None
            
            risk_amount = float(portfolio.total_balance) * 0.015  # 1.5% Risiko für Mean Reversion
            entry_price = reversion_signal['entry_price']
            stop_loss = reversion_signal['stop_loss']
            
            position_size = calculate_position_size(
                float(portfolio.total_balance),
                1.5,  # 1.5% Risiko
                entry_price,
                stop_loss
            )
            
            # TradeSignal erstellen
            trade_signal = TradeSignal(
                id=generate_signal_id(symbol, "REVERSION"),
                symbol=symbol,
                side=reversion_signal['side'],
                entry_price=Decimal(str(reversion_signal['entry_price'])),
                stop_loss=Decimal(str(reversion_signal['stop_loss'])),
                take_profit=Decimal(str(reversion_signal['take_profit'])),
                quantity=Decimal(str(position_size)),
                confidence=reversion_signal['confidence'],
                strategy="mean_reversion",
                timeframe=reversion_signal['timeframe'],
                timestamp=datetime.utcnow(),
                metadata={
                    'reversion_score': reversion_signal['reversion_score'],
                    'analysis': reversion_signal['analysis'],
                    'risk_reward_ratio': abs(reversion_signal['take_profit'] - reversion_signal['entry_price']) / abs(reversion_signal['entry_price'] - reversion_signal['stop_loss'])
                }
            )
            
            self.trades_today += 1
            return trade_signal
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Mean Reversion Trade Signal Generation: {e}")
            return None
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """📊 Strategie-Statistiken abrufen"""
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        
        return {
            'strategy_name': 'mean_reversion',
            'enabled': self.enabled,
            'signals_generated': self.signals_generated,
            'trades_today': self.trades_today,
            'max_trades_per_day': self.max_trades_per_day,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'target_win_rate': self.target_win_rate,
            'timeframes': self.timeframes,
            'cache_size': len(self.range_analysis_cache)
        }
    
    async def update_performance(self, signal_id: str, was_successful: bool):
        """📊 Performance Update"""
        try:
            if was_successful:
                self.win_count += 1
            else:
                self.loss_count += 1
            
            logger.info(f"📊 Mean Reversion Performance updated: {signal_id} -> {'Win' if was_successful else 'Loss'}")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Performance Update: {e}")
    
    async def shutdown(self):
        """🛑 Mean Reversion herunterfahren"""
        try:
            self.range_analysis_cache.clear()
            logger.info("✅ Mean Reversion heruntergefahren")
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren: {e}")
