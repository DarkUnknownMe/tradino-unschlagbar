"""
📊 TRADINO UNSCHLAGBAR - Market Intelligence
Advanced Market Analysis mit 20+ Technical Indicators

Author: AI Trading Systems
"""

import asyncio
import numpy as np
import pandas as pd
import ta
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from models.market_models import Candle, MarketData, TechnicalIndicators, MarketAnalysis, MarketRegime
from connectors.bitget_pro import BitgetPro
from utils.logger_pro import setup_logger, log_ai_decision
from utils.config_manager import ConfigManager
from utils.math_utils import safe_divide

logger = setup_logger("MarketIntelligence")


class TrendDirection(Enum):
    """Trend Richtung"""
    STRONG_BULLISH = 4
    BULLISH = 3
    NEUTRAL = 2
    BEARISH = 1
    STRONG_BEARISH = 0


@dataclass
class MarketSignal:
    """Market Signal Model"""
    symbol: str
    signal_strength: float  # 0-1
    trend_direction: TrendDirection
    volatility_level: float  # 0-1
    volume_confirmation: float  # 0-1
    momentum_score: float  # 0-1
    support_levels: List[float]
    resistance_levels: List[float]
    confidence: float  # 0-1
    timestamp: datetime


class MarketIntelligence:
    """📊 Advanced Market Intelligence System"""
    
    def __init__(self, config: ConfigManager, bitget_connector: BitgetPro):
        self.config = config
        self.bitget = bitget_connector
        
        # Market Data Cache
        self.candle_cache: Dict[str, List[Candle]] = {}
        self.indicator_cache: Dict[str, TechnicalIndicators] = {}
        self.analysis_cache: Dict[str, MarketAnalysis] = {}
        
        # Technical Analysis Settings
        self.indicator_config = config.get('technical_analysis.indicators', [])
        self.lookback_periods = {
            '1m': 1000,    # 1000 Minuten = ~16 Stunden
            '5m': 500,     # 500 * 5 = ~41 Stunden  
            '15m': 300,    # 300 * 15 = ~75 Stunden
            '1h': 168,     # 168 Stunden = 1 Woche
            '4h': 168,     # 168 * 4 = ~4 Wochen
            '1d': 365      # 365 Tage = 1 Jahr
        }
        
        # Market Regime Detection
        self.regime_thresholds = {
            'trend_strength': 0.6,
            'volatility_high': 0.7,
            'volatility_low': 0.3,
            'volume_surge': 2.0
        }
        
        # Performance Tracking
        self.analysis_count = 0
        self.accuracy_scores: List[float] = []
        
    async def initialize(self) -> bool:
        """🔥 Market Intelligence initialisieren"""
        try:
            logger.info("📊 Market Intelligence wird initialisiert...")
            
            # Indicator Validierung
            if not self.indicator_config:
                logger.warning("⚠️ Keine Technical Indicators konfiguriert")
                return False
            
            logger.success(f"✅ Market Intelligence mit {len(self.indicator_config)} Indikatoren initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ Market Intelligence Initialisierung fehlgeschlagen: {e}")
            return False
    
    # ==================== MAIN ANALYSIS METHODS ====================
    
    async def analyze_market(self, symbol: str, timeframes: List[str] = None) -> Optional[MarketAnalysis]:
        """🔍 Komplette Marktanalyse für Symbol"""
        try:
            if not timeframes:
                timeframes = ['5m', '15m', '1h', '4h']
            
            logger.info(f"🔍 Marktanalyse wird durchgeführt: {symbol}")
            
            # Multi-Timeframe Analysis
            timeframe_analyses = {}
            for tf in timeframes:
                analysis = await self._analyze_timeframe(symbol, tf)
                if analysis:
                    timeframe_analyses[tf] = analysis
            
            if not timeframe_analyses:
                logger.warning(f"⚠️ Keine Analyse-Daten für {symbol}")
                return None
            
            # Gesamtanalyse zusammenfassen
            combined_analysis = await self._combine_timeframe_analyses(symbol, timeframe_analyses)
            
            # Cache aktualisieren
            self.analysis_cache[symbol] = combined_analysis
            self.analysis_count += 1
            
            log_ai_decision(
                "MarketIntelligence", 
                f"{symbol} {combined_analysis.regime.value}", 
                combined_analysis.confidence
            )
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Marktanalyse für {symbol}: {e}")
            return None
    
    async def _analyze_timeframe(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """📈 Einzelner Timeframe Analysis"""
        try:
            # Kerzendaten abrufen
            candles = await self._get_candles(symbol, timeframe)
            if len(candles) < 50:  # Mindestens 50 Kerzen
                return None
            
            # DataFrame erstellen
            df = self._candles_to_dataframe(candles)
            
            # Technical Indicators berechnen
            indicators = await self._calculate_indicators(df, symbol, timeframe)
            
            # Trend Analysis
            trend_analysis = self._analyze_trend(df, indicators)
            
            # Volatility Analysis
            volatility_analysis = self._analyze_volatility(df, indicators)
            
            # Volume Analysis
            volume_analysis = self._analyze_volume(df, indicators)
            
            # Support/Resistance Levels
            sr_levels = self._find_support_resistance(df)
            
            # Momentum Analysis
            momentum_analysis = self._analyze_momentum(df, indicators)
            
            return {
                'timeframe': timeframe,
                'indicators': indicators,
                'trend': trend_analysis,
                'volatility': volatility_analysis,
                'volume': volume_analysis,
                'support_resistance': sr_levels,
                'momentum': momentum_analysis,
                'candles_count': len(candles)
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Timeframe-Analyse {symbol} {timeframe}: {e}")
            return None
    
    # ==================== TECHNICAL INDICATORS ====================
    
    async def _calculate_indicators(self, df: pd.DataFrame, symbol: str, timeframe: str) -> TechnicalIndicators:
        """📊 Technical Indicators berechnen"""
        try:
            indicators = TechnicalIndicators(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.utcnow()
            )
            
            # Trend Indicators
            if 'SMA_20' in self.indicator_config:
                indicators.sma_20 = Decimal(str(df['close'].rolling(20).mean().iloc[-1]))
            if 'SMA_50' in self.indicator_config:
                indicators.sma_50 = Decimal(str(df['close'].rolling(50).mean().iloc[-1]))
            if 'SMA_200' in self.indicator_config:
                indicators.sma_200 = Decimal(str(df['close'].rolling(200).mean().iloc[-1]))
            
            if 'EMA_12' in self.indicator_config:
                indicators.ema_12 = Decimal(str(df['close'].ewm(span=12).mean().iloc[-1]))
            if 'EMA_26' in self.indicator_config:
                indicators.ema_26 = Decimal(str(df['close'].ewm(span=26).mean().iloc[-1]))
            
            # Momentum Indicators
            if 'RSI' in self.indicator_config:
                rsi = ta.momentum.RSIIndicator(df['close'], window=14)
                indicators.rsi = float(rsi.rsi().iloc[-1])
            
            if 'MACD' in self.indicator_config:
                macd = ta.trend.MACD(df['close'])
                indicators.macd = Decimal(str(macd.macd().iloc[-1]))
                indicators.macd_signal = Decimal(str(macd.macd_signal().iloc[-1]))
                indicators.macd_histogram = Decimal(str(macd.macd_diff().iloc[-1]))
            
            # Volatility Indicators
            if 'Bollinger_Bands' in self.indicator_config:
                bb = ta.volatility.BollingerBands(df['close'])
                indicators.bb_upper = Decimal(str(bb.bollinger_hband().iloc[-1]))
                indicators.bb_middle = Decimal(str(bb.bollinger_mavg().iloc[-1]))
                indicators.bb_lower = Decimal(str(bb.bollinger_lband().iloc[-1]))
            
            if 'ATR' in self.indicator_config:
                atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'])
                indicators.atr = Decimal(str(atr.average_true_range().iloc[-1]))
            
            # Volume Indicators
            if 'OBV' in self.indicator_config:
                obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
                indicators.obv = Decimal(str(obv.on_balance_volume().iloc[-1]))
            
            if 'Volume_SMA' in self.indicator_config:
                indicators.volume_sma = Decimal(str(df['volume'].rolling(20).mean().iloc[-1]))
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Indicator-Berechnung: {e}")
            return TechnicalIndicators(symbol=symbol, timeframe=timeframe)
    
    # ==================== ANALYSIS METHODS ====================
    
    def _analyze_trend(self, df: pd.DataFrame, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """📈 Trend Analyse"""
        try:
            trend_signals = []
            current_price = df['close'].iloc[-1]
            
            # SMA Trend Analysis
            if indicators.sma_20 and indicators.sma_50:
                sma_20 = float(indicators.sma_20)
                sma_50 = float(indicators.sma_50)
                
                if sma_20 > sma_50 and current_price > sma_20:
                    trend_signals.append(1)  # Bullish
                elif sma_20 < sma_50 and current_price < sma_20:
                    trend_signals.append(-1)  # Bearish
                else:
                    trend_signals.append(0)  # Neutral
            
            # EMA Trend Analysis
            if indicators.ema_12 and indicators.ema_26:
                ema_12 = float(indicators.ema_12)
                ema_26 = float(indicators.ema_26)
                
                if ema_12 > ema_26 and current_price > ema_12:
                    trend_signals.append(1)
                elif ema_12 < ema_26 and current_price < ema_12:
                    trend_signals.append(-1)
                else:
                    trend_signals.append(0)
            
            # MACD Trend Analysis
            if indicators.macd and indicators.macd_signal:
                macd = float(indicators.macd)
                macd_signal = float(indicators.macd_signal)
                
                if macd > macd_signal and macd > 0:
                    trend_signals.append(1)
                elif macd < macd_signal and macd < 0:
                    trend_signals.append(-1)
                else:
                    trend_signals.append(0)
            
            # Price Action Trend
            price_change_5 = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]
            if price_change_5 > 0.02:  # 2% Anstieg
                trend_signals.append(1)
            elif price_change_5 < -0.02:  # 2% Rückgang
                trend_signals.append(-1)
            else:
                trend_signals.append(0)
            
            # Trend Score berechnen
            if trend_signals:
                trend_score = sum(trend_signals) / len(trend_signals)
                trend_strength = abs(trend_score)
                
                # Trend Direction klassifizieren
                if trend_score >= 0.75:
                    direction = TrendDirection.STRONG_BULLISH
                elif trend_score >= 0.25:
                    direction = TrendDirection.BULLISH
                elif trend_score <= -0.75:
                    direction = TrendDirection.STRONG_BEARISH
                elif trend_score <= -0.25:
                    direction = TrendDirection.BEARISH
                else:
                    direction = TrendDirection.NEUTRAL
            else:
                trend_score = 0
                trend_strength = 0
                direction = TrendDirection.NEUTRAL
            
            return {
                'trend_score': trend_score,
                'trend_strength': trend_strength,
                'direction': direction,
                'signals_count': len(trend_signals)
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Trend-Analyse: {e}")
            return {'trend_score': 0, 'trend_strength': 0, 'direction': TrendDirection.NEUTRAL}
    
    def _analyze_volatility(self, df: pd.DataFrame, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """📊 Volatilität Analyse"""
        try:
            volatility_scores = []
            
            # ATR-basierte Volatilität
            if indicators.atr:
                atr_value = float(indicators.atr)
                current_price = df['close'].iloc[-1]
                atr_percent = atr_value / current_price
                
                # Volatilität Score (0-1)
                volatility_scores.append(min(1.0, atr_percent * 50))  # Normalisierung
            
            # Bollinger Bands Width
            if indicators.bb_upper and indicators.bb_lower and indicators.bb_middle:
                bb_upper = float(indicators.bb_upper)
                bb_lower = float(indicators.bb_lower)
                bb_middle = float(indicators.bb_middle)
                
                bb_width = (bb_upper - bb_lower) / bb_middle
                volatility_scores.append(min(1.0, bb_width * 20))  # Normalisierung
            
            # Price Volatility (20-period)
            price_std = df['close'].rolling(20).std().iloc[-1]
            price_mean = df['close'].rolling(20).mean().iloc[-1]
            cv = price_std / price_mean  # Coefficient of Variation
            volatility_scores.append(min(1.0, cv * 10))
            
            # Durchschnittliche Volatilität
            volatility_level = sum(volatility_scores) / len(volatility_scores) if volatility_scores else 0
            
            # Volatilität klassifizieren
            if volatility_level >= self.regime_thresholds['volatility_high']:
                volatility_regime = "high"
            elif volatility_level <= self.regime_thresholds['volatility_low']:
                volatility_regime = "low"
            else:
                volatility_regime = "moderate"
            
            return {
                'volatility_level': volatility_level,
                'regime': volatility_regime,
                'atr_percent': atr_value / df['close'].iloc[-1] if indicators.atr else 0,
                'scores_count': len(volatility_scores)
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Volatilität-Analyse: {e}")
            return {'volatility_level': 0, 'regime': 'low'}
    
    def _analyze_volume(self, df: pd.DataFrame, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """📊 Volume Analyse"""
        try:
            current_volume = df['volume'].iloc[-1]
            
            # Volume SMA Vergleich
            volume_scores = []
            if indicators.volume_sma:
                volume_sma = float(indicators.volume_sma)
                volume_ratio = current_volume / volume_sma
                volume_scores.append(min(2.0, volume_ratio) / 2.0)  # Normalisierung 0-1
            
            # Volume Trend (5-period)
            volume_trend = (df['volume'].iloc[-5:].mean() - df['volume'].iloc[-10:-5].mean()) / df['volume'].iloc[-10:-5].mean()
            volume_scores.append(min(1.0, max(0.0, (volume_trend + 1) / 2)))  # Normalisierung
            
            # OBV Trend
            if indicators.obv:
                obv_values = df['volume'].cumsum()  # Vereinfachte OBV
                obv_trend = (obv_values.iloc[-5:].mean() - obv_values.iloc[-10:-5].mean()) / abs(obv_values.iloc[-10:-5].mean())
                volume_scores.append(min(1.0, max(0.0, (obv_trend + 1) / 2)))
            
            # Volume Score
            volume_score = sum(volume_scores) / len(volume_scores) if volume_scores else 0
            
            # Volume Confirmation Level
            if current_volume > df['volume'].rolling(20).mean().iloc[-1] * self.regime_thresholds['volume_surge']:
                confirmation_level = 1.0  # Starke Bestätigung
            elif current_volume > df['volume'].rolling(20).mean().iloc[-1]:
                confirmation_level = 0.7  # Moderate Bestätigung
            else:
                confirmation_level = 0.3  # Schwache Bestätigung
            
            return {
                'volume_score': volume_score,
                'confirmation_level': confirmation_level,
                'current_vs_average': current_volume / df['volume'].rolling(20).mean().iloc[-1],
                'trend': 'increasing' if volume_trend > 0.1 else 'decreasing' if volume_trend < -0.1 else 'stable'
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Volume-Analyse: {e}")
            return {'volume_score': 0, 'confirmation_level': 0.3}
    
    def _analyze_momentum(self, df: pd.DataFrame, indicators: TechnicalIndicators) -> Dict[str, Any]:
        """⚡ Momentum Analyse"""
        try:
            momentum_scores = []
            
            # RSI Momentum
            if indicators.rsi:
                rsi = indicators.rsi
                if rsi > 70:
                    momentum_scores.append(0.9)  # Overbought - starkes Momentum
                elif rsi > 50:
                    momentum_scores.append(0.7)  # Bullish Momentum
                elif rsi < 30:
                    momentum_scores.append(0.1)  # Oversold - schwaches Momentum
                else:
                    momentum_scores.append(0.3)  # Bearish Momentum
            
            # MACD Momentum
            if indicators.macd_histogram:
                macd_hist = float(indicators.macd_histogram)
                # MACD Histogram Änderung
                momentum_scores.append(min(1.0, max(0.0, (macd_hist + 1) / 2)))
            
            # Price Momentum (ROC - Rate of Change)
            roc_5 = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]
            roc_10 = (df['close'].iloc[-1] - df['close'].iloc[-11]) / df['close'].iloc[-11]
            
            avg_roc = (roc_5 + roc_10) / 2
            momentum_scores.append(min(1.0, max(0.0, (avg_roc + 0.1) / 0.2)))  # Normalisierung
            
            # Momentum Score
            momentum_score = sum(momentum_scores) / len(momentum_scores) if momentum_scores else 0.5
            
            # Momentum Richtung
            if momentum_score >= 0.7:
                momentum_direction = "strong_bullish"
            elif momentum_score >= 0.6:
                momentum_direction = "bullish"
            elif momentum_score <= 0.3:
                momentum_direction = "bearish"
            elif momentum_score <= 0.4:
                momentum_direction = "weak_bearish"
            else:
                momentum_direction = "neutral"
            
            return {
                'momentum_score': momentum_score,
                'direction': momentum_direction,
                'rsi_level': indicators.rsi if indicators.rsi else 50,
                'roc_5_period': roc_5,
                'signals_count': len(momentum_scores)
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Momentum-Analyse: {e}")
            return {'momentum_score': 0.5, 'direction': 'neutral'}
    
    def _find_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """📊 Support/Resistance Levels finden"""
        try:
            highs = df['high'].rolling(window, center=True).max()
            lows = df['low'].rolling(window, center=True).min()
            
            # Resistance Levels (lokale Hochs)
            resistance_levels = []
            for i in range(window, len(df) - window):
                if df['high'].iloc[i] == highs.iloc[i] and df['high'].iloc[i] > df['high'].iloc[i-5:i+5].mean() * 1.01:
                    resistance_levels.append(float(df['high'].iloc[i]))
            
            # Support Levels (lokale Tiefs)
            support_levels = []
            for i in range(window, len(df) - window):
                if df['low'].iloc[i] == lows.iloc[i] and df['low'].iloc[i] < df['low'].iloc[i-5:i+5].mean() * 0.99:
                    support_levels.append(float(df['low'].iloc[i]))
            
            # Nur die relevantesten Levels (letzte 5)
            resistance_levels = sorted(resistance_levels, reverse=True)[:5]
            support_levels = sorted(support_levels, reverse=True)[:5]
            
            return {
                'resistance_levels': resistance_levels,
                'support_levels': support_levels
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Support/Resistance Analyse: {e}")
            return {'resistance_levels': [], 'support_levels': []}
    
    # ==================== UTILITY METHODS ====================
    
    async def _get_candles(self, symbol: str, timeframe: str) -> List[Candle]:
        """📈 Kerzendaten abrufen (cached)"""
        cache_key = f"{symbol}_{timeframe}"
        
        # Cache Check
        if cache_key in self.candle_cache:
            cached_candles = self.candle_cache[cache_key]
            if cached_candles and (datetime.utcnow() - cached_candles[-1].timestamp).seconds < 300:  # 5 Min Cache
                return cached_candles
        
        # Neue Daten abrufen
        limit = self.lookback_periods.get(timeframe, 500)
        candles = await self.bitget.get_candles(symbol, timeframe, limit)
        
        if candles:
            self.candle_cache[cache_key] = candles
        
        return candles
    
    def _candles_to_dataframe(self, candles: List[Candle]) -> pd.DataFrame:
        """📊 Candles zu DataFrame konvertieren"""
        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': float(candle.open),
                'high': float(candle.high),
                'low': float(candle.low),
                'close': float(candle.close),
                'volume': float(candle.volume)
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    async def _combine_timeframe_analyses(self, symbol: str, analyses: Dict[str, Dict]) -> MarketAnalysis:
        """🔄 Multi-Timeframe Analysen kombinieren"""
        try:
            # Gewichtung nach Timeframe-Wichtigkeit
            weights = {'5m': 0.15, '15m': 0.20, '1h': 0.30, '4h': 0.35}
            
            # Gewichtete Scores berechnen
            trend_scores = []
            volatility_scores = []
            volume_scores = []
            momentum_scores = []
            
            all_support_levels = []
            all_resistance_levels = []
            
            for tf, analysis in analyses.items():
                weight = weights.get(tf, 0.25)
                
                if 'trend' in analysis:
                    trend_scores.append(analysis['trend']['trend_strength'] * weight)
                if 'volatility' in analysis:
                    volatility_scores.append(analysis['volatility']['volatility_level'] * weight)
                if 'volume' in analysis:
                    volume_scores.append(analysis['volume']['volume_score'] * weight)
                if 'momentum' in analysis:
                    momentum_scores.append(analysis['momentum']['momentum_score'] * weight)
                
                # Support/Resistance sammeln
                if 'support_resistance' in analysis:
                    all_support_levels.extend(analysis['support_resistance']['support_levels'])
                    all_resistance_levels.extend(analysis['support_resistance']['resistance_levels'])
            
            # Finale Scores
            final_trend_strength = sum(trend_scores) if trend_scores else 0
            final_volatility_score = sum(volatility_scores) if volatility_scores else 0
            final_volume_score = sum(volume_scores) if volume_scores else 0
            final_momentum_score = sum(momentum_scores) if momentum_scores else 0
            
            # Market Regime bestimmen
            regime = self._determine_market_regime(
                final_trend_strength, 
                final_volatility_score, 
                final_volume_score
            )
            
            # Confidence Score berechnen
            confidence = self._calculate_confidence(
                final_trend_strength,
                final_volatility_score,
                final_volume_score,
                final_momentum_score,
                len(analyses)
            )
            
            # Support/Resistance Levels clustern
            support_levels = self._cluster_levels(all_support_levels)[:3]
            resistance_levels = self._cluster_levels(all_resistance_levels)[:3]
            
            return MarketAnalysis(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                regime=regime,
                trend_strength=final_trend_strength,
                volatility_score=final_volatility_score,
                volume_score=final_volume_score,
                support_levels=[Decimal(str(level)) for level in support_levels],
                resistance_levels=[Decimal(str(level)) for level in resistance_levels],
                confidence=confidence,
                metadata={
                    'timeframes_analyzed': list(analyses.keys()),
                    'momentum_score': final_momentum_score,
                    'analysis_timestamp': datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Kombinieren der Analysen: {e}")
            return MarketAnalysis(
                symbol=symbol,
                regime=MarketRegime.RANGE_BOUND,
                trend_strength=0,
                volatility_score=0,
                volume_score=0,
                confidence=0
            )
    
    def _determine_market_regime(self, trend_strength: float, volatility: float, volume: float) -> MarketRegime:
        """🔍 Market Regime bestimmen"""
        try:
            # Regime-Logik
            if trend_strength >= 0.7 and volume >= 0.6:
                if volatility >= 0.7:
                    return MarketRegime.BREAKOUT
                else:
                    return MarketRegime.TRENDING_UP if trend_strength > 0 else MarketRegime.TRENDING_DOWN
            
            elif volatility >= 0.8:
                return MarketRegime.HIGH_VOLATILITY
            
            elif volatility <= 0.3 and trend_strength <= 0.4:
                return MarketRegime.LOW_VOLATILITY
            
            elif trend_strength <= 0.3 and volatility <= 0.6:
                return MarketRegime.RANGE_BOUND
            
            else:
                return MarketRegime.RANGE_BOUND  # Default
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Regime-Bestimmung: {e}")
            return MarketRegime.RANGE_BOUND
    
    def _calculate_confidence(self, trend: float, volatility: float, volume: float, 
                           momentum: float, timeframes_count: int) -> float:
        """🎯 Confidence Score berechnen"""
        try:
            # Basis-Confidence basierend auf Signal-Stärken
            base_confidence = (trend + volume + momentum) / 3
            
            # Volatilität Adjustment (moderate Volatilität ist besser)
            volatility_adjustment = 1 - abs(volatility - 0.5) * 0.4
            
            # Timeframe Adjustment (mehr Timeframes = höhere Confidence)
            timeframe_adjustment = min(1.0, timeframes_count / 4)
            
            # Finale Confidence
            confidence = base_confidence * volatility_adjustment * timeframe_adjustment
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Confidence-Berechnung: {e}")
            return 0.5
    
    def _cluster_levels(self, levels: List[float], tolerance: float = 0.02) -> List[float]:
        """📊 Support/Resistance Levels clustern"""
        try:
            if not levels:
                return []
            
            levels = sorted(levels)
            clustered = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                # Wenn Level innerhalb der Toleranz zum Cluster-Durchschnitt
                cluster_avg = sum(current_cluster) / len(current_cluster)
                if abs(level - cluster_avg) / cluster_avg <= tolerance:
                    current_cluster.append(level)
                else:
                    # Cluster abschließen und neuen beginnen
                    clustered.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [level]
            
            # Letzten Cluster hinzufügen
            if current_cluster:
                clustered.append(sum(current_cluster) / len(current_cluster))
            
            return clustered
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Clustern der Levels: {e}")
            return levels[:5] if levels else []
    
    # ==================== PUBLIC METHODS ====================
    
    async def get_market_signal(self, symbol: str) -> Optional[MarketSignal]:
        """🎯 Market Signal für Trading generieren"""
        try:
            analysis = await self.analyze_market(symbol)
            if not analysis:
                return None
            
            # Signal Strength basierend auf Trend und Volume
            signal_strength = (analysis.trend_strength + analysis.volume_score) / 2
            
            # Trend Direction
            trend_direction = TrendDirection.NEUTRAL
            if analysis.trend_strength >= 0.7:
                trend_direction = TrendDirection.STRONG_BULLISH if analysis.trend_strength > 0 else TrendDirection.STRONG_BEARISH
            elif analysis.trend_strength >= 0.5:
                trend_direction = TrendDirection.BULLISH if analysis.trend_strength > 0 else TrendDirection.BEARISH
            
            # Market Signal erstellen
            market_signal = MarketSignal(
                symbol=symbol,
                signal_strength=signal_strength,
                trend_direction=trend_direction,
                volatility_level=analysis.volatility_score,
                volume_confirmation=analysis.volume_score,
                momentum_score=analysis.metadata.get('momentum_score', 0.5),
                support_levels=[float(level) for level in analysis.support_levels],
                resistance_levels=[float(level) for level in analysis.resistance_levels],
                confidence=analysis.confidence,
                timestamp=datetime.utcnow()
            )
            
            return market_signal
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Market Signal Generation: {e}")
            return None
    
    def get_cached_analysis(self, symbol: str) -> Optional[MarketAnalysis]:
        """📊 Gecachte Analyse abrufen"""
        return self.analysis_cache.get(symbol)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """📈 Performance Statistiken"""
        avg_accuracy = sum(self.accuracy_scores) / len(self.accuracy_scores) if self.accuracy_scores else 0
        
        return {
            'total_analyses': self.analysis_count,
            'cached_symbols': len(self.analysis_cache),
            'average_accuracy': avg_accuracy,
            'indicators_configured': len(self.indicator_config)
        }
    
    async def shutdown(self):
        """🛑 Market Intelligence herunterfahren"""
        try:
            # Cache leeren
            self.candle_cache.clear()
            self.indicator_cache.clear()
            self.analysis_cache.clear()
            
            logger.info("✅ Market Intelligence heruntergefahren")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren: {e}")
