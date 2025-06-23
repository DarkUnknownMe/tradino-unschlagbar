"""
🔍 TRADINO UNSCHLAGBAR - Pattern Recognition
Advanced Candlestick & Chart Pattern Detection

Author: AI Trading Systems
"""

import asyncio
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from models.market_models import Candle
from connectors.bitget_pro import BitgetPro
from utils.logger_pro import setup_logger, log_ai_decision
from utils.config_manager import ConfigManager

logger = setup_logger("PatternRecognition")


class PatternType(Enum):
    """Pattern Typen"""
    CANDLESTICK = "candlestick"
    CHART = "chart"
    PRICE_ACTION = "price_action"


class PatternSignal(Enum):
    """Pattern Signal"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"


@dataclass
class DetectedPattern:
    """Erkanntes Pattern Model"""
    name: str
    type: PatternType
    signal: PatternSignal
    confidence: float  # 0-1
    strength: float  # 0-1
    timeframe: str
    symbol: str
    start_time: datetime
    end_time: datetime
    key_levels: List[float]
    description: str
    metadata: Dict[str, Any]


class PatternRecognition:
    """🔍 Advanced Pattern Recognition System"""
    
    def __init__(self, config: ConfigManager, bitget_connector: BitgetPro):
        self.config = config
        self.bitget = bitget_connector
        
        # Pattern Cache
        self.detected_patterns: Dict[str, List[DetectedPattern]] = {}
        
        # Pattern Configuration
        self.candlestick_patterns = self._init_candlestick_patterns()
        self.chart_patterns = self._init_chart_patterns()
        
        # Performance Tracking
        self.patterns_detected = 0
        self.pattern_accuracy: Dict[str, List[float]] = {}
        
    def _init_candlestick_patterns(self) -> Dict[str, Dict]:
        """🕯️ Candlestick Pattern Definitionen"""
        return {
            'doji': {
                'signal': PatternSignal.REVERSAL,
                'min_confidence': 0.6,
                'description': 'Doji - Unentschlossenheit'
            },
            'hammer': {
                'signal': PatternSignal.BULLISH,
                'min_confidence': 0.7,
                'description': 'Hammer - Bullish Reversal'
            },
            'shooting_star': {
                'signal': PatternSignal.BEARISH,
                'min_confidence': 0.7,
                'description': 'Shooting Star - Bearish Reversal'
            },
            'engulfing_bullish': {
                'signal': PatternSignal.BULLISH,
                'min_confidence': 0.8,
                'description': 'Bullish Engulfing - Starke Umkehr'
            },
            'engulfing_bearish': {
                'signal': PatternSignal.BEARISH,
                'min_confidence': 0.8,
                'description': 'Bearish Engulfing - Starke Umkehr'
            },
            'morning_star': {
                'signal': PatternSignal.BULLISH,
                'min_confidence': 0.85,
                'description': 'Morning Star - Sehr bullish'
            },
            'evening_star': {
                'signal': PatternSignal.BEARISH,
                'min_confidence': 0.85,
                'description': 'Evening Star - Sehr bearish'
            },
            'three_white_soldiers': {
                'signal': PatternSignal.BULLISH,
                'min_confidence': 0.9,
                'description': 'Three White Soldiers - Sehr stark bullish'
            },
            'three_black_crows': {
                'signal': PatternSignal.BEARISH,
                'min_confidence': 0.9,
                'description': 'Three Black Crows - Sehr stark bearish'
            }
        }
    
    def _init_chart_patterns(self) -> Dict[str, Dict]:
        """📊 Chart Pattern Definitionen"""
        return {
            'head_and_shoulders': {
                'signal': PatternSignal.BEARISH,
                'min_confidence': 0.75,
                'description': 'Head and Shoulders - Bearish Reversal'
            },
            'inverse_head_and_shoulders': {
                'signal': PatternSignal.BULLISH,
                'min_confidence': 0.75,
                'description': 'Inverse H&S - Bullish Reversal'
            },
            'double_top': {
                'signal': PatternSignal.BEARISH,
                'min_confidence': 0.7,
                'description': 'Double Top - Bearish Reversal'
            },
            'double_bottom': {
                'signal': PatternSignal.BULLISH,
                'min_confidence': 0.7,
                'description': 'Double Bottom - Bullish Reversal'
            },
            'triangle_ascending': {
                'signal': PatternSignal.BULLISH,
                'min_confidence': 0.65,
                'description': 'Ascending Triangle - Bullish Continuation'
            },
            'triangle_descending': {
                'signal': PatternSignal.BEARISH,
                'min_confidence': 0.65,
                'description': 'Descending Triangle - Bearish Continuation'
            },
            'flag_bullish': {
                'signal': PatternSignal.BULLISH,
                'min_confidence': 0.7,
                'description': 'Bullish Flag - Continuation'
            },
            'flag_bearish': {
                'signal': PatternSignal.BEARISH,
                'min_confidence': 0.7,
                'description': 'Bearish Flag - Continuation'
            }
        }
    
    async def initialize(self) -> bool:
        """🔥 Pattern Recognition initialisieren"""
        try:
            logger.info("🔍 Pattern Recognition wird initialisiert...")
            
            total_patterns = len(self.candlestick_patterns) + len(self.chart_patterns)
            logger.success(f"✅ Pattern Recognition mit {total_patterns} Patterns initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pattern Recognition Initialisierung fehlgeschlagen: {e}")
            return False
    
    # ==================== MAIN DETECTION METHODS ====================
    
    async def detect_patterns(self, symbol: str, timeframes: List[str] = None) -> List[DetectedPattern]:
        """🔍 Patterns in Symbol erkennen"""
        try:
            if not timeframes:
                timeframes = ['5m', '15m', '1h', '4h']
            
            all_patterns = []
            
            for timeframe in timeframes:
                # Kerzendaten abrufen
                candles = await self.bitget.get_candles(symbol, timeframe, limit=200)
                if len(candles) < 50:
                    continue
                
                # Candlestick Patterns
                candlestick_patterns = await self._detect_candlestick_patterns(candles, symbol, timeframe)
                all_patterns.extend(candlestick_patterns)
                
                # Chart Patterns
                chart_patterns = await self._detect_chart_patterns(candles, symbol, timeframe)
                all_patterns.extend(chart_patterns)
                
                # Price Action Patterns
                price_action_patterns = await self._detect_price_action_patterns(candles, symbol, timeframe)
                all_patterns.extend(price_action_patterns)
            
            # Cache aktualisieren
            self.detected_patterns[symbol] = all_patterns
            self.patterns_detected += len(all_patterns)
            
            # Top Patterns loggen
            top_patterns = sorted(all_patterns, key=lambda x: x.confidence, reverse=True)[:3]
            for pattern in top_patterns:
                log_ai_decision(
                    "PatternRecognition",
                    f"{symbol} {pattern.name} {pattern.signal.value}",
                    pattern.confidence
                )
            
            return all_patterns
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Pattern Detection für {symbol}: {e}")
            return []
    
    # ==================== CANDLESTICK PATTERNS ====================
    
    async def _detect_candlestick_patterns(self, candles: List[Candle], symbol: str, timeframe: str) -> List[DetectedPattern]:
        """🕯️ Candlestick Patterns erkennen"""
        try:
            patterns = []
            df = self._candles_to_dataframe(candles)
            
            # Single Candle Patterns
            patterns.extend(self._detect_doji(df, symbol, timeframe))
            patterns.extend(self._detect_hammer(df, symbol, timeframe))
            patterns.extend(self._detect_shooting_star(df, symbol, timeframe))
            
            # Two Candle Patterns
            patterns.extend(self._detect_engulfing(df, symbol, timeframe))
            
            # Three Candle Patterns
            patterns.extend(self._detect_morning_evening_star(df, symbol, timeframe))
            patterns.extend(self._detect_three_soldiers_crows(df, symbol, timeframe))
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Candlestick Pattern Detection: {e}")
            return []
    
    def _detect_doji(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[DetectedPattern]:
        """🕯️ Doji Pattern erkennen"""
        patterns = []
        
        for i in range(1, len(df)):
            open_price = df['open'].iloc[i]
            close_price = df['close'].iloc[i]
            high_price = df['high'].iloc[i]
            low_price = df['low'].iloc[i]
            
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            
            # Doji: Sehr kleiner Body im Verhältnis zur Gesamtrange
            if total_range > 0 and body_size / total_range <= 0.1:
                confidence = 1 - (body_size / total_range) * 10  # Je kleiner der Body, desto höher die Confidence
                
                if confidence >= self.candlestick_patterns['doji']['min_confidence']:
                    pattern = DetectedPattern(
                        name='doji',
                        type=PatternType.CANDLESTICK,
                        signal=PatternSignal.REVERSAL,
                        confidence=confidence,
                        strength=confidence * 0.8,  # Doji ist moderate Stärke
                        timeframe=timeframe,
                        symbol=symbol,
                        start_time=df.index[i],
                        end_time=df.index[i],
                        key_levels=[open_price, high_price, low_price],
                        description='Doji - Markt Unentschlossenheit',
                        metadata={'body_to_range_ratio': body_size / total_range}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_hammer(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[DetectedPattern]:
        """🔨 Hammer Pattern erkennen"""
        patterns = []
        
        for i in range(1, len(df)):
            open_price = df['open'].iloc[i]
            close_price = df['close'].iloc[i]
            high_price = df['high'].iloc[i]
            low_price = df['low'].iloc[i]
            
            body_size = abs(close_price - open_price)
            lower_shadow = min(open_price, close_price) - low_price
            upper_shadow = high_price - max(open_price, close_price)
            total_range = high_price - low_price
            
            # Hammer: Langer unterer Schatten, kurzer oberer Schatten, kleiner Body
            if (total_range > 0 and 
                lower_shadow >= body_size * 2 and  # Unterer Schatten mindestens 2x Body
                upper_shadow <= body_size * 0.5 and  # Oberer Schatten maximal 0.5x Body
                lower_shadow >= total_range * 0.6):  # Unterer Schatten mindestens 60% der Range
                
                confidence = min(0.95, lower_shadow / (body_size * 2) * 0.8)
                
                if confidence >= self.candlestick_patterns['hammer']['min_confidence']:
                    # Bullish wenn in Downtrend
                    prev_trend = self._analyze_trend_context(df, i, lookback=5)
                    signal = PatternSignal.BULLISH if prev_trend < 0 else PatternSignal.NEUTRAL
                    
                    pattern = DetectedPattern(
                        name='hammer',
                        type=PatternType.CANDLESTICK,
                        signal=signal,
                        confidence=confidence,
                        strength=confidence * 0.9,
                        timeframe=timeframe,
                        symbol=symbol,
                        start_time=df.index[i],
                        end_time=df.index[i],
                        key_levels=[low_price, min(open_price, close_price), max(open_price, close_price)],
                        description='Hammer - Potentielle bullish Umkehr',
                        metadata={'lower_shadow_ratio': lower_shadow / total_range}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_shooting_star(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[DetectedPattern]:
        """⭐ Shooting Star Pattern erkennen"""
        patterns = []
        
        for i in range(1, len(df)):
            open_price = df['open'].iloc[i]
            close_price = df['close'].iloc[i]
            high_price = df['high'].iloc[i]
            low_price = df['low'].iloc[i]
            
            body_size = abs(close_price - open_price)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            total_range = high_price - low_price
            
            # Shooting Star: Langer oberer Schatten, kurzer unterer Schatten, kleiner Body
            if (total_range > 0 and 
                upper_shadow >= body_size * 2 and  # Oberer Schatten mindestens 2x Body
                lower_shadow <= body_size * 0.5 and  # Unterer Schatten maximal 0.5x Body
                upper_shadow >= total_range * 0.6):  # Oberer Schatten mindestens 60% der Range
                
                confidence = min(0.95, upper_shadow / (body_size * 2) * 0.8)
                
                if confidence >= self.candlestick_patterns['shooting_star']['min_confidence']:
                    # Bearish wenn in Uptrend
                    prev_trend = self._analyze_trend_context(df, i, lookback=5)
                    signal = PatternSignal.BEARISH if prev_trend > 0 else PatternSignal.NEUTRAL
                    
                    pattern = DetectedPattern(
                        name='shooting_star',
                        type=PatternType.CANDLESTICK,
                        signal=signal,
                        confidence=confidence,
                        strength=confidence * 0.9,
                        timeframe=timeframe,
                        symbol=symbol,
                        start_time=df.index[i],
                        end_time=df.index[i],
                        key_levels=[high_price, max(open_price, close_price), min(open_price, close_price)],
                        description='Shooting Star - Potentielle bearish Umkehr',
                        metadata={'upper_shadow_ratio': upper_shadow / total_range}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_engulfing(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[DetectedPattern]:
        """🤝 Engulfing Pattern erkennen"""
        patterns = []
        
        for i in range(2, len(df)):
            # Erste Kerze (i-1)
            open1 = df['open'].iloc[i-1]
            close1 = df['close'].iloc[i-1]
            body1_size = abs(close1 - open1)
            
            # Zweite Kerze (i)
            open2 = df['open'].iloc[i]
            close2 = df['close'].iloc[i]
            body2_size = abs(close2 - open2)
            
            # Bullish Engulfing
            if (close1 < open1 and  # Erste Kerze bearish
                close2 > open2 and  # Zweite Kerze bullish
                open2 < close1 and  # Zweite öffnet unter erstem Close
                close2 > open1 and  # Zweite schließt über erstem Open
                body2_size > body1_size * 1.2):  # Zweite Body mindestens 20% größer
                
                confidence = min(0.95, (body2_size / body1_size - 1) * 2)
                
                if confidence >= self.candlestick_patterns['engulfing_bullish']['min_confidence']:
                    pattern = DetectedPattern(
                        name='engulfing_bullish',
                        type=PatternType.CANDLESTICK,
                        signal=PatternSignal.BULLISH,
                        confidence=confidence,
                        strength=confidence * 0.95,
                        timeframe=timeframe,
                        symbol=symbol,
                        start_time=df.index[i-1],
                        end_time=df.index[i],
                        key_levels=[min(open1, close1), max(open2, close2)],
                        description='Bullish Engulfing - Starke bullish Umkehr',
                        metadata={'size_ratio': body2_size / body1_size}
                    )
                    patterns.append(pattern)
            
            # Bearish Engulfing
            elif (close1 > open1 and  # Erste Kerze bullish
                  close2 < open2 and  # Zweite Kerze bearish
                  open2 > close1 and  # Zweite öffnet über erstem Close
                  close2 < open1 and  # Zweite schließt unter erstem Open
                  body2_size > body1_size * 1.2):  # Zweite Body mindestens 20% größer
                
                confidence = min(0.95, (body2_size / body1_size - 1) * 2)
                
                if confidence >= self.candlestick_patterns['engulfing_bearish']['min_confidence']:
                    pattern = DetectedPattern(
                        name='engulfing_bearish',
                        type=PatternType.CANDLESTICK,
                        signal=PatternSignal.BEARISH,
                        confidence=confidence,
                        strength=confidence * 0.95,
                        timeframe=timeframe,
                        symbol=symbol,
                        start_time=df.index[i-1],
                        end_time=df.index[i],
                        key_levels=[max(open1, close1), min(open2, close2)],
                        description='Bearish Engulfing - Starke bearish Umkehr',
                        metadata={'size_ratio': body2_size / body1_size}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    # ==================== CHART PATTERNS ====================
    
    async def _detect_chart_patterns(self, candles: List[Candle], symbol: str, timeframe: str) -> List[DetectedPattern]:
        """📊 Chart Patterns erkennen"""
        try:
            patterns = []
            df = self._candles_to_dataframe(candles)
            
            # Double Top/Bottom
            patterns.extend(self._detect_double_patterns(df, symbol, timeframe))
            
            # Head and Shoulders
            patterns.extend(self._detect_head_shoulders(df, symbol, timeframe))
            
            # Triangles
            patterns.extend(self._detect_triangles(df, symbol, timeframe))
            
            # Flags
            patterns.extend(self._detect_flags(df, symbol, timeframe))
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Chart Pattern Detection: {e}")
            return []
    
    def _detect_double_patterns(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[DetectedPattern]:
        """📊 Double Top/Bottom erkennen"""
        patterns = []
        
        # Vereinfachte Double Top/Bottom Detection
        highs = df['high'].rolling(10, center=True).max()
        lows = df['low'].rolling(10, center=True).min()
        
        # Double Top Suche
        for i in range(20, len(df) - 20):
            if df['high'].iloc[i] == highs.iloc[i]:
                # Nach zweitem Peak suchen
                for j in range(i + 10, min(i + 50, len(df) - 10)):
                    if df['high'].iloc[j] == highs.iloc[j]:
                        # Prüfen ob Peaks ähnlich hoch sind
                        peak1 = df['high'].iloc[i]
                        peak2 = df['high'].iloc[j]
                        
                        if abs(peak1 - peak2) / peak1 <= 0.03:  # Maximal 3% Unterschied
                            # Valley zwischen Peaks finden
                            valley_start = i + 5
                            valley_end = j - 5
                            valley_low = df['low'].iloc[valley_start:valley_end].min()
                            
                            # Double Top validieren
                            if (peak1 - valley_low) / peak1 >= 0.05:  # Mindestens 5% Korrektur
                                confidence = 1 - abs(peak1 - peak2) / peak1 * 10
                                
                                if confidence >= self.chart_patterns['double_top']['min_confidence']:
                                    pattern = DetectedPattern(
                                        name='double_top',
                                        type=PatternType.CHART,
                                        signal=PatternSignal.BEARISH,
                                        confidence=confidence,
                                        strength=confidence * 0.85,
                                        timeframe=timeframe,
                                        symbol=symbol,
                                        start_time=df.index[i],
                                        end_time=df.index[j],
                                        key_levels=[float(peak1), float(valley_low)],
                                        description='Double Top - Bearish Reversal Pattern',
                                        metadata={'peak_difference': abs(peak1 - peak2) / peak1}
                                    )
                                    patterns.append(pattern)
                                    break
        
        return patterns
    
    # ==================== UTILITY METHODS ====================
    
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
    
    def _analyze_trend_context(self, df: pd.DataFrame, index: int, lookback: int = 5) -> float:
        """📈 Trend Kontext analysieren"""
        try:
            if index < lookback:
                return 0
            
            start_price = df['close'].iloc[index - lookback]
            end_price = df['close'].iloc[index - 1]
            
            return (end_price - start_price) / start_price
            
        except Exception:
            return 0
    
    # ==================== PLACEHOLDER METHODS ====================
    
    def _detect_morning_evening_star(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[DetectedPattern]:
        """⭐ Morning/Evening Star - Placeholder"""
        return []
    
    def _detect_three_soldiers_crows(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[DetectedPattern]:
        """👥 Three Soldiers/Crows - Placeholder"""
        return []
    
    def _detect_head_shoulders(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[DetectedPattern]:
        """👤 Head and Shoulders - Placeholder"""
        return []
    
    def _detect_triangles(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[DetectedPattern]:
        """📐 Triangles - Placeholder"""
        return []
    
    def _detect_flags(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[DetectedPattern]:
        """🏁 Flags - Placeholder"""
        return []
    
    async def _detect_price_action_patterns(self, candles: List[Candle], symbol: str, timeframe: str) -> List[DetectedPattern]:
        """📊 Price Action Patterns - Placeholder"""
        return []
    
    # ==================== PUBLIC METHODS ====================
    
    def get_patterns_for_symbol(self, symbol: str) -> List[DetectedPattern]:
        """🔍 Patterns für Symbol abrufen"""
        return self.detected_patterns.get(symbol, [])
    
    def get_recent_patterns(self, hours: int = 24) -> List[DetectedPattern]:
        """⏰ Aktuelle Patterns abrufen"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_patterns = []
        
        for patterns in self.detected_patterns.values():
            for pattern in patterns:
                if pattern.end_time >= cutoff:
                    recent_patterns.append(pattern)
        
        return sorted(recent_patterns, key=lambda x: x.confidence, reverse=True)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """📊 Performance Statistiken"""
        total_patterns = sum(len(patterns) for patterns in self.detected_patterns.values())
        
        return {
            'total_patterns_detected': self.patterns_detected,
            'cached_symbols': len(self.detected_patterns),
            'candlestick_patterns_count': len(self.candlestick_patterns),
            'chart_patterns_count': len(self.chart_patterns),
            'recent_patterns_24h': len(self.get_recent_patterns(24))
        }
    
    async def shutdown(self):
        """🛑 Pattern Recognition herunterfahren"""
        try:
            self.detected_patterns.clear()
            self.pattern_accuracy.clear()
            logger.info("✅ Pattern Recognition heruntergefahren")
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren: {e}")
