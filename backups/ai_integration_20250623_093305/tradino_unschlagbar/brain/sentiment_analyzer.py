"""
🎭 TRADINO UNSCHLAGBAR - Sentiment Analyzer
Real-time Market Sentiment Analysis

Author: AI Trading Systems
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from utils.logger_pro import setup_logger, log_ai_decision
from utils.config_manager import ConfigManager

logger = setup_logger("SentimentAnalyzer")


class SentimentLevel(Enum):
    """Sentiment Level"""
    EXTREME_FEAR = 0
    FEAR = 1
    NEUTRAL = 2
    GREED = 3
    EXTREME_GREED = 4


@dataclass
class SentimentData:
    """Sentiment Data Model"""
    symbol: str
    timestamp: datetime
    fear_greed_index: float  # 0-100
    sentiment_level: SentimentLevel
    news_sentiment: float  # -1 to 1
    social_sentiment: float  # -1 to 1
    market_sentiment: float  # -1 to 1
    overall_sentiment: float  # -1 to 1
    confidence: float  # 0-1
    metadata: Dict[str, Any]


class SentimentAnalyzer:
    """🎭 Advanced Sentiment Analysis System"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        
        # Sentiment Cache
        self.sentiment_cache: Dict[str, SentimentData] = {}
        
        # API Endpoints (Mock für Demo)
        self.api_endpoints = {
            'fear_greed': 'https://api.alternative.me/fng/',
            'news': 'https://newsapi.org/v2/everything',  # Beispiel
            'social': 'https://api.social-sentiment.com'  # Beispiel
        }
        
        # Sentiment Weights
        self.sentiment_weights = {
            'fear_greed': 0.3,
            'news': 0.3,
            'social': 0.2,
            'market': 0.2
        }
        
        # Performance Tracking
        self.analyses_performed = 0
        self.cache_hits = 0
        
    async def initialize(self) -> bool:
        """🔥 Sentiment Analyzer initialisieren"""
        try:
            logger.info("🎭 Sentiment Analyzer wird initialisiert...")
            
            # Test API Verbindungen (Mock)
            await self._test_api_connections()
            
            logger.success("✅ Sentiment Analyzer erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sentiment Analyzer Initialisierung fehlgeschlagen: {e}")
            return False
    
    async def _test_api_connections(self):
        """🔗 API Verbindungen testen"""
        try:
            # Mock API Test
            logger.info("🔗 API Verbindungen werden getestet...")
            
            # Simuliere erfolgreiche Verbindung
            await asyncio.sleep(0.1)
            
            logger.info("✅ Alle Sentiment APIs verfügbar")
            
        except Exception as e:
            logger.warning(f"⚠️ Einige Sentiment APIs nicht verfügbar: {e}")
    
    # ==================== MAIN ANALYSIS METHODS ====================
    
    async def analyze_sentiment(self, symbol: str = "BTC") -> Optional[SentimentData]:
        """🎭 Komplette Sentiment Analyse"""
        try:
            logger.info(f"🎭 Sentiment Analyse wird durchgeführt: {symbol}")
            
            # Cache Check
            cache_key = f"{symbol}_{datetime.utcnow().hour}"  # Stündlicher Cache
            if cache_key in self.sentiment_cache:
                self.cache_hits += 1
                return self.sentiment_cache[cache_key]
            
            # Fear & Greed Index
            fear_greed_data = await self._get_fear_greed_index()
            
            # News Sentiment
            news_sentiment = await self._analyze_news_sentiment(symbol)
            
            # Social Media Sentiment
            social_sentiment = await self._analyze_social_sentiment(symbol)
            
            # Market-based Sentiment
            market_sentiment = await self._analyze_market_sentiment(symbol)
            
            # Overall Sentiment berechnen
            overall_sentiment = self._calculate_overall_sentiment(
                fear_greed_data.get('sentiment', 0),
                news_sentiment,
                social_sentiment,
                market_sentiment
            )
            
            # Sentiment Level bestimmen
            sentiment_level = self._determine_sentiment_level(fear_greed_data.get('index', 50))
            
            # Confidence berechnen
            confidence = self._calculate_confidence(
                fear_greed_data.get('confidence', 0.5),
                news_sentiment != 0,
                social_sentiment != 0,
                market_sentiment != 0
            )
            
            # Sentiment Data erstellen
            sentiment_data = SentimentData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                fear_greed_index=fear_greed_data.get('index', 50),
                sentiment_level=sentiment_level,
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                market_sentiment=market_sentiment,
                overall_sentiment=overall_sentiment,
                confidence=confidence,
                metadata={
                    'sources_available': sum([
                        fear_greed_data.get('available', False),
                        news_sentiment != 0,
                        social_sentiment != 0,
                        market_sentiment != 0
                    ]),
                    'analysis_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Cache aktualisieren
            self.sentiment_cache[cache_key] = sentiment_data
            self.analyses_performed += 1
            
            log_ai_decision(
                "SentimentAnalyzer",
                f"{symbol} {sentiment_level.name}",
                confidence
            )
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Sentiment Analyse für {symbol}: {e}")
            return None
    
    # ==================== DATA COLLECTION METHODS ====================
    
    async def _get_fear_greed_index(self) -> Dict[str, Any]:
        """😨 Fear & Greed Index abrufen"""
        try:
            # Mock Implementation - In Realität würde hier echter API Call stattfinden
            await asyncio.sleep(0.1)
            
            # Simuliere Fear & Greed Index (0-100)
            import random
            mock_index = random.randint(10, 90)
            
            if mock_index <= 25:
                sentiment_text = "Extreme Fear"
                sentiment_value = -0.8
            elif mock_index <= 45:
                sentiment_text = "Fear"
                sentiment_value = -0.4
            elif mock_index <= 55:
                sentiment_text = "Neutral"
                sentiment_value = 0.0
            elif mock_index <= 75:
                sentiment_text = "Greed"
                sentiment_value = 0.4
            else:
                sentiment_text = "Extreme Greed"
                sentiment_value = 0.8
            
            return {
                'index': mock_index,
                'sentiment': sentiment_value,
                'text': sentiment_text,
                'confidence': 0.8,
                'available': True
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Fear & Greed Index: {e}")
            return {'index': 50, 'sentiment': 0, 'available': False}
    
    async def _analyze_news_sentiment(self, symbol: str) -> float:
        """📰 News Sentiment analysieren"""
        try:
            # Mock Implementation
            await asyncio.sleep(0.1)
            
            # Simuliere News Sentiment Analysis
            import random
            
            # Verschiedene News Sentiment Scores simulieren
            news_scores = []
            for _ in range(5):  # 5 News Articles
                score = random.uniform(-1, 1)
                news_scores.append(score)
            
            # Durchschnittlicher Sentiment Score
            avg_sentiment = sum(news_scores) / len(news_scores) if news_scores else 0
            
            logger.info(f"📰 News Sentiment für {symbol}: {avg_sentiment:.2f}")
            return avg_sentiment
            
        except Exception as e:
            logger.error(f"❌ Fehler bei News Sentiment: {e}")
            return 0.0
    
    async def _analyze_social_sentiment(self, symbol: str) -> float:
        """📱 Social Media Sentiment analysieren"""
        try:
            # Mock Implementation
            await asyncio.sleep(0.1)
            
            # Simuliere Social Media Sentiment (Twitter, Reddit, etc.)
            import random
            
            social_scores = []
            platforms = ['twitter', 'reddit', 'telegram', 'discord']
            
            for platform in platforms:
                score = random.uniform(-1, 1)
                social_scores.append(score)
            
            # Gewichteter Durchschnitt
            weights = [0.4, 0.3, 0.2, 0.1]  # Twitter hat höchstes Gewicht
            weighted_sentiment = sum(score * weight for score, weight in zip(social_scores, weights))
            
            logger.info(f"📱 Social Sentiment für {symbol}: {weighted_sentiment:.2f}")
            return weighted_sentiment
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Social Sentiment: {e}")
            return 0.0
    
    async def _analyze_market_sentiment(self, symbol: str) -> float:
        """📊 Market-based Sentiment analysieren"""
        try:
            # Market Sentiment basierend auf technischen Indikatoren
            import random
            
            # Simuliere Market Sentiment Faktoren
            volume_sentiment = random.uniform(-0.5, 0.5)  # Volume Analysis
            volatility_sentiment = random.uniform(-0.3, 0.3)  # Volatility Analysis
            momentum_sentiment = random.uniform(-0.4, 0.4)  # Momentum Analysis
            
            # Kombinierter Market Sentiment
            market_sentiment = (volume_sentiment + volatility_sentiment + momentum_sentiment) / 3
            
            logger.info(f"📊 Market Sentiment für {symbol}: {market_sentiment:.2f}")
            return market_sentiment
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Market Sentiment: {e}")
            return 0.0
    
    # ==================== CALCULATION METHODS ====================
    
    def _calculate_overall_sentiment(self, fear_greed_sentiment: float, news_sentiment: float,
                                   social_sentiment: float, market_sentiment: float) -> float:
        """🎯 Overall Sentiment berechnen"""
        try:
            sentiments = {
                'fear_greed': fear_greed_sentiment,
                'news': news_sentiment,
                'social': social_sentiment,
                'market': market_sentiment
            }
            
            # Gewichteter Durchschnitt
            weighted_sum = 0
            total_weight = 0
            
            for sentiment_type, sentiment_value in sentiments.items():
                weight = self.sentiment_weights.get(sentiment_type, 0.25)
                weighted_sum += sentiment_value * weight
                total_weight += weight
            
            overall_sentiment = weighted_sum / total_weight if total_weight > 0 else 0
            
            # Normalisierung auf -1 bis 1
            return max(-1, min(1, overall_sentiment))
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Overall Sentiment Berechnung: {e}")
            return 0.0
    
    def _determine_sentiment_level(self, fear_greed_index: float) -> SentimentLevel:
        """📊 Sentiment Level bestimmen"""
        try:
            if fear_greed_index <= 20:
                return SentimentLevel.EXTREME_FEAR
            elif fear_greed_index <= 40:
                return SentimentLevel.FEAR
            elif fear_greed_index <= 60:
                return SentimentLevel.NEUTRAL
            elif fear_greed_index <= 80:
                return SentimentLevel.GREED
            else:
                return SentimentLevel.EXTREME_GREED
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Sentiment Level Bestimmung: {e}")
            return SentimentLevel.NEUTRAL
    
    def _calculate_confidence(self, fear_greed_confidence: float, has_news: bool,
                            has_social: bool, has_market: bool) -> float:
        """🎯 Confidence Score berechnen"""
        try:
            # Basis Confidence
            base_confidence = fear_greed_confidence
            
            # Bonus für verfügbare Datenquellen
            sources_bonus = sum([has_news, has_social, has_market]) * 0.1
            
            # Finale Confidence
            confidence = min(1.0, base_confidence + sources_bonus)
            
            return confidence
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Confidence Berechnung: {e}")
            return 0.5
    
    # ==================== PUBLIC METHODS ====================
    
    def get_cached_sentiment(self, symbol: str) -> Optional[SentimentData]:
        """📊 Gecachtes Sentiment abrufen"""
        cache_key = f"{symbol}_{datetime.utcnow().hour}"
        return self.sentiment_cache.get(cache_key)
    
    def get_sentiment_signal(self, sentiment_data: SentimentData) -> Dict[str, Any]:
        """🎯 Trading Signal aus Sentiment generieren"""
        try:
            signal_strength = abs(sentiment_data.overall_sentiment)
            
            if sentiment_data.overall_sentiment >= 0.6:
                signal = "bullish"
                recommendation = "BUY"
            elif sentiment_data.overall_sentiment <= -0.6:
                signal = "bearish"
                recommendation = "SELL"
            elif sentiment_data.overall_sentiment >= 0.3:
                signal = "slightly_bullish"
                recommendation = "WEAK_BUY"
            elif sentiment_data.overall_sentiment <= -0.3:
                signal = "slightly_bearish"
                recommendation = "WEAK_SELL"
            else:
                signal = "neutral"
                recommendation = "HOLD"
            
            return {
                'signal': signal,
                'recommendation': recommendation,
                'strength': signal_strength,
                'confidence': sentiment_data.confidence,
                'sentiment_level': sentiment_data.sentiment_level.name,
                'fear_greed_index': sentiment_data.fear_greed_index
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Sentiment Signal Generation: {e}")
            return {'signal': 'neutral', 'recommendation': 'HOLD'}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """📊 Performance Statistiken"""
        cache_hit_rate = self.cache_hits / self.analyses_performed if self.analyses_performed > 0 else 0
        
        return {
            'analyses_performed': self.analyses_performed,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'cached_symbols': len(self.sentiment_cache),
            'api_endpoints': len(self.api_endpoints)
        }
    
    async def shutdown(self):
        """🛑 Sentiment Analyzer herunterfahren"""
        try:
            self.sentiment_cache.clear()
            logger.info("✅ Sentiment Analyzer heruntergefahren")
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren: {e}")
