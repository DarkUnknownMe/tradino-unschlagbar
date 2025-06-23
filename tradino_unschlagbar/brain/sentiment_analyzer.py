#!/usr/bin/env python3
"""
💭 SENTIMENT ANALYSIS ENGINE - WELTKLASSE NLP
Echte Sentiment-Analyse für Trading-Entscheidungen
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import re
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
import aiohttp

class WorldClassSentimentEngine:
    """🌍 Weltklasse Sentiment Analysis für Trading"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.sentiment_history = []
        self.sources = {
            'news_apis': ['newsapi', 'alpha_vantage'],
            'social_media': ['twitter', 'reddit'],
            'crypto_specific': ['coindesk', 'cointelegraph']
        }
        
    async def analyze_comprehensive_sentiment(self, symbol: str) -> Dict[str, Any]:
        """🎯 Umfassende Sentiment-Analyse"""
        
        print(f"🔍 Analysiere Sentiment für {symbol}...")
        
        # Parallele Datensammlung
        tasks = [
            self._get_news_sentiment(symbol),
            self._get_social_sentiment(symbol),
            self._get_market_sentiment(symbol),
            self._get_options_sentiment(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        news_sentiment = results[0] if not isinstance(results[0], Exception) else {}
        social_sentiment = results[1] if not isinstance(results[1], Exception) else {}
        market_sentiment = results[2] if not isinstance(results[2], Exception) else {}
        options_sentiment = results[3] if not isinstance(results[3], Exception) else {}
        
        # Kombiniere alle Sentiment-Quellen
        composite_sentiment = self._calculate_composite_sentiment({
            'news': news_sentiment,
            'social': social_sentiment,
            'market': market_sentiment,
            'options': options_sentiment
        })
        
        # Trading Signal generieren
        trading_signal = self._generate_sentiment_trading_signal(composite_sentiment)
        
        result = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'composite_sentiment': composite_sentiment,
            'individual_sentiments': {
                'news': news_sentiment,
                'social': social_sentiment,
                'market': market_sentiment,
                'options': options_sentiment
            },
            'trading_signal': trading_signal,
            'confidence': self._calculate_confidence(composite_sentiment)
        }
        
        self.sentiment_history.append(result)
        return result
    
    async def _get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """📰 News Sentiment Analysis"""
        
        # Simuliere News-Daten (in Realität: echte APIs)
        sample_headlines = [
            f"{symbol} reaches new all-time high amid institutional adoption",
            f"Major cryptocurrency exchange lists {symbol} for trading",
            f"Regulatory concerns impact {symbol} trading volume",
            f"Technical analysis suggests {symbol} bullish momentum",
            f"Market volatility affects {symbol} price action"
        ]
        
        sentiments = []
        
        for headline in sample_headlines:
            # TextBlob Sentiment
            blob_sentiment = TextBlob(headline).sentiment.polarity
            
            # VADER Sentiment
            vader_scores = self.vader_analyzer.polarity_scores(headline)
            
            # Custom Financial Sentiment (vereinfacht)
            financial_score = self._analyze_financial_text(headline)
            
            sentiments.append({
                'text': headline,
                'textblob_sentiment': blob_sentiment,
                'vader_compound': vader_scores['compound'],
                'financial_sentiment': financial_score,
                'combined_sentiment': (blob_sentiment + vader_scores['compound'] + financial_score) / 3
            })
        
        avg_sentiment = np.mean([s['combined_sentiment'] for s in sentiments])
        
        return {
            'source': 'news',
            'sentiment_score': avg_sentiment,
            'sentiment_label': self._get_sentiment_label(avg_sentiment),
            'article_count': len(sentiments),
            'individual_sentiments': sentiments,
            'reliability': 0.8  # News haben hohe Reliability
        }
    
    async def _get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """📱 Social Media Sentiment"""
        
        # Simuliere Social Media Posts
        sample_posts = [
            f"Just bought more {symbol}! 🚀🚀🚀 #HODL",
            f"{symbol} looking bearish on the charts... might sell",
            f"Analysts predict {symbol} will moon soon 🌙",
            f"Not sure about {symbol} right now, market seems uncertain",
            f"{symbol} to the moon! Best crypto ever! 💎🙌"
        ]
        
        sentiments = []
        
        for post in sample_posts:
            # Emoji-aware Sentiment
            emoji_sentiment = self._analyze_emoji_sentiment(post)
            
            # Text Sentiment
            text_sentiment = self.vader_analyzer.polarity_scores(post)['compound']
            
            # Social Media specific analysis
            social_indicators = self._analyze_social_indicators(post)
            
            combined = (emoji_sentiment + text_sentiment + social_indicators) / 3
            
            sentiments.append({
                'text': post,
                'emoji_sentiment': emoji_sentiment,
                'text_sentiment': text_sentiment,
                'social_indicators': social_indicators,
                'combined_sentiment': combined
            })
        
        avg_sentiment = np.mean([s['combined_sentiment'] for s in sentiments])
        
        return {
            'source': 'social_media',
            'sentiment_score': avg_sentiment,
            'sentiment_label': self._get_sentiment_label(avg_sentiment),
            'post_count': len(sentiments),
            'engagement_score': np.random.uniform(0.6, 0.9),  # Simuliert
            'viral_potential': self._calculate_viral_potential(sentiments),
            'reliability': 0.6  # Social Media weniger reliable
        }
    
    async def _get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """📊 Market Data Sentiment"""
        
        # Simuliere Marktdaten-basierte Sentiment-Indikatoren
        
        # Fear & Greed Index (simuliert)
        fear_greed_index = np.random.uniform(20, 80)
        
        # Volume Analysis
        volume_sentiment = np.random.uniform(-0.3, 0.7)
        
        # Price Action Sentiment
        price_momentum = np.random.uniform(-0.5, 0.5)
        
        # Options Flow (simuliert)
        options_flow = np.random.uniform(-0.4, 0.6)
        
        # Technical Indicators Sentiment
        tech_sentiment = self._analyze_technical_sentiment()
        
        # Combine all market indicators
        market_sentiment = np.mean([
            (fear_greed_index - 50) / 50,  # Normalize Fear & Greed
            volume_sentiment,
            price_momentum,
            options_flow,
            tech_sentiment
        ])
        
        return {
            'source': 'market_data',
            'sentiment_score': market_sentiment,
            'sentiment_label': self._get_sentiment_label(market_sentiment),
            'components': {
                'fear_greed_index': fear_greed_index,
                'volume_sentiment': volume_sentiment,
                'price_momentum': price_momentum,
                'options_flow': options_flow,
                'technical_sentiment': tech_sentiment
            },
            'reliability': 0.9  # Market Data sehr reliable
        }
    
    async def _get_options_sentiment(self, symbol: str) -> Dict[str, Any]:
        """📈 Options Flow Sentiment"""
        
        # Simuliere Options-Daten
        put_call_ratio = np.random.uniform(0.6, 1.4)
        unusual_activity = np.random.uniform(0, 1)
        
        # Options Sentiment berechnen
        options_sentiment = 0.0
        
        # Put/Call Ratio Analysis
        if put_call_ratio < 0.8:
            options_sentiment += 0.3  # Bullish
        elif put_call_ratio > 1.2:
            options_sentiment -= 0.3  # Bearish
        
        # Unusual Activity
        if unusual_activity > 0.8:
            options_sentiment += 0.2  # Hohe Aktivität = Bullish
        
        return {
            'source': 'options_flow',
            'sentiment_score': options_sentiment,
            'sentiment_label': self._get_sentiment_label(options_sentiment),
            'put_call_ratio': put_call_ratio,
            'unusual_activity_score': unusual_activity,
            'reliability': 0.7
        }
    
    def _analyze_financial_text(self, text: str) -> float:
        """💰 Financial-specific Text Analysis"""
        
        # Financial Keywords
        bullish_keywords = ['bullish', 'moon', 'pump', 'gain', 'profit', 'high', 'buy', 'long']
        bearish_keywords = ['bearish', 'dump', 'crash', 'loss', 'sell', 'short', 'drop', 'fall']
        
        text_lower = text.lower()
        
        bullish_score = sum(1 for word in bullish_keywords if word in text_lower)
        bearish_score = sum(1 for word in bearish_keywords if word in text_lower)
        
        if bullish_score + bearish_score == 0:
            return 0.0
        
        return (bullish_score - bearish_score) / (bullish_score + bearish_score)
    
    def _analyze_emoji_sentiment(self, text: str) -> float:
        """😀 Emoji-based Sentiment Analysis"""
        
        positive_emojis = ['🚀', '🌙', '💎', '🙌', '📈', '💰', '🔥', '👍', '😊', '🎉']
        negative_emojis = ['📉', '💔', '😭', '😱', '👎', '🤮', '💸', '😰', '📻', '⚠️']
        
        positive_count = sum(1 for emoji in positive_emojis if emoji in text)
        negative_count = sum(1 for emoji in negative_emojis if emoji in text)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _analyze_social_indicators(self, text: str) -> float:
        """📱 Social Media Indicators"""
        
        indicators = {
            'HODL': 0.5, 'HODLing': 0.5,
            'diamond hands': 0.7, '💎🙌': 0.7,
            'paper hands': -0.5, '🧻🙌': -0.5,
            'to the moon': 0.8, 'moon': 0.6,
            'FOMO': 0.3, 'FUD': -0.6,
            'whale': 0.4, 'pump': 0.6, 'dump': -0.6
        }
        
        text_lower = text.lower()
        score = 0.0
        count = 0
        
        for indicator, value in indicators.items():
            if indicator.lower() in text_lower:
                score += value
                count += 1
        
        return score / max(count, 1)
    
    def _analyze_technical_sentiment(self) -> float:
        """📊 Technical Analysis Sentiment"""
        
        # Simuliere technische Indikatoren
        rsi = np.random.uniform(20, 80)
        macd_signal = np.random.uniform(-1, 1)
        bollinger_position = np.random.uniform(0, 1)
        
        # Technical Sentiment Score
        tech_score = 0.0
        
        # RSI Analysis
        if rsi < 30:
            tech_score += 0.3  # Oversold = Bullish
        elif rsi > 70:
            tech_score -= 0.3  # Overbought = Bearish
        
        # MACD Signal
        tech_score += macd_signal * 0.4
        
        # Bollinger Bands
        tech_score += (bollinger_position - 0.5) * 0.6
        
        return np.clip(tech_score, -1, 1)
    
    def _calculate_composite_sentiment(self, sentiments: Dict) -> Dict[str, Any]:
        """🔄 Kombiniere alle Sentiment-Quellen"""
        
        # Gewichtung basierend auf Reliability
        weights = {
            'news': 0.3,
            'social': 0.2,
            'market': 0.4,
            'options': 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for source, weight in weights.items():
            sentiment_data = sentiments.get(source, {})
            if sentiment_data and 'sentiment_score' in sentiment_data:
                reliability = sentiment_data.get('reliability', 1.0)
                adjusted_weight = weight * reliability
                weighted_score += sentiment_data['sentiment_score'] * adjusted_weight
                total_weight += adjusted_weight
        
        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        return {
            'composite_score': final_score,
            'composite_label': self._get_sentiment_label(final_score),
            'strength': abs(final_score),
            'weights_used': weights,
            'total_reliability': total_weight / sum(weights.values())
        }
    
    def _generate_sentiment_trading_signal(self, composite_sentiment: Dict) -> Dict[str, Any]:
        """💹 Generiere Trading Signal aus Sentiment"""
        
        score = composite_sentiment['composite_score']
        strength = composite_sentiment['strength']
        
        # Signal Logic
        if score > 0.3 and strength > 0.5:
            signal = 'STRONG_BUY'
            position_size = min(1.0, strength * 1.2)
        elif score > 0.1:
            signal = 'BUY'
            position_size = min(0.8, strength)
        elif score < -0.3 and strength > 0.5:
            signal = 'STRONG_SELL'
            position_size = min(1.0, strength * 1.2)
        elif score < -0.1:
            signal = 'SELL'
            position_size = min(0.8, strength)
        else:
            signal = 'HOLD'
            position_size = 0.0
        
        return {
            'signal': signal,
            'position_size_factor': position_size,
            'sentiment_score': score,
            'confidence': strength,
            'time_horizon': 'short_term'  # Sentiment meist kurzfristig
        }
    
    def _calculate_viral_potential(self, sentiments: List) -> float:
        """🦠 Berechne Viral Potential"""
        
        emoji_count = sum(1 for s in sentiments if any(e in s['text'] for e in ['🚀', '🌙', '💎']))
        extreme_sentiment = sum(1 for s in sentiments if abs(s['combined_sentiment']) > 0.7)
        
        return min(1.0, (emoji_count + extreme_sentiment) / len(sentiments))
    
    def _calculate_confidence(self, composite_sentiment: Dict) -> float:
        """📊 Berechne Gesamtconfidence"""
        
        strength = composite_sentiment['strength']
        reliability = composite_sentiment['total_reliability']
        
        return min(1.0, strength * reliability * 1.2)
    
    def _get_sentiment_label(self, score: float) -> str:
        """🏷️ Sentiment Label"""
        
        if score > 0.5:
            return "Very Bullish"
        elif score > 0.2:
            return "Bullish"
        elif score > -0.2:
            return "Neutral"
        elif score > -0.5:
            return "Bearish"
        else:
            return "Very Bearish"
    
    def get_sentiment_trends(self, days: int = 7) -> Dict[str, Any]:
        """📈 Sentiment Trends Analysis"""
        
        if len(self.sentiment_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_sentiments = self.sentiment_history[-days:] if len(self.sentiment_history) >= days else self.sentiment_history
        scores = [s['composite_sentiment']['composite_score'] for s in recent_sentiments]
        
        # Trend berechnen
        if len(scores) >= 2:
            trend = 'improving' if scores[-1] > scores[0] else 'declining'
            volatility = np.std(scores)
            avg_sentiment = np.mean(scores)
        else:
            trend = 'stable'
            volatility = 0
            avg_sentiment = scores[0] if scores else 0
        
        return {
            'trend': trend,
            'avg_sentiment': avg_sentiment,
            'volatility': volatility,
            'data_points': len(scores)
        }

# Legacy Compatibility für bestehende Integration
class SentimentAnalyzer(WorldClassSentimentEngine):
    """Legacy Wrapper für Backwards Compatibility"""
    
    def __init__(self):
        super().__init__()
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Legacy method für einfache Sentiment-Analyse"""
        vader_scores = self.vader_analyzer.polarity_scores(text)
        blob_sentiment = TextBlob(text).sentiment.polarity
        financial_score = self._analyze_financial_text(text)
        
        combined = (vader_scores['compound'] + blob_sentiment + financial_score) / 3
        
        return {
            'sentiment_score': combined,
            'sentiment_label': self._get_sentiment_label(combined),
            'vader_compound': vader_scores['compound'],
            'textblob_polarity': blob_sentiment,
            'financial_sentiment': financial_score
        }

# Verwendungsbeispiel
if __name__ == "__main__":
    async def main():
        engine = WorldClassSentimentEngine()
        
        # Analysiere Sentiment für BTC
        result = await engine.analyze_comprehensive_sentiment("BTC")
        
        print("💭 COMPREHENSIVE SENTIMENT ANALYSIS")
        print("=" * 50)
        print(f"Symbol: {result['symbol']}")
        print(f"Composite Sentiment: {result['composite_sentiment']['composite_label']}")
        print(f"Score: {result['composite_sentiment']['composite_score']:.3f}")
        print(f"Trading Signal: {result['trading_signal']['signal']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        print("\n📊 Individual Sources:")
        for source, data in result['individual_sentiments'].items():
            if data:
                print(f"  {source}: {data.get('sentiment_label', 'N/A')} ({data.get('sentiment_score', 0):.3f})")
    
    asyncio.run(main()) 