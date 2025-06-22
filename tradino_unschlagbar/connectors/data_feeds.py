"""
📊 TRADINO UNSCHLAGBAR - Data Feeds Manager
Multi-Source Market Data Integration mit Real-time Feeds

Author: AI Trading Systems
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import websockets

from connectors.bitget_pro import BitgetPro
from models.market_models import Candle, MarketData, TechnicalIndicators
from utils.logger_pro import setup_logger
from utils.config_manager import ConfigManager
from utils.helpers import generate_id

logger = setup_logger("DataFeeds")


class DataSource(Enum):
    """Datenquellen"""
    BITGET = "bitget"
    WEBSOCKET = "websocket"
    CACHE = "cache"


@dataclass
class DataFeed:
    """Data Feed Model"""
    id: str
    symbol: str
    timeframe: str
    source: DataSource
    last_update: datetime
    data_points: int
    quality_score: float
    is_active: bool = True


class DataFeedsManager:
    """📊 Multi-Source Market Data Manager"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.bitget: Optional[BitgetPro] = None
        
        # Data Feeds Registry
        self.active_feeds: Dict[str, DataFeed] = {}
        self.data_cache: Dict[str, List[Candle]] = {}
        
        # WebSocket Connections
        self.ws_connections: Dict[str, Any] = {}
        
        # Subscriptions & Callbacks
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.data_callbacks: Dict[str, Callable] = {}
        
        # Data Quality Monitoring
        self.quality_metrics = {
            'total_updates': 0,
            'failed_updates': 0,
            'avg_latency': 0,
            'data_gaps': 0
        }
        
        # Background Tasks
        self._feed_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
    
    async def initialize(self) -> bool:
        """🔥 Data Feeds Manager initialisieren"""
        try:
            logger.info("📊 Data Feeds Manager wird initialisiert...")
            
            # Bitget Connector initialisieren
            self.bitget = BitgetPro(self.config)
            if not await self.bitget.initialize():
                logger.error("❌ Bitget Initialisierung fehlgeschlagen")
                return False
            
            # Standard Feeds konfigurieren
            await self._setup_default_feeds()
            
            # WebSocket Verbindungen starten
            await self._start_websocket_feeds()
            
            self._running = True
            logger.success("✅ Data Feeds Manager erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data Feeds Manager Initialisierung fehlgeschlagen: {e}")
            return False
    
    # ==================== FEED MANAGEMENT ====================
    
    async def _setup_default_feeds(self):
        """🔧 Standard Data Feeds konfigurieren"""
        try:
            # Top Trading Pairs aus Config
            symbols = self.config.get('trading.symbols', ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
            timeframes = self.config.get('data.timeframes', ['1m', '5m', '15m', '1h'])
            
            for symbol in symbols:
                for timeframe in timeframes:
                    feed_id = f"{symbol}_{timeframe}"
                    
                    # Data Feed registrieren
                    feed = DataFeed(
                        id=feed_id,
                        symbol=symbol,
                        timeframe=timeframe,
                        source=DataSource.BITGET,
                        last_update=datetime.utcnow(),
                        data_points=0,
                        quality_score=1.0
                    )
                    
                    self.active_feeds[feed_id] = feed
                    self.data_cache[feed_id] = []
                    
                    # Initial Data laden
                    await self._load_initial_data(feed_id)
            
            logger.info(f"🔧 {len(self.active_feeds)} Data Feeds konfiguriert")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Setup der Default Feeds: {e}")
    
    async def _load_initial_data(self, feed_id: str):
        """📥 Initial Historical Data laden"""
        try:
            if feed_id not in self.active_feeds:
                return
                
            feed = self.active_feeds[feed_id]
            
            # Historical Candles von Bitget abrufen
            candles = await self.bitget.get_candles(
                symbol=feed.symbol,
                timeframe=feed.timeframe,
                limit=200  # Letzte 200 Candles
            )
            
            if candles:
                self.data_cache[feed_id] = candles
                feed.data_points = len(candles)
                feed.last_update = datetime.utcnow()
                
                logger.info(f"📥 Initial Data geladen: {feed_id} ({len(candles)} Candles)")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden der Initial Data für {feed_id}: {e}")
    
    async def _start_websocket_feeds(self):
        """🌐 WebSocket Feeds starten"""
        try:
            # WebSocket URL aus Config
            ws_url = self.config.get('data.websocket_url', 'wss://ws.bitget.com/mix/v1/stream')
            
            if not ws_url:
                logger.warning("⚠️ Keine WebSocket URL konfiguriert")
                return
            
            # WebSocket Task für jeden Feed starten
            for feed_id in self.active_feeds:
                task = asyncio.create_task(self._websocket_feed_handler(feed_id))
                self._feed_tasks[feed_id] = task
            
            logger.info(f"🌐 {len(self._feed_tasks)} WebSocket Feeds gestartet")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Starten der WebSocket Feeds: {e}")
    
    async def _websocket_feed_handler(self, feed_id: str):
        """🔄 WebSocket Feed Handler für einzelnen Feed"""
        try:
            if feed_id not in self.active_feeds:
                return
                
            feed = self.active_feeds[feed_id]
            ws_url = self.config.get('data.websocket_url')
            
            while self._running and feed.is_active:
                try:
                    async with websockets.connect(ws_url) as websocket:
                        # Subscribe Message senden
                        subscribe_msg = {
                            "op": "subscribe",
                            "args": [f"candle{feed.timeframe}:{feed.symbol}"]
                        }
                        await websocket.send(json.dumps(subscribe_msg))
                        
                        # Messages empfangen
                        async for message in websocket:
                            await self._process_websocket_message(feed_id, message)
                            
                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"⚠️ WebSocket Verbindung geschlossen für {feed_id}")
                    await asyncio.sleep(5)  # Reconnect Delay
                    
                except Exception as e:
                    logger.error(f"❌ WebSocket Fehler für {feed_id}: {e}")
                    await asyncio.sleep(10)
                    
        except Exception as e:
            logger.error(f"❌ WebSocket Handler Fehler für {feed_id}: {e}")
    
    async def _process_websocket_message(self, feed_id: str, message: str):
        """📨 WebSocket Message verarbeiten"""
        try:
            data = json.loads(message)
            
            # Candle Data extrahieren
            if 'data' in data and isinstance(data['data'], list):
                for candle_data in data['data']:
                    candle = self._parse_candle_data(candle_data)
                    if candle:
                        await self._update_feed_data(feed_id, candle)
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Verarbeiten der WebSocket Message: {e}")
    
    def _parse_candle_data(self, candle_data: List) -> Optional[Candle]:
        """🔄 Candle Data parsen"""
        try:
            if len(candle_data) >= 6:
                return Candle(
                    timestamp=datetime.fromtimestamp(int(candle_data[0]) / 1000),
                    open=float(candle_data[1]),
                    high=float(candle_data[2]),
                    low=float(candle_data[3]),
                    close=float(candle_data[4]),
                    volume=float(candle_data[5])
                )
        except Exception as e:
            logger.error(f"❌ Fehler beim Parsen der Candle Data: {e}")
        return None
    
    # ==================== DATA ACCESS ====================
    
    async def get_latest_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[Candle]:
        """📊 Neueste Candles abrufen"""
        try:
            feed_id = f"{symbol}_{timeframe}"
            
            if feed_id in self.data_cache:
                candles = self.data_cache[feed_id][-limit:]
                logger.debug(f"📊 {len(candles)} Candles abgerufen für {feed_id}")
                return candles
            else:
                # Fallback: Direkt von Bitget abrufen
                logger.warning(f"⚠️ Feed nicht gefunden, lade von Bitget: {feed_id}")
                return await self.bitget.get_candles(symbol, timeframe, limit)
                
        except Exception as e:
            logger.error(f"❌ Fehler beim Abrufen der Candles: {e}")
            return []
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """💰 Aktueller Preis abrufen"""
        try:
            # Versuche aus 1m Cache
            feed_id = f"{symbol}_1m"
            if feed_id in self.data_cache and self.data_cache[feed_id]:
                latest_candle = self.data_cache[feed_id][-1]
                return latest_candle.close
            
            # Fallback: Direkt von Bitget
            market_data = await self.bitget.get_market_data(symbol)
            return market_data.current_price if market_data else None
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Abrufen des aktuellen Preises: {e}")
            return None
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """📈 Vollständige Market Data abrufen"""
        try:
            # Candles aus verschiedenen Timeframes sammeln
            candles_1m = await self.get_latest_candles(symbol, '1m', 50)
            candles_5m = await self.get_latest_candles(symbol, '5m', 50)
            candles_1h = await self.get_latest_candles(symbol, '1h', 24)
            
            if not candles_1m:
                return None
            
            latest_candle = candles_1m[-1]
            
            # Technical Indicators berechnen (vereinfacht)
            indicators = self._calculate_basic_indicators(candles_1m)
            
            return MarketData(
                symbol=symbol,
                current_price=latest_candle.close,
                timestamp=latest_candle.timestamp,
                volume_24h=sum(c.volume for c in candles_1h),
                price_change_24h=((latest_candle.close - candles_1h[0].open) / candles_1h[0].open) * 100 if candles_1h else 0,
                high_24h=max(c.high for c in candles_1h) if candles_1h else latest_candle.high,
                low_24h=min(c.low for c in candles_1h) if candles_1h else latest_candle.low,
                indicators=indicators
            )
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Abrufen der Market Data: {e}")
            return None
    
    def _calculate_basic_indicators(self, candles: List[Candle]) -> TechnicalIndicators:
        """📊 Basic Technical Indicators berechnen"""
        try:
            if len(candles) < 20:
                return TechnicalIndicators()
            
            closes = [c.close for c in candles]
            
            # Simple Moving Averages
            sma_20 = sum(closes[-20:]) / 20
            sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else sma_20
            
            # RSI (vereinfacht)
            gains = []
            losses = []
            for i in range(1, min(15, len(closes))):
                change = closes[i] - closes[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0.001
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return TechnicalIndicators(
                sma_20=sma_20,
                sma_50=sma_50,
                rsi=rsi,
                volume_sma=sum(c.volume for c in candles[-20:]) / 20
            )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Indicator Berechnung: {e}")
            return TechnicalIndicators()
    
    # ==================== SUBSCRIPTION SYSTEM ====================
    
    def subscribe_to_symbol(self, symbol: str, timeframe: str, callback: Callable):
        """📡 Symbol Subscription"""
        try:
            feed_id = f"{symbol}_{timeframe}"
            
            if feed_id not in self.subscriptions:
                self.subscriptions[feed_id] = []
            
            self.subscriptions[feed_id].append(callback)
            logger.info(f"📡 Subscription hinzugefügt: {feed_id}")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Symbol Subscription: {e}")
    
    async def _update_feed_data(self, feed_id: str, new_candle: Candle):
        """🔄 Feed Data aktualisieren"""
        try:
            if feed_id not in self.data_cache:
                self.data_cache[feed_id] = []
            
            # Neue Candle hinzufügen
            self.data_cache[feed_id].append(new_candle)
            
            # Cache Größe begrenzen (letzte 1000 Candles)
            if len(self.data_cache[feed_id]) > 1000:
                self.data_cache[feed_id] = self.data_cache[feed_id][-1000:]
            
            # Feed Status aktualisieren
            if feed_id in self.active_feeds:
                feed = self.active_feeds[feed_id]
                feed.last_update = datetime.utcnow()
                feed.data_points = len(self.data_cache[feed_id])
            
            # Callbacks benachrichtigen
            if feed_id in self.subscriptions:
                for callback in self.subscriptions[feed_id]:
                    try:
                        await callback(new_candle)
                    except Exception as e:
                        logger.error(f"❌ Callback Fehler für {feed_id}: {e}")
            
            # Quality Metrics aktualisieren
            self.quality_metrics['total_updates'] += 1
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Aktualisieren der Feed Data: {e}")
            self.quality_metrics['failed_updates'] += 1
    
    # ==================== MONITORING & STATISTICS ====================
    
    def get_feed_statistics(self) -> Dict[str, Any]:
        """📊 Feed Statistiken abrufen"""
        try:
            active_feeds = sum(1 for feed in self.active_feeds.values() if feed.is_active)
            total_data_points = sum(feed.data_points for feed in self.active_feeds.values())
            avg_quality = sum(feed.quality_score for feed in self.active_feeds.values()) / len(self.active_feeds) if self.active_feeds else 0
            
            return {
                'total_feeds': len(self.active_feeds),
                'active_feeds': active_feeds,
                'total_data_points': total_data_points,
                'average_quality_score': avg_quality,
                'quality_metrics': self.quality_metrics,
                'cache_size': sum(len(data) for data in self.data_cache.values()),
                'websocket_connections': len(self.ws_connections)
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Abrufen der Feed Statistiken: {e}")
            return {}
    
    def get_feed_health(self) -> Dict[str, Any]:
        """🏥 Feed Health Check"""
        try:
            now = datetime.utcnow()
            healthy_feeds = 0
            stale_feeds = []
            
            for feed_id, feed in self.active_feeds.items():
                time_since_update = (now - feed.last_update).total_seconds()
                
                if time_since_update < 300:  # < 5 Minuten
                    healthy_feeds += 1
                else:
                    stale_feeds.append({
                        'feed_id': feed_id,
                        'last_update': feed.last_update,
                        'seconds_stale': time_since_update
                    })
            
            return {
                'healthy_feeds': healthy_feeds,
                'total_feeds': len(self.active_feeds),
                'health_percentage': (healthy_feeds / len(self.active_feeds)) * 100 if self.active_feeds else 0,
                'stale_feeds': stale_feeds
            }
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Feed Health Check: {e}")
            return {}
    
    # ==================== SHUTDOWN ====================
    
    async def shutdown(self):
        """🛑 Data Feeds Manager herunterfahren"""
        try:
            self._running = False
            
            # WebSocket Tasks stoppen
            for task in self._feed_tasks.values():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # WebSocket Verbindungen schließen
            for ws in self.ws_connections.values():
                try:
                    await ws.close()
                except:
                    pass
            
            # Bitget Connector herunterfahren
            if self.bitget:
                await self.bitget.shutdown()
            
            logger.info("✅ Data Feeds Manager heruntergefahren")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren des Data Feeds Managers: {e}")
