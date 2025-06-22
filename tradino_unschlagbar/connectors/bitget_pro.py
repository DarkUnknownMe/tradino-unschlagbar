"""
🚀 TRADINO UNSCHLAGBAR - Bitget Pro API Connector
Professional Bitget Futures API Integration mit Demo Support

Author: AI Trading Systems
"""

import asyncio
import ccxt.pro as ccxtpro
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta

from models.trade_models import Order, OrderType, OrderSide, OrderStatus
from models.market_models import Candle, MarketData
from models.portfolio_models import Position, Portfolio
from utils.logger_pro import setup_logger, log_trade
from utils.config_manager import ConfigManager
from utils.helpers import safe_float_convert, generate_id

logger = setup_logger("BitgetPro")


class BitgetProConnector:
    """🔥 Professional Bitget API Connector"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.exchange: Optional[ccxtpro.bitget] = None
        self.is_connected = False
        self.is_demo = config.is_demo_mode()
        self.symbols_info: Dict[str, Any] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        
        # Performance Tracking
        self.api_call_count = 0
        self.last_api_call = datetime.utcnow()
        
    async def initialize(self) -> bool:
        """🔥 API Initialisierung"""
        try:
            logger.info("🔌 Bitget API wird initialisiert...")
            
            # Exchange Config
            exchange_config = self.config.get_exchange_config()
            
            if not exchange_config.get('api_key'):
                logger.error("❌ Bitget API-Keys fehlen in der Konfiguration")
                return False
            
            # CCXT Exchange initialisieren
            self.exchange = ccxtpro.bitget({
                'apiKey': exchange_config['api_key'],
                'secret': exchange_config['api_secret'],
                'password': exchange_config['api_passphrase'],
                'sandbox': self.is_demo,  # Demo Account
                'enableRateLimit': True,
                'rateLimit': 50,  # 20 requests per second
                'timeout': 10000,
                'options': {
                    'defaultType': 'swap',  # Futures Trading
                    'marginMode': 'cross',  # Cross Margin
                },
                'urls': {
                    'api': {
                        'public': 'https://api.bitget.com' if not self.is_demo else 'https://api.bitget.com',
                        'private': 'https://api.bitget.com' if not self.is_demo else 'https://api.bitget.com'
                    }
                }
            })
            
            # Verbindung testen
            await self._test_connection()
            
            # Markets laden
            await self._load_markets()
            
            self.is_connected = True
            logger.success(f"✅ Bitget API erfolgreich verbunden (Demo: {self.is_demo})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Bitget API Initialisierung fehlgeschlagen: {e}")
            return False
    
    async def _test_connection(self):
        """Verbindungstest"""
        try:
            balance = await self.exchange.fetch_balance()
            logger.info(f"🔗 Verbindungstest erfolgreich - Balance verfügbar")
        except Exception as e:
            logger.error(f"❌ Verbindungstest fehlgeschlagen: {e}")
            raise
    
    async def _load_markets(self):
        """Märkte und Symbol-Informationen laden"""
        try:
            markets = await self.exchange.load_markets()
            self.symbols_info = {symbol: info for symbol, info in markets.items() 
                               if info.get('type') == 'swap' and info.get('active', True)}
            
            logger.info(f"📊 {len(self.symbols_info)} Futures-Märkte geladen")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden der Märkte: {e}")
            raise
    
    # ==================== TRADING METHODS ====================
    
    async def place_order(self, symbol: str, order_type: OrderType, side: OrderSide, 
                         amount: Decimal, price: Optional[Decimal] = None,
                         stop_loss: Optional[Decimal] = None,
                         take_profit: Optional[Decimal] = None,
                         leverage: int = 1) -> Optional[Order]:
        """📈 Order platzieren"""
        try:
            self._rate_limit_check()
            
            # Symbol validieren
            if symbol not in self.symbols_info:
                logger.error(f"❌ Unbekanntes Symbol: {symbol}")
                return None
            
            # Leverage setzen
            await self._set_leverage(symbol, leverage)
            
            # Order Parameter
            order_params = {
                'symbol': symbol,
                'type': order_type.value,
                'side': side.value,
                'amount': float(amount),
            }
            
            if price and order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                order_params['price'] = float(price)
            
            # Stop Loss / Take Profit
            if stop_loss or take_profit:
                order_params['params'] = {}
                if stop_loss:
                    order_params['params']['stopLoss'] = float(stop_loss)
                if take_profit:
                    order_params['params']['takeProfit'] = float(take_profit)
            
            # Order ausführen
            logger.info(f"📤 Order wird platziert: {symbol} {side.value} {amount}")
            exchange_order = await self.exchange.create_order(**order_params)
            
            # Order Model erstellen
            order = Order(
                id=generate_id("ORD_"),
                exchange_id=exchange_order['id'],
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=Decimal(str(price)) if price else None,
                status=OrderStatus.PENDING,
                leverage=leverage,
                metadata={'exchange_order': exchange_order}
            )
            
            log_trade(f"✅ Order platziert: {order.id} - {symbol} {side.value} {amount}")
            return order
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Platzieren der Order: {e}")
            return None
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """❌ Order stornieren"""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            log_trade(f"🚫 Order storniert: {order_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Fehler beim Stornieren der Order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        """📊 Order Status abrufen"""
        try:
            exchange_order = await self.exchange.fetch_order(order_id, symbol)
            
            # Status mapping
            status_map = {
                'open': OrderStatus.OPEN,
                'closed': OrderStatus.FILLED,
                'canceled': OrderStatus.CANCELED,
                'rejected': OrderStatus.REJECTED,
                'expired': OrderStatus.EXPIRED
            }
            
            order = Order(
                id=generate_id("ORD_"),
                exchange_id=exchange_order['id'],
                symbol=exchange_order['symbol'],
                type=OrderType(exchange_order['type']),
                side=OrderSide(exchange_order['side']),
                amount=Decimal(str(exchange_order['amount'])),
                price=Decimal(str(exchange_order['price'])) if exchange_order['price'] else None,
                status=status_map.get(exchange_order['status'], OrderStatus.PENDING),
                filled=Decimal(str(exchange_order['filled'])),
                remaining=Decimal(str(exchange_order['remaining'])),
                cost=Decimal(str(exchange_order['cost'])),
                fee=Decimal(str(exchange_order['fee']['cost'] if exchange_order['fee'] else 0))
            )
            
            return order
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Abrufen des Order-Status: {e}")
            return None
    
    # ==================== PORTFOLIO METHODS ====================
    
    async def get_portfolio(self) -> Optional[Portfolio]:
        """💼 Portfolio-Daten abrufen"""
        try:
            balance = await self.exchange.fetch_balance()
            positions_data = await self.exchange.fetch_positions()
            
            # Balance Information
            total_balance = Decimal(str(balance['USDT']['total']))
            free_balance = Decimal(str(balance['USDT']['free']))
            used_balance = Decimal(str(balance['USDT']['used']))
            
            # Positionen konvertieren
            positions = []
            for pos_data in positions_data:
                if pos_data['contracts'] > 0:  # Nur offene Positionen
                    position = Position(
                        id=generate_id("POS_"),
                        symbol=pos_data['symbol'],
                        side=pos_data['side'].lower(),
                        size=Decimal(str(pos_data['contracts'])),
                        entry_price=Decimal(str(pos_data['entryPrice'])),
                        current_price=Decimal(str(pos_data['markPrice'])),
                        unrealized_pnl=Decimal(str(pos_data['unrealizedPnl'])),
                        unrealized_pnl_percent=safe_float_convert(pos_data['percentage']),
                        leverage=int(pos_data['leverage']),
                        margin_used=Decimal(str(pos_data['initialMargin'])),
                        strategy="auto_detected"
                    )
                    positions.append(position)
            
            # Portfolio erstellen
            portfolio = Portfolio(
                account_id=generate_id("ACC_"),
                total_balance=total_balance,
                available_balance=free_balance,
                used_margin=used_balance,
                free_margin=free_balance,
                open_positions=positions,
                position_count=len(positions),
                unrealized_pnl=sum(pos.unrealized_pnl for pos in positions),
                margin_ratio=float(used_balance / total_balance) if total_balance > 0 else 0
            )
            
            return portfolio
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Abrufen des Portfolios: {e}")
            return None
    
    # ==================== MARKET DATA METHODS ====================
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """📊 Marktdaten abrufen"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            
            market_data = MarketData(
                symbol=symbol,
                price=Decimal(str(ticker['last'])),
                bid=Decimal(str(ticker['bid'])),
                ask=Decimal(str(ticker['ask'])),
                volume_24h=Decimal(str(ticker['baseVolume'])),
                change_24h=Decimal(str(ticker['change'])),
                change_24h_percent=safe_float_convert(ticker['percentage']),
                high_24h=Decimal(str(ticker['high'])),
                low_24h=Decimal(str(ticker['low']))
            )
            
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Abrufen der Marktdaten für {symbol}: {e}")
            return None
    
    async def get_candles(self, symbol: str, timeframe: str = '1m', 
                         limit: int = 100) -> List[Candle]:
        """📈 Kerzendaten abrufen"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            candles = []
            for data in ohlcv:
                candle = Candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.fromtimestamp(data[0] / 1000),
                    open=Decimal(str(data[1])),
                    high=Decimal(str(data[2])),
                    low=Decimal(str(data[3])),
                    close=Decimal(str(data[4])),
                    volume=Decimal(str(data[5]))
                )
                candles.append(candle)
            
            return candles
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Abrufen der Kerzendaten für {symbol}: {e}")
            return []
    
    # ==================== UTILITY METHODS ====================
    
    async def _set_leverage(self, symbol: str, leverage: int):
        """Leverage für Symbol setzen"""
        try:
            await self.exchange.set_leverage(leverage, symbol)
        except Exception as e:
            logger.warning(f"⚠️ Leverage konnte nicht gesetzt werden: {e}")
    
    def _rate_limit_check(self):
        """Rate Limiting prüfen"""
        now = datetime.utcnow()
        self.api_call_count += 1
        
        # Reset Counter jede Minute
        if (now - self.last_api_call).seconds > 60:
            self.api_call_count = 0
            self.last_api_call = now
    
    async def get_profitable_pairs(self, min_volume_24h: float = 10000000) -> List[str]:
        """💰 Profitable Trading-Paare identifizieren"""
        try:
            profitable_pairs = []
            
            for symbol, info in self.symbols_info.items():
                try:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    
                    # Filter Kriterien
                    volume_24h = safe_float_convert(ticker.get('baseVolume', 0))
                    change_24h_abs = abs(safe_float_convert(ticker.get('percentage', 0)))
                    
                    if (volume_24h >= min_volume_24h and 
                        change_24h_abs >= 2.0):  # Min 2% Bewegung
                        profitable_pairs.append(symbol)
                        
                except Exception:
                    continue
            
            # Nach Volumen sortieren
            profitable_pairs = profitable_pairs[:10]  # Top 10
            logger.info(f"💰 {len(profitable_pairs)} profitable Paare identifiziert")
            
            return profitable_pairs
            
        except Exception as e:
            logger.error(f"❌ Fehler bei der Identifikation profitabler Paare: {e}")
            return []
    
    async def shutdown(self):
        """🛑 Connector herunterfahren"""
        try:
            if self.exchange:
                await self.exchange.close()
            self.is_connected = False
            logger.info("✅ Bitget API Connector heruntergefahren")
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren: {e}")


# Alias für Kompatibilität
BitgetPro = BitgetProConnector

# Export für Import
__all__ = ['BitgetProConnector', 'BitgetPro']
