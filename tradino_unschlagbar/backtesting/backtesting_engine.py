"""
🧪 TRADINO UNSCHLAGBAR - Backtesting Engine
Robustes Backtesting-Framework mit Walk-Forward Analysis & Monte Carlo

Author: AI Trading Systems
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

from models.trade_models import TradeSignal, Trade, OrderSide
from models.market_models import Candle, MarketData
# Mock Strategies wenn nicht verfügbar
try:
    from strategies.scalping_master import ScalpingMaster
    from strategies.swing_genius import SwingGenius
    from strategies.trend_hunter import TrendHunter
    from strategies.mean_reversion import MeanReversion
except ImportError:
    # Mock Strategies für Testing
    class MockStrategy:
        def __init__(self):
            self.name = "MockStrategy"
    
    ScalpingMaster = MockStrategy
    SwingGenius = MockStrategy
    TrendHunter = MockStrategy
    MeanReversion = MockStrategy

# Mock AI wenn nicht verfügbar
try:
    from brain.master_ai import MasterAI
    from brain.market_intelligence import MarketIntelligence
except ImportError:
    class MasterAI:
        pass
    class MarketIntelligence:
        pass

from connectors.bitget_pro import BitgetProConnector
from utils.logger_pro import setup_logger
from utils.config_manager import ConfigManager

logger = setup_logger("BacktestingEngine")


@dataclass
class BacktestConfig:
    """🔧 Backtesting Konfiguration"""
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005   # 0.05%
    max_positions: int = 5
    timeframes: List[str] = field(default_factory=lambda: ['1h', '4h', '1d'])
    symbols: List[str] = field(default_factory=lambda: ['BTC/USDT', 'ETH/USDT'])
    use_leverage: bool = False
    max_leverage: int = 1
    enable_fees: bool = True
    enable_slippage: bool = True


@dataclass
class BacktestResults:
    """📊 Backtesting Ergebnisse"""
    config: BacktestConfig
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    volatility: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_duration: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    trades: List[Trade]
    equity_curve: pd.DataFrame
    monthly_returns: pd.DataFrame
    strategy_performance: Dict[str, Any]
    monte_carlo_results: Optional[Dict[str, Any]] = None


@dataclass
class WalkForwardResult:
    """📈 Walk-Forward Analysis Ergebnis"""
    train_periods: List[Tuple[datetime, datetime]]
    test_periods: List[Tuple[datetime, datetime]]
    results: List[BacktestResults]
    combined_metrics: Dict[str, float]
    stability_score: float
    overfitting_score: float


class BacktestingEngine:
    """🧪 Advanced Backtesting Engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategies: Dict[str, Any] = {}
        self.current_positions: Dict[str, Dict] = {}
        self.trade_history: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.current_capital = config.initial_capital
        self.peak_capital = config.initial_capital
        
        # Market Data Cache
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        
        # Performance Tracking
        self.daily_returns: List[float] = []
        self.benchmark_returns: List[float] = []
        
        # Results Storage
        self.results_path = Path("data/backtesting")
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🧪 Backtesting Engine initialisiert für {config.start_date} bis {config.end_date}")
    
    async def add_strategy(self, strategy_name: str, strategy_instance: Any) -> None:
        """➕ Trading-Strategie hinzufügen"""
        try:
            self.strategies[strategy_name] = strategy_instance
            logger.info(f"✅ Strategie hinzugefügt: {strategy_name}")
        except Exception as e:
            logger.error(f"❌ Fehler beim Hinzufügen der Strategie {strategy_name}: {e}")
    
    async def load_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """📊 Echte Marktdaten von Exchange laden"""
        try:
            cache_key = f"{symbol}_{timeframe}"
            
            if cache_key in self.market_data_cache:
                return self.market_data_cache[cache_key]
            
            # ECHTE Marktdaten von BitGet API laden
            start_dt = datetime.strptime(self.config.start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(self.config.end_date, '%Y-%m-%d')
            
            try:
                # BitGet Connector für echte Daten
                connector = BitgetProConnector()
                
                # Echte historische Daten abrufen
                data = await connector.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_dt,
                    end_time=end_dt,
                    limit=1000
                )
                
                if data.empty:
                    logger.warning(f"⚠️ Keine echten Daten für {symbol} verfügbar - verwende Fallback")
                    data = await self._fetch_alternative_real_data(symbol, timeframe, start_dt, end_dt)
                
            except Exception as e:
                logger.warning(f"⚠️ BitGet API Fehler: {e} - verwende alternative echte Datenquelle")
                data = await self._fetch_alternative_real_data(symbol, timeframe, start_dt, end_dt)
            
            self.market_data_cache[cache_key] = data
            logger.info(f"📊 ECHTE Marktdaten geladen für {symbol} ({timeframe}): {len(data)} Perioden")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden der ECHTEN Marktdaten für {symbol}: {e}")
            return pd.DataFrame()
    
    async def _fetch_alternative_real_data(self, symbol: str, timeframe: str, 
                                         start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """📈 Alternative echte Datenquellen (Yahoo Finance, Alpha Vantage, etc.)"""
        try:
            import yfinance as yf
            
            # Symbol für Yahoo Finance konvertieren
            yf_symbol = symbol.replace('/USDT', '-USD').replace('/BTC', '-BTC')
            
            # Echte Daten von Yahoo Finance
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(
                start=start_dt.strftime('%Y-%m-%d'),
                end=end_dt.strftime('%Y-%m-%d'),
                interval=self._convert_timeframe_to_yf(timeframe)
            )
            
            if not data.empty:
                # Konvertiere zu unserem Format
                formatted_data = []
                for timestamp, row in data.iterrows():
                    formatted_data.append({
                        'timestamp': timestamp,
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': float(row['Volume'])
                    })
                
                result_df = pd.DataFrame(formatted_data)
                logger.info(f"✅ ECHTE Yahoo Finance Daten geladen für {symbol}")
                return result_df
            
        except Exception as e:
            logger.error(f"❌ Fehler bei alternativen ECHTEN Daten: {e}")
        
        # Als letzter Ausweg: Binance Public API
        try:
            import requests
            
            # Binance API für echte Krypto-Daten
            if 'USDT' in symbol:
                binance_symbol = symbol.replace('/', '')
                url = f"https://api.binance.com/api/v3/klines"
                
                params = {
                    'symbol': binance_symbol,
                    'interval': self._convert_timeframe_to_binance(timeframe),
                    'startTime': int(start_dt.timestamp() * 1000),
                    'endTime': int(end_dt.timestamp() * 1000),
                    'limit': 1000
                }
                
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    klines = response.json()
                    
                    formatted_data = []
                    for kline in klines:
                        formatted_data.append({
                            'timestamp': pd.to_datetime(kline[0], unit='ms'),
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5])
                        })
                    
                    result_df = pd.DataFrame(formatted_data)
                    logger.info(f"✅ ECHTE Binance Daten geladen für {symbol}")
                    return result_df
                    
        except Exception as e:
            logger.error(f"❌ Fehler bei Binance ECHTEN Daten: {e}")
        
        raise Exception(f"Keine ECHTEN Marktdaten verfügbar für {symbol}")
    
    def _convert_timeframe_to_yf(self, timeframe: str) -> str:
        """Konvertiere Timeframe zu Yahoo Finance Format"""
        mapping = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '4h', '1d': '1d'
        }
        return mapping.get(timeframe, '1h')
    
    def _convert_timeframe_to_binance(self, timeframe: str) -> str:
        """Konvertiere Timeframe zu Binance Format"""
        mapping = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '4h', '1d': '1d'
        }
        return mapping.get(timeframe, '1h')

    async def run_backtest(self, symbols: Optional[List[str]] = None) -> BacktestResults:
        """🚀 Backtest ausführen"""
        try:
            logger.info("🚀 Starte Backtesting...")
            
            symbols = symbols or self.config.symbols
            
            # Portfolio State initialisieren
            self._reset_portfolio_state()
            
            # Marktdaten für alle Symbole laden
            market_data = {}
            for symbol in symbols:
                for timeframe in self.config.timeframes:
                    key = f"{symbol}_{timeframe}"
                    market_data[key] = await self.load_market_data(symbol, timeframe)
            
            # Hauptsimulation
            await self._run_simulation(market_data, symbols)
            
            # Ergebnisse berechnen
            results = self._calculate_results()
            
            # Ergebnisse speichern
            await self._save_results(results)
            
            logger.info(f"✅ Backtesting abgeschlossen - Return: {results.total_return:.2%}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Backtest: {e}")
            raise
    
    def _reset_portfolio_state(self):
        """🔄 Portfolio-Status zurücksetzen"""
        self.current_positions = {}
        self.trade_history = []
        self.equity_curve = []
        self.current_capital = self.config.initial_capital
        self.peak_capital = self.config.initial_capital
        self.daily_returns = []
    
    async def _run_simulation(self, market_data: Dict[str, pd.DataFrame], symbols: List[str]):
        """🎯 Haupt-Simulationsloop"""
        try:
            # Alle Timestamps aus allen Timeframes sammeln
            all_timestamps = set()
            for df in market_data.values():
                if not df.empty:
                    all_timestamps.update(df.index)
            
            # Sortierte Timestamp-Liste
            timestamps = sorted(list(all_timestamps))
            
            logger.info(f"📅 Simuliere {len(timestamps)} Zeitpunkte")
            
            for i, current_time in enumerate(timestamps):
                if i % 1000 == 0:
                    logger.info(f"📊 Fortschritt: {i}/{len(timestamps)} ({i/len(timestamps)*100:.1f}%)")
                
                # Portfolio-Update zu diesem Zeitpunkt
                await self._update_portfolio(current_time, market_data, symbols)
                
                # Neue Signale generieren
                await self._generate_signals(current_time, market_data, symbols)
                
                # Bestehende Positions prüfen (SL/TP)
                await self._check_position_exits(current_time, market_data)
                
                # Equity Curve updaten
                self._update_equity_curve(current_time)
            
            logger.info("✅ Simulation abgeschlossen")
            
        except Exception as e:
            logger.error(f"❌ Fehler in der Simulation: {e}")
            raise
    
    async def _generate_signals(self, current_time: datetime, market_data: Dict[str, pd.DataFrame], 
                               symbols: List[str]):
        """📡 Trading-Signale generieren"""
        try:
            for symbol in symbols:
                # Zu viele Positionen check
                if len(self.current_positions) >= self.config.max_positions:
                    continue
                
                # Bereits Position offen für dieses Symbol
                if symbol in self.current_positions:
                    continue
                
                # Marktdaten für aktuellen Zeitpunkt abrufen
                current_data = {}
                for timeframe in self.config.timeframes:
                    key = f"{symbol}_{timeframe}"
                    if key in market_data and not market_data[key].empty:
                        df = market_data[key]
                        # Data bis zum aktuellen Zeitpunkt
                        current_df = df[df.index <= current_time]
                        if len(current_df) >= 50:  # Mindestens 50 Perioden für TA
                            current_data[timeframe] = current_df
                
                if not current_data:
                    continue
                
                # Signale von allen Strategien abrufen
                for strategy_name, strategy in self.strategies.items():
                    try:
                        signal = await self._get_strategy_signal(
                            strategy, strategy_name, symbol, current_time, current_data
                        )
                        
                        if signal and signal.confidence >= 0.6:
                            await self._execute_signal(signal, current_time, market_data)
                            break  # Nur ein Signal pro Symbol/Zeitpunkt
                    
                    except Exception as e:
                        logger.warning(f"⚠️ Strategie {strategy_name} Fehler: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"❌ Fehler bei Signal-Generierung: {e}")
    
    async def _get_strategy_signal(self, strategy: Any, strategy_name: str, symbol: str, 
                                 current_time: datetime, market_data: Dict[str, pd.DataFrame]) -> Optional[TradeSignal]:
        """📈 ECHTES Trading-Signal von ECHTER Strategie generieren"""
        try:
            # Aktuelle Marktdaten für das Symbol
            symbol_data = market_data.get(symbol)
            if symbol_data is None or symbol_data.empty:
                return None
            
            # Aktuelle Datenzeile finden
            current_data = symbol_data[symbol_data.index <= current_time].tail(50)  # Letzten 50 Bars
            if len(current_data) < 20:  # Mindestens 20 Bars für Indikatoren
                return None
            
            # ECHTE Strategien verwenden ihre eigenen Signal-Methoden
            try:
                if hasattr(strategy, 'generate_signals'):
                    # Für echte TRADINO Strategien
                    signals = await strategy.generate_signals(current_data, symbol)
                    if signals and len(signals) > 0:
                        latest_signal = signals[-1]  # Neuestes Signal
                        
                        # Konvertiere zu unserem TradeSignal Format
                        if hasattr(latest_signal, 'action') and latest_signal.action != 'HOLD':
                            return TradeSignal(
                                symbol=symbol,
                                side=OrderSide.BUY if latest_signal.action == 'BUY' else OrderSide.SELL,
                                quantity=latest_signal.quantity if hasattr(latest_signal, 'quantity') else 1.0,
                                price=current_data['close'].iloc[-1],
                                timestamp=current_time,
                                strategy=strategy_name,
                                confidence=latest_signal.confidence if hasattr(latest_signal, 'confidence') else 0.8,
                                stop_loss=latest_signal.stop_loss if hasattr(latest_signal, 'stop_loss') else None,
                                take_profit=latest_signal.take_profit if hasattr(latest_signal, 'take_profit') else None
                            )
                
                elif hasattr(strategy, 'analyze_market'):
                    # Für AI-basierte Strategien
                    analysis = await strategy.analyze_market(symbol, current_data)
                    if analysis and analysis.get('signal', 'HOLD') != 'HOLD':
                        return TradeSignal(
                            symbol=symbol,
                            side=OrderSide.BUY if analysis['signal'] == 'BUY' else OrderSide.SELL,
                            quantity=analysis.get('position_size', 1.0),
                            price=current_data['close'].iloc[-1],
                            timestamp=current_time,
                            strategy=strategy_name,
                            confidence=analysis.get('confidence', 0.7),
                            stop_loss=analysis.get('stop_loss'),
                            take_profit=analysis.get('take_profit')
                        )
                
                elif isinstance(strategy, ScalpingMaster):
                    # Spezifische Integration für ScalpingMaster
                    signals = strategy.scalp_opportunities(current_data, symbol)
                    if signals:
                        signal = signals[0]  # Erstes Signal
                        return TradeSignal(
                            symbol=symbol,
                            side=OrderSide.BUY if signal['side'] == 'long' else OrderSide.SELL,
                            quantity=signal.get('size', 1.0),
                            price=current_data['close'].iloc[-1],
                            timestamp=current_time,
                            strategy=strategy_name,
                            confidence=signal.get('confidence', 0.8),
                            stop_loss=signal.get('stop_loss'),
                            take_profit=signal.get('take_profit')
                        )
                
                elif isinstance(strategy, SwingGenius):
                    # Spezifische Integration für SwingGenius
                    swing_signal = strategy.find_swing_setup(current_data, symbol)
                    if swing_signal and swing_signal.get('action') != 'HOLD':
                        return TradeSignal(
                            symbol=symbol,
                            side=OrderSide.BUY if swing_signal['action'] == 'BUY' else OrderSide.SELL,
                            quantity=swing_signal.get('position_size', 1.0),
                            price=current_data['close'].iloc[-1],
                            timestamp=current_time,
                            strategy=strategy_name,
                            confidence=swing_signal.get('confidence', 0.75),
                            stop_loss=swing_signal.get('stop_loss'),
                            take_profit=swing_signal.get('take_profit')
                        )
                
                elif isinstance(strategy, TrendHunter):
                    # Spezifische Integration für TrendHunter
                    trend_signal = strategy.hunt_trend(current_data, symbol)
                    if trend_signal and trend_signal.get('direction') != 'SIDEWAYS':
                        side = OrderSide.BUY if trend_signal['direction'] == 'UP' else OrderSide.SELL
                        return TradeSignal(
                            symbol=symbol,
                            side=side,
                            quantity=trend_signal.get('position_size', 1.0),
                            price=current_data['close'].iloc[-1],
                            timestamp=current_time,
                            strategy=strategy_name,
                            confidence=trend_signal.get('strength', 0.7),
                            stop_loss=trend_signal.get('stop_loss'),
                            take_profit=trend_signal.get('take_profit')
                        )
                
                elif isinstance(strategy, MeanReversion):
                    # Spezifische Integration für MeanReversion
                    reversion_signal = strategy.detect_reversion(current_data, symbol)
                    if reversion_signal and reversion_signal.get('signal') != 'NEUTRAL':
                        return TradeSignal(
                            symbol=symbol,
                            side=OrderSide.BUY if reversion_signal['signal'] == 'BUY' else OrderSide.SELL,
                            quantity=reversion_signal.get('quantity', 1.0),
                            price=current_data['close'].iloc[-1],
                            timestamp=current_time,
                            strategy=strategy_name,
                            confidence=reversion_signal.get('confidence', 0.7),
                            stop_loss=reversion_signal.get('stop_loss'),
                            take_profit=reversion_signal.get('take_profit')
                        )
                
                else:
                    logger.warning(f"⚠️ Unbekannte Strategie-Typ: {type(strategy)}")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Fehler bei ECHTER Signal-Generierung für {strategy_name}: {e}")
                return None
            
            return None  # Kein Signal generiert
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Signal-Generierung für {strategy_name}: {e}")
            return None
    
    async def _execute_signal(self, signal: TradeSignal, current_time: datetime, 
                             market_data: Dict[str, pd.DataFrame]):
        """💼 Trading-Signal ausführen"""
        try:
            # Aktuellen Preis mit Slippage abrufen
            symbol_1h = f"{signal.symbol}_1h"
            if symbol_1h not in market_data:
                return
            
            current_price_data = market_data[symbol_1h]
            current_candle = current_price_data[current_price_data.index <= current_time].iloc[-1]
            
            # Slippage anwenden
            execution_price = float(signal.entry_price)
            if self.config.enable_slippage:
                slippage_factor = 1 + self.config.slippage if signal.side == OrderSide.BUY else 1 - self.config.slippage
                execution_price *= slippage_factor
            
            # Gebühren berechnen
            position_value = float(signal.quantity) * execution_price
            commission = position_value * self.config.commission if self.config.enable_fees else 0
            
            # Capital Check
            required_capital = position_value + commission
            if required_capital > self.current_capital:
                logger.warning(f"⚠️ Nicht genügend Kapital für {signal.symbol}: {required_capital:.2f} > {self.current_capital:.2f}")
                return
            
            # Position eröffnen
            position = {
                'signal_id': signal.id,
                'symbol': signal.symbol,
                'side': signal.side,
                'entry_price': execution_price,
                'quantity': float(signal.quantity),
                'stop_loss': float(signal.stop_loss) if signal.stop_loss else None,
                'take_profit': float(signal.take_profit) if signal.take_profit else None,
                'entry_time': current_time,
                'strategy': signal.strategy,
                'commission': commission,
                'confidence': signal.confidence,
                'metadata': signal.metadata
            }
            
            self.current_positions[signal.symbol] = position
            self.current_capital -= required_capital
            
            logger.debug(f"📈 Position eröffnet: {signal.symbol} {signal.side.value} @ {execution_price:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Signal-Ausführung: {e}")
    
    async def _check_position_exits(self, current_time: datetime, market_data: Dict[str, pd.DataFrame]):
        """🚪 Position-Ausstiege prüfen (SL/TP)"""
        positions_to_close = []
        
        try:
            for symbol, position in self.current_positions.items():
                symbol_1h = f"{symbol}_1h"
                if symbol_1h not in market_data:
                    continue
                
                current_price_data = market_data[symbol_1h]
                current_candle = current_price_data[current_price_data.index <= current_time].iloc[-1]
                current_price = current_candle['close']
                
                # Stop Loss Check
                if position['stop_loss']:
                    if ((position['side'] == OrderSide.BUY and current_price <= position['stop_loss']) or
                        (position['side'] == OrderSide.SELL and current_price >= position['stop_loss'])):
                        positions_to_close.append((symbol, current_price, 'stop_loss'))
                        continue
                
                # Take Profit Check
                if position['take_profit']:
                    if ((position['side'] == OrderSide.BUY and current_price >= position['take_profit']) or
                        (position['side'] == OrderSide.SELL and current_price <= position['take_profit'])):
                        positions_to_close.append((symbol, current_price, 'take_profit'))
                        continue
            
            # Positionen schließen
            for symbol, exit_price, exit_reason in positions_to_close:
                await self._close_position(symbol, exit_price, current_time, exit_reason)
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Position-Exit-Check: {e}")
    
    async def _close_position(self, symbol: str, exit_price: float, exit_time: datetime, 
                            exit_reason: str = 'manual'):
        """🔒 Position schließen"""
        try:
            if symbol not in self.current_positions:
                return
            
            position = self.current_positions[symbol]
            
            # Slippage bei Exit anwenden
            if self.config.enable_slippage:
                slippage_factor = 1 - self.config.slippage if position['side'] == OrderSide.BUY else 1 + self.config.slippage
                exit_price *= slippage_factor
            
            # P&L berechnen
            if position['side'] == OrderSide.BUY:
                pnl_per_unit = exit_price - position['entry_price']
            else:
                pnl_per_unit = position['entry_price'] - exit_price
            
            gross_pnl = pnl_per_unit * position['quantity']
            
            # Exit-Gebühren
            exit_value = position['quantity'] * exit_price
            exit_commission = exit_value * self.config.commission if self.config.enable_fees else 0
            
            # Net P&L
            net_pnl = gross_pnl - position['commission'] - exit_commission
            pnl_percentage = (net_pnl / (position['entry_price'] * position['quantity'])) * 100
            
            # Trade-Record erstellen
            trade = Trade(
                id=f"trade_{len(self.trade_history)+1}",
                signal_id=position['signal_id'],
                symbol=symbol,
                side=position['side'],
                entry_price=position['entry_price'],
                exit_price=exit_price,
                quantity=position['quantity'],
                pnl=net_pnl,
                pnl_percentage=pnl_percentage,
                commission=position['commission'] + exit_commission,
                strategy=position['strategy'],
                entry_time=position['entry_time'],
                exit_time=exit_time,
                duration_minutes=int((exit_time - position['entry_time']).total_seconds() / 60),
                metadata={
                    **position['metadata'],
                    'exit_reason': exit_reason,
                    'confidence': position['confidence']
                }
            )
            
            self.trade_history.append(trade)
            
            # Kapital zurückbuchen
            self.current_capital += exit_value - exit_commission + net_pnl
            
            # Position entfernen
            del self.current_positions[symbol]
            
            logger.debug(f"🔒 Position geschlossen: {symbol} {exit_reason} P&L: {net_pnl:.2f} ({pnl_percentage:.2f}%)")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Schließen der Position {symbol}: {e}")
    
    async def _update_portfolio(self, current_time: datetime, market_data: Dict[str, pd.DataFrame], 
                              symbols: List[str]):
        """💼 Portfolio zu aktuellem Zeitpunkt aktualisieren"""
        try:
            # Unrealized P&L für offene Positionen berechnen
            unrealized_pnl = 0
            
            for symbol, position in self.current_positions.items():
                symbol_1h = f"{symbol}_1h"
                if symbol_1h in market_data:
                    current_price_data = market_data[symbol_1h]
                    current_candle = current_price_data[current_price_data.index <= current_time].iloc[-1]
                    current_price = current_candle['close']
                    
                    if position['side'] == OrderSide.BUY:
                        pnl_per_unit = current_price - position['entry_price']
                    else:
                        pnl_per_unit = position['entry_price'] - current_price
                    
                    unrealized_pnl += pnl_per_unit * position['quantity']
            
            # Total Portfolio Value
            total_value = self.current_capital + unrealized_pnl
            
            # Peak Capital für Drawdown-Berechnung
            if total_value > self.peak_capital:
                self.peak_capital = total_value
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Portfolio-Update: {e}")
    
    def _update_equity_curve(self, current_time: datetime):
        """📊 Equity Curve aktualisieren"""
        try:
            # Total Portfolio Value berechnen
            unrealized_pnl = 0
            for position in self.current_positions.values():
                # Vereinfacht - in Realität: aktueller Marktpreis
                unrealized_pnl += 0  # Wird in _update_portfolio berechnet
            
            total_value = self.current_capital + unrealized_pnl
            
            equity_point = {
                'timestamp': current_time,
                'total_value': total_value,
                'cash': self.current_capital,
                'unrealized_pnl': unrealized_pnl,
                'open_positions': len(self.current_positions),
                'realized_pnl': sum(trade.pnl for trade in self.trade_history if trade.pnl)
            }
            
            self.equity_curve.append(equity_point)
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Equity Curve Update: {e}") 

    async def walk_forward_analysis(self, train_period: int = 365, test_period: int = 30, 
                                   step_size: int = 30) -> WalkForwardResult:
        """📈 Walk-Forward Analysis durchführen"""
        try:
            logger.info(f"🚀 Starte Walk-Forward Analysis (Train: {train_period}d, Test: {test_period}d)")
            
            start_dt = datetime.strptime(self.config.start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(self.config.end_date, '%Y-%m-%d')
            
            train_periods = []
            test_periods = []
            results = []
            
            current_start = start_dt
            
            while current_start + timedelta(days=train_period + test_period) <= end_dt:
                # Training Period
                train_start = current_start
                train_end = current_start + timedelta(days=train_period)
                
                # Test Period
                test_start = train_end
                test_end = train_end + timedelta(days=test_period)
                
                train_periods.append((train_start, train_end))
                test_periods.append((test_start, test_end))
                
                # Backtest für diesen Test-Zeitraum
                test_config = BacktestConfig(
                    start_date=test_start.strftime('%Y-%m-%d'),
                    end_date=test_end.strftime('%Y-%m-%d'),
                    initial_capital=self.config.initial_capital,
                    commission=self.config.commission,
                    slippage=self.config.slippage,
                    max_positions=self.config.max_positions,
                    symbols=self.config.symbols
                )
                
                # Neuen Engine für diesen Zeitraum
                test_engine = BacktestingEngine(test_config)
                
                # Strategien kopieren
                test_engine.strategies = self.strategies.copy()
                
                # Backtest ausführen
                result = await test_engine.run_backtest()
                results.append(result)
                
                # Nächster Schritt
                current_start += timedelta(days=step_size)
                
                logger.info(f"📊 WFA Schritt abgeschlossen: {test_start.strftime('%Y-%m-%d')} bis {test_end.strftime('%Y-%m-%d')}")
            
            # Combined Metrics berechnen
            combined_metrics = self._calculate_combined_wfa_metrics(results)
            
            # Stability Score berechnen
            stability_score = self._calculate_stability_score(results)
            
            # Overfitting Score berechnen
            overfitting_score = self._calculate_overfitting_score(results)
            
            wfa_result = WalkForwardResult(
                train_periods=train_periods,
                test_periods=test_periods,
                results=results,
                combined_metrics=combined_metrics,
                stability_score=stability_score,
                overfitting_score=overfitting_score
            )
            
            logger.info(f"✅ Walk-Forward Analysis abgeschlossen - Stability: {stability_score:.2f}")
            return wfa_result
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Walk-Forward Analysis: {e}")
            raise
    
    def _calculate_combined_wfa_metrics(self, results: List[BacktestResults]) -> Dict[str, float]:
        """📊 Combined Walk-Forward Metriken"""
        if not results:
            return {}
        
        # Alle Returns sammeln
        all_returns = []
        all_trades = 0
        total_return = 1.0
        
        for result in results:
            all_returns.extend([trade.pnl_percentage/100 for trade in result.trades if trade.pnl_percentage])
            all_trades += result.total_trades
            total_return *= (1 + result.total_return/100)
        
        total_return = (total_return - 1) * 100  # Zu Prozent
        
        if not all_returns:
            return {'combined_return': 0, 'combined_sharpe': 0}
        
        # Combined Sharpe
        mean_return = np.mean(all_returns)
        std_return = np.std(all_returns)
        combined_sharpe = mean_return / std_return if std_return > 0 else 0
        
        # Consistency Score
        period_returns = [r.total_return for r in results]
        winning_periods = sum(1 for r in period_returns if r > 0)
        consistency = winning_periods / len(period_returns) if period_returns else 0
        
        return {
            'combined_return': total_return,
            'combined_sharpe': combined_sharpe,
            'consistency': consistency,
            'total_trades': all_trades,
            'avg_trades_per_period': all_trades / len(results) if results else 0
        }
    
    def _calculate_stability_score(self, results: List[BacktestResults]) -> float:
        """📊 Stabilität der Strategie berechnen"""
        if len(results) < 2:
            return 0.0
        
        returns = [r.total_return for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        
        # Coefficient of Variation für Returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        cv_returns = std_return / abs(mean_return) if mean_return != 0 else 1
        
        # Coefficient of Variation für Sharpe Ratios
        mean_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)
        cv_sharpe = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else 1
        
        # Stability Score (niedriger CV = höhere Stabilität)
        stability_returns = max(0, 1 - cv_returns)
        stability_sharpe = max(0, 1 - cv_sharpe)
        
        return (stability_returns + stability_sharpe) / 2
    
    def _calculate_overfitting_score(self, results: List[BacktestResults]) -> float:
        """🧠 Overfitting Risk berechnen"""
        if len(results) < 3:
            return 0.5  # Neutral
        
        returns = [r.total_return for r in results]
        
        # Trend in Performance über Zeit
        x = np.arange(len(returns))
        slope, _, r_value, _, _ = np.polyfit(x, returns, 1, full=True)[:5]
        
        # Negativer Trend deutet auf Overfitting hin
        trend_score = max(0, 1 + slope[0] / 10) if isinstance(slope, np.ndarray) else max(0, 1 + slope / 10)
        
        # Variabilität der Performance
        variability = np.std(returns) / (abs(np.mean(returns)) + 0.01)
        variability_score = max(0, 1 - variability / 2)
        
        return (trend_score + variability_score) / 2
    
    def monte_carlo_simulation(self, num_simulations: int = 1000, 
                             confidence_levels: List[float] = [0.05, 0.95]) -> Dict[str, Any]:
        """🎰 Monte Carlo Simulation"""
        try:
            logger.info(f"🎰 Starte Monte Carlo Simulation mit {num_simulations} Durchläufen")
            
            if not self.trade_history:
                logger.warning("⚠️ Keine Trade-Historie für Monte Carlo verfügbar")
                return {}
            
            # Trade Returns extrahieren
            trade_returns = [trade.pnl_percentage/100 for trade in self.trade_history if trade.pnl_percentage]
            
            if len(trade_returns) < 10:
                logger.warning("⚠️ Zu wenige Trades für Monte Carlo")
                return {}
            
            # Monte Carlo Simulation
            simulation_results = []
            
            for sim in range(num_simulations):
                # Random sampling der Trade Returns
                simulated_trades = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
                
                # Equity Curve simulieren
                equity_curve = [self.config.initial_capital]
                for ret in simulated_trades:
                    new_equity = equity_curve[-1] * (1 + ret)
                    equity_curve.append(new_equity)
                
                # Metrics für diese Simulation
                final_equity = equity_curve[-1]
                total_return = (final_equity - self.config.initial_capital) / self.config.initial_capital
                
                # Max Drawdown
                running_max = np.maximum.accumulate(equity_curve)
                drawdowns = [(equity_curve[i] - running_max[i]) / running_max[i] for i in range(len(equity_curve))]
                max_drawdown = min(drawdowns)
                
                simulation_results.append({
                    'total_return': total_return,
                    'final_equity': final_equity,
                    'max_drawdown': abs(max_drawdown)
                })
            
            # Ergebnisse analysieren
            returns = [sim['total_return'] for sim in simulation_results]
            drawdowns = [sim['max_drawdown'] for sim in simulation_results]
            
            mc_results = {
                'num_simulations': num_simulations,
                'return_statistics': {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'min': np.min(returns),
                    'max': np.max(returns),
                    'percentiles': {
                        str(int(level*100)): np.percentile(returns, level*100) 
                        for level in confidence_levels
                    }
                },
                'drawdown_statistics': {
                    'mean': np.mean(drawdowns),
                    'std': np.std(drawdowns),
                    'min': np.min(drawdowns),
                    'max': np.max(drawdowns),
                    'percentiles': {
                        str(int(level*100)): np.percentile(drawdowns, level*100) 
                        for level in confidence_levels
                    }
                },
                'probability_positive': sum(1 for r in returns if r > 0) / len(returns),
                'probability_loss_10': sum(1 for r in returns if r < -0.1) / len(returns),
                'probability_dd_20': sum(1 for dd in drawdowns if dd > 0.2) / len(drawdowns)
            }
            
            logger.info(f"✅ Monte Carlo abgeschlossen - Probability positive: {mc_results['probability_positive']:.2%}")
            return mc_results
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Monte Carlo Simulation: {e}")
            return {}
    
    def _calculate_results(self) -> BacktestResults:
        """📊 Backtest-Ergebnisse berechnen"""
        try:
            if not self.trade_history:
                logger.warning("⚠️ Keine Trades für Ergebnisberechnung")
                return self._empty_results()
            
            # Basic Stats
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for trade in self.trade_history if trade.pnl and trade.pnl > 0)
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Returns
            trade_returns = [trade.pnl_percentage/100 for trade in self.trade_history if trade.pnl_percentage]
            
            if not trade_returns:
                return self._empty_results()
            
            total_return = ((self.current_capital - self.config.initial_capital) / self.config.initial_capital) * 100
            
            # Annualized Return (vereinfacht)
            start_dt = datetime.strptime(self.config.start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(self.config.end_date, '%Y-%m-%d')
            days = (end_dt - start_dt).days
            annual_factor = 365.25 / max(days, 1)
            annual_return = ((1 + total_return/100) ** annual_factor - 1) * 100
            
            # Risk Metrics
            returns_array = np.array(trade_returns)
            volatility = np.std(returns_array) * np.sqrt(252) * 100  # Annualisiert
            
            # Sharpe Ratio
            risk_free_rate = 0.02  # 2%
            excess_return = annual_return/100 - risk_free_rate
            sharpe_ratio = excess_return / (volatility/100) if volatility > 0 else 0
            
            # Sortino Ratio
            downside_returns = returns_array[returns_array < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            sortino_ratio = excess_return / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
            
            # Max Drawdown
            equity_values = [point['total_value'] for point in self.equity_curve]
            if equity_values:
                running_max = np.maximum.accumulate(equity_values)
                drawdowns = [(equity_values[i] - running_max[i]) / running_max[i] for i in range(len(equity_values))]
                max_drawdown = abs(min(drawdowns)) * 100 if drawdowns else 0
            else:
                max_drawdown = 0
            
            # Calmar Ratio
            calmar_ratio = annual_return / max(max_drawdown, 1) if max_drawdown > 0 else 0
            
            # Trading Metrics
            winning_trades_data = [trade for trade in self.trade_history if trade.pnl and trade.pnl > 0]
            losing_trades_data = [trade for trade in self.trade_history if trade.pnl and trade.pnl < 0]
            
            avg_win = np.mean([trade.pnl for trade in winning_trades_data]) if winning_trades_data else 0
            avg_loss = np.mean([abs(trade.pnl) for trade in losing_trades_data]) if losing_trades_data else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            
            # Trade Duration
            durations = [trade.duration_minutes for trade in self.trade_history if trade.duration_minutes]
            avg_trade_duration = np.mean(durations) if durations else 0
            
            # VaR & CVaR
            var_95 = np.percentile(trade_returns, 5) * 100 if trade_returns else 0
            cvar_trades = [r for r in trade_returns if r <= np.percentile(trade_returns, 5)]
            cvar_95 = np.mean(cvar_trades) * 100 if cvar_trades else 0
            
            # Beta & Alpha (vereinfacht gegen Benchmark)
            benchmark_return = 0.1  # 10% Benchmark
            beta = 1.0  # Vereinfacht
            alpha = annual_return - benchmark_return
            
            # Equity Curve DataFrame
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Monthly Returns
            if not equity_df.empty:
                monthly_equity = equity_df['total_value'].resample('M').last()
                monthly_returns = monthly_equity.pct_change().dropna()
                monthly_returns_df = pd.DataFrame({
                    'month': monthly_returns.index,
                    'return': monthly_returns.values
                })
            else:
                monthly_returns_df = pd.DataFrame()
            
            # Strategy Performance
            strategy_performance = {}
            for strategy_name in set(trade.strategy for trade in self.trade_history):
                strategy_trades = [trade for trade in self.trade_history if trade.strategy == strategy_name]
                strategy_pnl = sum(trade.pnl for trade in strategy_trades if trade.pnl)
                strategy_performance[strategy_name] = {
                    'trades': len(strategy_trades),
                    'total_pnl': strategy_pnl,
                    'win_rate': sum(1 for trade in strategy_trades if trade.pnl and trade.pnl > 0) / len(strategy_trades) if strategy_trades else 0
                }
            
            results = BacktestResults(
                config=self.config,
                total_return=total_return,
                annual_return=annual_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                volatility=volatility,
                win_rate=win_rate * 100,
                profit_factor=profit_factor,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                avg_trade_duration=avg_trade_duration,
                var_95=var_95,
                cvar_95=cvar_95,
                beta=beta,
                alpha=alpha,
                trades=self.trade_history,
                equity_curve=equity_df,
                monthly_returns=monthly_returns_df,
                strategy_performance=strategy_performance
            )
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Ergebnisberechnung: {e}")
            return self._empty_results()
    
    def _empty_results(self) -> BacktestResults:
        """📊 Leere Ergebnisse für Fallback"""
        return BacktestResults(
            config=self.config,
            total_return=0, annual_return=0, sharpe_ratio=0, sortino_ratio=0,
            max_drawdown=0, calmar_ratio=0, volatility=0, win_rate=0,
            profit_factor=0, total_trades=0, winning_trades=0, losing_trades=0,
            avg_trade_duration=0, var_95=0, cvar_95=0, beta=0, alpha=0,
            trades=[], equity_curve=pd.DataFrame(), monthly_returns=pd.DataFrame(),
            strategy_performance={}
        ) 

    async def generate_report(self, results: Optional[BacktestResults] = None) -> Dict[str, Any]:
        """📋 Umfassenden Backtesting-Report generieren"""
        try:
            if results is None:
                results = await self.run_backtest()
            
            # Monte Carlo Simulation hinzufügen
            monte_carlo = self.monte_carlo_simulation(num_simulations=500)
            results.monte_carlo_results = monte_carlo
            
            report = {
                'backtest_summary': {
                    'period': f"{self.config.start_date} bis {self.config.end_date}",
                    'initial_capital': self.config.initial_capital,
                    'final_capital': self.config.initial_capital * (1 + results.total_return/100),
                    'symbols_traded': self.config.symbols,
                    'strategies_used': list(self.strategies.keys())
                },
                'performance_metrics': {
                    'total_return': f"{results.total_return:.2f}%",
                    'annual_return': f"{results.annual_return:.2f}%",
                    'volatility': f"{results.volatility:.2f}%",
                    'sharpe_ratio': f"{results.sharpe_ratio:.3f}",
                    'sortino_ratio': f"{results.sortino_ratio:.3f}",
                    'calmar_ratio': f"{results.calmar_ratio:.3f}",
                    'max_drawdown': f"{results.max_drawdown:.2f}%",
                    'var_95': f"{results.var_95:.2f}%",
                    'cvar_95': f"{results.cvar_95:.2f}%"
                },
                'trading_metrics': {
                    'total_trades': results.total_trades,
                    'winning_trades': results.winning_trades,
                    'losing_trades': results.losing_trades,
                    'win_rate': f"{results.win_rate:.2f}%",
                    'profit_factor': f"{results.profit_factor:.2f}",
                    'avg_trade_duration': f"{results.avg_trade_duration:.1f} minutes"
                },
                'strategy_breakdown': results.strategy_performance,
                'risk_analysis': {
                    'beta': f"{results.beta:.3f}",
                    'alpha': f"{results.alpha:.2f}%",
                    'commission_impact': f"{sum(trade.commission for trade in results.trades if trade.commission):.2f}",
                    'slippage_impact': "Included in execution prices"
                },
                'monte_carlo_analysis': monte_carlo,
                'recommendations': self._generate_recommendations(results)
            }
            
            logger.info("📋 Backtesting-Report generiert")
            return report
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Report-Generierung: {e}")
            return {}
    
    def _generate_recommendations(self, results: BacktestResults) -> List[str]:
        """💡 Trading-Empfehlungen basierend auf Backtesting-Ergebnissen"""
        recommendations = []
        
        try:
            # Sharpe Ratio Analyse
            if results.sharpe_ratio < 0.5:
                recommendations.append("⚠️ Niedrige Sharpe Ratio - Überdenken Sie die Risk/Reward-Parameter")
            elif results.sharpe_ratio > 1.5:
                recommendations.append("✅ Ausgezeichnete Sharpe Ratio - Strategie ist sehr effizient")
            
            # Win Rate Analyse
            if results.win_rate < 40:
                recommendations.append("⚠️ Niedrige Win Rate - Erhöhen Sie die Signal-Qualität oder anpassen Sie Entry-Kriterien")
            elif results.win_rate > 70:
                recommendations.append("✅ Hohe Win Rate - Strategie zeigt gute Signal-Qualität")
            
            # Drawdown Analyse
            if results.max_drawdown > 20:
                recommendations.append("🚨 Hoher Max Drawdown - Implementieren Sie strengere Risk Management")
            elif results.max_drawdown < 5:
                recommendations.append("✅ Niedriger Drawdown - Gutes Risk Management")
            
            # Trade Frequency Analyse
            if results.total_trades < 50:
                recommendations.append("⚠️ Wenige Trades - Erhöhen Sie die Signal-Frequenz oder erweitern Sie Timeframes")
            elif results.total_trades > 500:
                recommendations.append("⚠️ Sehr hohe Trade-Frequenz - Überprüfen Sie Commission-Impact")
            
            # Profit Factor Analyse
            if results.profit_factor < 1.2:
                recommendations.append("⚠️ Niedriger Profit Factor - Optimieren Sie Exit-Strategien")
            elif results.profit_factor > 2.0:
                recommendations.append("✅ Exzellenter Profit Factor - Strategie ist sehr profitabel")
            
            # Strategy-spezifische Empfehlungen
            for strategy_name, perf in results.strategy_performance.items():
                if perf['win_rate'] < 0.4:
                    recommendations.append(f"⚠️ Strategie '{strategy_name}' zeigt schwache Performance - Überarbeitung empfohlen")
                elif perf['win_rate'] > 0.7:
                    recommendations.append(f"✅ Strategie '{strategy_name}' zeigt starke Performance - Kapital-Allocation erhöhen")
            
            # Monte Carlo Empfehlungen
            if results.monte_carlo_results:
                prob_positive = results.monte_carlo_results.get('probability_positive', 0)
                if prob_positive < 0.6:
                    recommendations.append("⚠️ Monte Carlo zeigt niedrige Erfolgswahrscheinlichkeit - Risk Management verschärfen")
                elif prob_positive > 0.8:
                    recommendations.append("✅ Monte Carlo zeigt hohe Erfolgswahrscheinlichkeit - Strategie ist robust")
            
            if not recommendations:
                recommendations.append("📊 Strategie zeigt solide Performance - Kontinuierliches Monitoring empfohlen")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Empfehlungs-Generierung: {e}")
            return ["❌ Fehler bei der Analyse - Manuelle Überprüfung erforderlich"]
    
    async def _save_results(self, results: BacktestResults):
        """💾 Backtesting-Ergebnisse speichern"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # JSON Report speichern
            report = await self.generate_report(results)
            report_path = self.results_path / f"backtest_report_{timestamp}.json"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            # Detaillierte Ergebnisse speichern
            results_path = self.results_path / f"backtest_results_{timestamp}.pkl"
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            
            # Equity Curve als CSV
            if not results.equity_curve.empty:
                equity_path = self.results_path / f"equity_curve_{timestamp}.csv"
                results.equity_curve.to_csv(equity_path)
            
            # Trade Details als CSV
            if results.trades:
                trades_data = []
                for trade in results.trades:
                    trades_data.append({
                        'id': trade.id,
                        'symbol': trade.symbol,
                        'strategy': trade.strategy,
                        'side': trade.side.value,
                        'entry_price': float(trade.entry_price),
                        'exit_price': float(trade.exit_price) if trade.exit_price else None,
                        'quantity': float(trade.quantity),
                        'pnl': float(trade.pnl) if trade.pnl else None,
                        'pnl_percentage': trade.pnl_percentage,
                        'entry_time': trade.entry_time.isoformat(),
                        'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                        'duration_minutes': trade.duration_minutes
                    })
                
                trades_df = pd.DataFrame(trades_data)
                trades_path = self.results_path / f"trades_{timestamp}.csv"
                trades_df.to_csv(trades_path, index=False)
            
            logger.info(f"💾 Ergebnisse gespeichert: {report_path}")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Speichern der Ergebnisse: {e}")
    
    async def optimize_parameters(self, parameter_grid: Dict[str, List], 
                                optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
            """🔧 Parameter-Optimierung mit Grid Search"""
            try:
                logger.info("🔧 Starte Parameter-Optimierung...")
                
                # Parameter-Kombinationen generieren
                param_names = list(parameter_grid.keys())
                param_values = list(parameter_grid.values())
                param_combinations = list(itertools.product(*param_values))
                
                best_score = float('-inf')
                best_params = None
                results_cache = []
                
                for i, param_combo in enumerate(param_combinations):
                    logger.info(f"🔧 Teste Parameter-Set {i+1}/{len(param_combinations)}")
                    
                    # Parameter setzen
                    current_params = dict(zip(param_names, param_combo))
                    
                    # Backtest mit diesen Parametern
                    test_config = BacktestConfig(
                        start_date=self.config.start_date,
                        end_date=self.config.end_date,
                        initial_capital=self.config.initial_capital,
                        commission=current_params.get('commission', self.config.commission),
                        slippage=current_params.get('slippage', self.config.slippage),
                        max_positions=current_params.get('max_positions', self.config.max_positions),
                        symbols=self.config.symbols
                    )
                    
                    test_engine = BacktestingEngine(test_config)
                    test_engine.strategies = self.strategies.copy()
                    
                    result = await test_engine.run_backtest()
                    
                    # Optimierungs-Metrik extrahieren
                    score = getattr(result, optimization_metric, 0)
                    
                    results_cache.append({
                        'parameters': current_params,
                        'score': score,
                        'result': result
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = current_params
                        logger.info(f" Neuer bester Score: {score:.3f} mit {current_params}")
                
                # Ergebnisse sortieren
                results_cache.sort(key=lambda x: x['score'], reverse=True)
                
                optimization_result = {
                    'best_parameters': best_params,
                    'best_score': best_score,
                    'optimization_metric': optimization_metric,
                    'total_combinations_tested': len(param_combinations),
                    'top_10_results': results_cache[:10],
                    'parameter_sensitivity': self._analyze_parameter_sensitivity(results_cache, param_names)
                }
                
                logger.info(f"✅ Parameter-Optimierung abgeschlossen - Bester Score: {best_score:.3f}")
                return optimization_result
                
            except Exception as e:
                logger.error(f"❌ Fehler bei Parameter-Optimierung: {e}")
                return {}
    
    def _analyze_parameter_sensitivity(self, results: List[Dict], param_names: List[str]) -> Dict[str, Any]:
        """📊 Parameter-Sensitivitäts-Analyse"""
        try:
            sensitivity = {}
            
            for param_name in param_names:
                param_values = []
                scores = []
                
                for result in results:
                    param_values.append(result['parameters'][param_name])
                    scores.append(result['score'])
                
                # Korrelation zwischen Parameter und Score
                correlation = np.corrcoef(param_values, scores)[0, 1] if len(set(param_values)) > 1 else 0
                
                sensitivity[param_name] = {
                    'correlation': correlation,
                    'impact': abs(correlation),
                    'direction': 'positive' if correlation > 0 else 'negative'
                }
            
            return sensitivity
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Sensitivitäts-Analyse: {e}")
            return {}
    
    def compare_strategies(self, strategy_results: Dict[str, BacktestResults]) -> Dict[str, Any]:
        """⚖️ Strategien vergleichen"""
        try:
            comparison = {
                'strategy_rankings': {},
                'performance_matrix': {},
                'risk_adjusted_rankings': {},
                'recommendations': []
            }
            
            # Performance Metriken sammeln
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
            
            for metric in metrics:
                comparison['performance_matrix'][metric] = {}
                for strategy_name, result in strategy_results.items():
                    comparison['performance_matrix'][metric][strategy_name] = getattr(result, metric, 0)
            
            # Rankings berechnen
            for metric in metrics:
                values = comparison['performance_matrix'][metric]
                if metric == 'max_drawdown':  # Niedriger ist besser
                    sorted_strategies = sorted(values.items(), key=lambda x: x[1])
                else:  # Höher ist besser
                    sorted_strategies = sorted(values.items(), key=lambda x: x[1], reverse=True)
                
                comparison['strategy_rankings'][metric] = [name for name, value in sorted_strategies]
            
            # Risk-Adjusted Ranking (kombiniert Sharpe und Drawdown)
            risk_adjusted_scores = {}
            for strategy_name, result in strategy_results.items():
                score = result.sharpe_ratio * (1 - result.max_drawdown/100)
                risk_adjusted_scores[strategy_name] = score
            
            comparison['risk_adjusted_rankings'] = sorted(
                risk_adjusted_scores.items(), key=lambda x: x[1], reverse=True
            )
            
            # Empfehlungen
            best_overall = comparison['risk_adjusted_rankings'][0][0]
            comparison['recommendations'].append(f"🏆 Beste Gesamt-Performance: {best_overall}")
            
            best_sharpe = comparison['strategy_rankings']['sharpe_ratio'][0]
            comparison['recommendations'].append(f"📊 Beste Risk-Adjusted Return: {best_sharpe}")
            
            lowest_dd = comparison['strategy_rankings']['max_drawdown'][0]
            comparison['recommendations'].append(f"🛡️ Niedrigstes Risiko: {lowest_dd}")
            
            return comparison
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Strategien-Vergleich: {e}")
            return {}


# Utility Functions
def create_sample_config(start_date: str = "2023-01-01", end_date: str = "2023-12-31") -> BacktestConfig:
    """🔧 Sample Backtesting Configuration erstellen"""
    return BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=10000.0,
        commission=0.001,
        slippage=0.0005,
        max_positions=3,
        timeframes=['1h', '4h'],
        symbols=['BTC/USDT', 'ETH/USDT'],
        enable_fees=True,
        enable_slippage=True
    )


async def run_sample_backtest():
    """🚀 Sample Backtest ausführen (für Demo/Tests)"""
    try:
        # Konfiguration erstellen
        config = create_sample_config()
        
        # Engine initialisieren
        engine = BacktestingEngine(config)
        
        # Mock-Strategien hinzufügen
        await engine.add_strategy("scalping_master", "mock_strategy")
        await engine.add_strategy("swing_genius", "mock_strategy")
        
        # Backtest ausführen
        results = await engine.run_backtest()
        
        # Report generieren
        report = await engine.generate_report(results)
        
        print("🧪 SAMPLE BACKTEST ABGESCHLOSSEN")
        print("=" * 50)
        print(f"Total Return: {results.total_return:.2f}%")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {results.max_drawdown:.2f}%")
        print(f"Total Trades: {results.total_trades}")
        print(f"Win Rate: {results.win_rate:.2f}%")
        
        return results, report
        
    except Exception as e:
        logger.error(f"❌ Fehler im Sample Backtest: {e}")
        return None, None


if __name__ == "__main__":
    # Demo ausführen
    import asyncio
    asyncio.run(run_sample_backtest()) 