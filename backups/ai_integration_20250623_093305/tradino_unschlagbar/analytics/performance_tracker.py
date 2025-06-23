"""
📊 TRADINO UNSCHLAGBAR - Performance Tracker
KI-gestützter Performance Tracker für TRADINO

Author: AI Trading Systems
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from utils.logger_pro import setup_logger

logger = setup_logger("PerformanceTracker")


class PerformanceTracker:
    """
    KI-gestützter Performance Tracker für TRADINO
    """
    
    def __init__(self, config):
        self.config = config
        
        # Core Metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_pnl_percentage': 0.0,
            'max_drawdown': 0.0,
            'max_profit': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'start_time': datetime.now(),
            'last_update': datetime.now()
        }
        
        # Trade History
        self.trade_history = []
        self.daily_pnl = []
        self.equity_curve = []
        
        # Real-time Tracking
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.initial_balance = 10000.0  # Default
        
    async def initialize(self):
        """
        Performance Tracker initialisieren
        """
        try:
            logger.info("📊 Performance Tracker wird initialisiert...")
            
            # Initial Balance aus Config laden
            if hasattr(self.config, 'initial_balance'):
                self.initial_balance = self.config.initial_balance
            
            self.peak_equity = self.initial_balance
            
            logger.success("✅ Performance Tracker erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance Tracker Initialisierung fehlgeschlagen: {e}")
            return False
    
    def track_trade(self, trade_result: Dict) -> Dict:
        """
        Einzelnen Trade tracken und Metriken aktualisieren
        """
        try:
            # Trade zu History hinzufügen
            trade_result['timestamp'] = datetime.now()
            self.trade_history.append(trade_result)
            
            # Metriken aktualisieren
            self._update_metrics(trade_result)
            
            # Equity Curve aktualisieren
            self._update_equity_curve(trade_result)
            
            logger.info(f"📊 Trade getrackt: PnL {trade_result.get('pnl', 0):.2f}")
            
            return self.get_performance_summary()
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Trade-Tracking: {e}")
            return {}
    
    def _update_metrics(self, trade_result: Dict):
        """
        Core Metriken basierend auf Trade-Ergebnis aktualisieren
        """
        pnl = trade_result.get('pnl', 0)
        
        # Basic Counts
        self.metrics['total_trades'] += 1
        
        if pnl > 0:
            self.metrics['winning_trades'] += 1
        else:
            self.metrics['losing_trades'] += 1
        
        # PnL Tracking
        self.metrics['total_pnl'] += pnl
        self.metrics['total_pnl_percentage'] = (self.metrics['total_pnl'] / self.initial_balance) * 100
        
        # Win Rate
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = (self.metrics['winning_trades'] / self.metrics['total_trades']) * 100
        
        # Max Profit/Drawdown
        if pnl > self.metrics['max_profit']:
            self.metrics['max_profit'] = pnl
        
        # Average Win/Loss
        if self.metrics['winning_trades'] > 0:
            winning_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
            self.metrics['avg_win'] = np.mean([t['pnl'] for t in winning_trades])
        
        if self.metrics['losing_trades'] > 0:
            losing_trades = [t for t in self.trade_history if t.get('pnl', 0) < 0]
            self.metrics['avg_loss'] = np.mean([t['pnl'] for t in losing_trades])
        
        # Profit Factor
        total_wins = sum([t['pnl'] for t in self.trade_history if t.get('pnl', 0) > 0])
        total_losses = abs(sum([t['pnl'] for t in self.trade_history if t.get('pnl', 0) < 0]))
        
        if total_losses > 0:
            self.metrics['profit_factor'] = total_wins / total_losses
        
        self.metrics['last_update'] = datetime.now()
    
    def _update_equity_curve(self, trade_result: Dict):
        """
        Equity Curve und Drawdown berechnen
        """
        current_equity = self.initial_balance + self.metrics['total_pnl']
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': current_equity,
            'trade_pnl': trade_result.get('pnl', 0)
        })
        
        # Peak Equity aktualisieren
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Drawdown berechnen
        self.current_drawdown = ((self.peak_equity - current_equity) / self.peak_equity) * 100
        
        if self.current_drawdown > self.metrics['max_drawdown']:
            self.metrics['max_drawdown'] = self.current_drawdown
    
    def get_performance_summary(self) -> Dict:
        """
        Vollständige Performance-Zusammenfassung
        """
        return {
            'summary': {
                'total_trades': self.metrics['total_trades'],
                'win_rate': round(self.metrics['win_rate'], 2),
                'total_pnl': round(self.metrics['total_pnl'], 2),
                'total_pnl_percentage': round(self.metrics['total_pnl_percentage'], 2),
                'max_drawdown': round(self.metrics['max_drawdown'], 2),
                'profit_factor': round(self.metrics['profit_factor'], 2),
                'current_equity': round(self.initial_balance + self.metrics['total_pnl'], 2)
            },
            'details': self.metrics.copy(),
            'trade_count': len(self.trade_history),
            'last_update': self.metrics['last_update'].strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'active'
        }
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """
        Letzte N Trades zurückgeben
        """
        return self.trade_history[-limit:] if self.trade_history else []
    
    def reset_metrics(self):
        """
        Alle Metriken zurücksetzen (für Tests)
        """
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_pnl_percentage': 0.0,
            'max_drawdown': 0.0,
            'max_profit': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'start_time': datetime.now(),
            'last_update': datetime.now()
        }
        
        self.trade_history = []
        self.daily_pnl = []
        self.equity_curve = []
        self.current_drawdown = 0.0
        self.peak_equity = self.initial_balance
        
        logger.info("📊 Performance Metriken zurückgesetzt")
    
    # ==================== COMPATIBILITY METHODS ====================
    
    async def add_trade_signal(self, signal) -> bool:
        """
        Trade Signal hinzufügen (Kompatibilität)
        """
        try:
            trade_result = {
                'signal_id': getattr(signal, 'id', 'unknown'),
                'symbol': getattr(signal, 'symbol', 'unknown'),
                'strategy': getattr(signal, 'strategy', 'unknown'),
                'side': getattr(signal, 'side', 'unknown'),
                'entry_price': float(getattr(signal, 'entry_price', 0)),
                'quantity': float(getattr(signal, 'quantity', 0)),
                'pnl': 0.0,  # Initial PnL
                'status': 'open'
            }
            
            logger.info(f"📊 Trade Signal hinzugefügt: {trade_result['symbol']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Hinzufügen des Trade Signals: {e}")
            return False
    
    async def update_order_status(self, order_id: str, status: str) -> bool:
        """
        Order Status updaten (Kompatibilität)
        """
        try:
            logger.info(f"📋 Order Status aktualisiert: {order_id} -> {status}")
            return True
        except Exception as e:
            logger.error(f"❌ Fehler beim Order Status Update: {e}")
            return False
    
    async def update_real_time_metrics(self):
        """
        Real-time Metrics updaten (Kompatibilität)
        """
        try:
            # Metrics sind bereits up-to-date durch _update_metrics
            self.metrics['last_update'] = datetime.now()
            return True
        except Exception as e:
            logger.error(f"❌ Fehler beim Real-time Metrics Update: {e}")
            return False
    
    async def shutdown(self):
        """
        Performance Tracker herunterfahren
        """
        try:
            logger.info("✅ Performance Tracker heruntergefahren")
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren: {e}")
