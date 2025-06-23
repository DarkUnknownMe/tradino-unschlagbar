#!/usr/bin/env python3
"""
📈 PERFORMANCE MONITORING SYSTEM
Live performance tracking für TRADINO UNSCHLAGBAR
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class PerformanceMonitoringSystem:
    """📈 Live Performance Monitoring System"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self.get_default_config()
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
        
        # Time-based metrics
        self.daily_metrics = {}
        self.weekly_metrics = {}
        self.monthly_metrics = {}
        
        # Trade tracking
        self.trade_history = []
        self.equity_curve = []
        self.balance_history = []
        
        # Performance state
        self.initial_balance = 0.0
        self.current_balance = 0.0
        self.peak_balance = 0.0
        self.start_time = datetime.now()
        
        print("📈 Performance Monitoring System initialized")
    
    def get_default_config(self) -> Dict[str, Any]:
        """⚙️ Default configuration"""
        
        return {
            'track_equity_curve': True,
            'save_daily_reports': True,
            'benchmark_symbol': 'BTC/USDT',
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'update_interval': 60,  # Update every 60 seconds
            'max_history_length': 10000,
            'alert_on_drawdown': True,
            'drawdown_alert_threshold': 0.1,  # 10%
            'performance_alert_threshold': -0.05,  # -5% daily loss
            'save_detailed_logs': True
        }
    
    def update_balance(self, new_balance: float):
        """💰 Update current balance"""
        
        if self.initial_balance == 0.0:
            self.initial_balance = new_balance
            self.peak_balance = new_balance
        
        self.current_balance = new_balance
        
        # Update peak and drawdown
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
        
        # Calculate current drawdown
        self.metrics['current_drawdown'] = (self.peak_balance - new_balance) / self.peak_balance
        
        # Update max drawdown
        if self.metrics['current_drawdown'] > self.metrics['max_drawdown']:
            self.metrics['max_drawdown'] = self.metrics['current_drawdown']
        
        # Record balance history
        self.balance_history.append({
            'timestamp': datetime.now().isoformat(),
            'balance': new_balance,
            'drawdown': self.metrics['current_drawdown']
        })
        
        # Keep history limited
        if len(self.balance_history) > self.config['max_history_length']:
            self.balance_history = self.balance_history[-5000:]
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """📊 Record completed trade"""
        
        try:
            # Extract trade information
            result = trade_data.get('result', {})
            if not result.get('success'):
                return
            
            pnl = result.get('pnl', result.get('simulated_pnl', 0))
            if pnl == 0:
                return
            
            # Record trade
            trade_record = {
                'timestamp': trade_data.get('timestamp', datetime.now().isoformat()),
                'symbol': trade_data.get('symbol', 'Unknown'),
                'action': trade_data.get('signal', {}).get('action', 'Unknown'),
                'pnl': pnl,
                'confidence': trade_data.get('signal', {}).get('confidence', 0),
                'balance_after': self.current_balance
            }
            
            self.trade_history.append(trade_record)
            
            # Update metrics
            self.update_trade_metrics(trade_record)
            
            # Update time-based metrics
            self.update_time_based_metrics(trade_record)
            
            # Check for alerts
            self.check_performance_alerts()
            
            print(f"📊 Trade recorded: {trade_record['symbol']} "
                  f"PnL: ${pnl:.2f} | Total: ${self.metrics['total_pnl']:.2f}")
            
        except Exception as e:
            print(f"❌ Error recording trade: {e}")
    
    def update_trade_metrics(self, trade: Dict[str, Any]):
        """📊 Update trade-based metrics"""
        
        pnl = trade['pnl']
        
        # Basic counts
        self.metrics['total_trades'] += 1
        self.metrics['total_pnl'] += pnl
        
        if pnl > 0:
            # Winning trade
            self.metrics['winning_trades'] += 1
            self.metrics['consecutive_wins'] += 1
            self.metrics['consecutive_losses'] = 0
            
            if self.metrics['consecutive_wins'] > self.metrics['max_consecutive_wins']:
                self.metrics['max_consecutive_wins'] = self.metrics['consecutive_wins']
            
            if pnl > self.metrics['largest_win']:
                self.metrics['largest_win'] = pnl
            
            if pnl > self.metrics['max_profit']:
                self.metrics['max_profit'] = pnl
        
        else:
            # Losing trade
            self.metrics['losing_trades'] += 1
            self.metrics['consecutive_losses'] += 1
            self.metrics['consecutive_wins'] = 0
            
            if self.metrics['consecutive_losses'] > self.metrics['max_consecutive_losses']:
                self.metrics['max_consecutive_losses'] = self.metrics['consecutive_losses']
            
            if abs(pnl) > abs(self.metrics['largest_loss']):
                self.metrics['largest_loss'] = pnl
            
            if pnl < self.metrics['max_loss']:
                self.metrics['max_loss'] = pnl
        
        # Calculate ratios
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
        
        if self.metrics['winning_trades'] > 0:
            self.metrics['average_win'] = sum(t['pnl'] for t in self.trade_history if t['pnl'] > 0) / self.metrics['winning_trades']
        
        if self.metrics['losing_trades'] > 0:
            self.metrics['average_loss'] = sum(t['pnl'] for t in self.trade_history if t['pnl'] < 0) / self.metrics['losing_trades']
        
        # Profit factor
        total_wins = sum(t['pnl'] for t in self.trade_history if t['pnl'] > 0)
        total_losses = abs(sum(t['pnl'] for t in self.trade_history if t['pnl'] < 0))
        
        if total_losses > 0:
            self.metrics['profit_factor'] = total_wins / total_losses
        else:
            self.metrics['profit_factor'] = float('inf') if total_wins > 0 else 0
        
        # Calculate advanced ratios
        self.calculate_advanced_metrics()
    
    def calculate_advanced_metrics(self):
        """📊 Calculate advanced performance metrics"""
        
        if len(self.trade_history) < 10:
            return
        
        # Get returns
        returns = [trade['pnl'] / self.initial_balance for trade in self.trade_history[-100:]]
        
        if len(returns) < 5:
            return
        
        returns = np.array(returns)
        
        # Sharpe Ratio
        excess_returns = returns - (self.config['risk_free_rate'] / 252)  # Daily risk-free rate
        if np.std(returns) > 0:
            self.metrics['sharpe_ratio'] = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns)
            if downside_deviation > 0:
                self.metrics['sortino_ratio'] = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
        
        # Calmar Ratio
        if self.metrics['max_drawdown'] > 0:
            annual_return = (self.current_balance / self.initial_balance) ** (252 / len(self.trade_history)) - 1
            self.metrics['calmar_ratio'] = annual_return / self.metrics['max_drawdown']
    
    def update_time_based_metrics(self, trade: Dict[str, Any]):
        """📅 Update time-based performance metrics"""
        
        trade_date = datetime.fromisoformat(trade['timestamp']).date()
        today = datetime.now().date()
        
        # Daily metrics
        if trade_date == today:
            if today not in self.daily_metrics:
                self.daily_metrics[today] = {
                    'trades': 0,
                    'pnl': 0.0,
                    'wins': 0,
                    'losses': 0,
                    'start_balance': self.current_balance - trade['pnl']
                }
            
            daily = self.daily_metrics[today]
            daily['trades'] += 1
            daily['pnl'] += trade['pnl']
            
            if trade['pnl'] > 0:
                daily['wins'] += 1
            else:
                daily['losses'] += 1
            
            daily['end_balance'] = self.current_balance
            daily['return'] = (daily['end_balance'] - daily['start_balance']) / daily['start_balance']
        
        # Keep only recent daily metrics
        if len(self.daily_metrics) > 90:  # Keep 90 days
            old_dates = sorted(self.daily_metrics.keys())[:-90]
            for old_date in old_dates:
                del self.daily_metrics[old_date]
    
    def get_daily_performance(self, date: datetime.date = None) -> Dict[str, Any]:
        """📅 Get daily performance"""
        
        if date is None:
            date = datetime.now().date()
        
        if date in self.daily_metrics:
            daily = self.daily_metrics[date]
            
            return {
                'date': date.isoformat(),
                'trades': daily['trades'],
                'pnl': daily['pnl'],
                'return_percent': daily.get('return', 0) * 100,
                'wins': daily['wins'],
                'losses': daily['losses'],
                'win_rate': daily['wins'] / max(daily['trades'], 1),
                'start_balance': daily.get('start_balance', 0),
                'end_balance': daily.get('end_balance', 0)
            }
        
        return {
            'date': date.isoformat(),
            'trades': 0,
            'pnl': 0.0,
            'return_percent': 0.0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """📊 Get comprehensive performance summary"""
        
        runtime = datetime.now() - self.start_time
        runtime_days = runtime.total_seconds() / 86400
        
        # Calculate returns
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
        
        # Annual return (if running long enough)
        annual_return = 0.0
        if runtime_days > 0:
            annual_return = (1 + total_return) ** (365.25 / runtime_days) - 1
        
        return {
            'overview': {
                'initial_balance': self.initial_balance,
                'current_balance': self.current_balance,
                'total_pnl': self.metrics['total_pnl'],
                'total_return_percent': total_return * 100,
                'annual_return_percent': annual_return * 100,
                'runtime_days': runtime_days
            },
            'trading_stats': {
                'total_trades': self.metrics['total_trades'],
                'winning_trades': self.metrics['winning_trades'],
                'losing_trades': self.metrics['losing_trades'],
                'win_rate_percent': self.metrics['win_rate'] * 100,
                'profit_factor': self.metrics['profit_factor'],
                'average_win': self.metrics['average_win'],
                'average_loss': self.metrics['average_loss'],
                'largest_win': self.metrics['largest_win'],
                'largest_loss': self.metrics['largest_loss']
            },
            'risk_metrics': {
                'max_drawdown_percent': self.metrics['max_drawdown'] * 100,
                'current_drawdown_percent': self.metrics['current_drawdown'] * 100,
                'sharpe_ratio': self.metrics['sharpe_ratio'],
                'sortino_ratio': self.metrics['sortino_ratio'],
                'calmar_ratio': self.metrics['calmar_ratio'],
                'max_consecutive_wins': self.metrics['max_consecutive_wins'],
                'max_consecutive_losses': self.metrics['max_consecutive_losses']
            },
            'current_streak': {
                'consecutive_wins': self.metrics['consecutive_wins'],
                'consecutive_losses': self.metrics['consecutive_losses']
            },
            'daily_performance': self.get_daily_performance(),
            'last_update': datetime.now().isoformat()
        }
    
    def check_performance_alerts(self):
        """🚨 Check for performance alerts"""
        
        alerts = []
        
        # Drawdown alert
        if (self.config['alert_on_drawdown'] and 
            self.metrics['current_drawdown'] > self.config['drawdown_alert_threshold']):
            alerts.append({
                'type': 'drawdown_alert',
                'message': f"High drawdown: {self.metrics['current_drawdown']:.1%}",
                'severity': 'high',
                'timestamp': datetime.now().isoformat()
            })
        
        # Daily performance alert
        today_performance = self.get_daily_performance()
        if today_performance['return_percent'] < self.config['performance_alert_threshold'] * 100:
            alerts.append({
                'type': 'daily_loss_alert',
                'message': f"Daily loss: {today_performance['return_percent']:.1f}%",
                'severity': 'medium',
                'timestamp': datetime.now().isoformat()
            })
        
        # Consecutive losses alert
        if self.metrics['consecutive_losses'] >= 5:
            alerts.append({
                'type': 'consecutive_losses_alert',
                'message': f"Consecutive losses: {self.metrics['consecutive_losses']}",
                'severity': 'medium',
                'timestamp': datetime.now().isoformat()
            })
        
        # Print alerts
        for alert in alerts:
            print(f"🚨 PERFORMANCE ALERT: {alert['message']}")
        
        return alerts
    
    def get_equity_curve_data(self) -> List[Dict[str, Any]]:
        """📈 Get equity curve data"""
        
        if not self.balance_history:
            return []
        
        # Sample data for plotting (max 1000 points)
        if len(self.balance_history) <= 1000:
            return self.balance_history
        
        # Sample every nth point
        step = len(self.balance_history) // 1000
        return self.balance_history[::step]
    
    def get_trade_distribution(self) -> Dict[str, Any]:
        """📊 Get trade P&L distribution"""
        
        if not self.trade_history:
            return {}
        
        pnl_values = [trade['pnl'] for trade in self.trade_history]
        
        return {
            'mean': np.mean(pnl_values),
            'median': np.median(pnl_values),
            'std': np.std(pnl_values),
            'min': np.min(pnl_values),
            'max': np.max(pnl_values),
            'percentiles': {
                '25th': np.percentile(pnl_values, 25),
                '75th': np.percentile(pnl_values, 75),
                '90th': np.percentile(pnl_values, 90),
                '95th': np.percentile(pnl_values, 95)
            },
            'positive_trades': len([p for p in pnl_values if p > 0]),
            'negative_trades': len([p for p in pnl_values if p < 0]),
            'total_trades': len(pnl_values)
        }
    
    def save_performance_report(self):
        """💾 Save performance report"""
        
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'performance_summary': self.get_performance_summary(),
                'trade_distribution': self.get_trade_distribution(),
                'recent_trades': self.trade_history[-50:] if self.trade_history else [],
                'daily_metrics': {
                    str(date): metrics for date, metrics in 
                    list(self.daily_metrics.items())[-30:]  # Last 30 days
                }
            }
            
            os.makedirs('performance_reports', exist_ok=True)
            filename = f"performance_reports/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"💾 Performance report saved: {filename}")
            
        except Exception as e:
            print(f"❌ Error saving performance report: {e}")
    
    def reset_daily_metrics(self):
        """🔄 Reset for new trading day"""
        
        # This would be called at the start of each trading day
        today = datetime.now().date()
        print(f"🔄 New trading day: {today}")
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """📊 Get performance dashboard data"""
        
        return {
            'summary': self.get_performance_summary(),
            'today': self.get_daily_performance(),
            'alerts': self.check_performance_alerts(),
            'equity_curve': self.get_equity_curve_data()[-100:],  # Last 100 points
            'recent_trades': self.trade_history[-10:] if self.trade_history else [],
            'metrics': self.metrics,
            'distribution': self.get_trade_distribution()
        }

# Global instance
performance_monitor = None

def initialize_performance_monitoring(config: Dict[str, Any] = None) -> PerformanceMonitoringSystem:
    """📈 Initialize performance monitoring"""
    global performance_monitor
    performance_monitor = PerformanceMonitoringSystem(config)
    return performance_monitor

def get_performance_monitor() -> Optional[PerformanceMonitoringSystem]:
    """📈 Get performance monitor"""
    return performance_monitor

# Demo
if __name__ == "__main__":
    print("📈 PERFORMANCE MONITORING SYSTEM TEST")
    
    # Initialize
    pm = initialize_performance_monitoring()
    
    # Simulate initial balance
    pm.update_balance(10000.0)
    
    # Simulate some trades
    trades = [
        {'symbol': 'BTC/USDT', 'result': {'success': True, 'pnl': 150.0}, 'signal': {'action': 'buy', 'confidence': 0.8}},
        {'symbol': 'ETH/USDT', 'result': {'success': True, 'pnl': -75.0}, 'signal': {'action': 'sell', 'confidence': 0.7}},
        {'symbol': 'BTC/USDT', 'result': {'success': True, 'pnl': 200.0}, 'signal': {'action': 'buy', 'confidence': 0.9}},
    ]
    
    for i, trade in enumerate(trades):
        pm.update_balance(10000 + sum(t['result']['pnl'] for t in trades[:i+1]))
        pm.record_trade(trade)
    
    # Get summary
    summary = pm.get_performance_summary()
    print(f"✅ Performance Summary: Win Rate: {summary['trading_stats']['win_rate_percent']:.1f}%")
    print(f"📊 Total PnL: ${summary['overview']['total_pnl']:.2f}")
    
    dashboard = pm.get_performance_dashboard()
    print("📈 Performance monitoring ready!") 