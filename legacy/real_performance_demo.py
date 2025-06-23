#!/usr/bin/env python3
"""
📊 REAL PERFORMANCE ENGINE DEMO
Demonstration der echten Performance Analytics
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'tradino_unschlagbar'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

from tradino_unschlagbar.analytics.real_performance_engine import (
    RealPerformanceEngine, TradeRecord, PerformanceMetrics
)

def generate_realistic_trades(engine: RealPerformanceEngine, num_trades: int = 50):
    """🎲 Generiere realistische Trade-Daten"""
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 'XRP/USDT', 'DOT/USDT']
    strategies = ['AI_Sentiment', 'Market_Regime', 'Portfolio_Rebalance', 'Scalping', 'Trend_Following']
    
    base_time = datetime.now() - timedelta(days=60)
    
    for i in range(num_trades):
        # Random trade parameters
        symbol = np.random.choice(symbols)
        strategy = np.random.choice(strategies)
        side = np.random.choice(['buy', 'sell'])
        
        # Entry time (spread over 60 days)
        entry_time = base_time + timedelta(
            days=np.random.uniform(0, 60),
            hours=np.random.uniform(0, 24)
        )
        
        # Exit time (1-48 hours later)
        exit_time = entry_time + timedelta(hours=np.random.uniform(1, 48))
        
        # Price simulation based on symbol
        if 'BTC' in symbol:
            entry_price = np.random.uniform(95000, 105000)
            price_change = np.random.normal(0, 0.03)  # 3% std dev
            quantity = np.random.uniform(0.001, 0.01)
        elif 'ETH' in symbol:
            entry_price = np.random.uniform(3200, 3800)
            price_change = np.random.normal(0, 0.04)  # 4% std dev
            quantity = np.random.uniform(0.01, 0.1)
        else:
            entry_price = np.random.uniform(0.5, 50)
            price_change = np.random.normal(0, 0.05)  # 5% std dev
            quantity = np.random.uniform(1, 100)
        
        exit_price = entry_price * (1 + price_change)
        
        # Fee calculation
        fee = entry_price * quantity * 0.001  # 0.1% fee
        
        # PnL calculation
        if side == 'buy':
            pnl = (exit_price - entry_price) * quantity - fee
        else:
            pnl = (entry_price - exit_price) * quantity - fee
        
        # Strategy-based adjustments
        if strategy == 'AI_Sentiment':
            # AI sentiment tends to be more accurate
            if np.random.random() < 0.7:  # 70% win rate
                pnl = abs(pnl) if pnl < 0 else pnl
            confidence = np.random.uniform(0.7, 0.95)
        elif strategy == 'Market_Regime':
            # Market regime following
            if np.random.random() < 0.65:  # 65% win rate
                pnl = abs(pnl) if pnl < 0 else pnl
            confidence = np.random.uniform(0.6, 0.9)
        else:
            confidence = np.random.uniform(0.5, 0.8)
        
        trade = TradeRecord(
            trade_id=f"trade_{i+1:03d}",
            symbol=symbol,
            side=side,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            fee=fee,
            pnl=pnl,
            status='closed',
            strategy=strategy,
            confidence=confidence,
            market_regime=np.random.choice(['Bull', 'Bear', 'Sideways', 'High_Volatility']),
            sentiment_score=np.random.uniform(-1, 1)
        )
        
        engine.add_trade(trade)
    
    print(f"✅ {num_trades} realistische Trades generiert")

def demo_performance_analysis():
    """📊 Demo der Performance-Analyse"""
    
    print("🚀 REAL PERFORMANCE ENGINE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize Engine
    engine = RealPerformanceEngine("demo_performance.db")
    
    # Generate realistic trades
    print("\n📈 Generiere realistische Trading-Daten...")
    generate_realistic_trades(engine, 75)
    
    # Comprehensive metrics
    print("\n📊 Berechne umfassende Performance-Metriken...")
    metrics = engine.calculate_comprehensive_metrics()
    
    print(f"\n🎯 COMPREHENSIVE PERFORMANCE METRICS")
    print("=" * 45)
    print(f"📈 Returns:")
    print(f"├ Total Return: ${metrics.total_return:.2f}")
    print(f"├ Annual Return: {metrics.annual_return:.2%}")
    print(f"└ Volatility: {metrics.volatility:.2%}")
    print(f"")
    print(f"⚡ Risk-Adjusted:")
    print(f"├ Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"├ Sortino Ratio: {metrics.sortino_ratio:.3f}")
    print(f"├ Calmar Ratio: {metrics.calmar_ratio:.3f}")
    print(f"└ Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"")
    print(f"🎲 Trade Statistics:")
    print(f"├ Total Trades: {metrics.total_trades}")
    print(f"├ Win Rate: {metrics.win_rate:.2%}")
    print(f"├ Profit Factor: {metrics.profit_factor:.2f}")
    print(f"├ Avg Trade Duration: {metrics.avg_trade_duration:.1f} hours")
    print(f"├ Largest Win: ${metrics.largest_win:.2f}")
    print(f"├ Largest Loss: ${metrics.largest_loss:.2f}")
    print(f"├ Consecutive Wins: {metrics.consecutive_wins}")
    print(f"└ Consecutive Losses: {metrics.consecutive_losses}")
    print(f"")
    print(f"🔻 Risk Metrics:")
    print(f"├ VaR (95%): {metrics.var_95:.2%}")
    print(f"├ CVaR (95%): {metrics.cvar_95:.2%}")
    print(f"├ Beta: {metrics.beta:.3f}")
    print(f"├ Alpha: {metrics.alpha:.2%}")
    print(f"├ Tracking Error: {metrics.tracking_error:.2%}")
    print(f"└ Information Ratio: {metrics.information_ratio:.3f}")
    
    # Detailed performance report
    print(f"\n📋 DETAILED PERFORMANCE REPORT (30 Days)")
    print("=" * 50)
    report = engine.get_performance_report(30)
    
    print(f"Period: {report['period']}")
    print(f"Start: {report['start_date'][:10]}")
    print(f"End: {report['end_date'][:10]}")
    print(f"")
    print(f"📊 Overall Metrics:")
    for key, value in report['overall_metrics'].items():
        print(f"├ {key.replace('_', ' ').title()}: {value}")
    print(f"")
    print(f"📈 Trade Analysis:")
    for key, value in report['trade_analysis'].items():
        print(f"├ {key.replace('_', ' ').title()}: {value}")
    print(f"")
    print(f"🔻 Risk Metrics:")
    for key, value in report['risk_metrics'].items():
        print(f"├ {key.replace('_', ' ').title()}: {value}")
    
    # Strategy breakdown
    print(f"\n🎯 STRATEGY BREAKDOWN")
    print("=" * 25)
    strategy_data = report['strategy_breakdown']
    for strategy, data in sorted(strategy_data.items(), key=lambda x: x[1]['pnl'], reverse=True):
        win_rate = 0
        if data['trades'] > 0:
            # Calculate win rate for this strategy
            strategy_trades = [t for t in engine.trades if t.strategy == strategy and t.status == 'closed']
            wins = len([t for t in strategy_trades if t.pnl > 0])
            win_rate = wins / len(strategy_trades) if strategy_trades else 0
        
        print(f"├ {strategy}:")
        print(f"│  ├ Trades: {data['trades']}")
        print(f"│  ├ PnL: ${data['pnl']:.2f}")
        print(f"│  └ Win Rate: {win_rate:.1%}")
    
    # Symbol breakdown
    print(f"\n💰 SYMBOL BREAKDOWN")
    print("=" * 20)
    symbol_data = report['symbol_breakdown']
    for symbol, data in sorted(symbol_data.items(), key=lambda x: x[1]['pnl'], reverse=True):
        avg_pnl = data['pnl'] / data['trades'] if data['trades'] > 0 else 0
        print(f"├ {symbol}:")
        print(f"│  ├ Trades: {data['trades']}")
        print(f"│  ├ Total PnL: ${data['pnl']:.2f}")
        print(f"│  └ Avg PnL: ${avg_pnl:.2f}")
    
    # Equity curve
    print(f"\n📈 EQUITY CURVE (Last 30 Days)")
    print("=" * 35)
    equity_df = engine.get_equity_curve(30)
    
    if not equity_df.empty:
        # Show sample equity points
        sample_points = equity_df.iloc[::max(1, len(equity_df)//5)]  # Show 5 points max
        
        for _, row in sample_points.iterrows():
            print(f"├ {row['timestamp'].strftime('%m-%d %H:%M')}: "
                  f"${row['cumulative_pnl']:+.2f} "
                  f"({row['symbol']} - {row['strategy']})")
        
        final_pnl = equity_df['cumulative_pnl'].iloc[-1]
        print(f"└ Final PnL: ${final_pnl:+.2f}")
    
    # Monthly performance
    print(f"\n📅 MONTHLY PERFORMANCE")
    print("=" * 25)
    monthly_df = engine.get_monthly_performance()
    
    if not monthly_df.empty:
        for month, row in monthly_df.tail(6).iterrows():  # Last 6 months
            print(f"├ {month.strftime('%Y-%m')}:")
            print(f"│  ├ Trades: {row['trades']}")
            print(f"│  ├ PnL: ${row['pnl']:+.2f}")
            print(f"│  └ Win Rate: {row['win_rate']:.1%}")
    
    # Save performance snapshot
    print(f"\n📸 Speichere Performance Snapshot...")
    engine.save_performance_snapshot()
    
    print(f"\n🎯 REAL PERFORMANCE ENGINE DEMO ABGESCHLOSSEN!")
    print("=" * 50)
    print("✅ Alle Metriken basieren auf echten Berechnungen")
    print("📊 SQLite Database mit persistenter Speicherung")
    print("📈 Umfassende Performance-Analytics")
    print("🎲 Realistische Trading-Simulation")
    print("💾 Performance Snapshots für Tracking")
    print("📋 Detaillierte Reports verfügbar")
    print("\n🚀 TRADINO UNSCHLAGBAR Real Performance Engine Ready!")

if __name__ == "__main__":
    demo_performance_analysis() 