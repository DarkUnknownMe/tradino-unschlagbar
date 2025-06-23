#!/usr/bin/env python3
"""
🚀 PORTFOLIO OPTIMIZATION DEMO
Demonstration des weltklasse Portfolio Optimization Systems
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'tradino_unschlagbar'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio

from tradino_unschlagbar.brain.portfolio_optimizer import WorldClassPortfolioOptimizer
from tradino_unschlagbar.brain.portfolio_integration import TradingPortfolioManager

async def main():
    """🎯 Hauptdemo Funktion"""
    
    print("🚀 TRADINO UNSCHLAGBAR - PORTFOLIO OPTIMIZATION DEMO")
    print("=" * 60)
    
    # 1. Simuliere Crypto Asset Returns
    print("\n📊 Generiere Crypto Market Data...")
    np.random.seed(42)
    
    # Top Crypto Assets
    assets = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'AVAX', 'LINK', 'UNI']
    
    # 2 Jahre historische Daten
    start_date = datetime.now() - timedelta(days=730)
    dates = pd.date_range(start_date, datetime.now(), freq='D')
    
    # Generiere realistische Crypto Returns
    n_assets = len(assets)
    
    # Crypto-spezifische Parameter
    annual_returns = {
        'BTC': 0.40,   'ETH': 0.65,   'BNB': 0.35,   'ADA': 0.25,   'SOL': 0.80,
        'XRP': 0.20,   'DOT': 0.30,   'AVAX': 0.45,  'LINK': 0.35,  'UNI': 0.40
    }
    
    annual_vols = {
        'BTC': 0.65,   'ETH': 0.75,   'BNB': 0.70,   'ADA': 0.85,   'SOL': 0.95,
        'XRP': 0.80,   'DOT': 0.85,   'AVAX': 0.90,  'LINK': 0.80,  'UNI': 0.85
    }
    
    # Korrelationsmatrix (Crypto ist hoch korreliert)
    base_correlation = 0.7
    correlation_matrix = np.full((n_assets, n_assets), base_correlation)
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Etwas Variation hinzufügen
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            correlation_matrix[i,j] = correlation_matrix[j,i] = base_correlation + np.random.uniform(-0.2, 0.2)
    
    # Generiere tägliche Returns
    daily_returns_data = []
    for date in dates:
        # Marktzyklen simulieren (Bull/Bear)
        market_sentiment = np.sin((date - dates[0]).days / 365 * 2 * np.pi) * 0.3
        
        # Tägliche Returns für alle Assets
        daily_rets = []
        for asset in assets:
            base_return = annual_returns[asset] / 365 + market_sentiment / 365
            volatility = annual_vols[asset] / np.sqrt(365)
            daily_ret = np.random.normal(base_return, volatility)
            daily_rets.append(daily_ret)
        
        # Korrelierte Returns anwenden
        correlated_rets = np.random.multivariate_normal(
            mean=daily_rets,
            cov=correlation_matrix * 0.001,  # Kleinerer Cov für tägliche Returns
            size=1
        )[0]
        
        daily_returns_data.append(correlated_rets)
    
    # DataFrame erstellen
    returns_df = pd.DataFrame(daily_returns_data, index=dates, columns=assets)
    
    # Preise aus Returns generieren
    initial_prices = {'BTC': 30000, 'ETH': 2000, 'BNB': 300, 'ADA': 0.5, 'SOL': 50,
                     'XRP': 0.6, 'DOT': 10, 'AVAX': 20, 'LINK': 15, 'UNI': 8}
    
    prices_df = pd.DataFrame(index=dates, columns=assets)
    for asset in assets:
        prices_df[asset] = initial_prices[asset] * (1 + returns_df[asset]).cumprod()
    
    print(f"✅ {len(assets)} Assets, {len(dates)} Tage generiert")
    print(f"📈 Zeitraum: {dates[0].strftime('%Y-%m-%d')} bis {dates[-1].strftime('%Y-%m-%d')}")
    
    # 2. Portfolio Optimizer initialisieren
    print("\n🧠 Initialisiere Portfolio Optimizer...")
    optimizer = WorldClassPortfolioOptimizer()
    optimizer.add_assets(returns_df)
    
    # 3. Verschiedene Optimierungsmethoden testen
    print("\n⚡ Führe Portfolio Optimierungen durch...")
    
    # Vergleiche alle Methoden
    comparison = optimizer.compare_optimization_methods()
    print("\n📊 OPTIMIERUNGSMETHODEN-VERGLEICH:")
    print(comparison.to_string(index=False))
    
    # 4. Beste Lösung analysieren
    best_result = max(optimizer.optimization_history, 
                     key=lambda x: x.sharpe_ratio if x.success else -999)
    
    print(f"\n🏆 BESTE PORTFOLIO-LÖSUNG: {best_result.optimization_method}")
    print(f"📈 Expected Return: {best_result.expected_return:.2%}")
    print(f"📊 Volatility: {best_result.volatility:.2%}")
    print(f"⚡ Sharpe Ratio: {best_result.sharpe_ratio:.3f}")
    print(f"📉 Max Drawdown: {best_result.max_drawdown:.2%}")
    print(f"🔻 VaR (95%): {best_result.var_95:.2%}")
    print(f"💥 CVaR (95%): {best_result.cvar_95:.2%}")
    
    # 5. Portfolio Zusammensetzung
    print(f"\n📈 PORTFOLIO ZUSAMMENSETZUNG ({best_result.optimization_method}):")
    composition = optimizer.get_portfolio_composition(best_result)
    print(composition.to_string(index=False))
    
    # 6. Trading Portfolio Manager Demo
    print("\n🔗 Trading Portfolio Manager Demo...")
    portfolio_manager = TradingPortfolioManager(optimization_interval_hours=6)
    
    # Initialisiere mit historischen Daten
    success = await portfolio_manager.initialize_portfolio(assets, prices_df)
    
    if success:
        print("✅ Portfolio Manager erfolgreich initialisiert")
        
        # Performance Metriken
        performance = portfolio_manager.get_portfolio_performance()
        print(f"\n📊 PORTFOLIO PERFORMANCE:")
        print(f"🎯 Methode: {performance['method']}")
        print(f"📈 Expected Return: {performance['expected_return']:.2%}")
        print(f"📊 Volatility: {performance['volatility']:.2%}")
        print(f"⚡ Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
        
        # Asset Allocation
        allocation = portfolio_manager.get_asset_allocation()
        print(f"\n💰 ASSET ALLOCATION:")
        print(allocation.to_string(index=False))
        
        # 7. Rebalancing Simulation
        print("\n🔄 Rebalancing Simulation...")
        
        # Simuliere aktuelle Positionen (abweichend von Optimal)
        current_positions = {
            'BTC': 40000,  'ETH': 25000,  'BNB': 5000,   'ADA': 3000,   'SOL': 8000,
            'XRP': 2000,   'DOT': 4000,   'AVAX': 6000,  'LINK': 7000,  'UNI': 0  # UNI komplett verkauft
        }
        
        rebalance_needed = await portfolio_manager.check_rebalancing_needed(current_positions)
        print(f"🔄 Rebalancing erforderlich: {'JA' if rebalance_needed else 'NEIN'}")
        
        if rebalance_needed:
            orders = await portfolio_manager.generate_rebalancing_orders(current_positions, 10000)
            print(f"\n📋 REBALANCING ORDERS ({len(orders)}):")
            for order in orders[:5]:  # Top 5 Orders
                print(f"  {order['side'].upper()} {order['asset']}: ${order['amount']:.0f} (Priorität: {order['priority']:.2%})")
    
    # 8. Black-Litterman mit Investor Views Demo
    print("\n🔮 Black-Litterman mit Investor Views...")
    
    # Beispiel Views: Bullish auf SOL und ETH, Bearish auf XRP
    views = {
        'SOL': 0.60,   # 60% erwarteter Return
        'ETH': 0.50,   # 50% erwarteter Return  
        'XRP': -0.10   # -10% erwarteter Return
    }
    
    confidence = {
        'SOL': 0.8,    # 80% Vertrauen
        'ETH': 0.9,    # 90% Vertrauen
        'XRP': 0.7     # 70% Vertrauen
    }
    
    bl_result = optimizer.optimize_black_litterman(views, confidence)
    
    if bl_result.success:
        print(f"✅ Black-Litterman Optimization erfolgreich")
        print(f"📈 Expected Return: {bl_result.expected_return:.2%}")
        print(f"⚡ Sharpe Ratio: {bl_result.sharpe_ratio:.3f}")
        
        bl_composition = optimizer.get_portfolio_composition(bl_result)
        print(f"\n📈 BLACK-LITTERMAN PORTFOLIO:")
        print(bl_composition.to_string(index=False))
    
    print("\n🎯 PORTFOLIO OPTIMIZATION DEMO ABGESCHLOSSEN!")
    print("🚀 TRADINO UNSCHLAGBAR ist bereit für Weltklasse Portfolio Management!")

if __name__ == "__main__":
    asyncio.run(main()) 