#!/usr/bin/env python3
"""
🛡️ ADVANCED RISK MANAGER DEMO

Demonstration des institutionellen Risikomanagement-Systems.
Zeigt alle Features: VaR, CVaR, Stress Testing, Kelly Criterion, Real-time Monitoring.

Author: TRADINO Development Team
"""

import sys
import os
import logging
import numpy as np
from datetime import datetime
from typing import List

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from risk.advanced_risk_manager import AdvancedRiskManager
from risk.risk_models import (
    Position, TradingSignal, MarketConditions, Scenario,
    RiskLevel, SignalType, AssetClass
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_positions() -> List[Position]:
    """💼 Create sample portfolio positions"""
    positions = [
        Position(
            symbol="BTCUSDT",
            asset_class=AssetClass.CRYPTO,
            size=0.5,
            entry_price=42000.0,
            current_price=45000.0,
            timestamp=datetime.now(),
            stop_loss=40000.0,
            take_profit=50000.0,
            leverage=2.0
        ),
        Position(
            symbol="ETHUSDT",
            asset_class=AssetClass.CRYPTO,
            size=5.0,
            entry_price=3000.0,
            current_price=3200.0,
            timestamp=datetime.now(),
            stop_loss=2800.0,
            take_profit=3500.0,
            leverage=1.5
        )
    ]
    return positions

def create_sample_signals() -> List[TradingSignal]:
    """📡 Create sample trading signals"""
    signals = [
        TradingSignal(
            symbol="SOLUSDT",
            signal_type=SignalType.BUY,
            confidence=0.75,
            strength=0.8,
            timestamp=datetime.now(),
            expected_return=0.15,
            risk_score=0.08
        )
    ]
    return signals

def create_market_conditions() -> MarketConditions:
    """🌍 Create sample market conditions"""
    return MarketConditions(
        timestamp=datetime.now(),
        volatility=0.25,
        trend="bullish",
        momentum=0.6,
        liquidity=0.8,
        correlation_regime="medium",
        risk_appetite=0.7,
        market_stress=0.3
    )

def simulate_price_history(risk_manager: AdvancedRiskManager):
    """📈 Simulate historical price data"""
    print("📈 Simulating historical price data...")
    symbols = [pos.symbol for pos in risk_manager.current_positions]
    
    for day in range(100):
        price_updates = {}
        for symbol in symbols:
            if symbol not in risk_manager.price_history:
                if symbol == "BTCUSDT":
                    current_price = 45000.0
                elif symbol == "ETHUSDT":
                    current_price = 3200.0
                else:
                    current_price = 100.0
            else:
                current_price = risk_manager.price_history[symbol][-1]
            
            daily_return = np.random.normal(0.001, 0.03)
            new_price = current_price * (1 + daily_return)
            price_updates[symbol] = new_price
        
        risk_manager.update_position_prices(price_updates)

def demonstrate_var_calculation(risk_manager: AdvancedRiskManager):
    """📊 Demonstrate VaR calculations"""
    print("\n" + "="*60)
    print("📊 VALUE AT RISK (VaR) DEMONSTRATION")
    print("="*60)
    
    positions = create_sample_positions()
    for position in positions:
        violations = risk_manager.add_position(position)
        if violations:
            print(f"⚠️ Position warnings: {violations}")
    
    simulate_price_history(risk_manager)
    
    var_95 = risk_manager.calculate_var(risk_manager.current_positions, 0.95)
    var_99 = risk_manager.calculate_var(risk_manager.current_positions, 0.99)
    es_95 = risk_manager.calculate_expected_shortfall(risk_manager.current_positions, 0.95)
    
    portfolio_value = risk_manager.get_portfolio_value()
    
    print(f"💰 Portfolio Value: ${portfolio_value:,.2f}")
    print(f"📈 VaR (95%): ${var_95:,.2f} ({var_95/portfolio_value*100:.2f}%)")
    print(f"📉 VaR (99%): ${var_99:,.2f} ({var_99/portfolio_value*100:.2f}%)")
    print(f"💸 Expected Shortfall: ${es_95:,.2f}")
    
    return risk_manager

def demonstrate_position_sizing(risk_manager: AdvancedRiskManager):
    """🎯 Demonstrate Kelly Criterion position sizing"""
    print("\n" + "="*60)
    print("🎯 KELLY CRITERION POSITION SIZING")
    print("="*60)
    
    signals = create_sample_signals()
    market_conditions = create_market_conditions()
    
    for signal in signals:
        optimal_size = risk_manager.optimal_position_size(signal, market_conditions)
        portfolio_value = risk_manager.get_portfolio_value()
        
        print(f"\n📡 Signal: {signal.symbol}")
        print(f"   Confidence: {signal.confidence*100:.1f}%")
        expected_return = signal.expected_return or 0
        risk_score = signal.risk_score or 0
        print(f"   Expected Return: {expected_return*100:.1f}%")
        print(f"   Risk Score: {risk_score*100:.1f}%")
        print(f"   🎯 Optimal Size: ${optimal_size:,.2f}")
        print(f"   📊 Allocation: {optimal_size/portfolio_value*100:.2f}%")

def demonstrate_stress_testing(risk_manager: AdvancedRiskManager):
    """🧪 Demonstrate stress testing"""
    print("\n" + "="*60)
    print("🧪 STRESS TESTING & SCENARIO ANALYSIS")
    print("="*60)
    
    scenarios = [
        Scenario.create_market_crash(),
        Scenario.create_flash_crash(),
        Scenario.create_regulatory_shock()
    ]
    
    stress_results = risk_manager.stress_test_portfolio(scenarios)
    portfolio_value = risk_manager.get_portfolio_value()
    
    print(f"💰 Portfolio Value: ${portfolio_value:,.2f}\n")
    
    for scenario_name, result in stress_results.items():
        if isinstance(result, dict):
            loss = result['total_loss']
            loss_pct = result['loss_percentage']
            probability = result['scenario_probability']
        else:
            loss = result
            loss_pct = (loss / portfolio_value) * 100
            probability = 0.1
        
        print(f"🎭 Scenario: {scenario_name}")
        print(f"   💥 Loss: ${loss:,.2f} ({loss_pct:.2f}%)")
        print(f"   📊 Probability: {probability*100:.1f}%")
        print()

def demonstrate_risk_monitoring(risk_manager: AdvancedRiskManager):
    """🚨 Demonstrate risk monitoring"""
    print("\n" + "="*60)
    print("🚨 RISK MONITORING & ALERTS")
    print("="*60)
    
    metrics = risk_manager.calculate_risk_metrics()
    
    print("📊 RISK METRICS:")
    print(f"   Risk Score: {metrics.overall_risk_score:.1f}/100")
    print(f"   Risk Level: {metrics.risk_level.value.upper()}")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   Leverage: {metrics.leverage_ratio:.2f}x")
    
    alerts = risk_manager.monitor_risk_limits()
    
    if alerts:
        print(f"\n🚨 ALERTS ({len(alerts)}):")
        for alert in alerts:
            print(f"   {alert.risk_level.value.upper()}: {alert.message}")
    else:
        print("\n✅ No risk breaches detected")

def demonstrate_portfolio_report(risk_manager: AdvancedRiskManager):
    """📋 Demonstrate risk report"""
    print("\n" + "="*60)
    print("📋 COMPREHENSIVE RISK REPORT")
    print("="*60)
    
    report = risk_manager.generate_risk_report()
    
    exec_summary = report['executive_summary']
    print("📊 EXECUTIVE SUMMARY:")
    print(f"   Risk Level: {exec_summary['overall_risk_level'].upper()}")
    print(f"   Portfolio: ${exec_summary['portfolio_value']:,.2f}")
    print(f"   VaR (99%): ${exec_summary['var_1d_99']:,.2f}")
    
    portfolio = report['portfolio_summary']
    print(f"\n💼 PORTFOLIO:")
    print(f"   Positions: {portfolio['total_positions']}")
    print(f"   Exposure: ${portfolio['total_exposure']:,.2f}")
    print(f"   Return: {portfolio['total_return_pct']:.2f}%")

def main():
    """🚀 Main demonstration"""
    print("🛡️ ADVANCED RISK MANAGEMENT SYSTEM DEMO")
    print("="*60)
    print("Institutional-grade risk management for quantitative trading")
    
    portfolio_size = 100000.0
    max_drawdown = 0.15
    
    risk_manager = AdvancedRiskManager(portfolio_size, max_drawdown)
    
    try:
        demonstrate_var_calculation(risk_manager)
        demonstrate_position_sizing(risk_manager)
        demonstrate_stress_testing(risk_manager)
        demonstrate_risk_monitoring(risk_manager)
        demonstrate_portfolio_report(risk_manager)
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("🛡️ Risk management system ready for production!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    main() 