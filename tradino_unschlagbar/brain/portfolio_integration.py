#!/usr/bin/env python3
"""
🔗 PORTFOLIO INTEGRATION ENGINE
Verbindet Portfolio Optimization mit Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import asyncio
import logging

from .portfolio_optimizer import WorldClassPortfolioOptimizer, OptimizationResult

class TradingPortfolioManager:
    """🎯 Portfolio Manager für Trading System Integration"""
    
    def __init__(self, optimization_interval_hours: int = 24):
        self.optimizer = WorldClassPortfolioOptimizer()
        self.optimization_interval = optimization_interval_hours
        self.last_optimization = None
        self.current_portfolio = None
        self.target_weights = None
        self.rebalance_threshold = 0.05  # 5% Abweichung löst Rebalancing aus
        
        self.logger = logging.getLogger(__name__)
        
    async def initialize_portfolio(self, assets: List[str], historical_data: pd.DataFrame):
        """🚀 Initialisiere Portfolio mit historischen Daten"""
        
        try:
            # Berechne Returns
            returns_data = historical_data.pct_change().dropna()
            
            # Füge Assets zum Optimizer hinzu
            self.optimizer.add_assets(returns_data)
            
            # Führe initiale Optimierung durch
            await self.optimize_portfolio()
            
            self.logger.info(f"✅ Portfolio mit {len(assets)} Assets initialisiert")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio Initialisierung fehlgeschlagen: {e}")
            return False
    
    async def optimize_portfolio(self, method: str = "auto") -> OptimizationResult:
        """⚡ Portfolio Optimierung durchführen"""
        
        try:
            if method == "auto":
                # Automatische Methodenwahl basierend auf Marktbedingungen
                method = self._select_optimal_method()
            
            # Führe Optimierung durch
            if method == "markowitz":
                result = self.optimizer.optimize_markowitz()
            elif method == "risk_parity":
                result = self.optimizer.optimize_risk_parity()
            elif method == "max_diversification":
                result = self.optimizer.optimize_maximum_diversification()
            elif method == "cvar":
                result = self.optimizer.optimize_cvar()
            else:
                # Fallback: Vergleiche alle Methoden
                comparison_df = self.optimizer.compare_optimization_methods()
                # Wähle beste basierend auf Sharpe Ratio
                best_result = max(self.optimizer.optimization_history[-4:], 
                                key=lambda x: x.sharpe_ratio if x.success else -999)
                result = best_result
            
            if result.success:
                self.current_portfolio = result
                self.target_weights = result.weights
                self.last_optimization = datetime.now()
                
                self.logger.info(f"🎯 Portfolio optimiert: {result.optimization_method}")
                self.logger.info(f"📊 Sharpe Ratio: {result.sharpe_ratio:.3f}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio Optimierung fehlgeschlagen: {e}")
            return None
    
    def _select_optimal_method(self) -> str:
        """🧠 Wähle optimale Optimierungsmethode basierend auf Marktbedingungen"""
        
        if self.optimizer.returns is None:
            return "markowitz"
        
        # Analysiere Marktbedingungen
        recent_returns = self.optimizer.returns.tail(30)
        
        # Volatilität
        avg_volatility = recent_returns.std().mean()
        
        # Korrelation
        correlation_matrix = recent_returns.corr()
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        
        # Trend
        trend_strength = (recent_returns.mean() / recent_returns.std()).mean()
        
        # Methodenwahl basierend auf Bedingungen
        if avg_volatility > 0.03:  # Hohe Volatilität
            return "cvar"  # Fokus auf Downside Risk
        elif avg_correlation > 0.7:  # Hohe Korrelation
            return "max_diversification"  # Maximiere Diversifikation
        elif abs(trend_strength) < 0.1:  # Seitwärtsmarkt
            return "risk_parity"  # Gleichmäßige Risikoverteilung
        else:
            return "markowitz"  # Standard für normale Bedingungen
    
    async def check_rebalancing_needed(self, current_positions: Dict[str, float]) -> bool:
        """🔄 Prüfe ob Rebalancing erforderlich ist"""
        
        if self.target_weights is None:
            return True
        
        # Berechne aktuelle Gewichte
        total_value = sum(current_positions.values())
        if total_value == 0:
            return True
        
        current_weights = {asset: value/total_value for asset, value in current_positions.items()}
        
        # Prüfe Abweichungen
        max_deviation = 0
        for i, asset in enumerate(self.optimizer.assets):
            current_weight = current_weights.get(asset, 0)
            target_weight = self.target_weights[i]
            deviation = abs(current_weight - target_weight)
            max_deviation = max(max_deviation, deviation)
        
        rebalance_needed = max_deviation > self.rebalance_threshold
        
        if rebalance_needed:
            self.logger.info(f"🔄 Rebalancing erforderlich - Max Abweichung: {max_deviation:.2%}")
        
        return rebalance_needed
    
    async def generate_rebalancing_orders(self, current_positions: Dict[str, float], 
                                        available_balance: float) -> List[Dict]:
        """📋 Generiere Rebalancing Orders"""
        
        if self.target_weights is None:
            return []
        
        orders = []
        total_portfolio_value = sum(current_positions.values()) + available_balance
        
        for i, asset in enumerate(self.optimizer.assets):
            target_weight = self.target_weights[i]
            target_value = total_portfolio_value * target_weight
            current_value = current_positions.get(asset, 0)
            
            difference = target_value - current_value
            
            if abs(difference) > total_portfolio_value * 0.01:  # Min 1% des Portfolios
                if difference > 0:
                    # Kauforder
                    orders.append({
                        'asset': asset,
                        'side': 'buy',
                        'amount': difference,
                        'type': 'rebalance',
                        'priority': abs(difference / total_portfolio_value)
                    })
                else:
                    # Verkaufsorder
                    orders.append({
                        'asset': asset,
                        'side': 'sell',
                        'amount': abs(difference),
                        'type': 'rebalance',
                        'priority': abs(difference / total_portfolio_value)
                    })
        
        # Sortiere nach Priorität
        orders.sort(key=lambda x: x['priority'], reverse=True)
        
        self.logger.info(f"📋 {len(orders)} Rebalancing Orders generiert")
        return orders
    
    def get_portfolio_performance(self) -> Dict[str, Any]:
        """📊 Portfolio Performance Metriken"""
        
        if self.current_portfolio is None:
            return {}
        
        return {
            'method': self.current_portfolio.optimization_method,
            'expected_return': self.current_portfolio.expected_return,
            'volatility': self.current_portfolio.volatility,
            'sharpe_ratio': self.current_portfolio.sharpe_ratio,
            'max_drawdown': self.current_portfolio.max_drawdown,
            'var_95': self.current_portfolio.var_95,
            'cvar_95': self.current_portfolio.cvar_95,
            'last_optimization': self.last_optimization,
            'target_weights': dict(zip(self.optimizer.assets, self.target_weights)) if self.target_weights is not None else {}
        }
    
    def get_asset_allocation(self) -> pd.DataFrame:
        """📈 Aktuelle Asset Allocation"""
        
        if self.current_portfolio is None:
            return pd.DataFrame()
        
        return self.optimizer.get_portfolio_composition(self.current_portfolio)
    
    async def update_market_data(self, new_data: pd.DataFrame):
        """🔄 Aktualisiere Marktdaten für kontinuierliche Optimierung"""
        
        try:
            # Füge neue Daten hinzu
            if self.optimizer.returns is not None:
                # Kombiniere mit existierenden Daten
                combined_data = pd.concat([self.optimizer.returns, new_data]).tail(1000)  # Behalte letzten 1000 Tage
                returns_data = combined_data.pct_change().dropna()
            else:
                returns_data = new_data.pct_change().dropna()
            
            # Update Optimizer
            self.optimizer.add_assets(returns_data)
            
            # Prüfe ob Re-Optimierung nötig
            if self.last_optimization is None or \
               datetime.now() - self.last_optimization > timedelta(hours=self.optimization_interval):
                await self.optimize_portfolio()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Marktdaten Update fehlgeschlagen: {e}")
            return False
    
    def set_optimization_constraints(self, constraints: Dict[str, Any]):
        """⚙️ Setze Optimierungsconstraints"""
        
        # Portfolio-Level Constraints
        self.rebalance_threshold = constraints.get('rebalance_threshold', 0.05)
        self.optimization_interval = constraints.get('optimization_interval_hours', 24)
        
        # Asset-Level Constraints könnten hier implementiert werden
        # z.B. Min/Max Gewichte pro Asset, Sektorgrenzen, etc.
        
        self.logger.info("⚙️ Optimierungsconstraints aktualisiert")

# Integration mit Trading System
class PortfolioTradingIntegration:
    """🔗 Integration zwischen Portfolio Manager und Trading Engine"""
    
    def __init__(self, portfolio_manager: TradingPortfolioManager):
        self.portfolio_manager = portfolio_manager
        self.logger = logging.getLogger(__name__)
    
    async def execute_portfolio_strategy(self, trading_engine, market_data: pd.DataFrame):
        """🎯 Führe Portfolio-basierte Trading Strategie aus"""
        
        try:
            # Update Portfolio Manager mit neuen Daten
            await self.portfolio_manager.update_market_data(market_data)
            
            # Hole aktuelle Positionen vom Trading Engine
            current_positions = await trading_engine.get_current_positions()
            
            # Prüfe Rebalancing
            if await self.portfolio_manager.check_rebalancing_needed(current_positions):
                
                # Generiere Rebalancing Orders
                available_balance = await trading_engine.get_available_balance()
                orders = await self.portfolio_manager.generate_rebalancing_orders(
                    current_positions, available_balance
                )
                
                # Führe Orders aus
                for order in orders:
                    await trading_engine.place_order(
                        asset=order['asset'],
                        side=order['side'],
                        amount=order['amount'],
                        order_type='market',
                        metadata={'source': 'portfolio_rebalancing'}
                    )
                
                self.logger.info(f"🔄 Portfolio Rebalancing durchgeführt: {len(orders)} Orders")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio Strategy Execution fehlgeschlagen: {e}")
            return False 