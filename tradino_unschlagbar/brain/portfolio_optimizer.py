#!/usr/bin/env python3
"""
⚖️ PORTFOLIO OPTIMIZATION ENGINE - WELTKLASSE
AI-basierte Portfolio-Optimierung mit moderner Portfoliotheorie
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import minimize, differential_evolution
from sklearn.covariance import LedoitWolf
import cvxpy as cp
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class OptimizationResult:
    """📊 Portfolio Optimization Result"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    optimization_method: str
    success: bool
    message: str

class WorldClassPortfolioOptimizer:
    """🌍 Weltklasse Portfolio Optimization Engine"""
    
    def __init__(self):
        self.assets = []
        self.returns = None
        self.covariance_matrix = None
        self.expected_returns = None
        self.optimization_history = []
        
    def add_assets(self, returns_data: pd.DataFrame):
        """📈 Füge Assets für Optimierung hinzu"""
        
        self.returns = returns_data
        self.assets = list(returns_data.columns)
        
        # Berechne erwartete Returns (verschiedene Methoden)
        self.expected_returns = self._calculate_expected_returns()
        
        # Robuste Kovarianzmatrix-Schätzung
        self.covariance_matrix = self._estimate_covariance_matrix()
        
        print(f"✅ {len(self.assets)} Assets hinzugefügt für Optimierung")
        
    def _calculate_expected_returns(self) -> pd.Series:
        """📊 Berechne erwartete Returns mit verschiedenen Methoden"""
        
        methods = {
            'historical_mean': self.returns.mean() * 252,  # Annualisiert
            'exponential_weighted': self.returns.ewm(span=60).mean().iloc[-1] * 252,
            'robust_mean': self.returns.rolling(252).mean().median() * 252
        }
        
        # Kombiniere Methoden (Ensemble)
        expected_returns = pd.Series(index=self.assets, dtype=float)
        
        for asset in self.assets:
            asset_returns = []
            for method, values in methods.items():
                if asset in values:
                    asset_returns.append(values[asset])
            
            # Gewichteter Durchschnitt
            expected_returns[asset] = np.mean(asset_returns) if asset_returns else 0.0
        
        return expected_returns
    
    def _estimate_covariance_matrix(self) -> pd.DataFrame:
        """📊 Robuste Kovarianzmatrix-Schätzung"""
        
        # Ledoit-Wolf Shrinkage Estimator
        lw = LedoitWolf()
        cov_lw, _ = lw.fit(self.returns).covariance_, lw.shrinkage_
        
        # Standard Covariance Matrix (fallback)
        cov_standard = self.returns.cov() * 252  # Annualisiert
        
        # Exponentially Weighted Covariance
        try:
            cov_ewm = self.returns.ewm(span=60).cov().iloc[-len(self.assets):, :] * 252
            # Falls MultiIndex, nimm nur die letzten Werte für jedes Asset
            if isinstance(cov_ewm.index, pd.MultiIndex):
                cov_ewm = cov_ewm.groupby(level=1).last()
        except:
            # Fallback falls EWM Kovarianz fehlschlägt
            cov_ewm = cov_standard
        
        # Kombiniere beide Methoden
        alpha = 0.7  # Gewichtung für Ledoit-Wolf
        cov_lw_df = pd.DataFrame(cov_lw * 252, index=self.assets, columns=self.assets)
        
        # Stelle sicher, dass beide DataFrames dieselbe Struktur haben
        if cov_ewm.shape == cov_lw_df.shape and all(cov_ewm.index == cov_lw_df.index):
            combined_cov = alpha * cov_lw_df + (1 - alpha) * cov_ewm
        else:
            # Fallback zu Ledoit-Wolf falls Strukturen nicht übereinstimmen
            combined_cov = cov_lw_df
        
        return combined_cov
    
    def optimize_markowitz(self, target_return: Optional[float] = None, 
                          risk_aversion: float = 1.0) -> OptimizationResult:
        """🎯 Klassische Markowitz-Optimierung"""
        
        n_assets = len(self.assets)
        
        # Optimization Variables
        weights = cp.Variable(n_assets)
        
        # Portfolio Return und Risk
        portfolio_return = self.expected_returns.values @ weights
        portfolio_risk = cp.quad_form(weights, self.covariance_matrix.values)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= 0  # Long-only (kann angepasst werden)
        ]
        
        # Target Return Constraint
        if target_return is not None:
            constraints.append(portfolio_return >= target_return)
        
        # Objective Function
        if target_return is not None:
            # Minimize Risk für gegebenen Return
            objective = cp.Minimize(portfolio_risk)
        else:
            # Maximize Utility (Return - Risk Aversion * Risk)
            objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                weights_opt = weights.value
                
                # Berechne Portfolio-Metriken
                result = self._calculate_portfolio_metrics(weights_opt, "Markowitz")
                result.success = True
                result.message = f"Optimization successful - Status: {problem.status}"
                
            else:
                result = OptimizationResult(
                    weights=np.array([1/n_assets] * n_assets),
                    expected_return=0, volatility=0, sharpe_ratio=0,
                    max_drawdown=0, var_95=0, cvar_95=0,
                    optimization_method="Markowitz", success=False,
                    message=f"Optimization failed - Status: {problem.status}"
                )
                
        except Exception as e:
            result = OptimizationResult(
                weights=np.array([1/n_assets] * n_assets),
                expected_return=0, volatility=0, sharpe_ratio=0,
                max_drawdown=0, var_95=0, cvar_95=0,
                optimization_method="Markowitz", success=False,
                message=f"Optimization error: {str(e)}"
            )
        
        self.optimization_history.append(result)
        return result
    
    def optimize_black_litterman(self, views: Dict[str, float], 
                                confidence: Dict[str, float]) -> OptimizationResult:
        """🔮 Black-Litterman Optimierung mit Investor Views"""
        
        # Market Capitalization Weights (simuliert)
        market_caps = np.random.uniform(0.5, 2.0, len(self.assets))
        w_market = market_caps / market_caps.sum()
        
        # Implied Equilibrium Returns
        risk_aversion = 3.0  # Typischer Wert
        pi = risk_aversion * self.covariance_matrix.values @ w_market
        
        # Views Matrix P und Q
        P = np.zeros((len(views), len(self.assets)))
        Q = np.zeros(len(views))
        
        for i, (asset, view) in enumerate(views.items()):
            if asset in self.assets:
                asset_idx = self.assets.index(asset)
                P[i, asset_idx] = 1
                Q[i] = view
        
        # Confidence Matrix Omega
        omega = np.diag([1/confidence.get(asset, 1.0) for asset in views.keys()])
        
        # Black-Litterman Formula
        tau = 0.025  # Skalierungsfaktor
        
        M1 = np.linalg.inv(tau * self.covariance_matrix.values)
        M2 = P.T @ np.linalg.inv(omega) @ P
        M3 = np.linalg.inv(tau * self.covariance_matrix.values) @ pi
        M4 = P.T @ np.linalg.inv(omega) @ Q
        
        # Neue erwartete Returns
        mu_bl = np.linalg.inv(M1 + M2) @ (M3 + M4)
        
        # Neue Kovarianzmatrix
        cov_bl = np.linalg.inv(M1 + M2)
        
        # Optimierung mit Black-Litterman Inputs
        n_assets = len(self.assets)
        weights = cp.Variable(n_assets)
        
        portfolio_return = mu_bl @ weights
        portfolio_risk = cp.quad_form(weights, cov_bl)
        
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]
        
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve()
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                weights_opt = weights.value
                result = self._calculate_portfolio_metrics(weights_opt, "Black-Litterman")
                result.success = True
                result.message = "Black-Litterman optimization successful"
            else:
                result = OptimizationResult(
                    weights=w_market, expected_return=0, volatility=0, sharpe_ratio=0,
                    max_drawdown=0, var_95=0, cvar_95=0,
                    optimization_method="Black-Litterman", success=False,
                    message=f"Optimization failed - Status: {problem.status}"
                )
                
        except Exception as e:
            result = OptimizationResult(
                weights=w_market, expected_return=0, volatility=0, sharpe_ratio=0,
                max_drawdown=0, var_95=0, cvar_95=0,
                optimization_method="Black-Litterman", success=False,
                message=f"Error: {str(e)}"
            )
        
        self.optimization_history.append(result)
        return result
    
    def optimize_risk_parity(self) -> OptimizationResult:
        """⚖️ Risk Parity Optimierung"""
        
        def risk_parity_objective(weights):
            """Risk Parity Objective Function"""
            portfolio_vol = np.sqrt(weights.T @ self.covariance_matrix.values @ weights)
            marginal_contrib = self.covariance_matrix.values @ weights / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Minimize sum of squared differences from equal risk contribution
            target_contrib = portfolio_vol / len(weights)
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        }
        
        bounds = [(0.01, 0.5) for _ in range(len(self.assets))]  # Min 1%, Max 50%
        
        # Initial guess
        x0 = np.array([1/len(self.assets)] * len(self.assets))
        
        # Optimization
        result_opt = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        if result_opt.success:
            weights_opt = result_opt.x
            result = self._calculate_portfolio_metrics(weights_opt, "Risk Parity")
            result.success = True
            result.message = "Risk Parity optimization successful"
        else:
            result = OptimizationResult(
                weights=x0, expected_return=0, volatility=0, sharpe_ratio=0,
                max_drawdown=0, var_95=0, cvar_95=0,
                optimization_method="Risk Parity", success=False,
                message=f"Optimization failed: {result_opt.message}"
            )
        
        self.optimization_history.append(result)
        return result
    
    def optimize_maximum_diversification(self) -> OptimizationResult:
        """🌈 Maximum Diversification Optimierung"""
        
        def diversification_ratio(weights):
            """Diversification Ratio = Weighted Average Volatility / Portfolio Volatility"""
            weighted_vol = np.sum(weights * np.sqrt(np.diag(self.covariance_matrix.values)))
            portfolio_vol = np.sqrt(weights.T @ self.covariance_matrix.values @ weights)
            return -weighted_vol / portfolio_vol  # Negative für Maximierung
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0.01, 0.3) for _ in range(len(self.assets))]
        
        x0 = np.array([1/len(self.assets)] * len(self.assets))
        
        result_opt = minimize(
            diversification_ratio,
            x0,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        if result_opt.success:
            weights_opt = result_opt.x
            result = self._calculate_portfolio_metrics(weights_opt, "Maximum Diversification")
            result.success = True
            result.message = "Maximum Diversification optimization successful"
        else:
            result = OptimizationResult(
                weights=x0, expected_return=0, volatility=0, sharpe_ratio=0,
                max_drawdown=0, var_95=0, cvar_95=0,
                optimization_method="Maximum Diversification", success=False,
                message=f"Optimization failed: {result_opt.message}"
            )
        
        self.optimization_history.append(result)
        return result
    
    def optimize_cvar(self, alpha: float = 0.05) -> OptimizationResult:
        """📉 Conditional Value at Risk (CVaR) Optimierung"""
        
        # Convert returns to scenarios
        scenarios = self.returns.values
        n_scenarios, n_assets = scenarios.shape
        
        # CVaR Optimization using CVXPY
        weights = cp.Variable(n_assets)
        z = cp.Variable(n_scenarios)
        gamma = cp.Variable()
        
        # Portfolio returns for each scenario
        portfolio_returns = scenarios @ weights
        
        # CVaR constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            z >= 0,
            z >= -portfolio_returns - gamma
        ]
        
        # CVaR objective
        cvar = gamma + cp.sum(z) / (alpha * n_scenarios)
        
        # Expected return constraint (minimum)
        min_return = self.expected_returns.mean() * 0.8  # 80% of average expected return
        constraints.append(self.expected_returns.values @ weights >= min_return)
        
        objective = cp.Minimize(cvar)
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve()
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                weights_opt = weights.value
                result = self._calculate_portfolio_metrics(weights_opt, "CVaR")
                result.success = True
                result.message = f"CVaR optimization successful (alpha={alpha})"
            else:
                result = OptimizationResult(
                    weights=np.array([1/n_assets] * n_assets),
                    expected_return=0, volatility=0, sharpe_ratio=0,
                    max_drawdown=0, var_95=0, cvar_95=0,
                    optimization_method="CVaR", success=False,
                    message=f"CVaR optimization failed - Status: {problem.status}"
                )
                
        except Exception as e:
            result = OptimizationResult(
                weights=np.array([1/n_assets] * n_assets),
                expected_return=0, volatility=0, sharpe_ratio=0,
                max_drawdown=0, var_95=0, cvar_95=0,
                optimization_method="CVaR", success=False,
                message=f"CVaR optimization error: {str(e)}"
            )
        
        self.optimization_history.append(result)
        return result
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray, method: str) -> OptimizationResult:
        """📊 Berechne Portfolio Performance Metriken"""
        
        # Expected Return
        expected_return = np.sum(weights * self.expected_returns.values)
        
        # Volatility
        volatility = np.sqrt(weights.T @ self.covariance_matrix.values @ weights)
        
        # Sharpe Ratio (assume risk-free rate = 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Simulate portfolio returns for VaR/CVaR calculation
        portfolio_returns = self.returns.values @ weights
        
        # VaR und CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        
        # Maximum Drawdown (simplified)
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            optimization_method=method,
            success=True,
            message="Metrics calculated successfully"
        )
    
    def compare_optimization_methods(self) -> pd.DataFrame:
        """📊 Vergleiche verschiedene Optimierungsmethoden"""
        
        print("🔄 Führe Optimierung mit verschiedenen Methoden durch...")
        
        methods = [
            ("Markowitz", lambda: self.optimize_markowitz()),
            ("Risk Parity", lambda: self.optimize_risk_parity()),
            ("Max Diversification", lambda: self.optimize_maximum_diversification()),
            ("CVaR", lambda: self.optimize_cvar())
        ]
        
        results = []
        
        for method_name, method_func in methods:
            try:
                result = method_func()
                if result.success:
                    results.append({
                        'Method': method_name,
                        'Expected Return': f"{result.expected_return:.2%}",
                        'Volatility': f"{result.volatility:.2%}",
                        'Sharpe Ratio': f"{result.sharpe_ratio:.3f}",
                        'Max Drawdown': f"{result.max_drawdown:.2%}",
                        'VaR (95%)': f"{result.var_95:.2%}",
                        'CVaR (95%)': f"{result.cvar_95:.2%}"
                    })
                else:
                    results.append({
                        'Method': method_name,
                        'Expected Return': 'Failed',
                        'Volatility': 'Failed',
                        'Sharpe Ratio': 'Failed',
                        'Max Drawdown': 'Failed',
                        'VaR (95%)': 'Failed',
                        'CVaR (95%)': 'Failed'
                    })
            except Exception as e:
                print(f"⚠️ {method_name} failed: {e}")
        
        return pd.DataFrame(results)
    
    def get_portfolio_composition(self, result: OptimizationResult, min_weight: float = 0.01) -> pd.DataFrame:
        """📈 Detaillierte Portfolio-Zusammensetzung"""
        
        composition = pd.DataFrame({
            'Asset': self.assets,
            'Weight': result.weights,
            'Expected Return': self.expected_returns.values,
            'Individual Volatility': np.sqrt(np.diag(self.covariance_matrix.values))
        })
        
        # Filter small weights
        composition = composition[composition['Weight'] >= min_weight]
        composition = composition.sort_values('Weight', ascending=False)
        
        # Format
        composition['Weight'] = composition['Weight'].apply(lambda x: f"{x:.2%}")
        composition['Expected Return'] = composition['Expected Return'].apply(lambda x: f"{x:.2%}")
        composition['Individual Volatility'] = composition['Individual Volatility'].apply(lambda x: f"{x:.2%}")
        
        return composition

# Verwendungsbeispiel
if __name__ == "__main__":
    # Simuliere Asset Returns
    np.random.seed(42)
    assets = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE', 'COMP']
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Generiere korrelierte Returns
    n_assets = len(assets)
    correlation_matrix = np.random.uniform(0.3, 0.8, (n_assets, n_assets))
    np.fill_diagonal(correlation_matrix, 1.0)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    
    # Asset-spezifische Parameter
    annual_returns = np.random.uniform(0.05, 0.25, n_assets)  # 5-25% annual returns
    annual_vols = np.random.uniform(0.20, 0.60, n_assets)    # 20-60% annual volatility
    
    # Generiere Returns
    daily_returns = np.random.multivariate_normal(
        annual_returns / 252,
        np.outer(annual_vols, annual_vols) * correlation_matrix / 252,
        len(dates)
    )
    
    returns_df = pd.DataFrame(daily_returns, index=dates, columns=assets)
    
    # Portfolio Optimizer
    optimizer = WorldClassPortfolioOptimizer()
    optimizer.add_assets(returns_df)
    
    # Verschiedene Optimierungen durchführen
    print("🚀 PORTFOLIO OPTIMIZATION ANALYSIS")
    print("=" * 50)
    
    # Vergleiche Methoden
    comparison = optimizer.compare_optimization_methods()
    print("\n📊 Optimierungsmethoden-Vergleich:")
    print(comparison.to_string(index=False))
    
    # Beste Lösung
    best_result = max(optimizer.optimization_history, key=lambda x: x.sharpe_ratio if x.success else -999)
    
    print(f"\n🏆 BESTE LÖSUNG: {best_result.optimization_method}")
    print(f"Expected Return: {best_result.expected_return:.2%}")
    print(f"Volatility: {best_result.volatility:.2%}")
    print(f"Sharpe Ratio: {best_result.sharpe_ratio:.3f}")
    
    # Portfolio Zusammensetzung
    print(f"\n📈 Portfolio Zusammensetzung:")
    composition = optimizer.get_portfolio_composition(best_result)
    print(composition.to_string(index=False)) 