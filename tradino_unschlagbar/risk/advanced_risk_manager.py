#!/usr/bin/env python3
"""
🛡️ ADVANCED RISK MANAGEMENT SYSTEM

Institutionelles Risikomanagement-System mit VaR, CVaR, Stress Testing,
und Real-time Risk Monitoring für quantitative Trading Strategien.

Features:
- Value at Risk (VaR) Calculation
- Expected Shortfall (CVaR)
- Kelly Criterion Position Sizing
- Real-time Correlation Monitoring
- Stress Testing and Scenario Analysis
- Dynamic Risk Limits
- Real-time Alert System

Author: TRADINO Development Team
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from .risk_models import (
    Position, TradingSignal, MarketConditions, Scenario, RiskMetrics,
    RiskAlert, RiskLimits, RiskLevel, SignalType, AssetClass,
    create_default_risk_limits, validate_position
)

class VaRCalculator:
    """📊 Value at Risk Calculator with multiple methodologies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def historical_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """📈 Historical VaR calculation using empirical distribution"""
        if len(returns) == 0:
            return 0.0
            
        # Sort returns and find percentile
        sorted_returns = np.sort(returns)
        index = int((1 - confidence) * len(sorted_returns))
        
        if index >= len(sorted_returns):
            index = len(sorted_returns) - 1
            
        return abs(sorted_returns[index])
    
    def parametric_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """📐 Parametric VaR assuming normal distribution"""
        if len(returns) == 0:
            return 0.0
            
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence)
        
        # VaR calculation: μ + z*σ
        var = abs(mean_return + z_score * std_return)
        return var
    
    def monte_carlo_var(self, mean: float, std: float, confidence: float = 0.95, 
                       simulations: int = 10000) -> float:
        """🎲 Monte Carlo VaR simulation"""
        # Generate random returns
        random_returns = np.random.normal(mean, std, simulations)
        
        # Calculate VaR using historical method on simulated data
        return self.historical_var(random_returns, confidence)
    
    def expected_shortfall(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """💰 Expected Shortfall (Conditional VaR) - average loss beyond VaR"""
        if len(returns) == 0:
            return 0.0
            
        var = self.historical_var(returns, confidence)
        
        # Calculate average of losses beyond VaR
        tail_losses = returns[returns <= -var]
        
        if len(tail_losses) == 0:
            return var
            
        return abs(np.mean(tail_losses))

class CorrelationMonitor:
    """🔗 Portfolio Correlation Monitor for diversification analysis"""
    
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period
        self.logger = logging.getLogger(__name__)
        
    def calculate_correlation_matrix(self, price_data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """📊 Calculate rolling correlation matrix"""
        if not price_data:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(price_data)
        
        # Calculate returns
        returns = df.pct_change().dropna()
        
        # Use only recent data
        if len(returns) > self.lookback_period:
            returns = returns.tail(self.lookback_period)
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        return correlation_matrix
    
    def detect_correlation_breakdown(self, correlation_matrix: pd.DataFrame,
                                   threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """⚠️ Detect dangerous high correlation pairs"""
        high_correlations = []
        
        if correlation_matrix.empty:
            return high_correlations
            
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                asset1 = correlation_matrix.columns[i]
                asset2 = correlation_matrix.columns[j]
                correlation = correlation_matrix.iloc[i, j]
                
                if abs(correlation) >= threshold:
                    high_correlations.append((asset1, asset2, correlation))
                    
        return high_correlations
    
    def diversification_ratio(self, weights: np.ndarray, 
                            volatilities: np.ndarray,
                            correlation_matrix: pd.DataFrame) -> float:
        """🎯 Calculate diversification ratio"""
        if correlation_matrix.empty or len(weights) == 0:
            return 1.0
            
        # Weighted average volatility
        weighted_avg_vol = np.sum(weights * volatilities)
        
        # Portfolio volatility
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix.values
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        if portfolio_vol == 0:
            return 1.0
            
        return weighted_avg_vol / portfolio_vol

class AdvancedRiskManager:
    """
    🛡️ Advanced Institutional Risk Management System
    
    Provides comprehensive risk management including:
    - Portfolio-level VaR and Expected Shortfall
    - Dynamic position sizing using Kelly Criterion
    - Real-time correlation monitoring
    - Stress testing and scenario analysis
    - Risk limit monitoring and alerting
    """
    
    def __init__(self, portfolio_size: float, max_drawdown: float = 0.10):
        """
        Initialize Advanced Risk Manager
        
        Args:
            portfolio_size: Total portfolio size in base currency
            max_drawdown: Maximum allowed drawdown (default 10%)
        """
        self.portfolio_size = portfolio_size
        self.max_drawdown = max_drawdown
        
        # Initialize components
        self.var_calculator = VaRCalculator()
        self.correlation_monitor = CorrelationMonitor()
        self.risk_limits = create_default_risk_limits()
        
        # Historical data storage
        self.price_history: Dict[str, List[float]] = {}
        self.return_history: Dict[str, List[float]] = {}
        self.portfolio_history: List[float] = [portfolio_size]
        
        # Current state
        self.current_positions: List[Position] = []
        self.active_alerts: List[RiskAlert] = []
        
        # Performance tracking
        self.peak_portfolio_value = portfolio_size
        self.current_drawdown = 0.0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🛡️ Advanced Risk Manager initialized")
        self.logger.info(f"💰 Portfolio Size: ${portfolio_size:,.2f}")
        self.logger.info(f"📉 Max Drawdown: {max_drawdown*100:.1f}%")
    
    def add_position(self, position: Position) -> List[str]:
        """💼 Add position with comprehensive validation"""
        violations = validate_position(position, self.risk_limits)
        
        # Additional portfolio-level checks
        total_exposure = sum(pos.notional_value for pos in self.current_positions)
        new_exposure = total_exposure + position.notional_value
        
        # Check concentration risk
        position_weight = position.notional_value / self.portfolio_size
        if position_weight > self.risk_limits.max_position_size:
            violations.append(f"Position size {position_weight*100:.1f}% exceeds limit {self.risk_limits.max_position_size*100:.1f}%")
        
        # Check total leverage
        leverage_ratio = new_exposure / self.portfolio_size
        if leverage_ratio > self.risk_limits.max_leverage:
            violations.append(f"Total leverage {leverage_ratio:.2f}x would exceed limit {self.risk_limits.max_leverage:.2f}x")
        
        if not violations:
            self.current_positions.append(position)
            self.logger.info(f"✅ Position added: {position.symbol} ${position.notional_value:,.2f}")
            self.logger.info(f"📊 Total positions: {len(self.current_positions)}")
        else:
            self.logger.warning(f"❌ Position rejected for {position.symbol}: {violations}")
            
        return violations
    
    def remove_position(self, symbol: str) -> bool:
        """🗑️ Remove position by symbol"""
        for i, position in enumerate(self.current_positions):
            if position.symbol == symbol:
                removed_position = self.current_positions.pop(i)
                self.logger.info(f"🗑️ Position removed: {symbol} (P&L: ${removed_position.unrealized_pnl:,.2f})")
                return True
        return False
    
    def update_position_prices(self, price_updates: Dict[str, float]):
        """📈 Update current prices and calculate returns"""
        for position in self.current_positions:
            if position.symbol in price_updates:
                old_price = position.current_price
                position.current_price = price_updates[position.symbol]
                
                # Update price history
                if position.symbol not in self.price_history:
                    self.price_history[position.symbol] = []
                self.price_history[position.symbol].append(position.current_price)
                
                # Calculate and store return
                if old_price > 0:
                    return_pct = (position.current_price - old_price) / old_price
                    if position.symbol not in self.return_history:
                        self.return_history[position.symbol] = []
                    self.return_history[position.symbol].append(return_pct)
                    
                    # Limit history to recent data
                    if len(self.return_history[position.symbol]) > 504:  # 2 years
                        self.return_history[position.symbol] = self.return_history[position.symbol][-504:]
        
        # Update portfolio value history
        self.update_portfolio_history()
    
    def calculate_var(self, positions: List[Position], confidence: float = 0.95) -> float:
        """📊 Calculate portfolio Value at Risk"""
        if not positions:
            return 0.0
            
        # Aggregate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns()
        
        if len(portfolio_returns) < 30:  # Need sufficient data for reliable VaR
            # Use parametric VaR with conservative assumptions
            return self.portfolio_size * 0.03  # 3% default VaR
            
        # Calculate VaR using historical method
        var = self.var_calculator.historical_var(portfolio_returns, confidence)
        
        # Scale to current portfolio size
        current_portfolio_value = self.get_portfolio_value()
        return var * current_portfolio_value
    
    def calculate_expected_shortfall(self, positions: List[Position], 
                                   confidence: float = 0.95) -> float:
        """💸 Calculate Expected Shortfall (Conditional VaR)"""
        if not positions:
            return 0.0
            
        portfolio_returns = self._calculate_portfolio_returns()
        
        if len(portfolio_returns) < 30:
            # Use VaR + buffer for ES estimation
            var = self.calculate_var(positions, confidence)
            return var * 1.4  # ES typically 40% higher than VaR for normal distribution
            
        es = self.var_calculator.expected_shortfall(portfolio_returns, confidence)
        current_portfolio_value = self.get_portfolio_value()
        return es * current_portfolio_value
    
    def optimal_position_size(self, signal: TradingSignal, 
                            market_conditions: MarketConditions) -> float:
        """
        🎯 Calculate optimal position size using Kelly Criterion with risk adjustments
        
        Incorporates:
        - Kelly Criterion for optimal sizing
        - Market condition adjustments
        - Risk limit constraints
        - Correlation considerations
        """
        
        # Validate signal inputs
        if signal.expected_return is None or signal.confidence < 0.5:
            return self.portfolio_size * 0.01  # 1% conservative default
            
        # Kelly Criterion parameters
        win_probability = signal.confidence
        expected_return = abs(signal.expected_return)
        
        # Estimate loss parameters
        loss_probability = 1 - win_probability
        
        # Use risk score or default estimate
        if signal.risk_score and signal.risk_score > 0:
            expected_loss = signal.risk_score
        else:
            expected_loss = expected_return * 0.5  # Assume 50% of return as potential loss
        
        # Kelly fraction calculation: f = (bp - q) / b
        # where b = odds, p = win prob, q = loss prob
        if expected_loss == 0:
            kelly_fraction = 0.02  # Conservative default
        else:
            kelly_fraction = (win_probability * expected_return - loss_probability * expected_loss) / expected_loss
        
        # Apply Kelly fraction constraints (never risk more than 25%)
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        # Market condition adjustments
        market_adjustment = self._get_market_adjustment(market_conditions)
        adjusted_fraction = kelly_fraction * market_adjustment
        
        # Volatility adjustment
        if signal.symbol in self.return_history and len(self.return_history[signal.symbol]) > 10:
            volatility = np.std(self.return_history[signal.symbol]) * np.sqrt(252)
            vol_adjustment = max(0.3, min(1.0, 0.2 / volatility))  # Reduce size for high vol assets
            adjusted_fraction *= vol_adjustment
        
        # Portfolio-level constraints
        current_portfolio_value = self.get_portfolio_value()
        max_position = current_portfolio_value * self.risk_limits.max_position_size
        optimal_size = current_portfolio_value * adjusted_fraction
        
        return min(optimal_size, max_position)
    
    def stress_test_portfolio(self, stress_scenarios: List[Scenario]) -> Dict[str, float]:
        """🧪 Comprehensive stress testing with scenario analysis"""
        stress_results = {}
        
        for scenario in stress_scenarios:
            total_loss = 0.0
            scenario_positions = []
            
            for position in self.current_positions:
                # Determine applicable shock
                shock = 0.0
                if position.symbol in scenario.price_shocks:
                    shock = scenario.price_shocks[position.symbol]
                elif position.asset_class.value in scenario.price_shocks:
                    shock = scenario.price_shocks[position.asset_class.value]
                elif "market" in scenario.price_shocks:
                    shock = scenario.price_shocks["market"]
                
                # Apply volatility multiplier for additional uncertainty
                if scenario.volatility_multiplier > 1:
                    vol_shock = np.random.normal(0, 0.1) * (scenario.volatility_multiplier - 1)
                    shock += vol_shock
                
                # Calculate shocked price
                shocked_price = position.current_price * (1 + shock)
                
                # Calculate position P&L under stress
                position_pnl = (shocked_price - position.entry_price) * position.size
                
                # Apply leverage effect
                leveraged_pnl = position_pnl * position.leverage
                
                total_loss += leveraged_pnl
                scenario_positions.append({
                    'symbol': position.symbol,
                    'shock': shock,
                    'pnl': leveraged_pnl
                })
            
            # Apply correlation effects (increased correlation during stress)
            if scenario.correlation_shift > 0:
                # Increase losses due to correlation breakdown
                correlation_penalty = abs(total_loss) * scenario.correlation_shift * 0.1
                total_loss -= correlation_penalty
            
            # Apply liquidity impact
            if scenario.liquidity_reduction > 0:
                liquidity_cost = abs(total_loss) * scenario.liquidity_reduction * 0.1
                total_loss -= liquidity_cost
            
            # Store results
            stress_results[scenario.name] = {
                'total_loss': total_loss,
                'loss_percentage': (total_loss / self.get_portfolio_value()) * 100,
                'position_details': scenario_positions,
                'scenario_probability': scenario.probability
            }
            
        return stress_results
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """📊 Calculate comprehensive risk metrics suite"""
        timestamp = datetime.now()
        
        # VaR calculations at multiple confidence levels
        var_1d_95 = self.calculate_var(self.current_positions, 0.95)
        var_1d_99 = self.calculate_var(self.current_positions, 0.99)
        var_10d_95 = var_1d_95 * np.sqrt(10)  # Scale to 10-day using square root rule
        
        # Expected Shortfall calculations
        es_1d_95 = self.calculate_expected_shortfall(self.current_positions, 0.95)
        es_1d_99 = self.calculate_expected_shortfall(self.current_positions, 0.99)
        
        # Portfolio composition metrics
        current_portfolio_value = self.get_portfolio_value()
        total_exposure = sum(pos.notional_value for pos in self.current_positions)
        leverage_ratio = total_exposure / current_portfolio_value if current_portfolio_value > 0 else 0
        
        # Performance metrics
        portfolio_returns = self._calculate_portfolio_returns()
        
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        max_drawdown = self._calculate_max_drawdown()
        calmar_ratio = sharpe_ratio / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Correlation and diversification analysis
        price_data = {symbol: np.array(self.price_history.get(symbol, [0])) 
                     for symbol in [pos.symbol for pos in self.current_positions]}
        correlation_matrix = self.correlation_monitor.calculate_correlation_matrix(price_data)
        
        avg_correlation = 0.0
        if not correlation_matrix.empty and correlation_matrix.shape[0] > 1:
            # Extract upper triangle excluding diagonal
            upper_triangle = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
            avg_correlation = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
        
        # Diversification ratio calculation
        weights = np.array([pos.notional_value / total_exposure for pos in self.current_positions]) if total_exposure > 0 else np.array([])
        volatilities = np.array([np.std(self.return_history.get(pos.symbol, [0])) for pos in self.current_positions])
        
        diversification_ratio = 1.0
        if len(weights) > 1 and not correlation_matrix.empty:
            diversification_ratio = self.correlation_monitor.diversification_ratio(weights, volatilities, correlation_matrix)
        
        # Stress test analysis
        default_scenarios = [
            Scenario.create_market_crash(),
            Scenario.create_flash_crash(),
            Scenario.create_regulatory_shock()
        ]
        stress_results = self.stress_test_portfolio(default_scenarios)
        stress_loss_summary = {name: result['total_loss'] for name, result in stress_results.items()}
        
        # Risk utilization metrics
        var_utilization = (var_1d_95 / (self.portfolio_size * self.risk_limits.max_portfolio_var)) if self.risk_limits.max_portfolio_var > 0 else 0
        position_utilization = len(self.current_positions) / 20  # Assume max 20 positions
        
        return RiskMetrics(
            timestamp=timestamp,
            var_1d_95=var_1d_95,
            var_1d_99=var_1d_99,
            var_10d_95=var_10d_95,
            es_1d_95=es_1d_95,
            es_1d_99=es_1d_99,
            portfolio_value=current_portfolio_value,
            total_exposure=total_exposure,
            leverage_ratio=leverage_ratio,
            concentration_risk=self._calculate_concentration_risk(),
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=self.current_drawdown,
            risk_adjusted_return=sharpe_ratio,
            information_ratio=sharpe_ratio,  # Simplified - would need benchmark for true IR
            treynor_ratio=sharpe_ratio,     # Simplified - would need market beta for true TR
            avg_correlation=avg_correlation,
            diversification_ratio=diversification_ratio,
            risk_limit_utilization=var_utilization,
            position_limit_utilization=position_utilization,
            stress_test_loss=stress_loss_summary
        )
    
    def monitor_risk_limits(self) -> List[RiskAlert]:
        """🚨 Monitor risk limits and generate real-time alerts"""
        alerts = []
        current_time = datetime.now()
        
        # Calculate current metrics
        metrics = self.calculate_risk_metrics()
        
        # VaR limit monitoring
        var_ratio = metrics.var_1d_95 / (self.portfolio_size * self.risk_limits.max_portfolio_var) if self.risk_limits.max_portfolio_var > 0 else 0
        if var_ratio > self.risk_limits.var_alert_threshold:
            severity = RiskLevel.CRITICAL if var_ratio > 1.0 else RiskLevel.HIGH
            alert = RiskAlert(
                alert_id=f"VAR_{current_time.strftime('%Y%m%d_%H%M%S')}",
                timestamp=current_time,
                risk_level=severity,
                alert_type="VaR_BREACH",
                message=f"Portfolio VaR at {var_ratio*100:.1f}% of limit (${metrics.var_1d_95:,.2f})",
                affected_positions=[pos.symbol for pos in self.current_positions],
                recommended_action="Reduce position sizes, add hedges, or increase cash allocation",
                threshold_breached={"var_ratio": var_ratio, "var_limit": self.risk_limits.max_portfolio_var}
            )
            alerts.append(alert)
        
        # Drawdown monitoring
        drawdown_ratio = abs(self.current_drawdown) / self.risk_limits.max_drawdown
        if drawdown_ratio > self.risk_limits.drawdown_alert_threshold:
            severity = RiskLevel.CRITICAL if drawdown_ratio > 1.0 else RiskLevel.HIGH
            alert = RiskAlert(
                alert_id=f"DD_{current_time.strftime('%Y%m%d_%H%M%S')}",
                timestamp=current_time,
                risk_level=severity,
                alert_type="DRAWDOWN_BREACH",
                message=f"Current drawdown: {self.current_drawdown*100:.2f}% (limit: {self.risk_limits.max_drawdown*100:.1f}%)",
                affected_positions=[pos.symbol for pos in self.current_positions],
                recommended_action="Halt new positions, review strategy, consider defensive measures",
                threshold_breached={"drawdown": self.current_drawdown, "drawdown_limit": self.risk_limits.max_drawdown}
            )
            alerts.append(alert)
        
        # Leverage monitoring
        if metrics.leverage_ratio > self.risk_limits.max_leverage:
            alert = RiskAlert(
                alert_id=f"LEV_{current_time.strftime('%Y%m%d_%H%M%S')}",
                timestamp=current_time,
                risk_level=RiskLevel.HIGH,
                alert_type="LEVERAGE_BREACH",
                message=f"Portfolio leverage: {metrics.leverage_ratio:.2f}x (limit: {self.risk_limits.max_leverage:.2f}x)",
                affected_positions=[pos.symbol for pos in self.current_positions],
                recommended_action="Reduce leverage by closing positions or reducing sizes",
                threshold_breached={"leverage": metrics.leverage_ratio, "leverage_limit": self.risk_limits.max_leverage}
            )
            alerts.append(alert)
        
        # Concentration risk monitoring
        if metrics.concentration_risk > 0.5:  # High concentration
            alert = RiskAlert(
                alert_id=f"CONC_{current_time.strftime('%Y%m%d_%H%M%S')}",
                timestamp=current_time,
                risk_level=RiskLevel.MODERATE,
                alert_type="CONCENTRATION_RISK",
                message=f"High portfolio concentration detected (HHI: {metrics.concentration_risk:.3f})",
                affected_positions=[pos.symbol for pos in self.current_positions],
                recommended_action="Diversify portfolio across more assets and sectors",
                threshold_breached={"concentration": metrics.concentration_risk, "threshold": 0.5}
            )
            alerts.append(alert)
        
        # Correlation monitoring
        if metrics.avg_correlation > self.risk_limits.max_correlation:
            alert = RiskAlert(
                alert_id=f"CORR_{current_time.strftime('%Y%m%d_%H%M%S')}",
                timestamp=current_time,
                risk_level=RiskLevel.MODERATE,
                alert_type="CORRELATION_BREAKDOWN",
                message=f"High average correlation: {metrics.avg_correlation:.3f} (limit: {self.risk_limits.max_correlation:.3f})",
                affected_positions=[pos.symbol for pos in self.current_positions],
                recommended_action="Review portfolio diversification and consider uncorrelated assets",
                threshold_breached={"correlation": metrics.avg_correlation, "correlation_limit": self.risk_limits.max_correlation}
            )
            alerts.append(alert)
        
        # Update active alerts (remove old ones, add new ones)
        self.active_alerts = [alert for alert in self.active_alerts 
                            if (current_time - alert.timestamp).seconds < 3600]  # Keep alerts for 1 hour
        self.active_alerts.extend(alerts)
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(f"🚨 {alert.alert_type}: {alert.message}")
        
        return alerts
    
    def get_portfolio_value(self) -> float:
        """💰 Calculate current total portfolio value including unrealized P&L"""
        portfolio_value = self.portfolio_size
        
        for position in self.current_positions:
            portfolio_value += position.unrealized_pnl
            
        return max(0, portfolio_value)  # Ensure non-negative
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """📊 Get comprehensive portfolio summary"""
        metrics = self.calculate_risk_metrics()
        
        return {
            "portfolio_value": self.get_portfolio_value(),
            "initial_capital": self.portfolio_size,
            "total_return_pct": ((self.get_portfolio_value() - self.portfolio_size) / self.portfolio_size) * 100,
            "total_positions": len(self.current_positions),
            "total_exposure": metrics.total_exposure,
            "gross_leverage": metrics.leverage_ratio,
            "concentration_hhi": metrics.concentration_risk
        }
    
    # Private helper methods
    def _calculate_portfolio_returns(self) -> np.ndarray:
        """📈 Calculate portfolio returns from value history"""
        if len(self.portfolio_history) < 2:
            return np.array([])
            
        returns = []
        for i in range(1, len(self.portfolio_history)):
            if self.portfolio_history[i-1] != 0:
                ret = (self.portfolio_history[i] - self.portfolio_history[i-1]) / self.portfolio_history[i-1]
                returns.append(ret)
                
        return np.array(returns)
    
    def _get_market_adjustment(self, market_conditions: MarketConditions) -> float:
        """🌍 Calculate market condition adjustment factor for position sizing"""
        base_adjustment = 1.0
        
        # Volatility adjustment
        if market_conditions.volatility > 0.4:
            base_adjustment *= 0.3  # Severe reduction in crisis volatility
        elif market_conditions.volatility > 0.3:
            base_adjustment *= 0.5  # High volatility reduction
        elif market_conditions.volatility > 0.2:
            base_adjustment *= 0.7  # Moderate volatility reduction
            
        # Market stress adjustment
        if market_conditions.market_stress > 0.8:
            base_adjustment *= 0.2  # Extreme stress
        elif market_conditions.market_stress > 0.6:
            base_adjustment *= 0.4  # High stress
        elif market_conditions.market_stress > 0.4:
            base_adjustment *= 0.7  # Moderate stress
            
        # Risk appetite adjustment
        if market_conditions.risk_appetite < 0.2:
            base_adjustment *= 0.3  # Risk-off environment
        elif market_conditions.risk_appetite < 0.4:
            base_adjustment *= 0.6  # Low risk appetite
            
        # Market regime adjustment
        regime = market_conditions.market_regime
        if regime == "crisis":
            base_adjustment *= 0.1
        elif regime == "volatile":
            base_adjustment *= 0.4
        elif regime == "bear_market":
            base_adjustment *= 0.5
            
        return max(0.05, base_adjustment)  # Minimum 5% of normal size
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """📊 Calculate annualized Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
            
        # Convert annual risk-free rate to daily
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf
        
        if np.std(returns) == 0:
            return 0.0
            
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """📉 Calculate Sortino ratio (downside deviation)"""
        if len(returns) == 0:
            return 0.0
            
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf
        
        # Only consider negative returns for downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return np.mean(excess_returns) * np.sqrt(252) if len(excess_returns) > 0 else 0.0
            
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self) -> float:
        """📉 Calculate maximum drawdown from portfolio history"""
        if len(self.portfolio_history) < 2:
            return 0.0
            
        portfolio_values = np.array(self.portfolio_history)
        
        # Calculate running maximum (peak)
        peak = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdown
        drawdown = (portfolio_values - peak) / peak
        
        return np.min(drawdown)  # Most negative value
    
    def _calculate_concentration_risk(self) -> float:
        """🎯 Calculate portfolio concentration using Herfindahl-Hirschman Index"""
        if not self.current_positions:
            return 0.0
            
        total_exposure = sum(pos.notional_value for pos in self.current_positions)
        
        if total_exposure == 0:
            return 0.0
            
        # Calculate weights and HHI
        weights = [pos.notional_value / total_exposure for pos in self.current_positions]
        hhi = sum(w**2 for w in weights)
        
        return hhi
    
    def update_portfolio_history(self):
        """📊 Update portfolio value history and drawdown calculation"""
        current_value = self.get_portfolio_value()
        self.portfolio_history.append(current_value)
        
        # Update peak and current drawdown
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (current_value - self.peak_portfolio_value) / self.peak_portfolio_value
        
        # Limit history to manageable size (2 years of daily data)
        if len(self.portfolio_history) > 504:
            self.portfolio_history = self.portfolio_history[-504:]
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """📋 Generate comprehensive institutional risk report"""
        metrics = self.calculate_risk_metrics()
        stress_results = self.stress_test_portfolio([
            Scenario.create_market_crash(),
            Scenario.create_flash_crash(),
            Scenario.create_regulatory_shock()
        ])
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "risk_manager_version": "1.0.0",
            
            # Executive Summary
            "executive_summary": {
                "overall_risk_level": metrics.risk_level.value,
                "risk_score": metrics.overall_risk_score,
                "portfolio_value": self.get_portfolio_value(),
                "var_1d_99": metrics.var_1d_99,
                "max_stress_loss": min(stress_results.values(), key=lambda x: x['total_loss'])['total_loss'] if stress_results else 0,
                "active_breaches": len([alert for alert in self.active_alerts if alert.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]])
            },
            
            # Portfolio Summary
            "portfolio_summary": {
                "total_value": self.get_portfolio_value(),
                "initial_capital": self.portfolio_size,
                "total_return_pct": ((self.get_portfolio_value() - self.portfolio_size) / self.portfolio_size) * 100,
                "total_positions": len(self.current_positions),
                "total_exposure": metrics.total_exposure,
                "gross_leverage": metrics.leverage_ratio,
                "concentration_hhi": metrics.concentration_risk
            },
            
            # Risk Metrics
            "risk_metrics": {
                "value_at_risk": {
                    "var_1d_95_pct": (metrics.var_1d_95 / self.get_portfolio_value()) * 100,
                    "var_1d_99_pct": (metrics.var_1d_99 / self.get_portfolio_value()) * 100,
                    "var_10d_95_pct": (metrics.var_10d_95 / self.get_portfolio_value()) * 100,
                    "var_1d_95_usd": metrics.var_1d_95,
                    "var_1d_99_usd": metrics.var_1d_99
                },
                "expected_shortfall": {
                    "es_1d_95_pct": (metrics.es_1d_95 / self.get_portfolio_value()) * 100,
                    "es_1d_99_pct": (metrics.es_1d_99 / self.get_portfolio_value()) * 100,
                    "es_1d_95_usd": metrics.es_1d_95,
                    "es_1d_99_usd": metrics.es_1d_99
                },
                "drawdown_metrics": {
                    "current_drawdown_pct": self.current_drawdown * 100,
                    "max_drawdown_pct": metrics.max_drawdown * 100,
                    "peak_portfolio_value": self.peak_portfolio_value
                },
                "performance_metrics": {
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "sortino_ratio": metrics.sortino_ratio,
                    "calmar_ratio": metrics.calmar_ratio,
                    "information_ratio": metrics.information_ratio
                }
            },
            
            # Stress Test Results
            "stress_tests": {name: {
                "loss_usd": result['total_loss'],
                "loss_pct": result['loss_percentage'],
                "probability": result['scenario_probability']
            } for name, result in stress_results.items()},
            
            # Risk Limit Utilization
            "risk_limits": {
                "var_utilization_pct": metrics.risk_limit_utilization * 100,
                "position_utilization_pct": metrics.position_limit_utilization * 100,
                "leverage_utilization_pct": (metrics.leverage_ratio / self.risk_limits.max_leverage) * 100,
                "drawdown_utilization_pct": (abs(metrics.current_drawdown) / self.risk_limits.max_drawdown) * 100
            },
            
            # Active Alerts
            "active_alerts": [alert.to_dict() for alert in self.active_alerts],
            
            # Position Details
            "positions": [{
                "symbol": pos.symbol,
                "asset_class": pos.asset_class.value,
                "notional_value": pos.notional_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct * 100,
                "leverage": pos.leverage,
                "weight_pct": (pos.notional_value / metrics.total_exposure) * 100 if metrics.total_exposure > 0 else 0
            } for pos in self.current_positions],
            
            # Recommendations
            "recommendations": self._generate_recommendations(metrics, stress_results)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: RiskMetrics, stress_results: Dict) -> List[str]:
        """💡 Generate risk management recommendations"""
        recommendations = []
        
        # VaR recommendations
        if metrics.risk_limit_utilization > 0.8:
            recommendations.append("Consider reducing position sizes - VaR utilization above 80%")
        
        # Drawdown recommendations
        if abs(metrics.current_drawdown) > self.risk_limits.max_drawdown * 0.7:
            recommendations.append("Approaching maximum drawdown limit - consider defensive measures")
        
        # Leverage recommendations
        if metrics.leverage_ratio > self.risk_limits.max_leverage * 0.8:
            recommendations.append("High leverage detected - reduce position sizes or exposure")
        
        # Concentration recommendations
        if metrics.concentration_risk > 0.4:
            recommendations.append("High concentration risk - diversify across more assets")
        
        # Correlation recommendations
        if metrics.avg_correlation > 0.7:
            recommendations.append("High portfolio correlation - seek uncorrelated assets")
        
        # Stress test recommendations
        worst_stress_loss = min(stress_results.values(), key=lambda x: x['total_loss']) if stress_results else None
        if worst_stress_loss and abs(worst_stress_loss['total_loss']) > self.get_portfolio_value() * 0.15:
            recommendations.append("High stress test losses - consider hedging strategies")
        
        # Performance recommendations
        if metrics.sharpe_ratio < 0.5:
            recommendations.append("Low risk-adjusted returns - review strategy effectiveness")
        
        return recommendations 