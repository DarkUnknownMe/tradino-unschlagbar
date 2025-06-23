#!/usr/bin/env python3
"""
🛡️ RISK MANAGEMENT SYSTEM
Advanced risk management für TRADINO UNSCHLAGBAR
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class RiskManagementSystem:
    """🛡️ Advanced Risk Management System"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self.get_default_config()
        self.is_active = True
        
        # Portfolio state
        self.portfolio_state = {
            'total_balance': 0.0,
            'total_exposure': 0.0,
            'open_positions': {},
            'daily_pnl': 0.0,
            'max_drawdown': 0.0,
            'last_update': datetime.now()
        }
        
        # Risk metrics
        self.risk_metrics = {
            'var_1day': 0.0,
            'sharpe_ratio': 0.0,
            'max_consecutive_losses': 0,
            'current_drawdown': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0
        }
        
        self.trade_history = []
        self.active_alerts = []
        self.alert_callbacks = []
        
        print("🛡️ Risk Management System initialized")
    
    def get_default_config(self) -> Dict[str, Any]:
        """⚙️ Default risk configuration"""
        
        return {
            'max_portfolio_exposure': 0.8,
            'max_single_position': 0.1,
            'daily_loss_limit': 0.05,
            'max_drawdown_limit': 0.15,
            'stop_loss_percent': 0.02,
            'take_profit_percent': 0.04,
            'max_daily_trades': 20,
            'max_consecutive_losses': 5,
            'min_time_between_trades': 300,
            'high_volatility_threshold': 0.05,
            'low_confidence_threshold': 0.6,
            'emergency_stop_loss': 0.1,
            'circuit_breaker_enabled': True,
            'risk_check_interval': 30
        }
    
    def update_portfolio_state(self, balance: float, positions: Dict[str, Any], 
                             recent_trades: List[Dict[str, Any]]):
        """📊 Update portfolio state"""
        
        try:
            self.portfolio_state['total_balance'] = balance
            self.portfolio_state['open_positions'] = positions
            self.portfolio_state['last_update'] = datetime.now()
            
            # Calculate exposure
            total_exposure = sum(pos.get('value', 0) for pos in positions.values())
            self.portfolio_state['total_exposure'] = total_exposure
            
            # Update trade history
            for trade in recent_trades:
                if trade not in self.trade_history:
                    self.trade_history.append(trade)
            
            # Keep last 1000 trades
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
            
            self.calculate_daily_pnl()
            self.calculate_risk_metrics()
            self.check_risk_limits()
            
        except Exception as e:
            print(f"❌ Error updating portfolio: {e}")
    
    def calculate_daily_pnl(self):
        """📈 Calculate daily P&L"""
        
        today = datetime.now().date()
        daily_pnl = 0.0
        
        for trade in self.trade_history:
            trade_date = datetime.fromisoformat(trade['timestamp']).date()
            if trade_date == today:
                result = trade.get('result', {})
                if result.get('success'):
                    pnl = result.get('pnl', result.get('simulated_pnl', 0))
                    daily_pnl += pnl
        
        self.portfolio_state['daily_pnl'] = daily_pnl
    
    def calculate_risk_metrics(self):
        """📊 Calculate risk metrics"""
        
        if len(self.trade_history) < 10:
            return
        
        # Extract returns
        returns = []
        for trade in self.trade_history[-100:]:
            result = trade.get('result', {})
            if result.get('success'):
                pnl = result.get('pnl', result.get('simulated_pnl', 0))
                if pnl != 0:
                    returns.append(pnl)
        
        if len(returns) < 5:
            return
        
        returns = np.array(returns)
        
        # Calculate metrics
        self.risk_metrics['volatility'] = np.std(returns)
        
        if len(returns) >= 20:
            self.risk_metrics['var_1day'] = np.percentile(returns, 5)
        
        # Sharpe ratio
        mean_return = np.mean(returns)
        if self.risk_metrics['volatility'] > 0:
            self.risk_metrics['sharpe_ratio'] = mean_return / self.risk_metrics['volatility']
        
        # Win rate
        winning_trades = sum(1 for r in returns if r > 0)
        self.risk_metrics['win_rate'] = winning_trades / len(returns)
        
        # Current drawdown
        balance = self.portfolio_state['total_balance']
        if hasattr(self, 'peak_balance'):
            self.risk_metrics['current_drawdown'] = (self.peak_balance - balance) / self.peak_balance
        else:
            self.peak_balance = balance
        
        if balance > getattr(self, 'peak_balance', 0):
            self.peak_balance = balance
        
        # Consecutive losses
        consecutive = 0
        max_consecutive = 0
        
        for trade in reversed(self.trade_history[-50:]):
            result = trade.get('result', {})
            if result.get('success'):
                pnl = result.get('pnl', result.get('simulated_pnl', 0))
                if pnl < 0:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            else:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
        
        self.risk_metrics['max_consecutive_losses'] = max_consecutive
    
    def check_risk_limits(self) -> List[str]:
        """🚨 Check risk limits"""
        
        violations = []
        
        try:
            # Portfolio exposure
            exposure_ratio = (self.portfolio_state['total_exposure'] / 
                            max(self.portfolio_state['total_balance'], 1))
            
            if exposure_ratio > self.config['max_portfolio_exposure']:
                violations.append(f"Portfolio exposure: {exposure_ratio:.1%}")
            
            # Daily loss
            daily_loss_ratio = abs(self.portfolio_state['daily_pnl']) / max(self.portfolio_state['total_balance'], 1)
            if self.portfolio_state['daily_pnl'] < 0 and daily_loss_ratio > self.config['daily_loss_limit']:
                violations.append(f"Daily loss limit: {daily_loss_ratio:.1%}")
            
            # Max drawdown
            if self.risk_metrics['current_drawdown'] > self.config['max_drawdown_limit']:
                violations.append(f"Max drawdown: {self.risk_metrics['current_drawdown']:.1%}")
            
            # Consecutive losses
            if self.risk_metrics['max_consecutive_losses'] > self.config['max_consecutive_losses']:
                violations.append(f"Consecutive losses: {self.risk_metrics['max_consecutive_losses']}")
            
            # Emergency stop
            if daily_loss_ratio > self.config['emergency_stop_loss']:
                violations.append(f"EMERGENCY LOSS: {daily_loss_ratio:.1%}")
            
            if violations:
                self.handle_risk_violations(violations)
            
            return violations
            
        except Exception as e:
            print(f"❌ Error checking limits: {e}")
            return []
    
    def handle_risk_violations(self, violations: List[str]):
        """🚨 Handle violations"""
        
        for violation in violations:
            print(f"🚨 RISK ALERT: {violation}")
            
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': 'risk_violation',
                'message': violation,
                'severity': 'critical' if 'EMERGENCY' in violation else 'high'
            }
            
            self.active_alerts.append(alert)
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except:
                    pass
        
        # Emergency actions
        emergency_violations = [v for v in violations if 'EMERGENCY' in v]
        if emergency_violations and self.config['circuit_breaker_enabled']:
            self.trigger_emergency_stop()
    
    def trigger_emergency_stop(self):
        """🛑 Emergency stop"""
        
        print("🛑 EMERGENCY STOP TRIGGERED!")
        
        emergency_action = {
            'timestamp': datetime.now().isoformat(),
            'action': 'emergency_stop',
            'reason': 'Critical risk violation',
            'automatic': True
        }
        
        return emergency_action
    
    def validate_trade_signal(self, signal: Dict[str, Any], symbol: str, 
                            current_price: float) -> Dict[str, Any]:
        """🤔 Validate trade signal"""
        
        validation = {
            'approved': True,
            'risk_score': 0.0,
            'warnings': [],
            'restrictions': {}
        }
        
        try:
            # Position size check
            max_position_value = self.portfolio_state['total_balance'] * self.config['max_single_position']
            requested_size = signal.get('position_size', 0) * current_price
            
            if requested_size > max_position_value:
                new_size = max_position_value / current_price
                validation['restrictions']['max_position_size'] = new_size
                validation['warnings'].append("Position size reduced")
            
            # Confidence check
            confidence = signal.get('confidence', 0)
            if confidence < self.config['low_confidence_threshold']:
                validation['approved'] = False
                validation['warnings'].append(f"Low confidence: {confidence:.2f}")
            
            # Daily trades
            today_trades = self.get_todays_trade_count()
            if today_trades >= self.config['max_daily_trades']:
                validation['approved'] = False
                validation['warnings'].append("Daily trade limit reached")
            
            # Consecutive losses
            if self.risk_metrics['max_consecutive_losses'] >= self.config['max_consecutive_losses']:
                validation['approved'] = False
                validation['warnings'].append("Too many consecutive losses")
            
            # Exposure check
            current_exposure = self.portfolio_state['total_exposure']
            max_exposure = self.portfolio_state['total_balance'] * self.config['max_portfolio_exposure']
            
            if current_exposure + requested_size > max_exposure:
                validation['approved'] = False
                validation['warnings'].append("Would exceed exposure limit")
            
            # Risk score
            risk_factors = [
                min(confidence * 100, 100),
                max(0, 100 - (self.risk_metrics['volatility'] * 1000)),
                max(0, 100 - (self.risk_metrics['current_drawdown'] * 100)),
                min(self.risk_metrics['win_rate'] * 100, 100)
            ]
            
            validation['risk_score'] = np.mean(risk_factors) if risk_factors else 50
            
            if validation['risk_score'] < 30:
                validation['approved'] = False
                validation['warnings'].append(f"Risk score too low: {validation['risk_score']:.1f}")
            
        except Exception as e:
            validation['approved'] = False
            validation['warnings'].append(f"Validation error: {e}")
        
        return validation
    
    def calculate_optimal_position_size(self, signal: Dict[str, Any], symbol: str,
                                      current_price: float) -> float:
        """📊 Calculate optimal position size"""
        
        try:
            confidence = signal.get('confidence', 0.5)
            win_prob = (confidence + self.risk_metrics['win_rate']) / 2
            loss_prob = 1 - win_prob
            
            reward_ratio = self.config['take_profit_percent'] / self.config['stop_loss_percent']
            kelly_fraction = (reward_ratio * win_prob - loss_prob) / reward_ratio
            kelly_fraction = max(0, min(kelly_fraction, self.config['max_single_position']))
            
            # Risk adjustment
            risk_adjustment = 1.0 - (self.risk_metrics['current_drawdown'] * 2)
            risk_adjustment = max(0.1, min(1.0, risk_adjustment))
            
            optimal_fraction = kelly_fraction * risk_adjustment
            
            available_capital = self.portfolio_state['total_balance']
            optimal_position_value = available_capital * optimal_fraction
            optimal_position_size = optimal_position_value / current_price
            
            return optimal_position_size
            
        except Exception as e:
            # Fallback conservative size
            return (self.portfolio_state['total_balance'] * 0.02) / current_price
    
    def get_todays_trade_count(self) -> int:
        """📈 Today's trade count"""
        
        today = datetime.now().date()
        return sum(1 for trade in self.trade_history 
                  if datetime.fromisoformat(trade['timestamp']).date() == today)
    
    def subscribe_to_alerts(self, callback: callable):
        """📢 Subscribe to alerts"""
        
        if callback not in self.alert_callbacks:
            self.alert_callbacks.append(callback)
            print(f"✅ Alert subscriber added")
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """📊 Risk dashboard"""
        
        return {
            'portfolio_state': self.portfolio_state,
            'risk_metrics': self.risk_metrics,
            'risk_limits': {
                'exposure_limit': self.config['max_portfolio_exposure'],
                'daily_loss_limit': self.config['daily_loss_limit'],
                'max_drawdown': self.config['max_drawdown_limit']
            },
            'current_status': {
                'exposure_usage': self.portfolio_state['total_exposure'] / max(self.portfolio_state['total_balance'], 1),
                'daily_loss_usage': abs(self.portfolio_state['daily_pnl']) / max(self.portfolio_state['total_balance'], 1),
                'trades_today': self.get_todays_trade_count()
            },
            'active_alerts': self.active_alerts[-10:],
            'last_update': datetime.now().isoformat()
        }

# Global instance
risk_manager = None

def initialize_risk_management(config: Dict[str, Any] = None) -> RiskManagementSystem:
    """🛡️ Initialize risk management"""
    global risk_manager
    risk_manager = RiskManagementSystem(config)
    return risk_manager

def get_risk_manager() -> Optional[RiskManagementSystem]:
    """🛡️ Get risk manager"""
    return risk_manager

# Demo
if __name__ == "__main__":
    print("🛡️ RISK MANAGEMENT SYSTEM TEST")
    
    rm = initialize_risk_management()
    
    # Test
    rm.update_portfolio_state(
        balance=10000.0,
        positions={'BTC/USDT': {'value': 1000, 'size': 0.02}},
        recent_trades=[]
    )
    
    test_signal = {
        'action': 'buy',
        'confidence': 0.75,
        'position_size': 0.05
    }
    
    validation = rm.validate_trade_signal(test_signal, 'BTC/USDT', 50000)
    print(f"✅ Validation: {validation}")
    
    dashboard = rm.get_risk_dashboard()
    print(f"📊 Dashboard ready")
    
    print("🛡️ Risk Management ready!") 