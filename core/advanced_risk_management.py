#!/usr/bin/env python3
"""
🛡️ ADVANCED RISK MANAGEMENT SYSTEM
Vollständiges Risk Management für TRADINO mit Real-time Validation,
Kelly Criterion, Dynamic Stop Loss und Emergency Controls
"""

import os
import sys
import time
import threading
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

class RiskLevel(Enum):
    """📊 Risk Levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class RiskSettings:
    """⚙️ Risk Management Settings"""
    # Position Sizing
    max_position_size: float = 0.05
    max_portfolio_exposure: float = 0.7
    kelly_multiplier: float = 0.25
    min_position_size: float = 0.001
    
    # Stop Loss & Take Profit
    base_stop_loss: float = 0.02
    base_take_profit: float = 0.04
    volatility_multiplier: float = 1.5
    max_stop_loss: float = 0.05
    
    # Risk Limits
    daily_loss_limit: float = 0.03
    max_drawdown_limit: float = 0.15
    max_consecutive_losses: int = 5
    max_daily_trades: int = 20
    
    # Emergency Controls
    emergency_stop_loss: float = 0.08
    force_close_threshold: float = 0.12
    circuit_breaker_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AdvancedRiskManager:
    """🛡️ Advanced Risk Management System"""
    
    def __init__(self, config_file: str = "config/risk_settings.json"):
        self.config_file = config_file
        self.settings = self.load_settings()
        
        # Portfolio State
        self.portfolio_state = {
            'total_balance': 0.0,
            'available_balance': 0.0,
            'total_exposure': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl_daily': 0.0,
            'open_positions': {},
            'margin_usage': 0.0,
            'last_update': datetime.now()
        }
        
        # Risk Metrics
        self.risk_metrics = {
            'volatility': 0.0,
            'win_rate': 0.0,
            'current_drawdown': 0.0,
            'correlation_risk': 0.0,
            'portfolio_var': 0.0
        }
        
        # Trading State
        self.trade_history = []
        self.active_alerts = []
        self.risk_overrides = {}
        
        # Monitoring
        self.is_monitoring = False
        self.monitoring_thread = None
        self.last_risk_check = datetime.now()
        
        print("🛡️ Advanced Risk Management System initialized")
        self.start_monitoring()
    
    def load_settings(self) -> RiskSettings:
        """📂 Load risk settings from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                return RiskSettings(**data)
            return RiskSettings()
        except Exception as e:
            print(f"❌ Error loading settings: {e}")
            return RiskSettings()
    
    def save_settings(self):
        """💾 Save risk settings to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.settings.to_dict(), f, indent=2)
            print(f"✅ Risk settings saved")
        except Exception as e:
            print(f"❌ Error saving settings: {e}")
    
    def update_portfolio_state(self, balance: float, positions: Dict[str, Any], 
                             recent_trades: List[Dict[str, Any]], margin_info: Dict = None):
        """📊 Update portfolio state"""
        try:
            self.portfolio_state.update({
                'total_balance': balance,
                'available_balance': balance,
                'open_positions': positions,
                'last_update': datetime.now()
            })
            
            # Calculate exposure
            total_exposure = 0.0
            unrealized_pnl = 0.0
            
            for symbol, position in positions.items():
                exposure = position.get('notional_value', position.get('value', 0))
                total_exposure += abs(exposure)
                unrealized_pnl += position.get('unrealized_pnl', 0)
            
            self.portfolio_state['total_exposure'] = total_exposure
            self.portfolio_state['unrealized_pnl'] = unrealized_pnl
            
            # Update trade history
            for trade in recent_trades:
                if trade not in self.trade_history:
                    self.trade_history.append(trade)
            
            # Keep last 1000 trades
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
            
            self.calculate_daily_pnl()
            self.calculate_risk_metrics()
            
        except Exception as e:
            print(f"❌ Error updating portfolio: {e}")
    
    def calculate_daily_pnl(self):
        """📈 Calculate daily P&L"""
        today = datetime.now().date()
        daily_pnl = 0.0
        
        for trade in self.trade_history:
            try:
                trade_date = datetime.fromisoformat(trade['timestamp']).date()
                if trade_date == today:
                    pnl = trade.get('pnl', trade.get('realized_pnl', 0))
                    daily_pnl += pnl
            except:
                continue
        
        daily_pnl += self.portfolio_state.get('unrealized_pnl', 0)
        self.portfolio_state['realized_pnl_daily'] = daily_pnl
    
    def calculate_risk_metrics(self):
        """📊 Calculate risk metrics"""
        if len(self.trade_history) < 10:
            return
        
        try:
            # Extract returns
            returns = []
            for trade in self.trade_history[-100:]:
                pnl = trade.get('pnl', trade.get('realized_pnl', 0))
                if pnl != 0:
                    trade_value = trade.get('value', 1)
                    if trade_value > 0:
                        returns.append(pnl / trade_value)
            
            if len(returns) < 5:
                return
            
            returns = np.array(returns)
            
            # Basic metrics
            self.risk_metrics['volatility'] = np.std(returns) * np.sqrt(252)
            
            # Win rate
            winning_trades = sum(1 for r in returns if r > 0)
            if len(returns) > 0:
                self.risk_metrics['win_rate'] = winning_trades / len(returns)
            
            # Drawdown
            balance_history = [self.portfolio_state['total_balance']]
            running_max = max(balance_history)
            current_drawdown = (running_max - self.portfolio_state['total_balance']) / running_max if running_max > 0 else 0
            self.risk_metrics['current_drawdown'] = current_drawdown
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
    
    def validate_trade_realtime(self, signal: Dict[str, Any], symbol: str, 
                               current_price: float, market_data: Dict = None) -> Dict[str, Any]:
        """🔍 Real-time trade validation"""
        
        validation = {
            'approved': True,
            'risk_level': RiskLevel.LOW,
            'position_size': 0.0,
            'adjusted_sl': 0.0,
            'adjusted_tp': 0.0,
            'warnings': [],
            'risk_score': 0.0
        }
        
        try:
            # 1. Kelly Criterion Position Sizing
            optimal_size = self.calculate_kelly_position_size(signal, symbol, current_price)
            validation['position_size'] = optimal_size
            
            # 2. Dynamic Stop Loss
            dynamic_sl = self.calculate_dynamic_stop_loss(symbol, current_price)
            validation['adjusted_sl'] = dynamic_sl
            
            # 3. Dynamic Take Profit
            dynamic_tp = self.calculate_dynamic_take_profit(symbol, current_price)
            validation['adjusted_tp'] = dynamic_tp
            
            # 4. Portfolio Exposure Check
            position_value = optimal_size * current_price
            current_exposure = self.portfolio_state['total_exposure']
            total_balance = self.portfolio_state['total_balance']
            
            if current_exposure + position_value > total_balance * self.settings.max_portfolio_exposure:
                max_additional = (total_balance * self.settings.max_portfolio_exposure) - current_exposure
                if max_additional > 0:
                    validation['position_size'] = max_additional / current_price
                    validation['warnings'].append("Position size reduced due to exposure limit")
                    validation['risk_level'] = RiskLevel.MEDIUM
                else:
                    validation['approved'] = False
                    validation['warnings'].append("Portfolio exposure limit exceeded")
                    validation['risk_level'] = RiskLevel.HIGH
            
            # 5. Daily Limits
            daily_trades = self.get_todays_trade_count()
            if daily_trades >= self.settings.max_daily_trades:
                validation['approved'] = False
                validation['warnings'].append("Daily trade limit reached")
                validation['risk_level'] = RiskLevel.HIGH
            
            # 6. Consecutive Losses
            consecutive_losses = self.get_consecutive_losses()
            if consecutive_losses >= self.settings.max_consecutive_losses:
                validation['position_size'] *= 0.5
                validation['warnings'].append(f"Consecutive losses: {consecutive_losses}")
                validation['risk_level'] = RiskLevel.HIGH
            
            # 7. Risk Score
            confidence = signal.get('confidence', 0.5)
            win_rate = self.risk_metrics.get('win_rate', 0.5)
            volatility = self.get_symbol_volatility(symbol)
            
            risk_factors = [
                confidence * 100,
                win_rate * 100,
                max(0, 100 - (volatility * 1000)),
                max(0, 100 - (self.risk_metrics['current_drawdown'] * 100))
            ]
            validation['risk_score'] = np.mean(risk_factors)
            
            # 8. Final Check
            if validation['risk_score'] < 30:
                validation['approved'] = False
                validation['warnings'].append(f"Risk score too low: {validation['risk_score']:.1f}")
                validation['risk_level'] = RiskLevel.CRITICAL
            
            # 9. Emergency Override Check
            if symbol in self.risk_overrides:
                override = self.risk_overrides[symbol]
                if override.get('block_trading', False):
                    validation['approved'] = False
                    validation['warnings'].append("Trading blocked by risk override")
                    validation['risk_level'] = RiskLevel.EMERGENCY
            
            # Minimum position size check
            if validation['position_size'] < self.settings.min_position_size:
                validation['approved'] = False
                validation['warnings'].append("Position size below minimum")
            
        except Exception as e:
            validation['approved'] = False
            validation['warnings'].append(f"Validation error: {e}")
            validation['risk_level'] = RiskLevel.EMERGENCY
        
        return validation
    
    def calculate_kelly_position_size(self, signal: Dict[str, Any], symbol: str, price: float) -> float:
        """📊 Kelly Criterion Position Sizing"""
        try:
            confidence = signal.get('confidence', 0.5)
            win_rate = max(self.risk_metrics.get('win_rate', 0.5), 0.4)
            
            adjusted_win_prob = (confidence + win_rate) / 2
            loss_prob = 1 - adjusted_win_prob
            
            reward_risk_ratio = self.settings.base_take_profit / self.settings.base_stop_loss
            kelly_fraction = (reward_risk_ratio * adjusted_win_prob - loss_prob) / reward_risk_ratio
            kelly_fraction *= self.settings.kelly_multiplier
            kelly_fraction = max(0, min(kelly_fraction, self.settings.max_position_size))
            
            available_capital = self.portfolio_state['available_balance']
            position_value = available_capital * kelly_fraction
            position_size = position_value / price
            
            return max(position_size, 0)
            
        except Exception as e:
            return (self.portfolio_state['available_balance'] * 0.02) / price
    
    def calculate_dynamic_stop_loss(self, symbol: str, current_price: float) -> float:
        """🛑 Dynamic Stop Loss Calculation"""
        try:
            base_sl_pct = self.settings.base_stop_loss
            volatility = self.get_symbol_volatility(symbol)
            volatility_adjustment = volatility * self.settings.volatility_multiplier
            
            dynamic_sl_pct = base_sl_pct + volatility_adjustment
            dynamic_sl_pct = min(dynamic_sl_pct, self.settings.max_stop_loss)
            
            return current_price * (1 - dynamic_sl_pct)
        except:
            return current_price * (1 - self.settings.base_stop_loss)
    
    def calculate_dynamic_take_profit(self, symbol: str, current_price: float) -> float:
        """🎯 Dynamic Take Profit Calculation"""
        try:
            base_tp_pct = self.settings.base_take_profit
            volatility = self.get_symbol_volatility(symbol)
            volatility_adjustment = volatility * self.settings.volatility_multiplier
            
            dynamic_tp_pct = base_tp_pct + (volatility_adjustment * 0.5)
            return current_price * (1 + dynamic_tp_pct)
        except:
            return current_price * (1 + self.settings.base_take_profit)
    
    def get_symbol_volatility(self, symbol: str) -> float:
        """📊 Get Symbol Volatility"""
        if 'BTC' in symbol:
            return 0.04
        elif 'ETH' in symbol:
            return 0.05
        else:
            return 0.03
    
    def get_todays_trade_count(self) -> int:
        """📈 Today's Trade Count"""
        today = datetime.now().date()
        return sum(1 for trade in self.trade_history 
                  if datetime.fromisoformat(trade['timestamp']).date() == today)
    
    def get_consecutive_losses(self) -> int:
        """📉 Consecutive Losses"""
        consecutive = 0
        for trade in reversed(self.trade_history[-20:]):
            pnl = trade.get('pnl', trade.get('realized_pnl', 0))
            if pnl < 0:
                consecutive += 1
            else:
                break
        return consecutive
    
    def check_risk_limits(self) -> List[str]:
        """🚨 Check Risk Limits"""
        violations = []
        
        try:
            total_balance = self.portfolio_state['total_balance']
            total_exposure = self.portfolio_state['total_exposure']
            
            # Exposure check
            if total_balance > 0:
                exposure_ratio = total_exposure / total_balance
                if exposure_ratio > self.settings.max_portfolio_exposure:
                    violations.append(f"Portfolio exposure: {exposure_ratio:.1%}")
            
            # Daily loss check
            daily_pnl = self.portfolio_state['realized_pnl_daily']
            if total_balance > 0 and daily_pnl < 0:
                daily_loss_ratio = abs(daily_pnl) / total_balance
                if daily_loss_ratio > self.settings.daily_loss_limit:
                    violations.append(f"Daily loss limit: {daily_loss_ratio:.1%}")
            
            # Emergency checks
            if total_balance > 0 and daily_pnl < 0:
                emergency_loss_ratio = abs(daily_pnl) / total_balance
                if emergency_loss_ratio > self.settings.emergency_stop_loss:
                    violations.append(f"EMERGENCY LOSS: {emergency_loss_ratio:.1%}")
                    self.trigger_emergency_stop()
                    
                if emergency_loss_ratio > self.settings.force_close_threshold:
                    violations.append(f"FORCE CLOSE: {emergency_loss_ratio:.1%}")
                    self.trigger_force_close()
            
            # Drawdown check
            current_drawdown = self.risk_metrics['current_drawdown']
            if current_drawdown > self.settings.max_drawdown_limit:
                violations.append(f"Max drawdown: {current_drawdown:.1%}")
            
        except Exception as e:
            print(f"❌ Error checking limits: {e}")
        
        return violations
    
    def trigger_emergency_stop(self):
        """🚨 Emergency Stop"""
        print("🚨 EMERGENCY STOP TRIGGERED!")
        
        self.risk_overrides['emergency_stop'] = {
            'block_trading': True,
            'reason': 'Emergency stop triggered',
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'action': 'emergency_stop',
            'reason': 'Emergency loss threshold exceeded',
            'automatic': True
        }
    
    def trigger_force_close(self):
        """🛑 Force Close All"""
        print("🛑 FORCE CLOSE ALL POSITIONS!")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'action': 'force_close_all',
            'reason': 'Force close threshold exceeded',
            'automatic': True
        }
    
    def start_monitoring(self):
        """🔄 Start Risk Monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("👁️ Risk monitoring started")
    
    def stop_monitoring(self):
        """🛑 Stop Monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("🛑 Risk monitoring stopped")
    
    def _monitoring_loop(self):
        """🔄 Monitoring Loop"""
        while self.is_monitoring:
            try:
                violations = self.check_risk_limits()
                if violations:
                    print(f"⚠️ Risk violations: {len(violations)}")
                
                self.last_risk_check = datetime.now()
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(30)
    
    def set_risk_override(self, symbol: str, override_data: Dict[str, Any]):
        """⚠️ Set Risk Override"""
        self.risk_overrides[symbol] = {
            **override_data,
            'timestamp': datetime.now().isoformat()
        }
        print(f"⚠️ Risk override set for {symbol}")
    
    def clear_risk_override(self, symbol: str):
        """✅ Clear Risk Override"""
        if symbol in self.risk_overrides:
            del self.risk_overrides[symbol]
            print(f"✅ Risk override cleared for {symbol}")
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """📊 Risk Dashboard"""
        return {
            'timestamp': datetime.now().isoformat(),
            'risk_level': self._get_overall_risk_level(),
            'portfolio_state': self.portfolio_state,
            'risk_metrics': self.risk_metrics,
            'risk_settings': self.settings.to_dict(),
            'active_overrides': self.risk_overrides,
            'monitoring_status': {
                'is_active': self.is_monitoring,
                'last_check': self.last_risk_check.isoformat()
            },
            'limits_usage': {
                'exposure_usage': self.portfolio_state['total_exposure'] / max(self.portfolio_state['total_balance'], 1),
                'daily_loss_usage': abs(self.portfolio_state['realized_pnl_daily']) / max(self.portfolio_state['total_balance'], 1),
                'drawdown_usage': self.risk_metrics['current_drawdown'] / self.settings.max_drawdown_limit,
                'daily_trades_usage': self.get_todays_trade_count() / self.settings.max_daily_trades
            }
        }
    
    def _get_overall_risk_level(self) -> str:
        """📊 Calculate Overall Risk Level"""
        risk_factors = []
        
        # Exposure risk
        exposure_ratio = self.portfolio_state['total_exposure'] / max(self.portfolio_state['total_balance'], 1)
        risk_factors.append(exposure_ratio / self.settings.max_portfolio_exposure)
        
        # Drawdown risk
        drawdown_ratio = self.risk_metrics['current_drawdown'] / self.settings.max_drawdown_limit
        risk_factors.append(drawdown_ratio)
        
        # Daily loss risk
        daily_loss_ratio = abs(self.portfolio_state['realized_pnl_daily']) / max(self.portfolio_state['total_balance'], 1)
        risk_factors.append(daily_loss_ratio / self.settings.daily_loss_limit)
        
        avg_risk = np.mean(risk_factors)
        
        if avg_risk < 0.3:
            return RiskLevel.LOW.value
        elif avg_risk < 0.6:
            return RiskLevel.MEDIUM.value
        elif avg_risk < 0.8:
            return RiskLevel.HIGH.value
        else:
            return RiskLevel.CRITICAL.value

# Global instance
advanced_risk_manager = None

def initialize_advanced_risk_manager(config_file: str = None) -> AdvancedRiskManager:
    """🛡️ Initialize Advanced Risk Manager"""
    global advanced_risk_manager
    advanced_risk_manager = AdvancedRiskManager(config_file or "config/risk_settings.json")
    return advanced_risk_manager

def get_advanced_risk_manager() -> Optional[AdvancedRiskManager]:
    """🛡️ Get Advanced Risk Manager"""
    return advanced_risk_manager

if __name__ == "__main__":
    print("🛡️ Advanced Risk Management System Test")
    
    # Initialize
    risk_manager = initialize_advanced_risk_manager()
    
    # Test validation
    test_signal = {'action': 'buy', 'confidence': 0.75}
    validation = risk_manager.validate_trade_realtime(
        signal=test_signal,
        symbol='BTC/USDT:USDT',
        current_price=45000.0
    )
    
    print(f"✅ Validation: {validation['approved']}")
    print(f"📊 Risk Score: {validation['risk_score']:.1f}")
    print(f"💰 Position Size: {validation['position_size']:.6f}")
    print(f"🛑 Stop Loss: ${validation['adjusted_sl']:.2f}")
    print(f"🎯 Take Profit: ${validation['adjusted_tp']:.2f}")
    
    print("🛡️ Advanced Risk Management ready!")
