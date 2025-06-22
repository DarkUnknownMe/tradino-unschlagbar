import gym
from gym import spaces
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from loguru import logger
import asyncio
from collections import deque

class BitgetTradingEnvironment(gym.Env):
    """
    OpenAI Gym-kompatible Trading-Umgebung für Reinforcement Learning
    Simuliert reale Bitget-Trading-Bedingungen für RL-Agent Training
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config, market_data_manager, initial_balance=10000):
        super(BitgetTradingEnvironment, self).__init__()
        
        self.config = config
        self.market_data_manager = market_data_manager
        self.initial_balance = initial_balance
        
        # Action Space: Kontinuierliche Aktionen für Position Size
        # -1.0 = Max Short, 0.0 = Neutral, +1.0 = Max Long
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Observation Space: Market Features + Portfolio State
        # 50 Features: OHLCV, Technical Indicators, Portfolio State
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(50,), 
            dtype=np.float32
        )
        
        # Environment State
        self.current_step = 0
        self.max_steps = 1000
        self.current_position = 0.0  # -1 to 1
        self.current_balance = initial_balance
        self.current_equity = initial_balance
        self.peak_equity = initial_balance
        self.trade_history = []
        
        # Market Data
        self.market_data = None
        self.current_price = 0.0
        self.price_history = deque(maxlen=100)
        
        # Features für Observation
        self.feature_extractor = TradingFeatureExtractor()
        
        # Reward Calculation
        self.reward_calculator = RLRewardCalculator()
        
        # Risk Management
        self.max_drawdown_limit = 0.15  # 15% Max Drawdown
        self.max_position_size = 1.0    # 100% Portfolio
        
        logger.info("🎮 RL Trading Environment initialisiert")
    
    def reset(self):
        """
        Environment für neue Episode zurücksetzen
        """
        try:
            self.current_step = 0
            self.current_position = 0.0
            self.current_balance = self.initial_balance
            self.current_equity = self.initial_balance
            self.peak_equity = self.initial_balance
            self.trade_history = []
            self.price_history.clear()
            
            # Neue Market Data laden
            self.market_data = self._load_random_market_episode()
            self.current_price = self.market_data.iloc[0]['close']
            
            # Initial Observation
            observation = self._get_observation()
            
            logger.info(f"🔄 RL Environment Reset - Episode Start")
            return observation
            
        except Exception as e:
            logger.error(f"❌ RL Environment Reset Fehler: {e}")
            return np.zeros(50, dtype=np.float32)
    
    def step(self, action):
        """
        Einen Schritt in der Environment ausführen
        """
        try:
            # Action Processing
            desired_position = np.clip(action[0], -1.0, 1.0)
            
            # Market Data für aktuellen Schritt
            current_market_data = self.market_data.iloc[self.current_step]
            self.current_price = current_market_data['close']
            self.price_history.append(self.current_price)
            
            # Position Change ausführen
            position_change = desired_position - self.current_position
            trade_executed = False
            
            if abs(position_change) > 0.01:  # Mindest-Position-Change
                trade_result = self._execute_position_change(
                    position_change, current_market_data
                )
                trade_executed = True
                self.trade_history.append(trade_result)
            
            # Update Position
            self.current_position = desired_position
            
            # Portfolio Value Update
            self._update_portfolio_value(current_market_data)
            
            # Reward Calculation
            reward = self.reward_calculator.calculate_reward(
                portfolio_return=self._get_portfolio_return(),
                position_change=position_change,
                market_data=current_market_data,
                trade_executed=trade_executed
            )
            
            # Next Observation
            self.current_step += 1
            observation = self._get_observation()
            
            # Episode Ende Check
            done = self._check_episode_done()
            
            # Info Dictionary
            info = {
                'current_equity': self.current_equity,
                'current_position': self.current_position,
                'portfolio_return': self._get_portfolio_return(),
                'drawdown': self._get_current_drawdown(),
                'trade_executed': trade_executed,
                'total_trades': len(self.trade_history)
            }
            
            return observation, reward, done, info
            
        except Exception as e:
            logger.error(f"❌ RL Environment Step Fehler: {e}")
            return np.zeros(50), -1.0, True, {}
    
    def _get_observation(self):
        """
        Aktuelle Observation für RL-Agent erstellen
        """
        try:
            if self.current_step >= len(self.market_data):
                return np.zeros(50, dtype=np.float32)
            
            current_data = self.market_data.iloc[self.current_step]
            
            # Technical Features (35 Features)
            technical_features = self.feature_extractor.extract_technical_features(
                self.market_data.iloc[max(0, self.current_step-20):self.current_step+1]
            )
            
            # Portfolio Features (10 Features)
            portfolio_features = np.array([
                self.current_position,                          # Aktuelle Position
                self.current_equity / self.initial_balance,     # Equity Ratio
                self._get_portfolio_return(),                   # Portfolio Return
                self._get_current_drawdown(),                   # Current Drawdown
                len(self.trade_history) / 100.0,              # Trade Count (normalized)
                self._get_win_rate(),                          # Win Rate
                self._get_sharpe_ratio(),                      # Sharpe Ratio
                self._get_volatility(),                        # Portfolio Volatility
                self._get_momentum(),                          # Price Momentum
                self._get_rsi()                                # RSI
            ])
            
            # Market Context Features (5 Features)
            market_features = np.array([
                current_data.get('volume', 0) / 1e6,          # Volume (normalized)
                current_data.get('volatility', 0),            # Volatility
                self._get_market_trend(),                      # Trend Strength
                self._get_market_regime(),                     # Market Regime
                self.current_step / self.max_steps             # Episode Progress
            ])
            
            # Combine All Features
            observation = np.concatenate([
                technical_features[:35],  # 35 Technical Features
                portfolio_features,       # 10 Portfolio Features  
                market_features          # 5 Market Features
            ])
            
            # Ensure correct shape
            observation = observation[:50]  # Take first 50 features
            if len(observation) < 50:
                observation = np.pad(observation, (0, 50 - len(observation)))
            
            return observation.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Observation Generation Fehler: {e}")
            return np.zeros(50, dtype=np.float32)
    
    def _execute_position_change(self, position_change, market_data):
        """
        Position-Änderung ausführen und Trade-Result zurückgeben
        """
        try:
            # Trade Size berechnen
            trade_size = abs(position_change) * self.current_equity
            
            # Slippage und Gebühren simulieren
            slippage = 0.0005  # 0.05% Slippage
            fees = 0.0006      # 0.06% Trading Fees
            
            # Execution Price mit Slippage
            if position_change > 0:  # Long Position
                execution_price = self.current_price * (1 + slippage)
            else:  # Short Position
                execution_price = self.current_price * (1 - slippage)
            
            # Trade Costs
            total_costs = trade_size * fees
            
            # Portfolio Update
            self.current_balance -= total_costs
            
            trade_result = {
                'timestamp': datetime.now(),
                'action': 'BUY' if position_change > 0 else 'SELL',
                'size': abs(position_change),
                'price': execution_price,
                'costs': total_costs,
                'portfolio_value': self.current_equity
            }
            
            logger.debug(f"🔄 RL Trade: {trade_result['action']} {trade_result['size']:.3f} @ {execution_price:.2f}")
            
            return trade_result
            
        except Exception as e:
            logger.error(f"❌ Position Change Execution Fehler: {e}")
            return {}
    
    def _update_portfolio_value(self, market_data):
        """
        Portfolio Value basierend auf aktueller Position und Marktpreis aktualisieren
        """
        try:
            # Unrealized PnL von Position
            if self.current_position != 0:
                # Vereinfachte PnL Berechnung für Demo
                price_change = (self.current_price - self.price_history[-2]) if len(self.price_history) > 1 else 0
                position_pnl = self.current_position * price_change * self.current_equity * 0.01  # 1% per price unit
                
                self.current_equity = self.current_balance + position_pnl
            else:
                self.current_equity = self.current_balance
            
            # Peak Equity Update
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
                
        except Exception as e:
            logger.error(f"❌ Portfolio Value Update Fehler: {e}")
    
    def _check_episode_done(self):
        """
        Prüfen ob Episode beendet werden soll
        """
        # Max Steps erreicht
        if self.current_step >= min(self.max_steps, len(self.market_data) - 1):
            return True
        
        # Max Drawdown überschritten
        if self._get_current_drawdown() > self.max_drawdown_limit:
            logger.warning(f"⚠️ RL Episode beendet: Max Drawdown ({self.max_drawdown_limit*100:.1f}%) überschritten")
            return True
        
        # Equity zu niedrig
        if self.current_equity < self.initial_balance * 0.5:  # 50% Loss
            logger.warning(f"⚠️ RL Episode beendet: Equity zu niedrig")
            return True
        
        return False
    
    def _load_random_market_episode(self):
        """
        Zufällige Market Data Episode für Training laden
        """
        try:
            # Hier würde normalerweise historische Daten geladen werden
            # Für Demo: Generiere synthetische Daten
            np.random.seed()  # Random seed für Variation
            
            steps = min(self.max_steps, 1000)
            
            # Synthetische OHLCV Daten generieren
            base_price = 50000  # Starting price
            data = []
            
            for i in range(steps):
                # Random Walk mit Trend
                price_change = np.random.normal(0, 0.02)  # 2% daily volatility
                base_price *= (1 + price_change)
                
                # OHLCV simulieren
                high = base_price * (1 + abs(np.random.normal(0, 0.01)))
                low = base_price * (1 - abs(np.random.normal(0, 0.01)))
                volume = np.random.uniform(1e6, 5e6)
                
                data.append({
                    'open': base_price,
                    'high': high,
                    'low': low,
                    'close': base_price,
                    'volume': volume,
                    'volatility': abs(price_change)
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"❌ Market Episode Loading Fehler: {e}")
            # Fallback: Einfache synthetische Daten
            return pd.DataFrame({
                'open': [50000] * 100,
                'high': [50000] * 100,
                'low': [50000] * 100,
                'close': [50000] * 100,
                'volume': [1e6] * 100,
                'volatility': [0.01] * 100
            })
    
    # Helper Methods für Portfolio Metriken
    def _get_portfolio_return(self):
        return (self.current_equity - self.initial_balance) / self.initial_balance
    
    def _get_current_drawdown(self):
        if self.peak_equity == 0:
            return 0
        return (self.peak_equity - self.current_equity) / self.peak_equity
    
    def _get_win_rate(self):
        if not self.trade_history:
            return 0.5
        # Vereinfachte Win Rate Berechnung
        return 0.6  # Placeholder
    
    def _get_sharpe_ratio(self):
        # Vereinfachte Sharpe Ratio
        return max(-3, min(3, self._get_portfolio_return() * 10))
    
    def _get_volatility(self):
        if len(self.price_history) < 2:
            return 0
        returns = np.diff(list(self.price_history)) / list(self.price_history)[:-1]
        return np.std(returns) if len(returns) > 0 else 0
    
    def _get_momentum(self):
        if len(self.price_history) < 10:
            return 0
        return (self.price_history[-1] - self.price_history[-10]) / self.price_history[-10]
    
    def _get_rsi(self):
        # Vereinfachte RSI
        return 0.5  # Placeholder
    
    def _get_market_trend(self):
        if len(self.price_history) < 20:
            return 0
        return (self.price_history[-1] - self.price_history[-20]) / self.price_history[-20]
    
    def _get_market_regime(self):
        # Market Regime: 0 = Bear, 0.5 = Sideways, 1 = Bull
        trend = self._get_market_trend()
        if trend > 0.05:
            return 1.0  # Bull
        elif trend < -0.05:
            return 0.0  # Bear
        else:
            return 0.5  # Sideways

    def render(self, mode='human'):
        """
        Environment Rendering für Debugging
        """
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Position: {self.current_position:.3f}")
            print(f"Equity: ${self.current_equity:.2f}")
            print(f"Return: {self._get_portfolio_return()*100:.2f}%")
            print(f"Drawdown: {self._get_current_drawdown()*100:.2f}%")
            print("-" * 40)


class TradingFeatureExtractor:
    """
    Feature Extraction für RL Trading Environment
    """
    
    def extract_technical_features(self, market_data):
        """
        Technische Indikatoren als Features extrahieren
        """
        try:
            if len(market_data) < 2:
                return np.zeros(35)
            
            # Basic Price Features
            close_prices = market_data['close'].values
            high_prices = market_data['high'].values
            low_prices = market_data['low'].values
            volumes = market_data['volume'].values
            
            features = []
            
            # Price-based Features (10)
            features.extend([
                close_prices[-1] / close_prices[0] - 1,  # Total Return
                np.mean(close_prices[-5:]) / close_prices[-1] - 1,  # MA5 Ratio
                np.mean(close_prices[-10:]) / close_prices[-1] - 1,  # MA10 Ratio
                np.std(close_prices) / np.mean(close_prices),  # CV
                (close_prices[-1] - close_prices[-2]) / close_prices[-2],  # Daily Return
                np.max(close_prices) / close_prices[-1] - 1,  # Distance to High
                close_prices[-1] / np.min(close_prices) - 1,  # Distance to Low
                np.mean(high_prices - low_prices) / np.mean(close_prices),  # Average Range
                volumes[-1] / np.mean(volumes) - 1,  # Volume Ratio
                len(close_prices)  # Lookback Length
            ])
            
            # Momentum Features (10)
            if len(close_prices) >= 5:
                momentum_5 = (close_prices[-1] - close_prices[-5]) / close_prices[-5]
                momentum_3 = (close_prices[-1] - close_prices[-3]) / close_prices[-3]
            else:
                momentum_5 = momentum_3 = 0
                
            features.extend([
                momentum_5,
                momentum_3,
                np.corrcoef(range(len(close_prices)), close_prices)[0,1] if len(close_prices) > 2 else 0,  # Trend
                0.5,  # RSI Placeholder
                0.5,  # MACD Placeholder
                0.5,  # Bollinger Position Placeholder
                0.5,  # Stochastic Placeholder
                0.5,  # Williams %R Placeholder
                0.5,  # CCI Placeholder
                0.5   # ADX Placeholder
            ])
            
            # Volatility Features (10)
            returns = np.diff(close_prices) / close_prices[:-1] if len(close_prices) > 1 else [0]
            
            features.extend([
                np.std(returns),  # Historical Volatility
                np.std(returns[-5:]) if len(returns) >= 5 else 0,  # Short-term Vol
                np.std(returns[-10:]) if len(returns) >= 10 else 0,  # Medium-term Vol
                np.mean(np.abs(returns)),  # Mean Absolute Deviation
                len([r for r in returns if r > 0]) / len(returns) if returns else 0.5,  # Up Days Ratio
                np.max(returns) if returns else 0,  # Max Return
                np.min(returns) if returns else 0,  # Min Return
                np.mean(returns),  # Mean Return
                np.skew(returns) if len(returns) > 2 else 0,  # Skewness
                np.kurtosis(returns) if len(returns) > 3 else 0  # Kurtosis
            ])
            
            # Volume Features (5)
            features.extend([
                volumes[-1],  # Current Volume
                np.mean(volumes),  # Average Volume
                np.std(volumes),  # Volume Volatility
                np.corrcoef(volumes, close_prices)[0,1] if len(volumes) > 2 else 0,  # Price-Volume Correlation
                np.sum(volumes)  # Total Volume
            ])
            
            # Normalize and ensure correct length
            features = np.array(features[:35])
            if len(features) < 35:
                features = np.pad(features, (0, 35 - len(features)))
            
            # Replace NaN with 0
            features = np.nan_to_num(features)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Feature Extraction Fehler: {e}")
            return np.zeros(35)


class RLRewardCalculator:
    """
    Sophisticated Reward Function für RL Trading
    """
    
    def __init__(self):
        self.reward_weights = {
            'return': 1.0,       # Portfolio Return
            'risk': -0.5,        # Risk Penalty
            'transaction': -0.1,  # Transaction Cost
            'drawdown': -2.0,    # Drawdown Penalty
            'sharpe': 0.5        # Sharpe Ratio Bonus
        }
    
    def calculate_reward(self, portfolio_return, position_change, market_data, trade_executed):
        """
        Multi-Objective Reward Function
        """
        try:
            reward = 0.0
            
            # Return Component
            reward += portfolio_return * self.reward_weights['return']
            
            # Risk Penalty
            volatility = market_data.get('volatility', 0.01)
            risk_penalty = volatility * abs(position_change)
            reward += risk_penalty * self.reward_weights['risk']
            
            # Transaction Cost Penalty
            if trade_executed:
                transaction_penalty = abs(position_change) * 0.001  # 0.1% cost
                reward += transaction_penalty * self.reward_weights['transaction']
            
            # Drawdown Penalty (würde in _update_portfolio_value berechnet)
            drawdown_penalty = 0  # Placeholder
            reward += drawdown_penalty * self.reward_weights['drawdown']
            
            # Sharpe Bonus (vereinfacht)
            if portfolio_return > 0 and volatility > 0:
                sharpe_bonus = portfolio_return / volatility
                reward += sharpe_bonus * self.reward_weights['sharpe']
            
            # Clip reward to reasonable range
            reward = np.clip(reward, -10.0, 10.0)
            
            return float(reward)
            
        except Exception as e:
            logger.error(f"❌ Reward Calculation Fehler: {e}")
            return 0.0

# Zusätzliche Imports für Scipy Stats
try:
    from scipy.stats import skew, kurtosis
    np.skew = skew
    np.kurtosis = kurtosis
except ImportError:
    # Fallback implementations
    def skew(x):
        return 0.0
    def kurtosis(x):
        return 0.0
    np.skew = skew
    np.kurtosis = kurtosis 