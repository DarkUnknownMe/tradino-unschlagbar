import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
from loguru import logger
import asyncio
import joblib
import os
from datetime import datetime

# Stable Baselines3 für professionelle RL Implementation
try:
    from stable_baselines3 import PPO, A2C, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Stable Baselines3 nicht verfügbar - verwende Custom Implementation")
    STABLE_BASELINES_AVAILABLE = False

class TRADINOReinforcementAgent:
    """
    Hochentwickelter Reinforcement Learning Agent für TRADINO
    Verwendet PPO (Proximal Policy Optimization) für kontinuierliches Lernen
    """
    
    def __init__(self, config, trading_environment):
        self.config = config
        self.trading_env = trading_environment
        
        # RL Configuration
        self.learning_rate = 0.0003
        self.batch_size = 64
        self.n_steps = 2048
        self.gamma = 0.99  # Discount Factor
        self.gae_lambda = 0.95  # GAE Lambda
        
        # Training State
        self.total_timesteps = 0
        self.training_episodes = 0
        self.best_performance = -np.inf
        
        # Experience Buffer
        self.experience_buffer = deque(maxlen=10000)
        
        # Model Storage
        self.model_save_path = "data/models/rl_agent"
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Initialize Agent
        if STABLE_BASELINES_AVAILABLE:
            self.agent = self._initialize_stable_baselines_agent()
        else:
            self.agent = self._initialize_custom_agent()
        
        # Performance Tracking
        self.performance_history = []
        self.training_metrics = {
            'episodes': 0,
            'total_reward': 0,
            'average_reward': 0,
            'best_reward': -np.inf,
            'win_rate': 0,
            'last_training': None
        }
        
        logger.info("🧠 TRADINO Reinforcement Learning Agent initialisiert")
    
    def _initialize_stable_baselines_agent(self):
        """
        Stable Baselines3 PPO Agent initialisieren
        """
        try:
            # Vectorized Environment
            vec_env = DummyVecEnv([lambda: self.trading_env])
            
            # PPO Agent mit optimierten Hyperparametern für Trading
            agent = PPO(
                'MlpPolicy',
                vec_env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=0.2,
                ent_coef=0.01,          # Entropy Coefficient für Exploration
                vf_coef=0.5,            # Value Function Coefficient
                max_grad_norm=0.5,      # Gradient Clipping
                policy_kwargs=dict(
                    net_arch=[256, 256, 128],  # Neural Network Architecture
                    activation_fn=nn.ReLU
                ),
                verbose=1,
                tensorboard_log=f"{self.model_save_path}/tensorboard/"
            )
            
            logger.success("✅ Stable Baselines3 PPO Agent initialisiert")
            return agent
            
        except Exception as e:
            logger.error(f"❌ Stable Baselines3 Agent Initialization Fehler: {e}")
            return self._initialize_custom_agent()
    
    def _initialize_custom_agent(self):
        """
        Custom PPO Implementation als Fallback
        """
        try:
            custom_agent = CustomPPOAgent(
                state_dim=50,
                action_dim=1,
                learning_rate=self.learning_rate,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            logger.success("✅ Custom PPO Agent initialisiert")
            return custom_agent
            
        except Exception as e:
            logger.error(f"❌ Custom Agent Initialization Fehler: {e}")
            return None
    
    async def train_agent(self, total_timesteps=50000, episodes=100):
        """
        RL Agent Training mit kontinuierlichem Lernen
        """
        try:
            logger.info(f"🎓 RL Agent Training startet: {total_timesteps} timesteps")
            
            if STABLE_BASELINES_AVAILABLE and hasattr(self.agent, 'learn'):
                await self._train_stable_baselines(total_timesteps)
            else:
                await self._train_custom_agent(episodes)
            
            # Performance Evaluation nach Training
            await self._evaluate_agent_performance()
            
            # Model speichern
            await self.save_agent()
            
            logger.success("✅ RL Agent Training abgeschlossen")
            
        except Exception as e:
            logger.error(f"❌ RL Agent Training Fehler: {e}")
    
    async def _train_stable_baselines(self, total_timesteps):
        """
        Training mit Stable Baselines3
        """
        try:
            # Custom Callback für Training Monitoring
            callback = TradingTrainingCallback(self)
            
            # Training starten
            self.agent.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                tb_log_name="ppo_trading"
            )
            
            self.total_timesteps += total_timesteps
            self.training_metrics['last_training'] = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Stable Baselines Training Fehler: {e}")
    
    async def _train_custom_agent(self, episodes):
        """
        Training mit Custom Agent
        """
        try:
            for episode in range(episodes):
                logger.info(f"🎓 Training Episode {episode + 1}/{episodes}")
                
                # Episode Reset
                state = self.trading_env.reset()
                episode_reward = 0
                episode_steps = 0
                done = False
                
                episode_experiences = []
                
                while not done and episode_steps < 1000:
                    # Action Selection
                    action = self.agent.select_action(state)
                    
                    # Environment Step
                    next_state, reward, done, info = self.trading_env.step(action)
                    
                    # Experience Storage
                    experience = {
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'done': done
                    }
                    episode_experiences.append(experience)
                    
                    # Update für nächsten Step
                    state = next_state
                    episode_reward += reward
                    episode_steps += 1
                
                # Episode abgeschlossen - Agent Training
                if len(episode_experiences) > 10:
                    self.agent.train_on_batch(episode_experiences)
                
                # Performance Tracking
                self.performance_history.append({
                    'episode': episode + 1,
                    'reward': episode_reward,
                    'steps': episode_steps,
                    'final_equity': info.get('current_equity', 0),
                    'portfolio_return': info.get('portfolio_return', 0)
                })
                
                # Metrics Update
                self._update_training_metrics(episode_reward)
                
                # Progress Logging
                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean([p['reward'] for p in self.performance_history[-10:]])
                    logger.info(f"📊 Episode {episode + 1}: Avg Reward (last 10): {avg_reward:.3f}")
                
        except Exception as e:
            logger.error(f"❌ Custom Agent Training Fehler: {e}")
    
    async def get_trading_action(self, market_observation):
        """
        Trading Action vom RL Agent für Live Trading
        """
        try:
            # Observation Processing
            processed_obs = self._process_market_observation(market_observation)
            
            # Action Prediction
            if STABLE_BASELINES_AVAILABLE and hasattr(self.agent, 'predict'):
                action, _ = self.agent.predict(processed_obs, deterministic=True)
            else:
                action = self.agent.select_action(processed_obs, deterministic=True)
            
            # Action Processing
            position_size = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
            position_size = np.clip(position_size, -1.0, 1.0)
            
            # Trading Signal erstellen
            trading_signal = {
                'action': 'BUY' if position_size > 0.1 else 'SELL' if position_size < -0.1 else 'HOLD',
                'confidence': abs(position_size),
                'position_size': abs(position_size),
                'source': 'RL_Agent',
                'timestamp': datetime.now(),
                'raw_action': position_size
            }
            
            logger.debug(f"🤖 RL Agent Action: {trading_signal['action']} (Confidence: {trading_signal['confidence']:.3f})")
            
            return trading_signal
            
        except Exception as e:
            logger.error(f"❌ RL Trading Action Fehler: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'position_size': 0.0,
                'source': 'RL_Agent_Error',
                'error': str(e)
            }
    
    def _process_market_observation(self, market_observation):
        """
        Market Observation für RL Agent verarbeiten
        """
        try:
            # Market Data extrahieren
            if isinstance(market_observation, dict):
                # Feature Extraction von Market Data
                features = []
                
                # Price Features
                price_data = market_observation.get('price_data', {})
                features.extend([
                    price_data.get('close', 0) / 50000,  # Normalized Price
                    price_data.get('volume', 0) / 1e6,   # Normalized Volume
                    price_data.get('volatility', 0),     # Volatility
                ])
                
                # Technical Indicators
                indicators = market_observation.get('indicators', {})
                features.extend([
                    indicators.get('rsi', 50) / 100,     # RSI (0-1)
                    indicators.get('macd', 0),           # MACD
                    indicators.get('bb_position', 0.5),  # Bollinger Position
                ])
                
                # Sentiment Data
                sentiment = market_observation.get('sentiment', {})
                features.extend([
                    sentiment.get('news_sentiment', 0),
                    sentiment.get('social_sentiment', 0),
                    sentiment.get('market_sentiment', 0),
                ])
                
                # Pattern Recognition
                patterns = market_observation.get('patterns', {})
                features.extend([
                    patterns.get('trend_strength', 0),
                    patterns.get('support_resistance', 0),
                    patterns.get('breakout_probability', 0),
                ])
                
                # Portfolio State (wenn verfügbar)
                portfolio = market_observation.get('portfolio', {})
                features.extend([
                    portfolio.get('current_position', 0),
                    portfolio.get('unrealized_pnl', 0) / 1000,  # Normalized
                    portfolio.get('portfolio_return', 0),
                ])
                
                # Pad oder trim zu 50 Features
                while len(features) < 50:
                    features.append(0.0)
                features = features[:50]
                
                return np.array(features, dtype=np.float32)
            
            else:
                # Fallback: Direkte Observation verwenden
                return np.array(market_observation, dtype=np.float32)[:50]
                
        except Exception as e:
            logger.error(f"❌ Market Observation Processing Fehler: {e}")
            return np.zeros(50, dtype=np.float32)
    
    async def _evaluate_agent_performance(self):
        """
        Agent Performance nach Training evaluieren
        """
        try:
            logger.info("📊 RL Agent Performance Evaluation...")
            
            evaluation_episodes = 10
            total_rewards = []
            total_returns = []
            
            for episode in range(evaluation_episodes):
                state = self.trading_env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    # Deterministic Action (kein Exploration)
                    if STABLE_BASELINES_AVAILABLE and hasattr(self.agent, 'predict'):
                        action, _ = self.agent.predict(state, deterministic=True)
                    else:
                        action = self.agent.select_action(state, deterministic=True)
                    
                    state, reward, done, info = self.trading_env.step(action)
                    episode_reward += reward
                
                total_rewards.append(episode_reward)
                total_returns.append(info.get('portfolio_return', 0))
            
            # Performance Metriken
            avg_reward = np.mean(total_rewards)
            avg_return = np.mean(total_returns)
            win_rate = len([r for r in total_returns if r > 0]) / len(total_returns)
            
            # Best Performance Update
            if avg_reward > self.best_performance:
                self.best_performance = avg_reward
                logger.success(f"🏆 Neue beste Performance: {avg_reward:.3f}")
            
            # Metrics Update
            self.training_metrics.update({
                'average_reward': avg_reward,
                'average_return': avg_return,
                'win_rate': win_rate,
                'evaluation_episodes': evaluation_episodes
            })
            
            logger.success(f"✅ Evaluation abgeschlossen:")
            logger.info(f"   📈 Durchschnittlicher Reward: {avg_reward:.3f}")
            logger.info(f"   💰 Durchschnittlicher Return: {avg_return*100:.2f}%")
            logger.info(f"   🎯 Win Rate: {win_rate*100:.1f}%")
            
            return {
                'avg_reward': avg_reward,
                'avg_return': avg_return,
                'win_rate': win_rate
            }
            
        except Exception as e:
            logger.error(f"❌ Agent Performance Evaluation Fehler: {e}")
            return {}
    
    def _update_training_metrics(self, episode_reward):
        """
        Training Metriken aktualisieren
        """
        self.training_metrics['episodes'] += 1
        self.training_metrics['total_reward'] += episode_reward
        
        if episode_reward > self.training_metrics['best_reward']:
            self.training_metrics['best_reward'] = episode_reward
        
        if self.training_metrics['episodes'] > 0:
            self.training_metrics['average_reward'] = (
                self.training_metrics['total_reward'] / self.training_metrics['episodes']
            )
    
    async def save_agent(self, filename=None):
        """
        RL Agent Model speichern
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rl_agent_{timestamp}"
            
            save_path = os.path.join(self.model_save_path, filename)
            
            if STABLE_BASELINES_AVAILABLE and hasattr(self.agent, 'save'):
                # Stable Baselines3 Model speichern
                self.agent.save(save_path)
                
                # Zusätzliche Metriken speichern
                metrics_path = f"{save_path}_metrics.pkl"
                joblib.dump(self.training_metrics, metrics_path)
                
            else:
                # Custom Agent speichern
                torch.save({
                    'model_state_dict': self.agent.state_dict(),
                    'optimizer_state_dict': self.agent.optimizer.state_dict(),
                    'training_metrics': self.training_metrics,
                    'performance_history': self.performance_history
                }, f"{save_path}.pt")
            
            logger.success(f"✅ RL Agent gespeichert: {save_path}")
            
        except Exception as e:
            logger.error(f"❌ RL Agent Save Fehler: {e}")
    
    async def load_agent(self, filename):
        """
        RL Agent Model laden
        """
        try:
            load_path = os.path.join(self.model_save_path, filename)
            
            if STABLE_BASELINES_AVAILABLE:
                if os.path.exists(f"{load_path}.zip"):
                    self.agent = PPO.load(load_path)
                    
                    # Metriken laden
                    metrics_path = f"{load_path}_metrics.pkl"
                    if os.path.exists(metrics_path):
                        self.training_metrics = joblib.load(metrics_path)
                    
                    logger.success(f"✅ Stable Baselines3 Agent geladen: {load_path}")
                    
            else:
                if os.path.exists(f"{load_path}.pt"):
                    checkpoint = torch.load(f"{load_path}.pt")
                    self.agent.load_state_dict(checkpoint['model_state_dict'])
                    self.training_metrics = checkpoint.get('training_metrics', {})
                    self.performance_history = checkpoint.get('performance_history', [])
                    
                    logger.success(f"✅ Custom Agent geladen: {load_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ RL Agent Load Fehler: {e}")
            return False
    
    def get_training_status(self):
        """
        Aktueller Training Status
        """
        return {
            'is_trained': self.training_metrics['episodes'] > 0,
            'total_episodes': self.training_metrics['episodes'],
            'best_performance': self.best_performance,
            'last_training': self.training_metrics.get('last_training'),
            'metrics': self.training_metrics.copy(),
            'agent_type': 'Stable_Baselines3_PPO' if STABLE_BASELINES_AVAILABLE else 'Custom_PPO'
        }


class CustomPPOAgent:
    """
    Custom PPO Implementation als Fallback
    """
    
    def __init__(self, state_dim, action_dim, learning_rate=0.0003, device='cpu'):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.actor = self._build_actor_network().to(self.device)
        self.critic = self._build_critic_network().to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # PPO Parameters
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        
        logger.info("🧠 Custom PPO Agent initialisiert")
    
    def _build_actor_network(self):
        """
        Actor Network (Policy) erstellen
        """
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Tanh()  # Action zwischen -1 und 1
        )
    
    def _build_critic_network(self):
        """
        Critic Network (Value Function) erstellen
        """
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def select_action(self, state, deterministic=False):
        """
        Action Selection
        """
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_mean = self.actor(state_tensor)
                
                if deterministic:
                    action = action_mean
                else:
                    # Add noise for exploration
                    noise = torch.normal(0, 0.1, action_mean.shape).to(self.device)
                    action = action_mean + noise
                    action = torch.clamp(action, -1, 1)
            
            return action.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"❌ Custom Agent Action Selection Fehler: {e}")
            return np.array([0.0])
    
    def train_on_batch(self, experiences):
        """
        Training auf Batch von Experiences
        """
        try:
            if len(experiences) < 10:
                return
            
            # Batch preparation
            states = torch.FloatTensor([exp['state'] for exp in experiences]).to(self.device)
            actions = torch.FloatTensor([exp['action'] for exp in experiences]).to(self.device)
            rewards = torch.FloatTensor([exp['reward'] for exp in experiences]).to(self.device)
            next_states = torch.FloatTensor([exp['next_state'] for exp in experiences]).to(self.device)
            dones = torch.BoolTensor([exp['done'] for exp in experiences]).to(self.device)
            
            # Value predictions
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            
            # Calculate advantages
            returns = rewards + 0.99 * next_values * (~dones)
            advantages = returns - values
            
            # Actor loss (simplified PPO)
            action_probs = self.actor(states)
            actor_loss = -torch.mean(advantages.detach() * action_probs.squeeze())
            
            # Critic loss
            critic_loss = nn.MSELoss()(values, returns.detach())
            
            # Update networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
        except Exception as e:
            logger.error(f"❌ Custom Agent Training Fehler: {e}")
    
    def state_dict(self):
        """
        Model State für Speichern
        """
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """
        Model State laden
        """
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])


class TradingTrainingCallback(BaseCallback):
    """
    Custom Callback für Stable Baselines3 Training Monitoring
    """
    
    def __init__(self, rl_agent):
        super().__init__()
        self.rl_agent = rl_agent
        self.episode_rewards = []
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # Episode Ende Detection
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            
            # Episode Reward tracking
            if 'episode' in self.locals.get('infos', [{}])[0]:
                episode_reward = self.locals['infos'][0]['episode']['r']
                self.episode_rewards.append(episode_reward)
                
                # Logging alle 10 Episodes
                if self.episode_count % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    logger.info(f"🎓 RL Training Episode {self.episode_count}: Avg Reward: {avg_reward:.3f}")
        
        return True


# Integration in TRADINO System
class RLTradingIntegration:
    """
    Integration des RL Agents in das TRADINO Trading System
    """
    
    def __init__(self, config, trading_engine):
        self.config = config
        self.trading_engine = trading_engine
        
        # Import RL Environment
        from .rl_environment import BitgetTradingEnvironment
        
        # RL Environment Setup
        self.trading_env = BitgetTradingEnvironment(
            config=config,
            market_data_manager=trading_engine.market_intelligence,
            initial_balance=config.get('rl_initial_balance', 10000)
        )
        
        # RL Agent
        self.rl_agent = TRADINOReinforcementAgent(config, self.trading_env)
        
        # Integration Settings
        self.rl_enabled = config.get('rl_enabled', True)
        self.rl_weight = config.get('rl_signal_weight', 0.3)  # 30% Gewichtung
        
        logger.info("🚀 RL Trading Integration initialisiert")
    
    async def initialize(self):
        """
        RL Integration initialisieren
        """
        try:
            if self.rl_enabled:
                # Versuche existierendes Model zu laden
                existing_models = [f for f in os.listdir(self.rl_agent.model_save_path) 
                                 if f.startswith('rl_agent')]
                
                if existing_models:
                    latest_model = sorted(existing_models)[-1].replace('.zip', '').replace('.pt', '')
                    success = await self.rl_agent.load_agent(latest_model)
                    
                    if success:
                        logger.success(f"✅ Existierendes RL Model geladen: {latest_model}")
                    else:
                        logger.warning("⚠️ RL Model laden fehlgeschlagen - starte Training")
                        await self._initial_training()
                else:
                    logger.info("🎓 Kein existierendes RL Model - starte Training")
                    await self._initial_training()
            
            logger.success("✅ RL Integration bereit")
            
        except Exception as e:
            logger.error(f"❌ RL Integration Initialization Fehler: {e}")
    
    async def _initial_training(self):
        """
        Initial RL Agent Training
        """
        try:
            logger.info("🎓 Starte Initial RL Training...")
            
            # Schnelles Initial Training (weniger Timesteps für Demo)
            await self.rl_agent.train_agent(
                total_timesteps=10000,  # Reduziert für Demo
                episodes=50
            )
            
            logger.success("✅ Initial RL Training abgeschlossen")
            
        except Exception as e:
            logger.error(f"❌ Initial RL Training Fehler: {e}")
    
    async def get_rl_signal(self, market_data):
        """
        RL Signal für Trading Strategy Integration
        """
        try:
            if not self.rl_enabled:
                return None
            
            # Market Observation erstellen
            market_observation = await self._create_market_observation(market_data)
            
            # RL Action
            rl_signal = await self.rl_agent.get_trading_action(market_observation)
            
            # Signal Gewichtung
            rl_signal['weight'] = self.rl_weight
            rl_signal['priority'] = 'high' if rl_signal['confidence'] > 0.7 else 'medium'
            
            return rl_signal
            
        except Exception as e:
            logger.error(f"❌ RL Signal Generation Fehler: {e}")
            return None
    
    async def _create_market_observation(self, market_data):
        """
        Market Observation für RL Agent erstellen
        """
        try:
            # Integration mit bestehenden TRADINO Komponenten
            observation = {
                'price_data': market_data,
                'indicators': await self._get_technical_indicators(market_data),
                'sentiment': await self._get_sentiment_data(market_data),
                'patterns': await self._get_pattern_data(market_data),
                'portfolio': await self._get_portfolio_state()
            }
            
            return observation
            
        except Exception as e:
            logger.error(f"❌ Market Observation Creation Fehler: {e}")
            return {}
    
    async def _get_technical_indicators(self, market_data):
        """
        Technical Indicators von Market Intelligence holen
        """
        try:
            if hasattr(self.trading_engine, 'market_intelligence'):
                indicators = await self.trading_engine.market_intelligence.get_technical_indicators(
                    market_data.get('symbol', 'BTC/USDT')
                )
                return indicators
            return {}
        except:
            return {}
    
    async def _get_sentiment_data(self, market_data):
        """
        Sentiment Data holen
        """
        try:
            if hasattr(self.trading_engine, 'sentiment_analyzer'):
                sentiment = await self.trading_engine.sentiment_analyzer.analyze_sentiment(
                    market_data.get('symbol', 'BTC/USDT')
                )
                return sentiment
            return {}
        except:
            return {}
    
    async def _get_pattern_data(self, market_data):
        """
        Pattern Recognition Data holen
        """
        try:
            if hasattr(self.trading_engine, 'pattern_recognition'):
                patterns = await self.trading_engine.pattern_recognition.detect_patterns(
                    market_data.get('symbol', 'BTC/USDT')
                )
                return patterns
            return {}
        except:
            return {}
    
    async def _get_portfolio_state(self):
        """
        Aktueller Portfolio State
        """
        try:
            if hasattr(self.trading_engine, 'portfolio_manager'):
                portfolio = await self.trading_engine.portfolio_manager.get_portfolio_summary()
                return portfolio
            return {}
        except:
            return {}
    
    async def continuous_learning_update(self, trade_result):
        """
        Kontinuierliches Lernen basierend auf Trade-Ergebnissen
        """
        try:
            if not self.rl_enabled:
                return
            
            # Trade Ergebnis in RL Environment Format konvertieren
            rl_experience = self._convert_trade_to_experience(trade_result)
            
            # Experience Buffer Update
            self.rl_agent.experience_buffer.append(rl_experience)
            
            # Online Learning (alle 10 Trades)
            if len(self.rl_agent.experience_buffer) % 10 == 0:
                await self._perform_online_learning()
            
        except Exception as e:
            logger.error(f"❌ Continuous Learning Update Fehler: {e}")
    
    def _convert_trade_to_experience(self, trade_result):
        """
        Trade Result in RL Experience Format konvertieren
        """
        return {
            'timestamp': trade_result.get('timestamp', datetime.now()),
            'action': trade_result.get('action', 'HOLD'),
            'pnl': trade_result.get('pnl', 0),
            'success': trade_result.get('pnl', 0) > 0,
            'portfolio_return': trade_result.get('portfolio_return', 0)
        }
    
    async def _perform_online_learning(self):
        """
        Online Learning mit aktuellen Experiences
        """
        try:
            if len(self.rl_agent.experience_buffer) < 20:
                return
            
            # Neueste Experiences für Online Update
            recent_experiences = list(self.rl_agent.experience_buffer)[-20:]
            
            # Mini-Training Session
            if hasattr(self.rl_agent.agent, 'train_on_batch'):
                self.rl_agent.agent.train_on_batch(recent_experiences)
                logger.debug("🧠 RL Online Learning Update durchgeführt")
            
        except Exception as e:
            logger.error(f"❌ Online Learning Fehler: {e}")
    
    def get_rl_status(self):
        """
        RL System Status
        """
        return {
            'enabled': self.rl_enabled,
            'agent_trained': self.rl_agent.training_metrics['episodes'] > 0,
            'training_status': self.rl_agent.get_training_status(),
            'experience_buffer_size': len(self.rl_agent.experience_buffer),
            'integration_weight': self.rl_weight
        } 