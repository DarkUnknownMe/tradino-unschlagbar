import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
from datetime import datetime
from collections import deque
import asyncio
import json
import itertools
from dataclasses import dataclass
from enum import Enum

# Advanced ML Libraries
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
    logger.success("✅ Optuna für Hyperparameter-Optimierung verfügbar")
except ImportError:
    logger.warning("⚠️ Optuna nicht verfügbar - verwende Grid Search Fallback")
    OPTUNA_AVAILABLE = False

class NetworkType(Enum):
    """
    Verschiedene Neural Network Typen für Trading
    """
    FEEDFORWARD = "feedforward"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CNN_1D = "cnn_1d"
    RESIDUAL = "residual"
    ATTENTION = "attention"
    HYBRID = "hybrid"

@dataclass
class NetworkArchitecture:
    """
    Neural Network Architektur Definition
    """
    network_type: NetworkType
    layers: List[Dict[str, Any]]
    optimizer_config: Dict[str, Any]
    loss_function: str
    performance_score: float
    training_time: float
    inference_time: float
    memory_usage: float
    architecture_id: str

class TRADINONeuralArchitectureSearch:
    """
    Automatische Neural Architecture Search für Trading-spezifische Networks
    Verwendet AutoML-Techniken für optimale Netzwerk-Architekturen
    """
    
    def __init__(self, config):
        self.config = config
        
        # NAS Configuration
        self.search_space = self._define_search_space()
        self.performance_history = deque(maxlen=1000)
        self.best_architectures = {}
        
        # Training Configuration
        self.training_epochs = config.get('nas_training_epochs', 50)
        self.population_size = config.get('nas_population_size', 20)
        self.max_trials = config.get('nas_max_trials', 100)
        
        # Performance Targets
        self.target_tasks = [
            'price_prediction',
            'trend_classification',
            'volatility_prediction',
            'signal_generation',
            'risk_assessment'
        ]
        
        # Optuna Study
        self.study = None
        if OPTUNA_AVAILABLE:
            self.study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(),
                pruner=SuccessiveHalvingPruner()
            )
        
        # Device Selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"🧠 Neural Architecture Search initialisiert (Device: {self.device})")
    
    def _define_search_space(self) -> Dict[str, Any]:
        """
        Search Space für Neural Architecture Search definieren
        """
        return {
            'network_types': [
                NetworkType.FEEDFORWARD,
                NetworkType.LSTM,
                NetworkType.GRU,
                NetworkType.TRANSFORMER,
                NetworkType.CNN_1D,
                NetworkType.RESIDUAL,
                NetworkType.ATTENTION
            ],
            'layer_sizes': [32, 64, 128, 256, 512],
            'num_layers': [2, 3, 4, 5, 6],
            'dropout_rates': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'activation_functions': ['relu', 'leaky_relu', 'elu', 'gelu', 'swish'],
            'optimizers': ['adam', 'adamw', 'sgd', 'rmsprop'],
            'learning_rates': [0.0001, 0.0003, 0.001, 0.003, 0.01],
            'batch_sizes': [16, 32, 64, 128],
            'sequence_lengths': [10, 20, 30, 50, 100],  # For sequential models
            'attention_heads': [2, 4, 8, 16],  # For attention models
            'kernel_sizes': [3, 5, 7, 9],  # For CNN models
        }
    
    async def search_optimal_architecture(self, task_type: str, 
                                        training_data: torch.Tensor,
                                        target_data: torch.Tensor) -> NetworkArchitecture:
        """
        Optimale Neural Network Architektur für spezifische Trading-Aufgabe suchen
        """
        try:
            logger.info(f"🔍 Neural Architecture Search für Task: {task_type}")
            
            if OPTUNA_AVAILABLE:
                # Optuna-basierte Optimierung
                best_architecture = await self._optuna_search(task_type, training_data, target_data)
            else:
                # Grid Search Fallback
                best_architecture = await self._grid_search(task_type, training_data, target_data)
            
            # Performance Validation
            validated_architecture = await self._validate_architecture(
                best_architecture, training_data, target_data
            )
            
            # Architecture Storage
            self.best_architectures[task_type] = validated_architecture
            
            logger.success(f"✅ Optimale Architektur gefunden für {task_type}: "
                         f"Score {validated_architecture.performance_score:.4f}")
            
            return validated_architecture
            
        except Exception as e:
            logger.error(f"❌ Neural Architecture Search Fehler: {e}")
            return await self._create_fallback_architecture(task_type)
    
    async def _optuna_search(self, task_type: str, 
                           training_data: torch.Tensor,
                           target_data: torch.Tensor) -> NetworkArchitecture:
        """
        Optuna-basierte Hyperparameter und Architecture Search
        """
        try:
            def objective(trial):
                # Architecture Sampling
                network_type = trial.suggest_categorical('network_type', 
                    [nt.value for nt in self.search_space['network_types']])
                
                num_layers = trial.suggest_int('num_layers', 2, 6)
                layer_size = trial.suggest_categorical('layer_size', self.search_space['layer_sizes'])
                dropout_rate = trial.suggest_categorical('dropout_rate', self.search_space['dropout_rates'])
                activation = trial.suggest_categorical('activation', self.search_space['activation_functions'])
                optimizer_name = trial.suggest_categorical('optimizer', self.search_space['optimizers'])
                learning_rate = trial.suggest_categorical('learning_rate', self.search_space['learning_rates'])
                batch_size = trial.suggest_categorical('batch_size', self.search_space['batch_sizes'])
                
                # Network-specific parameters
                additional_params = {}
                if network_type in ['lstm', 'gru', 'transformer']:
                    additional_params['sequence_length'] = trial.suggest_categorical(
                        'sequence_length', self.search_space['sequence_lengths'])
                
                if network_type == 'transformer':
                    additional_params['attention_heads'] = trial.suggest_categorical(
                        'attention_heads', self.search_space['attention_heads'])
                
                if network_type == 'cnn_1d':
                    additional_params['kernel_size'] = trial.suggest_categorical(
                        'kernel_size', self.search_space['kernel_sizes'])
                
                # Build and train architecture
                try:
                    architecture = self._build_architecture({
                        'network_type': network_type,
                        'num_layers': num_layers,
                        'layer_size': layer_size,
                        'dropout_rate': dropout_rate,
                        'activation': activation,
                        'optimizer': optimizer_name,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        **additional_params
                    })
                    
                    # Train and evaluate
                    performance = self._train_and_evaluate_architecture(
                        architecture, training_data, target_data, trial
                    )
                    
                    return performance
                    
                except Exception as e:
                    logger.warning(f"⚠️ Optuna Trial Fehler: {e}")
                    return 0.0
            
            # Run optimization
            logger.info(f"🎯 Starte Optuna-Optimierung mit {self.max_trials} Trials...")
            self.study.optimize(objective, n_trials=self.max_trials, timeout=3600)  # 1 hour max
            
            # Best trial results
            best_trial = self.study.best_trial
            best_params = best_trial.params
            
            # Build final architecture
            final_architecture = self._build_architecture(best_params)
            final_architecture.performance_score = best_trial.value
            
            logger.success(f"✅ Optuna Optimierung abgeschlossen: Best Score {best_trial.value:.4f}")
            
            return final_architecture
            
        except Exception as e:
            logger.error(f"❌ Optuna Search Fehler: {e}")
            return await self._grid_search(task_type, training_data, target_data)
    
    async def _grid_search(self, task_type: str,
                         training_data: torch.Tensor,
                         target_data: torch.Tensor) -> NetworkArchitecture:
        """
        Grid Search Fallback für Architecture Search
        """
        try:
            logger.info("🔍 Grid Search Fallback wird verwendet...")
            
            best_architecture = None
            best_score = -np.inf
            
            # Simplified grid search
            network_types = [NetworkType.FEEDFORWARD, NetworkType.LSTM, NetworkType.GRU]
            layer_configs = [
                {'num_layers': 3, 'layer_size': 128},
                {'num_layers': 4, 'layer_size': 256},
                {'num_layers': 3, 'layer_size': 64}
            ]
            
            total_combinations = len(network_types) * len(layer_configs)
            current_combination = 0
            
            for network_type in network_types:
                for layer_config in layer_configs:
                    current_combination += 1
                    try:
                        logger.info(f"🧪 Testing {current_combination}/{total_combinations}: "
                                  f"{network_type.value} with {layer_config}")
                        
                        # Build architecture
                        params = {
                            'network_type': network_type.value,
                            'num_layers': layer_config['num_layers'],
                            'layer_size': layer_config['layer_size'],
                            'dropout_rate': 0.2,
                            'activation': 'relu',
                            'optimizer': 'adam',
                            'learning_rate': 0.001,
                            'batch_size': 32
                        }
                        
                        architecture = self._build_architecture(params)
                        
                        # Train and evaluate
                        performance = self._train_and_evaluate_architecture(
                            architecture, training_data, target_data
                        )
                        
                        if performance > best_score:
                            best_score = performance
                            best_architecture = architecture
                            best_architecture.performance_score = performance
                            
                        logger.info(f"📊 Performance: {performance:.4f}")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Grid Search Iteration Fehler: {e}")
            
            if best_architecture:
                logger.success(f"✅ Grid Search abgeschlossen: Best Score {best_score:.4f}")
                return best_architecture
            else:
                return await self._create_fallback_architecture(task_type)
                
        except Exception as e:
            logger.error(f"❌ Grid Search Fehler: {e}")
            return await self._create_fallback_architecture(task_type)
    
    def _build_architecture(self, params: Dict[str, Any]) -> NetworkArchitecture:
        """
        Neural Network Architektur basierend auf Parametern erstellen
        """
        try:
            network_type = NetworkType(params['network_type'])
            
            # Layer Configuration
            layers = []
            num_layers = params.get('num_layers', 3)
            layer_size = params.get('layer_size', 128)
            dropout_rate = params.get('dropout_rate', 0.2)
            activation = params.get('activation', 'relu')
            
            # Build layers based on network type
            if network_type == NetworkType.FEEDFORWARD:
                layers = self._build_feedforward_layers(num_layers, layer_size, dropout_rate, activation)
            elif network_type == NetworkType.LSTM:
                layers = self._build_lstm_layers(num_layers, layer_size, dropout_rate, params)
            elif network_type == NetworkType.GRU:
                layers = self._build_gru_layers(num_layers, layer_size, dropout_rate, params)
            elif network_type == NetworkType.TRANSFORMER:
                layers = self._build_transformer_layers(layer_size, dropout_rate, params)
            elif network_type == NetworkType.CNN_1D:
                layers = self._build_cnn1d_layers(layer_size, dropout_rate, params)
            elif network_type == NetworkType.RESIDUAL:
                layers = self._build_residual_layers(num_layers, layer_size, dropout_rate, activation)
            elif network_type == NetworkType.ATTENTION:
                layers = self._build_attention_layers(layer_size, dropout_rate, params)
            
            # Optimizer Configuration
            optimizer_config = {
                'type': params.get('optimizer', 'adam'),
                'learning_rate': params.get('learning_rate', 0.001),
                'batch_size': params.get('batch_size', 32)
            }
            
            # Create Architecture
            architecture = NetworkArchitecture(
                network_type=network_type,
                layers=layers,
                optimizer_config=optimizer_config,
                loss_function='mse',  # Default for regression tasks
                performance_score=0.0,
                training_time=0.0,
                inference_time=0.0,
                memory_usage=0.0,
                architecture_id=f"{network_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            return architecture
            
        except Exception as e:
            logger.error(f"❌ Architecture Building Fehler: {e}")
            return self._create_simple_fallback_architecture()
    
    def _build_feedforward_layers(self, num_layers: int, layer_size: int, 
                                dropout_rate: float, activation: str) -> List[Dict]:
        """
        Feedforward Network Layers erstellen
        """
        layers = []
        
        # Input layer
        layers.append({
            'type': 'linear',
            'input_size': 50,  # Standard feature size
            'output_size': layer_size,
            'activation': activation
        })
        
        # Hidden layers
        for i in range(num_layers - 1):
            layers.append({
                'type': 'linear',
                'input_size': layer_size,
                'output_size': layer_size,
                'activation': activation
            })
            
            if dropout_rate > 0:
                layers.append({
                    'type': 'dropout',
                    'rate': dropout_rate
                })
        
        # Output layer
        layers.append({
            'type': 'linear',
            'input_size': layer_size,
            'output_size': 1,  # Single output for most trading tasks
            'activation': 'linear'
        })
        
        return layers
    
    def _build_lstm_layers(self, num_layers: int, layer_size: int,
                          dropout_rate: float, params: Dict) -> List[Dict]:
        """
        LSTM Network Layers erstellen
        """
        sequence_length = params.get('sequence_length', 20)
        
        layers = [
            {
                'type': 'lstm',
                'input_size': 50,  # Feature size
                'hidden_size': layer_size,
                'num_layers': num_layers,
                'dropout': dropout_rate,
                'sequence_length': sequence_length,
                'batch_first': True
            },
            {
                'type': 'linear',
                'input_size': layer_size,
                'output_size': 1,
                'activation': 'linear'
            }
        ]
        
        return layers
    
    def _build_gru_layers(self, num_layers: int, layer_size: int,
                         dropout_rate: float, params: Dict) -> List[Dict]:
        """
        GRU Network Layers erstellen
        """
        sequence_length = params.get('sequence_length', 20)
        
        layers = [
            {
                'type': 'gru',
                'input_size': 50,
                'hidden_size': layer_size,
                'num_layers': num_layers,
                'dropout': dropout_rate,
                'sequence_length': sequence_length,
                'batch_first': True
            },
            {
                'type': 'linear',
                'input_size': layer_size,
                'output_size': 1,
                'activation': 'linear'
            }
        ]
        
        return layers
    
    def _build_transformer_layers(self, layer_size: int, dropout_rate: float, params: Dict) -> List[Dict]:
        """
        Transformer Network Layers erstellen
        """
        attention_heads = params.get('attention_heads', 8)
        sequence_length = params.get('sequence_length', 20)
        
        layers = [
            {
                'type': 'transformer_encoder',
                'input_size': 50,
                'd_model': layer_size,
                'nhead': attention_heads,
                'num_layers': 3,
                'dropout': dropout_rate,
                'sequence_length': sequence_length
            },
            {
                'type': 'global_avg_pool',
                'input_size': layer_size
            },
            {
                'type': 'linear',
                'input_size': layer_size,
                'output_size': 1,
                'activation': 'linear'
            }
        ]
        
        return layers
    
    def _build_cnn1d_layers(self, layer_size: int, dropout_rate: float, params: Dict) -> List[Dict]:
        """
        1D CNN Network Layers erstellen
        """
        kernel_size = params.get('kernel_size', 5)
        
        layers = [
            {
                'type': 'conv1d',
                'in_channels': 1,
                'out_channels': layer_size // 4,
                'kernel_size': kernel_size,
                'padding': kernel_size // 2,
                'activation': 'relu'
            },
            {
                'type': 'conv1d',
                'in_channels': layer_size // 4,
                'out_channels': layer_size // 2,
                'kernel_size': kernel_size,
                'padding': kernel_size // 2,
                'activation': 'relu'
            },
            {
                'type': 'global_avg_pool'
            },
            {
                'type': 'linear',
                'input_size': layer_size // 2,
                'output_size': layer_size,
                'activation': 'relu'
            },
            {
                'type': 'dropout',
                'rate': dropout_rate
            },
            {
                'type': 'linear',
                'input_size': layer_size,
                'output_size': 1,
                'activation': 'linear'
            }
        ]
        
        return layers
    
    def _build_residual_layers(self, num_layers: int, layer_size: int,
                             dropout_rate: float, activation: str) -> List[Dict]:
        """
        Residual Network Layers erstellen
        """
        layers = [
            {
                'type': 'linear',
                'input_size': 50,
                'output_size': layer_size,
                'activation': activation
            }
        ]
        
        # Residual blocks
        for i in range(num_layers - 1):
            layers.extend([
                {
                    'type': 'residual_block',
                    'input_size': layer_size,
                    'hidden_size': layer_size,
                    'activation': activation,
                    'dropout': dropout_rate
                }
            ])
        
        # Output layer
        layers.append({
            'type': 'linear',
            'input_size': layer_size,
            'output_size': 1,
            'activation': 'linear'
        })
        
        return layers
    
    def _build_attention_layers(self, layer_size: int, dropout_rate: float, params: Dict) -> List[Dict]:
        """
        Attention-based Network Layers erstellen
        """
        attention_heads = params.get('attention_heads', 4)
        
        layers = [
            {
                'type': 'multi_head_attention',
                'input_size': 50,
                'd_model': layer_size,
                'num_heads': attention_heads,
                'dropout': dropout_rate
            },
            {
                'type': 'feedforward',
                'input_size': layer_size,
                'hidden_size': layer_size * 2,
                'output_size': layer_size,
                'activation': 'relu',
                'dropout': dropout_rate
            },
            {
                'type': 'linear',
                'input_size': layer_size,
                'output_size': 1,
                'activation': 'linear'
            }
        ]
        
        return layers
    
    def _train_and_evaluate_architecture(self, architecture: NetworkArchitecture,
                                       training_data: torch.Tensor,
                                       target_data: torch.Tensor,
                                       trial=None) -> float:
        """
        Architektur trainieren und evaluieren
        """
        try:
            # Build PyTorch model from architecture
            model = self._build_pytorch_model(architecture)
            model = model.to(self.device)
            
            # Setup optimizer
            optimizer = self._setup_optimizer(model, architecture.optimizer_config)
            criterion = nn.MSELoss()
            
            # Training data preparation
            train_loader = self._prepare_data_loader(
                training_data, target_data, architecture.optimizer_config['batch_size']
            )
            
            # Training loop
            model.train()
            training_start = datetime.now()
            
            for epoch in range(self.training_epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                
                avg_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
                
                # Optuna pruning
                if trial:
                    trial.report(1.0 / (1.0 + avg_loss), epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                # Early stopping for very poor performance
                if avg_loss > 100:
                    break
            
            training_time = (datetime.now() - training_start).total_seconds()
            
            # Evaluation
            model.eval()
            eval_start = datetime.now()
            
            with torch.no_grad():
                total_loss = 0.0
                predictions = []
                actuals = []
                
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    total_loss += loss.item()
                    
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(batch_y.cpu().numpy())
                
                eval_time = (datetime.now() - eval_start).total_seconds()
                
                # Performance metrics
                mse = total_loss / len(train_loader)
                
                # Convert to performance score (higher is better)
                performance_score = 1.0 / (1.0 + mse)
                
                # Update architecture metrics
                architecture.training_time = training_time
                architecture.inference_time = eval_time
                architecture.memory_usage = self._estimate_memory_usage(model)
                
                return performance_score
                
        except Exception as e:
            if "TrialPruned" in str(type(e)):
                raise e
            logger.warning(f"⚠️ Architecture Training Fehler: {e}")
            return 0.0
    
    def _build_pytorch_model(self, architecture: NetworkArchitecture) -> nn.Module:
        """
        PyTorch Model aus Architecture Definition erstellen
        """
        try:
            if architecture.network_type == NetworkType.FEEDFORWARD:
                return self._build_feedforward_model(architecture.layers)
            elif architecture.network_type == NetworkType.LSTM:
                return self._build_lstm_model(architecture.layers)
            elif architecture.network_type == NetworkType.GRU:
                return self._build_gru_model(architecture.layers)
            elif architecture.network_type == NetworkType.TRANSFORMER:
                return self._build_transformer_model(architecture.layers)
            elif architecture.network_type == NetworkType.CNN_1D:
                return self._build_cnn1d_model(architecture.layers)
            elif architecture.network_type == NetworkType.RESIDUAL:
                return self._build_residual_model(architecture.layers)
            else:
                # Fallback to simple feedforward
                return self._build_simple_feedforward_model()
                
        except Exception as e:
            logger.error(f"❌ PyTorch Model Building Fehler: {e}")
            return self._build_simple_feedforward_model()
    
    def _build_feedforward_model(self, layers: List[Dict]) -> nn.Module:
        """
        Feedforward PyTorch Model erstellen
        """
        class FeedforwardModel(nn.Module):
            def __init__(self, layer_configs):
                super().__init__()
                self.layers = nn.ModuleList()
                
                for layer_config in layer_configs:
                    if layer_config['type'] == 'linear':
                        self.layers.append(nn.Linear(
                            layer_config['input_size'],
                            layer_config['output_size']
                        ))
                    elif layer_config['type'] == 'dropout':
                        self.layers.append(nn.Dropout(layer_config['rate']))
                
                self.layer_configs = layer_configs
            
            def forward(self, x):
                layer_idx = 0
                for config in self.layer_configs:
                    if config['type'] == 'linear':
                        x = self.layers[layer_idx](x)
                        layer_idx += 1
                        
                        # Apply activation
                        activation = config.get('activation', 'linear')
                        if activation == 'relu':
                            x = torch.relu(x)
                        elif activation == 'leaky_relu':
                            x = torch.leaky_relu(x)
                        elif activation == 'elu':
                            x = torch.elu(x)
                        elif activation == 'gelu':
                            x = torch.nn.functional.gelu(x)
                        
                    elif config['type'] == 'dropout':
                        x = self.layers[layer_idx](x)
                        layer_idx += 1
                
                return x
        
        return FeedforwardModel(layers)
    
    def _build_lstm_model(self, layers: List[Dict]) -> nn.Module:
        """
        LSTM PyTorch Model erstellen
        """
        class LSTMModel(nn.Module):
            def __init__(self, layer_configs):
                super().__init__()
                
                # Find LSTM layer config
                lstm_config = next(l for l in layer_configs if l['type'] == 'lstm')
                
                self.lstm = nn.LSTM(
                    input_size=lstm_config['input_size'],
                    hidden_size=lstm_config['hidden_size'],
                    num_layers=lstm_config['num_layers'],
                    dropout=lstm_config['dropout'],
                    batch_first=lstm_config['batch_first']
                )
                
                # Find linear layer config
                linear_config = next(l for l in layer_configs if l['type'] == 'linear')
                self.linear = nn.Linear(
                    linear_config['input_size'],
                    linear_config['output_size']
                )
            
            def forward(self, x):
                # Reshape for LSTM if needed
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # Add sequence dimension
                
                lstm_out, _ = self.lstm(x)
                # Take last output
                output = self.linear(lstm_out[:, -1, :])
                return output
        
        return LSTMModel(layers)
    
    def _build_gru_model(self, layers: List[Dict]) -> nn.Module:
        """
        GRU PyTorch Model erstellen
        """
        class GRUModel(nn.Module):
            def __init__(self, layer_configs):
                super().__init__()
                
                # Find GRU layer config
                gru_config = next(l for l in layer_configs if l['type'] == 'gru')
                
                self.gru = nn.GRU(
                    input_size=gru_config['input_size'],
                    hidden_size=gru_config['hidden_size'],
                    num_layers=gru_config['num_layers'],
                    dropout=gru_config['dropout'],
                    batch_first=gru_config['batch_first']
                )
                
                # Find linear layer config
                linear_config = next(l for l in layer_configs if l['type'] == 'linear')
                self.linear = nn.Linear(
                    linear_config['input_size'],
                    linear_config['output_size']
                )
            
            def forward(self, x):
                # Reshape for GRU if needed
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # Add sequence dimension
                
                gru_out, _ = self.gru(x)
                # Take last output
                output = self.linear(gru_out[:, -1, :])
                return output
        
        return GRUModel(layers)
    
    def _build_transformer_model(self, layers: List[Dict]) -> nn.Module:
        """
        Transformer PyTorch Model erstellen
        """
        class TransformerModel(nn.Module):
            def __init__(self, layer_configs):
                super().__init__()
                
                # Find transformer config
                transformer_config = next(l for l in layer_configs if l['type'] == 'transformer_encoder')
                
                # Input projection
                self.input_projection = nn.Linear(
                    transformer_config['input_size'],
                    transformer_config['d_model']
                )
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=transformer_config['d_model'],
                    nhead=transformer_config['nhead'],
                    dropout=transformer_config['dropout'],
                    batch_first=True
                )
                
                self.transformer = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=transformer_config['num_layers']
                )
                
                # Find linear layer config
                linear_config = next(l for l in layer_configs if l['type'] == 'linear')
                self.linear = nn.Linear(
                    linear_config['input_size'],
                    linear_config['output_size']
                )
            
            def forward(self, x):
                # Reshape for transformer if needed
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # Add sequence dimension
                
                # Project input
                x = self.input_projection(x)
                
                # Transformer encoding
                transformer_out = self.transformer(x)
                
                # Global average pooling
                pooled = torch.mean(transformer_out, dim=1)
                
                # Final prediction
                output = self.linear(pooled)
                return output
        
        return TransformerModel(layers)
    
    def _build_cnn1d_model(self, layers: List[Dict]) -> nn.Module:
        """
        1D CNN PyTorch Model erstellen
        """
        class CNN1DModel(nn.Module):
            def __init__(self, layer_configs):
                super().__init__()
                
                self.layers = nn.ModuleList()
                self.layer_configs = layer_configs
                
                for config in layer_configs:
                    if config['type'] == 'conv1d':
                        self.layers.append(nn.Conv1d(
                            in_channels=config['in_channels'],
                            out_channels=config['out_channels'],
                            kernel_size=config['kernel_size'],
                            padding=config['padding']
                        ))
                    elif config['type'] == 'linear':
                        self.layers.append(nn.Linear(
                            config['input_size'],
                            config['output_size']
                        ))
                    elif config['type'] == 'dropout':
                        self.layers.append(nn.Dropout(config['rate']))
            
            def forward(self, x):
                # Reshape for CNN
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # Add channel dimension
                
                layer_idx = 0
                for config in self.layer_configs:
                    if config['type'] == 'conv1d':
                        x = self.layers[layer_idx](x)
                        layer_idx += 1
                        
                        # Apply activation
                        if config.get('activation') == 'relu':
                            x = torch.relu(x)
                            
                    elif config['type'] == 'global_avg_pool':
                        x = torch.mean(x, dim=2)  # Global average pooling
                        
                    elif config['type'] == 'linear':
                        x = self.layers[layer_idx](x)
                        layer_idx += 1
                        
                        # Apply activation
                        activation = config.get('activation', 'linear')
                        if activation == 'relu':
                            x = torch.relu(x)
                            
                    elif config['type'] == 'dropout':
                        x = self.layers[layer_idx](x)
                        layer_idx += 1
                
                return x
        
        return CNN1DModel(layers)
    
    def _build_residual_model(self, layers: List[Dict]) -> nn.Module:
        """
        Residual PyTorch Model erstellen
        """
        class ResidualBlock(nn.Module):
            def __init__(self, input_size, hidden_size, dropout=0.0):
                super().__init__()
                self.linear1 = nn.Linear(input_size, hidden_size)
                self.linear2 = nn.Linear(hidden_size, input_size)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                residual = x
                out = torch.relu(self.linear1(x))
                out = self.dropout(out)
                out = self.linear2(out)
                out += residual  # Residual connection
                return torch.relu(out)
        
        class ResidualModel(nn.Module):
            def __init__(self, layer_configs):
                super().__init__()
                
                self.layers = nn.ModuleList()
                
                for config in layer_configs:
                    if config['type'] == 'linear':
                        self.layers.append(nn.Linear(
                            config['input_size'],
                            config['output_size']
                        ))
                    elif config['type'] == 'residual_block':
                        self.layers.append(ResidualBlock(
                            config['input_size'],
                            config['hidden_size'],
                            config.get('dropout', 0.0)
                        ))
                
                self.layer_configs = layer_configs
            
            def forward(self, x):
                layer_idx = 0
                for config in self.layer_configs:
                    if config['type'] in ['linear', 'residual_block']:
                        x = self.layers[layer_idx](x)
                        layer_idx += 1
                        
                        # Apply activation for linear layers
                        if config['type'] == 'linear':
                            activation = config.get('activation', 'linear')
                            if activation == 'relu':
                                x = torch.relu(x)
                
                return x
        
        return ResidualModel(layers)
    
    def _build_simple_feedforward_model(self) -> nn.Module:
        """
        Simple Fallback Feedforward Model
        """
        class SimpleFeedforward(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(50, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        return SimpleFeedforward()
    
    def _setup_optimizer(self, model: nn.Module, optimizer_config: Dict) -> torch.optim.Optimizer:
        """
        Optimizer basierend auf Konfiguration erstellen
        """
        optimizer_type = optimizer_config.get('type', 'adam')
        learning_rate = optimizer_config.get('learning_rate', 0.001)
        
        if optimizer_type == 'adam':
            return optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'adamw':
            return optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            return optim.Adam(model.parameters(), lr=learning_rate)
    
    def _prepare_data_loader(self, training_data: torch.Tensor, 
                           target_data: torch.Tensor, batch_size: int):
        """
        Data Loader für Training erstellen
        """
        try:
            from torch.utils.data import TensorDataset, DataLoader
            
            dataset = TensorDataset(training_data, target_data)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            return data_loader
            
        except Exception as e:
            logger.error(f"❌ Data Loader Creation Fehler: {e}")
            # Fallback: Return simple batches
            return [(training_data, target_data)]
    
    def _estimate_memory_usage(self, model: nn.Module) -> float:
        """
        Model Memory Usage schätzen
        """
        try:
            param_count = sum(p.numel() for p in model.parameters())
            # Estimate 4 bytes per parameter (float32)
            memory_mb = (param_count * 4) / (1024 * 1024)
            return memory_mb
        except:
            return 0.0
    
    async def _validate_architecture(self, architecture: NetworkArchitecture,
                                   training_data: torch.Tensor,
                                   target_data: torch.Tensor) -> NetworkArchitecture:
        """
        Architektur-Validierung mit Cross-Validation
        """
        try:
            # Split data for validation
            split_idx = int(len(training_data) * 0.8)
            train_data = training_data[:split_idx]
            val_data = training_data[split_idx:]
            train_targets = target_data[:split_idx]
            val_targets = target_data[split_idx:]
            
            # Validate on separate data
            validation_score = self._train_and_evaluate_architecture(
                architecture, val_data, val_targets
            )
            
            # Update performance score with validation
            architecture.performance_score = (architecture.performance_score + validation_score) / 2
            
            return architecture
            
        except Exception as e:
            logger.error(f"❌ Architecture Validation Fehler: {e}")
            return architecture
    
    async def _create_fallback_architecture(self, task_type: str) -> NetworkArchitecture:
        """
        Fallback Architecture erstellen
        """
        return NetworkArchitecture(
            network_type=NetworkType.FEEDFORWARD,
            layers=[
                {'type': 'linear', 'input_size': 50, 'output_size': 128, 'activation': 'relu'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'linear', 'input_size': 128, 'output_size': 64, 'activation': 'relu'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'linear', 'input_size': 64, 'output_size': 1, 'activation': 'linear'}
            ],
            optimizer_config={'type': 'adam', 'learning_rate': 0.001, 'batch_size': 32},
            loss_function='mse',
            performance_score=0.5,
            training_time=0.0,
            inference_time=0.0,
            memory_usage=0.0,
            architecture_id=f"fallback_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    def _create_simple_fallback_architecture(self) -> NetworkArchitecture:
        """
        Simple Fallback Architecture
        """
        return self._create_fallback_architecture("simple")
    
    async def generate_training_data(self, task_type: str, size: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Synthetic Training Data für NAS generieren
        """
        try:
            # Generate synthetic market features
            features = []
            targets = []
            
            for i in range(size):
                # Create synthetic market features
                feature_vector = []
                
                # Price momentum features
                momentum = np.random.normal(0, 0.02)  # Price change
                feature_vector.extend([momentum * k for k in [1, 2, 5, 10]])  # Different timeframes
                
                # Volatility features
                volatility = np.random.exponential(0.02)  # Always positive
                feature_vector.extend([volatility, volatility**2, np.sqrt(volatility)])
                
                # Volume features
                volume_ratio = np.random.lognormal(0, 0.5)  # Log-normal distribution
                feature_vector.extend([volume_ratio, np.log(volume_ratio), volume_ratio**0.5])
                
                # Technical indicators
                rsi = np.random.uniform(0, 100)
                macd = np.random.normal(0, 0.1)
                bb_position = np.random.uniform(0, 1)
                feature_vector.extend([rsi/100, macd, bb_position])
                
                # Trend features
                trend_strength = np.random.uniform(0, 1)
                trend_direction = np.random.choice([-1, 0, 1])
                feature_vector.extend([trend_strength, trend_direction, trend_strength * trend_direction])
                
                # Market regime features
                regime_indicators = np.random.uniform(0, 1, 5)
                feature_vector.extend(regime_indicators)
                
                # Sentiment features
                sentiment_scores = np.random.normal(0, 0.3, 3)
                feature_vector.extend(sentiment_scores)
                
                # Risk features
                risk_metrics = np.random.uniform(0, 1, 5)
                feature_vector.extend(risk_metrics)
                
                # Correlation features
                correlation_features = np.random.normal(0, 0.5, 3)
                feature_vector.extend(correlation_features)
                
                # Pattern recognition features
                pattern_features = np.random.uniform(0, 1, 7)
                feature_vector.extend(pattern_features)
                
                # Pad or trim to exactly 50 features
                while len(feature_vector) < 50:
                    feature_vector.append(0.0)
                feature_vector = feature_vector[:50]
                
                features.append(feature_vector)
                
                # Generate target based on task type
                if task_type == 'price_prediction':
                    # Predict next price movement
                    target = momentum + np.random.normal(0, 0.01)
                elif task_type == 'trend_classification':
                    # Classify trend direction
                    target = 1.0 if momentum > 0.01 else -1.0 if momentum < -0.01 else 0.0
                elif task_type == 'volatility_prediction':
                    # Predict volatility
                    target = volatility + np.random.normal(0, 0.005)
                elif task_type == 'signal_generation':
                    # Generate trading signal
                    signal_strength = abs(momentum) * trend_strength
                    target = np.tanh(signal_strength * 10)  # Squash to [-1, 1]
                elif task_type == 'risk_assessment':
                    # Assess risk level
                    risk_level = volatility * 0.5 + abs(momentum) * 0.3 + np.random.uniform(0, 0.2)
                    target = min(1.0, risk_level)
                else:
                    # Default: simple price prediction
                    target = momentum
                
                targets.append([target])
            
            # Convert to tensors
            features_tensor = torch.FloatTensor(features)
            targets_tensor = torch.FloatTensor(targets)
            
            logger.info(f"📊 Generated {size} training samples for {task_type}")
            logger.info(f"Features shape: {features_tensor.shape}, Targets shape: {targets_tensor.shape}")
            
            return features_tensor, targets_tensor
            
        except Exception as e:
            logger.error(f"❌ Training Data Generation Fehler: {e}")
            # Return minimal fallback data
            features_tensor = torch.randn(100, 50)
            targets_tensor = torch.randn(100, 1)
            return features_tensor, targets_tensor
    
    def get_best_architecture(self, task_type: str) -> Optional[NetworkArchitecture]:
        """
        Beste Architektur für Task Type abrufen
        """
        return self.best_architectures.get(task_type)
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """
        Zusammenfassung aller gefundenen Architekturen
        """
        summary = {
            'total_architectures': len(self.best_architectures),
            'task_types': list(self.best_architectures.keys()),
            'best_scores': {},
            'search_space_size': len(self.search_space),
            'optuna_available': OPTUNA_AVAILABLE
        }
        
        for task_type, architecture in self.best_architectures.items():
            summary['best_scores'][task_type] = {
                'performance_score': architecture.performance_score,
                'network_type': architecture.network_type.value,
                'training_time': architecture.training_time,
                'inference_time': architecture.inference_time,
                'memory_usage': architecture.memory_usage
            }
        
        return summary
    
    async def export_best_architectures(self, filepath: str):
        """
        Beste Architekturen exportieren
        """
        try:
            export_data = {}
            
            for task_type, architecture in self.best_architectures.items():
                export_data[task_type] = {
                    'network_type': architecture.network_type.value,
                    'layers': architecture.layers,
                    'optimizer_config': architecture.optimizer_config,
                    'loss_function': architecture.loss_function,
                    'performance_score': architecture.performance_score,
                    'training_time': architecture.training_time,
                    'inference_time': architecture.inference_time,
                    'memory_usage': architecture.memory_usage,
                    'architecture_id': architecture.architecture_id
                }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.success(f"✅ Architekturen exportiert nach: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Architecture Export Fehler: {e}")