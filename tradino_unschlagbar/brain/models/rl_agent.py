#!/usr/bin/env python3
"""
🤖 DEEP Q-NETWORK REINFORCEMENT LEARNING AGENT
Echter RL Agent für automatisierte Trading-Entscheidungen
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from collections import deque
from typing import Dict, List, Tuple, Any
import os

class DQNNetwork(nn.Module):
    """🧠 Deep Q-Network Architecture"""
    
    def __init__(self, input_size: int = 50, hidden_size: int = 256, output_size: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNTradingAgent:
    """🎯 DQN Trading Agent"""
    
    def __init__(self, state_size: int = 50, action_size: int = 3, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        
        # Neural Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, 256, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        print(f"🤖 RL Agent initialisiert - Device: {self.device}")
        
    def act(self, state, training=True):
        """🎯 Choose Action"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def save_model(self, filepath: str):
        """💾 Save Model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"✅ Modell gespeichert: {filepath}")
    
    def load_model(self, filepath: str):
        """📂 Load Model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            print(f"✅ Modell geladen: {filepath}")
            return True
        print(f"⚠️ Modell nicht gefunden: {filepath}")
        return False

if __name__ == "__main__":
    print("🎯 RL Agent Test")
    agent = DQNTradingAgent()
    
    # Test State
    test_state = np.random.random(50)
    action = agent.act(test_state, training=False)
    print(f"Test Action: {action}")
    
    # Test Model Save
    agent.save_model("tradino_unschlagbar/models/rl_agent_test.pth")
    print("✅ RL Agent Test erfolgreich!")
