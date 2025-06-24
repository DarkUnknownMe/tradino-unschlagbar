# 🚀 TRADINO - Advanced AI Trading System

[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production_Ready-success)](README.md)

## 📋 ÜBERSICHT

TRADINO ist ein hochentwickeltes AI-basiertes Trading-System für Kryptowährungen mit automatischem Risikomanagement, Take-Profit/Stop-Loss Funktionalität und umfassendem Monitoring.

### 🎯 HAUPTFEATURES

- **🤖 AI-Powered Trading:** Maschinelles Lernen mit XGBoost, LightGBM und Random Forest
- **🛡️ Advanced Risk Management:** Automatische Positionsgrößenanpassung und Portfolio-Schutz
- **🎯 TP/SL System:** Automatische Take-Profit und Stop-Loss Orders mit OCO-Fallback
- **📱 Telegram Integration:** Vollständige Kontrolle und Monitoring über Telegram Bot
- **📊 Real-time Monitoring:** Live Performance Tracking und System Health Monitoring
- **🔄 Multi-Exchange Support:** Optimiert für Bitget, erweiterbar für andere Exchanges

### ⚡ QUICK START

```bash
# 1. Environment Setup
python -m venv tradino_env
source tradino_env/bin/activate  # Linux/Mac
# tradino_env\Scripts\activate  # Windows

# 2. Dependencies Installation
pip install -r requirements.txt

# 3. Configuration
cp .env.example .env
# Edit .env with your API credentials

# 4. System Validation
python scripts/system_validation.py

# 5. Start Trading
python main.py
```

## 🏗️ SYSTEM ARCHITEKTUR

```
tradino/
├── 📁 tradino_unschlagbar/     # Haupt-Trading System
│   ├── core/                  # Trading Engine & Risk Management
│   ├── brain/                 # AI/ML Komponenten
│   ├── connectors/            # Exchange APIs & Data Feeds
│   ├── strategies/            # Trading Strategien
│   ├── analytics/             # Performance & Reporting
│   └── utils/                 # Helper Functions
├── 📁 core/                   # Legacy Core Components
├── 📁 config/                 # Konfigurationsdateien
├── 📁 data/                   # Daten & Trained Models
├── 📁 scripts/                # Utility & Test Scripts
├── 📁 tests/                  # Test Suite
└── 📁 docs/                   # Dokumentation
```

## 🔧 INSTALLATION & SETUP

### Systemanforderungen
- Python 3.12+
- 4GB+ RAM (8GB+ empfohlen für AI-Training)
- 2GB+ freier Speicherplatz
- Stabile Internetverbindung

### 1. Abhängigkeiten Installation
```bash
pip install -r requirements.txt
```

### 2. API Konfiguration
Erstelle `.env` Datei mit:
```env
# Bitget Exchange API
BITGET_API_KEY=your_api_key
BITGET_SECRET_KEY=your_secret_key
BITGET_PASSPHRASE=your_passphrase

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Trading Configuration
TRADING_MODE=paper  # paper oder live
MAX_POSITION_SIZE=100
RISK_PERCENTAGE=2.0
```

### 3. System Validation
```bash
python scripts/system_validation.py
```

## 🚀 USAGE

### Hauptsystem Starten
```bash
python main.py
```

### Telegram Bot Commands
- `/status` - System Status
- `/balance` - Account Balance
- `/positions` - Aktive Positionen
- `/trades` - Trade History
- `/risk` - Risk Management Settings
- `/stop` - Emergency Stop

### Trading Modi
- **Paper Trading:** Sicheres Testen ohne echtes Geld
- **Live Trading:** Echter Handel mit API-Integration

## 🤖 AI SYSTEM

### Trainierte Modelle
- **XGBoost Trend Model:** Trend-Erkennung (74.8% Accuracy)
- **LightGBM Volatility Model:** Volatilitäts-Prediction (75.1% Accuracy)
- **Random Forest Risk Model:** Risk Assessment (78.2% Accuracy)

### AI Training Pipeline
```bash
python scripts/train_models.py
```

### Reinforcement Learning
```bash
python tradino_unschlagbar/brain/rl_trading_agent.py
```

## 🛡️ RISK MANAGEMENT

### Automatic Features
- **Position Sizing:** Dynamische Positionsgrößenanpassung
- **Portfolio Protection:** Maximaler Drawdown Schutz
- **Volatility Adjustment:** Automatic Risk-Parameter Anpassung
- **Emergency Stop:** Sofortiger Trade-Stopp bei kritischen Ereignissen

### Manual Controls
- **Maximum Position Size:** Konfigurierbare Limits
- **Risk Percentage:** Risiko pro Trade
- **Stop Loss Levels:** Automatische SL-Berechnung
- **Take Profit Targets:** Profit-Ziele basierend auf Volatilität

## 🎯 TP/SL SYSTEM

### Features
- **Automatic TP/SL:** Sofortige Order-Platzierung nach Market Orders
- **OCO Orders:** One-Cancels-Other Implementation
- **Fallback Monitoring:** Manual monitoring bei OCO-Fehlern
- **Dynamic Targets:** Volatilitäts-basierte TP/SL Levels

### Configuration
```python
TP_SL_CONFIG = {
    "take_profit_percentage": 2.0,  # 2% Gewinn-Ziel
    "stop_loss_percentage": 1.0,    # 1% Verlust-Limit
    "use_oco": True,               # OCO Orders bevorzugt
    "fallback_monitoring": True     # Manual Fallback
}
```

## 📊 MONITORING & ANALYTICS

### System Health Monitoring
- **API Connectivity:** Latenz und Verfügbarkeit
- **Account Status:** Balance und Position Updates
- **Performance Metrics:** Win-Rate, Profit/Loss, Drawdown
- **Error Tracking:** Automatische Fehlerprotokollierung

### Telegram Notifications
- Trade Executions
- TP/SL Hits
- Risk Violations
- System Errors
- Performance Updates

### Dashboard Features
- Real-time P&L
- Position Overview
- Risk Metrics
- Performance Charts

## 🧪 TESTING

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Tests
```bash
python scripts/test_integration.py
```

### System Validation
```bash
python scripts/system_validation_standalone.py
```

## 📈 PERFORMANCE

### Backtesting Results
- **Total Return:** Konfigurierbar basierend auf Strategie
- **Sharpe Ratio:** Optimiert für Risk-Adjusted Returns
- **Maximum Drawdown:** Kontrolliert durch Risk Management
- **Win Rate:** Model-abhängig (70-80% typisch)

### Live Performance Tracking
- Real-time P&L
- Daily/Weekly/Monthly Reports
- Performance vs. Benchmark
- Risk-Adjusted Metrics

## 🔒 SICHERHEIT

### API Security
- Encrypted credential storage
- Read-only API permissions where possible
- IP whitelist recommendations
- Secure environment variable handling

### Trading Security
- Position size limits
- Maximum daily loss limits
- Emergency stop mechanisms
- Automated risk controls

## 🛠️ ENTWICKLUNG

### Code Structure
```bash
# Core Trading Logic
tradino_unschlagbar/core/trading_engine.py

# AI Models
tradino_unschlagbar/brain/master_ai.py

# Risk Management
tradino_unschlagbar/core/risk_guardian.py

# Exchange Connectivity
core/bitget_trading_api.py
```

### Adding New Features
1. Erstelle Feature Branch
2. Implementiere Tests
3. Validiere mit Backtesting
4. Deploy zu Paper Trading
5. Monitor Performance

### Custom Strategies
```python
from tradino_unschlagbar.strategies import BaseStrategy

class CustomStrategy(BaseStrategy):
    def generate_signals(self, data):
        # Implementiere deine Logik
        return signals
```

## 📚 DOKUMENTATION

### API Reference
- [Bitget API Integration](docs/api_reference.md)
- [AI Model Documentation](docs/ai_models.md)
- [Risk Management Guide](docs/risk_management.md)

### Troubleshooting
- [Common Issues](docs/troubleshooting.md)
- [Error Codes](docs/error_codes.md)
- [FAQ](docs/faq.md)

## 🤝 CONTRIBUTING

1. Fork das Repository
2. Erstelle Feature Branch
3. Implementiere Changes mit Tests
4. Submit Pull Request

## 📄 LICENSE

MIT License - siehe LICENSE Datei für Details.

## 🆘 SUPPORT

- **GitHub Issues:** Bug Reports und Feature Requests
- **Telegram Support:** @tradino_support
- **Documentation:** docs/ Verzeichnis
- **Email:** support@tradino.dev

## 📈 ROADMAP

### v2.0 Features
- Multi-Exchange Support
- Advanced ML Models
- Portfolio Optimization
- Social Trading Features

### v2.1 Features
- Options Trading
- DeFi Integration
- Advanced Analytics Dashboard
- Mobile App

---

⚠️ **WICHTIGER HINWEIS:** Trading mit Kryptowährungen beinhaltet erhebliche Risiken. Verwende nur Kapital, das du dir leisten kannst zu verlieren. Teste alle Strategien ausführlich im Paper Trading Modus bevor du zu Live Trading wechselst.

🚀 **Built with ❤️ by the TRADINO Team** 