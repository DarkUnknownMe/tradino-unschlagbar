# 🛡️ ADVANCED RISK MANAGEMENT SYSTEM

**Institutionelles Risikomanagement-System für quantitative Trading Strategien**

Das Advanced Risk Management System bietet umfassende, institutionelle Risikoanalyse und -kontrolle für professionelle Trading-Operationen.

---

## 🌟 HAUPTFEATURES

### 📊 Value at Risk (VaR) Berechnung
- **Historisches VaR**: Empirische Verteilungsanalyse
- **Parametrisches VaR**: Normalverteilungsannahme
- **Monte Carlo VaR**: Simulationsbasierte Risikoberechnung
- **Multiple Konfidenzniveaus**: 95%, 99%, 99.9%

### 💸 Expected Shortfall (CVaR)
- **Conditional VaR**: Durchschnittlicher Verlust jenseits des VaR
- **Tail Risk Analyse**: Extreme Verlustszenarien
- **Risk-adjusted Metriken**: Erwartete Verluste bei Stress

### 🎯 Kelly Criterion Position Sizing
- **Optimale Positionsgrößen**: Mathematisch fundierte Allokation
- **Marktbedingungen-Anpassung**: Dynamische Größenanpassung
- **Volatilitäts-Adjustierung**: Risikoadjustierte Positionsgrößen
- **Portfolio-Level Constraints**: Absolute Limits und Beschränkungen

### 🔗 Korrelations-Monitoring
- **Rolling Correlation Matrix**: Dynamische Korrelationsanalyse
- **Diversifikations-Ratio**: Portfoliodiversifikation Messung
- **Korrelations-Breakdown Detection**: Warnung bei hohen Korrelationen
- **Asset Class Clustering**: Sektorale Risiko-Konzentration

### 🧪 Stress Testing & Scenario Analysis
- **Market Crash Scenarios**: Marktcrash-Simulationen
- **Flash Crash Testing**: Plötzliche Marktbewegungen
- **Regulatory Shock**: Regulatorische Risiken
- **Custom Scenarios**: Benutzerdefinierte Stress-Tests
- **Liquidity Impact**: Liquiditätskosten-Berücksichtigung

### 🚨 Real-time Risk Monitoring
- **Dynamic Risk Limits**: Echtzeitüberwachung von Risikolimits
- **Alert System**: Sofortige Benachrichtigungen bei Grenzüberschreitungen
- **Risk Score**: Gesamtrisikobewertung (0-100)
- **Automated Actions**: Empfohlene Risikomanagement-Aktionen

---

## 🏗️ ARCHITEKTUR

### Core Komponenten

```
risk/
├── advanced_risk_manager.py    # Haupt-Risikomanager
├── risk_models.py             # Datenmodelle & Strukturen
├── __init__.py                # Modul-Exports
└── README.md                  # Diese Dokumentation
```

### Klassen-Hierarchie

```python
AdvancedRiskManager           # 🛡️ Haupt-Risikomanagement-System
├── VaRCalculator            # 📊 Value at Risk Berechnung
├── CorrelationMonitor       # 🔗 Korrelations-Überwachung
└── RiskLimits              # ⚙️ Risikolimit-Konfiguration

# Datenmodelle
Position                     # 💰 Trading-Position
TradingSignal               # 📡 Handelssignal  
MarketConditions            # 🌍 Marktbedingungen
Scenario                    # 🎭 Stress-Test-Szenario
RiskMetrics                 # 📊 Risikokennzahlen
RiskAlert                   # 🚨 Risiko-Warnung
```

---

## 🚀 SCHNELLSTART

### Installation & Import

```python
from tradino_unschlagbar.risk import (
    AdvancedRiskManager, Position, TradingSignal, 
    MarketConditions, AssetClass, SignalType
)
```

### Basis-Setup

```python
# Risk Manager initialisieren
portfolio_size = 100000.0  # $100k Portfolio
max_drawdown = 0.15        # 15% max Drawdown

risk_manager = AdvancedRiskManager(portfolio_size, max_drawdown)

# Position hinzufügen
position = Position(
    symbol="BTCUSDT",
    asset_class=AssetClass.CRYPTO,
    size=0.5,               # 0.5 BTC
    entry_price=42000.0,
    current_price=45000.0,
    timestamp=datetime.now(),
    leverage=2.0
)

violations = risk_manager.add_position(position)
if violations:
    print(f"Risk Violations: {violations}")
```

### VaR Berechnung

```python
# Value at Risk berechnen
var_95 = risk_manager.calculate_var(risk_manager.current_positions, 0.95)
var_99 = risk_manager.calculate_var(risk_manager.current_positions, 0.99)

# Expected Shortfall
es_95 = risk_manager.calculate_expected_shortfall(
    risk_manager.current_positions, 0.95
)

print(f"VaR (95%): ${var_95:,.2f}")
print(f"VaR (99%): ${var_99:,.2f}") 
print(f"Expected Shortfall: ${es_95:,.2f}")
```

### Kelly Criterion Position Sizing

```python
# Trading Signal erstellen
signal = TradingSignal(
    symbol="ETHUSDT",
    signal_type=SignalType.BUY,
    confidence=0.75,           # 75% Confidence
    strength=0.8,
    timestamp=datetime.now(),
    expected_return=0.15,      # 15% erwarteter Return
    risk_score=0.08           # 8% Risikoscore
)

# Marktbedingungen definieren
market_conditions = MarketConditions(
    timestamp=datetime.now(),
    volatility=0.25,          # 25% Volatilität
    trend="bullish",
    momentum=0.6,
    liquidity=0.8,
    correlation_regime="medium",
    risk_appetite=0.7,
    market_stress=0.3
)

# Optimale Positionsgröße berechnen
optimal_size = risk_manager.optimal_position_size(signal, market_conditions)
print(f"Optimal Position Size: ${optimal_size:,.2f}")
```

### Stress Testing

```python
# Vordefinierte Stress-Szenarien
scenarios = [
    Scenario.create_market_crash(),    # Marktcrash (-40% BTC, -45% ETH)
    Scenario.create_flash_crash(),     # Flash Crash (-20% BTC, -25% ETH)
    Scenario.create_regulatory_shock() # Regulatorischer Schock (-25% BTC)
]

# Stress Tests durchführen
stress_results = risk_manager.stress_test_portfolio(scenarios)

for scenario_name, result in stress_results.items():
    loss = result['total_loss']
    loss_pct = result['loss_percentage']
    print(f"{scenario_name}: ${loss:,.2f} ({loss_pct:.2f}%)")
```

### Risk Monitoring

```python
# Risikometriken berechnen
metrics = risk_manager.calculate_risk_metrics()

print(f"Overall Risk Score: {metrics.overall_risk_score:.1f}/100")
print(f"Risk Level: {metrics.risk_level.value}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Current Drawdown: {metrics.current_drawdown*100:.2f}%")

# Risikolimits überwachen
alerts = risk_manager.monitor_risk_limits()

for alert in alerts:
    print(f"🚨 {alert.alert_type}: {alert.message}")
    print(f"🔧 Recommendation: {alert.recommended_action}")
```

---

## 📊 RISIKOMETRIKEN

### Portfolio-Level Metriken
- **VaR (95%, 99%)**: Value at Risk bei verschiedenen Konfidenzniveaus
- **Expected Shortfall**: Durchschnittlicher Verlust jenseits VaR
- **Maximum Drawdown**: Größter historischer Verlust
- **Current Drawdown**: Aktueller Drawdown vom Peak
- **Leverage Ratio**: Gesamthebel des Portfolios
- **Concentration Risk (HHI)**: Herfindahl-Hirschman Index

### Performance-Metriken
- **Sharpe Ratio**: Risikoadjustierte Rendite
- **Sortino Ratio**: Downside-risikoadjustierte Rendite
- **Calmar Ratio**: Rendite geteilt durch Maximum Drawdown
- **Information Ratio**: Aktive Rendite vs. Tracking Error
- **Treynor Ratio**: Rendite pro Einheit systematisches Risiko

### Korrelations-Metriken
- **Average Correlation**: Durchschnittliche Asset-Korrelation
- **Diversification Ratio**: Portfoliodiversifikation
- **Correlation Regime**: Niedrig/Mittel/Hoch Korrelationsumfeld

---

## ⚙️ KONFIGURATION

### Risk Limits

```python
from tradino_unschlagbar.risk import RiskLimits

# Custom Risk Limits
risk_limits = RiskLimits(
    max_portfolio_var=0.05,        # Max 5% daily VaR
    max_position_size=0.10,        # Max 10% per Position
    max_sector_concentration=0.25,  # Max 25% per Sektor
    max_leverage=3.0,              # Max 3x Leverage
    max_drawdown=0.15,             # Max 15% Drawdown
    max_correlation=0.8,           # Max 80% Korrelation
    max_stress_loss=0.20,          # Max 20% Stress-Verlust
    var_alert_threshold=0.8,       # Alert bei 80% VaR Limit
    drawdown_alert_threshold=0.8   # Alert bei 80% DD Limit
)

risk_manager.risk_limits = risk_limits
```

### Custom Stress Scenarios

```python
# Benutzerdefiniertes Szenario
custom_scenario = Scenario(
    name="DeFi Crash",
    description="DeFi sector collapse",
    probability=0.08,              # 8% Wahrscheinlichkeit
    impact_duration=timedelta(days=14),
    price_shocks={
        "UNIUSDT": -0.60,         # 60% Verlust
        "AAVEUSDT": -0.55,        # 55% Verlust
        "COMPUSDT": -0.50         # 50% Verlust
    },
    volatility_multiplier=4.0,
    correlation_shift=0.4,         # Erhöhte Korrelation
    liquidity_reduction=0.7        # 70% Liquiditätsverlust
)
```

---

## 📋 RISK REPORT

### Umfassender Risk Report

```python
# Vollständigen Risikobericht generieren
report = risk_manager.generate_risk_report()

# Report Struktur
{
    "executive_summary": {
        "overall_risk_level": "low|moderate|high|critical",
        "risk_score": 0-100,
        "portfolio_value": float,
        "var_1d_99": float,
        "max_stress_loss": float,
        "active_breaches": int
    },
    "portfolio_summary": {
        "total_positions": int,
        "total_exposure": float,
        "gross_leverage": float,
        "total_return_pct": float
    },
    "risk_metrics": {
        "value_at_risk": {...},
        "expected_shortfall": {...},
        "drawdown_metrics": {...},
        "performance_metrics": {...}
    },
    "stress_tests": {...},
    "risk_limits": {...},
    "active_alerts": [...],
    "positions": [...],
    "recommendations": [...]
}
```

---

## 🚨 ALERT SYSTEM

### Alert Types

- **VaR_BREACH**: VaR-Limit überschritten
- **DRAWDOWN_BREACH**: Drawdown-Limit überschritten  
- **LEVERAGE_BREACH**: Leverage-Limit überschritten
- **CONCENTRATION_RISK**: Hohe Portfolio-Konzentration
- **CORRELATION_BREAKDOWN**: Erhöhte Asset-Korrelation

### Alert Levels

- **LOW**: Informative Warnungen
- **MODERATE**: Aufmerksamkeit erforderlich
- **HIGH**: Sofortige Maßnahmen empfohlen
- **CRITICAL**: Notfall-Intervention erforderlich

---

## 🧪 TESTING

### Demo ausführen

```bash
cd tradino_unschlagbar
python scripts/demo_advanced_risk_manager.py
```

### Unit Tests

```bash
cd tradino_unschlagbar
python -m pytest tests/test_risk_management.py -v
```

---

## 📚 WISSENSCHAFTLICHE BASIS

### Implementierte Methodologien

1. **Kelly Criterion** (1956): Optimale Positionsgrößenbestimmung
2. **Value at Risk** (JP Morgan, 1994): Quantifizierung von Marktrisiken
3. **Expected Shortfall** (Artzner et al., 1999): Kohärente Risikomaße
4. **Herfindahl-Hirschman Index**: Konzentrationsmessung
5. **Markowitz Portfolio Theory**: Moderne Portfoliotheorie
6. **GARCH Models**: Volatilitätsmodellierung

### Regulatorische Compliance

- **Basel III**: Internationale Bankenregulierung
- **UCITS**: EU Investmentfonds-Richtlinien  
- **AIFMD**: Alternative Investment Fund Managers Directive
- **MiFID II**: Markets in Financial Instruments Directive

---

## 🔧 ERWEITERTE FEATURES

### Real-time Integration

```python
# Preise in Echtzeit aktualisieren
price_updates = {
    "BTCUSDT": 45500.0,
    "ETHUSDT": 3250.0
}

risk_manager.update_position_prices(price_updates)

# Portfolio-Historie aktualisieren
risk_manager.update_portfolio_history()
```

### Custom Risk Models

```python
# Eigene VaR-Methode implementieren
class CustomVaRCalculator(VaRCalculator):
    def custom_var(self, returns, confidence=0.95):
        # Ihre eigene VaR-Implementierung
        pass

# Integration in Risk Manager
risk_manager.var_calculator = CustomVaRCalculator()
```

---

## 🎯 PRODUCTION DEPLOYMENT

### Performance Optimierung

- **Numpy/Scipy**: Optimierte numerische Berechnungen
- **Caching**: Wiederverwendung von Korrelationsmatrizen
- **Parallel Processing**: Multi-threaded Stress-Tests
- **Memory Management**: Begrenzte Historie (2 Jahre)

### Monitoring & Logging

```python
import logging

# Risk Manager Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('risk_manager')

# Alle Risk Events werden geloggt
risk_manager.monitor_risk_limits()  # Logs alerts automatisch
```

### Integration in Trading Systems

```python
# Beispiel Integration
class TradingSystem:
    def __init__(self):
        self.risk_manager = AdvancedRiskManager(1000000.0, 0.15)
    
    def place_order(self, signal, market_conditions):
        # Optimale Größe berechnen
        size = self.risk_manager.optimal_position_size(signal, market_conditions)
        
        # Pre-trade Risk Check
        if self.risk_manager.monitor_risk_limits():
            print("🚨 Risk limits breached - order rejected")
            return False
            
        # Order ausführen
        return self.execute_order(signal.symbol, size)
```

---

## 📞 SUPPORT & CONTRIBUTION

### Issues & Bug Reports
- GitHub Issues für Bug-Reports
- Detaillierte Reproduktionsschritte angeben
- Log-Ausgaben beifügen

### Feature Requests
- Neue Risikometriken
- Zusätzliche Stress-Szenarien
- Integrations mit externen Systemen

### Code Contributions
- Fork → Branch → Pull Request Workflow
- Unit Tests für neue Features
- Dokumentation für neue APIs

---

**🛡️ TRADINO ADVANCED RISK MANAGEMENT SYSTEM**  
*Institutional-grade risk management for quantitative trading*

Version: 1.0.0 | License: MIT | Author: TRADINO Development Team 