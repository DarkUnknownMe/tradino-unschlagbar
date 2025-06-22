# 🐺 WOT ALPHA - LIVE LAUNCH PROTOCOL

## 🚨 SCHRITT-FÜR-SCHRITT ANLEITUNG

### **PHASE 1: VORBEREITUNG** ⚙️

#### 1.1 **ECHTE API KEYS ERSTELLEN**
```bash
# Bei Bitget einloggen:
# 1. Gehe zu: API Management
# 2. Erstelle neue API Keys
# 3. WICHTIG: Nur "Spot Trading" + "Futures Trading" aktivieren
# 4. NIEMALS "Withdraw" aktivieren!
# 5. IP Whitelist aktivieren (deine Server IP)
# 6. 2FA aktivieren
```

#### 1.2 **LIVE KONFIGURATION**
```bash
# .env Datei für LIVE anpassen:
cp .env .env.demo.backup  # Backup erstellen

# Neue .env für LIVE:
BITGET_API_KEY=DEINE_ECHTE_API_KEY
BITGET_SECRET_KEY=DEIN_ECHTER_SECRET
BITGET_PASSPHRASE=DEINE_ECHTE_PASSPHRASE
BITGET_SANDBOX=false

ENVIRONMENT=live
INITIAL_CAPITAL=500.0  # START KLEIN!
MAX_RISK_PER_TRADE=0.02  # 2% für Start
MAX_DAILY_DRAWDOWN=0.03  # 3% für Start
```

### **PHASE 2: SOFT LAUNCH** 🧪

#### 2.1 **MINIMALER START**
```bash
# Erste 48 Stunden:
INITIAL_CAPITAL=500.0     # Nur $500
MAX_RISK_PER_TRADE=0.02   # 2% = max $10/Trade
MAX_DAILY_DRAWDOWN=0.03   # 3% = max $15/Tag
```

#### 2.2 **INTENSIVE ÜBERWACHUNG**
```bash
# System starten:
python3 system_launcher.py --mode=live --capital=500

# Monitoring (separates Terminal):
tail -f logs/tradino_*.log

# Telegram Überwachung aktiviert
```

### **PHASE 3: SCALING** 📈

#### 3.1 **NACH 1 WOCHE ERFOLG**
```bash
# Kapital erhöhen (nur bei Profit):
INITIAL_CAPITAL=1000.0
MAX_RISK_PER_TRADE=0.025  # 2.5%
MAX_DAILY_DRAWDOWN=0.04   # 4%
```

#### 3.2 **NACH 1 MONAT ERFOLG**
```bash
# Standard Settings:
INITIAL_CAPITAL=2000.0
MAX_RISK_PER_TRADE=0.03   # 3%
MAX_DAILY_DRAWDOWN=0.05   # 5%
```

---

## ⚠️ **KRITISCHE SICHERHEITS-CHECKS**

### 🛡️ **VOR JEDEM START PRÜFEN:**
- [ ] **Nur Risikokapital** verwenden
- [ ] **API Keys** haben KEINE Withdrawal-Rechte
- [ ] **Stop-Loss** Mechanismen funktionieren
- [ ] **Emergency Stop** Button bereit
- [ ] **Internet Verbindung** stabil
- [ ] **Backup Plan** vorhanden

### 🚨 **NOTFALL-PROTOKOLL:**
```bash
# EMERGENCY STOP (sofort alle Positionen schließen):
python3 -c "
from core.trading_engine import TradingEngine
from utils.config_manager import ConfigManager
import asyncio

async def emergency_stop():
    config = ConfigManager()
    engine = TradingEngine(config)
    await engine.emergency_shutdown()

asyncio.run(emergency_stop())
"
```

---

## 📊 **MONITORING BEFEHLE**

### **System Status:**
```bash
# Performance Check:
python3 -c "
from analytics.performance_tracker import PerformanceTracker
tracker = PerformanceTracker()
print(tracker.get_daily_summary())
"

# Portfolio Status:
python3 -c "
from core.portfolio_manager import PortfolioManager
pm = PortfolioManager()
print(pm.get_portfolio_summary())
"
```

### **Telegram Commands:**
```
/status     - System Status
/portfolio  - Portfolio Übersicht
/performance - Performance Metrics
/positions  - Aktuelle Positionen
/stop       - EMERGENCY STOP
```

---

## 🎯 **SUCCESS METRICS**

### **WOCHE 1 ZIELE:**
- ✅ **Uptime > 95%**
- ✅ **Max Drawdown < 5%**
- ✅ **Win Rate > 55%**
- ✅ **Keine kritischen Errors**

### **MONAT 1 ZIELE:**
- ✅ **Uptime > 99%**
- ✅ **Profit > 10%**
- ✅ **Win Rate > 60%**
- ✅ **Sharpe Ratio > 1.0**

---

## 🐺 **ALPHA'S FINAL WORDS**

**"Hunt Smart, Not Hard!"**

1. **Start Klein** - Lerne das System kennen
2. **Monitor Intensiv** - Erste Wochen 24/7 überwachen
3. **Scale Vorsichtig** - Nur bei bewiesenem Erfolg
4. **Risk Management** - Niemals das gesamte Kapital riskieren
5. **Emergency Ready** - Immer bereit zu stoppen

**Der Alpha schützt sein Rudel. Sicherheit geht vor Profit!** 🛡️

---

*Erstellt von: WOT Alpha System*
*Datum: $(date)*
