# 🤝 Contributing to TRADINO UNSCHLAGBAR

Vielen Dank für dein Interesse, zu TRADINO UNSCHLAGBAR beizutragen! 🚀

## 📋 **Contribution Guidelines**

### 🎯 **Wie du beitragen kannst**

- 🐛 **Bug Reports**: Fehler melden und reproduzierbare Schritte angeben
- 💡 **Feature Requests**: Neue Funktionen vorschlagen
- 📝 **Documentation**: Dokumentation verbessern
- 🧪 **Tests**: Test-Coverage erhöhen
- 🔧 **Code**: Bugfixes und neue Features implementieren

### 🛠️ **Development Setup**

1. **Repository forken**
   ```bash
   git clone https://github.com/DarkUnknownMe/tradino-unschlagbar.git
   cd tradino-unschlagbar
   ```

2. **Virtual Environment erstellen**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # oder
   venv\Scripts\activate     # Windows
   ```

3. **Dependencies installieren**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Environment konfigurieren**
   ```bash
   cp .env.example .env
   # .env mit deinen Test-API Keys bearbeiten
   ```

### 🔄 **Development Workflow**

1. **Feature Branch erstellen**
   ```bash
   git checkout -b feature/amazing-new-feature
   ```

2. **Code schreiben**
   - Folge den bestehenden Code-Konventionen
   - Füge Tests für neue Features hinzu
   - Dokumentiere deine Änderungen

3. **Tests ausführen**
   ```bash
   # Alle Tests
   python -m pytest tests/ -v
   
   # Mit Coverage
   python -m pytest tests/ --cov=tradino_unschlagbar
   
   # Spezifische Tests
   python -m pytest tests/test_trading_engine.py
   ```

4. **Code Quality prüfen**
   ```bash
   # Linting
   flake8 tradino_unschlagbar/
   
   # Formatting
   black tradino_unschlagbar/
   
   # Type checking
   mypy tradino_unschlagbar/
   ```

5. **Commit und Push**
   ```bash
   git add .
   git commit -m "✨ Add amazing new feature"
   git push origin feature/amazing-new-feature
   ```

6. **Pull Request erstellen**
   - Beschreibe deine Änderungen ausführlich
   - Verlinke relevante Issues
   - Füge Screenshots hinzu (falls UI-Änderungen)

### 📝 **Commit Message Guidelines**

Verwende [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: Neue Features
- `fix`: Bugfixes
- `docs`: Dokumentation
- `style`: Code-Formatierung
- `refactor`: Code-Refactoring
- `test`: Tests hinzufügen/ändern
- `chore`: Build-Prozess, Dependencies

**Beispiele:**
```
feat(trading): add new scalping strategy
fix(api): handle connection timeout errors
docs(readme): update installation instructions
test(risk): add unit tests for risk management
```

### 🧪 **Testing Guidelines**

- **Unit Tests**: Teste einzelne Funktionen/Klassen
- **Integration Tests**: Teste Komponenten-Interaktionen
- **E2E Tests**: Teste komplette Workflows
- **Mock External APIs**: Verwende Mocks für externe Services

**Test-Struktur:**
```python
def test_should_do_something_when_condition():
    # Given
    setup_test_data()
    
    # When
    result = function_under_test()
    
    # Then
    assert result == expected_value
```

### 🏗️ **Code Architecture**

**Verzeichnisstruktur:**
```
tradino_unschlagbar/
├── brain/          # AI/ML Komponenten
├── core/           # Core Trading Logic
├── connectors/     # Exchange Connections
├── strategies/     # Trading Strategies
├── analytics/      # Performance Analytics
├── security/       # Security & Validation
└── utils/          # Utilities
```

**Design Principles:**
- **Single Responsibility**: Eine Klasse = eine Verantwortung
- **Dependency Injection**: Lose Kopplung zwischen Komponenten
- **Interface Segregation**: Kleine, spezifische Interfaces
- **Error Handling**: Robuste Fehlerbehandlung
- **Logging**: Umfassendes Logging für Debugging

### 🔒 **Security Guidelines**

- **Niemals** API Keys oder Secrets committen
- **Immer** Input-Validierung verwenden
- **Sichere** Konfiguration für Production
- **Rate Limiting** für API-Calls implementieren
- **Encryption** für sensitive Daten

### 📊 **Performance Guidelines**

- **Profiling**: Verwende `cProfile` für Performance-Analyse
- **Async/Await**: Für I/O-intensive Operationen
- **Caching**: Für häufig abgerufene Daten
- **Database Optimization**: Effiziente Queries
- **Memory Management**: Vermeide Memory Leaks

### 🐛 **Bug Report Template**

```markdown
## 🐛 Bug Report

**Beschreibung:**
Kurze Beschreibung des Problems

**Schritte zur Reproduktion:**
1. Schritt 1
2. Schritt 2
3. Schritt 3

**Erwartetes Verhalten:**
Was sollte passieren

**Aktuelles Verhalten:**
Was passiert tatsächlich

**Environment:**
- OS: [z.B. Ubuntu 20.04]
- Python Version: [z.B. 3.9.7]
- TRADINO Version: [z.B. 1.0.0]

**Logs:**
```
Relevante Log-Ausgaben hier einfügen
```

**Screenshots:**
Falls relevant, Screenshots hinzufügen
```

### 💡 **Feature Request Template**

```markdown
## 💡 Feature Request

**Problem/Use Case:**
Welches Problem löst diese Feature?

**Vorgeschlagene Lösung:**
Wie könnte die Lösung aussehen?

**Alternativen:**
Welche Alternativen hast du erwogen?

**Zusätzlicher Kontext:**
Weitere relevante Informationen
```

### 📞 **Hilfe & Support**

- 💬 **Discussions**: Für allgemeine Fragen
- 🐛 **Issues**: Für Bugs und Feature Requests
- 📧 **Email**: support@tradino-unschlagbar.com
- 💬 **Telegram**: @TradinoSupport

### 🏆 **Contributors**

Vielen Dank an alle Contributors! 🙏

<!-- ALL-CONTRIBUTORS-LIST:START -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

**Code of Conduct**: Bitte lies unseren [Code of Conduct](CODE_OF_CONDUCT.md) bevor du beiträgst.

**Happy Coding! 🚀** 