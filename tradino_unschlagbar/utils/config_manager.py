"""
⚙️ TRADINO UNSCHLAGBAR - Configuration Manager
Zentrales Konfigurationsmanagement mit YAML und Environment Support

Author: AI Trading Systems
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from utils.logger_pro import setup_logger

logger = setup_logger("ConfigManager")


class ConfigManager:
    """🔧 Professional Configuration Manager"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = Path(config_file)
        self.config_data: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self):
        """Konfiguration laden (YAML + Environment)"""
        try:
            # Environment Variablen laden
            load_dotenv()
            
            # YAML Konfiguration laden
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config_data = yaml.safe_load(f) or {}
                logger.success(f"✅ Konfiguration geladen: {self.config_file}")
            else:
                logger.warning(f"⚠️ Konfigurationsdatei nicht gefunden: {self.config_file}")
                self.config_data = {}
            
            # Environment Overrides anwenden
            self._apply_env_overrides()
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden der Konfiguration: {e}")
            self.config_data = {}
    
    def _apply_env_overrides(self):
        """Environment-Variablen als Overrides anwenden"""
        env_mappings = {
            'INITIAL_CAPITAL': 'trading.initial_capital',
            'MAX_RISK_PER_TRADE': 'trading.risk_per_trade',
            'MAX_DAILY_DRAWDOWN': 'trading.max_daily_drawdown',
            'BITGET_API_KEY': 'exchange.api_key',
            'BITGET_SECRET_KEY': 'exchange.api_secret',
            'BITGET_PASSPHRASE': 'exchange.api_passphrase',
            'BITGET_SANDBOX': 'exchange.sandbox',
            'TELEGRAM_BOT_TOKEN': 'telegram.bot_token',
            'TELEGRAM_CHAT_ID': 'telegram.chat_id',
            'LOG_LEVEL': 'system.log_level',
            'ENVIRONMENT': 'system.environment',
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(config_path, env_value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Konfigurationswert abrufen (Dot-Notation unterstützt)"""
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Konfigurationswert setzen (Dot-Notation unterstützt)"""
        self._set_nested_value(key, value)
    
    def _set_nested_value(self, key: str, value: Any):
        """Verschachtelten Wert setzen"""
        keys = key.split('.')
        data = self.config_data
        
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        
        # Type Conversion für spezielle Werte
        if isinstance(value, str):
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.replace('.', '', 1).isdigit():
                value = float(value) if '.' in value else int(value)
        
        data[keys[-1]] = value
    
    def save_config(self):
        """Konfiguration speichern"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
            logger.success(f"✅ Konfiguration gespeichert: {self.config_file}")
        except Exception as e:
            logger.error(f"❌ Fehler beim Speichern der Konfiguration: {e}")
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Trading-spezifische Konfiguration abrufen"""
        return self.get('trading', {})
    
    def get_exchange_config(self) -> Dict[str, Any]:
        """Exchange-spezifische Konfiguration abrufen"""
        return self.get('exchange', {})
    
    def get_telegram_config(self) -> Dict[str, Any]:
        """Telegram-spezifische Konfiguration abrufen"""
        return self.get('telegram', {})
    
    def is_demo_mode(self) -> bool:
        """Prüfen ob Demo-Modus aktiv ist"""
        return self.get('system.environment', 'demo') == 'demo'
    
    def validate_config(self) -> bool:
        """Konfiguration validieren"""
        required_keys = [
            'exchange.api_key',
            'exchange.api_secret', 
            'exchange.api_passphrase',
            'telegram.bot_token',
            'telegram.chat_id'
        ]
        
        missing_keys = []
        for key in required_keys:
            if not self.get(key):
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"❌ Fehlende Konfiguration: {', '.join(missing_keys)}")
            return False
      
        logger.success("✅ Konfiguration vollständig validiert")
        return True
