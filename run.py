#!/usr/bin/env python3
"""
🚀 TRADINO UNSCHLAGBAR - Main Launcher
Unschlagbarer AI-Trading Bot für Bitget Futures

Author: AI Trading Systems
Version: 1.0.0
"""

import asyncio
import signal
import sys
from pathlib import Path

# Projekt Root Path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger_pro import setup_logger
from utils.config_manager import ConfigManager
from core.trading_engine import TradingEngine

# Global Logger
logger = setup_logger("TRADINO_LAUNCHER")

class TradinoLauncher:
    """🚀 Haupt-Launcher für TRADINO UNSCHLAGBAR"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.trading_engine = None
        self.running = False
        
    async def startup(self):
        """🔥 System-Startup Sequenz"""
        try:
            logger.info("🚀 TRADINO UNSCHLAGBAR wird gestartet...")
            logger.info(f"📊 Version: {self.config.get('system.version')}")
            logger.info(f"🌍 Environment: {self.config.get('system.environment')}")
            
            # Trading Engine initialisieren
            self.trading_engine = TradingEngine(self.config)
            await self.trading_engine.initialize()
            
            logger.success("✅ TRADINO UNSCHLAGBAR erfolgreich gestartet!")
            logger.info("💪 System ist bereit für unschlagbares Trading!")
            
            self.running = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Startup Fehler: {e}")
            return False
    
    async def shutdown(self):
        """🛑 Graceful Shutdown"""
        logger.info("🛑 TRADINO UNSCHLAGBAR wird beendet...")
        self.running = False
        
        if self.trading_engine:
            await self.trading_engine.shutdown()
            
        logger.info("✅ TRADINO UNSCHLAGBAR erfolgreich beendet!")
    
    async def run(self):
        """🏃‍♂️ Main Run Loop"""
        if not await self.startup():
            sys.exit(1)
            
        try:
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("⌨️  Benutzer-Unterbrechung erkannt")
        except Exception as e:
            logger.error(f"❌ Runtime Fehler: {e}")
        finally:
            await self.shutdown()

def signal_handler(signum, frame):
    """Signal Handler für graceful shutdown"""
    logger.info(f"🛑 Signal {signum} empfangen, beende System...")
    sys.exit(0)

async def main():
    """🎯 Main Entry Point"""
    # Signal Handler registrieren
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # TRADINO starten
    launcher = TradinoLauncher()
    await launcher.run()

if __name__ == "__main__":
    print("🚀 TRADINO UNSCHLAGBAR - Der ultimative AI-Trading Bot")
    print("💪 Bereit für unschlagbare Performance!")
    
    asyncio.run(main()) 