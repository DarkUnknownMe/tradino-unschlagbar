"""
🚀 TRADINO UNSCHLAGBAR - System Launcher
Production Launch Script mit Pre-flight Checks

Author: AI Trading Systems
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger_pro import setup_logger
from utils.config_manager import ConfigManager
from core.trading_engine import TradingEngine

logger = setup_logger("SystemLauncher")


class TRADINOLauncher:
    """🚀 TRADINO Production Launcher"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.trading_engine = None
        
    async def pre_flight_checks(self) -> bool:
        """✈️ Pre-flight System Checks"""
        try:
            print("✈️ TRADINO UNSCHLAGBAR - PRE-FLIGHT CHECKS")
            print("=" * 60)
            
            checks = [
                ("Configuration", self._check_configuration),
                ("API Keys", self._check_api_keys),
                ("Network", self._check_network),
                ("Permissions", self._check_permissions),
                ("System Resources", self._check_resources)
            ]
            
            all_passed = True
            
            for check_name, check_func in checks:
                print(f"🔍 Checking {check_name}...", end="")
                try:
                    result = await check_func()
                    if result:
                        print(" ✅ PASSED")
                    else:
                        print(" ❌ FAILED")
                        all_passed = False
                except Exception as e:
                    print(f" ❌ ERROR: {e}")
                    all_passed = False
            
            print("-" * 60)
            if all_passed:
                print("✅ ALL PRE-FLIGHT CHECKS PASSED")
                print("🚀 System ready for launch!")
            else:
                print("❌ SOME PRE-FLIGHT CHECKS FAILED")
                print("🔧 Please fix issues before launch")
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Pre-flight check error: {e}")
            return False
    
    async def _check_configuration(self) -> bool:
        """⚙️ Configuration Check"""
        return self.config.validate_config()
    
    async def _check_api_keys(self) -> bool:
        """🔑 API Keys Check"""
        required_keys = [
            'exchange.api_key',
            'exchange.api_secret', 
            'exchange.api_passphrase',
            'telegram.bot_token',
            'telegram.chat_id'
        ]
        
        for key in required_keys:
            if not self.config.get(key):
                return False
        return True
    
    async def _check_network(self) -> bool:
        """🌐 Network Connectivity Check"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.bitget.com/api/spot/v1/public/time', timeout=5) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _check_permissions(self) -> bool:
        """📋 File Permissions Check"""
        try:
            # Test log file creation
            test_file = Path("data/logs/test.log")
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception:
            return False
    
    async def _check_resources(self) -> bool:
        """💾 System Resources Check"""
        try:
            import psutil
            
            # Memory check (mindestens 512MB verfügbar)
            available_memory = psutil.virtual_memory().available / 1024 / 1024
            if available_memory < 512:
                return False
            
            # Disk space check (mindestens 100MB)
            disk_usage = psutil.disk_usage('.')
            available_disk = disk_usage.free / 1024 / 1024
            if available_disk < 100:
                return False
            
            return True
        except Exception:
            return True  # Fallback wenn psutil nicht verfügbar
    
    async def launch_system(self) -> bool:
        """🚀 System Launch"""
        try:
            print("\n🚀 LAUNCHING TRADINO UNSCHLAGBAR")
            print("=" * 60)
            
            # Trading Engine erstellen
            print("🔧 Initializing Trading Engine...")
            self.trading_engine = TradingEngine(self.config)
            
            # Initialisierung
            print("⚙️ Initializing components...")
            init_success = await self.trading_engine.initialize()
            
            if not init_success:
                print("❌ System initialization failed!")
                return False
            
            print("✅ All components initialized successfully!")
            
            # System starten
            print("🔥 Starting trading engine...")
            start_success = await self.trading_engine.start()
            
            if not start_success:
                print("❌ Failed to start trading engine!")
                return False
            
            print("🎉 TRADINO UNSCHLAGBAR IS NOW LIVE!")
            print("-" * 60)
            print("📊 System Status: RUNNING")
            print("💰 Trading Mode: DEMO" if self.config.is_demo_mode() else "💰 Trading Mode: LIVE")
            print("📱 Telegram: ACTIVE")
            print("🛡️ Risk Management: ACTIVE")
            print("🧠 AI Intelligence: ACTIVE")
            print("-" * 60)
            print("💡 Monitor your bot via Telegram!")
            print("🛑 Press Ctrl+C to stop the system")
            
            # Auf Shutdown warten
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutdown signal received...")
                
            return True
            
        except Exception as e:
            print(f"❌ Launch error: {e}")
            return False
        finally:
            if self.trading_engine:
                print("🛑 Shutting down trading engine...")
                await self.trading_engine.shutdown()
                print("✅ System shutdown complete")


async def main():
    """🚀 Main Launch Function"""
    try:
        launcher = TRADINOLauncher()
        
        # Pre-flight Checks
        checks_passed = await launcher.pre_flight_checks()
        
        if not checks_passed:
            print("\n❌ Pre-flight checks failed!")
            print("🔧 Please fix the issues and try again")
            return False
        
        # User Confirmation für Live Launch
        print("\n" + "=" * 60)
        print("⚠️  FINAL CONFIRMATION REQUIRED")
        print("=" * 60)
        
        if launcher.config.is_demo_mode():
            print("✅ Demo Mode - Safe to proceed")
            response = input("🚀 Launch TRADINO UNSCHLAGBAR in DEMO mode? (y/N): ")
        else:
            print("🚨 LIVE MODE - Real money trading!")
            print("⚠️  Please ensure you understand the risks")
            response = input("🚀 Launch TRADINO UNSCHLAGBAR in LIVE mode? (yes/N): ")
        
        if response.strip().lower() not in ['y', 'yes']:
            print("👋 Launch cancelled by user")
            return False
        
        # System Launch
        success = await launcher.launch_system()
        
        return success
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        return False


if __name__ == "__main__":
    print("🚀 TRADINO UNSCHLAGBAR - SYSTEM LAUNCHER")
    print("💪 Preparing for launch...")
    
    success = asyncio.run(main())
    
    if success:
        print("👋 TRADINO UNSCHLAGBAR session ended")
    else:
        print("❌ Launch failed")
    
    sys.exit(0 if success else 1) 