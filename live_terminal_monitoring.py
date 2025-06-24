#!/usr/bin/env python3
"""
📊 TRADINO LIVE TERMINAL MONITORING
Real-time system monitoring and control via terminal
"""

import os
import sys
import time
import json
import psutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue

# Add project paths
sys.path.append('/root/tradino')
sys.path.append('/root/tradino/tradino_unschlagbar')

class LiveTerminalMonitoring:
    """📊 Live Terminal Monitoring System"""
    
    def __init__(self):
        self.running = True
        self.start_time = datetime.now()
        self.stats = {
            'system_uptime': 0,
            'tradino_processes': [],
            'api_calls': 0,
            'balance': 0.0,
            'positions': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0
        }
        
        # Monitoring configuration
        self.refresh_interval = 2  # seconds
        self.log_file = 'logs/live_monitoring.log'
        Path('logs').mkdir(exist_ok=True)
        
    def clear_screen(self):
        """🖥️ Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def get_system_stats(self):
        """💻 Get current system statistics"""
        try:
            # System resources
            self.stats['cpu_usage'] = psutil.cpu_percent(interval=0.1)
            self.stats['memory_usage'] = psutil.virtual_memory().percent
            self.stats['disk_usage'] = psutil.disk_usage('/').percent
            
            # TRADINO processes
            tradino_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent']):
                try:
                    if proc.info['cmdline'] and any('tradino' in str(arg).lower() for arg in proc.info['cmdline']):
                        tradino_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmd': ' '.join(proc.info['cmdline'][-2:]) if proc.info['cmdline'] else 'unknown',
                            'cpu': proc.info['cpu_percent'] or 0
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            self.stats['tradino_processes'] = tradino_processes
            self.stats['system_uptime'] = datetime.now() - self.start_time
            
            # Try to get trading data
            self.update_trading_stats()
            
        except Exception as e:
            self.log_error(f"Stats error: {e}")
    
    def update_trading_stats(self):
        """💰 Update trading-specific statistics"""
        try:
            # Check if API is available
            try:
                from core.bitget_trading_api import BitgetTradingAPI
                
                # Try to get balance
                api_key = os.getenv('BITGET_API_KEY', 'demo')
                secret = os.getenv('BITGET_SECRET_KEY', 'demo')
                passphrase = os.getenv('BITGET_PASSPHRASE', 'demo')
                
                api = BitgetTradingAPI(
                    api_key=api_key,
                    secret=secret,
                    passphrase=passphrase,
                    sandbox=True
                )
                
                if api.is_connected:
                    self.stats['balance'] = api.get_total_balance()
                    # Try to get positions
                    try:
                        positions = api.get_positions()
                        self.stats['positions'] = len(positions) if positions else 0
                    except:
                        self.stats['positions'] = 0
                
            except Exception as e:
                self.stats['balance'] = 'N/A'
                self.stats['positions'] = 'N/A'
            
            # Check for trading stats files
            stats_files = list(Path('data').glob('trading_stats_*.json'))
            if stats_files:
                latest_stats = max(stats_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_stats) as f:
                        data = json.load(f)
                        self.stats['signals_generated'] = data.get('signals_generated', 0)
                        self.stats['trades_executed'] = data.get('trades_executed', 0)
                except:
                    pass
                    
        except Exception as e:
            self.log_error(f"Trading stats error: {e}")
    
    def display_header(self):
        """🎨 Display monitoring header"""
        uptime_str = str(self.stats['system_uptime']).split('.')[0]
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        header = f"""
{'='*80}
🚀 TRADINO UNSCHLAGBAR - LIVE SYSTEM MONITORING
{'='*80}
⏰ Current Time: {current_time}
⏱️  System Uptime: {uptime_str}
🔄 Refresh Rate: {self.refresh_interval}s | Press CTRL+C to stop
{'='*80}
"""
        print(header)
    
    def display_system_health(self):
        """💻 Display system health metrics"""
        cpu_color = "🟢" if self.stats['cpu_usage'] < 50 else "🟡" if self.stats['cpu_usage'] < 80 else "🔴"
        mem_color = "🟢" if self.stats['memory_usage'] < 70 else "🟡" if self.stats['memory_usage'] < 85 else "🔴"
        disk_color = "🟢" if self.stats['disk_usage'] < 80 else "🟡" if self.stats['disk_usage'] < 90 else "🔴"
        
        print("💻 SYSTEM HEALTH:")
        print(f"   {cpu_color} CPU Usage:    {self.stats['cpu_usage']:5.1f}%")
        print(f"   {mem_color} Memory Usage: {self.stats['memory_usage']:5.1f}%")
        print(f"   {disk_color} Disk Usage:   {self.stats['disk_usage']:5.1f}%")
        print()
    
    def display_tradino_processes(self):
        """🔧 Display TRADINO processes"""
        processes = self.stats['tradino_processes']
        
        print("🔧 TRADINO PROCESSES:")
        if processes:
            for proc in processes:
                status = "🟢" if proc['cpu'] > 0 else "🟡"
                print(f"   {status} PID {proc['pid']:6} | {proc['name'][:15]:15} | CPU: {proc['cpu']:4.1f}% | {proc['cmd'][:40]}")
        else:
            print("   🔴 No TRADINO processes detected")
        print()
    
    def display_trading_status(self):
        """💰 Display trading status"""
        print("💰 TRADING STATUS:")
        
        # Balance
        if isinstance(self.stats['balance'], (int, float)):
            balance_color = "🟢" if self.stats['balance'] > 1000 else "🟡"
            print(f"   {balance_color} Account Balance: ${self.stats['balance']:,.2f} USDT (SANDBOX)")
        else:
            print(f"   🔴 Account Balance: {self.stats['balance']}")
        
        # Positions
        if isinstance(self.stats['positions'], int):
            pos_color = "🟢" if self.stats['positions'] == 0 else "🟡"
            print(f"   {pos_color} Open Positions:  {self.stats['positions']}")
        else:
            print(f"   🔴 Open Positions:  {self.stats['positions']}")
        
        # AI Activity
        print(f"   🤖 AI Signals:      {self.stats['signals_generated']}")
        print(f"   📈 Trades Executed: {self.stats['trades_executed']}")
        print()
    
    def display_recent_logs(self):
        """📋 Display recent system logs"""
        print("📋 RECENT ACTIVITY:")
        
        # Check for recent log files
        log_files = []
        log_dirs = ['logs', 'data']
        
        for log_dir in log_dirs:
            if Path(log_dir).exists():
                log_files.extend(Path(log_dir).glob('*.log'))
        
        if log_files:
            # Get most recent log file
            recent_log = max(log_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(recent_log, 'r') as f:
                    lines = f.readlines()
                    
                # Show last 5 lines
                for line in lines[-5:]:
                    timestamp = line[:19] if len(line) > 19 else "Unknown"
                    message = line[20:].strip() if len(line) > 20 else line.strip()
                    print(f"   📝 {timestamp} | {message[:60]}")
                    
            except Exception as e:
                print(f"   ❌ Error reading logs: {e}")
        else:
            print("   📝 No recent logs found")
        print()
    
    def display_control_info(self):
        """🎮 Display control information"""
        print("🎮 CONTROLS:")
        print("   ⌨️  CTRL+C         - Stop monitoring")
        print("   📱 Telegram        - Not available (use this monitor)")
        print("   🔧 Manual Control  - Available via API")
        print("   📊 Full Logs       - Check logs/ directory")
        print("="*80)
    
    def log_error(self, message):
        """📝 Log error message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{timestamp} [ERROR] {message}\n")
        except:
            pass
    
    def run_monitoring(self):
        """🚀 Main monitoring loop"""
        print("🚀 Starting TRADINO Live Terminal Monitoring...")
        print("📊 Initializing system...")
        time.sleep(2)
        
        try:
            while self.running:
                # Clear screen and update stats
                self.clear_screen()
                self.get_system_stats()
                
                # Display all sections
                self.display_header()
                self.display_system_health()
                self.display_tradino_processes()
                self.display_trading_status()
                self.display_recent_logs()
                self.display_control_info()
                
                # Wait for next refresh
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
        finally:
            print("📊 Live monitoring session ended")

def main():
    """🚀 Main function"""
    print("📊 TRADINO Live Terminal Monitoring")
    print("Initializing...")
    
    monitor = LiveTerminalMonitoring()
    monitor.run_monitoring()

if __name__ == "__main__":
    main() 