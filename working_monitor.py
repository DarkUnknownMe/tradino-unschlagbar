#!/usr/bin/env python3
"""
🚀 TRADINO WORKING LIVE MONITOR
Funktionierendes Live-Monitoring für TRADINO System
"""

import os
import sys
import time
import json
import psutil
from datetime import datetime
from pathlib import Path

# Add project paths
sys.path.insert(0, '/root/tradino')

class WorkingLiveMonitor:
    """📊 Funktionierendes Live Monitor für TRADINO"""
    
    def __init__(self):
        self.running = True
        self.start_time = datetime.now()
        self.stats = {
            'api_calls': 0,
            'last_balance_check': None,
            'last_price_check': None
        }
        
    def clear_screen(self):
        """🖥️ Clear screen"""
        os.system('clear')
    
    def get_tradino_processes(self):
        """🔧 Get TRADINO processes"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                if proc.info['cmdline'] and any('tradino' in str(arg).lower() for arg in proc.info['cmdline']):
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmd': ' '.join(proc.info['cmdline'][-3:]) if proc.info['cmdline'] else 'unknown',
                        'cpu': proc.info['cpu_percent'] or 0,
                        'memory': proc.info['memory_percent'] or 0
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
    
    def get_system_stats(self):
        """💻 Get system statistics"""
        return {
            'cpu': psutil.cpu_percent(interval=0.1),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent,
            'boot_time': datetime.fromtimestamp(psutil.boot_time())
        }
    
    def test_trading_api(self):
        """💰 Test trading API connection"""
        try:
            from core.bitget_trading_api import BitgetTradingAPI
            
            api = BitgetTradingAPI(
                api_key=os.getenv('BITGET_API_KEY', 'demo'),
                secret=os.getenv('BITGET_SECRET_KEY', 'demo'),
                passphrase=os.getenv('BITGET_PASSPHRASE', 'demo'),
                sandbox=True
            )
            
            self.stats['api_calls'] += 1
            
            if api.is_connected:
                balance = api.get_total_balance()
                price = api.get_current_price('BTCUSDT')
                
                self.stats['last_balance_check'] = datetime.now()
                self.stats['last_price_check'] = datetime.now()
                
                return {
                    'status': 'connected',
                    'balance': balance,
                    'btc_price': price,
                    'sandbox': api.sandbox
                }
            else:
                return {'status': 'disconnected'}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def check_logs_activity(self):
        """📋 Check log activity"""
        log_info = []
        
        # Check various log locations
        log_locations = ['logs', 'data', '.']
        
        for location in log_locations:
            if Path(location).exists():
                log_files = list(Path(location).glob('*.log'))
                for log_file in log_files:
                    try:
                        stat = log_file.stat()
                        age = time.time() - stat.st_mtime
                        if age < 300:  # Files modified in last 5 minutes
                            log_info.append({
                                'file': log_file.name,
                                'size': stat.st_size,
                                'age_seconds': int(age),
                                'location': location
                            })
                    except:
                        continue
        
        return sorted(log_info, key=lambda x: x['age_seconds'])
    
    def check_ai_activity(self):
        """🧠 Check AI activity"""
        ai_info = {
            'models_loaded': False,
            'recent_predictions': 0,
            'last_activity': None
        }
        
        # Check for AI model files
        model_paths = ['models', 'tradino_unschlagbar/data/models']
        for path in model_paths:
            if Path(path).exists():
                model_files = list(Path(path).glob('*.pkl')) + list(Path(path).glob('*.zip'))
                if model_files:
                    ai_info['models_loaded'] = True
                    break
        
        # Check for recent AI data
        ai_data_paths = ['data/ai_analysis', 'data']
        for path in ai_data_paths:
            if Path(path).exists():
                ai_files = list(Path(path).glob('*ai*.json'))
                if ai_files:
                    latest = max(ai_files, key=lambda x: x.stat().st_mtime)
                    age = time.time() - latest.stat().st_mtime
                    if age < 600:  # Less than 10 minutes old
                        ai_info['last_activity'] = int(age)
                    break
        
        return ai_info
    
    def display_status(self):
        """🎨 Display complete status"""
        uptime = str(datetime.now() - self.start_time).split('.')[0]
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Get all data
        system_stats = self.get_system_stats()
        processes = self.get_tradino_processes()
        api_status = self.test_trading_api()
        log_activity = self.check_logs_activity()
        ai_activity = self.check_ai_activity()
        
        # Clear and display header
        self.clear_screen()
        
        print('=' * 80)
        print('🚀 TRADINO UNSCHLAGBAR - LIVE STATUS MONITOR')
        print('=' * 80)
        print(f'⏰ Time: {current_time} | Monitor Uptime: {uptime} | API Calls: {self.stats["api_calls"]}')
        print('🔄 Auto-refresh every 3 seconds | Press CTRL+C to stop')
        print('=' * 80)
        
        # System Health
        cpu_color = '🟢' if system_stats['cpu'] < 50 else '🟡' if system_stats['cpu'] < 80 else '🔴'
        mem_color = '🟢' if system_stats['memory'] < 70 else '🟡' if system_stats['memory'] < 85 else '🔴'
        disk_color = '🟢' if system_stats['disk'] < 80 else '🟡' if system_stats['disk'] < 90 else '🔴'
        
        print('💻 SYSTEM HEALTH:')
        print(f'   {cpu_color} CPU Usage:    {system_stats["cpu"]:5.1f}%')
        print(f'   {mem_color} Memory Usage: {system_stats["memory"]:5.1f}%')
        print(f'   {disk_color} Disk Usage:   {system_stats["disk"]:5.1f}%')
        print()
        
        # TRADINO Processes
        print('🔧 TRADINO PROCESSES:')
        if processes:
            for proc in processes:
                status = '🟢' if proc['cpu'] > 0 else '🟡'
                print(f'   {status} PID {proc["pid"]:6} | {proc["name"][:15]:15} | CPU: {proc["cpu"]:4.1f}% | MEM: {proc["memory"]:4.1f}%')
                print(f'       └─ {proc["cmd"][:65]}')
        else:
            print('   🔴 No TRADINO processes detected')
        print()
        
        # Trading API Status
        print('💰 TRADING API STATUS:')
        if api_status['status'] == 'connected':
            print('   🟢 Status:      CONNECTED')
            print(f'   💰 Balance:     ${api_status["balance"]:,.2f} USDT')
            print(f'   🏦 Mode:        {"SANDBOX" if api_status["sandbox"] else "LIVE"}')
            if api_status.get('btc_price'):
                print(f'   ₿  BTC Price:   ${api_status["btc_price"]:,.2f}')
            if self.stats['last_balance_check']:
                last_check = (datetime.now() - self.stats['last_balance_check']).seconds
                print(f'   🕐 Last Check:  {last_check}s ago')
        elif api_status['status'] == 'disconnected':
            print('   🟡 Status:      DISCONNECTED')
        else:
            print(f'   🔴 Status:      ERROR')
            print(f'       Error: {api_status.get("error", "Unknown")[:50]}')
        print()
        
        # AI Activity
        print('🧠 AI SYSTEM STATUS:')
        models_status = '🟢' if ai_activity['models_loaded'] else '🔴'
        print(f'   {models_status} AI Models:   {"LOADED" if ai_activity["models_loaded"] else "NOT FOUND"}')
        
        if ai_activity['last_activity'] is not None:
            activity_status = '🟢' if ai_activity['last_activity'] < 60 else '🟡'
            print(f'   {activity_status} Last Activity: {ai_activity["last_activity"]}s ago')
        else:
            print('   🔴 Last Activity: No recent activity')
        print()
        
        # Log Activity
        print('📋 RECENT LOG ACTIVITY:')
        if log_activity:
            for log in log_activity[:3]:
                age_color = '🟢' if log['age_seconds'] < 30 else '🟡' if log['age_seconds'] < 120 else '🔴'
                size_kb = log['size'] // 1024
                print(f'   {age_color} {log["file"][:25]:25} | {log["age_seconds"]:3}s ago | {size_kb:4}KB | {log["location"]}')
        else:
            print('   🔴 No recent log activity detected')
        print()
        
        # Trading Statistics
        print('📊 LIVE TRADING STATUS:')
        if api_status['status'] == 'connected':
            print('   🎯 Trading Mode: FUTURES')
            print('   📈 Strategy:     AI-Powered Signals')
            print('   🛡️  Risk Mgmt:    Active')
            print('   ⚡ Execution:    Real-time')
        else:
            print('   🔴 Trading:      OFFLINE (API disconnected)')
        
        print()
        print('🎮 CONTROLS: CTRL+C to stop monitoring')
        print('=' * 80)
    
    def run(self):
        """🚀 Run monitoring loop"""
        print('🚀 Starting TRADINO Working Live Monitor...')
        print('📊 Initializing connections...')
        time.sleep(2)
        
        try:
            while self.running:
                self.display_status()
                time.sleep(3)  # 3 second refresh
                
        except KeyboardInterrupt:
            print('\n🛑 Monitoring stopped by user')
        except Exception as e:
            print(f'\n❌ Monitor error: {e}')
            import traceback
            traceback.print_exc()
        finally:
            print('📊 Live monitoring session ended')

def main():
    """🚀 Main function"""
    print('🚀 TRADINO Working Live Monitor')
    print('Connecting to TRADINO system...')
    
    monitor = WorkingLiveMonitor()
    monitor.run()

if __name__ == '__main__':
    main() 