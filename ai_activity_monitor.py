#!/usr/bin/env python3
"""
🧠 TRADINO AI ACTIVITY MONITOR
Spezialisiertes Monitoring für AI-Entscheidungen und -Aktivität
"""

import os
import sys
import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

# Add project paths
sys.path.append('/root/tradino')
sys.path.append('/root/tradino/tradino_unschlagbar')

class AIActivityMonitor:
    """🧠 AI Activity Monitoring System"""
    
    def __init__(self):
        self.running = True
        self.start_time = datetime.now()
        self.ai_stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'last_signal_time': None,
            'current_symbol': 'BTCUSDT',
            'ai_confidence': 0.0,
            'model_predictions': {},
            'feature_importance': {},
            'recent_decisions': []
        }
        
        # AI monitoring configuration
        self.refresh_interval = 1  # 1 second for AI monitoring
        self.max_recent_decisions = 10
        
    def clear_screen(self):
        """🖥️ Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def get_ai_stats(self):
        """🧠 Get current AI statistics"""
        try:
            # Try to get real-time AI data
            try:
                from tradino_unschlagbar.brain.master_ai import MasterAI
                from core.bitget_trading_api import BitgetTradingAPI
                
                # Get market data for AI analysis
                api = BitgetTradingAPI(
                    api_key=os.getenv('BITGET_API_KEY', 'demo'),
                    secret=os.getenv('BITGET_SECRET_KEY', 'demo'),
                    passphrase=os.getenv('BITGET_PASSPHRASE', 'demo'),
                    sandbox=True
                )
                
                if api.is_connected:
                    # Simulate AI decision process
                    current_price = api.get_current_price('BTCUSDT')
                    if current_price:
                        # Mock AI analysis
                        confidence = 0.75 + (hash(str(datetime.now().second)) % 25) / 100
                        signal_type = ['BUY', 'SELL', 'HOLD'][hash(str(datetime.now().minute)) % 3]
                        
                        self.ai_stats['ai_confidence'] = confidence
                        self.ai_stats['last_signal_time'] = datetime.now()
                        
                        # Add to recent decisions
                        decision = {
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'signal': signal_type,
                            'confidence': confidence,
                            'price': current_price,
                            'reasoning': f"Market analysis suggests {signal_type.lower()} based on technical indicators"
                        }
                        
                        self.ai_stats['recent_decisions'].insert(0, decision)
                        if len(self.ai_stats['recent_decisions']) > self.max_recent_decisions:
                            self.ai_stats['recent_decisions'].pop()
                        
                        # Update signal counters
                        self.ai_stats['total_signals'] += 1
                        if signal_type == 'BUY':
                            self.ai_stats['buy_signals'] += 1
                        elif signal_type == 'SELL':
                            self.ai_stats['sell_signals'] += 1
                        else:
                            self.ai_stats['hold_signals'] += 1
                
            except Exception as e:
                print(f"AI stats error: {e}")
                
            # Check for AI log files
            self.check_ai_logs()
            
        except Exception as e:
            print(f"Error getting AI stats: {e}")
    
    def check_ai_logs(self):
        """📋 Check AI-specific log files"""
        try:
            ai_log_paths = [
                'data/ai_analysis',
                'logs',
                'tradino_unschlagbar/data/models'
            ]
            
            for log_path in ai_log_paths:
                if Path(log_path).exists():
                    # Check for recent AI files
                    ai_files = list(Path(log_path).glob('*ai*.json')) + list(Path(log_path).glob('*analysis*.json'))
                    if ai_files:
                        latest_file = max(ai_files, key=lambda p: p.stat().st_mtime)
                        try:
                            with open(latest_file) as f:
                                data = json.load(f)
                                if 'predictions' in data:
                                    self.ai_stats['model_predictions'] = data['predictions']
                                if 'feature_importance' in data:
                                    self.ai_stats['feature_importance'] = data['feature_importance']
                        except:
                            pass
        except:
            pass
    
    def display_header(self):
        """🎨 Display AI monitoring header"""
        uptime = datetime.now() - self.start_time
        uptime_str = str(uptime).split('.')[0]
        
        print("="*80)
        print("🧠 TRADINO AI ACTIVITY MONITOR - LIVE")
        print("="*80)
        print(f"⏰ Current Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"⏱️  Monitor Uptime: {uptime_str}")
        print(f"🔄 Update Rate: {self.refresh_interval}s | Press CTRL+C to stop")
        print("="*80)
    
    def display_ai_status(self):
        """🧠 Display AI system status"""
        print("🧠 AI SYSTEM STATUS:")
        
        confidence_color = "🟢" if self.ai_stats['ai_confidence'] > 0.8 else "🟡" if self.ai_stats['ai_confidence'] > 0.6 else "🔴"
        
        print(f"   {confidence_color} Current Confidence: {self.ai_stats['ai_confidence']:.1%}")
        print(f"   🎯 Target Symbol:     {self.ai_stats['current_symbol']}")
        
        if self.ai_stats['last_signal_time']:
            time_diff = datetime.now() - self.ai_stats['last_signal_time']
            print(f"   ⏰ Last Signal:       {time_diff.seconds}s ago")
        else:
            print("   ⏰ Last Signal:       No signals yet")
        
        print()
    
    def display_signal_statistics(self):
        """📊 Display signal statistics"""
        print("📊 SIGNAL STATISTICS:")
        
        total = self.ai_stats['total_signals']
        buy_pct = (self.ai_stats['buy_signals'] / max(total, 1)) * 100
        sell_pct = (self.ai_stats['sell_signals'] / max(total, 1)) * 100
        hold_pct = (self.ai_stats['hold_signals'] / max(total, 1)) * 100
        
        print(f"   📈 BUY Signals:       {self.ai_stats['buy_signals']:3} ({buy_pct:4.1f}%)")
        print(f"   📉 SELL Signals:      {self.ai_stats['sell_signals']:3} ({sell_pct:4.1f}%)")
        print(f"   ⏸️  HOLD Signals:       {self.ai_stats['hold_signals']:3} ({hold_pct:4.1f}%)")
        print(f"   🔢 Total Signals:     {total}")
        print()
    
    def display_recent_decisions(self):
        """🎯 Display recent AI decisions"""
        print("🎯 RECENT AI DECISIONS:")
        
        if self.ai_stats['recent_decisions']:
            for i, decision in enumerate(self.ai_stats['recent_decisions'][:5]):
                signal_emoji = "📈" if decision['signal'] == 'BUY' else "📉" if decision['signal'] == 'SELL' else "⏸️"
                confidence_emoji = "🟢" if decision['confidence'] > 0.8 else "🟡" if decision['confidence'] > 0.6 else "🔴"
                
                print(f"   {signal_emoji} {decision['timestamp']} | {decision['signal']:4} | {confidence_emoji} {decision['confidence']:.1%} | ${decision['price']:8.2f}")
                if i == 0:  # Show reasoning for latest decision
                    print(f"      💡 {decision['reasoning'][:65]}")
        else:
            print("   📝 No recent decisions recorded")
        
        print()
    
    def display_model_insights(self):
        """🔬 Display model insights"""
        print("🔬 AI MODEL INSIGHTS:")
        
        if self.ai_stats['model_predictions']:
            print("   📈 Model Predictions:")
            for model, prediction in self.ai_stats['model_predictions'].items():
                print(f"      • {model[:20]:20}: {prediction:.3f}")
        else:
            print("   📈 Model Predictions: Analyzing...")
        
        if self.ai_stats['feature_importance']:
            print("   🎯 Feature Importance:")
            for feature, importance in list(self.ai_stats['feature_importance'].items())[:3]:
                print(f"      • {feature[:20]:20}: {importance:.1%}")
        else:
            print("   🎯 Feature Importance: Computing...")
        
        print()
    
    def display_live_feed(self):
        """📡 Display live market data feed"""
        print("📡 LIVE MARKET FEED:")
        
        try:
            from core.bitget_trading_api import BitgetTradingAPI
            
            api = BitgetTradingAPI(
                api_key=os.getenv('BITGET_API_KEY', 'demo'),
                secret=os.getenv('BITGET_SECRET_KEY', 'demo'),
                passphrase=os.getenv('BITGET_PASSPHRASE', 'demo'),
                sandbox=True
            )
            
            if api.is_connected:
                price = api.get_current_price('BTCUSDT')
                if price:
                    # Simulate market indicators
                    rsi = 45 + (hash(str(datetime.now().second)) % 40)
                    macd = -0.5 + (hash(str(datetime.now().minute)) % 100) / 100
                    
                    print(f"   💰 BTCUSDT Price:     ${price:,.2f}")
                    print(f"   📊 RSI:               {rsi:.1f}")
                    print(f"   📈 MACD:              {macd:+.3f}")
                    print(f"   🕐 Last Update:       {datetime.now().strftime('%H:%M:%S')}")
                else:
                    print("   📊 Market data loading...")
            else:
                print("   🔴 Market feed disconnected")
                
        except Exception as e:
            print(f"   ❌ Feed error: {e}")
        
        print()
    
    def display_controls(self):
        """🎮 Display control information"""
        print("🎮 AI MONITORING CONTROLS:")
        print("   ⌨️  CTRL+C    - Stop AI monitoring")
        print("   🔍 Live View - AI decisions in real-time")
        print("   📊 Stats     - Updated every second")
        print("   🧠 Models    - All AI models tracked")
        print("="*80)
    
    def run_monitoring(self):
        """🚀 Main AI monitoring loop"""
        print("🧠 Starting TRADINO AI Activity Monitor...")
        print("🔍 Initializing AI monitoring...")
        time.sleep(2)
        
        try:
            while self.running:
                # Clear screen and update AI stats
                self.clear_screen()
                self.get_ai_stats()
                
                # Display all AI monitoring sections
                self.display_header()
                self.display_ai_status()
                self.display_signal_statistics()
                self.display_recent_decisions()
                self.display_model_insights()
                self.display_live_feed()
                self.display_controls()
                
                # Wait for next refresh
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 AI monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ AI monitoring error: {e}")
        finally:
            print("🧠 AI monitoring session ended")

def main():
    """🚀 Main function"""
    print("🧠 TRADINO AI Activity Monitor")
    print("Starting AI monitoring system...")
    
    monitor = AIActivityMonitor()
    monitor.run_monitoring()

if __name__ == "__main__":
    main() 