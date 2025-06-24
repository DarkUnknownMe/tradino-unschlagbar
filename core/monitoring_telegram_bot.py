#!/usr/bin/env python3
"""
📱 MONITORING TELEGRAM BOT
Comprehensive Telegram integration for TRADINO monitoring system
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Telegram Integration
try:
    import telegram
    from telegram import Bot, Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

# Import monitoring system
try:
    from monitoring_system import (
        get_monitoring_system, AlertEvent, LogCategory, 
        AlertSeverity, SystemMetrics
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

class MonitoringTelegramBot:
    """📱 Comprehensive Monitoring Telegram Bot"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = None
        self.application = None
        self.is_running = False
        
        if not TELEGRAM_AVAILABLE:
            raise ImportError("Telegram library not available")
        
        try:
            self.bot = Bot(token=self.bot_token)
            self.application = Application.builder().token(self.bot_token).build()
            self.register_handlers()
            print("✅ Monitoring Telegram Bot initialized")
        except Exception as e:
            print(f"❌ Bot initialization failed: {e}")
            raise
    
    def register_handlers(self):
        """📋 Register command handlers"""
        
        # System Monitoring Commands
        self.application.add_handler(CommandHandler("system_status", self.system_status))
        self.application.add_handler(CommandHandler("system_health", self.system_health))
        self.application.add_handler(CommandHandler("system_metrics", self.system_metrics))
        self.application.add_handler(CommandHandler("performance", self.performance_summary))
        
        # Alert Commands
        self.application.add_handler(CommandHandler("alerts", self.active_alerts))
        self.application.add_handler(CommandHandler("alert_history", self.alert_history))
        self.application.add_handler(CommandHandler("clear_alerts", self.clear_alerts))
        
        # Log Commands
        self.application.add_handler(CommandHandler("logs", self.recent_logs))
        self.application.add_handler(CommandHandler("errors", self.recent_errors))
        self.application.add_handler(CommandHandler("trades", self.recent_trades))
        self.application.add_handler(CommandHandler("ai_decisions", self.recent_ai_decisions))
        
        # Analytics Commands
        self.application.add_handler(CommandHandler("analytics", self.analytics_summary))
        self.application.add_handler(CommandHandler("risk_events", self.recent_risk_events))
        self.application.add_handler(CommandHandler("api_status", self.api_status))
        
        # Control Commands
        self.application.add_handler(CommandHandler("monitoring_start", self.start_monitoring))
        self.application.add_handler(CommandHandler("monitoring_stop", self.stop_monitoring))
        self.application.add_handler(CommandHandler("monitoring_restart", self.restart_monitoring))
        
        # Help Commands
        self.application.add_handler(CommandHandler("monitor_help", self.monitor_help))
        self.application.add_handler(CommandHandler("start", self.start_command))
        
        print("✅ Monitoring command handlers registered")
    
    async def system_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📊 Show comprehensive system status"""
        try:
            monitoring_system = get_monitoring_system()
            if not monitoring_system:
                await update.message.reply_text("❌ Monitoring system not available")
                return
            
            # Get performance summary
            performance = monitoring_system.get_performance_summary()
            
            # Get latest metrics
            metrics = monitoring_system.get_system_metrics()
            
            message = f"📊 TRADINO SYSTEM STATUS\n\n"
            message += f"🕐 Session: {performance['session_id']}\n"
            message += f"⏱️ Uptime: {performance['uptime_hours']:.1f} hours\n"
            message += f"🏥 Health: {'✅ Healthy' if performance['system_healthy'] else '❌ Issues'}\n"
            message += f"📝 Total Logs: {performance['total_logs']}\n"
            message += f"🚨 Active Alerts: {performance['active_alerts']}\n"
            message += f"🧵 Monitoring Threads: {performance['monitoring_threads']}\n\n"
            
            if metrics:
                message += f"💻 CURRENT METRICS:\n"
                message += f"CPU: {metrics.cpu_usage:.1f}%\n"
                message += f"RAM: {metrics.memory_usage:.1f}%\n"
                message += f"Disk: {metrics.disk_usage:.1f}%\n"
                message += f"Threads: {metrics.active_threads}\n"
                message += f"API Latency: {metrics.api_latency:.1f}ms\n\n"
            
            message += f"🔧 Log Categories: {performance['log_categories']}"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def system_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """💓 Show detailed system health"""
        try:
            monitoring_system = get_monitoring_system()
            if not monitoring_system:
                await update.message.reply_text("❌ Monitoring system not available")
                return
            
            is_healthy = monitoring_system._is_system_healthy()
            metrics = monitoring_system.get_system_metrics()
            
            message = f"💓 SYSTEM HEALTH REPORT\n\n"
            message += f"Overall Status: {'✅ HEALTHY' if is_healthy else '❌ UNHEALTHY'}\n\n"
            
            if metrics:
                thresholds = monitoring_system.config['alert_thresholds']
                
                # CPU Status
                cpu_status = "✅" if metrics.cpu_usage <= thresholds['cpu_usage'] else "⚠️"
                message += f"{cpu_status} CPU: {metrics.cpu_usage:.1f}% (limit: {thresholds['cpu_usage']}%)\n"
                
                # Memory Status
                mem_status = "✅" if metrics.memory_usage <= thresholds['memory_usage'] else "⚠️"
                message += f"{mem_status} Memory: {metrics.memory_usage:.1f}% (limit: {thresholds['memory_usage']}%)\n"
                
                # Disk Status
                disk_status = "✅" if metrics.disk_usage <= thresholds['disk_usage'] else "⚠️"
                message += f"{disk_status} Disk: {metrics.disk_usage:.1f}% (limit: {thresholds['disk_usage']}%)\n"
                
                # API Latency Status
                api_status = "✅" if metrics.api_latency <= thresholds['api_latency'] else "⚠️"
                message += f"{api_status} API Latency: {metrics.api_latency:.1f}ms (limit: {thresholds['api_latency']*1000:.0f}ms)\n\n"
                
                # Network I/O
                message += f"📡 NETWORK I/O:\n"
                message += f"Bytes Sent: {metrics.network_io.get('bytes_sent', 0):,}\n"
                message += f"Bytes Received: {metrics.network_io.get('bytes_recv', 0):,}\n"
                message += f"Packets Sent: {metrics.network_io.get('packets_sent', 0):,}\n"
                message += f"Packets Received: {metrics.network_io.get('packets_recv', 0):,}\n"
            
            # Check monitoring threads
            message += f"\n🧵 MONITORING THREADS:\n"
            for name, thread in monitoring_system.monitoring_threads.items():
                status = "✅ Running" if thread.is_alive() else "❌ Stopped"
                message += f"{name}: {status}\n"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def system_metrics(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📈 Show detailed system metrics"""
        try:
            monitoring_system = get_monitoring_system()
            if not monitoring_system:
                await update.message.reply_text("❌ Monitoring system not available")
                return
            
            metrics = monitoring_system.get_system_metrics()
            if not metrics:
                await update.message.reply_text("📊 No metrics data available")
                return
            
            message = f"📈 SYSTEM METRICS\n\n"
            message += f"🕐 Timestamp: {metrics.timestamp[11:19]}\n\n"
            
            message += f"💻 SYSTEM RESOURCES:\n"
            message += f"CPU Usage: {metrics.cpu_usage:.1f}%\n"
            message += f"Memory Usage: {metrics.memory_usage:.1f}%\n"
            message += f"Disk Usage: {metrics.disk_usage:.1f}%\n\n"
            
            message += f"🔄 SYSTEM ACTIVITY:\n"
            message += f"Active Threads: {metrics.active_threads}\n"
            message += f"Open Positions: {metrics.open_positions}\n"
            message += f"API Latency: {metrics.api_latency:.1f}ms\n"
            message += f"AI Processing: {metrics.ai_processing_time:.1f}ms\n\n"
            
            if metrics.last_trade_time:
                message += f"💰 Last Trade: {metrics.last_trade_time[11:19]}\n"
            else:
                message += f"💰 Last Trade: No trades today\n"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def performance_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🚀 Show performance summary"""
        try:
            monitoring_system = get_monitoring_system()
            if not monitoring_system:
                await update.message.reply_text("❌ Monitoring system not available")
                return
            
            performance = monitoring_system.get_performance_summary()
            
            message = f"🚀 PERFORMANCE SUMMARY\n\n"
            message += f"📅 Session: {performance['session_id']}\n"
            message += f"⏱️ Uptime: {performance['uptime_hours']:.2f} hours\n"
            message += f"🏥 System Status: {'✅ Healthy' if performance['system_healthy'] else '❌ Issues'}\n\n"
            
            message += f"📊 ACTIVITY METRICS:\n"
            message += f"Total Log Entries: {performance['total_logs']:,}\n"
            message += f"Active Alerts: {performance['active_alerts']}\n"
            message += f"Monitoring Threads: {performance['monitoring_threads']}\n"
            message += f"Log Categories: {performance['log_categories']}\n\n"
            
            # Calculate logs per hour
            if performance['uptime_hours'] > 0:
                logs_per_hour = performance['total_logs'] / performance['uptime_hours']
                message += f"📈 Log Rate: {logs_per_hour:.0f} entries/hour\n"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def active_alerts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🚨 Show active alerts"""
        try:
            monitoring_system = get_monitoring_system()
            if not monitoring_system:
                await update.message.reply_text("❌ Monitoring system not available")
                return
            
            active_alerts = monitoring_system.active_alerts
            
            if not active_alerts:
                await update.message.reply_text("✅ No active alerts")
                return
            
            message = f"🚨 ACTIVE ALERTS ({len(active_alerts)})\n\n"
            
            for alert_key, last_time in active_alerts.items():
                category, title = alert_key.split('_', 1)
                time_ago = (time.time() - last_time) / 60  # minutes
                message += f"⚠️ {title.replace('_', ' ').title()}\n"
                message += f"   Category: {category}\n"
                message += f"   Time: {time_ago:.0f}m ago\n\n"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def recent_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📝 Show recent log entries"""
        try:
            # Get count from args or use default
            count = 5
            if context.args and len(context.args) > 0:
                try:
                    count = min(int(context.args[0]), 20)  # Max 20 entries
                except ValueError:
                    pass
            
            # Read recent log entries from log files
            monitoring_system = get_monitoring_system()
            if not monitoring_system:
                await update.message.reply_text("❌ Monitoring system not available")
                return
            
            message = f"📝 RECENT LOG ENTRIES (Last {count})\n\n"
            
            # Try to read from log files
            try:
                import os
                log_dir = monitoring_system.log_dir
                
                # Get most recent log entries across all categories
                recent_entries = []
                
                for category in LogCategory:
                    log_file = os.path.join(log_dir, f"{category.value}.log")
                    if os.path.exists(log_file):
                        try:
                            with open(log_file, 'r') as f:
                                lines = f.readlines()
                                for line in lines[-count:]:
                                    try:
                                        entry = json.loads(line)
                                        recent_entries.append(entry)
                                    except json.JSONDecodeError:
                                        continue
                        except Exception:
                            continue
                
                # Sort by timestamp
                recent_entries.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                
                for entry in recent_entries[:count]:
                    timestamp = entry.get('timestamp', '')[:19].replace('T', ' ')
                    level = entry.get('level', 'INFO')
                    category = entry.get('category', 'unknown')
                    msg = entry.get('message', '')[:50]
                    
                    level_emoji = {
                        'INFO': 'ℹ️',
                        'WARN': '⚠️',
                        'ERROR': '❌',
                        'CRITICAL': '🚨',
                        'DEBUG': '🐛',
                        'TRACE': '🔍'
                    }.get(level, 'ℹ️')
                    
                    message += f"{level_emoji} {timestamp}\n"
                    message += f"   {category}: {msg}\n\n"
                
                if not recent_entries:
                    message += "No log entries found"
                
            except Exception as e:
                message += f"Error reading logs: {e}"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def recent_errors(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """❌ Show recent error logs"""
        try:
            # Get count from args or use default
            count = 5
            if context.args and len(context.args) > 0:
                try:
                    count = min(int(context.args[0]), 10)
                except ValueError:
                    pass
            
            monitoring_system = get_monitoring_system()
            if not monitoring_system:
                await update.message.reply_text("❌ Monitoring system not available")
                return
            
            message = f"❌ RECENT ERRORS (Last {count})\n\n"
            
            # Read error logs
            try:
                import os
                log_dir = monitoring_system.log_dir
                
                error_entries = []
                
                for category in LogCategory:
                    log_file = os.path.join(log_dir, f"{category.value}.log")
                    if os.path.exists(log_file):
                        try:
                            with open(log_file, 'r') as f:
                                lines = f.readlines()
                                for line in lines[-50:]:  # Check last 50 lines
                                    try:
                                        entry = json.loads(line)
                                        if entry.get('level') in ['ERROR', 'CRITICAL']:
                                            error_entries.append(entry)
                                    except json.JSONDecodeError:
                                        continue
                        except Exception:
                            continue
                
                # Sort by timestamp
                error_entries.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                
                for entry in error_entries[:count]:
                    timestamp = entry.get('timestamp', '')[:19].replace('T', ' ')
                    level = entry.get('level', 'ERROR')
                    category = entry.get('category', 'unknown')
                    msg = entry.get('message', '')
                    
                    level_emoji = '🚨' if level == 'CRITICAL' else '❌'
                    
                    message += f"{level_emoji} {timestamp}\n"
                    message += f"   {category}: {msg[:100]}\n\n"
                
                if not error_entries:
                    message += "✅ No recent errors found"
                
            except Exception as e:
                message += f"Error reading error logs: {e}"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def recent_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """💰 Show recent trade logs"""
        try:
            # Get count from args or use default
            count = 5
            if context.args and len(context.args) > 0:
                try:
                    count = min(int(context.args[0]), 10)
                except ValueError:
                    pass
            
            monitoring_system = get_monitoring_system()
            if not monitoring_system:
                await update.message.reply_text("❌ Monitoring system not available")
                return
            
            message = f"💰 RECENT TRADES (Last {count})\n\n"
            
            # Read trade logs
            try:
                import os
                log_dir = monitoring_system.log_dir
                trade_log_file = os.path.join(log_dir, "trade.log")
                
                if os.path.exists(trade_log_file):
                    with open(trade_log_file, 'r') as f:
                        lines = f.readlines()
                        trade_entries = []
                        
                        for line in lines[-50:]:  # Check last 50 lines
                            try:
                                entry = json.loads(line)
                                trade_entries.append(entry)
                            except json.JSONDecodeError:
                                continue
                    
                    # Sort by timestamp
                    trade_entries.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                    
                    for entry in trade_entries[:count]:
                        timestamp = entry.get('timestamp', '')[:19].replace('T', ' ')
                        data = entry.get('data', {})
                        
                        symbol = data.get('symbol', 'Unknown')
                        action = data.get('action', 'Unknown')
                        quantity = data.get('quantity', 0)
                        price = data.get('price', 0)
                        total_value = data.get('total_value', 0)
                        
                        action_emoji = '🟢' if action.lower() == 'buy' else '🔴'
                        
                        message += f"{action_emoji} {timestamp}\n"
                        message += f"   {action.upper()} {symbol}\n"
                        message += f"   Qty: {quantity:.6f}\n"
                        message += f"   Price: ${price:.2f}\n"
                        message += f"   Value: ${total_value:.2f}\n\n"
                    
                    if not trade_entries:
                        message += "No recent trades found"
                else:
                    message += "No trade log file found"
                
            except Exception as e:
                message += f"Error reading trade logs: {e}"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def monitor_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """❓ Show monitoring bot help"""
        message = f"📊 MONITORING BOT COMMANDS\n\n"
        
        message += f"📊 SYSTEM STATUS:\n"
        message += f"/system_status - Complete system overview\n"
        message += f"/system_health - Detailed health report\n"
        message += f"/system_metrics - Current system metrics\n"
        message += f"/performance - Performance summary\n\n"
        
        message += f"🚨 ALERTS:\n"
        message += f"/alerts - Show active alerts\n"
        message += f"/alert_history - Recent alert history\n"
        message += f"/clear_alerts - Clear all active alerts\n\n"
        
        message += f"📝 LOGS:\n"
        message += f"/logs [count] - Recent log entries (max 20)\n"
        message += f"/errors [count] - Recent error logs (max 10)\n"
        message += f"/trades [count] - Recent trades (max 10)\n"
        message += f"/ai_decisions - Recent AI decisions\n\n"
        
        message += f"📈 ANALYTICS:\n"
        message += f"/analytics - Analytics summary\n"
        message += f"/risk_events - Recent risk events\n"
        message += f"/api_status - API connection status\n\n"
        
        message += f"🔧 CONTROL:\n"
        message += f"/monitoring_start - Start monitoring\n"
        message += f"/monitoring_stop - Stop monitoring\n"
        message += f"/monitoring_restart - Restart monitoring\n\n"
        
        message += f"❓ HELP:\n"
        message += f"/monitor_help - This help message\n"
        
        await update.message.reply_text(message)
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🚀 Start command"""
        await update.message.reply_text(
            f"📊 TRADINO MONITORING BOT\n\n"
            f"Professional monitoring and alerting system.\n\n"
            f"Use /monitor_help for all available commands.\n"
            f"Use /system_status for a quick overview."
        )
    
    def start_bot(self):
        """🚀 Start the Telegram bot"""
        if not TELEGRAM_AVAILABLE:
            print("❌ Telegram not available")
            return
        
        try:
            self.is_running = True
            print("🚀 Starting Monitoring Telegram Bot...")
            self.application.run_polling()
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"❌ Bot error: {e}")
        finally:
            self.is_running = False
    
    def stop_bot(self):
        """🛑 Stop the Telegram bot"""
        if self.application:
            self.application.stop()
        self.is_running = False
        print("🛑 Monitoring Bot stopped")

# Alert notification handler
def send_alert_notification(alert: AlertEvent):
    """📱 Send alert notification via Telegram"""
    try:
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            return
        
        # Create alert message
        severity_emoji = {
            AlertSeverity.LOW: '🔵',
            AlertSeverity.MEDIUM: '🟡',
            AlertSeverity.HIGH: '🟠',
            AlertSeverity.CRITICAL: '🔴'
        }.get(alert.severity, '⚪')
        
        category_emoji = {
            LogCategory.TRADE: '💰',
            LogCategory.AI_DECISION: '🤖',
            LogCategory.RISK_EVENT: '🛡️',
            LogCategory.SYSTEM_HEALTH: '💻',
            LogCategory.API_CONNECTION: '🌐',
            LogCategory.MARKET_DATA: '📊',
            LogCategory.PERFORMANCE: '📈'
        }.get(alert.category, '⚠️')
        
        message = f"{severity_emoji} {category_emoji} ALERT\n\n"
        message += f"Title: {alert.title}\n"
        message += f"Severity: {alert.severity.value.upper()}\n"
        message += f"Category: {alert.category.value}\n"
        message += f"Time: {alert.timestamp[11:19]}\n\n"
        message += f"Message: {alert.message}\n"
        
        # Add data if available
        if alert.data:
            message += f"\nDetails:\n"
            for key, value in alert.data.items():
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        message += f"  {key}: {value:.2f}\n"
                    else:
                        message += f"  {key}: {value}\n"
                else:
                    message += f"  {key}: {str(value)[:50]}\n"
        
        # Send notification
        import telegram
        bot = telegram.Bot(token=bot_token)
        asyncio.create_task(bot.send_message(chat_id=chat_id, text=message))
        
    except Exception as e:
        print(f"❌ Failed to send alert notification: {e}")

# Global instance
monitoring_telegram_bot = None

def initialize_monitoring_telegram_bot() -> Optional[MonitoringTelegramBot]:
    """📱 Initialize Monitoring Telegram Bot"""
    global monitoring_telegram_bot
    
    if not TELEGRAM_AVAILABLE:
        print("❌ Telegram not available")
        return None
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("❌ Telegram credentials not found")
        print("Required: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
        return None
    
    try:
        monitoring_telegram_bot = MonitoringTelegramBot(bot_token, chat_id)
        return monitoring_telegram_bot
    except Exception as e:
        print(f"❌ Failed to initialize Monitoring Telegram Bot: {e}")
        return None

def get_monitoring_telegram_bot() -> Optional[MonitoringTelegramBot]:
    """📱 Get Monitoring Telegram Bot instance"""
    return monitoring_telegram_bot

if __name__ == "__main__":
    print("📱 TRADINO MONITORING TELEGRAM BOT")
    print("=" * 50)
    
    bot = initialize_monitoring_telegram_bot()
    if bot:
        print("🚀 Starting bot...")
        bot.start_bot()
    else:
        print("❌ Bot not available")
 