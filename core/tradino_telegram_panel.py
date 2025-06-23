#!/usr/bin/env python3
"""
📱 TRADINO TELEGRAM CONTROL PANEL
Live Trading Control via Telegram für TRADINO UNSCHLAGBAR
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import traceback

# Load environment variables
try:
    from dotenv import load_dotenv
    if os.path.exists('tradino_unschlagbar/.env'):
        load_dotenv('tradino_unschlagbar/.env')
        print("✅ Loaded .env from tradino_unschlagbar/.env")
except ImportError:
    print("⚠️ python-dotenv not installed")

# Add project path
sys.path.append('/root/tradino')

# Telegram imports
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    print("❌ telegram library not installed: pip install python-telegram-bot")
    TELEGRAM_AVAILABLE = False

# Trading system imports
try:
    from final_live_trading_system import RealLiveTradingSystem, initialize_real_tradino_system
    TRADING_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"❌ Trading system not available: {e}")
    TRADING_SYSTEM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TradinoTelegramBot:
    """📱 TRADINO Telegram Control Panel"""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.authorized_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.application = None
        self.trading_system = None
        self.is_running = False
        
        # Bot state
        self.notifications_enabled = True
        self.last_update = datetime.now()
        
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")
        
        if not self.authorized_chat_id:
            raise ValueError("TELEGRAM_CHAT_ID not found in environment variables")
        
        print(f"📱 Telegram Bot initialized")
        print(f"🔑 Bot Token: {self.bot_token[:20]}...")
        print(f"👤 Authorized Chat ID: {self.authorized_chat_id}")
    
    def is_authorized(self, chat_id: str) -> bool:
        """🔐 Check if user is authorized"""
        return str(chat_id) == str(self.authorized_chat_id)
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🚀 Start command handler"""
        
        if not self.is_authorized(update.effective_chat.id):
            await update.message.reply_text("❌ Unauthorized access")
            return
        
        keyboard = [
            [InlineKeyboardButton("📊 System Status", callback_data="status")],
            [InlineKeyboardButton("🚀 Start Trading", callback_data="start_trading"),
             InlineKeyboardButton("🛑 Stop Trading", callback_data="stop_trading")],
            [InlineKeyboardButton("💰 Account Info", callback_data="account"),
             InlineKeyboardButton("📈 Performance", callback_data="performance")],
            [InlineKeyboardButton("🤖 AI Signals", callback_data="signals"),
             InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = (
            "🚀 *TRADINO UNSCHLAGBAR CONTROL PANEL*\n\n"
            "💡 Choose an option to control your trading system:\n\n"
            "📊 *System Status* - Check current status\n"
            "🚀 *Start/Stop* - Control trading\n"
            "💰 *Account* - View balance & positions\n"
            "📈 *Performance* - Trading statistics\n"
            "🤖 *AI Signals* - Latest AI predictions\n"
            "⚙️ *Settings* - Configuration options"
        )
        
        await update.message.reply_text(message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🔘 Handle button callbacks"""
        
        query = update.callback_query
        await query.answer()
        
        if not self.is_authorized(query.message.chat_id):
            await query.edit_message_text("❌ Unauthorized access")
            return
        
        action = query.data
        
        try:
            if action == "status":
                await self.send_system_status(query)
            elif action == "start_trading":
                await self.start_trading_system(query)
            elif action == "stop_trading":
                await self.stop_trading_system(query)
            elif action == "account":
                await self.send_account_info(query)
            elif action == "performance":
                await self.send_performance_info(query)
            elif action == "signals":
                await self.send_ai_signals(query)
            elif action == "settings":
                await self.send_settings_menu(query)
            elif action == "toggle_notifications":
                await self.toggle_notifications(query)
            elif action == "back_main":
                await self.back_to_main_menu(query)
                
        except Exception as e:
            logger.error(f"Error handling button {action}: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)}")
    
    async def send_system_status(self, query):
        """📊 Send system status"""
        
        if not self.trading_system:
            self.trading_system = initialize_real_tradino_system()
        
        if self.trading_system:
            status = self.trading_system.get_real_status()
            
            status_text = (
                f"📊 *TRADINO SYSTEM STATUS*\n\n"
                f"🔧 *Initialized:* {'✅' if status['initialized'] else '❌'}\n"
                f"🏃 *Running:* {'✅' if status['running'] else '❌'}\n"
                f"🤖 *AI Models:* {'✅' if status['ai_ready'] else '❌'}\n"
                f"📊 *Market Feed:* {'✅' if status['market_feed_connected'] else '❌'}\n"
                f"🏦 *Trading API:* {'✅' if status['trading_api_connected'] else '❌'}\n\n"
                f"⏰ *Uptime:* {status['session_stats']['uptime']}\n"
                f"🤖 *Signals Generated:* {status['session_stats']['signals_generated']}\n"
                f"📈 *Trades Executed:* {status['session_stats']['trades_executed']}\n"
                f"🎯 *Mode:* FUTURES TRADING\n"
                f"📱 *Last Update:* {datetime.now().strftime('%H:%M:%S')}"
            )
        else:
            status_text = "❌ *Trading system not initialized*"
        
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="status"),
                    InlineKeyboardButton("🏠 Main Menu", callback_data="back_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(status_text, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def start_trading_system(self, query):
        """🚀 Start trading system"""
        
        try:
            if not self.trading_system:
                self.trading_system = initialize_real_tradino_system()
            
            if not self.trading_system or not self.trading_system.is_initialized:
                await query.edit_message_text("❌ Trading system not ready!")
                return
            
            if self.trading_system.is_running:
                await query.edit_message_text("⚠️ Trading system already running!")
                return
            
            self.trading_system.start_real_trading()
            
            message = (
                "🚀 *TRADING SYSTEM STARTED!*\n\n"
                "✅ FUTURES trading is now active\n"
                "🤖 AI models are generating signals\n"
                "📊 Real-time market data connected\n"
                "💰 Ready to execute trades\n\n"
                "⚠️ *Monitor your positions carefully!*"
            )
            
            keyboard = [[InlineKeyboardButton("📊 Status", callback_data="status"),
                        InlineKeyboardButton("🛑 Stop", callback_data="stop_trading")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            await query.edit_message_text(f"❌ Error starting trading: {str(e)}")
    
    async def stop_trading_system(self, query):
        """🛑 Stop trading system"""
        
        try:
            if not self.trading_system or not self.trading_system.is_running:
                await query.edit_message_text("⚠️ Trading system not running!")
                return
            
            self.trading_system.stop_real_trading()
            
            message = (
                "🛑 *TRADING SYSTEM STOPPED*\n\n"
                "✅ All trading operations halted\n"
                "📊 Final session statistics saved\n"
                "💰 Positions remain open (if any)\n\n"
                "💡 *You can restart anytime*"
            )
            
            keyboard = [[InlineKeyboardButton("🚀 Start", callback_data="start_trading"),
                        InlineKeyboardButton("📊 Status", callback_data="status")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            await query.edit_message_text(f"❌ Error stopping trading: {str(e)}")
    
    async def send_account_info(self, query):
        """💰 Send account information"""
        
        try:
            if not self.trading_system:
                self.trading_system = initialize_real_tradino_system()
            
            if self.trading_system and self.trading_system.trading_api:
                api = self.trading_system.trading_api
                
                # Get balance
                balance = api.get_total_balance()
                free_balance = api.get_free_balance()
                
                message = (
                    f"💰 *ACCOUNT INFORMATION*\n\n"
                    f"💵 *Total Balance:* ${balance:.2f} USDT\n"
                    f"💸 *Available:* ${free_balance:.2f} USDT\n"
                    f"🔒 *Used:* ${balance - free_balance:.2f} USDT\n\n"
                    f"🎯 *Account Type:* {'SANDBOX' if api.sandbox else 'LIVE'}\n"
                    f"🏦 *Exchange:* Bitget Futures\n"
                    f"📱 *Last Update:* {datetime.now().strftime('%H:%M:%S')}"
                )
            else:
                message = "❌ *Account information not available*"
            
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="account"),
                        InlineKeyboardButton("🏠 Main Menu", callback_data="back_main")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            await query.edit_message_text(f"❌ Error getting account info: {str(e)}")
    
    async def send_performance_info(self, query):
        """📈 Send performance information"""
        
        try:
            if not self.trading_system:
                await query.edit_message_text("❌ Trading system not available")
                return
            
            stats = self.trading_system.session_stats
            
            message = (
                f"📈 *TRADING PERFORMANCE*\n\n"
                f"⏰ *Session Duration:* {stats['uptime']}\n"
                f"🤖 *AI Signals:* {stats['signals_generated']}\n"
                f"📈 *Trades Executed:* {stats['trades_executed']}\n"
                f"💰 *Total Trade Value:* ${sum(trade.get('value', 0) for trade in stats.get('real_trades', [])):.2f}\n\n"
            )
            
            if stats.get('real_trades'):
                message += "📊 *Recent Trades:*\n"
                for trade in stats['real_trades'][-3:]:  # Last 3 trades
                    timestamp = trade['timestamp'][:16].replace('T', ' ')
                    message += f"• {timestamp} | {trade['symbol']} {trade['side']} | ${trade['value']:.2f}\n"
            else:
                message += "📊 *No trades executed yet*\n"
            
            message += f"\n📱 *Updated:* {datetime.now().strftime('%H:%M:%S')}"
            
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="performance"),
                        InlineKeyboardButton("🏠 Main Menu", callback_data="back_main")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            await query.edit_message_text(f"❌ Error getting performance: {str(e)}")
    
    async def send_ai_signals(self, query):
        """🤖 Send latest AI signals"""
        
        try:
            if not self.trading_system:
                await query.edit_message_text("❌ Trading system not available")
                return
            
            signals = self.trading_system.session_stats.get('latest_signals', [])
            
            message = "🤖 *LATEST AI SIGNALS*\n\n"
            
            if signals:
                for signal in signals[-3:]:  # Last 3 signals
                    timestamp = signal['timestamp'][:16].replace('T', ' ')
                    confidence = signal['confidence']
                    action = signal['action'].upper()
                    symbol = signal['symbol']
                    
                    confidence_emoji = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
                    action_emoji = "📈" if action == "BUY" else "📉" if action == "SELL" else "⏸️"
                    
                    message += (
                        f"{action_emoji} *{symbol}* - {action}\n"
                        f"  {confidence_emoji} Confidence: {confidence:.1%}\n"
                        f"  ⏰ {timestamp}\n\n"
                    )
            else:
                message += "📊 *No signals generated yet*\n"
            
            message += f"📱 *Updated:* {datetime.now().strftime('%H:%M:%S')}"
            
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="signals"),
                        InlineKeyboardButton("🏠 Main Menu", callback_data="back_main")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            await query.edit_message_text(f"❌ Error getting signals: {str(e)}")
    
    async def send_settings_menu(self, query):
        """⚙️ Send settings menu"""
        
        message = (
            f"⚙️ *SETTINGS*\n\n"
            f"🔔 *Notifications:* {'✅ ON' if self.notifications_enabled else '❌ OFF'}\n"
            f"🎯 *Trading Mode:* FUTURES\n"
            f"💰 *Position Size:* 2% per trade\n"
            f"🛡️ *Stop Loss:* 1.5%\n"
            f"🎯 *Take Profit:* 3%\n"
            f"🤖 *Min Confidence:* 75%\n"
            f"📈 *Max Daily Trades:* 10\n\n"
            f"💡 *Contact admin to change trading parameters*"
        )
        
        keyboard = [
            [InlineKeyboardButton("🔔 Toggle Notifications", callback_data="toggle_notifications")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="back_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def toggle_notifications(self, query):
        """🔔 Toggle notifications"""
        
        self.notifications_enabled = not self.notifications_enabled
        status = "enabled" if self.notifications_enabled else "disabled"
        
        await query.edit_message_text(
            f"🔔 Notifications {status}!",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
                InlineKeyboardButton("🏠 Main Menu", callback_data="back_main")
            ]])
        )
    
    async def back_to_main_menu(self, query):
        """🏠 Back to main menu"""
        
        await self.start_command(query, None)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """❓ Help command"""
        
        if not self.is_authorized(update.effective_chat.id):
            await update.message.reply_text("❌ Unauthorized access")
            return
        
        help_text = (
            "📱 *TRADINO TELEGRAM CONTROLS*\n\n"
            "🚀 `/start` - Main control panel\n"
            "❓ `/help` - Show this help\n"
            "📊 `/status` - Quick system status\n"
            "💰 `/balance` - Account balance\n"
            "🤖 `/signals` - Latest AI signals\n\n"
            "💡 *Use the buttons for full control!*"
        )
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📊 Quick status command"""
        
        if not self.is_authorized(update.effective_chat.id):
            await update.message.reply_text("❌ Unauthorized access")
            return
        
        if not self.trading_system:
            self.trading_system = initialize_real_tradino_system()
        
        if self.trading_system:
            running_status = "🟢 RUNNING" if self.trading_system.is_running else "🔴 STOPPED"
            await update.message.reply_text(f"📊 *TRADINO STATUS:* {running_status}", parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ Trading system not available")
    
    async def send_notification(self, message: str):
        """📢 Send notification to authorized user"""
        
        if not self.notifications_enabled:
            return
        
        try:
            await self.application.bot.send_message(
                chat_id=self.authorized_chat_id,
                text=f"🤖 *TRADINO ALERT*\n\n{message}",
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def run(self):
        """🚀 Start the Telegram bot"""
        
        if not TELEGRAM_AVAILABLE:
            print("❌ Telegram library not available")
            return
        
        self.application = Application.builder().token(self.bot_token).build()
        
        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CallbackQueryHandler(self.button_handler))
        
        print("🚀 Starting Telegram bot...")
        print(f"📱 Bot ready for authorized user: {self.authorized_chat_id}")
        
        # Start the bot
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)

def main():
    """🚀 Main function"""
    
    try:
        bot = TradinoTelegramBot()
        bot.run()
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 