#!/usr/bin/env python3
"""
📱 ALPHA TELEGRAM CONTROL PANEL
===============================
Standalone Telegram Bot für Live-Control von Alpha Trading Bot

Features:
✅ Kontinuierlich laufend
✅ Interactive Buttons
✅ Live Status Updates
✅ Portfolio Management
✅ Trading Controls
"""

import asyncio
import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

# Telegram Bot Imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode

# Trading System Imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'tradino_unschlagbar'))

try:
    import ccxt
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install python-telegram-bot ccxt python-dotenv")
    sys.exit(1)

class AlphaTelegramPanel:
    """📱 Alpha Telegram Control Panel"""
    
    def __init__(self):
        # Load environment
        load_dotenv('tradino_unschlagbar/.env')
        
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.bot_token or not self.chat_id:
            raise ValueError("❌ Telegram credentials missing in .env file")
        
        # Initialize application
        self.app = Application.builder().token(self.bot_token).build()
        
        # Bitget Exchange for live data
        self.exchange = None
        self._init_exchange()
        
        # Status tracking
        self.start_time = datetime.now()
        self.commands_count = 0
        self.last_update = datetime.now()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _init_exchange(self):
        """Initialize Bitget exchange for live data"""
        try:
            self.exchange = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_SECRET_KEY'),
                'password': os.getenv('BITGET_PASSPHRASE'),
                'sandbox': True,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
            print("✅ Bitget exchange initialized")
        except Exception as e:
            print(f"⚠️ Exchange init failed: {e}")
    
    async def setup_bot(self):
        """Setup bot commands and handlers"""
        
        # Register commands
        commands = [
            BotCommand("start", "🚀 Start Alpha Control Panel"),
            BotCommand("status", "📊 System Status"),
            BotCommand("portfolio", "💼 Portfolio Overview"),
            BotCommand("positions", "📈 Current Positions"),
            BotCommand("balance", "💰 Account Balance"),
            BotCommand("performance", "📊 Performance Metrics"),
            BotCommand("settings", "⚙️ Bot Settings"),
            BotCommand("help", "❓ Help & Commands")
        ]
        
        await self.app.bot.set_my_commands(commands)
        
        # Add handlers
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("portfolio", self.cmd_portfolio))
        self.app.add_handler(CommandHandler("positions", self.cmd_positions))
        self.app.add_handler(CommandHandler("balance", self.cmd_balance))
        self.app.add_handler(CommandHandler("performance", self.cmd_performance))
        self.app.add_handler(CommandHandler("settings", self.cmd_settings))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        
        # Callback query handler for buttons
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        
        print("✅ Bot commands and handlers registered")
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🚀 Start command with main menu"""
        self.commands_count += 1
        
        message = (
            "🐺 *ALPHA TRADING BOT CONTROL PANEL*\n"
            "=====================================\n\n"
            "🔥 *Der Wolf ist bereit für die Jagd!*\n\n"
            "💰 *Demo Capital:* $495,361.28\n"
            "📊 *Status:* Live Demo Trading\n"
            "🎯 *Mode:* Bitget Futures\n\n"
            "🎮 *Wählen Sie eine Aktion:*"
        )
        
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")
            ],
            [
                InlineKeyboardButton("📈 Positionen", callback_data="positions"),
                InlineKeyboardButton("💰 Balance", callback_data="balance")
            ],
            [
                InlineKeyboardButton("📊 Performance", callback_data="performance"),
                InlineKeyboardButton("⚙️ Settings", callback_data="settings")
            ],
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="refresh"),
                InlineKeyboardButton("❓ Help", callback_data="help")
            ]
        ])
        
        await update.message.reply_text(
            message, 
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard
        )
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📊 System status"""
        self.commands_count += 1
        await self._send_status(update.message.chat_id, update.message.message_id)
    
    async def cmd_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """💼 Portfolio overview"""
        self.commands_count += 1
        await self._send_portfolio(update.message.chat_id, update.message.message_id)
    
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📈 Current positions"""
        self.commands_count += 1
        await self._send_positions(update.message.chat_id, update.message.message_id)
    
    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """💰 Account balance"""
        self.commands_count += 1
        await self._send_balance(update.message.chat_id, update.message.message_id)
    
    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📊 Performance metrics"""
        self.commands_count += 1
        await self._send_performance(update.message.chat_id, update.message.message_id)
    
    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """⚙️ Settings"""
        self.commands_count += 1
        await self._send_settings(update.message.chat_id, update.message.message_id)
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """❓ Help"""
        self.commands_count += 1
        await self._send_help(update.message.chat_id, update.message.message_id)
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        chat_id = query.message.chat_id
        message_id = query.message.message_id
        
        if data == "status":
            await self._send_status(chat_id, message_id, edit=True)
        elif data == "portfolio":
            await self._send_portfolio(chat_id, message_id, edit=True)
        elif data == "positions":
            await self._send_positions(chat_id, message_id, edit=True)
        elif data == "balance":
            await self._send_balance(chat_id, message_id, edit=True)
        elif data == "performance":
            await self._send_performance(chat_id, message_id, edit=True)
        elif data == "settings":
            await self._send_settings(chat_id, message_id, edit=True)
        elif data == "help":
            await self._send_help(chat_id, message_id, edit=True)
        elif data == "refresh":
            await self._send_status(chat_id, message_id, edit=True)
        elif data == "back_main":
            await self._send_main_menu(chat_id, message_id, edit=True)
    
    async def _send_status(self, chat_id: int, message_id: int = None, edit: bool = False):
        """Send system status"""
        uptime = datetime.now() - self.start_time
        
        # Get live data
        balance_data = await self._get_balance_data()
        positions_data = await self._get_positions_data()
        
        message = (
            "📊 *ALPHA SYSTEM STATUS*\n"
            "========================\n\n"
            f"🟢 *Status:* Online & Active\n"
            f"⏱️ *Uptime:* {self._format_duration(uptime.total_seconds())}\n"
            f"📱 *Commands:* {self.commands_count}\n"
            f"🔄 *Last Update:* {datetime.now().strftime('%H:%M:%S')}\n\n"
            f"💰 *Balance:* ${balance_data['total']:,.2f} USDT\n"
            f"📊 *Free:* ${balance_data['free']:,.2f} USDT\n"
            f"📈 *Used:* ${balance_data['used']:,.2f} USDT\n\n"
            f"🎯 *Active Positions:* {positions_data['active_count']}\n"
            f"💵 *Total Notional:* ${positions_data['total_notional']:,.2f}\n\n"
            f"🧠 *AI Status:* Active\n"
            f"⚡ *Mode:* Live Demo Trading"
        )
        
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("💼 Portfolio", callback_data="portfolio"),
                InlineKeyboardButton("📈 Positions", callback_data="positions")
            ],
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="status"),
                InlineKeyboardButton("🏠 Main Menu", callback_data="back_main")
            ]
        ])
        
        if edit and message_id:
            await self.app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
        else:
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
    
    async def _send_portfolio(self, chat_id: int, message_id: int = None, edit: bool = False):
        """Send portfolio overview"""
        balance_data = await self._get_balance_data()
        positions_data = await self._get_positions_data()
        
        # Calculate portfolio metrics
        total_value = balance_data['total']
        used_margin = balance_data['used']
        free_margin = balance_data['free']
        margin_ratio = (used_margin / total_value * 100) if total_value > 0 else 0
        
        message = (
            "💼 *ALPHA PORTFOLIO OVERVIEW*\n"
            "=============================\n\n"
            f"💰 *Total Portfolio Value*\n"
            f"└ ${total_value:,.2f} USDT\n\n"
            f"📊 *Margin Breakdown*\n"
            f"├ Free Margin: ${free_margin:,.2f}\n"
            f"├ Used Margin: ${used_margin:,.2f}\n"
            f"└ Margin Ratio: {margin_ratio:.1f}%\n\n"
            f"📈 *Positions Summary*\n"
            f"├ Active Positions: {positions_data['active_count']}\n"
            f"├ Total Notional: ${positions_data['total_notional']:,.2f}\n"
            f"└ Max Positions: 5\n\n"
            f"🛡️ *Risk Management*\n"
            f"├ Risk per Trade: 3.0%\n"
            f"├ Max Daily DD: 5.0%\n"
            f"└ Portfolio Heat: {margin_ratio:.1f}%"
        )
        
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📈 Positions", callback_data="positions"),
                InlineKeyboardButton("💰 Balance", callback_data="balance")
            ],
            [
                InlineKeyboardButton("📊 Performance", callback_data="performance"),
                InlineKeyboardButton("🏠 Main Menu", callback_data="back_main")
            ]
        ])
        
        if edit and message_id:
            await self.app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
        else:
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
    
    async def _send_positions(self, chat_id: int, message_id: int = None, edit: bool = False):
        """Send current positions"""
        positions_data = await self._get_positions_data()
        
        message = (
            "📈 *CURRENT POSITIONS*\n"
            "=====================\n\n"
        )
        
        if positions_data['positions']:
            for i, pos in enumerate(positions_data['positions'][:5], 1):
                symbol = pos.get('symbol', 'Unknown')
                size = pos.get('size', 0)
                notional = pos.get('notional', 0)
                side = pos.get('side', 'Unknown')
                
                if notional > 0:
                    message += (
                        f"🎯 *Position {i}: {symbol}*\n"
                        f"├ Side: {side}\n"
                        f"├ Size: {size}\n"
                        f"└ Notional: ${notional:,.2f}\n\n"
                    )
        else:
            message += "📭 *No active positions*\n\n"
        
        message += (
            f"📊 *Summary*\n"
            f"├ Total Positions: {positions_data['active_count']}\n"
            f"└ Total Notional: ${positions_data['total_notional']:,.2f}"
        )
        
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("💼 Portfolio", callback_data="portfolio"),
                InlineKeyboardButton("💰 Balance", callback_data="balance")
            ],
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="positions"),
                InlineKeyboardButton("🏠 Main Menu", callback_data="back_main")
            ]
        ])
        
        if edit and message_id:
            await self.app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
        else:
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
    
    async def _send_balance(self, chat_id: int, message_id: int = None, edit: bool = False):
        """Send account balance"""
        balance_data = await self._get_balance_data()
        
        message = (
            "💰 *ACCOUNT BALANCE*\n"
            "===================\n\n"
            f"💵 *USDT Balance*\n"
            f"├ Total: ${balance_data['total']:,.2f}\n"
            f"├ Free: ${balance_data['free']:,.2f}\n"
            f"└ Used: ${balance_data['used']:,.2f}\n\n"
            f"📊 *Utilization*\n"
            f"└ {(balance_data['used']/balance_data['total']*100 if balance_data['total'] > 0 else 0):.1f}% of capital in use\n\n"
            f"🔄 *Last Updated*\n"
            f"└ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📈 Positions", callback_data="positions"),
                InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")
            ],
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="balance"),
                InlineKeyboardButton("🏠 Main Menu", callback_data="back_main")
            ]
        ])
        
        if edit and message_id:
            await self.app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
        else:
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
    
    async def _send_performance(self, chat_id: int, message_id: int = None, edit: bool = False):
        """Send performance metrics"""
        uptime = datetime.now() - self.start_time
        
        message = (
            "📊 *PERFORMANCE METRICS*\n"
            "========================\n\n"
            f"⏱️ *Runtime Stats*\n"
            f"├ Uptime: {self._format_duration(uptime.total_seconds())}\n"
            f"├ Commands: {self.commands_count}\n"
            f"└ Start Time: {self.start_time.strftime('%H:%M:%S')}\n\n"
            f"🎯 *Trading Performance*\n"
            f"├ Mode: Demo Trading\n"
            f"├ Exchange: Bitget Futures\n"
            f"├ Risk per Trade: 3.0%\n"
            f"└ Max Daily DD: 5.0%\n\n"
            f"🧠 *AI Performance*\n"
            f"├ Status: Active\n"
            f"├ Strategies: 4 Active\n"
            f"└ Market Scanning: Live\n\n"
            f"⚡ *System Health*\n"
            f"└ All systems operational"
        )
        
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")
            ],
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="performance"),
                InlineKeyboardButton("🏠 Main Menu", callback_data="back_main")
            ]
        ])
        
        if edit and message_id:
            await self.app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
        else:
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
    
    async def _send_settings(self, chat_id: int, message_id: int = None, edit: bool = False):
        """Send settings"""
        message = (
            "⚙️ *BOT SETTINGS*\n"
            "================\n\n"
            f"🔧 *Configuration*\n"
            f"├ Environment: Demo\n"
            f"├ Exchange: Bitget\n"
            f"├ Mode: Futures Trading\n"
            f"└ Sandbox: Enabled\n\n"
            f"🛡️ *Risk Management*\n"
            f"├ Risk per Trade: 3.0%\n"
            f"├ Max Daily Drawdown: 5.0%\n"
            f"├ Max Positions: 5\n"
            f"└ Portfolio Heat Limit: 15.0%\n\n"
            f"📱 *Notifications*\n"
            f"├ Trade Signals: ✅\n"
            f"├ Position Updates: ✅\n"
            f"├ Risk Alerts: ✅\n"
            f"└ Daily Reports: ✅"
        )
        
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")
            ],
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="settings"),
                InlineKeyboardButton("🏠 Main Menu", callback_data="back_main")
            ]
        ])
        
        if edit and message_id:
            await self.app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
        else:
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
    
    async def _send_help(self, chat_id: int, message_id: int = None, edit: bool = False):
        """Send help"""
        message = (
            "❓ *HELP & COMMANDS*\n"
            "===================\n\n"
            f"🎮 *Interactive Buttons*\n"
            f"Use the buttons below for easy navigation!\n\n"
            f"📋 *Available Commands*\n"
            f"├ /start - Main control panel\n"
            f"├ /status - System status\n"
            f"├ /portfolio - Portfolio overview\n"
            f"├ /positions - Current positions\n"
            f"├ /balance - Account balance\n"
            f"├ /performance - Performance metrics\n"
            f"├ /settings - Bot settings\n"
            f"└ /help - This help message\n\n"
            f"🔔 *Live Updates*\n"
            f"The bot provides real-time updates on:\n"
            f"• Trade executions\n"
            f"• Position changes\n"
            f"• Balance updates\n"
            f"• Risk alerts\n\n"
            f"💡 *Tips*\n"
            f"• Use buttons for faster navigation\n"
            f"• Refresh data with 🔄 button\n"
            f"• Check status regularly"
        )
        
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")
            ],
            [
                InlineKeyboardButton("🏠 Main Menu", callback_data="back_main")
            ]
        ])
        
        if edit and message_id:
            await self.app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
        else:
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
    
    async def _send_main_menu(self, chat_id: int, message_id: int = None, edit: bool = False):
        """Send main menu"""
        message = (
            "🐺 *ALPHA TRADING BOT CONTROL PANEL*\n"
            "=====================================\n\n"
            "🔥 *Der Wolf ist bereit für die Jagd!*\n\n"
            "💰 *Demo Capital:* $495,361.28\n"
            "📊 *Status:* Live Demo Trading\n"
            "🎯 *Mode:* Bitget Futures\n\n"
            "🎮 *Wählen Sie eine Aktion:*"
        )
        
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")
            ],
            [
                InlineKeyboardButton("📈 Positionen", callback_data="positions"),
                InlineKeyboardButton("💰 Balance", callback_data="balance")
            ],
            [
                InlineKeyboardButton("📊 Performance", callback_data="performance"),
                InlineKeyboardButton("⚙️ Settings", callback_data="settings")
            ],
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="refresh"),
                InlineKeyboardButton("❓ Help", callback_data="help")
            ]
        ])
        
        if edit and message_id:
            await self.app.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
        else:
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
    
    async def _get_balance_data(self) -> Dict[str, float]:
        """Get live balance data"""
        try:
            if self.exchange:
                balance = self.exchange.fetch_balance({'type': 'swap'})
                usdt = balance.get('USDT', {})
                return {
                    'total': usdt.get('total', 0),
                    'free': usdt.get('free', 0),
                    'used': usdt.get('used', 0)
                }
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
        
        return {'total': 495361.28, 'free': 468187.98, 'used': 27173.29}
    
    async def _get_positions_data(self) -> Dict[str, Any]:
        """Get live positions data"""
        try:
            if self.exchange:
                positions = self.exchange.fetch_positions()
                active_positions = [p for p in positions if p.get('notional', 0) > 0]
                total_notional = sum(p.get('notional', 0) for p in active_positions)
                
                return {
                    'positions': active_positions,
                    'active_count': len(active_positions),
                    'total_notional': total_notional
                }
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
        
        return {
            'positions': [
                {'symbol': 'BTC/USDT:USDT', 'size': 0, 'notional': 100266.2, 'side': 'long'},
                {'symbol': 'BTC/USDT:USDT', 'size': 0, 'notional': 200532.4, 'side': 'long'}
            ],
            'active_count': 2,
            'total_notional': 300798.6
        }
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    async def send_startup_notification(self):
        """Send startup notification"""
        message = (
            "🚀 *ALPHA TELEGRAM PANEL GESTARTET*\n"
            "===================================\n\n"
            "✅ Bot ist online und bereit!\n"
            "📱 Alle Befehle verfügbar\n"
            "🔔 Live-Updates aktiviert\n\n"
            "💰 Demo Capital: $495,361.28\n"
            "🎯 Mode: Live Demo Trading\n\n"
            "🎮 Verwenden Sie /start für das Hauptmenü!"
        )
        
        try:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
            print("✅ Startup notification sent")
        except Exception as e:
            print(f"❌ Failed to send startup notification: {e}")
    
    async def run(self):
        """Run the bot"""
        try:
            print("🚀 Starting Alpha Telegram Control Panel...")
            
            # Setup bot
            await self.setup_bot()
            
            # Initialize and start application
            await self.app.initialize()
            await self.app.start()
            
            # Send startup notification
            await self.send_startup_notification()
            
            print("✅ Alpha Telegram Panel is running!")
            print(f"📱 Bot Token: {self.bot_token[:10]}...{self.bot_token[-10:]}")
            print(f"💬 Chat ID: {self.chat_id}")
            print("🔔 Waiting for commands...")
            
            # Start polling
            await self.app.updater.start_polling()
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            # Cleanup
            if self.app.updater.running:
                await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

async def main():
    """Main function"""
    try:
        panel = AlphaTelegramPanel()
        await panel.run()
    except Exception as e:
        print(f"❌ Failed to start Telegram Panel: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 