#!/usr/bin/env python3
"""
📱 RISK MANAGEMENT TELEGRAM PANEL
Telegram Bot Interface für Live Risk Control und Monitoring
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Telegram Integration
try:
    import telegram
    from telegram import Update, Bot
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

# Add project path
sys.path.append('/root/tradino')

class RiskTelegramPanel:
    """📱 Telegram Panel for Risk Management"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = None
        self.application = None
        self.is_running = False
        
        if TELEGRAM_AVAILABLE:
            self.initialize_bot()
        else:
            print("❌ Telegram not available")
    
    def initialize_bot(self):
        """🤖 Initialize Telegram Bot"""
        try:
            self.bot = Bot(token=self.bot_token)
            self.application = Application.builder().token(self.bot_token).build()
            self.register_handlers()
            print("✅ Risk Management Telegram Bot initialized")
        except Exception as e:
            print(f"❌ Bot initialization failed: {e}")
    
    def register_handlers(self):
        """📋 Register command handlers"""
        self.application.add_handler(CommandHandler("risk_status", self.risk_status))
        self.application.add_handler(CommandHandler("emergency_stop", self.emergency_stop))
        self.application.add_handler(CommandHandler("help", self.help_command))
        print("✅ Command handlers registered")
    
    async def risk_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📊 Show risk status"""
        try:
            from advanced_risk_management import get_advanced_risk_manager
            risk_manager = get_advanced_risk_manager()
            
            if not risk_manager:
                await update.message.reply_text("❌ Risk Manager not available")
                return
            
            dashboard = risk_manager.get_risk_dashboard()
            
            message = f"📊 RISK STATUS\n\n"
            message += f"Risk Level: {dashboard['risk_level'].upper()}\n"
            message += f"Balance: ${dashboard['portfolio_state']['total_balance']:.2f}\n"
            message += f"Exposure: {dashboard['limits_usage']['exposure_usage']:.1%}\n"
            message += f"Daily PnL: ${dashboard['portfolio_state']['realized_pnl_daily']:.2f}\n"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def emergency_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🚨 Trigger emergency stop"""
        try:
            from advanced_risk_management import get_advanced_risk_manager
            risk_manager = get_advanced_risk_manager()
            
            if not risk_manager:
                await update.message.reply_text("❌ Risk Manager not available")
                return
            
            result = risk_manager.trigger_emergency_stop()
            
            message = f"🚨 EMERGENCY STOP TRIGGERED!\n\n"
            message += f"All trading has been halted.\n"
            message += f"Manual intervention required."
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """❓ Help command"""
        message = f"📱 TRADINO RISK MANAGEMENT BOT\n\n"
        message += f"/risk_status - Risk overview\n"
        message += f"/emergency_stop - Stop all trading\n"
        message += f"/help - Show this help"
        
        await update.message.reply_text(message)
    
    def start_bot(self):
        """🚀 Start the Telegram bot"""
        if not self.application:
            print("❌ Bot not initialized")
            return
        
        try:
            print("🚀 Starting Risk Management Telegram Bot...")
            self.is_running = True
            self.application.run_polling()
        except Exception as e:
            print(f"❌ Bot error: {e}")
        finally:
            self.is_running = False

# Global instance
risk_telegram_panel = None

def initialize_risk_telegram_panel():
    """📱 Initialize Risk Telegram Panel"""
    global risk_telegram_panel
    
    if not TELEGRAM_AVAILABLE:
        return None
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("❌ Telegram credentials not found")
        return None
    
    try:
        risk_telegram_panel = RiskTelegramPanel(bot_token, chat_id)
        return risk_telegram_panel
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return None

if __name__ == "__main__":
    panel = initialize_risk_telegram_panel()
    if panel:
        panel.start_bot()
