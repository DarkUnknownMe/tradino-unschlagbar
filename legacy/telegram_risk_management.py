#!/usr/bin/env python3
"""
��️ TELEGRAM ADVANCED RISK MANAGEMENT SYSTEM
============================================
Vollständige Risiko-Steuerung über Telegram mit allen wichtigen Parametern
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
    from dotenv import load_dotenv
    import ccxt
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install python-telegram-bot ccxt python-dotenv")
    sys.exit(1)

class TelegramRiskManager:
    """🛡️ Advanced Telegram Risk Management System"""
    
    def __init__(self):
        load_dotenv('tradino_unschlagbar/.env')
        
        # Telegram configuration
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.bot_token:
            raise ValueError("❌ TELEGRAM_BOT_TOKEN nicht gefunden!")
        
        # Risk management parameters
        self.risk_config = {
            'max_drawdown_percent': 10.0,      # Max 10% Drawdown
            'max_risk_per_trade': 2.0,         # Max 2% Risk pro Trade
            'max_daily_loss': 5.0,             # Max 5% Verlust pro Tag
            'max_weekly_loss': 15.0,           # Max 15% Verlust pro Woche
            'max_monthly_loss': 25.0,          # Max 25% Verlust pro Monat
            'max_leverage': 10,                # Max 10x Leverage
            'max_positions': 5,                # Max 5 gleichzeitige Positionen
            'max_position_size': 1000,         # Max $1000 pro Position
            'stop_loss_percent': 3.0,          # Standard 3% Stop Loss
            'take_profit_percent': 6.0,        # Standard 6% Take Profit
            'risk_reward_ratio': 2.0,          # Min 1:2 Risk/Reward
            'max_correlation': 0.7,            # Max 70% Korrelation zwischen Positionen
            'emergency_stop': False,           # Emergency Stop aktiviert
            'risk_level': 'MEDIUM',            # LOW, MEDIUM, HIGH
            'auto_reduce_size': True,          # Auto Size Reduction bei Verlusten
            'trailing_stop': True,             # Trailing Stop aktiviert
            'break_even_move': True,           # Break Even Move aktiviert
        }
        
        # Load existing config
        self._load_risk_config()
        
        # Initialize exchange for monitoring
        self.exchange = None
        self._init_exchange()
        
        # Risk monitoring data
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.active_positions = 0
        self.total_exposure = 0.0
        
        # Setup logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_risk_config(self):
        """Load existing risk configuration"""
        try:
            config_path = 'tradino_unschlagbar/config/risk_config.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                    self.risk_config.update(saved_config)
                    print("✅ Risk configuration loaded from file")
        except Exception as e:
            print(f"⚠️ Could not load risk config: {e}")
    
    def _save_risk_config(self):
        """Save current risk configuration"""
        try:
            os.makedirs('tradino_unschlagbar/config', exist_ok=True)
            config_path = 'tradino_unschlagbar/config/risk_config.json'
            
            config_to_save = self.risk_config.copy()
            config_to_save['last_updated'] = datetime.now().isoformat()
            
            with open(config_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            
            print("✅ Risk configuration saved")
            return True
        except Exception as e:
            print(f"❌ Could not save risk config: {e}")
            return False
    
    def _init_exchange(self):
        """Initialize exchange for monitoring"""
        try:
            self.exchange = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_SECRET_KEY'),
                'password': os.getenv('BITGET_PASSPHRASE'),
                'sandbox': True,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
            print("✅ Exchange initialized for risk monitoring")
        except Exception as e:
            print(f"⚠️ Exchange initialization failed: {e}")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command - Main risk management menu"""
        keyboard = [
            [
                InlineKeyboardButton("🛡️ Risk Settings", callback_data="risk_settings"),
                InlineKeyboardButton("📊 Risk Status", callback_data="risk_status")
            ],
            [
                InlineKeyboardButton("⚠️ Drawdown Limits", callback_data="drawdown_limits"),
                InlineKeyboardButton("💰 Position Limits", callback_data="position_limits")
            ],
            [
                InlineKeyboardButton("🎯 SL/TP Settings", callback_data="sl_tp_settings"),
                InlineKeyboardButton("⚡ Risk Presets", callback_data="risk_presets")
            ],
            [
                InlineKeyboardButton("🚨 Emergency Stop", callback_data="emergency_stop"),
                InlineKeyboardButton("🚀 START TRADING", callback_data="start_trading")
            ],
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="refresh_main")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""
🛡️ **ALPHA UNSCHLAGBAR - RISK MANAGEMENT**
==========================================

🎯 **Current Risk Level:** {self.risk_config['risk_level']}
📊 **Max Drawdown:** {self.risk_config['max_drawdown_percent']}%
💰 **Risk per Trade:** {self.risk_config['max_risk_per_trade']}%
🔥 **Max Leverage:** {self.risk_config['max_leverage']}x

⚠️ **Current Status:**
├ Emergency Stop: {'🔴 ON' if self.risk_config['emergency_stop'] else '🟢 OFF'}
├ Active Positions: {self.active_positions}/{self.risk_config['max_positions']}
├ Current Drawdown: {self.current_drawdown:.2f}%
└ Daily P&L: {self.daily_pnl:.2f}%

🎛️ **Use buttons below to configure risk parameters**
        """
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle all callback queries"""
        query = update.callback_query
        await query.answer()
        
        if query.data == "risk_settings":
            await self._show_risk_settings(query)
        elif query.data == "risk_status":
            await self._show_risk_status(query)
        elif query.data == "drawdown_limits":
            await self._show_drawdown_limits(query)
        elif query.data == "position_limits":
            await self._show_position_limits(query)
        elif query.data == "sl_tp_settings":
            await self._show_sl_tp_settings(query)
        elif query.data == "risk_presets":
            await self._show_risk_presets(query)
        elif query.data == "emergency_stop":
            await self._toggle_emergency_stop(query)
        elif query.data == "start_trading":
            await self._start_trading(query)
        elif query.data == "refresh_main":
            await self._refresh_main_menu(query)
        elif query.data.startswith("set_"):
            await self._handle_setting_change(query)
        elif query.data.startswith("preset_"):
            await self._apply_risk_preset(query)
    
    async def _show_risk_settings(self, query):
        """Show main risk settings menu"""
        keyboard = [
            [
                InlineKeyboardButton(f"Max Drawdown: {self.risk_config['max_drawdown_percent']}%", 
                                   callback_data="set_max_drawdown"),
                InlineKeyboardButton(f"Risk/Trade: {self.risk_config['max_risk_per_trade']}%", 
                                   callback_data="set_risk_per_trade")
            ],
            [
                InlineKeyboardButton(f"Max Leverage: {self.risk_config['max_leverage']}x", 
                                   callback_data="set_max_leverage"),
                InlineKeyboardButton(f"Max Positions: {self.risk_config['max_positions']}", 
                                   callback_data="set_max_positions")
            ],
            [
                InlineKeyboardButton(f"Daily Loss: {self.risk_config['max_daily_loss']}%", 
                                   callback_data="set_daily_loss"),
                InlineKeyboardButton(f"Weekly Loss: {self.risk_config['max_weekly_loss']}%", 
                                   callback_data="set_weekly_loss")
            ],
            [
                InlineKeyboardButton(f"Position Size: ${self.risk_config['max_position_size']}", 
                                   callback_data="set_position_size"),
                InlineKeyboardButton(f"R:R Ratio: 1:{self.risk_config['risk_reward_ratio']}", 
                                   callback_data="set_risk_reward")
            ],
            [
                InlineKeyboardButton("💾 Save Settings", callback_data="save_settings"),
                InlineKeyboardButton("🔙 Back", callback_data="refresh_main")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""
🛡️ **RISK SETTINGS CONFIGURATION**
==================================

🎯 **Core Risk Parameters:**
├ Max Drawdown: {self.risk_config['max_drawdown_percent']}%
├ Risk per Trade: {self.risk_config['max_risk_per_trade']}%
├ Max Leverage: {self.risk_config['max_leverage']}x
└ Max Positions: {self.risk_config['max_positions']}

📊 **Loss Limits:**
├ Daily Loss Limit: {self.risk_config['max_daily_loss']}%
├ Weekly Loss Limit: {self.risk_config['max_weekly_loss']}%
└ Monthly Loss Limit: {self.risk_config['max_monthly_loss']}%

💰 **Position Management:**
├ Max Position Size: ${self.risk_config['max_position_size']}
├ Risk/Reward Ratio: 1:{self.risk_config['risk_reward_ratio']}
└ Max Correlation: {self.risk_config['max_correlation']}

**Tap any parameter to modify it**
        """
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def _show_risk_status(self, query):
        """Show current risk status and monitoring"""
        # Update current status
        await self._update_risk_status()
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Detailed Report", callback_data="detailed_report"),
                InlineKeyboardButton("⚠️ Risk Alerts", callback_data="risk_alerts")
            ],
            [
                InlineKeyboardButton("🔄 Update Status", callback_data="risk_status"),
                InlineKeyboardButton("�� Back", callback_data="refresh_main")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Risk level indicators
        drawdown_status = "🟢 SAFE" if self.current_drawdown < self.risk_config['max_drawdown_percent'] * 0.5 else "🟡 CAUTION" if self.current_drawdown < self.risk_config['max_drawdown_percent'] * 0.8 else "🔴 DANGER"
        
        daily_status = "🟢 SAFE" if abs(self.daily_pnl) < self.risk_config['max_daily_loss'] * 0.5 else "🟡 CAUTION" if abs(self.daily_pnl) < self.risk_config['max_daily_loss'] * 0.8 else "🔴 DANGER"
        
        position_status = "🟢 SAFE" if self.active_positions < self.risk_config['max_positions'] * 0.8 else "🟡 FULL" if self.active_positions < self.risk_config['max_positions'] else "🔴 LIMIT"
        
        message = f"""
📊 **REAL-TIME RISK STATUS**
===========================

⚠️ **Risk Monitoring:**
├ Current Drawdown: {self.current_drawdown:.2f}% {drawdown_status}
├ Daily P&L: {self.daily_pnl:+.2f}% {daily_status}
├ Weekly P&L: {self.weekly_pnl:+.2f}%
└ Monthly P&L: {self.monthly_pnl:+.2f}%

📈 **Position Status:**
├ Active Positions: {self.active_positions}/{self.risk_config['max_positions']} {position_status}
├ Total Exposure: ${self.total_exposure:,.2f}
├ Average Leverage: {self.total_exposure / max(1, self.active_positions):.1f}x
└ Risk Level: {self.risk_config['risk_level']}

🛡️ **Safety Systems:**
├ Emergency Stop: {'�� ACTIVE' if self.risk_config['emergency_stop'] else '🟢 INACTIVE'}
├ Auto Size Reduction: {'✅ ON' if self.risk_config['auto_reduce_size'] else '❌ OFF'}
├ Trailing Stop: {'✅ ON' if self.risk_config['trailing_stop'] else '❌ OFF'}
└ Break Even Move: {'✅ ON' if self.risk_config['break_even_move'] else '❌ OFF'}

🕐 **Last Update:** {datetime.now().strftime('%H:%M:%S')}
        """
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def _show_drawdown_limits(self, query):
        """Show drawdown and loss limit settings"""
        keyboard = [
            [
                InlineKeyboardButton("📉 Max DD: 5%", callback_data="set_drawdown_5"),
                InlineKeyboardButton("📉 Max DD: 10%", callback_data="set_drawdown_10"),
                InlineKeyboardButton("📉 Max DD: 15%", callback_data="set_drawdown_15")
            ],
            [
                InlineKeyboardButton("📅 Daily: 3%", callback_data="set_daily_3"),
                InlineKeyboardButton("📅 Daily: 5%", callback_data="set_daily_5"),
                InlineKeyboardButton("📅 Daily: 8%", callback_data="set_daily_8")
            ],
            [
                InlineKeyboardButton("📊 Weekly: 10%", callback_data="set_weekly_10"),
                InlineKeyboardButton("📊 Weekly: 15%", callback_data="set_weekly_15"),
                InlineKeyboardButton("📊 Weekly: 20%", callback_data="set_weekly_20")
            ],
            [
                InlineKeyboardButton("🔙 Back", callback_data="risk_settings")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""
📉 **DRAWDOWN & LOSS LIMITS**
============================

🎯 **Current Settings:**
├ Max Drawdown: {self.risk_config['max_drawdown_percent']}%
├ Max Daily Loss: {self.risk_config['max_daily_loss']}%
├ Max Weekly Loss: {self.risk_config['max_weekly_loss']}%
└ Max Monthly Loss: {self.risk_config['max_monthly_loss']}%

⚠️ **Current Status:**
├ Current Drawdown: {self.current_drawdown:.2f}%
├ Daily P&L: {self.daily_pnl:+.2f}%
├ Weekly P&L: {self.weekly_pnl:+.2f}%
└ Monthly P&L: {self.monthly_pnl:+.2f}%

**Select new limits from buttons above**
        """
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def _show_position_limits(self, query):
        """Show position and size limit settings"""
        keyboard = [
            [
                InlineKeyboardButton("📊 Max Pos: 3", callback_data="set_max_pos_3"),
                InlineKeyboardButton("📊 Max Pos: 5", callback_data="set_max_pos_5"),
                InlineKeyboardButton("📊 Max Pos: 8", callback_data="set_max_pos_8")
            ],
            [
                InlineKeyboardButton("💰 Size: $500", callback_data="set_size_500"),
                InlineKeyboardButton("💰 Size: $1000", callback_data="set_size_1000"),
                InlineKeyboardButton("💰 Size: $2000", callback_data="set_size_2000")
            ],
            [
                InlineKeyboardButton("⚡ Lev: 5x", callback_data="set_lev_5"),
                InlineKeyboardButton("⚡ Lev: 10x", callback_data="set_lev_10"),
                InlineKeyboardButton("⚡ Lev: 20x", callback_data="set_lev_20")
            ],
            [
                InlineKeyboardButton("🔙 Back", callback_data="risk_settings")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""
💰 **POSITION & SIZE LIMITS**
============================

🎯 **Current Settings:**
├ Max Positions: {self.risk_config['max_positions']}
├ Max Position Size: ${self.risk_config['max_position_size']}
├ Max Leverage: {self.risk_config['max_leverage']}x
└ Risk per Trade: {self.risk_config['max_risk_per_trade']}%

📊 **Current Status:**
├ Active Positions: {self.active_positions}
├ Total Exposure: ${self.total_exposure:,.2f}
├ Average Size: ${self.total_exposure / max(1, self.active_positions):,.2f}
└ Position Utilization: {(self.active_positions / self.risk_config['max_positions']) * 100:.1f}%

**Select new limits from buttons above**
        """
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def _show_sl_tp_settings(self, query):
        """Show Stop Loss and Take Profit settings"""
        keyboard = [
            [
                InlineKeyboardButton(f"🛡️ SL: {self.risk_config['stop_loss_percent']}%", callback_data="set_sl"),
                InlineKeyboardButton(f"🎯 TP: {self.risk_config['take_profit_percent']}%", callback_data="set_tp")
            ],
            [
                InlineKeyboardButton("🛡️ SL: 2%", callback_data="set_sl_2"),
                InlineKeyboardButton("🛡️ SL: 3%", callback_data="set_sl_3"),
                InlineKeyboardButton("🛡️ SL: 5%", callback_data="set_sl_5")
            ],
            [
                InlineKeyboardButton("🎯 TP: 4%", callback_data="set_tp_4"),
                InlineKeyboardButton("🎯 TP: 6%", callback_data="set_tp_6"),
                InlineKeyboardButton("🎯 TP: 8%", callback_data="set_tp_8")
            ],
            [
                InlineKeyboardButton(f"Trailing: {'✅' if self.risk_config['trailing_stop'] else '❌'}", 
                                   callback_data="toggle_trailing"),
                InlineKeyboardButton(f"Break Even: {'✅' if self.risk_config['break_even_move'] else '❌'}", 
                                   callback_data="toggle_break_even")
            ],
            [
                InlineKeyboardButton("🔙 Back", callback_data="risk_settings")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""
🎯 **STOP LOSS & TAKE PROFIT SETTINGS**
======================================

🛡️ **Current Settings:**
├ Stop Loss: {self.risk_config['stop_loss_percent']}%
├ Take Profit: {self.risk_config['take_profit_percent']}%
├ Risk/Reward Ratio: 1:{self.risk_config['risk_reward_ratio']}
└ Max Correlation: {self.risk_config['max_correlation']}

⚙️ **Advanced Features:**
├ Trailing Stop: {'✅ ENABLED' if self.risk_config['trailing_stop'] else '❌ DISABLED'}
├ Break Even Move: {'✅ ENABLED' if self.risk_config['break_even_move'] else '❌ DISABLED'}
├ Auto Size Reduction: {'✅ ENABLED' if self.risk_config['auto_reduce_size'] else '❌ DISABLED'}
└ Dynamic SL/TP: Based on volatility

**Configure your SL/TP strategy above**
        """
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def _show_risk_presets(self, query):
        """Show risk management presets"""
        keyboard = [
            [
                InlineKeyboardButton("🟢 CONSERVATIVE", callback_data="preset_conservative"),
                InlineKeyboardButton("🟡 MODERATE", callback_data="preset_moderate"),
                InlineKeyboardButton("🔴 AGGRESSIVE", callback_data="preset_aggressive")
            ],
            [
                InlineKeyboardButton("🛡️ SAFE MODE", callback_data="preset_safe"),
                InlineKeyboardButton("⚡ SCALPING", callback_data="preset_scalping"),
                InlineKeyboardButton("📈 SWING", callback_data="preset_swing")
            ],
            [
                InlineKeyboardButton("🔙 Back", callback_data="refresh_main")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""
⚡ **RISK MANAGEMENT PRESETS**
=============================

🟢 **CONSERVATIVE:**
├ Max Drawdown: 5%
├ Risk per Trade: 1%
├ Max Leverage: 5x
└ Max Positions: 3

🟡 **MODERATE:**
├ Max Drawdown: 10%
├ Risk per Trade: 2%
├ Max Leverage: 10x
└ Max Positions: 5

🔴 **AGGRESSIVE:**
├ Max Drawdown: 15%
├ Risk per Trade: 3%
├ Max Leverage: 20x
└ Max Positions: 8

🛡️ **SAFE MODE:**
├ Max Drawdown: 3%
├ Risk per Trade: 0.5%
├ Max Leverage: 3x
└ Max Positions: 2

**Current Preset:** {self.risk_config['risk_level']}
        """
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def _toggle_emergency_stop(self, query):
        """Toggle emergency stop"""
        self.risk_config['emergency_stop'] = not self.risk_config['emergency_stop']
        self._save_risk_config()
        
        status = "🔴 ACTIVATED" if self.risk_config['emergency_stop'] else "🟢 DEACTIVATED"
        
        await query.answer(f"Emergency Stop {status}")
        await self._refresh_main_menu(query)
    
    async def _start_trading(self, query):
        """Start trading with current risk settings"""
        if self.risk_config['emergency_stop']:
            await query.answer("❌ Cannot start - Emergency Stop is active!")
            return
        
        # Save current configuration
        self._save_risk_config()
        
        keyboard = [
            [
                InlineKeyboardButton("✅ CONFIRM START", callback_data="confirm_start"),
                InlineKeyboardButton("❌ Cancel", callback_data="refresh_main")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""
🚀 **READY TO START TRADING**
============================

🛡️ **Risk Configuration Summary:**
├ Max Drawdown: {self.risk_config['max_drawdown_percent']}%
├ Risk per Trade: {self.risk_config['max_risk_per_trade']}%
├ Max Leverage: {self.risk_config['max_leverage']}x
├ Max Positions: {self.risk_config['max_positions']}
├ Stop Loss: {self.risk_config['stop_loss_percent']}%
├ Take Profit: {self.risk_config['take_profit_percent']}%
└ Risk Level: {self.risk_config['risk_level']}

⚠️ **Safety Features:**
├ Emergency Stop: {'🔴 ON' if self.risk_config['emergency_stop'] else '🟢 OFF'}
├ Auto Size Reduction: {'✅' if self.risk_config['auto_reduce_size'] else '❌'}
├ Trailing Stop: {'✅' if self.risk_config['trailing_stop'] else '❌'}
└ Break Even Move: {'✅' if self.risk_config['break_even_move'] else '❌'}

**⚠️ CONFIRM TO START LIVE TRADING ⚠️**
        """
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def _handle_setting_change(self, query):
        """Handle setting changes from callback data"""
        setting = query.data.replace("set_", "")
        
        # Handle different setting types
        if setting.startswith("drawdown_"):
            value = int(setting.split("_")[1])
            self.risk_config['max_drawdown_percent'] = value
            await query.answer(f"Max Drawdown set to {value}%")
            
        elif setting.startswith("daily_"):
            value = int(setting.split("_")[1])
            self.risk_config['max_daily_loss'] = value
            await query.answer(f"Daily Loss Limit set to {value}%")
            
        elif setting.startswith("weekly_"):
            value = int(setting.split("_")[1])
            self.risk_config['max_weekly_loss'] = value
            await query.answer(f"Weekly Loss Limit set to {value}%")
            
        elif setting.startswith("max_pos_"):
            value = int(setting.split("_")[2])
            self.risk_config['max_positions'] = value
            await query.answer(f"Max Positions set to {value}")
            
        elif setting.startswith("size_"):
            value = int(setting.split("_")[1])
            self.risk_config['max_position_size'] = value
            await query.answer(f"Max Position Size set to ${value}")
            
        elif setting.startswith("lev_"):
            value = int(setting.split("_")[1])
            self.risk_config['max_leverage'] = value
            await query.answer(f"Max Leverage set to {value}x")
            
        elif setting.startswith("sl_"):
            value = int(setting.split("_")[1])
            self.risk_config['stop_loss_percent'] = value
            await query.answer(f"Stop Loss set to {value}%")
            
        elif setting.startswith("tp_"):
            value = int(setting.split("_")[1])
            self.risk_config['take_profit_percent'] = value
            await query.answer(f"Take Profit set to {value}%")
            
        elif setting == "toggle_trailing":
            self.risk_config['trailing_stop'] = not self.risk_config['trailing_stop']
            status = "enabled" if self.risk_config['trailing_stop'] else "disabled"
            await query.answer(f"Trailing Stop {status}")
            
        elif setting == "toggle_break_even":
            self.risk_config['break_even_move'] = not self.risk_config['break_even_move']
            status = "enabled" if self.risk_config['break_even_move'] else "disabled"
            await query.answer(f"Break Even Move {status}")
        
        # Save configuration
        self._save_risk_config()
        
        # Refresh the current menu
        if "drawdown" in setting or "daily" in setting or "weekly" in setting:
            await self._show_drawdown_limits(query)
        elif "pos" in setting or "size" in setting or "lev" in setting:
            await self._show_position_limits(query)
        elif "sl" in setting or "tp" in setting or "trailing" in setting or "break_even" in setting:
            await self._show_sl_tp_settings(query)
        else:
            await self._show_risk_settings(query)
    
    async def _apply_risk_preset(self, query):
        """Apply risk management preset"""
        preset = query.data.replace("preset_", "")
        
        if preset == "conservative":
            self.risk_config.update({
                'max_drawdown_percent': 5.0,
                'max_risk_per_trade': 1.0,
                'max_leverage': 5,
                'max_positions': 3,
                'max_daily_loss': 3.0,
                'stop_loss_percent': 2.0,
                'take_profit_percent': 4.0,
                'risk_level': 'CONSERVATIVE'
            })
        elif preset == "moderate":
            self.risk_config.update({
                'max_drawdown_percent': 10.0,
                'max_risk_per_trade': 2.0,
                'max_leverage': 10,
                'max_positions': 5,
                'max_daily_loss': 5.0,
                'stop_loss_percent': 3.0,
                'take_profit_percent': 6.0,
                'risk_level': 'MODERATE'
            })
        elif preset == "aggressive":
            self.risk_config.update({
                'max_drawdown_percent': 15.0,
                'max_risk_per_trade': 3.0,
                'max_leverage': 20,
                'max_positions': 8,
                'max_daily_loss': 8.0,
                'stop_loss_percent': 4.0,
                'take_profit_percent': 8.0,
                'risk_level': 'AGGRESSIVE'
            })
        elif preset == "safe":
            self.risk_config.update({
                'max_drawdown_percent': 3.0,
                'max_risk_per_trade': 0.5,
                'max_leverage': 3,
                'max_positions': 2,
                'max_daily_loss': 2.0,
                'stop_loss_percent': 1.5,
                'take_profit_percent': 3.0,
                'risk_level': 'SAFE'
            })
        elif preset == "scalping":
            self.risk_config.update({
                'max_drawdown_percent': 8.0,
                'max_risk_per_trade': 1.5,
                'max_leverage': 15,
                'max_positions': 10,
                'max_daily_loss': 6.0,
                'stop_loss_percent': 1.0,
                'take_profit_percent': 2.0,
                'risk_level': 'SCALPING'
            })
        elif preset == "swing":
            self.risk_config.update({
                'max_drawdown_percent': 12.0,
                'max_risk_per_trade': 2.5,
                'max_leverage': 8,
                'max_positions': 4,
                'max_daily_loss': 4.0,
                'stop_loss_percent': 5.0,
                'take_profit_percent': 10.0,
                'risk_level': 'SWING'
            })
        
        self._save_risk_config()
        await query.answer(f"Applied {preset.upper()} preset")
        await self._refresh_main_menu(query)
    
    async def _refresh_main_menu(self, query):
        """Refresh main menu"""
        # Update status first
        await self._update_risk_status()
        
        keyboard = [
            [
                InlineKeyboardButton("🛡️ Risk Settings", callback_data="risk_settings"),
                InlineKeyboardButton("📊 Risk Status", callback_data="risk_status")
            ],
            [
                InlineKeyboardButton("⚠️ Drawdown Limits", callback_data="drawdown_limits"),
                InlineKeyboardButton("💰 Position Limits", callback_data="position_limits")
            ],
            [
                InlineKeyboardButton("🎯 SL/TP Settings", callback_data="sl_tp_settings"),
                InlineKeyboardButton("⚡ Risk Presets", callback_data="risk_presets")
            ],
            [
                InlineKeyboardButton("🚨 Emergency Stop", callback_data="emergency_stop"),
                InlineKeyboardButton("🚀 START TRADING", callback_data="start_trading")
            ],
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="refresh_main")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""
🛡️ **ALPHA UNSCHLAGBAR - RISK MANAGEMENT**
==========================================

🎯 **Current Risk Level:** {self.risk_config['risk_level']}
📊 **Max Drawdown:** {self.risk_config['max_drawdown_percent']}%
💰 **Risk per Trade:** {self.risk_config['max_risk_per_trade']}%
🔥 **Max Leverage:** {self.risk_config['max_leverage']}x

⚠️ **Current Status:**
├ Emergency Stop: {'🔴 ON' if self.risk_config['emergency_stop'] else '🟢 OFF'}
├ Active Positions: {self.active_positions}/{self.risk_config['max_positions']}
├ Current Drawdown: {self.current_drawdown:.2f}%
└ Daily P&L: {self.daily_pnl:.2f}%

🎛️ **Use buttons below to configure risk parameters**
        """
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def _update_risk_status(self):
        """Update current risk status from exchange"""
        try:
            if self.exchange:
                # Get current positions
                positions = self.exchange.fetch_positions()
                active_positions = [p for p in positions if abs(p.get('size', 0)) > 0.0001]
                
                self.active_positions = len(active_positions)
                self.total_exposure = sum(abs(p.get('notional', 0)) for p in active_positions)
                
                # Get balance for drawdown calculation
                balance = self.exchange.fetch_balance()
                current_balance = balance.get('USDT', {}).get('total', 0)
                
                # Calculate drawdown (simplified)
                initial_balance = 500000  # Demo account starting balance
                self.current_drawdown = max(0, ((initial_balance - current_balance) / initial_balance) * 100)
                
                # Calculate P&L (simplified - would need historical data for accurate calculation)
                self.daily_pnl = sum(p.get('unrealizedPnl', 0) for p in active_positions) / current_balance * 100 if current_balance > 0 else 0
                
        except Exception as e:
            self.logger.error(f"Failed to update risk status: {e}")
    
    def run(self):
        """Run the telegram risk management bot"""
        if not self.bot_token:
            print("❌ TELEGRAM_BOT_TOKEN not found!")
            return
        
        print("🛡️ Starting Telegram Risk Management System...")
        print(f"📊 Current Risk Level: {self.risk_config['risk_level']}")
        print(f"⚠️ Max Drawdown: {self.risk_config['max_drawdown_percent']}%")
        print(f"💰 Risk per Trade: {self.risk_config['max_risk_per_trade']}%")
        print("🎛️ Use /start in Telegram to configure risk parameters")
        
        # Create application
        application = Application.builder().token(self.bot_token).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Run the bot
        application.run_polling(allowed_updates=Update.ALL_TYPES)

def main():
    """Main function"""
    print("🛡️ TELEGRAM ADVANCED RISK MANAGEMENT SYSTEM")
    print("============================================")
    print()
    print("🎯 Features:")
    print("   ├ Max Drawdown Control")
    print("   ├ Position Size Limits")
    print("   ├ Risk per Trade Management")
    print("   ├ Stop Loss / Take Profit Settings")
    print("   ├ Emergency Stop Function")
    print("   ├ Risk Presets (Conservative/Moderate/Aggressive)")
    print("   └ Real-time Risk Monitoring")
    print()
    
    try:
        risk_manager = TelegramRiskManager()
        risk_manager.run()
    except KeyboardInterrupt:
        print("\n🛑 Risk Management System stopped by user")
    except Exception as e:
        print(f"❌ Risk Management System failed: {e}")

if __name__ == "__main__":
    main()
