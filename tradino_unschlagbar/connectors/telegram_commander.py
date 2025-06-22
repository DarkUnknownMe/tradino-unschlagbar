"""
📱 TRADINO UNSCHLAGBAR - Telegram Commander
Vollständige Telegram Bot Integration mit Rich Notifications

Author: AI Trading Systems
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode

from models.trade_models import Order, Trade
from models.portfolio_models import Position, Portfolio
from models.signal_models import AISignal
from utils.logger_pro import setup_logger
from utils.config_manager import ConfigManager
from utils.helpers import format_currency, format_percentage, generate_id

logger = setup_logger("TelegramCommander")


class TelegramCommander:
    """📱 Professional Telegram Bot Controller"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.app: Optional[Application] = None
        self.chat_id = config.get('telegram.chat_id')
        self.is_running = False
        
        # Notification Settings (VOLLSTÄNDIG)
        self.notifications = config.get('telegram.notifications', {})
        
        # Command Callbacks
        self.command_callbacks: Dict[str, Callable] = {}
        
        # Message Templates
        self.templates = {
            'trade_signal': self._template_trade_signal,
            'trade_execution': self._template_trade_execution,
            'position_update': self._template_position_update,
            'pnl_update': self._template_pnl_update,
            'risk_alert': self._template_risk_alert,
            'system_status': self._template_system_status,
            'daily_report': self._template_daily_report,
            'emergency_alert': self._template_emergency_alert
        }
        
        # Statistics Tracking
        self.messages_sent = 0
        self.commands_executed = 0
        self.last_activity = datetime.utcnow()
    
    async def initialize(self) -> bool:
        """🔥 Telegram Bot initialisieren"""
        try:
            logger.info("📱 Telegram Bot wird initialisiert...")
            
            bot_token = self.config.get('telegram.bot_token')
            if not bot_token:
                logger.error("❌ Telegram Bot Token fehlt")
                return False
            
            # Application erstellen
            self.app = Application.builder().token(bot_token).build()
            
            # Command Handlers registrieren
            await self._register_handlers()
            
            # Bot starten
            await self.app.initialize()
            await self.app.start()
            
            # Startup-Nachricht senden
            await self.send_startup_message()
            
            self.is_running = True
            logger.success("✅ Telegram Bot erfolgreich gestartet")
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram Bot Initialisierung fehlgeschlagen: {e}")
            return False
    
    async def _register_handlers(self):
        """🔧 Command Handlers registrieren"""
        try:
            # Command Handlers
            commands = [
                ('start', self._cmd_start),
                ('stop', self._cmd_stop),
                ('status', self._cmd_status),
                ('portfolio', self._cmd_portfolio),
                ('performance', self._cmd_performance),
                ('positions', self._cmd_positions),
                ('settings', self._cmd_settings),
                ('help', self._cmd_help)
            ]
            
            for command, handler in commands:
                self.app.add_handler(CommandHandler(command, handler))
            
            # Callback Query Handler für Inline Keyboards
            self.app.add_handler(CallbackQueryHandler(self._handle_callback_query))
            
            logger.info(f"🔧 {len(commands)} Command Handlers registriert")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Handler-Registrierung: {e}")
    
    # ==================== COMMAND HANDLERS ====================
    
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🚀 /start Command"""
        try:
            self.commands_executed += 1
            
            message = (
                "🚀 *TRADINO UNSCHLAGBAR GESTARTET*\n\n"
                "💪 Der ultimative AI-Trading Bot ist bereit!\n\n"
                "📊 *Verfügbare Befehle:*\n"
                "• `/status` - System Status\n"
                "• `/portfolio` - Portfolio Übersicht\n"
                "• `/positions` - Aktuelle Positionen\n"
                "• `/performance` - Performance Metriken\n"
                "• `/settings` - Einstellungen\n"
                "• `/stop` - Bot stoppen\n"
                "• `/help` - Hilfe anzeigen\n\n"
                "🔥 *Bereit für unschlagbare Performance!*"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Status", callback_data="status"),
                 InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")],
                [InlineKeyboardButton("📈 Performance", callback_data="performance"),
                 InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
            ])
            
            await update.message.reply_text(
                message, 
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
            
            # Callback auslösen
            if 'bot_started' in self.command_callbacks:
                await self.command_callbacks['bot_started']()
            
        except Exception as e:
            logger.error(f"❌ Fehler bei /start Command: {e}")
    
    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🛑 /stop Command"""
        try:
            self.commands_executed += 1
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Ja, stoppen", callback_data="confirm_stop"),
                 InlineKeyboardButton("❌ Abbrechen", callback_data="cancel_stop")]
            ])
            
            await update.message.reply_text(
                "🛑 *Bot wirklich stoppen?*\n\n"
                "⚠️ Alle laufenden Trades werden beibehalten,\n"
                "aber Benachrichtigungen werden gestoppt.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei /stop Command: {e}")
    
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📊 /status Command"""
        try:
            self.commands_executed += 1
            
            # Status-Daten von Callbacks abrufen
            status_data = {}
            if 'get_system_status' in self.command_callbacks:
                status_data = await self.command_callbacks['get_system_status']()
            
            uptime = datetime.utcnow() - self.last_activity
            
            message = (
                "📊 *TRADINO SYSTEM STATUS*\n\n"
                f"🟢 *Status:* {'Online' if self.is_running else 'Offline'}\n"
                f"⏱️ *Uptime:* {self._format_duration(uptime.total_seconds())}\n"
                f"📱 *Nachrichten:* {self.messages_sent}\n"
                f"⌨️ *Befehle:* {self.commands_executed}\n\n"
                f"💰 *Balance:* {status_data.get('balance', 'N/A')}\n"
                f"📊 *Aktive Positionen:* {status_data.get('positions', 0)}\n"
                f"📈 *Heute P&L:* {status_data.get('daily_pnl', 'N/A')}\n"
                f"🔥 *Portfolio Heat:* {status_data.get('portfolio_heat', 'N/A')}\n\n"
                f"🧠 *AI Status:* {status_data.get('ai_status', 'Active')}\n"
                f"⚡ *API Latenz:* {status_data.get('api_latency', 'N/A')}ms"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Refresh", callback_data="status"),
                 InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")]
            ])
            
            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei /status Command: {e}")
    
    async def _cmd_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """💼 /portfolio Command"""
        try:
            self.commands_executed += 1
            
            # Portfolio-Daten von Callbacks abrufen
            portfolio_data = {}
            if 'get_portfolio_data' in self.command_callbacks:
                portfolio_data = await self.command_callbacks['get_portfolio_data']()
            
            total_balance = portfolio_data.get('total_balance', 0)
            available_balance = portfolio_data.get('available_balance', 0)
            unrealized_pnl = portfolio_data.get('unrealized_pnl', 0)
            daily_pnl = portfolio_data.get('daily_pnl', 0)
            
            pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
            pnl_color = "🟢" if unrealized_pnl >= 0 else "🔴"
            
            message = (
                "💼 *PORTFOLIO ÜBERSICHT*\n\n"
                f"💰 *Gesamtbalance:* {format_currency(total_balance)}\n"
                f"💵 *Verfügbar:* {format_currency(available_balance)}\n"
                f"📊 *Verwendet:* {format_currency(total_balance - available_balance)}\n\n"
                f"{pnl_color} *Unrealisiert P&L:* {format_currency(unrealized_pnl)}\n"
                f"{pnl_emoji} *Heute P&L:* {format_currency(daily_pnl)}\n\n"
                f"📊 *Positionen:* {portfolio_data.get('position_count', 0)}\n"
                f"🔥 *Portfolio Heat:* {format_percentage(portfolio_data.get('portfolio_heat', 0))}\n"
                f"⚖️ *Margin Ratio:* {format_percentage(portfolio_data.get('margin_ratio', 0))}"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Positionen", callback_data="positions"),
                 InlineKeyboardButton("📈 Performance", callback_data="performance")],
                [InlineKeyboardButton("🔄 Refresh", callback_data="portfolio")]
            ])
            
            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei /portfolio Command: {e}")
    
    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📊 /positions Command"""
        try:
            self.commands_executed += 1
            
            # Positions-Daten von Callbacks abrufen
            positions_data = []
            if 'get_positions_data' in self.command_callbacks:
                positions_data = await self.command_callbacks['get_positions_data']()
            
            if not positions_data:
                message = (
                    "📊 *AKTUELLE POSITIONEN*\n\n"
                    "🚫 Keine aktiven Positionen vorhanden.\n\n"
                    "💡 Der Bot analysiert kontinuierlich den Markt\n"
                    "und öffnet Positionen bei profitablen Gelegenheiten."
                )
            else:
                message = "📊 *AKTUELLE POSITIONEN*\n\n"
                
                for i, pos in enumerate(positions_data[:5], 1):  # Top 5 Positionen
                    side_emoji = "📈" if pos.get('side') == 'long' else "📉"
                    pnl_emoji = "🟢" if pos.get('pnl', 0) >= 0 else "🔴"
                    
                    message += (
                        f"{side_emoji} *{pos.get('symbol')}* ({pos.get('side', '').upper()})\n"
                        f"💰 Size: {format_currency(pos.get('size', 0))}\n"
                        f"📍 Entry: {pos.get('entry_price', 'N/A')}\n"
                        f"📍 Current: {pos.get('current_price', 'N/A')}\n"
                        f"{pnl_emoji} P&L: {format_currency(pos.get('pnl', 0))} "
                        f"({format_percentage(pos.get('pnl_percent', 0))})\n"
                        f"⚖️ Leverage: {pos.get('leverage', 1)}x\n\n"
                    )
                
                if len(positions_data) > 5:
                    message += f"... und {len(positions_data) - 5} weitere Positionen"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("💼 Portfolio", callback_data="portfolio"),
                 InlineKeyboardButton("📈 Performance", callback_data="performance")],
                [InlineKeyboardButton("🔄 Refresh", callback_data="positions")]
            ])
            
            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei /positions Command: {e}")
    
    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📈 /performance Command"""
        try:
            self.commands_executed += 1
            
            # Performance-Daten von Callbacks abrufen
            perf_data = {}
            if 'get_performance_data' in self.command_callbacks:
                perf_data = await self.command_callbacks['get_performance_data']()
            
            win_rate = perf_data.get('win_rate', 0)
            profit_factor = perf_data.get('profit_factor', 0)
            sharpe_ratio = perf_data.get('sharpe_ratio', 0)
            max_drawdown = perf_data.get('max_drawdown', 0)
            
            win_emoji = "🟢" if win_rate >= 0.6 else "🟡" if win_rate >= 0.5 else "🔴"
            pf_emoji = "🟢" if profit_factor >= 1.5 else "🟡" if profit_factor >= 1.0 else "🔴"
            
            message = (
                "📈 *PERFORMANCE METRIKEN*\n\n"
                f"🎯 *Win Rate:* {win_emoji} {format_percentage(win_rate)}\n"
                f"💰 *Profit Factor:* {pf_emoji} {profit_factor:.2f}\n"
                f"📊 *Sharpe Ratio:* {sharpe_ratio:.2f}\n"
                f"📉 *Max Drawdown:* {format_percentage(max_drawdown)}\n\n"
                f"📊 *Total Trades:* {perf_data.get('total_trades', 0)}\n"
                f"✅ *Gewinn Trades:* {perf_data.get('winning_trades', 0)}\n"
                f"❌ *Verlust Trades:* {perf_data.get('losing_trades', 0)}\n\n"
                f"💹 *Ø Gewinn:* {format_currency(perf_data.get('avg_win', 0))}\n"
                f"💸 *Ø Verlust:* {format_currency(perf_data.get('avg_loss', 0))}\n\n"
                f"🚀 *ROI:* {format_percentage(perf_data.get('roi', 0))}"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Portfolio", callback_data="portfolio"),
                 InlineKeyboardButton("📊 Positionen", callback_data="positions")],
                [InlineKeyboardButton("🔄 Refresh", callback_data="performance")]
            ])
            
            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei /performance Command: {e}")
    
    async def _cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """⚙️ /settings Command"""
        try:
            self.commands_executed += 1
            
            risk_per_trade = self.config.get('trading.risk_per_trade', 0.03)
            max_positions = self.config.get('trading.max_positions', 5)
            auto_trading = self.config.get('trading.auto_execution', True)
            
            message = (
                "⚙️ *SYSTEM EINSTELLUNGEN*\n\n"
                f"💰 *Risiko pro Trade:* {format_percentage(risk_per_trade)}\n"
                f"📊 *Max Positionen:* {max_positions}\n"
                f"🤖 *Auto Trading:* {'✅ An' if auto_trading else '❌ Aus'}\n"
                f"📱 *Notifications:* ✅ Vollständig\n\n"
                "🔧 *Strategien:*\n"
                f"• Scalping Master: {'✅' if self.config.get('strategies.scalping_master.enabled') else '❌'}\n"
                f"• Swing Genius: {'✅' if self.config.get('strategies.swing_genius.enabled') else '❌'}\n"
                f"• Trend Hunter: {'✅' if self.config.get('strategies.trend_hunter.enabled') else '❌'}\n"
                f"• Mean Reversion: {'✅' if self.config.get('strategies.mean_reversion.enabled') else '❌'}"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔧 Trading Config", callback_data="config_trading"),
                 InlineKeyboardButton("🧪 Strategien", callback_data="config_strategies")],
                [InlineKeyboardButton("📱 Notifications", callback_data="config_notifications")],
                [InlineKeyboardButton("🔄 Refresh", callback_data="settings")]
            ])
            
            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei /settings Command: {e}")
    
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """❓ /help Command"""
        try:
            self.commands_executed += 1
            
            message = (
                "❓ *TRADINO HILFE*\n\n"
                "🚀 *TRADINO UNSCHLAGBAR* ist ein AI-Trading Bot\n"
                "für Bitget Futures mit überlegener Performance.\n\n"
                "📋 *Verfügbare Befehle:*\n"
                "• `/start` - Bot starten\n"
                "• `/stop` - Bot stoppen\n"
                "• `/status` - System Status anzeigen\n"
                "• `/portfolio` - Portfolio Übersicht\n"
                "• `/positions` - Aktuelle Positionen\n"
                "• `/performance` - Performance Metriken\n"
                "• `/settings` - Einstellungen anzeigen\n"
                "• `/help` - Diese Hilfe anzeigen\n\n"
                "🎯 *Features:*\n"
                "• Multi-Layer AI Intelligence\n"
                "• 4 Trading Strategien\n"
                "• Bulletproof Risk Management\n"
                "• Real-time Notifications\n"
                "• <50ms Trade Execution\n\n"
                "💪 *Bereit für unschlagbare Performance!*"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Status", callback_data="status"),
                 InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")]
            ])
            
            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"❌ Fehler bei /help Command: {e}")
    
    async def _handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🔘 Callback Query Handler für Inline Keyboards"""
        try:
            query = update.callback_query
            await query.answer()
            
            # Command basierend auf callback_data ausführen
            data = query.data
            
            if data == "status":
                await self._cmd_status(update, context)
            elif data == "portfolio":
                await self._cmd_portfolio(update, context)
            elif data == "positions":
                await self._cmd_positions(update, context)
            elif data == "performance":
                await self._cmd_performance(update, context)
            elif data == "settings":
                await self._cmd_settings(update, context)
            elif data == "confirm_stop":
                await self._handle_stop_confirmation(update, context)
            elif data == "cancel_stop":
                await query.edit_message_text("🚀 Bot läuft weiter!")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Callback Query: {e}")
    
    async def _handle_stop_confirmation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """✅ Stop-Bestätigung verarbeiten"""
        try:
            await update.callback_query.edit_message_text(
                "🛑 *Bot wird gestoppt...*\n\n"
                "✅ Alle Trades bleiben aktiv\n"
                "📱 Benachrichtigungen gestoppt\n\n"
                "Auf Wiedersehen! 👋",
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Stop Callback auslösen
            if 'bot_stopped' in self.command_callbacks:
                await self.command_callbacks['bot_stopped']()
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Stop-Bestätigung: {e}")
    
    # ==================== NOTIFICATION SYSTEM ====================
    
    async def send_trade_signal(self, signal: AISignal):
        """🎯 Trading Signal Notification"""
        if not self.notifications.get('trade_signals', True):
            return
        
        try:
            message = self._template_trade_signal(signal)
            await self._send_message(message)
        except Exception as e:
            logger.error(f"❌ Fehler bei Signal-Notification: {e}")
    
    async def send_trade_execution(self, trade: Trade):
        """⚡ Trade Execution Notification"""
        if not self.notifications.get('trade_execution', True):
            return
        
        try:
            message = self._template_trade_execution(trade)
            await self._send_message(message)
        except Exception as e:
            logger.error(f"❌ Fehler bei Execution-Notification: {e}")
    
    async def send_position_update(self, position: Position):
        """📊 Position Update Notification"""
        if not self.notifications.get('position_updates', True):
            return
        
        try:
            # Nur bei signifikanten Änderungen senden
            if abs(position.unrealized_pnl_percent) >= 5:  # >= 5% Änderung
                message = self._template_position_update(position)
                await self._send_message(message)
        except Exception as e:
            logger.error(f"❌ Fehler bei Position-Update: {e}")
    
    async def send_pnl_update(self, daily_pnl: Decimal, total_pnl: Decimal):
        """💰 P&L Update Notification"""
        if not self.notifications.get('pnl_updates', True):
            return
        
        try:
            message = self._template_pnl_update(daily_pnl, total_pnl)
            await self._send_message(message)
        except Exception as e:
            logger.error(f"❌ Fehler bei PnL-Update: {e}")
    
    async def send_risk_alert(self, alert_type: str, message: str, severity: str):
        """🚨 Risk Alert Notification"""
        if not self.notifications.get('risk_alerts', True):
            return
        
        try:
            formatted_message = self._template_risk_alert(alert_type, message, severity)
            await self._send_message(formatted_message)
        except Exception as e:
            logger.error(f"❌ Fehler bei Risk-Alert: {e}")
    
    async def send_daily_report(self, report_data: Dict):
        """📊 Daily Report Notification"""
        if not self.notifications.get('daily_reports', True):
            return
        
        try:
            message = self._template_daily_report(report_data)
            await self._send_message(message)
        except Exception as e:
            logger.error(f"❌ Fehler bei Daily-Report: {e}")
    
    async def send_emergency_alert(self, alert_message: str):
        """🚨 Emergency Alert Notification"""
        if not self.notifications.get('emergency_alerts', True):
            return
        
        try:
            message = self._template_emergency_alert(alert_message)
            await self._send_message(message, priority=True)
        except Exception as e:
            logger.error(f"❌ Fehler bei Emergency-Alert: {e}")
    
    # ==================== MESSAGE TEMPLATES ====================
    
    def _template_trade_signal(self, signal: AISignal) -> str:
        """🎯 Trade Signal Template"""
        side_emoji = "📈" if signal.signal_type.value == 'buy' else "📉"
        strength_emoji = {"weak": "🟡", "moderate": "🟠", "strong": "🟢", "very_strong": "🔥"}.get(signal.strength.value, "🟡")
        
        return (
            f"🎯 *TRADING SIGNAL*\n\n"
            f"{side_emoji} *{signal.symbol}* - {signal.signal_type.value.upper()}\n"
            f"{strength_emoji} *Stärke:* {signal.strength.value.title()}\n"
            f"🧠 *Confidence:* {format_percentage(signal.confidence)}\n"
            f"💰 *Entry:* {signal.entry_price}\n"
            f"🛡️ *Stop Loss:* {signal.stop_loss or 'N/A'}\n"
            f"🎯 *Take Profit:* {signal.take_profit or 'N/A'}\n"
            f"📊 *Strategie:* {signal.strategy_source}\n"
            f"⏰ *Timeframe:* {signal.timeframe}"
        )
    
    def _template_trade_execution(self, trade: Trade) -> str:
        """⚡ Trade Execution Template"""
        side_emoji = "📈" if trade.side.value == 'buy' else "📉"
        
        return (
            f"⚡ *TRADE AUSGEFÜHRT*\n\n"
            f"{side_emoji} *{trade.symbol}* - {trade.side.value.upper()}\n"
            f"💰 *Preis:* {trade.entry_price}\n"
            f"📊 *Menge:* {trade.quantity}\n"
            f"⚖️ *Leverage:* {trade.leverage}x\n"
            f"🧪 *Strategie:* {trade.strategy}\n"
            f"⏰ *Zeit:* {trade.entry_time.strftime('%H:%M:%S')}"
        )
    
    def _template_position_update(self, position: Position) -> str:
        """📊 Position Update Template"""
        side_emoji = "📈" if position.side.value == 'long' else "📉"
        pnl_emoji = "🟢" if position.unrealized_pnl >= 0 else "🔴"
        
        return (
            f"📊 *POSITION UPDATE*\n\n"
            f"{side_emoji} *{position.symbol}* ({position.side.value.upper()})\n"
            f"💰 *Entry:* {position.entry_price}\n"
            f"📍 *Current:* {position.current_price}\n"
            f"{pnl_emoji} *P&L:* {format_currency(position.unrealized_pnl)} "
            f"({format_percentage(position.unrealized_pnl_percent)})\n"
            f"⚖️ *Leverage:* {position.leverage}x"
        )
    
    def _template_pnl_update(self, daily_pnl: Decimal, total_pnl: Decimal) -> str:
        """💰 P&L Update Template"""
        daily_emoji = "📈" if daily_pnl >= 0 else "📉"
        total_emoji = "🟢" if total_pnl >= 0 else "🔴"
        
        return (
            f"💰 *P&L UPDATE*\n\n"
            f"{daily_emoji} *Heute:* {format_currency(daily_pnl)}\n"
            f"{total_emoji} *Gesamt:* {format_currency(total_pnl)}\n\n"
            f"📊 *Performance:* {'Stark' if daily_pnl > 0 else 'Neutral' if daily_pnl == 0 else 'Schwach'}"
        )
    
    def _template_risk_alert(self, alert_type: str, message: str, severity: str) -> str:
        """🚨 Risk Alert Template"""
        severity_emojis = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
        emoji = severity_emojis.get(severity, "⚠️")
        
        return (
            f"{emoji} *RISK ALERT*\n\n"
            f"🛡️ *Type:* {alert_type.title()}\n"
            f"📝 *Message:* {message}\n"
            f"⚡ *Severity:* {severity.upper()}\n"
            f"⏰ *Zeit:* {datetime.utcnow().strftime('%H:%M:%S')}"
        )
    
    def _template_system_status(self, status_data: Dict) -> str:
        """📊 System Status Template"""
        return (
            f"📊 *SYSTEM STATUS*\n\n"
            f"🟢 *Online* - Alle Systeme aktiv\n"
            f"💰 *Balance:* {status_data.get('balance', 'N/A')}\n"
            f"📊 *Positionen:* {status_data.get('positions', 0)}\n"
            f"🧠 *AI:* Active\n"
            f"⚡ *Latenz:* {status_data.get('latency', 'N/A')}ms"
        )
    
    def _template_daily_report(self, report_data: Dict) -> str:
        """📊 Daily Report Template"""
        trades_today = report_data.get('trades_today', 0)
        daily_pnl = report_data.get('daily_pnl', 0)
        win_rate = report_data.get('win_rate_today', 0)
        
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        performance_emoji = "🔥" if daily_pnl > 0 else "💪" if daily_pnl == 0 else "🛡️"
        
        return (
            f"📊 *TAGESBERICHT*\n\n"
            f"📅 *Datum:* {datetime.utcnow().strftime('%d.%m.%Y')}\n\n"
            f"📊 *Trades heute:* {trades_today}\n"
            f"🎯 *Win Rate:* {format_percentage(win_rate)}\n"
            f"{pnl_emoji} *Tages P&L:* {format_currency(daily_pnl)}\n\n"
            f"{performance_emoji} *Status:* {'Excellent' if daily_pnl > 0 else 'Stable' if daily_pnl == 0 else 'Protected'}\n\n"
            f"💪 *TRADINO bleibt unschlagbar!*"
        )
    
    def _template_emergency_alert(self, alert_message: str) -> str:
        """🚨 Emergency Alert Template"""
        return (
            f"🚨 *EMERGENCY ALERT* 🚨\n\n"
            f"⚠️ *KRITISCH:* {alert_message}\n\n"
            f"🛡️ *Sofortige Aufmerksamkeit erforderlich!*\n"
            f"⏰ *Zeit:* {datetime.utcnow().strftime('%d.%m.%Y %H:%M:%S')}"
        )
    
    # ==================== UTILITY METHODS ====================
    
    async def _send_message(self, message: str, priority: bool = False):
        """📤 Nachricht senden"""
        try:
            if not self.chat_id:
                logger.warning("⚠️ Chat ID nicht konfiguriert")
                return
            
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True
            )
            
            self.messages_sent += 1
            self.last_activity = datetime.utcnow()
            
            logger.info(f"📤 Telegram Nachricht gesendet (Priority: {priority})")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Senden der Nachricht: {e}")
    
    def _format_duration(self, seconds: float) -> str:
        """⏰ Dauer formatieren"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            return f"{minutes}m"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    async def send_startup_message(self):
        """🚀 Startup-Nachricht senden"""
        try:
            message = (
                "🚀 *TRADINO UNSCHLAGBAR GESTARTET!*\n\n"
                "💪 Der ultimative AI-Trading Bot ist online!\n\n"
                f"🌍 *Environment:* {'Demo' if self.config.is_demo_mode() else 'Live'}\n"
                f"📱 *Notifications:* Vollständig aktiviert\n"
                f"🧠 *AI Status:* Aktiv und bereit\n"
                f"⚡ *Performance:* Optimiert für <50ms Execution\n\n"
                "🔥 *Bereit für unschlagbare Performance!*\n\n"
                "💡 Verwende `/help` für alle verfügbaren Befehle."
            )
            
            await self._send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Startup-Nachricht: {e}")
    
    # ==================== CALLBACK SYSTEM ====================
    
    def register_callback(self, event: str, callback: Callable):
        """Callback für Events registrieren"""
        self.command_callbacks[event] = callback
        logger.info(f"🔧 Callback registriert: {event}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """📊 Telegram Statistiken abrufen"""
        return {
            'messages_sent': self.messages_sent,
            'commands_executed': self.commands_executed,
            'last_activity': self.last_activity,
            'is_running': self.is_running,
            'notifications_enabled': self.notifications
        }
    
    async def shutdown(self):
        """🛑 Telegram Bot herunterfahren"""
        try:
            if self.is_running and self.app:
                # Shutdown-Nachricht senden
                await self._send_message(
                    "🛑 *TRADINO SHUTDOWN*\n\n"
                    "Bot wird heruntergefahren...\n"
                    "Alle Trades bleiben aktiv!\n\n"
                    "Auf Wiedersehen! 👋"
                )
                
                # App stoppen
                await self.app.stop()
                await self.app.shutdown()
                
                self.is_running = False
                logger.info("✅ Telegram Bot heruntergefahren")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Herunterfahren des Telegram Bots: {e}")
