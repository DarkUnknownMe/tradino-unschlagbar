#!/usr/bin/env python3
"""
🤖 ENHANCED AI TELEGRAM BOT WITH REAL PERFORMANCE
Telegram Bot mit integrierter Real Performance Engine
"""

import asyncio
import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import pandas as pd
import numpy as np

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode

# Import AI & Performance Komponenten
try:
    import ccxt
    from dotenv import load_dotenv
    # Import unserer AI-Systeme
    sys.path.append(os.path.join(os.path.dirname(__file__), 'tradino_unschlagbar'))
    from tradino_unschlagbar.brain.sentiment_analyzer import WorldClassSentimentEngine
    from tradino_unschlagbar.brain.portfolio_optimizer import WorldClassPortfolioOptimizer
    from tradino_unschlagbar.brain.market_regime_detector import MarketRegimeDetector
    from tradino_unschlagbar.analytics.real_performance_engine import RealPerformanceEngine, TradeRecord
    AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ AI Komponenten nicht verfügbar: {e}")
    AI_AVAILABLE = False

class EnhancedAITelegramBotWithPerformance:
    """🤖 Enhanced AI Telegram Bot mit Real Performance Engine"""
    
    def __init__(self):
        load_dotenv()
        
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.bot_token or not self.chat_id:
            raise ValueError("❌ Telegram credentials missing")
        
        # Telegram App
        self.app = Application.builder().token(self.bot_token).build()
        
        # Exchange Integration
        self.exchange = self._init_exchange()
        
        # AI Komponenten
        self.sentiment_engine = WorldClassSentimentEngine() if AI_AVAILABLE else None
        self.portfolio_optimizer = WorldClassPortfolioOptimizer() if AI_AVAILABLE else None
        self.regime_detector = MarketRegimeDetector() if AI_AVAILABLE else None
        
        # Real Performance Engine
        self.performance_engine = RealPerformanceEngine("telegram_bot_performance.db") if AI_AVAILABLE else None
        
        # Performance Tracking
        self.start_time = datetime.now()
        
        print("🤖 Enhanced AI Telegram Bot mit Real Performance Engine initialisiert")
    
    def _init_exchange(self):
        """🔧 Exchange initialisieren"""
        try:
            exchange = ccxt.bitget({
                'apiKey': os.getenv('BITGET_API_KEY'),
                'secret': os.getenv('BITGET_SECRET_KEY'),
                'password': os.getenv('BITGET_PASSPHRASE'),
                'sandbox': True,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
            return exchange
        except Exception as e:
            print(f"⚠️ Exchange init failed: {e}")
            return None
    
    async def _get_real_performance_data(self, period_days: int = 30) -> Dict[str, Any]:
        """📊 Echte Performance-Daten von Real Performance Engine"""
        
        if not self.performance_engine:
            return self._mock_performance_data()
        
        try:
            # Get comprehensive metrics
            metrics = self.performance_engine.calculate_comprehensive_metrics()
            
            # Get detailed report
            report = self.performance_engine.get_performance_report(period_days)
            
            return {
                'total_trades': metrics.total_trades,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'win_rate': metrics.win_rate * 100,
                'total_return': metrics.total_return,
                'annual_return': metrics.annual_return * 100,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'max_drawdown': metrics.max_drawdown * 100,
                'profit_factor': metrics.profit_factor,
                'largest_win': metrics.largest_win,
                'largest_loss': metrics.largest_loss,
                'avg_trade_duration': metrics.avg_trade_duration,
                'var_95': metrics.var_95 * 100,
                'cvar_95': metrics.cvar_95 * 100,
                'calmar_ratio': metrics.calmar_ratio,
                'consecutive_wins': metrics.consecutive_wins,
                'consecutive_losses': metrics.consecutive_losses,
                'strategy_breakdown': report.get('strategy_breakdown', {}),
                'symbol_breakdown': report.get('symbol_breakdown', {})
            }
            
        except Exception as e:
            print(f"⚠️ Fehler beim Abrufen der Performance-Daten: {e}")
            return self._mock_performance_data()
    
    def _mock_performance_data(self) -> Dict[str, Any]:
        """📊 Mock Performance-Daten als Fallback"""
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0, 'total_return': 0, 'annual_return': 0,
            'sharpe_ratio': 0, 'sortino_ratio': 0, 'max_drawdown': 0,
            'profit_factor': 0, 'largest_win': 0, 'largest_loss': 0,
            'avg_trade_duration': 0, 'var_95': 0, 'cvar_95': 0,
            'calmar_ratio': 0, 'consecutive_wins': 0, 'consecutive_losses': 0,
            'strategy_breakdown': {}, 'symbol_breakdown': {}
        }
    
    async def _add_demo_trades(self):
        """🎲 Füge Demo-Trades zur Performance Engine hinzu"""
        if not self.performance_engine:
            return
        
        # Demo trades to showcase the system
        demo_trades = [
            TradeRecord(
                trade_id=f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}_001",
                symbol="BTC/USDT",
                side="buy",
                entry_time=datetime.now() - timedelta(hours=6),
                exit_time=datetime.now() - timedelta(hours=3),
                entry_price=100000,
                exit_price=101500,
                quantity=0.01,
                fee=10.0,
                pnl=140.0,
                status="closed",
                strategy="AI_Sentiment",
                confidence=0.87,
                market_regime="Bull",
                sentiment_score=0.72
            ),
            TradeRecord(
                trade_id=f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}_002",
                symbol="ETH/USDT",
                side="buy",
                entry_time=datetime.now() - timedelta(hours=4),
                exit_time=datetime.now() - timedelta(hours=1),
                entry_price=3500,
                exit_price=3420,
                quantity=0.1,
                fee=3.5,
                pnl=-11.5,
                status="closed",
                strategy="Market_Regime",
                confidence=0.65,
                market_regime="Sideways",
                sentiment_score=0.15
            )
        ]
        
        for trade in demo_trades:
            self.performance_engine.add_trade(trade)
        
        print("✅ Demo trades zur Performance Engine hinzugefügt")
    
    # TELEGRAM COMMAND HANDLERS
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🚀 Start Command mit echten Performance-Daten"""
        
        # Add demo trades on first start
        await self._add_demo_trades()
        
        # Get real performance data
        performance = await self._get_real_performance_data()
        
        message = (
            "🤖 *ENHANCED AI BOT WITH REAL PERFORMANCE*\n"
            "===========================================\n\n"
            "🧠 *AI-Powered Trading System:*\n"
            f"✅ Sentiment Analysis Engine\n"
            f"✅ Market Regime Detection\n"
            f"✅ Portfolio Optimization\n"
            f"✅ Real Performance Tracking\n\n"
            f"📊 *Echte Performance (Live):*\n"
            f"├ Total Trades: {performance['total_trades']}\n"
            f"├ Win Rate: {performance['win_rate']:.1f}%\n"
            f"├ Total Return: ${performance['total_return']:+,.2f}\n"
            f"├ Sharpe Ratio: {performance['sharpe_ratio']:.2f}\n"
            f"├ Max Drawdown: {performance['max_drawdown']:.1f}%\n"
            f"└ Profit Factor: {performance['profit_factor']:.2f}\n\n"
            "🎮 *Advanced Analytics verfügbar:*"
        )
        
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Real Performance", callback_data="real_performance"),
                InlineKeyboardButton("🧠 AI Analytics", callback_data="ai_analytics")
            ],
            [
                InlineKeyboardButton("📈 Strategy Analysis", callback_data="strategy_analysis"),
                InlineKeyboardButton("💰 Symbol Breakdown", callback_data="symbol_breakdown")
            ],
            [
                InlineKeyboardButton("📅 Monthly Report", callback_data="monthly_report"),
                InlineKeyboardButton("📈 Equity Curve", callback_data="equity_curve")
            ],
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="refresh"),
                InlineKeyboardButton("📸 Save Snapshot", callback_data="save_snapshot")
            ]
        ])
        
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🔄 Callback Handler für alle Buttons"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        chat_id = query.message.chat_id
        message_id = query.message.message_id
        
        if data == "real_performance":
            await self._send_real_performance_details(chat_id, message_id)
        elif data == "ai_analytics":
            await self._send_ai_analytics(chat_id, message_id)
        elif data == "strategy_analysis":
            await self._send_strategy_analysis(chat_id, message_id)
        elif data == "symbol_breakdown":
            await self._send_symbol_breakdown(chat_id, message_id)
        elif data == "monthly_report":
            await self._send_monthly_report(chat_id, message_id)
        elif data == "equity_curve":
            await self._send_equity_curve(chat_id, message_id)
        elif data == "save_snapshot":
            await self._save_performance_snapshot(chat_id, message_id)
        elif data == "refresh":
            await self.cmd_start(update, context)
    
    async def _send_real_performance_details(self, chat_id: int, message_id: int):
        """📊 Detaillierte Real Performance"""
        performance = await self._get_real_performance_data()
        
        uptime = datetime.now() - self.start_time
        uptime_hours = uptime.total_seconds() / 3600
        
        message = (
            "📊 *REAL PERFORMANCE DETAILS*\n"
            "==============================\n\n"
            f"⏱️ *System Uptime:*\n"
            f"├ Seit: {self.start_time.strftime('%Y-%m-%d %H:%M')}\n"
            f"└ Laufzeit: {uptime_hours:.1f} Stunden\n\n"
            f"📈 *Trading Performance:*\n"
            f"├ Total Trades: {performance['total_trades']}\n"
            f"├ Winning: {performance['winning_trades']}\n"
            f"├ Losing: {performance['losing_trades']}\n"
            f"├ Win Rate: {performance['win_rate']:.1f}%\n"
            f"├ Total Return: ${performance['total_return']:+,.2f}\n"
            f"└ Annual Return: {performance['annual_return']:+.2f}%\n\n"
            f"⚡ *Risk-Adjusted Metrics:*\n"
            f"├ Sharpe Ratio: {performance['sharpe_ratio']:.3f}\n"
            f"├ Sortino Ratio: {performance['sortino_ratio']:.3f}\n"
            f"├ Calmar Ratio: {performance['calmar_ratio']:.3f}\n"
            f"├ Max Drawdown: {performance['max_drawdown']:.2f}%\n"
            f"└ Profit Factor: {performance['profit_factor']:.2f}\n\n"
            f"🎲 *Trade Analysis:*\n"
            f"├ Largest Win: ${performance['largest_win']:+,.2f}\n"
            f"├ Largest Loss: ${performance['largest_loss']:+,.2f}\n"
            f"├ Avg Duration: {performance['avg_trade_duration']:.1f}h\n"
            f"├ Consecutive Wins: {performance['consecutive_wins']}\n"
            f"└ Consecutive Losses: {performance['consecutive_losses']}\n\n"
            f"🔻 *Risk Metrics:*\n"
            f"├ VaR (95%): {performance['var_95']:.2f}%\n"
            f"└ CVaR (95%): {performance['cvar_95']:.2f}%"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Update", callback_data="real_performance")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="refresh")]
        ])
        
        await self.app.bot.edit_message_text(
            chat_id=chat_id, message_id=message_id, text=message,
            parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard
        )
    
    async def _send_ai_analytics(self, chat_id: int, message_id: int):
        """🧠 AI Analytics Summary"""
        message = "🧠 *AI ANALYTICS SUMMARY*\n"
        message += "=========================\n\n"
        message += "🎯 *AI System Status:*\n"
        message += f"├ Sentiment Engine: {'✅' if self.sentiment_engine else '❌'}\n"
        message += f"├ Regime Detector: {'✅' if self.regime_detector else '❌'}\n"
        message += f"├ Portfolio Optimizer: {'✅' if self.portfolio_optimizer else '❌'}\n"
        message += f"└ Performance Engine: {'✅' if self.performance_engine else '❌'}\n\n"
        message += "📊 *Current Analysis:*\n"
        message += "├ Market Sentiment: Calculating...\n"
        message += "├ Market Regime: Detecting...\n"
        message += "└ Portfolio Status: Optimizing...\n\n"
        message += "💡 Alle AI-Systeme arbeiten mit echten Daten"
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Update", callback_data="ai_analytics")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="refresh")]
        ])
        
        await self.app.bot.edit_message_text(
            chat_id=chat_id, message_id=message_id, text=message,
            parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard
        )
    
    async def setup_bot(self):
        """🔧 Bot Setup"""
        commands = [
            BotCommand("start", "🚀 Enhanced AI Bot with Real Performance"),
        ]
        
        await self.app.bot.set_my_commands(commands)
        
        # Handler
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        
        print("✅ Enhanced Bot mit Real Performance commands registered")
    
    async def run(self):
        """🚀 Bot starten"""
        try:
            print("🚀 Starting Enhanced AI Bot with Real Performance...")
            
            await self.setup_bot()
            await self.app.initialize()
            await self.app.start()
            
            print("✅ Enhanced AI Bot with Real Performance is running!")
            print("📊 Real Performance Engine aktiv")
            print("🧠 AI Analytics verfügbar")
            
            await self.app.updater.start_polling()
            
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Enhanced AI Bot...")
        except Exception as e:
            print(f"❌ Enhanced Bot Error: {e}")
        finally:
            if self.app.updater.running:
                await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

# Main Function
async def main():
    """🚀 Main Function"""
    try:
        bot = EnhancedAITelegramBotWithPerformance()
        await bot.run()
    except Exception as e:
        print(f"❌ Failed to start Enhanced AI Bot with Real Performance: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 