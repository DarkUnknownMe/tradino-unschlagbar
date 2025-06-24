#!/usr/bin/env python3
"""
📱 AI ANALYSIS TELEGRAM INTEGRATION
Telegram Bot Integration für AI Analysis Monitoring in TRADINO
"""

import os
import sys
import json
import asyncio
from datetime import datetime
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

# Import AI Monitoring System
try:
    from ai_analysis_monitor import get_ai_monitoring_system
    from trained_model_integration import trained_models
    AI_MONITORING_AVAILABLE = True
except ImportError:
    AI_MONITORING_AVAILABLE = False

class AITelegramBot:
    """📱 AI Analysis Telegram Bot"""
    
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
            print("✅ AI Analysis Telegram Bot initialized")
        except Exception as e:
            print(f"❌ Bot initialization failed: {e}")
            raise
    
    def register_handlers(self):
        """📋 Register command handlers"""
        
        # AI Analysis Commands
        self.application.add_handler(CommandHandler("ai_status", self.ai_status))
        self.application.add_handler(CommandHandler("ai_dashboard", self.ai_dashboard))
        self.application.add_handler(CommandHandler("ai_report", self.ai_report))
        self.application.add_handler(CommandHandler("ai_models", self.ai_models))
        self.application.add_handler(CommandHandler("ai_features", self.ai_features))
        self.application.add_handler(CommandHandler("ai_history", self.ai_history))
        self.application.add_handler(CommandHandler("ai_agreement", self.ai_agreement))
        
        # Monitoring Controls
        self.application.add_handler(CommandHandler("start_ai_monitoring", self.start_monitoring))
        self.application.add_handler(CommandHandler("stop_ai_monitoring", self.stop_monitoring))
        
        # General Commands
        self.application.add_handler(CommandHandler("ai_help", self.ai_help))
        self.application.add_handler(CommandHandler("start", self.start_command))
        
        print("✅ AI Analysis command handlers registered")
    
    async def ai_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📊 Show AI analysis status"""
        try:
            if not AI_MONITORING_AVAILABLE:
                await update.message.reply_text("❌ AI Monitoring nicht verfügbar")
                return
            
            # Get data from trained models
            ai_data = trained_models.export_ai_analysis_for_telegram()
            
            if not ai_data:
                await update.message.reply_text("📊 Keine AI-Analysen verfügbar")
                return
            
            summary = ai_data['summary']
            latest = ai_data['latest_decision']
            
            message = f"🤖 AI ANALYSIS STATUS\n\n"
            message += f"📈 Analysen total: {summary['total_analyses']}\n"
            message += f"🎯 Ø Konfidenz: {summary['avg_confidence']:.1%}\n"
            message += f"🤝 Ø Einigkeit: {summary['avg_agreement']:.1%}\n"
            message += f"📊 Trend: {summary['confidence_trend']}\n\n"
            
            message += f"🔍 LETZTE ENTSCHEIDUNG:\n"
            message += f"Aktion: {latest['action'].upper()}\n"
            message += f"Konfidenz: {latest['confidence']:.1%}\n"
            message += f"Einigkeit: {latest['agreement']:.1%}\n"
            message += f"Hauptfaktor: {latest['top_feature']}\n"
            message += f"Zeit: {latest['timestamp'][:19]}\n"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def ai_dashboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📊 Show AI dashboard"""
        try:
            if not AI_MONITORING_AVAILABLE:
                await update.message.reply_text("❌ AI Monitoring nicht verfügbar")
                return
            
            ai_data = trained_models.export_ai_analysis_for_telegram()
            
            if not ai_data:
                await update.message.reply_text("📊 Keine AI-Analysen verfügbar")
                return
            
            message = f"📊 AI DASHBOARD\n\n"
            message += f"🕐 Zeit: {datetime.now().strftime('%H:%M:%S')}\n\n"
            
            # Model Performance
            message += f"🤖 MODELL-PERFORMANCE:\n"
            for model, perf in ai_data['model_performance'].items():
                message += f"  {model.upper()}:\n"
                message += f"    Konfidenz: {perf['avg_confidence']:.1%}\n"
                message += f"    Genauigkeit: {perf['accuracy']:.1%}\n"
                message += f"    Predictions: {perf['predictions']}\n"
                message += f"    Ø Zeit: {perf['avg_processing_time']:.1f}ms\n\n"
            
            # Recent Decisions
            message += f"📈 LETZTE ENTSCHEIDUNGEN:\n"
            for decision, count in ai_data['recent_decisions'].items():
                message += f"  {decision.upper()}: {count}\n"
            
            # Top Features
            message += f"\n🔝 TOP FEATURES:\n"
            for feature, importance in list(ai_data['top_features'].items())[:5]:
                message += f"  {feature}: {importance:.1%}\n"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def ai_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📝 Generate AI analysis report"""
        try:
            if not AI_MONITORING_AVAILABLE:
                await update.message.reply_text("❌ AI Monitoring nicht verfügbar")
                return
            
            report = trained_models.get_ai_analysis_report()
            
            if not report:
                await update.message.reply_text("📊 Keine AI-Analysen für Report verfügbar")
                return
            
            # Split long message if needed
            if len(report) > 4000:
                # Send first part
                await update.message.reply_text(report[:4000] + "...\n\n[Fortsetzung folgt]")
                # Send second part
                await update.message.reply_text("..." + report[4000:])
            else:
                await update.message.reply_text(report)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def ai_models(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🤖 Show AI model information"""
        try:
            if not AI_MONITORING_AVAILABLE:
                await update.message.reply_text("❌ AI Monitoring nicht verfügbar")
                return
            
            status = trained_models.get_model_status()
            
            message = f"🤖 AI MODELLE\n\n"
            message += f"Status: {'✅ Ready' if status['is_ready'] else '❌ Not Ready'}\n"
            message += f"Modelle geladen: {status['models_loaded']}\n"
            message += f"Features: {status['feature_count']}\n\n"
            
            message += f"📊 MODELL-GENAUIGKEITEN:\n"
            for model, accuracy in status['model_accuracies'].items():
                message += f"  {model.upper()}: {accuracy:.1%}\n"
            
            message += f"\n🔧 VERFÜGBARE MODELLE:\n"
            for model in status['models_available']:
                message += f"  • {model}\n"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def ai_features(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📊 Show feature importance trends"""
        try:
            if not AI_MONITORING_AVAILABLE:
                await update.message.reply_text("❌ AI Monitoring nicht verfügbar")
                return
            
            ai_logger, _, _ = get_ai_monitoring_system()
            
            if not ai_logger:
                await update.message.reply_text("❌ AI Logger nicht verfügbar")
                return
            
            trends = ai_logger.get_feature_importance_trends()
            
            message = f"📊 FEATURE IMPORTANCE TRENDS\n\n"
            
            # Sort by absolute trend value
            sorted_trends = sorted(trends.items(), key=lambda x: abs(x[1]), reverse=True)
            
            for feature, trend in sorted_trends[:10]:  # Top 10
                trend_symbol = "📈" if trend > 0.01 else "📉" if trend < -0.01 else "➡️"
                message += f"{trend_symbol} {feature}: {trend:+.3f}\n"
            
            if not trends:
                message += "Keine Trend-Daten verfügbar"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def ai_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """📋 Show AI analysis history"""
        try:
            if not AI_MONITORING_AVAILABLE:
                await update.message.reply_text("❌ AI Monitoring nicht verfügbar")
                return
            
            ai_logger, _, _ = get_ai_monitoring_system()
            
            if not ai_logger:
                await update.message.reply_text("❌ AI Logger nicht verfügbar")
                return
            
            recent_analyses = ai_logger.get_recent_analyses(5)
            
            if not recent_analyses:
                await update.message.reply_text("📋 Keine Analyse-Historie verfügbar")
                return
            
            message = f"📋 AI ANALYSE HISTORIE\n\n"
            
            for i, analysis in enumerate(reversed(recent_analyses), 1):
                message += f"{i}. {analysis.decision.value.upper()}\n"
                message += f"   Konfidenz: {analysis.final_confidence:.1%}\n"
                message += f"   Einigkeit: {analysis.agreement_score:.1%}\n"
                message += f"   Zeit: {analysis.timestamp[11:19]}\n"
                message += f"   Top Feature: {max(analysis.dominant_features.items(), key=lambda x: x[1])[0] if analysis.dominant_features else 'N/A'}\n\n"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def ai_agreement(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🤝 Show model agreement trends"""
        try:
            if not AI_MONITORING_AVAILABLE:
                await update.message.reply_text("❌ AI Monitoring nicht verfügbar")
                return
            
            ai_logger, _, ai_display = get_ai_monitoring_system()
            
            if not ai_logger or not ai_display:
                await update.message.reply_text("❌ AI Components nicht verfügbar")
                return
            
            agreement_trend = ai_display.get_model_agreement_trend()
            
            message = f"🤝 MODELL-EINIGKEIT TREND\n\n"
            message += f"Aktuell: {agreement_trend['current']:.1%}\n"
            message += f"Trend: {agreement_trend['trend'].upper()}\n"
            message += f"Änderung: {agreement_trend['change']:+.1%}\n\n"
            
            if agreement_trend['current'] > 0.8:
                message += "✅ Hohe Modell-Einigkeit"
            elif agreement_trend['current'] > 0.6:
                message += "⚠️ Mittlere Modell-Einigkeit"
            else:
                message += "❌ Niedrige Modell-Einigkeit"
            
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def start_monitoring(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🔄 Start AI monitoring"""
        try:
            if not AI_MONITORING_AVAILABLE:
                await update.message.reply_text("❌ AI Monitoring nicht verfügbar")
                return
            
            # Get interval from args or use default
            interval = 60  # Default 60 seconds
            if context.args and len(context.args) > 0:
                try:
                    interval = int(context.args[0])
                    interval = max(30, min(300, interval))  # Limit between 30-300 seconds
                except ValueError:
                    pass
            
            trained_models.start_ai_monitoring(interval)
            
            await update.message.reply_text(
                f"🔄 AI Monitoring gestartet\n"
                f"Intervall: {interval} Sekunden\n"
                f"Verwende /stop_ai_monitoring zum Stoppen"
            )
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def stop_monitoring(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🛑 Stop AI monitoring"""
        try:
            if not AI_MONITORING_AVAILABLE:
                await update.message.reply_text("❌ AI Monitoring nicht verfügbar")
                return
            
            trained_models.stop_ai_monitoring()
            
            await update.message.reply_text("🛑 AI Monitoring gestoppt")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def ai_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """❓ Show AI analysis help"""
        message = f"🤖 AI ANALYSIS BOT COMMANDS\n\n"
        
        message += f"📊 STATUS & DASHBOARD:\n"
        message += f"/ai_status - AI Status Übersicht\n"
        message += f"/ai_dashboard - Vollständiges Dashboard\n"
        message += f"/ai_report - Detaillierter Analyse-Report\n\n"
        
        message += f"🤖 MODELL-INFO:\n"
        message += f"/ai_models - Modell-Status & Genauigkeiten\n"
        message += f"/ai_features - Feature Importance Trends\n"
        message += f"/ai_agreement - Modell-Einigkeit Trends\n\n"
        
        message += f"📋 HISTORIE:\n"
        message += f"/ai_history - Letzte AI-Analysen\n\n"
        
        message += f"🔄 MONITORING:\n"
        message += f"/start_ai_monitoring [interval] - Start Live Monitoring\n"
        message += f"/stop_ai_monitoring - Stop Monitoring\n\n"
        
        message += f"❓ HILFE:\n"
        message += f"/ai_help - Diese Hilfe anzeigen\n"
        
        await update.message.reply_text(message)
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """🚀 Start command"""
        await update.message.reply_text(
            f"🤖 TRADINO AI ANALYSIS BOT\n\n"
            f"Willkommen zum AI Analysis Monitoring Bot!\n\n"
            f"Verwende /ai_help für alle verfügbaren Befehle.\n"
            f"Verwende /ai_status für einen schnellen Überblick."
        )
    
    def start_bot(self):
        """🚀 Start the Telegram bot"""
        if not TELEGRAM_AVAILABLE:
            print("❌ Telegram not available")
            return
        
        try:
            self.is_running = True
            print("🚀 Starting AI Analysis Telegram Bot...")
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
        print("🛑 AI Analysis Bot stopped")

# Global instance
ai_telegram_bot = None

def initialize_ai_telegram_bot() -> Optional[AITelegramBot]:
    """📱 Initialize AI Telegram Bot"""
    global ai_telegram_bot
    
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
        ai_telegram_bot = AITelegramBot(bot_token, chat_id)
        return ai_telegram_bot
    except Exception as e:
        print(f"❌ Failed to initialize AI Telegram Bot: {e}")
        return None

def get_ai_telegram_bot() -> Optional[AITelegramBot]:
    """📱 Get AI Telegram Bot instance"""
    return ai_telegram_bot

if __name__ == "__main__":
    print("📱 TRADINO AI ANALYSIS TELEGRAM BOT")
    print("=" * 50)
    
    bot = initialize_ai_telegram_bot()
    if bot:
        print("🚀 Starting bot...")
        bot.start_bot()
    else:
        print("❌ Could not start bot")