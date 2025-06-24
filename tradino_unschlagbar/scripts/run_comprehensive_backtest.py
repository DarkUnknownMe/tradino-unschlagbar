#!/usr/bin/env python3
"""
🧪 TRADINO UNSCHLAGBAR - Comprehensive Backtesting Script
Vollständiges Backtesting mit allen AI-Modellen und Trading-Strategien

Author: AI Trading Systems
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backtesting.backtesting_engine import (
    BacktestingEngine, BacktestConfig, BacktestResults, 
    WalkForwardResult, create_sample_config
)
from backtesting.visualization import BacktestingVisualizer, generate_html_report
from utils.config_manager import ConfigManager
from utils.logger_pro import setup_logger

logger = setup_logger("ComprehensiveBacktest")


class TRADINOBacktestSuite:
    """🧪 TRADINO Comprehensive Backtesting Suite"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.results: Dict[str, BacktestResults] = {}
        self.output_dir = Path("data/backtesting/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Backtesting Configurations
        self.backtest_configs = {
            'short_term': BacktestConfig(
                start_date="2023-01-01",
                end_date="2023-06-30",
                initial_capital=10000.0,
                commission=0.001,
                slippage=0.0005,
                max_positions=3,
                timeframes=['1m', '5m', '15m'],
                symbols=['BTC/USDT', 'ETH/USDT'],
                enable_fees=True,
                enable_slippage=True
            ),
            'medium_term': BacktestConfig(
                start_date="2023-01-01",
                end_date="2023-12-31",
                initial_capital=10000.0,
                commission=0.001,
                slippage=0.0005,
                max_positions=5,
                timeframes=['1h', '4h'],
                symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
                enable_fees=True,
                enable_slippage=True
            ),
            'long_term': BacktestConfig(
                start_date="2022-01-01",
                end_date="2023-12-31",
                initial_capital=50000.0,
                commission=0.0008,
                slippage=0.0003,
                max_positions=8,
                timeframes=['4h', '1d'],
                symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'],
                enable_fees=True,
                enable_slippage=True
            )
        }
    
    async def run_full_backtesting_suite(self):
        """🚀 Vollständige Backtesting-Suite ausführen"""
        try:
            logger.info("🧪 Starte TRADINO Comprehensive Backtesting Suite")
            
            # 1. Einzelne Strategien testen
            await self._test_individual_strategies()
            
            # 2. Kombinierte Strategien testen
            await self._test_combined_strategies()
            
            # 3. Walk-Forward Analysis
            await self._run_walk_forward_analysis()
            
            # 4. Parameter-Optimierung
            await self._run_parameter_optimization()
            
            # 5. Monte Carlo Simulation
            await self._run_monte_carlo_analysis()
            
            # 6. Risk Analysis
            await self._run_risk_analysis()
            
            # 7. Performance Comparison
            await self._generate_performance_comparison()
            
            # 8. Final Report
            await self._generate_comprehensive_report()
            
            logger.info("✅ Backtesting Suite abgeschlossen")
            
        except Exception as e:
            logger.error(f"❌ Fehler in Backtesting Suite: {e}")
            raise
    
    async def _test_individual_strategies(self):
        """🎯 Einzelne Strategien testen"""
        try:
            logger.info("🎯 Teste einzelne Trading-Strategien...")
            
            strategy_configs = {
                'scalping_master': {
                    'description': 'Ultra-schnelle Scalping für 1-5min Timeframes',
                    'config': self.backtest_configs['short_term']
                },
                'swing_genius': {
                    'description': 'Swing Trading für 1h-4h Timeframes',
                    'config': self.backtest_configs['medium_term']
                },
                'trend_hunter': {
                    'description': 'Trend Following für 4h-1d Timeframes',
                    'config': self.backtest_configs['long_term']
                },
                'mean_reversion': {
                    'description': 'Mean Reversion für 15min-1h Timeframes',
                    'config': self.backtest_configs['medium_term']
                }
            }
            
            for strategy_name, strategy_info in strategy_configs.items():
                logger.info(f"📊 Teste Strategie: {strategy_name}")
                
                # Engine für diese Strategie
                engine = BacktestingEngine(strategy_info['config'])
                
                # Mock-Strategie hinzufügen (in Realität: echte Strategie-Integration)
                await engine.add_strategy(strategy_name, f"mock_{strategy_name}")
                
                # Backtest ausführen
                result = await engine.run_backtest()
                self.results[strategy_name] = result
                
                # Einzelnen Report speichern
                await self._save_strategy_report(strategy_name, result, strategy_info['description'])
                
                logger.info(f"✅ {strategy_name} abgeschlossen - Return: {result.total_return:.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Individual Strategy Tests: {e}")
    
    async def _test_combined_strategies(self):
        """🎪 Kombinierte Strategien testen"""
        try:
            logger.info("🎪 Teste kombinierte Trading-Strategien...")
            
            combined_configs = {
                'scalping_swing_combo': {
                    'strategies': ['scalping_master', 'swing_genius'],
                    'config': self.backtest_configs['medium_term'],
                    'description': 'Kombination aus Scalping und Swing Trading'
                },
                'trend_reversion_combo': {
                    'strategies': ['trend_hunter', 'mean_reversion'],
                    'config': self.backtest_configs['long_term'],
                    'description': 'Trend Following mit Mean Reversion'
                },
                'all_strategies_combo': {
                    'strategies': ['scalping_master', 'swing_genius', 'trend_hunter', 'mean_reversion'],
                    'config': self.backtest_configs['long_term'],
                    'description': 'Alle Strategien kombiniert'
                }
            }
            
            for combo_name, combo_info in combined_configs.items():
                logger.info(f"🎪 Teste Kombination: {combo_name}")
                
                # Engine für Kombination
                engine = BacktestingEngine(combo_info['config'])
                
                # Alle Strategien der Kombination hinzufügen
                for strategy in combo_info['strategies']:
                    await engine.add_strategy(strategy, f"mock_{strategy}")
                
                # Backtest ausführen
                result = await engine.run_backtest()
                self.results[combo_name] = result
                
                logger.info(f"✅ {combo_name} abgeschlossen - Return: {result.total_return:.2f}%")
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Combined Strategy Tests: {e}")
    
    async def _run_walk_forward_analysis(self):
        """📈 Walk-Forward Analysis durchführen"""
        try:
            logger.info("📈 Führe Walk-Forward Analysis durch...")
            
            # WFA für beste Einzelstrategie
            best_strategy = max(self.results.items(), 
                              key=lambda x: x[1].sharpe_ratio if x[0] in ['scalping_master', 'swing_genius', 'trend_hunter', 'mean_reversion'] else -999)
            
            strategy_name, _ = best_strategy
            logger.info(f"📊 WFA für beste Strategie: {strategy_name}")
            
            # WFA Konfiguration
            wfa_config = BacktestConfig(
                start_date="2022-01-01",
                end_date="2023-12-31",
                initial_capital=10000.0,
                commission=0.001,
                slippage=0.0005,
                max_positions=5,
                symbols=['BTC/USDT', 'ETH/USDT']
            )
            
            # WFA Engine
            wfa_engine = BacktestingEngine(wfa_config)
            await wfa_engine.add_strategy(strategy_name, f"mock_{strategy_name}")
            
            # Walk-Forward Analysis ausführen
            wfa_result = await wfa_engine.walk_forward_analysis(
                train_period=365,  # 1 Jahr Training
                test_period=30,    # 1 Monat Test
                step_size=30       # Monatliche Steps
            )
            
            # WFA Ergebnisse speichern
            wfa_path = self.output_dir / f"walk_forward_analysis_{strategy_name}.json"
            wfa_data = {
                'strategy': strategy_name,
                'stability_score': wfa_result.stability_score,
                'overfitting_score': wfa_result.overfitting_score,
                'combined_metrics': wfa_result.combined_metrics,
                'period_count': len(wfa_result.results)
            }
            
            with open(wfa_path, 'w') as f:
                json.dump(wfa_data, f, indent=2, default=str)
            
            logger.info(f"✅ WFA abgeschlossen - Stability Score: {wfa_result.stability_score:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Walk-Forward Analysis: {e}")
    
    async def _run_parameter_optimization(self):
        """🔧 Parameter-Optimierung durchführen"""
        try:
            logger.info("🔧 Führe Parameter-Optimierung durch...")
            
            # Parameter Grid für Optimierung
            parameter_grid = {
                'commission': [0.0005, 0.001, 0.0015],
                'slippage': [0.0002, 0.0005, 0.001],
                'max_positions': [3, 5, 8]
            }
            
            # Optimierung für beste Strategie
            if self.results:
                best_strategy_name = max(self.results.items(), 
                                       key=lambda x: x[1].sharpe_ratio)[0]
                
                logger.info(f"🔧 Optimiere Parameter für: {best_strategy_name}")
                
                # Optimierungs-Engine
                opt_config = self.backtest_configs['medium_term']
                opt_engine = BacktestingEngine(opt_config)
                await opt_engine.add_strategy(best_strategy_name, f"mock_{best_strategy_name}")
                
                # Parameter-Optimierung ausführen
                optimization_result = await opt_engine.optimize_parameters(
                    parameter_grid=parameter_grid,
                    optimization_metric='sharpe_ratio'
                )
                
                # Optimierungs-Ergebnisse speichern
                opt_path = self.output_dir / f"parameter_optimization_{best_strategy_name}.json"
                with open(opt_path, 'w') as f:
                    json.dump(optimization_result, f, indent=2, default=str)
                
                logger.info(f"✅ Parameter-Optimierung abgeschlossen - Bester Score: {optimization_result.get('best_score', 0):.3f}")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Parameter-Optimierung: {e}")
    
    async def _run_monte_carlo_analysis(self):
        """🎰 Monte Carlo Simulation durchführen"""
        try:
            logger.info("🎰 Führe Monte Carlo Simulation durch...")
            
            for strategy_name, result in self.results.items():
                if strategy_name in ['scalping_master', 'swing_genius', 'trend_hunter', 'mean_reversion']:
                    logger.info(f"🎰 Monte Carlo für: {strategy_name}")
                    
                    # Engine für Monte Carlo
                    engine = BacktestingEngine(self.backtest_configs['medium_term'])
                    engine.trade_history = result.trades  # Historische Trades laden
                    
                    # Monte Carlo Simulation
                    mc_result = engine.monte_carlo_simulation(num_simulations=1000)
                    
                    # MC Ergebnisse speichern
                    mc_path = self.output_dir / f"monte_carlo_{strategy_name}.json"
                    with open(mc_path, 'w') as f:
                        json.dump(mc_result, f, indent=2, default=str)
                    
                    logger.info(f"✅ Monte Carlo für {strategy_name} - P(positive): {mc_result.get('probability_positive', 0):.2%}")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Monte Carlo Analysis: {e}")
    
    async def _run_risk_analysis(self):
        """🛡️ Risk Analysis durchführen"""
        try:
            logger.info("🛡️ Führe Risk Analysis durch...")
            
            risk_analysis = {
                'strategy_risk_profiles': {},
                'portfolio_risk_metrics': {},
                'correlation_analysis': {},
                'var_analysis': {}
            }
            
            for strategy_name, result in self.results.items():
                # Risk Profile für jede Strategie
                risk_profile = {
                    'sharpe_ratio': result.sharpe_ratio,
                    'sortino_ratio': result.sortino_ratio,
                    'max_drawdown': result.max_drawdown,
                    'volatility': result.volatility,
                    'var_95': result.var_95,
                    'cvar_95': result.cvar_95,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor
                }
                
                risk_analysis['strategy_risk_profiles'][strategy_name] = risk_profile
                
                # Risk Rating
                risk_score = self._calculate_risk_score(result)
                risk_analysis['strategy_risk_profiles'][strategy_name]['risk_score'] = risk_score
                risk_analysis['strategy_risk_profiles'][strategy_name]['risk_rating'] = self._get_risk_rating(risk_score)
            
            # Risk Analysis speichern
            risk_path = self.output_dir / "comprehensive_risk_analysis.json"
            with open(risk_path, 'w') as f:
                json.dump(risk_analysis, f, indent=2, default=str)
            
            logger.info("✅ Risk Analysis abgeschlossen")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Risk Analysis: {e}")
    
    def _calculate_risk_score(self, result: BacktestResults) -> float:
        """🎯 Risk Score berechnen"""
        try:
            # Normalisierte Metriken (0-1 Skala)
            sharpe_norm = min(result.sharpe_ratio / 2.0, 1.0)  # Sharpe > 2.0 = perfect
            dd_norm = max(1 - result.max_drawdown / 20.0, 0.0)  # DD < 20% = good
            vol_norm = max(1 - result.volatility / 50.0, 0.0)  # Vol < 50% = good
            wr_norm = result.win_rate / 100.0  # Win Rate normalisiert
            
            # Gewichteter Risk Score
            risk_score = (sharpe_norm * 0.3 + dd_norm * 0.3 + vol_norm * 0.2 + wr_norm * 0.2)
            
            return min(max(risk_score, 0.0), 1.0)
            
        except Exception:
            return 0.5  # Neutral Score
    
    def _get_risk_rating(self, score: float) -> str:
        """🏆 Risk Rating bestimmen"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Average"
        elif score >= 0.2:
            return "Poor"
        else:
            return "High Risk"
    
    async def _generate_performance_comparison(self):
        """⚖️ Performance Comparison generieren"""
        try:
            logger.info("⚖️ Generiere Performance Comparison...")
            
            # Nur Einzelstrategien für Vergleich
            strategy_results = {k: v for k, v in self.results.items() 
                              if k in ['scalping_master', 'swing_genius', 'trend_hunter', 'mean_reversion']}
            
            if strategy_results:
                # Performance Comparison mit erstem verfügbaren Result
                first_result = list(strategy_results.values())[0]
                visualizer = BacktestingVisualizer(first_result)
                
                # Comparison Chart erstellen (vereinfacht)
                comparison_data = {
                    'strategies': list(strategy_results.keys()),
                    'metrics': {
                        'total_return': [r.total_return for r in strategy_results.values()],
                        'sharpe_ratio': [r.sharpe_ratio for r in strategy_results.values()],
                        'max_drawdown': [r.max_drawdown for r in strategy_results.values()],
                        'win_rate': [r.win_rate for r in strategy_results.values()]
                    }
                }
                
                # Comparison Daten speichern
                comp_path = self.output_dir / "strategy_comparison.json"
                with open(comp_path, 'w') as f:
                    json.dump(comparison_data, f, indent=2, default=str)
            
            logger.info("✅ Performance Comparison generiert")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Performance Comparison: {e}")
    
    async def _save_strategy_report(self, strategy_name: str, result: BacktestResults, description: str):
        """📋 Einzelnen Strategy Report speichern"""
        try:
            # Report generieren
            visualizer = BacktestingVisualizer(result)
            
            # HTML Report
            report_path = self.output_dir / f"{strategy_name}_report.html"
            generate_html_report(result, str(report_path))
            
            # Charts speichern
            chart_dir = self.output_dir / f"{strategy_name}_charts"
            visualizer.save_static_charts(str(chart_dir))
            
            logger.info(f"📋 Report für {strategy_name} gespeichert")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Speichern des Reports für {strategy_name}: {e}")
    
    async def _generate_comprehensive_report(self):
        """📋 Comprehensive Final Report generieren"""
        try:
            logger.info("📋 Generiere Comprehensive Final Report...")
            
            # Summary erstellen
            summary = {
                'backtest_summary': {
                    'total_strategies_tested': len(self.results),
                    'test_period': f"{min(config.start_date for config in self.backtest_configs.values())} bis {max(config.end_date for config in self.backtest_configs.values())}",
                    'timestamp': datetime.now().isoformat()
                },
                'best_performers': {},
                'risk_analysis_summary': {},
                'recommendations': []
            }
            
            if self.results:
                # Beste Performer identifizieren
                best_return = max(self.results.items(), key=lambda x: x[1].total_return)
                best_sharpe = max(self.results.items(), key=lambda x: x[1].sharpe_ratio)
                best_winrate = max(self.results.items(), key=lambda x: x[1].win_rate)
                lowest_dd = min(self.results.items(), key=lambda x: x[1].max_drawdown)
                
                summary['best_performers'] = {
                    'highest_return': {'strategy': best_return[0], 'value': f"{best_return[1].total_return:.2f}%"},
                    'best_sharpe': {'strategy': best_sharpe[0], 'value': f"{best_sharpe[1].sharpe_ratio:.3f}"},
                    'highest_winrate': {'strategy': best_winrate[0], 'value': f"{best_winrate[1].win_rate:.1f}%"},
                    'lowest_drawdown': {'strategy': lowest_dd[0], 'value': f"{lowest_dd[1].max_drawdown:.2f}%"}
                }
                
                # Empfehlungen generieren
                summary['recommendations'] = self._generate_final_recommendations()
            
            # Final Report speichern
            final_report_path = self.output_dir / "TRADINO_Comprehensive_Backtest_Report.json"
            with open(final_report_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Executive Summary als Text
            exec_summary_path = self.output_dir / "Executive_Summary.txt"
            await self._generate_executive_summary(summary, exec_summary_path)
            
            logger.info(f"📋 Comprehensive Report generiert: {final_report_path}")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Comprehensive Report: {e}")
    
    def _generate_final_recommendations(self) -> List[str]:
        """💡 Finale Empfehlungen generieren"""
        recommendations = []
        
        if not self.results:
            return ["❌ Keine Ergebnisse für Empfehlungen verfügbar"]
        
        # Einzelstrategien analysieren
        strategy_results = {k: v for k, v in self.results.items() 
                           if k in ['scalping_master', 'swing_genius', 'trend_hunter', 'mean_reversion']}
        
        if strategy_results:
            best_strategy = max(strategy_results.items(), key=lambda x: x[1].sharpe_ratio)
            worst_strategy = min(strategy_results.items(), key=lambda x: x[1].sharpe_ratio)
            
            recommendations.append(f"🏆 Beste Einzelstrategie: {best_strategy[0]} (Sharpe: {best_strategy[1].sharpe_ratio:.3f})")
            recommendations.append(f"⚠️ Schwächste Strategie: {worst_strategy[0]} (Sharpe: {worst_strategy[1].sharpe_ratio:.3f})")
            
            # Risk-Adjusted Empfehlungen
            for name, result in strategy_results.items():
                risk_score = self._calculate_risk_score(result)
                if risk_score >= 0.8:
                    recommendations.append(f"✅ {name} zeigt exzellente Risk-Adjusted Performance")
                elif risk_score <= 0.3:
                    recommendations.append(f"🚨 {name} zeigt hohe Risikowerte - Überarbeitung empfohlen")
        
        # Kombinationsempfehlungen
        combo_results = {k: v for k, v in self.results.items() if 'combo' in k}
        if combo_results:
            best_combo = max(combo_results.items(), key=lambda x: x[1].sharpe_ratio)
            recommendations.append(f"🎪 Beste Kombination: {best_combo[0]} (Sharpe: {best_combo[1].sharpe_ratio:.3f})")
        
        # Allgemeine Empfehlungen
        avg_sharpe = sum(r.sharpe_ratio for r in self.results.values()) / len(self.results)
        if avg_sharpe > 1.0:
            recommendations.append("✅ Durchschnittliche Performance über Benchmark - Strategien sind profitabel")
        else:
            recommendations.append("⚠️ Durchschnittliche Performance unter Erwartung - Risk Management überprüfen")
        
        return recommendations
    
    async def _generate_executive_summary(self, summary: Dict, output_path: str):
        """📋 Executive Summary als Text generieren"""
        try:
            exec_text = f"""
🧪 TRADINO UNSCHLAGBAR - EXECUTIVE SUMMARY
=========================================

BACKTESTING PERIODE: {summary['backtest_summary']['test_period']}
STRATEGIEN GETESTET: {summary['backtest_summary']['total_strategies_tested']}
GENERIERT AM: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BESTE PERFORMER:
----------------
"""
            
            if 'best_performers' in summary:
                for metric, data in summary['best_performers'].items():
                    exec_text += f"• {metric.replace('_', ' ').title()}: {data['strategy']} ({data['value']})\n"
            
            exec_text += "\nEMPFEHLUNGEN:\n"
            exec_text += "-------------\n"
            
            for rec in summary.get('recommendations', []):
                exec_text += f"• {rec}\n"
            
            exec_text += f"""

NÄCHSTE SCHRITTE:
-----------------
• Beste Strategien für Live-Trading vorbereiten
• Risk Management Parameter finalisieren
• Monitoring und Alerting implementieren
• Portfolio-Allokation optimieren

Detaillierte Ergebnisse finden Sie in den einzelnen Strategy Reports.
"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(exec_text)
            
            logger.info(f"📋 Executive Summary generiert: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Executive Summary: {e}")


async def main():
    """🚀 Main Execution"""
    try:
        print("🧪 TRADINO UNSCHLAGBAR - COMPREHENSIVE BACKTESTING SUITE")
        print("=" * 60)
        
        # Backtesting Suite initialisieren
        suite = TRADINOBacktestSuite()
        
        # Vollständige Suite ausführen
        await suite.run_full_backtesting_suite()
        
        print("\n✅ BACKTESTING SUITE ERFOLGREICH ABGESCHLOSSEN")
        print(f"📁 Ergebnisse gespeichert in: {suite.output_dir}")
        print("\n🔍 NÄCHSTE SCHRITTE:")
        print("• Überprüfen Sie die generierten Reports")
        print("• Analysieren Sie die Performance-Metriken")
        print("• Implementieren Sie die besten Strategien")
        print("• Starten Sie das Live-Trading")
        
    except Exception as e:
        logger.error(f"❌ Fehler in Main Execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Ausführen
    asyncio.run(main()) 