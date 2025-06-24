"""
📊 TRADINO UNSCHLAGBAR - Backtesting Visualization
Visualisierung von Backtesting-Ergebnissen mit Charts und Dashboards

Author: AI Trading Systems
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .backtesting_engine import BacktestResults, WalkForwardResult
from utils.logger_pro import setup_logger

logger = setup_logger("BacktestingVisualization")

# Set style
plt.style.use('dark_background')
sns.set_theme(style="darkgrid")


class BacktestingVisualizer:
    """📊 Backtesting Results Visualizer"""
    
    def __init__(self, results: BacktestResults):
        self.results = results
        self.colors = {
            'profit': '#00ff88',
            'loss': '#ff4444',
            'neutral': '#4488ff',
            'background': '#1e1e1e',
            'text': '#ffffff'
        }
    
    def create_comprehensive_dashboard(self, save_path: Optional[str] = None) -> go.Figure:
        """📊 Umfassendes Dashboard erstellen"""
        try:
            # Einfache Dashboard-Version für Demo
            fig = go.Figure()
            
            # Equity Curve
            if not self.results.equity_curve.empty:
                fig.add_trace(
                    go.Scatter(
                        x=self.results.equity_curve.index,
                        y=self.results.equity_curve['total_value'],
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color=self.colors['profit'], width=2)
                    )
                )
            
            fig.update_layout(
                title=f"📊 Backtesting Results - Return: {self.results.total_return:.2f}%",
                template="plotly_dark",
                xaxis_title="Time",
                yaxis_title="Portfolio Value"
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"📊 Dashboard gespeichert: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Fehler bei Dashboard-Erstellung: {e}")
            return go.Figure()
    
    def save_static_charts(self, output_dir: str = "data/backtesting/charts"):
        """💾 Statische Charts speichern"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Equity Curve
            if not self.results.equity_curve.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(self.results.equity_curve.index, 
                        self.results.equity_curve['total_value'],
                        color=self.colors['profit'], linewidth=2)
                plt.title('📈 Portfolio Equity Curve')
                plt.xlabel('Time')
                plt.ylabel('Portfolio Value')
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{output_dir}/equity_curve.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"📊 Charts gespeichert: {output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Speichern der Charts: {e}")


def generate_html_report(results: BacktestResults, output_path: str = "data/backtesting/report.html"):
    """📋 HTML Report generieren"""
    try:
        visualizer = BacktestingVisualizer(results)
        dashboard = visualizer.create_comprehensive_dashboard()
        
        # HTML Template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🧪 TRADINO Backtesting Report</title>
            <meta charset="utf-8">
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    background: #1e1e1e; 
                    color: #ffffff; 
                    margin: 20px; 
                }}
                .header {{ 
                    text-align: center; 
                    padding: 20px; 
                    background: #2d2d2d;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .metric {{ 
                    display: inline-block; 
                    margin: 10px; 
                    padding: 15px; 
                    background: #333; 
                    border-radius: 8px; 
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧪 TRADINO Backtesting Report</h1>
                <p>Period: {results.config.start_date} bis {results.config.end_date}</p>
            </div>
            
            <div class="metric">Total Return: {results.total_return:.2f}%</div>
            <div class="metric">Sharpe Ratio: {results.sharpe_ratio:.3f}</div>
            <div class="metric">Max Drawdown: {results.max_drawdown:.2f}%</div>
            <div class="metric">Win Rate: {results.win_rate:.1f}%</div>
            <div class="metric">Total Trades: {results.total_trades}</div>
            
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"📋 HTML Report generiert: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Fehler bei HTML Report: {e}")


if __name__ == "__main__":
    # Demo Visualization
    from .backtesting_engine import create_sample_config, BacktestingEngine
    import asyncio
    
    async def demo_visualization():
        config = create_sample_config()
        engine = BacktestingEngine(config)
        await engine.add_strategy("demo_strategy", "mock")
        
        results = await engine.run_backtest()
        
        visualizer = BacktestingVisualizer(results)
        dashboard = visualizer.create_comprehensive_dashboard("demo_dashboard.html")
        
        print("📊 Demo Visualization erstellt!")
    
    asyncio.run(demo_visualization()) 