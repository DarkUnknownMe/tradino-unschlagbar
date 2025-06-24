#!/usr/bin/env python3
"""
🌐 NOBELBRETT.DE TRADING DASHBOARD

Specialized deployment script for www.nobelbrett.de
with professional crypto branding and production configuration.

Author: TRADINO Development Team
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from config.domain_config import get_domain_config, get_custom_css, get_favicon_svg
from core.trading_dashboard import TradingDashboard

class NobelBrettDashboard(TradingDashboard):
    """🏆 Nobelbrett.de Trading Dashboard with custom branding"""
    
    def __init__(self):
        # Get domain configuration
        config = get_domain_config()
        domain_config = config['domain']
        branding_config = config['branding']
        
        # Initialize with domain-specific settings
        super().__init__(
            host=domain_config['host'],
            port=domain_config['port']
        )
        
        self.domain_config = domain_config
        self.branding_config = branding_config
        self.security_config = config['security']
        self.production_config = config['production']
        
        # Update app configuration
        self.app.title = branding_config['title']
        self.app.description = branding_config['subtitle']
        
        # Setup domain-specific routes
        self._setup_domain_routes()
        
        # Create branded static files
        self._create_branded_assets()
    
    def _setup_domain_routes(self):
        """🛤️ Setup domain-specific routes"""
        
        @self.app.get("/favicon.ico")
        async def favicon():
            """🎯 Custom favicon for Nobelbrett"""
            from fastapi.responses import Response
            svg_content = get_favicon_svg()
            return Response(content=svg_content, media_type="image/svg+xml")
        
        @self.app.get("/robots.txt")
        async def robots():
            """🤖 Robots.txt for SEO"""
            from fastapi.responses import PlainTextResponse
            robots_content = """User-agent: *
Disallow: /api/
Disallow: /ws/
Allow: /

Sitemap: https://www.nobelbrett.de/sitemap.xml
"""
            return PlainTextResponse(robots_content)
        
        @self.app.get("/health")
        async def health_check():
            """🏥 Health check endpoint for monitoring"""
            return {"status": "healthy", "domain": "www.nobelbrett.de", "service": "tradino-dashboard"}
    
    def _create_branded_assets(self):
        """🎨 Create custom branded assets"""
        # Create static directory
        static_dir = Path("static")
        static_dir.mkdir(exist_ok=True)
        
        # Create custom CSS file
        css_file = static_dir / "nobelbrett.css"
        with open(css_file, 'w', encoding='utf-8') as f:
            f.write(get_custom_css())
        
        # Create favicon ICO (simple conversion)
        favicon_file = static_dir / "favicon.ico"
        if not favicon_file.exists():
            # Create a simple favicon placeholder
            with open(favicon_file, 'wb') as f:
                # Simple 16x16 ICO header + green pixel data
                ico_data = b'\x00\x00\x01\x00\x01\x00\x10\x10\x00\x00\x01\x00\x20\x00\x68\x04\x00\x00\x16\x00\x00\x00'
                f.write(ico_data)
        
        print(f"✅ Custom assets created in {static_dir}")
    
    def _get_branded_html_template(self) -> str:
        """🎨 Get Nobelbrett-branded HTML template"""
        return '''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NOBELBRETT - Trading Dashboard</title>
    <meta name="description" content="Professional Cryptocurrency Trading Dashboard - Real-time AI-powered trading analytics">
    <meta name="keywords" content="cryptocurrency, trading, dashboard, AI, bitcoin, altcoins">
    <meta name="author" content="Nobelbrett Trading">
    
    <!-- Favicon -->
    <link rel="icon" type="image/svg+xml" href="/favicon.ico">
    
    <!-- Bootstrap & Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;700&display=swap" rel="stylesheet">
    
    <!-- Plotly -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    
    <!-- Custom Nobelbrett Styles -->
    <link href="/static/nobelbrett.css" rel="stylesheet">
    
    <!-- SEO Meta Tags -->
    <meta property="og:title" content="NOBELBRETT - Trading Dashboard">
    <meta property="og:description" content="Professional Cryptocurrency Trading Platform">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://www.nobelbrett.de">
    
    <style>
        /* Immediate loading styles */
        body { 
            background: #0d0d0d; 
            color: #ffffff; 
            font-family: 'Roboto Mono', monospace;
            overflow-x: hidden;
        }
        .loading-screen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            flex-direction: column;
        }
        .loading-logo {
            font-size: 3rem;
            color: #00ff88;
            font-weight: bold;
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
            margin-bottom: 20px;
            animation: pulse 2s infinite;
        }
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 3px solid rgba(0, 255, 136, 0.3);
            border-top: 3px solid #00ff88;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <!-- Loading Screen -->
    <div class="loading-screen" id="loading-screen">
        <div class="loading-logo">NOBELBRETT</div>
        <div class="loading-spinner"></div>
        <div style="margin-top: 20px; color: #b0b0b0;">Initialisiere Trading Dashboard...</div>
    </div>

    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">
                <i class="fas fa-chart-line"></i> NOBELBRETT
            </a>
            <div class="navbar-nav ms-auto">
                <span class="navbar-text">
                    <i class="fas fa-circle real-time-indicator"></i>
                    <span id="connection-status">Verbindung aufbauen...</span>
                </span>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4" style="display: none;" id="main-content">
        <!-- System Overview Row -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h6 class="card-title">
                            <i class="fas fa-heartbeat"></i> System Status
                        </h6>
                        <h3 id="system-status" class="text-success">
                            <span class="status-indicator status-healthy"></span>Online
                        </h3>
                        <small class="text-muted">CPU: <span id="cpu-usage">0%</span> | RAM: <span id="memory-usage">0%</span></small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h6 class="card-title">
                            <i class="fas fa-wallet"></i> Portfolio Wert
                        </h6>
                        <h3 id="account-balance" class="text-warning">$0.00</h3>
                        <small class="text-muted">Verfügbar: <span id="available-balance">$0.00</span></small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h6 class="card-title">
                            <i class="fas fa-robot"></i> KI Konfidenz
                        </h6>
                        <h3 id="ai-confidence" class="text-info">0%</h3>
                        <small class="text-muted">Signale: <span id="total-signals">0</span></small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h6 class="card-title">
                            <i class="fas fa-chart-pie"></i> Aktive Positionen
                        </h6>
                        <h3 id="open-positions" class="text-primary">0</h3>
                        <small class="text-muted">P&L: <span id="daily-pnl">$0.00</span></small>
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="row mb-4">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-line"></i> Portfolio Performance</h5>
                    </div>
                    <div class="card-body">
                        <div id="performance-chart" class="chart-container"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-brain"></i> KI Signal Verteilung</h5>
                    </div>
                    <div class="card-body">
                        <div id="ai-chart" class="chart-container"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Data Tables Row -->
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-exclamation-triangle"></i> Aktive Alerts</h5>
                    </div>
                    <div class="card-body">
                        <div id="alerts-container">
                            <p class="text-muted">Keine aktiven Alerts</p>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-history"></i> Letzte KI Entscheidungen</h5>
                    </div>
                    <div class="card-body">
                        <div id="ai-decisions-container">
                            <p class="text-muted">Keine Entscheidungen verfügbar</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <footer class="mt-5 py-4" style="border-top: 1px solid var(--primary-color);">
            <div class="row">
                <div class="col-md-6">
                    <p class="text-muted">&copy; 2024 NOBELBRETT - Professional Trading Platform</p>
                </div>
                <div class="col-md-6 text-end">
                    <p class="text-muted">Powered by TRADINO AI</p>
                </div>
            </div>
        </footer>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // WebSocket connection
        let socket = null;
        let reconnectInterval = null;

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/nobelbrett_client`;
            
            socket = new WebSocket(wsUrl);
            
            socket.onopen = function(event) {
                console.log('✅ WebSocket verbunden');
                document.getElementById('connection-status').textContent = 'Live Verbindung';
                clearInterval(reconnectInterval);
                
                // Subscribe to all updates
                socket.send(JSON.stringify({
                    type: 'subscription',
                    subscription: 'all'
                }));
            };
            
            socket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleRealtimeUpdate(data);
            };
            
            socket.onclose = function(event) {
                console.log('❌ WebSocket getrennt');
                document.getElementById('connection-status').textContent = 'Getrennt';
                
                // Attempt to reconnect
                reconnectInterval = setInterval(connectWebSocket, 5000);
            };
            
            socket.onerror = function(error) {
                console.error('WebSocket Fehler:', error);
                document.getElementById('connection-status').textContent = 'Fehler';
            };
        }

        function handleRealtimeUpdate(data) {
            if (data.update_type === 'dashboard_data') {
                updateDashboard(data.data);
            }
        }

        function updateDashboard(data) {
            // Update system health
            const systemHealth = data.system_health;
            document.getElementById('cpu-usage').textContent = systemHealth.cpu_usage.toFixed(1) + '%';
            document.getElementById('memory-usage').textContent = systemHealth.memory_usage.toFixed(1) + '%';
            
            // Update trading status
            const tradingStatus = data.trading_status;
            document.getElementById('account-balance').textContent = '$' + tradingStatus.account_balance.toLocaleString();
            document.getElementById('available-balance').textContent = '$' + tradingStatus.available_balance.toLocaleString();
            document.getElementById('open-positions').textContent = tradingStatus.open_positions;
            document.getElementById('daily-pnl').textContent = '$' + tradingStatus.daily_pnl.toFixed(2);
            
            // Update AI performance
            const aiPerformance = data.ai_performance;
            document.getElementById('ai-confidence').textContent = (aiPerformance.overall_confidence * 100).toFixed(1) + '%';
            document.getElementById('total-signals').textContent = aiPerformance.total_signals;
            
            // Update alerts
            updateAlerts(data.alerts);
            
            // Update AI decisions
            updateAIDecisions(aiPerformance.recent_decisions);
        }

        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');
            if (alerts.length === 0) {
                container.innerHTML = '<p class="text-muted">Keine aktiven Alerts</p>';
                return;
            }
            
            let html = '';
            alerts.forEach(alert => {
                const severityClass = {
                    'low': 'info',
                    'medium': 'warning', 
                    'high': 'danger',
                    'critical': 'danger'
                }[alert.severity] || 'secondary';
                
                html += `
                    <div class="alert alert-${severityClass} alert-dismissible">
                        <strong>${alert.category.toUpperCase()}:</strong> ${alert.message}
                        <small class="d-block">${new Date(alert.timestamp).toLocaleTimeString()}</small>
                    </div>
                `;
            });
            container.innerHTML = html;
        }

        function updateAIDecisions(decisions) {
            const container = document.getElementById('ai-decisions-container');
            if (decisions.length === 0) {
                container.innerHTML = '<p class="text-muted">Keine Entscheidungen verfügbar</p>';
                return;
            }
            
            let html = '<div class="list-group list-group-flush">';
            decisions.slice(0, 5).forEach(decision => {
                const signalClass = {
                    'BUY': 'success',
                    'SELL': 'danger',
                    'HOLD': 'warning'
                }[decision.signal] || 'secondary';
                
                html += `
                    <div class="list-group-item bg-transparent">
                        <div class="d-flex justify-content-between">
                            <span class="badge bg-${signalClass}">${decision.signal}</span>
                            <small>${decision.timestamp}</small>
                        </div>
                        <small class="text-muted">Konfidenz: ${(decision.confidence * 100).toFixed(1)}%</small>
                    </div>
                `;
            });
            html += '</div>';
            container.innerHTML = html;
        }

        // Load charts
        async function loadCharts() {
            try {
                // Load performance chart
                const perfResponse = await fetch('/api/charts/performance');
                const perfData = await perfResponse.json();
                if (perfData.chart) {
                    Plotly.newPlot('performance-chart', JSON.parse(perfData.chart).data, JSON.parse(perfData.chart).layout);
                }
                
                // Load AI chart
                const aiResponse = await fetch('/api/charts/ai-analysis');
                const aiData = await aiResponse.json();
                if (aiData.chart) {
                    Plotly.newPlot('ai-chart', JSON.parse(aiData.chart).data, JSON.parse(aiData.chart).layout);
                }
            } catch (error) {
                console.error('Fehler beim Laden der Charts:', error);
            }
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            // Hide loading screen
            setTimeout(() => {
                document.getElementById('loading-screen').style.display = 'none';
                document.getElementById('main-content').style.display = 'block';
                
                // Initialize connections and charts
                connectWebSocket();
                loadCharts();
                
                // Reload charts every 5 minutes
                setInterval(loadCharts, 5 * 60 * 1000);
            }, 2000);
        });
    </script>
</body>
</html>'''
    
    def run_production(self):
        """🚀 Run in production mode for nobelbrett.de"""
        print("🌐 Starting NOBELBRETT.DE Trading Dashboard...")
        print(f"🎯 Domain: {self.domain_config['domain']}")
        print(f"🎨 Theme: Professional Crypto Design")
        print(f"🔒 Security: HTTPS + Rate Limiting")
        
        # Create HTML template with branding
        template_path = Path("templates/dashboard.html")
        template_path.parent.mkdir(exist_ok=True)
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(self._get_branded_html_template())
        
        # Start in production mode
        try:
            import uvicorn
            uvicorn.run(
                self.app,
                host=self.domain_config['host'],
                port=self.domain_config['port'],
                ssl_keyfile=None,  # Will be handled by reverse proxy
                ssl_certfile=None,
                log_level="info",
                access_log=True,
                server_header=False,
                workers=1  # Single worker for WebSocket support
            )
        except KeyboardInterrupt:
            print("\n🛑 Dashboard gestoppt")
        except Exception as e:
            print(f"❌ Dashboard Fehler: {e}")

def create_nginx_config():
    """🌐 Create Nginx configuration for nobelbrett.de"""
    nginx_config = """
# NOBELBRETT.DE - Trading Dashboard Nginx Configuration
server {
    listen 80;
    server_name www.nobelbrett.de nobelbrett.de;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name www.nobelbrett.de nobelbrett.de;
    
    # SSL Configuration (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/www.nobelbrett.de/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/www.nobelbrett.de/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000" always;
    
    # Dashboard Application
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
    }
    
    # WebSocket Support
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
    
    # Static Files Caching
    location /static/ {
        proxy_pass http://127.0.0.1:8000;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Logs
    access_log /var/log/nginx/nobelbrett_access.log;
    error_log /var/log/nginx/nobelbrett_error.log;
}
"""
    
    nginx_file = Path("/etc/nginx/sites-available/nobelbrett.de")
    try:
        with open(nginx_file, 'w') as f:
            f.write(nginx_config)
        print(f"✅ Nginx config created: {nginx_file}")
        
        # Enable site
        enabled_link = Path("/etc/nginx/sites-enabled/nobelbrett.de")
        if not enabled_link.exists():
            enabled_link.symlink_to(nginx_file)
            print("✅ Site enabled in Nginx")
    except PermissionError:
        print("⚠️ Need root permissions for Nginx config")
        print("Run manually: sudo python deploy/nobelbrett_dashboard.py --setup-nginx")

def create_systemd_service():
    """🔧 Create systemd service for auto-start"""
    service_config = """
[Unit]
Description=NOBELBRETT Trading Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/tradino
Environment=PATH=/root/tradino/tradino_env/bin
ExecStart=/root/tradino/tradino_env/bin/python deploy/nobelbrett_dashboard.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    service_file = Path("/etc/systemd/system/nobelbrett-dashboard.service")
    try:
        with open(service_file, 'w') as f:
            f.write(service_config)
        print(f"✅ Systemd service created: {service_file}")
        
        # Reload and enable
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "enable", "nobelbrett-dashboard"], check=True)
        print("✅ Service enabled for auto-start")
    except PermissionError:
        print("⚠️ Need root permissions for systemd service")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Systemctl error: {e}")

def run_dashboard():
    """🚀 Run Nobelbrett Dashboard"""
    dashboard = NobelBrettDashboard()
    print("🌐 Starting NOBELBRETT.DE Trading Dashboard...")
    print("🎨 Professional Crypto Design loaded")
    dashboard.run()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NOBELBRETT.DE Trading Dashboard")
    parser.add_argument("--setup-nginx", action="store_true", help="Setup Nginx configuration")
    parser.add_argument("--setup-service", action="store_true", help="Setup systemd service")
    parser.add_argument("--production", action="store_true", help="Run in production mode")
    
    args = parser.parse_args()
    
    if args.setup_nginx:
        create_nginx_config()
    elif args.setup_service:
        create_systemd_service()
    elif args.production:
        run_dashboard()
    else:
        print("🌐 NOBELBRETT.DE Dashboard Setup")
        print("Available options:")
        print("  --setup-nginx     Setup Nginx reverse proxy")
        print("  --setup-service   Setup systemd auto-start service")
        print("  --production      Run dashboard in production mode")
        print("\nFor immediate testing: python deploy/nobelbrett_dashboard.py --production") 