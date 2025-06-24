#!/usr/bin/env python3
"""
🌐 DOMAIN CONFIGURATION - NOBELBRETT.DE

Domain-spezifische Konfiguration für das TRADINO Trading Dashboard
auf www.nobelbrett.de mit professionellem Krypto-Design.

Author: TRADINO Development Team
"""

import os
from typing import Dict, Any

# Domain Configuration
DOMAIN_CONFIG = {
    'domain': 'www.nobelbrett.de',
    'host': '0.0.0.0',  # Listen on all interfaces
    'port': 8000,
    'ssl_enabled': True,
    'ssl_redirect': True,
    'environment': 'production'
}

# Professional Crypto Branding
BRANDING_CONFIG = {
    'title': 'NOBELBRETT - Trading Dashboard',
    'subtitle': 'Professional Cryptocurrency Trading Platform',
    'logo_text': 'NOBELBRETT',
    'favicon_path': '/static/favicon.ico',
    'theme': 'crypto_professional',
    'colors': {
        'primary': '#00ff88',      # Neon Green
        'secondary': '#1a1a1a',    # Deep Black
        'background': '#0d0d0d',   # Darker Black
        'surface': '#1f1f1f',      # Card Background
        'text_primary': '#ffffff',  # White Text
        'text_secondary': '#b0b0b0', # Gray Text
        'success': '#00ff88',      # Neon Green
        'warning': '#ffaa00',      # Orange
        'danger': '#ff4444',       # Red
        'accent': '#00ccff'        # Cyber Blue
    }
}

# Security Configuration
SECURITY_CONFIG = {
    'https_only': True,
    'secure_headers': True,
    'rate_limiting': True,
    'cors_origins': ['https://www.nobelbrett.de', 'https://nobelbrett.de'],
    'allowed_hosts': ['www.nobelbrett.de', 'nobelbrett.de', 'localhost']
}

# Production Server Configuration
PRODUCTION_CONFIG = {
    'workers': 4,
    'worker_class': 'uvicorn.workers.UvicornWorker',
    'bind': '0.0.0.0:8000',
    'daemon': False,
    'user': 'root',
    'group': 'root',
    'timeout': 120,
    'keepalive': 5,
    'max_requests': 1000,
    'max_requests_jitter': 50,
    'log_level': 'info',
    'access_log': '/root/tradino/logs/access.log',
    'error_log': '/root/tradino/logs/error.log'
}

def get_domain_config() -> Dict[str, Any]:
    """🌐 Get complete domain configuration"""
    return {
        'domain': DOMAIN_CONFIG,
        'branding': BRANDING_CONFIG,
        'security': SECURITY_CONFIG,
        'production': PRODUCTION_CONFIG
    }

def get_custom_css() -> str:
    """🎨 Get custom CSS for Nobelbrett branding"""
    return """
    :root {
        --primary-color: #00ff88;
        --secondary-color: #1a1a1a;
        --background-color: #0d0d0d;
        --surface-color: #1f1f1f;
        --text-primary: #ffffff;
        --text-secondary: #b0b0b0;
        --success-color: #00ff88;
        --warning-color: #ffaa00;
        --danger-color: #ff4444;
        --accent-color: #00ccff;
        --glow-color: rgba(0, 255, 136, 0.3);
    }

    body {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 100%);
        color: var(--text-primary);
        font-family: 'Roboto Mono', monospace;
    }

    .navbar {
        background: linear-gradient(90deg, #000000 0%, #1a1a1a 50%, #000000 100%) !important;
        border-bottom: 2px solid var(--primary-color);
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.2);
    }

    .navbar-brand {
        color: var(--primary-color) !important;
        font-weight: bold;
        font-size: 1.5rem;
        text-shadow: 0 0 10px var(--glow-color);
    }

    .card {
        background: var(--surface-color) !important;
        border: 1px solid var(--primary-color) !important;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
    }

    .card-header {
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color)) !important;
        color: #000000 !important;
        font-weight: bold;
        border-radius: 12px 12px 0 0 !important;
    }

    .metric-card {
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.1), transparent);
        transition: left 0.5s;
    }

    .metric-card:hover::before {
        left: 100%;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 255, 136, 0.3);
        border-color: var(--accent-color) !important;
    }

    .status-indicator {
        box-shadow: 0 0 10px currentColor;
    }

    .status-healthy {
        background-color: var(--success-color) !important;
    }

    .status-warning {
        background-color: var(--warning-color) !important;
    }

    .status-error {
        background-color: var(--danger-color) !important;
    }

    .chart-container {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 8px;
        padding: 10px;
    }

    .real-time-indicator {
        animation: pulse 2s infinite;
        color: var(--success-color);
        text-shadow: 0 0 5px var(--glow-color);
    }

    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.05); }
        100% { opacity: 1; transform: scale(1); }
    }

    .btn-primary {
        background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
        border: none;
        color: #000000;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .btn-primary:hover {
        background: linear-gradient(45deg, var(--accent-color), var(--primary-color));
        box-shadow: 0 5px 15px var(--glow-color);
    }

    .alert {
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
    }

    .list-group-item {
        background: rgba(31, 31, 31, 0.8) !important;
        border-color: rgba(0, 255, 136, 0.2) !important;
        color: var(--text-primary);
    }

    .badge {
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 0.5px;
    }

    /* Crypto-style animations */
    @keyframes matrix-rain {
        0% { transform: translateY(-100vh); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateY(100vh); opacity: 0; }
    }

    /* Professional glow effects */
    .text-success { color: var(--success-color) !important; text-shadow: 0 0 5px var(--glow-color); }
    .text-warning { color: var(--warning-color) !important; }
    .text-danger { color: var(--danger-color) !important; }
    .text-info { color: var(--accent-color) !important; }
    .text-primary { color: var(--primary-color) !important; }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--background-color);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-color);
    }
    """

def get_favicon_svg() -> str:
    """🎯 Generate SVG favicon for Nobelbrett"""
    return """
    <svg width="32" height="32" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
        <rect width="32" height="32" fill="#000000"/>
        <path d="M8 8 L24 8 L20 16 L24 24 L8 24 L12 16 Z" fill="#00ff88" stroke="#00ccff" stroke-width="1"/>
        <circle cx="16" cy="16" r="2" fill="#ffffff"/>
    </svg>
    """ 