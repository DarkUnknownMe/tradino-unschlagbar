#!/bin/bash

# 🔧 TRADINO UNSCHLAGBAR - Auto Update Setup
# ===========================================

echo "🔧 Setting up automatic update system..."

# Create systemd service for automatic updates
cat > /etc/systemd/system/tradino-updater.service << 'SERVICE_EOF'
[Unit]
Description=TRADINO UNSCHLAGBAR Auto Updater
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/tradino
ExecStart=/usr/bin/python3 /root/tradino/update_watcher.py 30
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Create cron job for regular updates
echo "# TRADINO UNSCHLAGBAR Auto Update - Every 30 minutes" >> /etc/crontab
echo "*/30 * * * * root cd /root/tradino && ./auto_update.sh >/dev/null 2>&1" >> /etc/crontab

echo "✅ Auto update system configured!"
echo ""
echo "📋 Available options:"
echo "   1. Manual update:     ./auto_update.sh"
echo "   2. Watch mode:        python3 update_watcher.py [minutes]"
echo "   3. Systemd service:   systemctl enable tradino-updater"
echo "   4. Cron job:          Already configured (every 30 min)"
echo ""
echo "🚀 Your TRADINO system will now auto-update from GitHub!"
