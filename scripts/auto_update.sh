#!/bin/bash

# 🔄 TRADINO UNSCHLAGBAR - Auto Update Script
# ============================================

echo "�� TRADINO UNSCHLAGBAR - Automatic Update System"
echo "================================================="

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# 1. Check for updates
log "🔍 Checking for updates on GitHub..."
git fetch origin main

# Compare local with remote
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    success "Repository is already up to date!"
    exit 0
fi

# 2. Show what will be updated
log "📥 New updates available!"
echo ""
echo "📋 Changes that will be applied:"
git log HEAD..origin/main --oneline --decorate
echo ""

# 3. Ask for confirmation (optional)
read -p "🤔 Do you want to apply these updates? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    warning "Update cancelled by user"
    exit 0
fi

# 4. Backup local changes (if any)
if ! git diff-index --quiet HEAD --; then
    log "💾 Backing up local changes..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    git stash push -m "Auto-backup before update $TIMESTAMP"
    success "Local changes backed up"
fi

# 5. Pull updates
log "📥 Pulling updates from GitHub..."
if git pull origin main; then
    success "Updates successfully applied!"
else
    error "Failed to pull updates"
    exit 1
fi

# 6. Update Python dependencies
if [ -f "requirements.txt" ]; then
    log "📦 Updating Python dependencies..."
    if pip install -r requirements.txt --upgrade; then
        success "Dependencies updated"
    else
        warning "Failed to update dependencies"
    fi
fi

# 7. Check if services need restart
log "🔄 Checking services..."
SERVICES=("telegram_control_panel_enhanced.py" "alpha_smart_position_manager.py")

for service in "${SERVICES[@]}"; do
    if [ -f "$service" ]; then
        warning "Service $service may need manual restart"
    fi
done

success "🎉 Update completed successfully!"
echo ""
echo "📊 Summary:"
echo "   • Repository updated to latest version"
echo "   • Dependencies refreshed"
echo "   • Local changes backed up (if any)"
echo ""
echo "💡 Don't forget to restart any running services!"
