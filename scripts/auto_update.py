#!/usr/bin/env python3
"""
🔄 TRADINO UNSCHLAGBAR - Automatic Update System
================================================

Automatisches Update-System für GitHub Repository Synchronisation
"""

import subprocess
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

class TradinoAutoUpdater:
    """🤖 Automatischer Updater für TRADINO UNSCHLAGBAR"""
    
    def __init__(self):
        self.setup_logging()
        self.repo_path = Path.cwd()
        self.remote_name = "origin"
        self.branch_name = "main"
        
    def setup_logging(self):
        """Setup logging für Update-Prozess"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('auto_update.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_git_command(self, command: list) -> tuple:
        """Führt Git-Kommando aus und gibt Ergebnis zurück"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            self.logger.error(f"Git command failed: {e}")
            return False, "", str(e)
    
    def check_for_updates(self) -> bool:
        """Prüft ob Updates auf GitHub verfügbar sind"""
        self.logger.info("🔍 Checking for updates on GitHub...")
        
        # Fetch remote changes
        success, stdout, stderr = self.run_git_command([
            "git", "fetch", self.remote_name, self.branch_name
        ])
        
        if not success:
            self.logger.error(f"Failed to fetch: {stderr}")
            return False
        
        # Check if local is behind remote
        success, stdout, stderr = self.run_git_command([
            "git", "rev-list", "--count", f"HEAD..{self.remote_name}/{self.branch_name}"
        ])
        
        if success:
            commits_behind = int(stdout.strip()) if stdout.strip().isdigit() else 0
            if commits_behind > 0:
                self.logger.info(f"📥 {commits_behind} new commits available!")
                return True
            else:
                self.logger.info("✅ Repository is up to date!")
                return False
        
        return False
    
    def run_update(self) -> bool:
        """Führt kompletten Update-Prozess aus"""
        self.logger.info("🚀 Starting TRADINO UNSCHLAGBAR auto-update...")
        
        try:
            # Check for updates
            if not self.check_for_updates():
                return True  # No updates needed
            
            # Pull updates
            self.logger.info("�� Pulling updates from GitHub...")
            success, stdout, stderr = self.run_git_command([
                "git", "pull", self.remote_name, self.branch_name
            ])
            
            if success:
                self.logger.info("✅ Successfully pulled updates!")
                return True
            else:
                self.logger.error(f"Failed to pull updates: {stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Update failed: {e}")
            return False

def main():
    updater = TradinoAutoUpdater()
    success = updater.run_update()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
