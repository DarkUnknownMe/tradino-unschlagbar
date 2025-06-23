#!/usr/bin/env python3
"""
👁️ TRADINO UNSCHLAGBAR - Update Watcher
======================================

Überwacht GitHub Repository auf Änderungen und führt automatische Updates durch
"""

import subprocess
import time
import logging
import sys
from datetime import datetime

class UpdateWatcher:
    def __init__(self, check_interval=1800):  # 30 minutes default
        self.check_interval = check_interval
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('update_watcher.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_and_update(self):
        """Prüft auf Updates und führt sie aus"""
        try:
            self.logger.info("🔍 Checking for updates...")
            
            # Run the update script
            result = subprocess.run(['./auto_update.sh'], 
                                  capture_output=True, 
                                  text=True, 
                                  input='y\n')  # Auto-confirm updates
            
            if result.returncode == 0:
                if "already up to date" in result.stdout:
                    self.logger.info("✅ No updates available")
                else:
                    self.logger.info("🎉 Updates applied successfully!")
                    self.logger.info(f"Output: {result.stdout}")
            else:
                self.logger.error(f"❌ Update failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error during update check: {e}")
    
    def start_watching(self):
        """Startet den Update-Watcher"""
        self.logger.info(f"👁️ Starting update watcher (checking every {self.check_interval//60} minutes)")
        
        try:
            while True:
                self.check_and_update()
                self.logger.info(f"😴 Sleeping for {self.check_interval//60} minutes...")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Update watcher stopped by user")
        except Exception as e:
            self.logger.error(f"Watcher error: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    interval = 1800  # 30 minutes default
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1]) * 60  # Convert minutes to seconds
        except ValueError:
            print("Usage: python update_watcher.py [interval_in_minutes]")
            sys.exit(1)
    
    watcher = UpdateWatcher(interval)
    watcher.start_watching()
