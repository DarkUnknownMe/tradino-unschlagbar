#!/usr/bin/env python3
"""
🤖 TRADINO UNSCHLAGBAR - AI INTEGRATION SETUP
Automatisierte Integration aller AI-Komponenten
"""

import os
import sys
import shutil
import json
from datetime import datetime
from pathlib import Path

class TradinoAISetup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.backup_dir = self.project_root / "backup_before_ai_integration"
        self.tradino_dir = self.project_root / "tradino_unschlagbar"
        
        print("🤖 TRADINO UNSCHLAGBAR - AI INTEGRATION SETUP")
        print("=" * 60)
        print(f"📁 Project Root: {self.project_root}")
        print(f"🔄 Integration Target: {self.tradino_dir}")
    
    def create_backup(self):
        """📦 Erstelle Backup der bestehenden Dateien"""
        print("\n📦 ERSTELLE BACKUP...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        # Backup wichtiger Dateien
        files_to_backup = [
            "alpha_smart_position_manager.py",
            "telegram_control_panel_enhanced.py",
            "comprehensive_system_test.py",
            "requirements.txt"
        ]
        
        self.backup_dir.mkdir(exist_ok=True)
        
        for file in files_to_backup:
            src = self.project_root / file
            if src.exists():
                dst = self.backup_dir / file
                shutil.copy2(src, dst)
                print(f"✅ Backup: {file}")
        
        print(f"✅ Backup erstellt in: {self.backup_dir}")
    
    def create_directory_structure(self):
        """📁 Erstelle AI-Verzeichnisstruktur"""
        print("\n📁 ERSTELLE VERZEICHNISSTRUKTUR...")
        
        directories = [
            "tradino_unschlagbar/brain",
            "tradino_unschlagbar/brain/models",
            "tradino_unschlagbar/brain/agents", 
            "tradino_unschlagbar/brain/nas",
            "tradino_unschlagbar/brain/training",
            "tradino_unschlagbar/core",
            "tradino_unschlagbar/utils",
            "tradino_unschlagbar/config",
            "tradino_unschlagbar/models",
            "tradino_unschlagbar/data",
            "tradino_unschlagbar/logs",
            "tradino_unschlagbar/tests",
            "scripts"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
    
    def create_init_files(self):
        """🔧 Erstelle __init__.py Dateien"""
        print("\n🔧 ERSTELLE __init__.py DATEIEN...")
        
        init_files = [
            "tradino_unschlagbar/__init__.py",
            "tradino_unschlagbar/brain/__init__.py",
            "tradino_unschlagbar/brain/models/__init__.py",
            "tradino_unschlagbar/brain/agents/__init__.py",
            "tradino_unschlagbar/brain/nas/__init__.py",
            "tradino_unschlagbar/brain/training/__init__.py",
            "tradino_unschlagbar/core/__init__.py",
            "tradino_unschlagbar/utils/__init__.py",
        ]
        
        for init_file in init_files:
            file_path = self.project_root / init_file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('"""TRADINO UNSCHLAGBAR AI Trading System"""\n')
            print(f"✅ Created: {init_file}")
    
    def update_requirements(self):
        """📋 Update requirements.txt"""
        print("\n📋 UPDATE REQUIREMENTS.TXT...")
        
        new_requirements = [
            "# AI/ML Dependencies",
            "torch>=1.11.0",
            "torchvision>=0.12.0", 
            "tensorflow>=2.8.0",
            "gym>=0.21.0",
            "stable-baselines3>=1.6.0",
            "optuna>=2.10.0",
            "",
            "# Data Science",
            "scikit-learn>=1.1.0",
            "xgboost>=1.6.0",
            "lightgbm>=3.3.0",
            "catboost>=1.0.0",
            "",
            "# Technical Analysis Enhanced",
            "ta>=0.10.2",
            "talib-binary>=0.4.19",
            "",
            "# Sentiment Analysis",
            "nltk>=3.7",
            "textblob>=0.17.1",
            "vaderSentiment>=3.3.2",
            "",
            "# Performance & Optimization",
            "numba>=0.56.0",
            "joblib>=1.1.0",
            "",
            "# Visualization Enhanced",
            "plotly>=5.8.0",
            "seaborn>=0.11.0",
            "",
            "# Configuration & Utilities",
            "pydantic>=1.9.0",
            "hydra-core>=1.1.0",
            "omegaconf>=2.1.0"
        ]
        
        # Lese bestehende requirements
        req_file = self.project_root / "requirements.txt"
        existing_reqs = []
        if req_file.exists():
            with open(req_file, 'r') as f:
                existing_reqs = f.readlines()
        
        # Schreibe erweiterte requirements
        with open(req_file, 'w', encoding='utf-8') as f:
            # Schreibe bestehende requirements
            for req in existing_reqs:
                f.write(req)
            
            # Füge neue requirements hinzu
            f.write("\n\n# AI TRADING SYSTEM REQUIREMENTS\n")
            for req in new_requirements:
                f.write(req + "\n")
        
        print("✅ Requirements.txt erweitert")
    
    def run_setup(self):
        """🚀 Führe komplettes Setup aus"""
        try:
            print("🚀 STARTE AI INTEGRATION SETUP...")
            
            # Bestätige dass wir im richtigen Verzeichnis sind
            if not (self.project_root / "alpha_smart_position_manager.py").exists():
                print("❌ FEHLER: Nicht im TRADINO UNSCHLAGBAR Projekt-Verzeichnis!")
                print("💡 Führen Sie dieses Script im Repository-Root aus")
                return False
            
            # Setup Steps
            self.create_backup()
            self.create_directory_structure()
            self.create_init_files()
            self.update_requirements()
            
            print("\n🎉 AI INTEGRATION SETUP ABGESCHLOSSEN!")
            print("=" * 60)
            print("📁 Verzeichnisstruktur erstellt")
            print("📦 Backup erstellt")
            print("📋 Requirements erweitert")
            print("\n🔄 NÄCHSTE SCHRITTE:")
            print("1. Führen Sie: python create_ai_components.py")
            print("2. Installieren Sie neue Dependencies: pip install -r requirements.txt")
            print("3. Starten Sie das AI System: python scripts/start_ai_trading.py")
            
            return True
            
        except Exception as e:
            print(f"❌ SETUP FEHLER: {e}")
            return False

if __name__ == "__main__":
    setup = TradinoAISetup()
    success = setup.run_setup()
    
    if success:
        print("\n✅ BEREIT FÜR AI-KOMPONENTEN INSTALLATION!")
    else:
        print("\n❌ SETUP FEHLGESCHLAGEN!")
        sys.exit(1)
