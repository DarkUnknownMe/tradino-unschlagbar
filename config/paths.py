#!/usr/bin/env python3
"""
🗂️ TRADINO PATH CONFIGURATION
Cross-Platform Path Management für alle TRADINO Komponenten

Automatische Pfad-Erkennung, Umgebungsvariablen-Support und
plattformübergreifende Kompatibilität für Windows, Linux und macOS.
"""

import os
import sys
import platform
from pathlib import Path
from typing import Dict, Optional, Union, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SystemInfo:
    """📊 System Information Container"""
    platform: str
    system: str
    machine: str
    python_version: str
    is_windows: bool
    is_linux: bool
    is_macos: bool
    path_separator: str
    home_dir: Path

class PathConfig:
    """
    🗂️ Zentrale Pfad-Konfiguration für TRADINO
    
    Automatische Erkennung der Projektstruktur und flexible
    Pfad-Verwaltung für verschiedene Betriebssysteme.
    """
    
    def __init__(self, custom_root: Optional[Union[str, Path]] = None):
        """
        Initialize PathConfig
        
        Args:
            custom_root: Optionaler custom root path für Testing
        """
        self._system_info = self._detect_system()
        self._root_dir = self._detect_project_root(custom_root)
        self._ensure_directories()
        
        logger.info(f"🗂️ PathConfig initialized for {self._system_info.system}")
        logger.info(f"📁 Project Root: {self._root_dir}")
    
    def _detect_system(self) -> SystemInfo:
        """🖥️ Detect system information"""
        system = platform.system()
        return SystemInfo(
            platform=platform.platform(),
            system=system,
            machine=platform.machine(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            is_windows=(system == "Windows"),
            is_linux=(system == "Linux"),
            is_macos=(system == "Darwin"),
            path_separator=os.sep,
            home_dir=Path.home()
        )
    
    def _detect_project_root(self, custom_root: Optional[Union[str, Path]] = None) -> Path:
        """
        🔍 Automatische Erkennung des Projekt-Root Verzeichnisses
        
        Sucht nach charakteristischen Dateien um das Root zu finden:
        - main.py
        - requirements.txt  
        - tradino.py
        - tradino_unschlagbar/ Verzeichnis
        """
        if custom_root:
            root = Path(custom_root).resolve()
            if root.exists():
                return root
            else:
                logger.warning(f"⚠️ Custom root {custom_root} does not exist, falling back to auto-detection")
        
        # Check environment variable first
        env_root = os.getenv('TRADINO_ROOT')
        if env_root:
            env_path = Path(env_root).resolve()
            if env_path.exists() and self._is_valid_tradino_root(env_path):
                logger.info(f"✅ Using TRADINO_ROOT environment variable: {env_path}")
                return env_path
        
        # Start from current file location and search upwards
        current = Path(__file__).parent.resolve()
        
        # Search upwards for project root indicators
        for _ in range(10):  # Limit search depth
            if self._is_valid_tradino_root(current):
                logger.info(f"✅ Auto-detected project root: {current}")
                return current
            
            parent = current.parent
            if parent == current:  # Reached filesystem root
                break
            current = parent
        
        # Fallback: use current file's parent's parent (config/../)
        fallback = Path(__file__).parent.parent.resolve()
        logger.warning(f"⚠️ Could not auto-detect root, using fallback: {fallback}")
        return fallback
    
    def _is_valid_tradino_root(self, path: Path) -> bool:
        """✅ Check if path is a valid TRADINO project root"""
        indicators = [
            'main.py',
            'requirements.txt',
            'tradino.py',
            'tradino_unschlagbar',
            'config',
            'core'
        ]
        
        found_indicators = sum(1 for indicator in indicators if (path / indicator).exists())
        return found_indicators >= 3  # Need at least 3 indicators
    
    def _ensure_directories(self) -> None:
        """📁 Ensure all required directories exist"""
        required_dirs = [
            self.logs_dir,
            self.data_dir,
            self.models_dir,
            self.config_dir,
            self.backup_dir,
            self.temp_dir
        ]
        
        for directory in required_dirs:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"📁 Ensured directory: {directory}")
            except Exception as e:
                logger.warning(f"⚠️ Could not create directory {directory}: {e}")
    
    # ==================
    # CORE PROPERTIES
    # ==================
    
    @property
    def system_info(self) -> SystemInfo:
        """🖥️ System information"""
        return self._system_info
    
    @property
    def root_dir(self) -> Path:
        """📁 Project root directory"""
        return self._root_dir
    
    @property
    def is_portable(self) -> bool:
        """🎒 Check if installation is portable (no hardcoded paths)"""
        return not any(str(self._root_dir).startswith(path) for path in ['/root/', 'C:\\Users\\'])
    
    # ==================
    # MAIN DIRECTORIES
    # ==================
    
    @property
    def config_dir(self) -> Path:
        """⚙️ Configuration directory"""
        return self._root_dir / "config"
    
    @property
    def core_dir(self) -> Path:
        """🔧 Core system directory"""
        return self._root_dir / "core"
    
    @property
    def tradino_unschlagbar_dir(self) -> Path:
        """🤖 Main trading system directory"""
        return self._root_dir / "tradino_unschlagbar"
    
    @property
    def scripts_dir(self) -> Path:
        """📜 Scripts directory"""
        return self._root_dir / "scripts"
    
    @property
    def tests_dir(self) -> Path:
        """🧪 Tests directory"""
        return self._root_dir / "tests"
    
    @property
    def logs_dir(self) -> Path:
        """📋 Logs directory"""
        return self._root_dir / "logs"
    
    @property
    def data_dir(self) -> Path:
        """💾 Data directory"""
        return self._root_dir / "data"
    
    @property
    def models_dir(self) -> Path:
        """🧠 ML Models directory"""
        return self._root_dir / "models"
    
    @property
    def backup_dir(self) -> Path:
        """💿 Backup directory"""
        return self._root_dir / "backup"
    
    @property
    def temp_dir(self) -> Path:
        """🗂️ Temporary files directory"""
        return self._root_dir / "temp"
    
    # ==================
    # TRADINO_UNSCHLAGBAR SUBDIRECTORIES
    # ==================
    
    @property
    def brain_dir(self) -> Path:
        """🧠 AI Brain directory"""
        return self.tradino_unschlagbar_dir / "brain"
    
    @property
    def connectors_dir(self) -> Path:
        """🔌 Connectors directory"""
        return self.tradino_unschlagbar_dir / "connectors"
    
    @property
    def strategies_dir(self) -> Path:
        """📈 Trading strategies directory"""
        return self.tradino_unschlagbar_dir / "strategies"
    
    @property
    def analytics_dir(self) -> Path:
        """📊 Analytics directory"""
        return self.tradino_unschlagbar_dir / "analytics"
    
    @property
    def tradino_core_dir(self) -> Path:
        """🔧 TRADINO core directory"""
        return self.tradino_unschlagbar_dir / "core"
    
    @property
    def tradino_utils_dir(self) -> Path:
        """🛠️ TRADINO utils directory"""
        return self.tradino_unschlagbar_dir / "utils"
    
    @property
    def tradino_models_dir(self) -> Path:
        """🧠 TRADINO models directory"""
        return self.tradino_unschlagbar_dir / "models"
    
    @property
    def tradino_data_dir(self) -> Path:
        """💾 TRADINO data directory"""
        return self.tradino_unschlagbar_dir / "data"
    
    @property
    def tradino_config_dir(self) -> Path:
        """⚙️ TRADINO config directory"""
        return self.tradino_unschlagbar_dir / "config"
    
    # ==================
    # SPECIFIC FILES
    # ==================
    
    @property
    def main_file(self) -> Path:
        """🚀 Main entry point"""
        return self._root_dir / "main.py"
    
    @property
    def tradino_file(self) -> Path:
        """🎯 TRADINO launcher"""
        return self._root_dir / "tradino.py"
    
    @property
    def requirements_file(self) -> Path:
        """📋 Requirements file"""
        return self._root_dir / "requirements.txt"
    
    @property
    def requirements_dev_file(self) -> Path:
        """📋 Development requirements"""
        return self._root_dir / "requirements-dev.txt"
    
    @property
    def requirements_minimal_file(self) -> Path:
        """📋 Minimal requirements"""
        return self._root_dir / "requirements-minimal.txt"
    
    @property
    def env_file(self) -> Path:
        """🔑 Environment variables file"""
        return self._root_dir / ".env"
    
    @property
    def gitignore_file(self) -> Path:
        """🚫 Git ignore file"""
        return self._root_dir / ".gitignore"
    
    # ==================
    # CONFIGURATION FILES
    # ==================
    
    @property
    def trading_config_file(self) -> Path:
        """💰 Trading configuration"""
        return self.tradino_config_dir / "final_trading_config.json"
    
    @property
    def risk_config_file(self) -> Path:
        """🛡️ Risk management configuration"""
        return self.tradino_config_dir / "risk_config.json"
    
    @property
    def system_config_file(self) -> Path:
        """⚙️ System configuration"""
        return self.config_dir / "system_config.yaml"
    
    # ==================
    # VIRTUAL ENVIRONMENT
    # ==================
    
    @property
    def venv_dir(self) -> Path:
        """🐍 Virtual environment directory"""
        # Common virtual environment names
        venv_names = ['tradino_env', 'venv', '.venv', 'env']
        
        for venv_name in venv_names:
            venv_path = self._root_dir / venv_name
            if venv_path.exists() and (venv_path / 'pyvenv.cfg').exists():
                return venv_path
        
        # Default to tradino_env if none found
        return self._root_dir / 'tradino_env'
    
    @property
    def venv_bin_dir(self) -> Path:
        """🐍 Virtual environment bin/Scripts directory"""
        if self._system_info.is_windows:
            return self.venv_dir / 'Scripts'
        else:
            return self.venv_dir / 'bin'
    
    @property
    def python_executable(self) -> Path:
        """🐍 Python executable in virtual environment"""
        if self._system_info.is_windows:
            return self.venv_bin_dir / 'python.exe'
        else:
            return self.venv_bin_dir / 'python'
    
    @property
    def pip_executable(self) -> Path:
        """📦 Pip executable in virtual environment"""
        if self._system_info.is_windows:
            return self.venv_bin_dir / 'pip.exe'
        else:
            return self.venv_bin_dir / 'pip'
    
    # ==================
    # UTILITY METHODS
    # ==================
    
    def get_relative_path(self, target: Union[str, Path]) -> Path:
        """
        📍 Get relative path from project root
        
        Args:
            target: Target path (absolute or relative)
            
        Returns:
            Relative path from project root
        """
        target_path = Path(target)
        
        if target_path.is_absolute():
            try:
                return target_path.relative_to(self._root_dir)
            except ValueError:
                # Path is not relative to root, return as-is
                return target_path
        else:
            return target_path
    
    def get_absolute_path(self, relative_path: Union[str, Path]) -> Path:
        """
        📍 Get absolute path from project root
        
        Args:
            relative_path: Relative path from project root
            
        Returns:
            Absolute path
        """
        return self._root_dir / relative_path
    
    def add_to_python_path(self) -> None:
        """
        🐍 Add project directories to Python path
        
        Adds project root and key directories to sys.path
        """
        paths_to_add = [
            self._root_dir,
            self.core_dir,
            self.tradino_unschlagbar_dir,
            self.scripts_dir,
            self.tests_dir
        ]
        
        for path in paths_to_add:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
                logger.debug(f"🐍 Added to Python path: {path_str}")
    
    def validate_structure(self) -> Dict[str, bool]:
        """
        ✅ Validate project structure
        
        Returns:
            Dictionary with validation results
        """
        checks = {
            'root_exists': self._root_dir.exists(),
            'config_dir_exists': self.config_dir.exists(),
            'core_dir_exists': self.core_dir.exists(),
            'tradino_unschlagbar_exists': self.tradino_unschlagbar_dir.exists(),
            'main_file_exists': self.main_file.exists(),
            'requirements_exists': self.requirements_file.exists(),
            'venv_exists': self.venv_dir.exists(),
            'python_executable_exists': self.python_executable.exists(),
        }
        
        return checks
    
    def get_environment_variables(self) -> Dict[str, str]:
        """
        🔑 Get recommended environment variables
        
        Returns:
            Dictionary with environment variable suggestions
        """
        return {
            'TRADINO_ROOT': str(self._root_dir),
            'TRADINO_CONFIG_DIR': str(self.config_dir),
            'TRADINO_LOGS_DIR': str(self.logs_dir),
            'TRADINO_DATA_DIR': str(self.data_dir),
            'TRADINO_MODELS_DIR': str(self.models_dir),
            'PYTHONPATH': f"{self._root_dir}{os.pathsep}{self.tradino_unschlagbar_dir}{os.pathsep}{self.core_dir}",
            'VIRTUAL_ENV': str(self.venv_dir) if self.venv_dir.exists() else '',
        }
    
    def export_to_file(self, filepath: Union[str, Path] = None) -> Path:
        """
        💾 Export path configuration to file
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path to exported file
        """
        if filepath is None:
            filepath = self.config_dir / 'path_export.json'
        
        filepath = Path(filepath)
        
        export_data = {
            'system_info': {
                'platform': self._system_info.platform,
                'system': self._system_info.system,
                'python_version': self._system_info.python_version,
                'is_portable': self.is_portable
            },
            'paths': {
                'root_dir': str(self._root_dir),
                'config_dir': str(self.config_dir),
                'logs_dir': str(self.logs_dir),
                'data_dir': str(self.data_dir),
                'models_dir': str(self.models_dir),
                'venv_dir': str(self.venv_dir),
            },
            'validation': self.validate_structure(),
            'environment_variables': self.get_environment_variables()
        }
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Path configuration exported to: {filepath}")
        return filepath
    
    def __str__(self) -> str:
        """String representation"""
        return f"PathConfig(root={self._root_dir}, system={self._system_info.system})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return f"PathConfig(root='{self._root_dir}', system='{self._system_info.system}', portable={self.is_portable})"

# ==================
# GLOBAL INSTANCE
# ==================

# Global PathConfig instance
_path_config: Optional[PathConfig] = None

def get_path_config(custom_root: Optional[Union[str, Path]] = None, 
                   force_reload: bool = False) -> PathConfig:
    """
    🗂️ Get global PathConfig instance
    
    Args:
        custom_root: Optional custom root directory
        force_reload: Force creation of new instance
        
    Returns:
        PathConfig instance
    """
    global _path_config
    
    if _path_config is None or force_reload:
        _path_config = PathConfig(custom_root)
        _path_config.add_to_python_path()
    
    return _path_config

def setup_tradino_paths(custom_root: Optional[Union[str, Path]] = None) -> PathConfig:
    """
    🚀 Setup TRADINO path configuration
    
    Convenience function to initialize and configure all paths.
    
    Args:
        custom_root: Optional custom root directory
        
    Returns:
        Configured PathConfig instance
    """
    config = get_path_config(custom_root)
    
    # Set environment variables
    env_vars = config.get_environment_variables()
    for key, value in env_vars.items():
        if key not in os.environ and value:
            os.environ[key] = value
    
    logger.info(f"🚀 TRADINO paths configured for {config.system_info.system}")
    return config

# ==================
# CONVENIENCE FUNCTIONS
# ==================

def get_project_root() -> Path:
    """📁 Get project root directory"""
    return get_path_config().root_dir

def get_logs_dir() -> Path:
    """📋 Get logs directory"""
    return get_path_config().logs_dir

def get_data_dir() -> Path:
    """💾 Get data directory"""
    return get_path_config().data_dir

def get_models_dir() -> Path:
    """🧠 Get models directory"""
    return get_path_config().models_dir

def get_config_dir() -> Path:
    """⚙️ Get config directory"""
    return get_path_config().config_dir

def get_python_executable() -> Path:
    """🐍 Get Python executable path"""
    return get_path_config().python_executable

def is_portable_installation() -> bool:
    """🎒 Check if installation is portable"""
    return get_path_config().is_portable

# ==================
# MAIN ENTRY POINT
# ==================

if __name__ == "__main__":
    """🔍 CLI interface for path configuration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TRADINO Path Configuration")
    parser.add_argument('--export', action='store_true', help='Export configuration to file')
    parser.add_argument('--validate', action='store_true', help='Validate project structure')
    parser.add_argument('--info', action='store_true', help='Show system information')
    parser.add_argument('--root', help='Custom project root directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Initialize path config
    config = setup_tradino_paths(args.root)
    
    if args.info:
        print(f"🖥️ System: {config.system_info.system} ({config.system_info.platform})")
        print(f"🐍 Python: {config.system_info.python_version}")
        print(f"📁 Root: {config.root_dir}")
        print(f"🎒 Portable: {config.is_portable}")
    
    if args.validate:
        validation = config.validate_structure()
        print("✅ Project Structure Validation:")
        for check, result in validation.items():
            status = "✅" if result else "❌"
            print(f"  {status} {check}: {result}")
    
    if args.export:
        export_path = config.export_to_file()
        print(f"📄 Configuration exported to: {export_path}")
    
    if not any([args.export, args.validate, args.info]):
        print(f"🗂️ {config}")
        print(f"📁 Project structure validated: {all(config.validate_structure().values())}") 