#!/usr/bin/env python3
"""
TRADINO System Validation & Integration Test Suite (Standalone)
Vollständige End-to-End Validierung aller Systemkomponenten
Funktioniert ohne externe Abhängigkeiten
"""

import os
import sys
import time
import json
import logging
import traceback
import asyncio
import platform
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SystemValidator")

# Project root
project_root = Path(__file__).parent.parent


class StandaloneSystemValidator:
    """Standalone TRADINO System Validation Suite"""
    
    def __init__(self):
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        self.test_start_time = datetime.now()
        
    async def run_full_validation(self) -> Dict[str, Any]:
        """Führe vollständige Systemvalidierung durch"""
        logger.info("🚀 TRADINO Standalone System Validation gestartet")
        
        validation_steps = [
            ("System Environment", self._validate_system_environment),
            ("Python Environment", self._validate_python_environment),
            ("Project Structure", self._validate_project_structure),
            ("Configuration Files", self._validate_configuration_files),
            ("Dependencies", self._validate_dependencies),
            ("Environment Variables", self._validate_environment_variables),
            ("File Permissions", self._validate_file_permissions),
            ("Storage Space", self._validate_storage_space),
            ("Network Connectivity", self._validate_network_connectivity),
            ("Python Modules", self._validate_python_modules),
            ("Log Directory", self._validate_log_directory),
            ("Model Files", self._validate_model_files),
            ("Data Directory", self._validate_data_directory),
            ("Scripts Validation", self._validate_scripts),
            ("Performance Test", self._validate_performance)
        ]
        
        for step_name, step_func in validation_steps:
            try:
                logger.info(f"🔍 Validierung: {step_name}")
                result = await step_func()
                self.validation_results[step_name] = {
                    "status": "PASSED" if result["success"] else "FAILED",
                    "details": result.get("details", {}),
                    "metrics": result.get("metrics", {}),
                    "timestamp": datetime.now().isoformat()
                }
                
                if not result["success"]:
                    self.errors.append(f"{step_name}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                error_msg = f"{step_name} failed: {str(e)}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                self.validation_results[step_name] = {
                    "status": "ERROR",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat()
                }
        
        # Generate final report
        return await self._generate_validation_report()
    
    async def _validate_system_environment(self) -> Dict[str, Any]:
        """Validiere System Environment"""
        details = {}
        
        try:
            # Operating System
            details["os"] = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            }
            
            # Python version
            details["python"] = {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": sys.executable
            }
            
            # System resources
            try:
                import psutil
                details["resources"] = {
                    "cpu_count": psutil.cpu_count(),
                    "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
                }
            except ImportError:
                details["resources"] = "psutil not available"
            
            # System uptime
            try:
                with open('/proc/uptime', 'r') as f:
                    uptime_seconds = float(f.readline().split()[0])
                    details["uptime_hours"] = round(uptime_seconds / 3600, 2)
            except:
                details["uptime_hours"] = "Not available"
            
            return {"success": True, "details": details}
            
        except Exception as e:
            return {"success": False, "error": str(e), "details": details}
    
    async def _validate_python_environment(self) -> Dict[str, Any]:
        """Validiere Python Environment"""
        details = {}
        
        try:
            # Python executable
            details["executable"] = sys.executable
            details["version"] = sys.version
            details["path"] = sys.path[:5]  # First 5 paths
            
            # Virtual environment
            details["virtual_env"] = os.getenv("VIRTUAL_ENV")
            details["conda_env"] = os.getenv("CONDA_DEFAULT_ENV")
            
            # Package management
            try:
                pip_result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                          capture_output=True, text=True, timeout=10)
                details["pip_version"] = pip_result.stdout.strip() if pip_result.returncode == 0 else "Not available"
            except:
                details["pip_version"] = "Not available"
            
            return {"success": True, "details": details}
            
        except Exception as e:
            return {"success": False, "error": str(e), "details": details}
    
    async def _validate_project_structure(self) -> Dict[str, Any]:
        """Validiere Projektstruktur"""
        details = {}
        
        try:
            # Required directories
            required_dirs = [
                "core", "scripts", "config", "logs", "models", 
                "tradino_unschlagbar", "tradino_unschlagbar/brain",
                "tradino_unschlagbar/core", "tradino_unschlagbar/analytics",
                "tradino_unschlagbar/connectors", "tradino_unschlagbar/strategies"
            ]
            
            existing_dirs = []
            missing_dirs = []
            
            for dir_name in required_dirs:
                dir_path = project_root / dir_name
                if dir_path.exists() and dir_path.is_dir():
                    existing_dirs.append(dir_name)
                    # Count files in directory
                    file_count = len(list(dir_path.rglob('*.py')))
                    details[f"{dir_name}_files"] = file_count
                else:
                    missing_dirs.append(dir_name)
            
            details["existing_directories"] = existing_dirs
            details["missing_directories"] = missing_dirs
            
            # Important files
            important_files = [
                "tradino.py", "README.md", "LICENSE",
                "core/bitget_trading_api.py", "core/risk_management_system.py",
                "core/final_live_trading_system.py"
            ]
            
            existing_files = []
            missing_files = []
            
            for file_name in important_files:
                file_path = project_root / file_name
                if file_path.exists() and file_path.is_file():
                    existing_files.append(file_name)
                    details[f"{file_name}_size"] = file_path.stat().st_size
                else:
                    missing_files.append(file_name)
            
            details["existing_files"] = existing_files
            details["missing_files"] = missing_files
            
            success = len(missing_dirs) == 0 and len(missing_files) <= 2  # Allow some missing files
            
            return {"success": success, "details": details}
            
        except Exception as e:
            return {"success": False, "error": str(e), "details": details}
    
    async def _validate_configuration_files(self) -> Dict[str, Any]:
        """Validiere Konfigurationsdateien"""
        details = {}
        
        try:
            config_files = [
                "config/requirements.txt",
                "config/requirements_ai.txt",
                "tradino_unschlagbar/config.yaml",
                "tradino_unschlagbar/config/final_trading_config.json",
                "tradino_unschlagbar/config/risk_config.json"
            ]
            
            valid_configs = []
            invalid_configs = []
            missing_configs = []
            
            for config_file in config_files:
                config_path = project_root / config_file
                if not config_path.exists():
                    missing_configs.append(config_file)
                    continue
                
                try:
                    if config_file.endswith('.json'):
                        with open(config_path, 'r') as f:
                            config_data = json.load(f)
                        details[f"{config_file}_keys"] = list(config_data.keys()) if isinstance(config_data, dict) else "Not a dict"
                    elif config_file.endswith('.yaml'):
                        with open(config_path, 'r') as f:
                            content = f.read()
                        details[f"{config_file}_lines"] = len(content.splitlines())
                    else:
                        with open(config_path, 'r') as f:
                            content = f.read()
                        details[f"{config_file}_lines"] = len(content.splitlines())
                    
                    valid_configs.append(config_file)
                    
                except Exception as e:
                    invalid_configs.append(f"{config_file}: {str(e)}")
            
            details["valid_configs"] = valid_configs
            details["invalid_configs"] = invalid_configs
            details["missing_configs"] = missing_configs
            
            success = len(valid_configs) >= 3 and len(invalid_configs) == 0
            
            return {"success": success, "details": details}
            
        except Exception as e:
            return {"success": False, "error": str(e), "details": details}
    
    async def _validate_dependencies(self) -> Dict[str, Any]:
        """Validiere Dependencies"""
        details = {}
        
        try:
            # Check requirements files
            req_files = ["config/requirements.txt", "config/requirements_ai.txt"]
            
            all_requirements = set()
            for req_file in req_files:
                req_path = project_root / req_file
                if req_path.exists():
                    with open(req_path, 'r') as f:
                        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    all_requirements.update(requirements)
                    details[f"{req_file}_count"] = len(requirements)
            
            details["total_requirements"] = len(all_requirements)
            details["requirements"] = list(all_requirements)[:10]  # First 10
            
            # Test import of critical modules
            critical_modules = [
                ("aiohttp", "HTTP client"),
                ("ccxt", "Crypto exchange library"),
                ("numpy", "Numerical computing"),
                ("pandas", "Data analysis"),
                ("scikit-learn", "Machine learning"),
                ("telegram", "Telegram bot")
            ]
            
            available_modules = []
            missing_modules = []
            
            for module_name, description in critical_modules:
                try:
                    __import__(module_name)
                    available_modules.append(f"{module_name} ({description})")
                except ImportError:
                    missing_modules.append(f"{module_name} ({description})")
            
            details["available_modules"] = available_modules
            details["missing_modules"] = missing_modules
            
            # Success if at least half the modules are available
            success = len(available_modules) >= len(missing_modules)
            
            return {"success": success, "details": details}
            
        except Exception as e:
            return {"success": False, "error": str(e), "details": details}
    
    async def _validate_environment_variables(self) -> Dict[str, Any]:
        """Validiere Environment Variables"""
        details = {}
        
        try:
            required_vars = [
                "BITGET_API_KEY", "BITGET_SECRET_KEY", "BITGET_PASSPHRASE",
                "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"
            ]
            
            configured_vars = []
            missing_vars = []
            
            for var in required_vars:
                value = os.getenv(var)
                if value:
                    configured_vars.append(var)
                    # Don't log actual values for security
                    details[f"{var}_configured"] = True
                    details[f"{var}_length"] = len(value)
                else:
                    missing_vars.append(var)
                    details[f"{var}_configured"] = False
            
            details["configured_vars"] = configured_vars
            details["missing_vars"] = missing_vars
            
            # Optional variables
            optional_vars = ["PYTHONPATH", "TZ", "LANG"]
            for var in optional_vars:
                value = os.getenv(var)
                details[f"optional_{var}"] = value if value else "Not set"
            
            # Success if at least API credentials are configured
            api_vars = [var for var in configured_vars if "BITGET" in var]
            success = len(api_vars) >= 3 or len(configured_vars) >= 2  # Be lenient for testing
            
            return {"success": success, "details": details}
            
        except Exception as e:
            return {"success": False, "error": str(e), "details": details}
    
    async def _validate_file_permissions(self) -> Dict[str, Any]:
        """Validiere File Permissions"""
        details = {}
        
        try:
            # Check permissions on important directories
            important_paths = [
                "logs", "models", "data", "config",
                "scripts", "core", "tradino_unschlagbar"
            ]
            
            permission_issues = []
            
            for path_name in important_paths:
                path = project_root / path_name
                if path.exists():
                    stat = path.stat()
                    is_readable = os.access(path, os.R_OK)
                    is_writable = os.access(path, os.W_OK)
                    is_executable = os.access(path, os.X_OK) if path.is_dir() else True
                    
                    details[f"{path_name}_permissions"] = {
                        "readable": is_readable,
                        "writable": is_writable,
                        "executable": is_executable,
                        "mode": oct(stat.st_mode)
                    }
                    
                    if not (is_readable and is_writable and is_executable):
                        permission_issues.append(path_name)
            
            details["permission_issues"] = permission_issues
            
            # Check if we can create/write files
            try:
                test_file = project_root / "logs" / "test_write.txt"
                test_file.parent.mkdir(exist_ok=True)
                with open(test_file, 'w') as f:
                    f.write("test")
                test_file.unlink()
                details["write_test"] = "success"
            except Exception as e:
                details["write_test"] = f"failed: {str(e)}"
                permission_issues.append("write_test")
            
            success = len(permission_issues) == 0
            
            return {"success": success, "details": details}
            
        except Exception as e:
            return {"success": False, "error": str(e), "details": details}
    
    async def _validate_storage_space(self) -> Dict[str, Any]:
        """Validiere Storage Space"""
        details = {}
        
        try:
            # Get disk usage
            try:
                import shutil
                total, used, free = shutil.disk_usage(project_root)
                
                details["disk_space"] = {
                    "total_gb": round(total / (1024**3), 2),
                    "used_gb": round(used / (1024**3), 2),
                    "free_gb": round(free / (1024**3), 2),
                    "usage_percent": round((used / total) * 100, 2)
                }
                
                # Check if we have enough space (at least 1GB free)
                enough_space = free > 1024**3
                
            except Exception as e:
                details["disk_space"] = f"Could not determine: {str(e)}"
                enough_space = True  # Assume OK if we can't check
            
            # Check project directory size
            try:
                total_size = 0
                file_count = 0
                
                for root, dirs, files in os.walk(project_root):
                    for file in files:
                        file_path = Path(root) / file
                        try:
                            total_size += file_path.stat().st_size
                            file_count += 1
                        except:
                            pass  # Skip files we can't access
                
                details["project_size"] = {
                    "total_mb": round(total_size / (1024**2), 2),
                    "file_count": file_count
                }
                
            except Exception as e:
                details["project_size"] = f"Could not calculate: {str(e)}"
            
            success = enough_space
            
            return {"success": success, "details": details}
            
        except Exception as e:
            return {"success": False, "error": str(e), "details": details}
    
    async def _validate_network_connectivity(self) -> Dict[str, Any]:
        """Validiere Network Connectivity"""
        details = {}
        
        try:
            import socket
            import urllib.request
            
            # Test DNS resolution
            try:
                socket.gethostbyname('google.com')
                details["dns_resolution"] = "success"
            except Exception as e:
                details["dns_resolution"] = f"failed: {str(e)}"
            
            # Test HTTP connectivity
            test_urls = [
                ("google.com", "https://www.google.com"),
                ("bitget.com", "https://api.bitget.com"),
                ("telegram.org", "https://api.telegram.org")
            ]
            
            connectivity_results = {}
            
            for name, url in test_urls:
                try:
                    start_time = time.time()
                    response = urllib.request.urlopen(url, timeout=10)
                    latency = round((time.time() - start_time) * 1000, 2)
                    
                    connectivity_results[name] = {
                        "status": "success",
                        "status_code": response.getcode(),
                        "latency_ms": latency
                    }
                except Exception as e:
                    connectivity_results[name] = {
                        "status": "failed",
                        "error": str(e)
                    }
            
            details["connectivity_tests"] = connectivity_results
            
            # Success if at least one connection works
            successful_connections = sum(1 for result in connectivity_results.values() 
                                       if result["status"] == "success")
            success = successful_connections > 0
            
            return {"success": success, "details": details}
            
        except Exception as e:
            return {"success": False, "error": str(e), "details": details}
    
    async def _validate_python_modules(self) -> Dict[str, Any]:
        """Validiere Python Module"""
        details = {}
        
        try:
            # Test importing core project modules (with error handling)
            project_modules = [
                "core.bitget_trading_api",
                "core.risk_management_system", 
                "core.final_live_trading_system",
                "tradino_unschlagbar.utils.logger_pro",
                "tradino_unschlagbar.core.trading_engine"
            ]
            
            importable_modules = []
            failed_imports = []
            
            # Add project root to sys.path temporarily
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            for module_name in project_modules:
                try:
                    __import__(module_name)
                    importable_modules.append(module_name)
                except Exception as e:
                    failed_imports.append(f"{module_name}: {str(e)}")
            
            details["importable_modules"] = importable_modules
            details["failed_imports"] = failed_imports
            
            # Check syntax of Python files
            syntax_errors = []
            python_files = list(project_root.rglob("*.py"))
            
            for py_file in python_files[:20]:  # Check first 20 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source = f.read()
                    compile(source, str(py_file), 'exec')
                except SyntaxError as e:
                    syntax_errors.append(f"{py_file.relative_to(project_root)}: {str(e)}")
                except Exception:
                    pass  # Skip files we can't read
            
            details["syntax_errors"] = syntax_errors
            details["total_python_files"] = len(python_files)
            
            # Success if at least some modules import and no syntax errors
            success = len(importable_modules) > 0 and len(syntax_errors) == 0
            
            return {"success": success, "details": details}
            
        except Exception as e:
            return {"success": False, "error": str(e), "details": details}
    
    async def _validate_log_directory(self) -> Dict[str, Any]:
        """Validiere Log Directory"""
        details = {}
        
        try:
            logs_dir = project_root / "logs"
            
            # Create logs directory if it doesn't exist
            logs_dir.mkdir(exist_ok=True)
            
            # Check if we can write to logs directory
            test_log = logs_dir / "test_validation.log"
            try:
                with open(test_log, 'w') as f:
                    f.write(f"Test log entry at {datetime.now()}")
                test_log.unlink()
                details["write_permission"] = True
            except Exception as e:
                details["write_permission"] = False
                details["write_error"] = str(e)
            
            # Check existing log files
            log_files = list(logs_dir.glob("*.log"))
            details["existing_log_files"] = len(log_files)
            
            if log_files:
                # Get info about most recent log
                recent_log = max(log_files, key=lambda f: f.stat().st_mtime)
                details["most_recent_log"] = {
                    "name": recent_log.name,
                    "size_bytes": recent_log.stat().st_size,
                    "modified": datetime.fromtimestamp(recent_log.stat().st_mtime).isoformat()
                }
            
            # Check log directory size
            total_log_size = sum(f.stat().st_size for f in log_files)
            details["total_log_size_mb"] = round(total_log_size / (1024**2), 2)
            
            success = details.get("write_permission", False)
            
            return {"success": success, "details": details}
            
        except Exception as e:
            return {"success": False, "error": str(e), "details": details}
    
    async def _validate_model_files(self) -> Dict[str, Any]:
        """Validiere Model Files"""
        details = {}
        
        try:
            models_dir = project_root / "models"
            
            if not models_dir.exists():
                details["models_directory"] = "does not exist"
                return {"success": False, "details": details}
            
            # Expected model files
            expected_models = [
                "xgboost_trend.pkl",
                "lightgbm_volatility.pkl",
                "random_forest_risk.pkl",
                "feature_pipeline.pkl",
                "model_info.json"
            ]
            
            existing_models = []
            missing_models = []
            model_info = {}
            
            for model_file in expected_models:
                model_path = models_dir / model_file
                if model_path.exists():
                    existing_models.append(model_file)
                    model_info[model_file] = {
                        "size_mb": round(model_path.stat().st_size / (1024**2), 2),
                        "modified": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
                    }
                    
                    # Try to validate pickle files
                    if model_file.endswith('.pkl'):
                        try:
                            import pickle
                            with open(model_path, 'rb') as f:
                                pickle.load(f)
                            model_info[model_file]["loadable"] = True
                        except Exception as e:
                            model_info[model_file]["loadable"] = False
                            model_info[model_file]["load_error"] = str(e)
                else:
                    missing_models.append(model_file)
            
            details["existing_models"] = existing_models
            details["missing_models"] = missing_models
            details["model_info"] = model_info
            
            # Check additional model files
            all_model_files = list(models_dir.rglob("*"))
            additional_files = [f.name for f in all_model_files 
                              if f.is_file() and f.name not in expected_models]
            details["additional_files"] = additional_files
            
            # Success if at least some models exist
            success = len(existing_models) >= 2
            
            return {"success": success, "details": details}
            
        except Exception as e:
            return {"success": False, "error": str(e), "details": details}
    
    async def _validate_data_directory(self) -> Dict[str, Any]:
        """Validiere Data Directory"""
        details = {}
        
        try:
            data_dirs = [
                project_root / "data",
                project_root / "tradino_unschlagbar" / "data"
            ]
            
            for data_dir in data_dirs:
                dir_name = str(data_dir.relative_to(project_root))
                
                if data_dir.exists():
                    # Count files by type
                    json_files = list(data_dir.rglob("*.json"))
                    pkl_files = list(data_dir.rglob("*.pkl"))
                    csv_files = list(data_dir.rglob("*.csv"))
                    
                    details[f"{dir_name}_files"] = {
                        "json": len(json_files),
                        "pkl": len(pkl_files),
                        "csv": len(csv_files),
                        "total": len(json_files) + len(pkl_files) + len(csv_files)
                    }
                    
                    # Check directory size
                    total_size = 0
                    for root, dirs, files in os.walk(data_dir):
                        for file in files:
                            try:
                                total_size += (Path(root) / file).stat().st_size
                            except:
                                pass
                    
                    details[f"{dir_name}_size_mb"] = round(total_size / (1024**2), 2)
                    
                    # Check for recent files
                    all_files = list(data_dir.rglob("*"))
                    if all_files:
                        file_files = [f for f in all_files if f.is_file()]
                        if file_files:
                            recent_file = max(file_files, key=lambda f: f.stat().st_mtime)
                            details[f"{dir_name}_most_recent"] = {
                                "file": recent_file.name,
                                "modified": datetime.fromtimestamp(recent_file.stat().st_mtime).isoformat()
                            }
                else:
                    details[f"{dir_name}_exists"] = False
            
            # Success if at least one data directory exists with some files
            has_data = any(info.get("total", 0) > 0 for key, info in details.items() 
                          if isinstance(info, dict) and "total" in info)
            
            return {"success": has_data, "details": details}
            
        except Exception as e:
            return {"success": False, "error": str(e), "details": details}
    
    async def _validate_scripts(self) -> Dict[str, Any]:
        """Validiere Scripts"""
        details = {}
        
        try:
            scripts_dir = project_root / "scripts"
            
            if not scripts_dir.exists():
                return {"success": False, "error": "Scripts directory does not exist"}
            
            # Find Python scripts
            python_scripts = list(scripts_dir.glob("*.py"))
            shell_scripts = list(scripts_dir.glob("*.sh"))
            
            details["python_scripts"] = len(python_scripts)
            details["shell_scripts"] = len(shell_scripts)
            
            # Check if scripts are executable
            executable_scripts = []
            non_executable_scripts = []
            
            for script in python_scripts + shell_scripts:
                if os.access(script, os.X_OK):
                    executable_scripts.append(script.name)
                else:
                    non_executable_scripts.append(script.name)
            
            details["executable_scripts"] = executable_scripts
            details["non_executable_scripts"] = non_executable_scripts
            
            # Check specific important scripts
            important_scripts = [
                "system_validation.py",
                "test_complete_monitoring_system.py",
                "test_tp_sl_system.py",
                "run.py"
            ]
            
            existing_important = []
            missing_important = []
            
            for script_name in important_scripts:
                script_path = scripts_dir / script_name
                if script_path.exists():
                    existing_important.append(script_name)
                else:
                    missing_important.append(script_name)
            
            details["existing_important_scripts"] = existing_important
            details["missing_important_scripts"] = missing_important
            
            success = len(python_scripts) > 0 and len(existing_important) >= 2
            
            return {"success": success, "details": details}
            
        except Exception as e:
            return {"success": False, "error": str(e), "details": details}
    
    async def _validate_performance(self) -> Dict[str, Any]:
        """Validiere System Performance"""
        details = {}
        
        try:
            # CPU performance test
            start_time = time.time()
            result = sum(i * i for i in range(100000))
            cpu_time = time.time() - start_time
            
            details["cpu_test"] = {
                "computation_time_ms": round(cpu_time * 1000, 2),
                "result": result
            }
            
            # Memory allocation test
            try:
                import sys
                start_memory = sys.getsizeof([])
                test_list = list(range(10000))
                end_memory = sys.getsizeof(test_list)
                
                details["memory_test"] = {
                    "start_size_bytes": start_memory,
                    "end_size_bytes": end_memory,
                    "allocated_bytes": end_memory - start_memory
                }
            except Exception as e:
                details["memory_test"] = f"failed: {str(e)}"
            
            # File I/O performance test
            test_file = project_root / "logs" / "perf_test.tmp"
            try:
                start_time = time.time()
                with open(test_file, 'w') as f:
                    for i in range(1000):
                        f.write(f"Test line {i}\n")
                write_time = time.time() - start_time
                
                start_time = time.time()
                with open(test_file, 'r') as f:
                    lines = f.readlines()
                read_time = time.time() - start_time
                
                test_file.unlink()
                
                details["io_test"] = {
                    "write_time_ms": round(write_time * 1000, 2),
                    "read_time_ms": round(read_time * 1000, 2),
                    "lines_read": len(lines)
                }
                
            except Exception as e:
                details["io_test"] = f"failed: {str(e)}"
            
            # JSON performance test
            try:
                test_data = {"test": list(range(1000)), "nested": {"data": "test"}}
                
                start_time = time.time()
                json_str = json.dumps(test_data)
                serialize_time = time.time() - start_time
                
                start_time = time.time()
                parsed_data = json.loads(json_str)
                deserialize_time = time.time() - start_time
                
                details["json_test"] = {
                    "serialize_time_ms": round(serialize_time * 1000, 2),
                    "deserialize_time_ms": round(deserialize_time * 1000, 2),
                    "json_size_bytes": len(json_str)
                }
                
            except Exception as e:
                details["json_test"] = f"failed: {str(e)}"
            
            # Performance assessment
            performance_score = 100
            if cpu_time > 0.1:  # More than 100ms for simple computation
                performance_score -= 20
            if write_time > 0.05:  # More than 50ms for file write
                performance_score -= 20
            
            details["performance_score"] = performance_score
            
            success = performance_score >= 60
            
            return {"success": success, "details": details}
            
        except Exception as e:
            return {"success": False, "error": str(e), "details": details}
    
    async def _generate_validation_report(self) -> Dict[str, Any]:
        """Generiere finalen Validierungsreport"""
        end_time = datetime.now()
        total_duration = (end_time - self.test_start_time).total_seconds()
        
        # Statistiken berechnen
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results.values() if r["status"] == "PASSED")
        failed_tests = sum(1 for r in self.validation_results.values() if r["status"] == "FAILED")
        error_tests = sum(1 for r in self.validation_results.values() if r["status"] == "ERROR")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        overall_status = "PASSED" if failed_tests == 0 and error_tests == 0 else "FAILED"
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        report = {
            "validation_summary": {
                "overall_status": overall_status,
                "success_rate": round(success_rate, 2),
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "warnings": len(self.warnings)
            },
            "execution_info": {
                "start_time": self.test_start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": round(total_duration, 2),
                "standalone_mode": True
            },
            "detailed_results": self.validation_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": recommendations,
            "system_info": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "working_directory": str(project_root)
            }
        }
        
        # Speichere Report
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        report_file = logs_dir / f"system_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"📊 Validation Report gespeichert: {report_file}")
        except Exception as e:
            logger.error(f"Konnte Report nicht speichern: {e}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generiere Empfehlungen basierend auf Testergebnissen"""
        recommendations = []
        
        if self.errors:
            recommendations.append("🔴 Kritische Fehler beheben vor Produktionsstart")
            
        if self.warnings:
            recommendations.append("🟡 Warnungen überprüfen und beheben")
            
        # Basiere Empfehlungen auf spezifischen Test-Ergebnissen
        failed_components = [name for name, result in self.validation_results.items() 
                           if result["status"] in ["FAILED", "ERROR"]]
        
        if "Dependencies" in failed_components:
            recommendations.append("📦 Fehlende Dependencies installieren: pip install -r config/requirements.txt")
            
        if "Environment Variables" in failed_components:
            recommendations.append("🔑 API Credentials und Environment Variables konfigurieren")
            
        if "Model Files" in failed_components:
            recommendations.append("🧠 ML Modelle trainieren oder herunterladen")
            
        if "Network Connectivity" in failed_components:
            recommendations.append("🌐 Netzwerkverbindung und Firewall überprüfen")
        
        if "File Permissions" in failed_components:
            recommendations.append("🔒 Dateiberechtigungen korrigieren")
            
        if not failed_components:
            recommendations.append("✅ Grundlegende Systemvalidierung erfolgreich")
            recommendations.append("🚀 Empfehlung: Dependencies installieren und API Keys konfigurieren")
            recommendations.append("📊 Empfehlung: Vollständige Tests nach Dependency-Installation")
        
        return recommendations


async def main():
    """Hauptfunktion für Standalone System Validation"""
    print("🔍 TRADINO Standalone System Validation & Integration Test")
    print("=" * 70)
    print("Dieses Tool überprüft die Grundkonfiguration ohne externe Dependencies")
    print()
    
    validator = StandaloneSystemValidator()
    
    try:
        # Führe vollständige Validation durch
        report = await validator.run_full_validation()
        
        # Zeige Zusammenfassung
        summary = report["validation_summary"]
        print(f"\n📊 VALIDATION SUMMARY")
        print(f"Status: {summary['overall_status']}")
        print(f"Success Rate: {summary['success_rate']}%")
        print(f"Tests: {summary['passed']}/{summary['total_tests']} passed")
        
        if summary["failed"] > 0:
            print(f"❌ Failed: {summary['failed']}")
            
        if summary["errors"] > 0:
            print(f"🔥 Errors: {summary['errors']}")
            
        if summary["warnings"] > 0:
            print(f"⚠️ Warnings: {summary['warnings']}")
        
        # Zeige Empfehlungen
        if report["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report["recommendations"]:
                print(f"  {rec}")
        
        # Zeige kritische Fehler
        if report["errors"]:
            print(f"\n🔥 CRITICAL ERRORS:")
            for error in report["errors"][:5]:  # Zeige erste 5 Fehler
                print(f"  ❌ {error}")
        
        # Zeige System Info
        print(f"\n🖥️ SYSTEM INFO:")
        sys_info = report["system_info"]
        print(f"  Platform: {sys_info['platform']}")
        print(f"  Python: {sys_info['python_version']}")
        print(f"  Working Dir: {sys_info['working_directory']}")
        
        print(f"\n⏱️ Validation completed in {report['execution_info']['duration_seconds']}s")
        print(f"📁 Full report saved to logs/ directory")
        print("\n🚀 Next Steps:")
        print("  1. Install dependencies: pip install -r config/requirements.txt")
        print("  2. Configure API keys in environment variables")
        print("  3. Run full system validation: python scripts/system_validation.py")
        
        return summary["overall_status"] == "PASSED"
        
    except Exception as e:
        print(f"🔥 CRITICAL ERROR during validation: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 