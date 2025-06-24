#!/usr/bin/env python3
"""
TRADINO Standalone Test Suite
============================

Eigenständige Testsuite für TRADINO ohne externe Dependencies
Für sofortige Ausführung und Systemvalidierung
"""

import sys
import os
import time
import json
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TestResults:
    """Sammelt und verwaltet alle Testergebnisse"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.results = {}
        self.errors = []
        self.warnings = []
        self.performance_metrics = {}
        
    def add_result(self, test_name: str, status: str, details: str = "", duration: float = 0):
        """Fügt ein Testergebnis hinzu"""
        self.results[test_name] = {
            'status': status,
            'details': details,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
    def add_error(self, test_name: str, error: str):
        """Fügt einen Fehler hinzu"""
        self.errors.append({
            'test': test_name,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        
    def add_warning(self, test_name: str, warning: str):
        """Fügt eine Warnung hinzu"""
        self.warnings.append({
            'test': test_name,
            'warning': warning,
            'timestamp': datetime.now().isoformat()
        })
        
    def add_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Fügt eine Performance-Metrik hinzu"""
        self.performance_metrics[metric_name] = {
            'value': value,
            'unit': unit,
            'timestamp': datetime.now().isoformat()
        }
        
    def generate_report(self) -> str:
        """Generiert einen detaillierten Testbericht"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASSED')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAILED')
        total = len(self.results)
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                         TRADINO STANDALONE TEST REPORT                                    ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║ Test Execution Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}                                            ║
║ Total Duration: {total_duration:.2f} seconds                                                        ║
║ Tests Run: {total:3d} | Passed: {passed:3d} | Failed: {failed:3d} | Success Rate: {(passed/total*100) if total > 0 else 0:.1f}%       ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

📊 TEST RESULTS SUMMARY:
"""
        
        for test_name, result in self.results.items():
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            report += f"  {status_icon} {test_name:<50} [{result['status']:>6}] ({result['duration']:.3f}s)\n"
            if result['details']:
                report += f"     └─ {result['details']}\n"
        
        if self.performance_metrics:
            report += "\n🚀 PERFORMANCE METRICS:\n"
            for metric, data in self.performance_metrics.items():
                report += f"  📈 {metric:<40} {data['value']:>10.3f} {data['unit']}\n"
        
        if self.warnings:
            report += "\n⚠️  WARNINGS:\n"
            for warning in self.warnings:
                report += f"  • {warning['test']}: {warning['warning']}\n"
        
        if self.errors:
            report += "\n🚨 ERRORS:\n"
            for error in self.errors:
                report += f"  • {error['test']}: {error['error']}\n"
        
        # System Readiness Assessment
        critical_tests = [
            'file_structure_validation', 'config_file_validation', 'model_files_check',
            'python_environment_check', 'system_performance'
        ]
        
        critical_passed = sum(1 for test in critical_tests 
                            if test in self.results and self.results[test]['status'] == 'PASSED')
        
        if critical_passed == len(critical_tests) and failed == 0:
            readiness = "🟢 SYSTEM READY FOR FURTHER INTEGRATION"
        elif critical_passed >= len(critical_tests) * 0.8:
            readiness = "🟡 SYSTEM MOSTLY READY - REVIEW WARNINGS"
        else:
            readiness = "🔴 SYSTEM NOT READY - CRITICAL ISSUES DETECTED"
        
        report += f"\n{readiness}\n"
        report += "="*90 + "\n"
        
        return report

class TRADINOStandaloneTestSuite:
    """Eigenständige TRADINO Testsuite ohne externe Dependencies"""
    
    def __init__(self):
        self.results = TestResults()
        self.project_root = Path(__file__).parent.parent
        
    def run_test(self, test_name: str, test_func):
        """Führt einen einzelnen Test aus und misst die Zeit"""
        start_time = time.time()
        try:
            logger.info(f"🧪 Running test: {test_name}")
            result = test_func()
            duration = time.time() - start_time
            
            if result is True or (isinstance(result, tuple) and result[0]):
                self.results.add_result(test_name, 'PASSED', 
                                      result[1] if isinstance(result, tuple) else "", duration)
                logger.info(f"✅ {test_name} PASSED ({duration:.3f}s)")
            else:
                details = result[1] if isinstance(result, tuple) and len(result) > 1 else ""
                self.results.add_result(test_name, 'FAILED', details, duration)
                logger.error(f"❌ {test_name} FAILED ({duration:.3f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{str(e)[:100]}..."
            self.results.add_result(test_name, 'FAILED', error_msg, duration)
            self.results.add_error(test_name, str(e))
            logger.error(f"❌ {test_name} ERROR: {str(e)}")

    def test_file_structure_validation(self) -> Tuple[bool, str]:
        """Testet die TRADINO Dateistruktur"""
        try:
            required_dirs = [
                'tradino_unschlagbar',
                'tests',
                'models',
                'core',
                'scripts'
            ]
            
            missing_dirs = []
            for dir_name in required_dirs:
                dir_path = self.project_root / dir_name
                if not dir_path.exists():
                    missing_dirs.append(dir_name)
            
            if missing_dirs:
                return False, f"Missing directories: {', '.join(missing_dirs)}"
            
            return True, f"File structure validated: {len(required_dirs)} directories found"
            
        except Exception as e:
            return False, f"File structure test error: {str(e)}"
    
    def test_system_performance(self) -> Tuple[bool, str]:
        """Testet System Performance"""
        try:
            start_time = time.time()
            
            # Simulate data processing
            data_points = 10000
            total = 0
            
            for i in range(data_points):
                total += i * 2 + 1
            
            processing_time = time.time() - start_time
            processing_speed = data_points / processing_time if processing_time > 0 else 0
            
            self.results.add_performance_metric("processing_speed", processing_speed, "ops/sec")
            
            return True, f"Performance: {processing_speed:.0f} ops/sec"
            
        except Exception as e:
            return False, f"Performance test error: {str(e)}"
    
    def run_all_tests(self):
        """Führt alle Tests aus"""
        logger.info("🚀 Starting TRADINO Standalone Test Suite")
        logger.info("="*70)
        
        tests = [
            ("file_structure_validation", self.test_file_structure_validation),
            ("system_performance", self.test_system_performance),
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        report = self.results.generate_report()
        logger.info("\n" + report)
        
        return self.results

def main():
    """Hauptfunktion zum Ausführen der Tests"""
    print("🔥 TRADINO Standalone Test Suite 🔥")
    print("=" * 45)
    
    try:
        test_suite = TRADINOStandaloneTestSuite()
        results = test_suite.run_all_tests()
        
        total_tests = len(results.results)
        passed_tests = sum(1 for r in results.results.values() if r['status'] == 'PASSED')
        failed_tests = total_tests - passed_tests
        
        print(f"\n🏁 Test Suite Completed!")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests*100) if total_tests > 0 else 0:.1f}%")
        
        return 0 if failed_tests == 0 else 1
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
