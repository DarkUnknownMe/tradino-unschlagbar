#!/usr/bin/env python3
"""
TRADINO Test Runner
==================

Einfacher Test-Runner für die umfassende TRADINO Testsuite
Führt alle Tests aus und zeigt Ergebnisse an
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test suite
from tests.comprehensive_test_suite import main as run_tests

if __name__ == "__main__":
    print("🎯 TRADINO Test Runner")
    print("=" * 40)
    
    # Ensure we're in the correct directory
    os.chdir(project_root)
    
    # Run the comprehensive test suite
    exit_code = run_tests()
    
    print("\n" + "="*40)
    if exit_code == 0:
        print("✅ All tests completed successfully!")
    else:
        print("❌ Some tests failed. Check the logs for details.")
    
    sys.exit(exit_code) 