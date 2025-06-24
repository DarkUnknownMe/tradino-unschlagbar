"""
TRADINO Test Suite
Unit Tests und Integration Tests für alle Systemkomponenten
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

__version__ = "1.0.0"
__author__ = "TRADINO Team" 