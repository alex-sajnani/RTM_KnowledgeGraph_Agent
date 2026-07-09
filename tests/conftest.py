"""
conftest.py — pytest configuration for the RTM Knowledge Graph Agent test suite.

Adds src/ to sys.path (bare imports, matching how app.py runs) and changes
the working directory to the project root so that regulations_cache.json is
found by the module-level load_regulations() call in agent.py / sme_agent.py.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root / "src"))
