"""
conftest.py — pytest configuration
Adds the project root to sys.path so all packages (agent, retriever, api)
are importable during tests without needing an install step.
"""
import sys
from pathlib import Path

# Insert project root at the front of sys.path
sys.path.insert(0, str(Path(__file__).parent))
