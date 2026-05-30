# conftest.py
import sys
from pathlib import Path

# Add the project root to sys.path so pytest can find `src`
sys.path.insert(0, str(Path(__file__).parent))