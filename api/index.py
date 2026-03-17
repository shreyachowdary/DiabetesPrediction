"""
Vercel serverless entry point for FastAPI.

Exports the FastAPI app for Vercel's Python runtime.
All routes are handled by the main app.
"""

import sys
from pathlib import Path

# Ensure project root is in path when running from api/
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.main import app

# Vercel expects the ASGI app to be named 'app'
# No additional wrapper needed for FastAPI
