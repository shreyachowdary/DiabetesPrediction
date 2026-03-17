"""
Desktop-style launcher using pywebview.

Opens the app in a native window (no browser chrome).
Requires: pip install pywebview

Run: python run_desktop.py
"""

import sys
import threading
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def run_server():
    """Start uvicorn in a background thread."""
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8765, log_level="warning")


def main():
    try:
        import webview
    except ImportError:
        print("Install pywebview: pip install pywebview")
        sys.exit(1)

    # Start server in background
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # Wait for server to be ready

    # Create native window
    webview.create_window(
        "Diabetes Prediction - GA-Based ML",
        "http://127.0.0.1:8765",
        width=950,
        height=800,
        resizable=True,
    )
    webview.start()


if __name__ == "__main__":
    main()
