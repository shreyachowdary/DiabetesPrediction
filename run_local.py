"""
Local development server launcher.

Starts Uvicorn and optionally opens the app in the default browser.
Run: python run_local.py
"""

import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Ensure we're in project root
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    port = 8000
    url = f"http://127.0.0.1:{port}"

    if is_port_in_use(port):
        print(f"Server already running at {url}")
        webbrowser.open(url)
        return

    print("Starting Diabetes Prediction server...")
    print(f"Open {url} in your browser")
    print("Press Ctrl+C to stop\n")

    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open(url)

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    subprocess.run(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", str(port)],
        cwd=str(project_root),
    )


if __name__ == "__main__":
    main()
