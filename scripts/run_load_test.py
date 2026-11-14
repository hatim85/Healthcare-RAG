#!/usr/bin/env python3
"""Automated runner for Locust Load & Concurrency Benchmark.

Starts a test instance of the FastAPI service on port 8008, executes Locust
with 20 concurrent users, collects performance metrics, and prints a benchmark summary.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import requests

SERVER_PORT = 8008
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
LOCUSTFILE = "tests/load_testing/locustfile.py"
REPORT_PREFIX = "data/evaluation/locust_benchmark"


def wait_for_server(url: str, timeout: int = 25) -> bool:
    """Wait until server responds 200 on /health."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    print(f"=== Step 2: Starting Load & Concurrency Benchmark (20 Concurrent Users) ===")
    env = os.environ.copy()
    env["PORT"] = str(SERVER_PORT)
    env["PYTHONPATH"] = "."
    env["RATE_LIMIT_ASK_RPM"] = "30"  # Set lower threshold so rate limiter stress test triggers reliably

    server_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
        "--log-level",
        "warning",
    ]

    print(f"Launching FastAPI test server on {SERVER_URL}...")
    server_proc = subprocess.Popen(server_cmd, env=env)

    try:
        if not wait_for_server(SERVER_URL):
            print("ERROR: Server failed to start within timeout.")
            return 1

        print("Server is UP and HEALTHY! Warming up pipeline singleton...")
        # Send one warm-up health and ask request
        try:
            requests.get(f"{SERVER_URL}/health", timeout=5)
        except Exception:
            pass
        time.sleep(2)
        print("Warmup complete. Launching Locust load test...")
        Path("data/evaluation").mkdir(parents=True, exist_ok=True)

        locust_cmd = [
            "./.venv/bin/locust",
            "-f",
            LOCUSTFILE,
            "--headless",
            "-u",
            "20",
            "-r",
            "5",
            "--run-time",
            "15s",
            "--host",
            SERVER_URL,
            "--csv",
            REPORT_PREFIX,
            "--only-summary",
        ]

        result = subprocess.run(locust_cmd, capture_output=True, text=True)
        print("\n=== Locust Output Summary ===")
        print(result.stdout)
        if result.stderr:
            print("Locust info/warnings:", result.stderr[-400:])

    finally:
        print("Shutting down FastAPI test server...")
        server_proc.send_signal(signal.SIGINT)
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("Test server stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
