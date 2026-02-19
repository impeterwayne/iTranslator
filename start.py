"""
Start both FastAPI backend and Vite frontend dev server with a single command.

Usage:
    python start.py
"""
import subprocess
import sys
import os
import signal
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

def main():
    procs = []

    try:
        # 1. Auto-install frontend deps if node_modules is missing
        if not os.path.isdir(os.path.join(FRONTEND_DIR, "node_modules")):
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, shell=True, check=True)

        print("🚀 Starting iTranslator...")
        print("   Backend  → http://localhost:8000")
        print("   Frontend → http://localhost:5173")
        print("   Press Ctrl+C to stop both.\n")

        # 2. Start FastAPI backend
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server:app",
             "--host", "0.0.0.0", "--port", "8000", "--reload"],
            cwd=ROOT_DIR,
            shell=False,
        )
        procs.append(backend)

        # 3. Start Vite frontend
        frontend = subprocess.Popen(
            ["npx", "vite", "--port", "5173"],
            cwd=FRONTEND_DIR,
            shell=True,
        )
        procs.append(frontend)

        # Wait for either to exit
        while True:
            for p in procs:
                ret = p.poll()
                if ret is not None:
                    raise SystemExit(ret)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
    finally:
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        print("✅ Stopped.")

if __name__ == "__main__":
    main()
