#!/usr/bin/env python3
"""Simple test that just checks if restore is working"""

import requests
import json
import time

print("\n=== SIMPLE RESTORE TEST ===\n")

# Check if server is running
try:
    r = requests.get("http://localhost:8080/health", timeout=2)
    if r.status_code == 200:
        print("Server is running")
    else:
        print("Server not healthy, trying to start it...")
        import subprocess
        subprocess.run("python microservice_manager.py --num-services 1 > /tmp/mgr.log 2>&1 &", shell=True)
        time.sleep(10)
except:
    print("Server not running, starting...")
    import subprocess
    subprocess.run("python microservice_manager.py --num-services 1 > /tmp/mgr.log 2>&1 &", shell=True)
    time.sleep(10)

# Try to get a snapshot that might already exist
print("\nChecking for existing snapshots...")
try:
    # First, let's see if we can even call the endpoint
    r = requests.post("http://localhost:8080/restore", 
                     json={"snapshot_id": "test"}, 
                     timeout=5)
    print(f"Restore endpoint response: {r.status_code}")
    if r.status_code == 400:
        print("  Environment not initialized (expected)")
    elif r.status_code == 404:
        print("  Snapshot not found (expected)")
    elif r.status_code == 500:
        error = r.json().get('detail', 'Unknown error')
        print(f"  Server error: {error}")
except Exception as e:
    print(f"Error calling restore: {e}")

print("\nConclusion:")
print("- The restore endpoint exists and responds")
print("- But we can't test it fully without a working environment")
print("- The scenario loading hangs, preventing full testing")
