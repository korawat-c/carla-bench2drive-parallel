#!/usr/bin/env python3
"""
Test script to verify CARLA server startup and connection
"""
import time
import subprocess
import requests
import sys
from pathlib import Path

def test_server_startup():
    """Test CARLA server startup and basic connection"""
    print("🚀 Testing CARLA server startup...")

    # Start server
    print("Starting CARLA server...")
    server_process = subprocess.Popen([
        'python', 'server/carla_server.py',
        '--port', '8080',
        '--carla-port', '2000',
        '--server-id', 'test-service'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for server to start
    print("Waiting for server to initialize (30 seconds)...")
    time.sleep(30)

    # Test server connection
    try:
        print("Testing server health endpoint...")
        response = requests.get("http://localhost:8080/health", timeout=30)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Server is healthy: {health_data.get('status', 'unknown')}")
            print(f"   Server ID: {health_data.get('server_id', 'unknown')}")
            print(f"   Step count: {health_data.get('step_count', 0)}")

            # Test basic reset
            print("Testing server reset endpoint...")
            reset_response = requests.post(
                "http://localhost:8080/reset",
                json={"route_id": 0},
                timeout=60
            )
            if reset_response.status_code == 200:
                print("✅ Reset endpoint working")
                return True
            else:
                print(f"⚠️ Reset endpoint returned status: {reset_response.status_code}")
        else:
            print(f"⚠️ Server responded with status: {response.status_code}")

    except Exception as e:
        print(f"❌ Server connection test failed: {e}")

    # Clean up
    print("Stopping server...")
    server_process.terminate()
    try:
        server_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_process.kill()

    return False

if __name__ == "__main__":
    success = test_server_startup()
    if success:
        print("\n✅ Server test passed!")
        sys.exit(0)
    else:
        print("\n❌ Server test failed!")
        sys.exit(1)