#!/usr/bin/env python3
"""Test script to verify the notebook fix works."""

import requests
import time
import subprocess
import os

def test_service_health():
    """Test which services are responding."""
    available_services = []
    for i in range(2):
        try:
            response = requests.get(f"http://localhost:{8080 + i}/health", timeout=5)
            if response.status_code == 200:
                available_services.append(f"http://localhost:{8080 + i}")
                print(f"✅ Service {i} is ready at {8080 + i}")
            else:
                print(f"⚠️  Service {i} returned status {response.status_code}")
        except Exception as e:
            print(f"❌ Service {i} is not responding: {e}")

    print(f"Available services: {len(available_services)}/2")
    return available_services

def start_servers():
    """Start servers with robust error handling."""
    print("Starting servers...")

    # Clean up first
    subprocess.run(["pkill", "-9", "-f", "microservice_manager.py"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "carla_server.py"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "CarlaUE4"], capture_output=True)

    # Wait a bit
    time.sleep(2)

    # Start microservice manager
    cmd = [
        "python",
        "/mnt3/Documents/AD_Framework/bench2drive-gymnasium/bench2drive_microservices/server/microservice_manager.py",
        "--num-services", "2",
        "--startup-delay", "30",
    ]

    print(f"Launching: {' '.join(cmd)}")
    subprocess.Popen(cmd)

    # Wait for services
    max_wait = 60
    wait_interval = 5
    total_wait = 0

    print(f"Waiting up to {max_wait} seconds for services...")

    while total_wait < max_wait:
        available_services = []
        all_ready = True

        for i in range(2):
            try:
                response = requests.get(f"http://localhost:{8080 + i}/health", timeout=5)
                if response.status_code == 200:
                    available_services.append(f"http://localhost:{8080 + i}")
                else:
                    all_ready = False
            except:
                all_ready = False

        print(f"Waited {total_wait}s: {len(available_services)}/2 services ready")

        if all_ready and len(available_services) == 2:
            print("✅ All services ready!")
            return available_services

        time.sleep(wait_interval)
        total_wait += wait_interval

    print(f"⚠️  Only {len(available_services)}/2 services ready after {max_wait}s")
    return available_services

if __name__ == "__main__":
    print("Testing service availability...")

    # Test initial state
    print("=== Initial State ===")
    services = test_service_health()

    if len(services) == 0:
        print("\n=== Starting Servers ===")
        services = start_servers()

        print("\n=== Final State ===")
        services = test_service_health()

    if len(services) > 0:
        print(f"\n✅ Success! Found {len(services)} services: {services}")
    else:
        print(f"\n❌ Failed to start any services")