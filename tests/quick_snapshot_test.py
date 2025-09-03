#!/usr/bin/env python3
"""Minimal test for snapshot/restore - just 3 steps"""

import requests
import time
import json
import subprocess
import sys

def cleanup_processes():
    """Clean up existing CARLA and server processes"""
    print("Cleaning up existing servers...")
    commands = [
        "pkill -f 'python.*carla_server'",
        "pkill -f 'python.*microservice_manager'",
        "pkill -f 'CarlaUE4.sh'",
        "pkill -f 'CarlaUE4-Linux'"
    ]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)
    
    time.sleep(3)

def start_server():
    """Start the microservice manager"""
    print("Starting server...")
    cleanup_processes()
    
    cmd = "python microservice_manager.py --num-services 1 --startup-delay 0"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for server to be ready
    for i in range(30):
        try:
            r = requests.get("http://localhost:8080/health", timeout=2)
            if r.status_code == 200:
                print("Server is ready!")
                return proc
        except:
            pass
        time.sleep(1)
    
    print("ERROR: Server failed to start")
    return None

def quick_test():
    base_url = "http://localhost:8080"
    
    print("\n=== QUICK SNAPSHOT TEST ===\n")
    
    # 1. Check server
    print("1. Checking server...")
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Server status: {r.status_code}")
    except:
        print("   ERROR: Server not running!")
        return
    
    # 2. Reset - Use route 1 which might be simpler
    print("\n2. Resetting environment (this may take 60-90 seconds)...")
    try:
        r = requests.post(f"{base_url}/reset", json={"route_id": 1}, timeout=180)
        if r.status_code != 200:
            print(f"   ERROR: Reset failed: {r.status_code}")
            return
        print("   Reset OK")
    except Exception as e:
        print(f"   ERROR: Reset failed: {e}")
        return
    
    # 3. Take 3 steps and record position
    print("\n3. Taking 3 steps...")
    positions = []
    for i in range(3):
        action = {"throttle": 1.0, "brake": 0.0, "steer": 0.0}
        r = requests.post(f"{base_url}/step", json={"action": action}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            obs = data.get('observation', {})
            if 'vehicle_state' in obs and 'position' in obs['vehicle_state']:
                pos = obs['vehicle_state']['position']
                x = pos.get('x', 0)
                positions.append(x)
                print(f"   Step {i+1}: x={x:.2f}")
    
    if len(positions) < 3:
        print("   ERROR: Could not get positions")
        return
    
    step3_x = positions[-1]
    print(f"\n   Position after 3 steps: x={step3_x:.2f}")
    
    # 4. Save snapshot
    print("\n4. Saving snapshot...")
    r = requests.post(f"{base_url}/snapshot", json={}, timeout=10)
    if r.status_code != 200:
        print(f"   ERROR: Save failed: {r.status_code}")
        return
    snapshot_data = r.json()
    snapshot_id = snapshot_data['snapshot_id']
    stats = snapshot_data.get('stats', {})
    print(f"   Saved: {snapshot_id}")
    print(f"   Vehicles: {stats.get('vehicles', 0)}")
    print(f"   Has scenario manager: {stats.get('has_scenario_manager', False)}")
    
    # 5. Take 3 more steps (position should change)
    print("\n5. Taking 3 more steps...")
    for i in range(3):
        action = {"throttle": 1.0, "brake": 0.0, "steer": 0.0}
        r = requests.post(f"{base_url}/step", json={"action": action}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            obs = data.get('observation', {})
            if 'vehicle_state' in obs and 'position' in obs['vehicle_state']:
                pos = obs['vehicle_state']['position']
                x = pos.get('x', 0)
                print(f"   Step {i+4}: x={x:.2f}")
                step6_x = x
    
    print(f"\n   Position after 6 steps: x={step6_x:.2f}")
    print(f"   Moved {step6_x - step3_x:.2f}m from step 3")
    
    # 6. Restore snapshot
    print(f"\n6. Restoring to step 3 (x={step3_x:.2f})...")
    r = requests.post(f"{base_url}/restore", json={"snapshot_id": snapshot_id}, timeout=30)
    if r.status_code != 200:
        print(f"   ERROR: Restore failed: {r.status_code}")
        if r.status_code == 500:
            try:
                error = r.json().get('detail', 'Unknown')
                print(f"   Error: {error}")
            except:
                print(f"   Response: {r.text}")
        return
    print("   Restore OK")
    
    # 7. Get position after restore
    print("\n7. Getting position after restore...")
    r = requests.post(f"{base_url}/get_observation", json={}, timeout=10)
    if r.status_code == 200:
        data = r.json()
        obs = data.get('observation', {})
        if 'vehicle_state' in obs and 'position' in obs['vehicle_state']:
            pos = obs['vehicle_state']['position']
            restored_x = pos.get('x', 0)
            print(f"   Position after restore: x={restored_x:.2f}")
            
            drift = restored_x - step3_x
            print(f"\n=== RESULT ===")
            print(f"Expected: x={step3_x:.2f}")
            print(f"Got:      x={restored_x:.2f}")
            print(f"Drift:    {drift:.2f}m")
            
            if abs(drift) < 1.0:
                print("\n✓ SUCCESS: Restore worked!")
            else:
                print(f"\n✗ FAILED: Position drift of {drift:.2f}m")
    else:
        print(f"   ERROR: Could not get observation")

if __name__ == "__main__":
    if "--start-server" in sys.argv:
        server = start_server()
        if not server:
            sys.exit(1)
    
    quick_test()
    
    if "--start-server" in sys.argv:
        cleanup_processes()
