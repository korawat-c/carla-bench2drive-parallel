#!/usr/bin/env python3
"""Simple test to debug snapshot/restore issue"""

import requests
import time
import json
import sys

def test_snapshot_restore():
    """Test snapshot and restore functionality"""
    base_url = "http://localhost:8080"
    
    print("1. Testing server health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Health check: {response.status_code}")
        if response.status_code != 200:
            print("   Server not healthy!")
            return False
    except Exception as e:
        print(f"   Server not responding: {e}")
        return False
    
    print("\n2. Resetting environment...")
    try:
        response = requests.post(f"{base_url}/reset", json={"route_id": 0}, timeout=120)
        print(f"   Reset status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Reset failed: {response.text}")
            return False
        reset_data = response.json()
        print("   Reset successful")
    except Exception as e:
        print(f"   Reset error: {e}")
        return False
    
    print("\n3. Taking initial steps...")
    for i in range(5):
        action = {"throttle": 1.0, "brake": 0.0, "steer": 0.0}
        response = requests.post(f"{base_url}/step", json={"action": action}, timeout=10)
        if response.status_code == 200:
            step_data = response.json()
            # Try to extract position
            obs = step_data.get('observation', {})
            if 'vehicle_state' in obs and 'position' in obs['vehicle_state']:
                pos = obs['vehicle_state']['position']
                print(f"   Step {i+1}: x={pos.get('x', 0):.2f}")
            else:
                print(f"   Step {i+1}: OK")
    
    print("\n4. Saving snapshot...")
    try:
        response = requests.post(f"{base_url}/snapshot", json={}, timeout=10)
        print(f"   Save status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Save failed: {response.text}")
            return False
        snapshot_data = response.json()
        snapshot_id = snapshot_data['snapshot_id']
        print(f"   Saved snapshot: {snapshot_id}")
        print(f"   Stats: {snapshot_data.get('stats', {})}")
    except Exception as e:
        print(f"   Save error: {e}")
        return False
    
    print("\n5. Taking more steps (should change position)...")
    for i in range(5):
        action = {"throttle": 1.0, "brake": 0.0, "steer": 0.0}
        response = requests.post(f"{base_url}/step", json={"action": action}, timeout=10)
        if response.status_code == 200:
            step_data = response.json()
            obs = step_data.get('observation', {})
            if 'vehicle_state' in obs and 'position' in obs['vehicle_state']:
                pos = obs['vehicle_state']['position']
                print(f"   Step {i+6}: x={pos.get('x', 0):.2f}")
    
    print("\n6. Restoring snapshot...")
    try:
        response = requests.post(f"{base_url}/restore", json={"snapshot_id": snapshot_id}, timeout=10)
        print(f"   Restore status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Restore failed: {response.text}")
            # Try to get more error details
            if response.status_code == 500:
                try:
                    error_data = response.json()
                    print(f"   Error details: {error_data.get('detail', 'No details')}")
                except:
                    pass
            return False
        print("   Restore successful")
    except Exception as e:
        print(f"   Restore error: {e}")
        return False
    
    print("\n7. Getting observation after restore...")
    try:
        response = requests.post(f"{base_url}/get_observation", json={}, timeout=10)
        if response.status_code == 200:
            obs_data = response.json()
            obs = obs_data.get('observation', {})
            if 'vehicle_state' in obs and 'position' in obs['vehicle_state']:
                pos = obs['vehicle_state']['position']
                print(f"   Position after restore: x={pos.get('x', 0):.2f}")
        else:
            print(f"   Could not get observation: {response.status_code}")
    except:
        pass
    
    return True

if __name__ == "__main__":
    print("Simple Snapshot/Restore Test")
    print("="*50)
    
    success = test_snapshot_restore()
    
    print("\n" + "="*50)
    if success:
        print("Test completed (check positions)")
    else:
        print("Test FAILED")
        sys.exit(1)
