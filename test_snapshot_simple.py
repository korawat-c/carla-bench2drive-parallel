#!/usr/bin/env python3
"""
Simple test for snapshot/restore functionality.
Tests if vehicle can move after restore.
"""

import requests
import json
import time
import sys

def test_snapshot_restore():
    """Test basic snapshot and restore functionality"""
    
    base_url = "http://localhost:8080"
    
    print("=" * 60)
    print("SNAPSHOT/RESTORE SIMPLE TEST")
    print("=" * 60)
    
    # 1. Reset environment
    print("\n1. Resetting environment...")
    response = requests.post(f"{base_url}/reset", json={"route_id": 0})
    if response.status_code != 200:
        print(f"ERROR: Reset failed with status {response.status_code}")
        return False
    print("✓ Reset successful")
    
    # Get initial position
    result = response.json()
    obs = result['observation']
    info = result['info']
    if 'vehicle_state' in info:
        pos = info['vehicle_state']['position']
        print(f"Initial position: x={pos['x']:.2f}, y={pos['y']:.2f}, z={pos['z']:.2f}")
        initial_x = pos['x']
        initial_y = pos['y']
    else:
        print("WARNING: No vehicle_state in info")
        initial_x = initial_y = 0
    
    # 2. Drive forward for 5 steps
    print("\n2. Driving forward for 5 steps...")
    action = {'throttle': 1.0, 'brake': 0.0, 'steer': 0.0}
    
    pos = None
    for i in range(5):
        response = requests.post(f"{base_url}/step", json={'action': action})
        if response.status_code != 200:
            print(f"ERROR: Step {i+1} failed with status {response.status_code}")
            return False
        
        result = response.json()
        info = result['info']
        if 'vehicle_state' in info:
            pos = info['vehicle_state']['position']
            print(f"  Step {i+1}: x={pos['x']:.2f}, y={pos['y']:.2f}")
        else:
            print(f"  Step {i+1}: No vehicle_state in info")
    
    if pos is None:
        print("ERROR: No position data available")
        return False
    
    mid_x = pos['x']
    mid_y = pos['y']
    print(f"✓ After 5 steps: x={mid_x:.2f}, y={mid_y:.2f}")
    print(f"  Distance moved: {((mid_x-initial_x)**2 + (mid_y-initial_y)**2)**0.5:.2f} meters")
    
    # 3. Save snapshot
    print("\n3. Saving snapshot...")
    response = requests.post(f"{base_url}/snapshot", json={})
    if response.status_code != 200:
        print(f"ERROR: Snapshot failed with status {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    snapshot_result = response.json()
    snapshot_id = snapshot_result['snapshot_id']
    print(f"✓ Snapshot saved with ID: {snapshot_id}")
    
    # Print debug info from snapshot
    debug_info = snapshot_result.get('debug_info', {})
    if debug_info:
        print(f"  Debug info:")
        for key, value in debug_info.items():
            print(f"    - {key}: {value}")
    
    # 4. Continue driving for 5 more steps
    print("\n4. Continuing forward for 5 more steps...")
    for i in range(5):
        response = requests.post(f"{base_url}/step", json={'action': action})
        if response.status_code != 200:
            print(f"ERROR: Step {i+1} failed with status {response.status_code}")
            return False
        
        result = response.json()
        info = result['info']
        if 'vehicle_state' in info:
            pos = info['vehicle_state']['position']
            print(f"  Step {i+1}: x={pos['x']:.2f}, y={pos['y']:.2f}")
    
    continue_x = pos['x']
    continue_y = pos['y']
    print(f"✓ After 10 total steps: x={continue_x:.2f}, y={continue_y:.2f}")
    print(f"  Distance from snapshot: {((continue_x-mid_x)**2 + (continue_y-mid_y)**2)**0.5:.2f} meters")
    
    # 5. Restore snapshot
    print("\n5. Restoring snapshot...")
    response = requests.post(f"{base_url}/restore", json={'snapshot_id': snapshot_id})
    if response.status_code != 200:
        print(f"ERROR: Restore failed with status {response.status_code}")
        print(f"Response: {response.text}")
        return False
    print("✓ Snapshot restored")
    
    # Get position after restore
    response = requests.post(f"{base_url}/step", json={'action': {'throttle': 0.0, 'brake': 0.0, 'steer': 0.0}})
    if response.status_code == 200:
        result = response.json()
        info = result['info']
        if 'vehicle_state' in info:
            pos = info['vehicle_state']['position']
            print(f"Position after restore: x={pos['x']:.2f}, y={pos['y']:.2f}")
            restored_x = pos['x']
            restored_y = pos['y']
        else:
            restored_x = restored_y = 0
    
    # 6. Try to turn right after restore
    print("\n6. Turning right for 5 steps after restore...")
    action = {'throttle': 1.0, 'brake': 0.0, 'steer': 1.0}  # Full right turn
    
    positions_after_restore = []
    for i in range(5):
        response = requests.post(f"{base_url}/step", json={'action': action})
        if response.status_code != 200:
            print(f"ERROR: Step {i+1} failed with status {response.status_code}")
            return False
        
        result = response.json()
        info = result['info']
        if 'vehicle_state' in info:
            pos = info['vehicle_state']['position']
            print(f"  Step {i+1}: x={pos['x']:.2f}, y={pos['y']:.2f}")
            positions_after_restore.append((pos['x'], pos['y']))
    
    # Check if vehicle moved after restore
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print("=" * 60)
    
    if len(positions_after_restore) >= 2:
        first_pos = positions_after_restore[0]
        last_pos = positions_after_restore[-1]
        distance_moved = ((last_pos[0]-first_pos[0])**2 + (last_pos[1]-first_pos[1])**2)**0.5
        
        print(f"Distance moved after restore: {distance_moved:.2f} meters")
        
        if distance_moved < 0.5:
            print("❌ FAILED: Vehicle appears to be stuck after restore!")
            print(f"   Vehicle only moved {distance_moved:.2f} meters in 5 steps")
            return False
        else:
            print("✅ SUCCESS: Vehicle can move after restore!")
            print(f"   Vehicle moved {distance_moved:.2f} meters in 5 steps")
            return True
    else:
        print("❌ FAILED: Could not get enough position data")
        return False

if __name__ == "__main__":
    # Check if server is running
    try:
        response = requests.get("http://localhost:8080/health")
        if response.status_code == 200:
            print("Server is healthy")
        else:
            print("ERROR: Server not responding properly")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server on port 8080")
        print("Please start the server with: ./RUN_SERVER.sh 0")
        sys.exit(1)
    
    # Run the test
    success = test_snapshot_restore()
    sys.exit(0 if success else 1)