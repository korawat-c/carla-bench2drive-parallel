#!/usr/bin/env python3
"""
Test reset endpoint to debug hanging issue
"""

import requests
import time
import sys

def test_reset():
    """Test reset endpoint"""
    
    print("Testing reset endpoint...")
    
    # Check health first
    try:
        response = requests.get("http://localhost:8080/health", timeout=2)
        if response.status_code == 200:
            health = response.json()
            print(f"Server is healthy: {health}")
        else:
            print(f"Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Cannot connect to server: {e}")
        return False
    
    # Try reset with timeout
    print("\nAttempting reset with 10 second timeout...")
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:8080/reset",
            json={"route_id": 0},
            timeout=10
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✓ Reset successful in {elapsed:.2f} seconds")
            result = response.json()
            print(f"  Keys in result: {list(result.keys())}")
            
            if 'observation' in result:
                obs = result['observation']
                print(f"  Observation type: {type(obs)}")
                if isinstance(obs, dict):
                    print(f"  Observation keys: {list(obs.keys())}")
            
            if 'info' in result:
                info = result['info']
                print(f"  Info type: {type(info)}")
                if isinstance(info, dict):
                    print(f"  Info keys: {list(info.keys())}")
                    if 'vehicle_state' in info:
                        print("  ✓ vehicle_state present in info")
                    else:
                        print("  ✗ vehicle_state NOT in info")
            
            return True
        else:
            print(f"✗ Reset failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Reset timed out after 10 seconds")
        print("  The reset endpoint is hanging")
        return False
    except Exception as e:
        print(f"✗ Reset failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_reset()
    sys.exit(0 if success else 1)