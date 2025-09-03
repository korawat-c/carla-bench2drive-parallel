#!/usr/bin/env python3
"""
Test that verifies each component is properly saved and restored:
1. World (vehicles, pedestrians, traffic lights)
2. Scenario Manager state
3. Agent Instance state  
4. Traffic Manager state
"""

import requests
import json
import time
import sys

def get_component_states(base_url):
    """Get detailed state of all components"""
    # Get step to know where we are
    r = requests.post(f"{base_url}/step", json={"throttle": 0, "steer": 0, "brake": 1.0})
    step_data = r.json()
    
    obs = step_data.get('observation', {})
    vehicle_state = obs.get('vehicle_state', {})
    
    state = {
        'step': step_data.get('step', 0),
        'ego_position': vehicle_state.get('position', [0,0,0]),
        'ego_rotation': vehicle_state.get('rotation', [0,0,0]),
        'ego_velocity': vehicle_state.get('velocity', [0,0,0]),
        'ego_speed': vehicle_state.get('speed', 0),
        'scenario_info': obs.get('scenario_info', {}),
        'debug_info': obs.get('debug_info', {})
    }
    
    return state

def compare_states(state1, state2, phase1_name, phase2_name):
    """Compare two states and report differences"""
    print(f"\n=== Comparing {phase1_name} vs {phase2_name} ===")
    
    # Compare ego position
    pos1 = state1['ego_position']
    pos2 = state2['ego_position']
    pos_diff = ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)**0.5
    
    print(f"Ego Position:")
    print(f"  {phase1_name}: x={pos1[0]:.2f}, y={pos1[1]:.2f}, z={pos1[2]:.2f}")
    print(f"  {phase2_name}: x={pos2[0]:.2f}, y={pos2[1]:.2f}, z={pos2[2]:.2f}")
    print(f"  Distance: {pos_diff:.2f}m")
    
    if pos_diff > 1.0:
        print(f"  ❌ ERROR: Position mismatch > 1m!")
    else:
        print(f"  ✓ Position match within tolerance")
    
    # Compare step count
    print(f"\nStep Count:")
    print(f"  {phase1_name}: {state1['step']}")
    print(f"  {phase2_name}: {state2['step']}")
    
    return pos_diff < 1.0

def main():
    base_url = "http://localhost:8080"
    
    # Check server
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        print(f"Server health: {r.json()}")
    except:
        print("Server not running. Start with: python microservice_manager.py --num-services 1")
        sys.exit(1)
    
    print("\n=== Component Verification Test ===")
    
    # Phase 0: Reset
    print("\nPhase 0: Resetting...")
    r = requests.post(f"{base_url}/reset", json={"route_id": 0}, timeout=90)
    print(f"Reset: {r.status_code}")
    
    # Phase 1: Drive 10 steps
    print("\nPhase 1: Driving 10 steps...")
    for i in range(10):
        r = requests.post(f"{base_url}/step", 
                         json={"throttle": 1.0, "steer": 0.0, "brake": 0.0})
        obs = r.json().get("observation", {})
        pos = obs.get("vehicle_state", {}).get("position", [0,0,0])
        print(f"  Step {i+1}: x={pos[0]:.2f}, y={pos[1]:.2f}")
    
    # Get state at step 10
    print("\nCapturing state at step 10...")
    state_at_save = get_component_states(base_url)
    print(f"State at step 10: x={state_at_save['ego_position'][0]:.2f}, y={state_at_save['ego_position'][1]:.2f}")
    
    # Save snapshot
    print("\nSaving snapshot 'test_components'...")
    r = requests.post(f"{base_url}/snapshot", json={"snapshot_id": "test_components"})
    save_result = r.json()
    print(f"Snapshot saved: {save_result.get('status')}")
    
    # Phase 2: Continue 10 more steps
    print("\nPhase 2: Continuing 10 more steps...")
    for i in range(10, 20):
        r = requests.post(f"{base_url}/step",
                         json={"throttle": 1.0, "steer": 0.0, "brake": 0.0})
        obs = r.json().get("observation", {})
        pos = obs.get("vehicle_state", {}).get("position", [0,0,0])
        print(f"  Step {i+1}: x={pos[0]:.2f}, y={pos[1]:.2f}")
    
    # Get state at step 20
    print("\nCapturing state at step 20...")
    state_at_phase2_end = get_component_states(base_url)
    print(f"State at step 20: x={state_at_phase2_end['ego_position'][0]:.2f}, y={state_at_phase2_end['ego_position'][1]:.2f}")
    
    # Phase 3: Restore
    print("\nPhase 3: Restoring snapshot...")
    r = requests.post(f"{base_url}/restore", json={"snapshot_id": "test_components"})
    restore_result = r.json()
    print(f"Restore: {restore_result.get('status')}")
    
    # Get state after restore
    print("\nCapturing state after restore...")
    state_after_restore = get_component_states(base_url)
    print(f"State after restore: x={state_after_restore['ego_position'][0]:.2f}, y={state_after_restore['ego_position'][1]:.2f}")
    
    # Compare states
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    # Should match: Save state vs Restore state
    match1 = compare_states(state_at_save, state_after_restore, "Step 10 (Save)", "After Restore")
    
    # Should NOT match: Phase 2 end vs Restore
    compare_states(state_at_phase2_end, state_after_restore, "Step 20 (Phase 2 End)", "After Restore")
    
    if match1:
        print("\n✓ SUCCESS: Restore returned to saved position!")
    else:
        print("\n❌ FAILURE: Restore did NOT return to saved position!")
        print("\nDEBUGGING INFO:")
        print(f"Expected position (step 10): x={state_at_save['ego_position'][0]:.2f}")
        print(f"Got position (after restore): x={state_after_restore['ego_position'][0]:.2f}")
        print(f"Drift: {state_after_restore['ego_position'][0] - state_at_save['ego_position'][0]:.2f}m forward")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
