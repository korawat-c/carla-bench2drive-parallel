#!/usr/bin/env python3
"""
Test script to verify that vehicle spawning is frozen after restore
"""

import requests
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:8080"

def main():
    logger.info("Testing vehicle freeze after restore")
    
    # 1. Reset environment
    logger.info("1. Resetting environment...")
    response = requests.post(f"{SERVER_URL}/reset", json={"route_id": 0}, timeout=120)
    if response.status_code != 200:
        logger.error(f"Failed to reset: {response.text}")
        return
    
    # 2. Take some steps
    logger.info("2. Taking 20 steps...")
    for i in range(20):
        action = {"throttle": 1.0, "brake": 0.0, "steer": 0.0}
        response = requests.post(f"{SERVER_URL}/step", json={"action": action})
        if response.status_code == 200 and i == 19:
            obs = response.json()['observation']
            pos = obs['vehicle_state']['position']
            logger.info(f"   Position at step 20: x={pos['x']:.2f}, y={pos['y']:.2f}")
    
    # 3. Get vehicle count before snapshot
    logger.info("3. Getting vehicle count...")
    response = requests.get(f"{SERVER_URL}/world/vehicles")
    if response.status_code == 200:
        vehicles_before = response.json()
        logger.info(f"   Vehicles before snapshot: {vehicles_before['count']}")
    
    # 4. Create snapshot
    logger.info("4. Creating snapshot...")
    response = requests.post(f"{SERVER_URL}/snapshot", json={"snapshot_id": "freeze_test"})
    if response.status_code != 200:
        logger.error(f"Failed to create snapshot: {response.text}")
        return
    logger.info("   Snapshot created")
    
    # 5. Take more steps (continue timeline)
    logger.info("5. Taking 10 more steps (continue timeline)...")
    for i in range(10):
        action = {"throttle": 1.0, "brake": 0.0, "steer": 0.0}
        requests.post(f"{SERVER_URL}/step", json={"action": action})
    
    # 6. Restore snapshot
    logger.info("6. Restoring snapshot...")
    response = requests.post(f"{SERVER_URL}/restore", json={"snapshot_id": "freeze_test"}, timeout=30)
    if response.status_code != 200:
        logger.error(f"Failed to restore: {response.text}")
        return
    logger.info("   Snapshot restored - vehicles should now be FROZEN")
    
    # 7. Monitor vehicle count for next 10 steps
    logger.info("7. Taking 10 steps and monitoring vehicle count...")
    for i in range(10):
        # Take a step
        action = {"throttle": 0.5, "brake": 0.0, "steer": 0.2}  # Turn slightly
        response = requests.post(f"{SERVER_URL}/step", json={"action": action})
        
        # Check vehicle count
        response = requests.get(f"{SERVER_URL}/world/vehicles")
        if response.status_code == 200:
            vehicles = response.json()
            logger.info(f"   Step {i+1}: {vehicles['count']} vehicles")
            
            # Check if count changed
            if i > 0 and vehicles['count'] != vehicles_before['count']:
                logger.error(f"   ❌ FAILURE: Vehicle count changed! Was {vehicles_before['count']}, now {vehicles['count']}")
                logger.error("   Vehicles are still spawning/despawning after restore!")
                return
    
    logger.info("✅ SUCCESS: Vehicle count remained stable - freeze is working!")
    
    # 8. Test unfreeze endpoint
    logger.info("8. Testing unfreeze...")
    response = requests.post(f"{SERVER_URL}/freeze_vehicles", json={"freeze": False})
    if response.status_code == 200:
        logger.info("   Vehicles unfrozen")
    
    logger.info("Test complete!")

if __name__ == "__main__":
    main()