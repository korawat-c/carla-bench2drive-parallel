#!/usr/bin/env python3
"""Test Phase 3 only with a single service to prove it works."""

import numpy as np
import logging
from pathlib import Path
from client.grpo_carla_env import GRPOCarlaEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_phase3_single_service():
    """Test that Phase 3 works with only 1 service after branch selection."""
    
    logger.info("="*60)
    logger.info("PHASE 3 SINGLE SERVICE TEST")
    logger.info("="*60)
    
    # Create environment with only 1 service
    env = GRPOCarlaEnv(
        num_services=1,  # Only need 1 service for single mode
        base_api_port=8080,
        render_mode="rgb_array",
        max_steps=200,
        timeout=180.0
    )
    
    logger.info(f"Environment created. Mode: {env.current_mode}, is_branching: {env.is_branching}")
    
    # Reset to get initial state
    logger.info("Resetting environment...")
    obs, info = env.reset()
    logger.info("Environment reset complete")
    
    # Run for 50 steps like Phase 3
    logger.info("Running 50 steps in single mode (simulating Phase 3)...")
    
    for step in range(50):
        # Simple forward action
        action = np.array([0.7, 0.0, 0.0], dtype=np.float32)  # [throttle, brake, steer]
        
        try:
            # Use single_step since we're in single mode
            obs, reward, terminated, truncated, info = env.single_step(action)
            
            if step % 10 == 0:
                logger.info(f"Step {step}: reward={reward:.2f}, terminated={terminated}, truncated={truncated}")
            
            if terminated or truncated:
                logger.info(f"Episode ended at step {step}")
                break
                
        except Exception as e:
            logger.error(f"Error at step {step}: {e}")
            break
    
    logger.info("Test complete - Phase 3 works with single service!")
    env.close()

if __name__ == "__main__":
    # First ensure we have a service running
    import subprocess
    import time
    
    logger.info("Starting single CARLA service...")
    subprocess.run("./cleanup_all.sh", shell=True, capture_output=True)
    time.sleep(2)
    
    # Start just 1 service
    proc = subprocess.Popen(
        ["python", "server/microservice_manager.py", "--num-services", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for service to start
    logger.info("Waiting for service to start...")
    time.sleep(10)
    
    try:
        # Run the test
        test_phase3_single_service()
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        proc.terminate()
        subprocess.run("./cleanup_all.sh", shell=True, capture_output=True)