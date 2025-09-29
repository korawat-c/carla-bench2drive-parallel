#!/usr/bin/env python3
"""
Basic test to verify snapshot functionality works.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import time

sys.path.append(str(Path(__file__).parent.parent))

from client.grpo_carla_env import GRPOCarlaEnv, EnvStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_snapshot():
    """Test basic snapshot save/restore functionality."""
    
    # Create environment
    env = GRPOCarlaEnv(num_services=1)
    
    try:
        logger.info("=== Testing Basic Snapshot ===")
        
        # 1. Reset environment
        obs, info = env.reset()
        logger.info(f"Environment reset. Mode: {env.current_mode}")
        
        # 2. Take a few steps first (important!)
        logger.info("Taking a few steps before snapshot...")
        for i in range(5):
            action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.single_step(action)
            logger.info(f"Step {i+1}: reward={reward:.2f}")
            
            if terminated or truncated:
                logger.warning("Episode ended early")
                break
        
        # 3. Now try to save snapshot
        logger.info("\nAttempting to save snapshot...")
        try:
            snapshot_id = env.save_snapshot()
            logger.info(f"SUCCESS: Snapshot saved with ID: {snapshot_id}")
        except Exception as e:
            logger.error(f"FAILED to save snapshot: {e}")
            return False
        
        # 4. Take a few more steps
        logger.info("\nTaking more steps after snapshot...")
        for i in range(5):
            action = np.array([0.0, 0.5, 0.0], dtype=np.float32)  # Brake
            obs, reward, terminated, truncated, info = env.single_step(action)
            logger.info(f"Step {i+6}: reward={reward:.2f}")
        
        # 5. Try to restore
        logger.info("\nAttempting to restore snapshot...")
        try:
            env.restore_snapshot(snapshot_id)
            logger.info("SUCCESS: Snapshot restored")
        except Exception as e:
            logger.error(f"FAILED to restore snapshot: {e}")
            return False
        
        logger.info("\n=== Test Complete ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        env.close()


def main():
    """Run basic snapshot test."""
    import subprocess
    import time
    
    # COMPREHENSIVE CLEANUP BEFORE STARTING - USE THE CLEANUP SCRIPT
    logger.info("=== Running comprehensive cleanup script ===")
    
    cleanup_script = Path(__file__).parent.parent / "cleanup_all.sh"
    result = subprocess.run(["bash", str(cleanup_script)], capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            if line.startswith('✓'):
                logger.info(f"  {line}")
    
    # Extra wait to ensure everything is cleaned
    time.sleep(3)
    logger.info("Cleanup complete")
    
    # Start servers
    logger.info("Starting CARLA servers...")
    cmd = [
        "python", 
        str(Path(__file__).parent.parent / "server" / "microservice_manager.py"),
        "--num-services", "1"
    ]
    
    server_proc = subprocess.Popen(cmd)
    logger.info("Waiting for servers to start...")
    time.sleep(20)  # Give servers time to start
    
    try:
        success = test_basic_snapshot()
        if success:
            logger.info("\n✓ Basic snapshot test PASSED")
        else:
            logger.error("\n✗ Basic snapshot test FAILED")
            
    finally:
        logger.info("Stopping servers...")
        server_proc.terminate()
        server_proc.wait()
        
        # Extra cleanup
        subprocess.run(["pkill", "-f", "CarlaUE4"], capture_output=True)
        subprocess.run(["pkill", "-f", "server_manager.py"], capture_output=True)


if __name__ == "__main__":
    main()