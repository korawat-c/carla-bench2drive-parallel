#!/usr/bin/env python3
"""
Test GRPO environment status API.

Demonstrates how to use the status information returned by the environment
to handle async operations and error conditions.
"""

import numpy as np
import logging
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from client.grpo_carla_env import GRPOCarlaEnv, EnvStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_status_api():
    """Test the status API functionality."""
    
    # Create environment
    env = GRPOCarlaEnv(num_services=4)
    
    try:
        logger.info("=== Testing Status API ===")
        
        # 1. Reset and check status
        obs, info = env.reset()
        logger.info(f"After reset - Mode: {env.current_mode}, Ready: {env.is_branching == False}")
        
        # 2. Single step with status
        action = np.array([0.5, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.single_step(action)
        
        if 'status' in info:
            status = info['status']
            logger.info(f"Single step status: {status.status.value} - {status.message}")
            logger.info(f"  Ready: {status.ready}, Details: {status.details}")
        
        # 3. Try incorrect method (should fail with clear error)
        try:
            logger.info("\nTrying branch_step in single mode (should fail)...")
            env.branch_step([action, action])
        except RuntimeError as e:
            logger.info(f"Expected error: {e}")
        
        # 4. Save snapshot
        snapshot_id = env.save_snapshot()
        logger.info(f"\nSnapshot saved: {snapshot_id}")
        
        # 5. Enable branching with async setup
        logger.info("\nEnabling branching with async setup...")
        setup_status = env.enable_branching(snapshot_id, num_branches=4, async_setup=True)
        logger.info(f"Setup status: {setup_status.status.value} - {setup_status.message}")
        logger.info(f"  Ready: {setup_status.ready}, Retry after: {setup_status.retry_after}s")
        
        # 6. Poll for setup completion
        max_attempts = 10
        for attempt in range(max_attempts):
            time.sleep(1)
            status = env.check_branching_status()
            logger.info(f"Check {attempt+1}: {status.status.value} - {status.message}")
            
            if status.progress is not None:
                logger.info(f"  Progress: {int(status.progress * 100)}%")
            
            if status.status == EnvStatus.BRANCHING_READY:
                logger.info("Branching setup complete!")
                break
            elif status.status == EnvStatus.ERROR:
                logger.error("Branching setup failed!")
                break
            elif status.retry_after:
                logger.info(f"  Waiting {status.retry_after}s before retry...")
        
        # 7. Try branch step after setup
        if env.is_branching:
            logger.info("\nTrying branch_step after setup...")
            actions = [np.array([0.5, 0.0, 0.0]) for _ in range(env.active_branches)]
            observations, rewards, terminateds, truncateds, infos = env.branch_step(actions)
            
            for i, info in enumerate(infos):
                if 'status' in info:
                    status = info['status']
                    logger.info(f"Branch {i} status: {status.status.value} - {status.message}")
        
        # 8. Select branch and return to single mode
        logger.info("\nSelecting best branch...")
        env.select_branch(0)
        logger.info(f"After select - Mode: {env.current_mode}, Ready: {env.is_branching == False}")
        
        # 9. Try single step again
        obs, reward, terminated, truncated, info = env.single_step(action)
        if 'status' in info:
            status = info['status']
            logger.info(f"Final single step status: {status.status.value} - {status.message}")
        
        logger.info("\n=== Status API Test Complete ===")
        
    finally:
        env.close()


def test_synchronous_setup():
    """Test synchronous branching setup with immediate status."""
    
    env = GRPOCarlaEnv(num_services=2)
    
    try:
        logger.info("\n=== Testing Synchronous Setup ===")
        
        # Reset and save snapshot
        obs, _ = env.reset()
        snapshot_id = env.save_snapshot()
        
        # Enable branching synchronously
        logger.info("Enabling branching synchronously...")
        setup_status = env.enable_branching(snapshot_id, num_branches=2, async_setup=False)
        
        logger.info(f"Setup status: {setup_status.status.value} - {setup_status.message}")
        logger.info(f"  Ready: {setup_status.ready}")
        logger.info(f"  Progress: {setup_status.progress}")
        logger.info(f"  Details: {setup_status.details}")
        
        if setup_status.status == EnvStatus.BRANCHING_READY:
            logger.info("Branching immediately ready!")
            
            # Test branch step
            actions = [np.array([0.5, 0.0, 0.0]) for _ in range(2)]
            observations, rewards, terminateds, truncateds, infos = env.branch_step(actions)
            logger.info(f"Branch step successful, got {len(observations)} observations")
        
    finally:
        env.close()


def main():
    """Run status API tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test GRPO status API')
    parser.add_argument('--async', action='store_true', help='Test async setup')
    parser.add_argument('--sync', action='store_true', help='Test sync setup')
    
    args = parser.parse_args()
    
    if not args.async and not args.sync:
        # Run both by default
        test_status_api()
        test_synchronous_setup()
    else:
        if args.async:
            test_status_api()
        if args.sync:
            test_synchronous_setup()


if __name__ == "__main__":
    main()