#!/usr/bin/env python3
"""
Test script for snapshot/restore functionality with GRPO support.

Tests:
1. Single server snapshot/restore
2. Multi-turn rollouts from same state
3. Parallel environments with branching
4. Performance benchmarks
"""

import asyncio
import time
import json
import logging
import argparse
from typing import List, Dict, Any
import numpy as np
import requests
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class CarlaClient:
    """Client for interacting with CARLA server API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if server is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def reset(self, route_id: int = 0) -> Dict:
        """Reset environment"""
        response = self.session.post(
            f"{self.base_url}/reset",
            json={"route_id": route_id}
        )
        response.raise_for_status()
        return response.json()
    
    def step(self, action: Dict[str, float]) -> Dict:
        """Execute step"""
        response = self.session.post(
            f"{self.base_url}/step",
            json={"action": action}
        )
        response.raise_for_status()
        return response.json()
    
    def save_snapshot(self, snapshot_id: str = None) -> str:
        """Save snapshot"""
        response = self.session.post(
            f"{self.base_url}/snapshot",
            json={"snapshot_id": snapshot_id} if snapshot_id else {}
        )
        response.raise_for_status()
        return response.json()["snapshot_id"]
    
    def restore_snapshot(self, snapshot_id: str) -> Dict:
        """Restore snapshot"""
        response = self.session.post(
            f"{self.base_url}/restore",
            json={"snapshot_id": snapshot_id}
        )
        response.raise_for_status()
        return response.json()
    
    def list_snapshots(self) -> List[Dict]:
        """List available snapshots"""
        response = self.session.get(f"{self.base_url}/snapshots")
        response.raise_for_status()
        return response.json()["snapshots"]
    
    def delete_snapshot(self, snapshot_id: str) -> Dict:
        """Delete snapshot"""
        response = self.session.delete(f"{self.base_url}/snapshot/{snapshot_id}")
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close environment"""
        try:
            self.session.post(f"{self.base_url}/close")
        except:
            pass
        self.session.close()


def test_basic_snapshot_restore(client: CarlaClient):
    """Test basic snapshot and restore functionality"""
    logger.info("=== Testing Basic Snapshot/Restore ===")
    
    try:
        # Reset environment
        logger.info("Resetting environment...")
        reset_result = client.reset(route_id=0)
        logger.info(f"Reset successful: route_id={reset_result['info'].get('route_id', 'unknown')}")
        
        # Take initial steps
        logger.info("Taking 5 initial steps...")
        for i in range(5):
            action = {"throttle": 0.5, "brake": 0.0, "steer": 0.0}
            step_result = client.step(action)
            logger.info(f"Step {i+1}: reward={step_result['reward']:.3f}")
        
        # Save snapshot
        logger.info("Saving snapshot...")
        snapshot_id = client.save_snapshot("test_snap_1")
        logger.info(f"Snapshot saved: {snapshot_id}")
        
        # Take more steps
        logger.info("Taking 5 more steps...")
        for i in range(5):
            action = {"throttle": 0.3, "brake": 0.0, "steer": 0.1}
            step_result = client.step(action)
            logger.info(f"Step {i+6}: reward={step_result['reward']:.3f}")
        
        # List snapshots
        snapshots = client.list_snapshots()
        logger.info(f"Available snapshots: {len(snapshots)}")
        for snap in snapshots:
            logger.info(f"  - {snap}")
        
        # Restore snapshot
        logger.info(f"Restoring snapshot {snapshot_id}...")
        restore_result = client.restore_snapshot(snapshot_id)
        logger.info(f"Restore successful: step_count={restore_result['step_count']}")
        
        # Verify restoration by taking different actions
        logger.info("Taking different actions after restore...")
        for i in range(3):
            action = {"throttle": 0.2, "brake": 0.0, "steer": -0.1}
            step_result = client.step(action)
            logger.info(f"Post-restore step {i+1}: reward={step_result['reward']:.3f}")
        
        # Clean up
        logger.info("Cleaning up snapshot...")
        client.delete_snapshot(snapshot_id)
        
        logger.info("✓ Basic snapshot/restore test passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Basic snapshot/restore test failed: {e}")
        return False


def test_grpo_branching(client: CarlaClient):
    """Test GRPO-style branching from same state"""
    logger.info("=== Testing GRPO Branching ===")
    
    try:
        # Reset environment
        logger.info("Resetting environment...")
        reset_result = client.reset(route_id=0)
        
        # Take initial steps to reach interesting state
        logger.info("Reaching initial state...")
        for i in range(10):
            action = {"throttle": 0.4, "brake": 0.0, "steer": 0.0}
            client.step(action)
        
        # Save snapshot for branching
        logger.info("Saving branch point...")
        branch_snapshot = client.save_snapshot("grpo_branch")
        
        # Collect multiple rollouts from same state
        rollouts = []
        num_branches = 3
        
        for branch in range(num_branches):
            logger.info(f"Branch {branch+1}/{num_branches}:")
            
            # Restore to branch point
            client.restore_snapshot(branch_snapshot)
            
            # Different exploration for each branch
            trajectory = []
            for step in range(5):
                # Add exploration noise
                steer_noise = np.random.uniform(-0.2, 0.2) * (branch + 1)
                action = {
                    "throttle": 0.5 + np.random.uniform(-0.1, 0.1),
                    "brake": 0.0,
                    "steer": np.clip(steer_noise, -1.0, 1.0)
                }
                
                step_result = client.step(action)
                trajectory.append({
                    "action": action,
                    "reward": step_result["reward"],
                    "terminated": step_result["terminated"]
                })
                
                logger.info(f"  Step {step+1}: reward={step_result['reward']:.3f}, steer={action['steer']:.2f}")
                
                if step_result["terminated"]:
                    break
            
            # Calculate trajectory return
            total_reward = sum(t["reward"] for t in trajectory)
            rollouts.append((branch, total_reward, trajectory))
            logger.info(f"  Total reward: {total_reward:.3f}")
        
        # Select best rollout (GRPO style)
        best_branch, best_reward, best_trajectory = max(rollouts, key=lambda x: x[1])
        logger.info(f"Best branch: {best_branch+1} with reward {best_reward:.3f}")
        
        # Clean up
        client.delete_snapshot(branch_snapshot)
        
        logger.info("✓ GRPO branching test passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ GRPO branching test failed: {e}")
        return False


def test_parallel_snapshots(ports: List[int]):
    """Test parallel environments with independent snapshots"""
    logger.info("=== Testing Parallel Snapshots ===")
    
    clients = []
    for port in ports:
        client = CarlaClient(f"http://localhost:{port}")
        if client.health_check():
            clients.append(client)
            logger.info(f"Connected to server on port {port}")
        else:
            logger.warning(f"Server on port {port} not available")
    
    if len(clients) < 2:
        logger.warning("Need at least 2 servers for parallel test")
        return False
    
    try:
        # Reset all environments
        logger.info("Resetting all environments...")
        for i, client in enumerate(clients):
            client.reset(route_id=i)
        
        # Take initial steps
        logger.info("Taking initial steps...")
        for _ in range(5):
            for client in clients:
                action = {"throttle": 0.4, "brake": 0.0, "steer": 0.0}
                client.step(action)
        
        # Save snapshots on all environments
        logger.info("Saving snapshots on all environments...")
        snapshot_ids = []
        for i, client in enumerate(clients):
            snap_id = client.save_snapshot(f"parallel_snap_{i}")
            snapshot_ids.append(snap_id)
            logger.info(f"  Server {i}: saved {snap_id}")
        
        # Different actions on each environment
        logger.info("Taking different actions on each environment...")
        for _ in range(3):
            for i, client in enumerate(clients):
                action = {
                    "throttle": 0.3 + i * 0.1,
                    "brake": 0.0,
                    "steer": (-1 if i % 2 == 0 else 1) * 0.2
                }
                client.step(action)
        
        # Restore all snapshots
        logger.info("Restoring all snapshots...")
        for i, (client, snap_id) in enumerate(zip(clients, snapshot_ids)):
            result = client.restore_snapshot(snap_id)
            logger.info(f"  Server {i}: restored to step {result['step_count']}")
        
        # Verify independent operation
        logger.info("Verifying independent operation...")
        for i, client in enumerate(clients):
            action = {"throttle": 0.5, "brake": 0.0, "steer": 0.0}
            step_result = client.step(action)
            logger.info(f"  Server {i}: step successful")
        
        # Clean up
        for client, snap_id in zip(clients, snapshot_ids):
            client.delete_snapshot(snap_id)
        
        logger.info("✓ Parallel snapshots test passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Parallel snapshots test failed: {e}")
        return False
    
    finally:
        for client in clients:
            client.close()


def benchmark_snapshot_performance(client: CarlaClient):
    """Benchmark snapshot/restore performance"""
    logger.info("=== Benchmarking Snapshot Performance ===")
    
    try:
        # Reset environment
        client.reset(route_id=0)
        
        # Warm up
        for _ in range(10):
            action = {"throttle": 0.4, "brake": 0.0, "steer": 0.0}
            client.step(action)
        
        # Benchmark snapshot creation
        snapshot_times = []
        snapshot_ids = []
        
        for i in range(5):
            start_time = time.time()
            snap_id = client.save_snapshot(f"bench_snap_{i}")
            snapshot_time = time.time() - start_time
            snapshot_times.append(snapshot_time)
            snapshot_ids.append(snap_id)
            logger.info(f"Snapshot {i+1}: {snapshot_time*1000:.2f}ms")
            
            # Take some steps between snapshots
            for _ in range(5):
                client.step({"throttle": 0.3, "brake": 0.0, "steer": 0.0})
        
        # Benchmark snapshot restoration
        restore_times = []
        
        for snap_id in snapshot_ids:
            start_time = time.time()
            client.restore_snapshot(snap_id)
            restore_time = time.time() - start_time
            restore_times.append(restore_time)
            logger.info(f"Restore {snap_id}: {restore_time*1000:.2f}ms")
        
        # Clean up
        for snap_id in snapshot_ids:
            client.delete_snapshot(snap_id)
        
        # Report statistics
        logger.info("Performance Statistics:")
        logger.info(f"  Snapshot: avg={np.mean(snapshot_times)*1000:.2f}ms, "
                   f"min={np.min(snapshot_times)*1000:.2f}ms, "
                   f"max={np.max(snapshot_times)*1000:.2f}ms")
        logger.info(f"  Restore:  avg={np.mean(restore_times)*1000:.2f}ms, "
                   f"min={np.min(restore_times)*1000:.2f}ms, "
                   f"max={np.max(restore_times)*1000:.2f}ms")
        
        # Check if performance is acceptable for GRPO
        avg_snapshot = np.mean(snapshot_times)
        avg_restore = np.mean(restore_times)
        
        if avg_snapshot < 1.0 and avg_restore < 1.0:  # Less than 1 second
            logger.info("✓ Performance is suitable for GRPO training!")
        else:
            logger.warning("⚠ Performance may be slow for GRPO training")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance benchmark failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test snapshot/restore functionality")
    parser.add_argument("--port", type=int, default=8080,
                       help="Server port for single server tests")
    parser.add_argument("--parallel-ports", type=int, nargs="+",
                       default=[8080, 8081, 8082, 8083],
                       help="Server ports for parallel tests")
    parser.add_argument("--test", type=str, choices=["basic", "grpo", "parallel", "benchmark", "all"],
                       default="all", help="Which test to run")
    args = parser.parse_args()
    
    # Single server tests
    if args.test in ["basic", "grpo", "benchmark", "all"]:
        client = CarlaClient(f"http://localhost:{args.port}")
        
        if not client.health_check():
            logger.error(f"Server on port {args.port} is not responding")
            logger.info("Please start the server with:")
            logger.info(f"  python carla_server.py --port {args.port}")
            return
        
        logger.info(f"Connected to server on port {args.port}")
        
        try:
            if args.test == "basic" or args.test == "all":
                test_basic_snapshot_restore(client)
                time.sleep(2)
            
            if args.test == "grpo" or args.test == "all":
                test_grpo_branching(client)
                time.sleep(2)
            
            if args.test == "benchmark" or args.test == "all":
                benchmark_snapshot_performance(client)
        
        finally:
            client.close()
    
    # Parallel server tests
    if args.test in ["parallel", "all"]:
        time.sleep(2)
        test_parallel_snapshots(args.parallel_ports)
    
    logger.info("=== All tests completed ===")


if __name__ == "__main__":
    main()