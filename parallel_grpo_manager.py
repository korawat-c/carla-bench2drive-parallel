#!/usr/bin/env python3
"""
True Parallel GRPO Manager for CARLA Microservices

This manager enables true parallel initialization where multiple CARLA instances
load their worlds simultaneously rather than sequentially. This is critical for
efficient GRPO training where time is saved by parallel world loading.

Key Features:
- Simultaneous world loading across multiple CARLA instances
- Parallel scenario initialization
- Coordinated snapshot creation for branching
- Efficient resource utilization
"""

import os
import sys
import time
import logging
import threading
import requests
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

# Add the project root to Python path
sys.path.insert(0, '/mnt3/Documents/AD_Framework/bench2drive-gymnasium')

from bench2drive_microservices.multi_instance_manager import MultiInstanceManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ParallelEnvironment:
    """Represents a parallel CARLA environment instance"""
    service_id: int
    api_url: str
    carla_port: int
    status: str = "initializing"
    world_loaded: bool = False
    scenario_loaded: bool = False
    ready_for_snapshot: bool = False


class ParallelGRPOManager:
    """Manager for true parallel GRPO environments"""

    def __init__(self, num_instances: int = 2, base_port: int = 8080):
        """
        Initialize parallel GRPO manager

        Args:
            num_instances: Number of parallel CARLA instances
            base_port: Base API port
        """
        self.num_instances = num_instances
        self.base_port = base_port
        self.environments: Dict[int, ParallelEnvironment] = {}
        self.multi_manager = MultiInstanceManager(num_instances=num_instances)

    def start_all_instances(self) -> bool:
        """Start all CARLA instances with parallel initialization"""
        logger.info(f"Starting {self.num_instances} CARLA instances in parallel mode")

        # Start all instances first
        if not self.multi_manager.start_all_instances(startup_delay=5):
            logger.error("Failed to start all instances")
            return False

        # Create environment objects
        for i in range(self.num_instances):
            api_port = self.base_port + i
            self.environments[i] = ParallelEnvironment(
                service_id=i,
                api_url=f"http://localhost:{api_port}",
                carla_port=2000 + (i * 6),  # Based on resource manager allocation
                status="starting"
            )

        return True

    def parallel_world_loading(self, route_id: int = 0) -> bool:
        """
        Load worlds in parallel across all instances

        Args:
            route_id: Route ID to use for all instances

        Returns:
            True if all instances loaded successfully
        """
        logger.info("Starting parallel world loading...")

        def load_single_world(env: ParallelEnvironment) -> bool:
            """Load world for a single environment"""
            try:
                logger.info(f"Loading world for environment {env.service_id}")

                # Reset environment to trigger world loading
                response = requests.post(
                    f"{env.api_url}/reset",
                    json={"route_id": route_id},
                    timeout=120  # Extended timeout for world loading
                )

                if response.status_code == 200:
                    env.world_loaded = True
                    env.status = "world_loaded"
                    logger.info(f"Environment {env.service_id} world loaded successfully")
                    return True
                else:
                    logger.error(f"Environment {env.service_id} failed to load world: {response.status_code}")
                    return False

            except Exception as e:
                logger.error(f"Error loading world for environment {env.service_id}: {e}")
                return False

        # Execute world loading in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_instances) as executor:
            future_to_env = {
                executor.submit(load_single_world, env): env
                for env in self.environments.values()
            }

            success_count = 0
            for future in concurrent.futures.as_completed(future_to_env):
                env = future_to_env[future]
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    logger.error(f"Environment {env.service_id} failed during parallel loading: {e}")

        logger.info(f"Parallel world loading complete: {success_count}/{self.num_instances} successful")
        return success_count == self.num_instances

    def warm_up_environments(self, warmup_steps: int = 5) -> bool:
        """
        Warm up all environments with a few steps to ensure they're ready

        Args:
            warmup_steps: Number of warmup steps

        Returns:
            True if all environments warmed up successfully
        """
        logger.info(f"Warming up {self.num_instances} environments with {warmup_steps} steps...")

        def warmup_single_env(env: ParallelEnvironment) -> bool:
            """Warm up a single environment"""
            try:
                # Send warmup actions
                for step in range(warmup_steps):
                    action = {"throttle": 0.5, "brake": 0.0, "steer": 0.0}

                    response = requests.post(
                        f"{env.api_url}/step",
                        json=action,
                        timeout=30
                    )

                    if response.status_code != 200:
                        logger.error(f"Environment {env.service_id} warmup failed at step {step}")
                        return False

                    # Small delay between steps
                    time.sleep(0.1)

                env.scenario_loaded = True
                env.ready_for_snapshot = True
                env.status = "ready"
                logger.info(f"Environment {env.service_id} warmup complete")
                return True

            except Exception as e:
                logger.error(f"Error warming up environment {env.service_id}: {e}")
                return False

        # Execute warmup in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_instances) as executor:
            future_to_env = {
                executor.submit(warmup_single_env, env): env
                for env in self.environments.values()
            }

            success_count = 0
            for future in concurrent.futures.as_completed(future_to_env):
                env = future_to_env[future]
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    logger.error(f"Environment {env.service_id} failed during warmup: {e}")

        logger.info(f"Environment warmup complete: {success_count}/{self.num_instances} successful")
        return success_count == self.num_instances

    def create_parallel_snapshots(self, snapshot_prefix: str = "parallel") -> Dict[int, str]:
        """
        Create snapshots on all environments in parallel

        Args:
            snapshot_prefix: Prefix for snapshot names

        Returns:
            Dictionary mapping service_id to snapshot_id
        """
        logger.info("Creating parallel snapshots...")

        def create_single_snapshot(env: ParallelEnvironment) -> Tuple[int, Optional[str]]:
            """Create snapshot for a single environment"""
            try:
                response = requests.post(
                    f"{env.api_url}/snapshot",
                    json={"snapshot_id": f"{snapshot_prefix}_service_{env.service_id}"},
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    snapshot_id = result.get('snapshot_id')
                    logger.info(f"Environment {env.service_id} snapshot created: {snapshot_id}")
                    return env.service_id, snapshot_id
                else:
                    logger.error(f"Environment {env.service_id} snapshot creation failed")
                    return env.service_id, None

            except Exception as e:
                logger.error(f"Error creating snapshot for environment {env.service_id}: {e}")
                return env.service_id, None

        # Execute snapshot creation in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_instances) as executor:
            future_to_env = {
                executor.submit(create_single_snapshot, env): env
                for env in self.environments.values()
            }

            snapshots = {}
            for future in concurrent.futures.as_completed(future_to_env):
                env = future_to_env[future]
                try:
                    service_id, snapshot_id = future.result()
                    if snapshot_id:
                        snapshots[service_id] = snapshot_id
                except Exception as e:
                    logger.error(f"Environment {env.service_id} failed during snapshot creation: {e}")

        logger.info(f"Parallel snapshot creation complete: {len(snapshots)}/{self.num_instances} successful")
        return snapshots

    def get_environment_status(self) -> Dict[int, Dict]:
        """Get status of all environments"""
        status = {}

        for service_id, env in self.environments.items():
            # Check health via API
            healthy = False
            try:
                response = requests.get(f"{env.api_url}/health", timeout=5)
                healthy = response.status_code == 200
            except:
                pass

            status[service_id] = {
                'service_id': service_id,
                'api_url': env.api_url,
                'carla_port': env.carla_port,
                'status': env.status,
                'world_loaded': env.world_loaded,
                'scenario_loaded': env.scenario_loaded,
                'ready_for_snapshot': env.ready_for_snapshot,
                'healthy': healthy
            }

        return status

    def print_status(self) -> None:
        """Print status of all parallel environments"""
        status = self.get_environment_status()

        print("\n" + "="*60)
        print("PARALLEL GRPO ENVIRONMENTS STATUS")
        print("="*60)

        for service_id, info in status.items():
            health_icon = "✓" if info['healthy'] else "✗"
            world_icon = "✓" if info['world_loaded'] else "✗"
            scenario_icon = "✓" if info['scenario_loaded'] else "✗"
            ready_icon = "✓" if info['ready_for_snapshot'] else "✗"

            print(f"\nEnvironment {service_id}:")
            print(f"  Status: {info['status']}")
            print(f"  Health: {health_icon} | World: {world_icon} | Scenario: {scenario_icon} | Ready: {ready_icon}")
            print(f"  API URL: {info['api_url']}")
            print(f"  CARLA Port: {info['carla_port']}")

        print("\n" + "="*60)

    def stop_all_instances(self) -> None:
        """Stop all CARLA instances"""
        logger.info("Stopping all parallel CARLA instances")
        self.multi_manager.stop_all_instances()
        self.environments.clear()
        logger.info("All parallel instances stopped")


def main():
    """Main function for testing parallel GRPO manager"""
    import argparse

    parser = argparse.ArgumentParser(description='Parallel GRPO Manager for CARLA')
    parser.add_argument('--num-instances', type=int, default=2,
                       help='Number of parallel CARLA instances')
    parser.add_argument('--base-port', type=int, default=8080,
                       help='Base API port (default: 8080)')
    parser.add_argument('--route-id', type=int, default=0,
                       help='Route ID for all instances (default: 0)')
    parser.add_argument('--warmup-steps', type=int, default=5,
                       help='Number of warmup steps (default: 5)')

    args = parser.parse_args()

    # Create parallel manager
    manager = ParallelGRPOManager(
        num_instances=args.num_instances,
        base_port=args.base_port
    )

    try:
        print("=== PARALLEL GRPO MANAGER TEST ===\n")

        # Step 1: Start all instances
        print("Step 1: Starting CARLA instances...")
        if not manager.start_all_instances():
            print("❌ Failed to start instances")
            return

        print("✅ All instances started")
        manager.print_status()

        # Step 2: Parallel world loading
        print(f"\nStep 2: Parallel world loading (Route {args.route_id})...")
        start_time = time.time()

        if not manager.parallel_world_loading(args.route_id):
            print("❌ Parallel world loading failed")
            manager.stop_all_instances()
            return

        loading_time = time.time() - start_time
        print(f"✅ Parallel world loading completed in {loading_time:.1f} seconds")
        manager.print_status()

        # Step 3: Warm up environments
        print(f"\nStep 3: Warm up environments ({args.warmup_steps} steps)...")
        start_time = time.time()

        if not manager.warm_up_environments(args.warmup_steps):
            print("❌ Environment warmup failed")
            manager.stop_all_instances()
            return

        warmup_time = time.time() - start_time
        print(f"✅ Environment warmup completed in {warmup_time:.1f} seconds")
        manager.print_status()

        # Step 4: Create parallel snapshots
        print(f"\nStep 4: Create parallel snapshots...")
        start_time = time.time()

        snapshots = manager.create_parallel_snapshots()

        snapshot_time = time.time() - start_time
        print(f"✅ Parallel snapshots created in {snapshot_time:.1f} seconds")
        print(f"   Snapshots: {snapshots}")

        # Final status
        print(f"\n🎉 PARALLEL GRPO SETUP COMPLETE!")
        print(f"   Total time: {loading_time + warmup_time + snapshot_time:.1f} seconds")
        print(f"   Efficiency: {args.num_instances} instances ready simultaneously")

        manager.print_status()

        # Keep running for manual testing
        print(f"\n📊 Environment Status Summary:")
        status = manager.get_environment_status()
        ready_count = sum(1 for info in status.values() if info['ready_for_snapshot'])
        print(f"   Ready environments: {ready_count}/{args.num_instances}")

        print(f"\n💡 Test URLs:")
        for service_id, info in status.items():
            print(f"   Environment {service_id}: {info['api_url']}")

        print(f"\nPress Ctrl+C to stop all instances...")

        # Monitor loop
        try:
            while True:
                time.sleep(30)
                ready_count = sum(1 for info in manager.get_environment_status().values()
                                 if info['healthy'] and info['ready_for_snapshot'])
                logger.info(f"Monitoring: {ready_count}/{args.num_instances} environments ready")
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")

    except Exception as e:
        logger.error(f"Error in parallel GRPO manager: {e}")
    finally:
        manager.stop_all_instances()


if __name__ == "__main__":
    main()