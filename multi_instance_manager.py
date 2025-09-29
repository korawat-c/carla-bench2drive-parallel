#!/usr/bin/env python3
"""
Enhanced Multi-Instance CARLA Initialization Script
This script properly initializes multiple CARLA instances with robust port management
and resource allocation to avoid conflicts during parallel execution.
"""

import os
import sys
import time
import signal
import subprocess
import logging
import socket
import threading
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

# Add the project root to Python path
sys.path.insert(0, '/mnt3/Documents/AD_Framework/bench2drive-gymnasium')

# Import resource manager
from bench2drive_microservices.server.resource_manager import get_resource_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CarlaInstance:
    """Represents a CARLA instance with its resources"""
    service_id: int
    api_port: int
    carla_port: int
    streaming_port: int
    tm_port: int
    gpu_id: int
    process: Optional[subprocess.Popen] = None
    healthy: bool = False

    def get_api_url(self) -> str:
        return f"http://localhost:{self.api_port}"

    def get_health_url(self) -> str:
        return f"{self.get_api_url()}/health"


class MultiInstanceManager:
    """Manages multiple CARLA instances with proper resource allocation"""

    def __init__(self, num_instances: int = 2, gpus: List[int] = None):
        """
        Initialize multi-instance manager

        Args:
            num_instances: Number of CARLA instances to start
            gpus: List of GPU IDs to use (default: [0])
        """
        self.num_instances = num_instances
        self.gpus = gpus or [0]
        self.instances: Dict[int, CarlaInstance] = {}
        self.resource_manager = get_resource_manager()

        # Configure resource manager for distributed GPU usage
        self.resource_manager.config['gpu']['strategy'] = 'distributed'
        self.resource_manager.config['gpu']['available_gpus'] = self.gpus

        # Enhanced port configuration to avoid conflicts
        self.resource_manager.config['ports']['carla_offset'] = 6  # Larger offset
        self.resource_manager.config['ports']['streaming_offset'] = 20

        logger.info(f"Initializing {num_instances} CARLA instances on GPUs: {self.gpus}")

    def check_port_availability(self, service_id: int) -> Tuple[bool, List[str]]:
        """Check if all required ports are available for a service"""
        resources = self.resource_manager.allocate_resources(service_id)

        # CARLA needs multiple consecutive ports
        carla_ports_needed = [
            resources.carla_port,      # Main RPC port
            resources.carla_port + 1,  # Secondary port
            resources.carla_port + 2,  # Tertiary port
        ]

        all_ports = [
            ('API', resources.api_port),
            ('Streaming', resources.streaming_port),
            ('TM', resources.tm_port),
        ] + [(f'CARLA-{i}', port) for i, port in enumerate(carla_ports_needed)]

        unavailable = []
        for name, port in all_ports:
            if not self._is_port_free(port):
                unavailable.append(f"{name}:{port}")

        return len(unavailable) == 0, unavailable

    def _is_port_free(self, port: int) -> bool:
        """Check if a port is free"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return True
            except OSError:
                return False

    def cleanup_ports_for_service(self, service_id: int) -> bool:
        """Clean up all ports for a specific service"""
        resources = self.resource_manager.allocate_resources(service_id)

        ports_to_clean = [
            resources.api_port,
            resources.streaming_port,
            resources.tm_port,
            resources.carla_port,
            resources.carla_port + 1,
            resources.carla_port + 2,
        ]

        cleaned_any = False
        for port in ports_to_clean:
            if self._kill_process_on_port(port):
                cleaned_any = True
                time.sleep(1.0)  # Wait for process to terminate

        return cleaned_any

    def _kill_process_on_port(self, port: int) -> bool:
        """Kill process using a specific port"""
        try:
            result = subprocess.run(
                f"lsof -ti:{port}",
                shell=True,
                capture_output=True,
                text=True
            )

            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        logger.info(f"Killing process {pid} on port {port}")
                        os.kill(int(pid), signal.SIGKILL)
                    except:
                        pass
                return True
            return False
        except:
            return False

    def start_instance(self, service_id: int, max_retries: int = 3) -> bool:
        """Start a single CARLA instance"""
        logger.info(f"Starting CARLA instance {service_id}")

        for attempt in range(max_retries):
            try:
                # Clean up ports before starting
                self.cleanup_ports_for_service(service_id)

                # Check port availability
                available, unavailable = self.check_port_availability(service_id)
                if not available:
                    logger.warning(f"Ports not available for service {service_id}: {unavailable}")
                    if attempt < max_retries - 1:
                        time.sleep(5)  # Wait before retry
                        continue
                    return False

                # Get resources
                resources = self.resource_manager.allocate_resources(service_id)

                # Create instance
                instance = CarlaInstance(
                    service_id=service_id,
                    api_port=resources.api_port,
                    carla_port=resources.carla_port,
                    streaming_port=resources.streaming_port,
                    tm_port=resources.tm_port,
                    gpu_id=resources.gpu_id
                )

                # Start CARLA server
                carla_cmd = [
                    sys.executable,
                    '/mnt3/Documents/AD_Framework/bench2drive-gymnasium/bench2drive_microservices/server/carla_server.py',
                    '--port', str(resources.api_port),
                    '--carla-port', str(resources.carla_port),
                    '--server-id', f'service-{service_id}'
                ]

                # Set environment variables for GPU
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(resources.gpu_id)

                logger.info(f"Starting CARLA server for service-{service_id}")
                logger.info(f"Command: {' '.join(carla_cmd)}")

                process = subprocess.Popen(
                    carla_cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )

                instance.process = process
                self.instances[service_id] = instance

                # Wait for service to be healthy
                if self._wait_for_healthy(instance, timeout=60):
                    logger.info(f"CARLA instance {service_id} started successfully")
                    return True
                else:
                    logger.error(f"CARLA instance {service_id} failed to become healthy")
                    self.stop_instance(service_id)

            except Exception as e:
                logger.error(f"Error starting CARLA instance {service_id}: {e}")
                if service_id in self.instances:
                    self.stop_instance(service_id)

            if attempt < max_retries - 1:
                logger.info(f"Retrying service {service_id} (attempt {attempt + 2})")
                time.sleep(10)

        return False

    def _wait_for_healthy(self, instance: CarlaInstance, timeout: int = 60) -> bool:
        """Wait for instance to become healthy"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(instance.get_health_url(), timeout=5)
                if response.status_code == 200:
                    instance.healthy = True
                    return True
            except:
                pass

            time.sleep(2)

        return False

    def stop_instance(self, service_id: int) -> bool:
        """Stop a specific CARLA instance"""
        if service_id not in self.instances:
            return True

        instance = self.instances[service_id]

        try:
            # Try graceful shutdown first
            requests.post(f"{instance.get_api_url()}/close", timeout=5)
        except:
            pass

        # Kill the process
        if instance.process:
            try:
                os.killpg(os.getpgid(instance.process.pid), signal.SIGTERM)
                time.sleep(2)
                os.killpg(os.getpgid(instance.process.pid), signal.SIGKILL)
            except:
                pass

        # Clean up ports
        self.cleanup_ports_for_service(service_id)

        del self.instances[service_id]
        logger.info(f"Stopped CARLA instance {service_id}")
        return True

    def start_all_instances(self, startup_delay: int = 15) -> bool:
        """Start all CARLA instances with delays to avoid conflicts"""
        logger.info(f"Starting {self.num_instances} CARLA instances")

        success_count = 0

        for i in range(self.num_instances):
            logger.info(f"Starting instance {i}/{self.num_instances}")

            if self.start_instance(i):
                success_count += 1

                # Don't delay after the last instance
                if i < self.num_instances - 1:
                    logger.info(f"Waiting {startup_delay} seconds before starting next instance...")
                    time.sleep(startup_delay)
            else:
                logger.error(f"Failed to start instance {i}")

        logger.info(f"Successfully started {success_count}/{self.num_instances} instances")
        return success_count == self.num_instances

    def stop_all_instances(self) -> None:
        """Stop all CARLA instances"""
        logger.info("Stopping all CARLA instances")

        for service_id in list(self.instances.keys()):
            self.stop_instance(service_id)

        logger.info("All CARLA instances stopped")

    def get_instance_status(self) -> Dict[int, Dict]:
        """Get status of all instances"""
        status = {}

        for service_id, instance in self.instances.items():
            status[service_id] = {
                'service_id': service_id,
                'api_url': instance.get_api_url(),
                'api_port': instance.api_port,
                'carla_port': instance.carla_port,
                'streaming_port': instance.streaming_port,
                'tm_port': instance.tm_port,
                'gpu_id': instance.gpu_id,
                'healthy': instance.healthy,
                'process_alive': instance.process and instance.process.poll() is None
            }

        return status

    def print_status(self) -> None:
        """Print status of all instances"""
        status = self.get_instance_status()

        print("\n" + "="*60)
        print("CARLA Multi-Instance Status")
        print("="*60)

        for service_id, info in status.items():
            health_status = "✓ Healthy" if info['healthy'] else "✗ Unhealthy"
            process_status = "Running" if info['process_alive'] else "Stopped"

            print(f"\nService {service_id}:")
            print(f"  Status: {health_status} | Process: {process_status}")
            print(f"  API URL: {info['api_url']}")
            print(f"  Ports: CARLA={info['carla_port']}, API={info['api_port']}")
            print(f"         Streaming={info['streaming_port']}, TM={info['tm_port']}")
            print(f"  GPU: {info['gpu_id']}")

        print("\n" + "="*60)

    def wait_for_all_healthy(self, timeout: int = 120) -> bool:
        """Wait for all instances to become healthy"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_healthy = all(instance.healthy for instance in self.instances.values())
            if all_healthy:
                logger.info("All instances are healthy")
                return True

            logger.info("Waiting for all instances to be healthy...")
            time.sleep(5)

        return False


def main():
    """Main function for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Instance CARLA Manager')
    parser.add_argument('--num-instances', type=int, default=2,
                       help='Number of CARLA instances to start')
    parser.add_argument('--gpus', type=str, default='0',
                       help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--startup-delay', type=int, default=15,
                       help='Delay between instance startups (seconds)')
    parser.add_argument('--status-only', action='store_true',
                       help='Only print status, do not start/stop instances')
    parser.add_argument('--stop', action='store_true',
                       help='Stop all instances instead of starting')

    args = parser.parse_args()

    # Parse GPU list
    gpus = [int(g.strip()) for g in args.gpus.split(',') if g.strip()]

    # Create manager
    manager = MultiInstanceManager(num_instances=args.num_instances, gpus=gpus)

    if args.status_only:
        manager.print_status()
        return

    if args.stop:
        manager.stop_all_instances()
        return

    try:
        # Start all instances
        if manager.start_all_instances(startup_delay=args.startup_delay):
            logger.info("All instances started successfully")

            # Wait for all to be healthy
            if manager.wait_for_all_healthy():
                logger.info("All instances are ready!")
                manager.print_status()

                # Keep running until interrupted
                logger.info("Press Ctrl+C to stop all instances...")
                while True:
                    time.sleep(1)
            else:
                logger.error("Not all instances became healthy")
                manager.stop_all_instances()
                sys.exit(1)
        else:
            logger.error("Failed to start all instances")
            manager.stop_all_instances()
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        manager.stop_all_instances()


if __name__ == "__main__":
    main()