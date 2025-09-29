#!/usr/bin/env python3
"""
ROBUST Microservice Manager for Bench2Drive

Enhanced version with automatic port cleanup, self-recovery, and health monitoring.
This correctly uses carla_server.py which includes the building blocks.
Each microservice is just a carla_server.py instance that handles:
1. Starting CARLA through building blocks (with leaderboard)
2. Providing REST API for Gymnasium interface

Features:
- Automatic port cleanup before starting
- Health monitoring with auto-restart
- Graceful shutdown with cleanup
- Port conflict resolution
"""

import os
import sys
import time
import signal
import subprocess
import psutil
import logging
import yaml
import json
import socket
import atexit
import threading
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import requests

# Import resource manager for modular configuration
try:
    from resource_manager import get_resource_manager, ServiceResources
except ImportError:
    get_resource_manager = None
    ServiceResources = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('MicroserviceManager')


class PortManager:
    """Manages port allocation and cleanup"""
    
    @staticmethod
    def is_port_in_use(port: int) -> bool:
        """Check if a port is in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return False
            except:
                return True
    
    @staticmethod
    def kill_process_on_port(port: int) -> bool:
        """Kill any process using the specified port"""
        try:
            # Find process using the port
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
                        logger.info(f"Killing process {pid} using port {port}")
                        os.kill(int(pid), signal.SIGKILL)
                        time.sleep(2.0)  # Wait longer for CARLA processes to fully terminate
                    except:
                        pass
                return True
            return False
        except Exception as e:
            logger.debug(f"Error killing process on port {port}: {e}")
            return False
    
    @staticmethod
    def cleanup_ports(ports: List[int], force: bool = True) -> None:
        """Clean up a list of ports"""
        for port in ports:
            if PortManager.is_port_in_use(port):
                logger.info(f"Port {port} is in use, cleaning up...")
                if force:
                    PortManager.kill_process_on_port(port)
                    time.sleep(2.0)  # Wait longer for CARLA processes to fully terminate
                    if not PortManager.is_port_in_use(port):
                        logger.info(f"Port {port} successfully freed")
                    else:
                        logger.warning(f"Port {port} still in use after cleanup")


class ProcessManager:
    """Manages process cleanup"""
    
    @staticmethod
    def kill_carla_processes():
        """Kill all CARLA-related processes"""
        patterns = [
            "CarlaUE4",
            "carla_server.py",
            "microservice_manager",
            "CarlaUE4-Linux-Shipping"
        ]
        
        for pattern in patterns:
            try:
                subprocess.run(
                    f"pkill -f {pattern}",
                    shell=True,
                    capture_output=True
                )
            except:
                pass
    
    @staticmethod
    def cleanup_zombie_processes():
        """Clean up zombie processes"""
        for proc in psutil.process_iter(['pid', 'name', 'status']):
            try:
                if proc.info['status'] == psutil.STATUS_ZOMBIE:
                    logger.info(f"Cleaning zombie process: {proc.info['pid']} ({proc.info['name']})")
                    proc.kill()
            except:
                pass


@dataclass
class ServiceConfig:
    """Configuration for a single Bench2Drive microservice"""
    service_id: int
    api_port: int
    carla_port: int  # CARLA port that carla_server.py will use
    gpu_id: int
    auto_restart: bool = True
    max_restarts: int = 3
    health_check_interval: int = 300  # 5 minutes for CARLA world loading


class Bench2DriveService:
    """
    Single Bench2Drive microservice.
    
    This just manages ONE carla_server.py process.
    The carla_server.py handles:
    - Starting CARLA via building blocks
    - Loading leaderboard scenarios
    - Providing REST API
    """
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.process = None
        self.is_running = False
        self.start_time = None
        self.restart_count = 0
        self.health_thread = None
        self.should_monitor = True
        self.logger = logging.getLogger(f'Service-{config.service_id}')
        
        # Path to carla_server.py
        self.server_script = Path(__file__).parent / "carla_server.py"
        
        # Required ports for this service
        self.required_ports = [
            config.api_port,
            config.carla_port,
            config.carla_port + 1,  # CARLA streaming port
            config.carla_port + 2,  # CARLA secondary port
        ]
        
    def cleanup_ports(self):
        """Clean up all ports used by this service"""
        self.logger.info(f"Cleaning up ports for service {self.config.service_id}")
        PortManager.cleanup_ports(self.required_ports)
    
    def start(self) -> bool:
        """Start the carla_server.py process with port cleanup"""
        self.logger.info(f"Starting service {self.config.service_id}")
        
        # Clean up ports first
        self.cleanup_ports()
        time.sleep(1)
        
        # Verify ports are free
        for port in self.required_ports:
            if PortManager.is_port_in_use(port):
                self.logger.error(f"Port {port} still in use after cleanup")
                return False
        
        # Build command to start carla_server.py
        cmd = [
            sys.executable,  # Python interpreter
            str(self.server_script),
            "--port", str(self.config.api_port),
            "--carla-port", str(self.config.carla_port),
            "--server-id", f"service-{self.config.service_id}"
        ]
        
        # Set environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.config.gpu_id)
        # The carla_server.py will handle DISPLAY internally
        
        self.logger.info(f"Starting carla_server.py:")
        self.logger.info(f"  API port: {self.config.api_port}")
        self.logger.info(f"  CARLA port: {self.config.carla_port}")
        self.logger.info(f"  GPU: {self.config.gpu_id}")
        
        try:
            # Create log directory
            log_dir = Path(f"logs/service_{self.config.service_id}")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Start the server process
            with open(log_dir / "server.log", "w") as log_file:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    preexec_fn=os.setsid  # Create new process group
                )
            
            self.logger.info(f"Server process started with PID {self.process.pid}")
            
            # Wait for server to be ready (check health endpoint)
            if self._wait_for_server():
                self.is_running = True
                self.start_time = time.time()
                self.logger.info(f"✓ Service {self.config.service_id} ready")
                
                # Start health monitoring if enabled
                if self.config.auto_restart:
                    self.start_health_monitor()
                
                return True
            else:
                self.logger.error("Server did not become ready")
                self.stop()
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            return False
    
    def _wait_for_server(self, timeout: int = 60) -> bool:
        """Wait for the server to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if process is still running
            if self.process and self.process.poll() is not None:
                self.logger.error("Server process died")
                return False
            
            # Try health endpoint
            try:
                response = requests.get(
                    f"http://localhost:{self.config.api_port}/health",
                    timeout=2
                )
                if response.status_code == 200:
                    self.logger.info("Server is healthy")
                    return True
            except:
                pass
            
            time.sleep(2)
        
        self.logger.error(f"Server not ready after {timeout} seconds")
        return False
    
    def stop(self):
        """Stop the service and clean up"""
        self.logger.info(f"Stopping service {self.config.service_id}")
        self.should_monitor = False
        self.is_running = False
        
        # Stop health monitoring thread
        if self.health_thread:
            self.health_thread.join(timeout=5)
        
        if self.process and self.process.poll() is None:
            try:
                # Try graceful shutdown first
                self.process.terminate()
                self.process.wait(timeout=10)
            except:
                # Force kill if needed
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass
        
        # Clean up ports after stopping
        self.cleanup_ports()
        self.logger.info("Service stopped and ports cleaned")
    
    def check_health(self) -> bool:
        """Check if the service is healthy"""
        if not self.is_running or not self.process:
            return False
        
        # Check process is still running
        if self.process.poll() is not None:
            self.logger.warning(f"Process died with exit code {self.process.returncode}")
            return False
        
        # Check API health endpoint
        try:
            response = requests.get(
                f"http://localhost:{self.config.api_port}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def restart(self) -> bool:
        """Restart the service"""
        self.logger.info(f"Restarting service {self.config.service_id}")
        self.stop()
        time.sleep(2)
        
        if self.restart_count >= self.config.max_restarts:
            self.logger.error(f"Maximum restart attempts ({self.config.max_restarts}) reached")
            return False
        
        self.restart_count += 1
        success = self.start()
        
        if success:
            self.logger.info(f"Service restarted successfully (attempt {self.restart_count})")
        else:
            self.logger.error(f"Service restart failed (attempt {self.restart_count})")
        
        return success
    
    def start_health_monitor(self):
        """Start health monitoring thread"""
        self.should_monitor = True
        self.health_thread = threading.Thread(target=self._health_monitor_loop)
        self.health_thread.daemon = True
        self.health_thread.start()
    
    def _health_monitor_loop(self):
        """Health monitoring loop"""
        current_thread = threading.current_thread()
        while self.should_monitor:
            time.sleep(self.config.health_check_interval)

            if not self.should_monitor:
                break

            if not self.check_health():
                self.logger.warning(f"Service {self.config.service_id} is unhealthy")
                if self.config.auto_restart:
                    # Don't call restart directly from the health thread
                    # Just signal and let another thread handle it
                    self.logger.info(f"Scheduling restart for service {self.config.service_id}")
                    restart_thread = threading.Thread(target=self.restart)
                    restart_thread.daemon = True
                    restart_thread.start()
    
    def health_check(self) -> Dict:
        """Check health of the service"""
        health = {
            "service_id": self.config.service_id,
            "api_port": self.config.api_port,
            "carla_port": self.config.carla_port,
            "status": "unknown",
            "pid": None,
            "uptime": None
        }
        
        # Check process
        if self.process and self.process.poll() is None:
            health["pid"] = self.process.pid
            
            # Check API health
            try:
                response = requests.get(
                    f"http://localhost:{self.config.api_port}/health",
                    timeout=10  # Increased timeout for health checks
                )
                if response.status_code == 200:
                    health["status"] = "healthy"
                    data = response.json()
                    health["carla_connected"] = data.get("has_evaluator", False)
            except:
                health["status"] = "unhealthy"
        else:
            health["status"] = "dead"
        
        # Calculate uptime
        if self.start_time:
            health["uptime"] = time.time() - self.start_time
        
        return health


class MicroserviceManager:
    """
    Manages multiple Bench2Drive microservices.
    
    Each service is a carla_server.py instance that:
    1. Uses building blocks to start CARLA with leaderboard
    2. Provides REST API for Gymnasium interface
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.services: Dict[int, Bench2DriveService] = {}
        self.config = self._load_config(config_file)
        self.running = False
        self.monitor_thread = None
        
        # Port allocation with proper spacing
        # CARLA uses 3 consecutive ports, so we space by 4
        self.port_spacing = 4
        
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration"""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)

        # Default configuration matching config.yaml structure
        return {
            "ports": {
                "carla_base": 2000,
                "api_base": 8080,
                "streaming_base": 3000,
                "api_offset": 1,
                "carla_offset": 4,
                "streaming_offset": 10
            },
            "services": {
                "num_services": 2,
                "auto_restart": True,
                "health_check_interval": 30,
                "health_check_timeout": 5,
                "max_restart_attempts": 3
            },
            "gpu": {
                "available_gpus": [0]
            }
        }
    
    def spawn_service(self, service_id: Optional[int] = None) -> Bench2DriveService:
        """Spawn a new service"""
        if service_id is None:
            service_id = len(self.services)
        
        # Use resource manager if available
        if get_resource_manager:
            resource_mgr = get_resource_manager("config.yaml")
            resources = resource_mgr.allocate_resources(service_id)
            
            # Create service configuration from resource allocation
            config = ServiceConfig(
                service_id=service_id,
                api_port=resources.api_port,
                carla_port=resources.carla_port,
                gpu_id=resources.gpu_id
            )
            
            logger.info(f"[Resource Manager] Allocated resources for service {service_id}")
        else:
            # Fallback to config-based logic
            carla_port = self.config["ports"]["carla_base"] + (service_id * self.config["ports"]["carla_offset"])
            api_port = self.config["ports"]["api_base"] + (service_id * self.config["ports"]["api_offset"])

            # GPU assignment (round-robin)
            available_gpus = self.config["gpu"]["available_gpus"]
            gpu_id = available_gpus[service_id % len(available_gpus)]

            # Create service configuration
            config = ServiceConfig(
                service_id=service_id,
                api_port=api_port,
                carla_port=carla_port,
                gpu_id=gpu_id
            )

            logger.info(f"[Config-based] Allocated resources for service {service_id}")
        
        # Create and start service
        service = Bench2DriveService(config)
        if service.start():
            self.services[service_id] = service
            logger.info(f"✓ Spawned service {service_id}")
            logger.info(f"  API: http://localhost:{config.api_port}")
            logger.info(f"  CARLA port: {config.carla_port}")
            logger.info(f"  GPU: {config.gpu_id}")
            return service
        else:
            logger.error(f"Failed to spawn service {service_id}")
            return None
    
    def scale_to(self, num_services: int):
        """Scale to specified number of services"""
        current = len(self.services)
        
        if num_services > current:
            # Scale up
            for i in range(current, num_services):
                self.spawn_service(i)
        elif num_services < current:
            # Scale down
            for i in range(num_services, current):
                if i in self.services:
                    self.services[i].stop()
                    del self.services[i]
    
    def get_service_urls(self) -> List[str]:
        """Get API URLs of all healthy services"""
        urls = []
        for service in self.services.values():
            health = service.health_check()
            if health["status"] == "healthy":
                urls.append(f"http://localhost:{service.config.api_port}")
        return urls
    
    def health_check_all(self) -> Dict:
        """Check health of all services"""
        report = {
            "timestamp": time.time(),
            "services": {},
            "summary": {
                "total": len(self.services),
                "healthy": 0,
                "unhealthy": 0,
                "dead": 0
            }
        }
        
        for sid, service in self.services.items():
            health = service.health_check()
            report["services"][sid] = health
            
            status = health["status"]
            if status == "healthy":
                report["summary"]["healthy"] += 1
            elif status == "unhealthy":
                report["summary"]["unhealthy"] += 1
            else:
                report["summary"]["dead"] += 1
        
        return report
    
    def _monitor_services(self):
        """Background monitoring thread"""
        while self.running:
            time.sleep(self.config["services"].get("health_check_interval", 30))

            if not self.config["services"].get("auto_restart", True):
                continue
            
            # Auto-restart dead services
            report = self.health_check_all()
            for sid, health in report["services"].items():
                if health["status"] == "dead":
                    logger.warning(f"Service {sid} is dead, restarting...")
                    self.services[sid].stop()
                    self.services[sid].start()
    
    def start(self):
        """Start the manager and spawn initial services in parallel"""
        self.running = True

        # Spawn initial services in parallel for faster startup
        num_services = self.config["services"].get("num_services", 2)
        startup_delay = self.config["services"].get("startup_delay", 0)  # Delay between service starts
        logger.info(f"Starting {num_services} services in parallel...")

        # For true parallel startup, skip delays when startup_delay is 0
        if startup_delay == 0:
            # Start all services simultaneously using threads
            import concurrent.futures

            def start_service_thread(service_id):
                """Thread function to start a single service"""
                try:
                    self.spawn_service(service_id)
                    return service_id, True
                except Exception as e:
                    logger.error(f"Failed to start service {service_id}: {e}")
                    return service_id, False

            # Use ThreadPoolExecutor to start services in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_services) as executor:
                futures = [executor.submit(start_service_thread, i) for i in range(num_services)]

                # Wait for all services to start
                for future in concurrent.futures.as_completed(futures):
                    service_id, success = future.result()
                    if success:
                        logger.info(f"✓ Service {service_id} started successfully")
                    else:
                        logger.error(f"✗ Service {service_id} failed to start")
        else:
            # Sequential startup with delays (fallback for compatibility)
            for i in range(num_services):
                self.spawn_service(i)

                # Add delay between service starts if configured
                # This helps when running multiple services on the same GPU
                if startup_delay > 0 and i < num_services - 1:
                    logger.info(f"Waiting {startup_delay}s before starting next service...")
                    time.sleep(startup_delay)

        # Start monitoring
        self.monitor_thread = threading.Thread(target=self._monitor_services, daemon=True)
        self.monitor_thread.start()

        logger.info(f"Manager started with {len(self.services)} services")
    
    def stop(self):
        """Stop all services"""
        self.running = False
        
        logger.info("Stopping all services...")
        for service in self.services.values():
            service.stop()
        
        logger.info("All services stopped")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bench2Drive Microservice Manager")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--num-services", type=int, default=2, help="Number of services")
    parser.add_argument("--startup-delay", type=int, default=0, 
                       help="Delay in seconds between starting services (useful for same GPU)")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()
    
    manager = MicroserviceManager(args.config)
    
    if args.num_services:
        manager.config["services"]["num_services"] = args.num_services

    if args.startup_delay:
        manager.config["services"]["startup_delay"] = args.startup_delay
    
    manager.start()
    
    try:
        if args.test:
            # Test mode - check health and exit
            time.sleep(30)
            report = manager.health_check_all()
            print(json.dumps(report, indent=2))
        else:
            # Run forever
            logger.info("Services running. Press Ctrl+C to stop.")
            while True:
                time.sleep(60)
                report = manager.health_check_all()
                logger.info(f"Status: {report['summary']}")
                
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        manager.stop()


if __name__ == "__main__":
    main()