#!/usr/bin/env python3
"""
FIXED Microservice Manager for Bench2Drive

This correctly uses carla_server.py which includes the building blocks.
Each microservice is just a carla_server.py instance that handles:
1. Starting CARLA through building blocks (with leaderboard)
2. Providing REST API for Gymnasium interface

The building blocks handle all CARLA startup, so we DON'T start CARLA directly!
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


@dataclass
class ServiceConfig:
    """Configuration for a single Bench2Drive microservice"""
    service_id: int
    api_port: int
    carla_port: int  # CARLA port that carla_server.py will use
    gpu_id: int


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
        self.logger = logging.getLogger(f'Service-{config.service_id}')
        
        # Path to carla_server.py
        self.server_script = Path(__file__).parent / "carla_server.py"
        
    def start(self) -> bool:
        """Start the carla_server.py process"""
        self.logger.info(f"Starting service {self.config.service_id}")
        
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
        """Stop the service"""
        self.logger.info(f"Stopping service {self.config.service_id}")
        self.is_running = False
        
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
            
            self.logger.info("Service stopped")
    
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
                    timeout=2
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
        
        # Default configuration
        return {
            "base_carla_port": 2000,
            "base_api_port": 8080,
            "num_services": 2,
            "gpus": [0],  # Available GPUs
            "auto_restart": True,
            "health_check_interval": 30
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
            # Fallback to original logic
            carla_port = self.config["base_carla_port"] + (service_id * self.port_spacing)
            api_port = self.config["base_api_port"] + service_id
            
            # GPU assignment (round-robin)
            available_gpus = self.config.get("gpus", [0])
            gpu_id = available_gpus[service_id % len(available_gpus)]
            
            # Create service configuration
            config = ServiceConfig(
                service_id=service_id,
                api_port=api_port,
                carla_port=carla_port,
                gpu_id=gpu_id
            )
            
            logger.info(f"[Fallback] Using hardcoded resource allocation")
        
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
            time.sleep(self.config.get("health_check_interval", 30))
            
            if not self.config.get("auto_restart", True):
                continue
            
            # Auto-restart dead services
            report = self.health_check_all()
            for sid, health in report["services"].items():
                if health["status"] == "dead":
                    logger.warning(f"Service {sid} is dead, restarting...")
                    self.services[sid].stop()
                    self.services[sid].start()
    
    def start(self):
        """Start the manager and spawn initial services"""
        self.running = True
        
        # Spawn initial services
        num_services = self.config.get("num_services", 2)
        startup_delay = self.config.get("startup_delay", 0)  # Delay between service starts
        logger.info(f"Starting {num_services} services...")
        
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
        
        logger.info(f"Manager started with {num_services} services")
    
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
        manager.config["num_services"] = args.num_services
    
    if args.startup_delay:
        manager.config["startup_delay"] = args.startup_delay
    
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