#!/usr/bin/env python3
"""
ROBUST Microservice Manager for Bench2Drive

Enhanced version with:
- Automatic port cleanup before starting
- Self-recovery on failures
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
import socket
import atexit
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('RobustMicroserviceManager')


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
                        time.sleep(0.5)
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
                    time.sleep(0.5)
                    if not PortManager.is_port_in_use(port):
                        logger.info(f"Port {port} successfully freed")
                    else:
                        logger.warning(f"Port {port} still in use after cleanup")
                else:
                    logger.warning(f"Port {port} is in use (cleanup disabled)")
    
    @staticmethod
    def find_free_port(starting_port: int, max_attempts: int = 100) -> Optional[int]:
        """Find a free port starting from the given port"""
        for offset in range(max_attempts):
            port = starting_port + offset
            if not PortManager.is_port_in_use(port):
                return port
        return None


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
    carla_port: int
    gpu_id: int
    auto_restart: bool = True
    max_restarts: int = 3
    health_check_interval: int = 30


class RobustBench2DriveService:
    """Enhanced Bench2Drive microservice with robustness features"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.process = None
        self.is_running = False
        self.start_time = None
        self.restart_count = 0
        self.logger = logging.getLogger(f'Service-{config.service_id}')
        self.health_thread = None
        self.should_monitor = True
        
        # Path to carla_server.py
        self.server_script = Path(__file__).parent / "carla_server.py"
        
        # Required ports
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
        """Start the service with automatic port cleanup"""
        self.logger.info(f"Starting service {self.config.service_id}")
        
        # Clean up ports first
        self.cleanup_ports()
        time.sleep(1)
        
        # Verify ports are free
        for port in self.required_ports:
            if PortManager.is_port_in_use(port):
                self.logger.error(f"Port {port} still in use after cleanup")
                return False
        
        # Build command
        cmd = [
            sys.executable,
            str(self.server_script),
            "--port", str(self.config.api_port),
            "--carla-port", str(self.config.carla_port),
            "--server-id", f"service-{self.config.service_id}"
        ]
        
        # Set environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.config.gpu_id)
        
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
            
            # Wait for server to be ready
            if self._wait_for_server():
                self.is_running = True
                self.start_time = time.time()
                self.logger.info(f"✓ Service {self.config.service_id} ready")
                
                # Start health monitoring
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
        health_url = f"http://localhost:{self.config.api_port}/health"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=1)
                if response.status_code == 200:
                    return True
            except:
                pass
            
            # Check if process died
            if self.process and self.process.poll() is not None:
                self.logger.error(f"Process died with exit code {self.process.returncode}")
                return False
            
            time.sleep(1)
        
        return False
    
    def stop(self):
        """Stop the service and clean up"""
        self.logger.info(f"Stopping service {self.config.service_id}")
        self.should_monitor = False
        
        if self.health_thread:
            self.health_thread.join(timeout=5)
        
        if self.process:
            try:
                # Try graceful shutdown
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except:
                try:
                    # Force kill if needed
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass
            
            self.process = None
        
        # Clean up ports
        self.cleanup_ports()
        self.is_running = False
        self.logger.info("Service stopped and ports cleaned")
    
    def check_health(self) -> bool:
        """Check if the service is healthy"""
        if not self.is_running or not self.process:
            return False
        
        # Check process
        if self.process.poll() is not None:
            self.logger.warning(f"Process died with exit code {self.process.returncode}")
            return False
        
        # Check API health
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
        while self.should_monitor:
            time.sleep(self.config.health_check_interval)
            
            if not self.should_monitor:
                break
            
            if not self.check_health():
                self.logger.warning(f"Service {self.config.service_id} is unhealthy")
                if self.config.auto_restart:
                    self.restart()


class RobustMicroserviceManager:
    """Enhanced manager for multiple Bench2Drive microservices"""
    
    def __init__(self, num_services: int = 1, startup_delay: float = 0):
        self.num_services = num_services
        self.startup_delay = startup_delay
        self.services: List[RobustBench2DriveService] = []
        self.is_running = False
        
        # Load configuration
        self.load_config()
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def load_config(self):
        """Load configuration from config.yaml"""
        config_paths = [
            Path(__file__).parent.parent / "configs" / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
        ]
        
        self.config = None
        for config_path in config_paths:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                break
        
        if not self.config:
            logger.warning("No config.yaml found, using defaults")
            self.config = {}
    
    def initial_cleanup(self):
        """Perform initial cleanup before starting services"""
        logger.info("Performing initial cleanup...")
        
        # Kill existing CARLA processes
        ProcessManager.kill_carla_processes()
        time.sleep(2)
        
        # Clean up zombie processes
        ProcessManager.cleanup_zombie_processes()
        
        # Clean up all ports that will be used
        all_ports = []
        for i in range(self.num_services):
            all_ports.extend([
                8080 + i,  # API ports
                2000 + (i * 4),  # CARLA ports
                2001 + (i * 4),
                2002 + (i * 4),
            ])
        
        PortManager.cleanup_ports(all_ports)
        logger.info("Initial cleanup complete")
    
    def create_services(self):
        """Create service instances"""
        self.services = []
        
        for i in range(self.num_services):
            config = ServiceConfig(
                service_id=i,
                api_port=8080 + i,
                carla_port=2000 + (i * 4),
                gpu_id=i % 2,  # Alternate between GPU 0 and 1
                auto_restart=True,
                max_restarts=3,
                health_check_interval=30
            )
            
            service = RobustBench2DriveService(config)
            self.services.append(service)
    
    def start(self) -> bool:
        """Start all services with cleanup"""
        logger.info(f"Starting {self.num_services} services...")
        
        # Initial cleanup
        self.initial_cleanup()
        time.sleep(2)
        
        # Create services
        self.create_services()
        
        # Start services
        success_count = 0
        for i, service in enumerate(self.services):
            logger.info(f"Starting service {i}/{self.num_services}")
            
            if service.start():
                success_count += 1
                
                if self.startup_delay > 0 and i < len(self.services) - 1:
                    logger.info(f"Waiting {self.startup_delay}s before next service...")
                    time.sleep(self.startup_delay)
            else:
                logger.error(f"Failed to start service {i}")
        
        self.is_running = success_count > 0
        
        if success_count == self.num_services:
            logger.info(f"✓ All {self.num_services} services started successfully")
            return True
        else:
            logger.warning(f"Only {success_count}/{self.num_services} services started")
            return success_count > 0
    
    def stop(self):
        """Stop all services"""
        logger.info("Stopping all services...")
        
        for service in self.services:
            try:
                service.stop()
            except Exception as e:
                logger.error(f"Error stopping service: {e}")
        
        self.services = []
        self.is_running = False
        logger.info("All services stopped")
    
    def cleanup(self):
        """Cleanup on exit"""
        if self.is_running:
            self.stop()
        
        # Final cleanup
        ProcessManager.kill_carla_processes()
        logger.info("Cleanup complete")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)
    
    def monitor_loop(self):
        """Main monitoring loop"""
        try:
            while self.is_running:
                time.sleep(60)
                
                # Check overall health
                healthy_count = sum(1 for s in self.services if s.check_health())
                logger.info(f"Health check: {healthy_count}/{len(self.services)} services healthy")
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Robust Bench2Drive Microservice Manager')
    parser.add_argument('--num-services', type=int, default=1,
                        help='Number of services to start')
    parser.add_argument('--startup-delay', type=float, default=0,
                        help='Delay between starting services (seconds)')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Skip initial cleanup')
    
    args = parser.parse_args()
    
    manager = RobustMicroserviceManager(
        num_services=args.num_services,
        startup_delay=args.startup_delay
    )
    
    if manager.start():
        logger.info("Services running. Press Ctrl+C to stop.")
        manager.monitor_loop()
    else:
        logger.error("Failed to start services")
        sys.exit(1)


if __name__ == '__main__':
    main()