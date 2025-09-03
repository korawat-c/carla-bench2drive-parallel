"""
CARLA server manager for handling multiple instances.
Manages port allocation, process lifecycle, and health monitoring.
"""

import os
import subprocess
import time
import signal
import atexit
import psutil
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import socket
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CarlaServer:
    """Information about a running CARLA server."""
    server_id: str
    carla_port: int
    tm_port: int  # Traffic Manager port
    api_port: int
    api_url: str
    carla_process: Optional[subprocess.Popen] = None
    api_process: Optional[subprocess.Popen] = None
    gpu_id: int = 0
    status: str = "starting"


class CarlaServerManager:
    """
    Manages multiple CARLA server instances for parallel environments.
    
    This class handles the critical challenge of running multiple CARLA
    instances on the same machine without conflicts.
    
    Key features:
    - Automatic port allocation with proper spacing
    - GPU assignment for load distribution
    - Process lifecycle management
    - Health monitoring and auto-restart
    - Clean shutdown on exit
    """
    
    def __init__(
        self,
        carla_root: Optional[str] = None,
        bench2drive_root: Optional[str] = None,
        api_script: Optional[str] = None,
        log_dir: Optional[str] = None
    ):
        """
        Initialize server manager.
        
        Args:
            carla_root: Path to CARLA installation
            bench2drive_root: Path to Bench2Drive
            api_script: Path to API server script
            log_dir: Directory for log files
        """
        # Set paths from environment or arguments
        self.carla_root = carla_root or os.environ.get(
            "CARLA_ROOT",
            "/mnt3/Documents/AD_Framework/carla0915"
        )
        self.bench2drive_root = bench2drive_root or os.environ.get(
            "WORK_DIR",
            "/mnt3/Documents/AD_Framework/Bench2Drive"
        )
        
        # API server script - use our integrated carla_server.py
        if api_script is None:
            api_script = Path(__file__).parent / "carla_server.py"
        self.api_script = str(api_script)
        
        # Log directory
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Track running servers
        self.servers: Dict[str, CarlaServer] = {}
        
        # Register cleanup on exit
        atexit.register(self.cleanup_all)
        
        logger.info(f"Initialized CarlaServerManager")
        logger.info(f"CARLA root: {self.carla_root}")
        logger.info(f"Bench2Drive root: {self.bench2drive_root}")
    
    def start_servers(
        self,
        num_servers: int = 4,
        carla_ports: Optional[List[int]] = None,
        tm_ports: Optional[List[int]] = None,
        api_ports: Optional[List[int]] = None,
        gpu_assignment: Optional[List[int]] = None,
        render_offscreen: bool = True,
        quality: str = "Low"  # Low/Epic
    ) -> List[CarlaServer]:
        """
        Start multiple CARLA servers with proper port spacing.
        
        Args:
            num_servers: Number of servers to start
            carla_ports: CARLA RPC ports (auto-generated if None)
            tm_ports: Traffic Manager ports (auto-generated if None)
            api_ports: API wrapper ports (auto-generated if None)
            gpu_assignment: GPU IDs for each server
            render_offscreen: Run in offscreen mode
            quality: Rendering quality
        
        Returns:
            List of CarlaServer instances
        """
        # Generate ports with proper spacing if not provided
        if carla_ports is None:
            carla_ports = [2000 + i * 2 for i in range(num_servers)]
        if tm_ports is None:
            tm_ports = [3000 + i * 2 for i in range(num_servers)]
        if api_ports is None:
            api_ports = [8080 + i for i in range(num_servers)]
        
        # Default GPU assignment
        if gpu_assignment is None:
            # Distribute across available GPUs
            try:
                import torch
                num_gpus = torch.cuda.device_count()
                gpu_assignment = [i % num_gpus for i in range(num_servers)]
            except:
                gpu_assignment = [0] * num_servers
        
        logger.info(f"Starting {num_servers} CARLA servers...")
        logger.info(f"CARLA ports: {carla_ports}")
        logger.info(f"TM ports: {tm_ports}")
        logger.info(f"API ports: {api_ports}")
        logger.info(f"GPU assignment: {gpu_assignment}")
        
        servers = []
        for i in range(num_servers):
            server_id = f"carla_server_{i}"
            
            # Check port availability
            if not self._is_port_available(carla_ports[i]):
                logger.warning(f"CARLA port {carla_ports[i]} is in use, cleaning up...")
                self._kill_process_on_port(carla_ports[i])
                time.sleep(2)
            
            if not self._is_port_available(api_ports[i]):
                logger.warning(f"API port {api_ports[i]} is in use, cleaning up...")
                self._kill_process_on_port(api_ports[i])
                time.sleep(2)
            
            # Create server info
            server = CarlaServer(
                server_id=server_id,
                carla_port=carla_ports[i],
                tm_port=tm_ports[i],
                api_port=api_ports[i],
                api_url=f"http://localhost:{api_ports[i]}",
                gpu_id=gpu_assignment[i]
            )
            
            # Start CARLA instance
            self._start_carla_instance(server, render_offscreen, quality)
            
            # Wait a bit between starts to avoid conflicts
            time.sleep(5)
            
            # Start API wrapper
            self._start_api_wrapper(server)
            
            # Wait for server to be ready
            if self._wait_for_server(server, timeout=60):
                server.status = "running"
                servers.append(server)
                self.servers[server_id] = server
                logger.info(f"Started {server_id} successfully")
            else:
                logger.error(f"Failed to start {server_id}")
                self._cleanup_server(server)
        
        logger.info(f"Successfully started {len(servers)}/{num_servers} servers")
        return servers
    
    def _start_carla_instance(self, server: CarlaServer, render_offscreen: bool, quality: str):
        """Start a CARLA instance with proper configuration."""
        carla_executable = os.path.join(self.carla_root, "CarlaUE4.sh")
        
        # Build command
        cmd = [carla_executable]
        cmd.extend(["-carla-rpc-port", str(server.carla_port)])
        cmd.extend(["-carla-streaming-port", "0"])  # Disable streaming
        
        if render_offscreen:
            cmd.append("-RenderOffScreen")
        
        cmd.extend(["-quality-level", quality])
        cmd.append("-nosound")
        
        # GPU selection
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(server.gpu_id)
        
        # Log file
        log_file = self.log_dir / f"{server.server_id}_carla.log"
        
        logger.info(f"Starting CARLA: {' '.join(cmd)}")
        
        # Start process
        with open(log_file, "w") as log:
            server.carla_process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
        
        # Wait for CARLA to start
        time.sleep(10)
    
    def _start_api_wrapper(self, server: CarlaServer):
        """Start the API wrapper for a CARLA instance."""
        # Build command
        cmd = [
            "python",  # Use current python interpreter
            self.api_script,
            "--port", str(server.api_port),
            "--carla-port", str(server.carla_port),
            "--server-id", server.server_id
        ]
        
        # Environment setup
        env = os.environ.copy()
        env["CARLA_ROOT"] = self.carla_root
        env["WORK_DIR"] = self.bench2drive_root
        env["CUDA_VISIBLE_DEVICES"] = str(server.gpu_id)
        
        # Log file
        log_file = self.log_dir / f"{server.server_id}_api.log"
        
        logger.info(f"Starting API wrapper: {' '.join(cmd)}")
        
        # Start process
        with open(log_file, "w") as log:
            server.api_process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
    
    def _wait_for_server(self, server: CarlaServer, timeout: int = 60) -> bool:
        """Wait for server to be ready."""
        import requests
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if processes are still running
            if server.carla_process and server.carla_process.poll() is not None:
                logger.error(f"CARLA process died for {server.server_id}")
                return False
            
            if server.api_process and server.api_process.poll() is not None:
                logger.error(f"API process died for {server.server_id}")
                return False
            
            # Try health check
            try:
                response = requests.get(
                    f"{server.api_url}/health",
                    timeout=2
                )
                if response.status_code == 200:
                    return True
            except:
                pass
            
            time.sleep(2)
        
        return False
    
    def check_health(self) -> Dict[str, str]:
        """Check health of all servers."""
        import requests
        
        health_status = {}
        for server_id, server in self.servers.items():
            try:
                response = requests.get(
                    f"{server.api_url}/health",
                    timeout=5
                )
                if response.status_code == 200:
                    health_status[server_id] = "healthy"
                else:
                    health_status[server_id] = "unhealthy"
            except:
                health_status[server_id] = "unreachable"
            
            # Check process status
            if server.carla_process and server.carla_process.poll() is not None:
                health_status[server_id] = "carla_dead"
            if server.api_process and server.api_process.poll() is not None:
                health_status[server_id] = "api_dead"
        
        return health_status
    
    def restart_server(self, server_id: str) -> bool:
        """Restart a specific server."""
        if server_id not in self.servers:
            logger.error(f"Unknown server: {server_id}")
            return False
        
        server = self.servers[server_id]
        logger.info(f"Restarting {server_id}...")
        
        # Cleanup existing processes
        self._cleanup_server(server)
        time.sleep(5)
        
        # Restart CARLA
        self._start_carla_instance(server, True, "Low")
        time.sleep(5)
        
        # Restart API
        self._start_api_wrapper(server)
        
        # Wait for ready
        if self._wait_for_server(server, timeout=60):
            server.status = "running"
            logger.info(f"Successfully restarted {server_id}")
            return True
        else:
            logger.error(f"Failed to restart {server_id}")
            return False
    
    def _cleanup_server(self, server: CarlaServer):
        """Clean up a server's processes."""
        # Kill API process
        if server.api_process:
            try:
                if os.name != 'nt':
                    os.killpg(os.getpgid(server.api_process.pid), signal.SIGTERM)
                else:
                    server.api_process.terminate()
                server.api_process.wait(timeout=5)
            except:
                if server.api_process.poll() is None:
                    server.api_process.kill()
        
        # Kill CARLA process
        if server.carla_process:
            try:
                if os.name != 'nt':
                    os.killpg(os.getpgid(server.carla_process.pid), signal.SIGTERM)
                else:
                    server.carla_process.terminate()
                server.carla_process.wait(timeout=5)
            except:
                if server.carla_process.poll() is None:
                    server.carla_process.kill()
    
    def cleanup_all(self):
        """Clean up all servers."""
        logger.info("Cleaning up all servers...")
        
        for server in self.servers.values():
            self._cleanup_server(server)
        
        # Additional cleanup of any orphaned CARLA processes
        try:
            os.system("pkill -f CarlaUE4")
            os.system("pkill -f 'python.*server.py'")
        except:
            pass
        
        logger.info("Cleanup complete")
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return True
        except OSError:
            return False
    
    def _kill_process_on_port(self, port: int):
        """Kill process using a specific port."""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        logger.info(f"Killing process {proc.pid} on port {port}")
                        proc.kill()
                        return
        except:
            # Fallback to system commands
            os.system(f"lsof -ti:{port} | xargs kill -9 2>/dev/null")


def spawn_carla_servers(
    num_servers: int = 4,
    **kwargs
) -> Tuple[CarlaServerManager, List[CarlaServer]]:
    """
    Convenience function to spawn CARLA servers.
    
    Args:
        num_servers: Number of servers to spawn
        **kwargs: Additional arguments for CarlaServerManager
    
    Returns:
        Tuple of (manager, servers)
    """
    manager = CarlaServerManager()
    servers = manager.start_servers(num_servers, **kwargs)
    return manager, servers