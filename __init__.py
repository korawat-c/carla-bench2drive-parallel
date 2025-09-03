"""
Gymnasium-compliant CARLA environment for Bench2Drive.

This package provides:
- CarlaEnv: Main Gymnasium environment for single CARLA instance
- VectorizedCarlaEnv: Parallel environments for training  
- CarlaServerManager: Manages multiple CARLA servers
- APIAgent: Agent that receives actions via REST API
- Utility functions for starting servers and environments

Key features:
- NO MOCKING - uses real CARLA instances
- Full Gymnasium API compliance
- GRPO support with snapshot/restore
- Multi-instance support for parallel training
- Bench2Drive integration with building blocks
"""

from typing import List, Tuple, Any
from client.carla_env import CarlaEnv
from client.vectorized_env import VectorizedCarlaEnv
from server.microservice_manager import MicroserviceManager, Bench2DriveService

# Try to import old components for compatibility
try:
    from tests.server_manager import CarlaServerManager, CarlaServer, spawn_carla_servers
except ImportError:
    CarlaServerManager = None
    CarlaServer = None
    spawn_carla_servers = None

try:
    from client.api_agent import APIAgent, get_entry_point, create_api_agent, validate_action
except ImportError:
    APIAgent = None
    get_entry_point = None
    create_api_agent = None
    validate_action = None

# Convenience functions
def make_carla_env(server_url: str = "http://localhost:8080", **kwargs) -> CarlaEnv:
    """
    Create a single CARLA environment.
    
    Args:
        server_url: URL of the CARLA API server
        **kwargs: Additional environment arguments
        
    Returns:
        CarlaEnv instance
    """
    return CarlaEnv(server_url=server_url, **kwargs)

def make_parallel_envs(
    num_envs: int = 4,
    base_api_port: int = 8080,
    **kwargs
) -> VectorizedCarlaEnv:
    """
    Create parallel CARLA environments.
    
    Args:
        num_envs: Number of parallel environments
        base_api_port: Starting API port
        **kwargs: Additional environment arguments
        
    Returns:
        VectorizedCarlaEnv instance
    """
    server_urls = [f"http://localhost:{base_api_port + i}" for i in range(num_envs)]
    return VectorizedCarlaEnv(server_urls=server_urls, **kwargs)

def start_carla_servers(
    num_servers: int = 4,
    **kwargs
) -> Tuple[CarlaServerManager, List[CarlaServer]]:
    """
    Start CARLA servers and return manager and server list.
    
    Args:
        num_servers: Number of servers to start
        **kwargs: Additional server manager arguments
        
    Returns:
        Tuple of (manager, servers)
    """
    return spawn_carla_servers(num_servers=num_servers, **kwargs)

__all__ = [
    # Core classes
    "CarlaEnv",
    "VectorizedCarlaEnv",
    # "ServiceManager",
    # "CarlaService", 
    # "ServiceConfig",
    "MicroserviceManager",  # The FIXED manager
    "Bench2DriveService",
    
    # Convenience functions
    "make_carla_env",
    "make_parallel_envs", 
    "start_carla_servers",
]

# Add optional exports if available
if CarlaServerManager:
    __all__.extend(["CarlaServerManager", "CarlaServer", "spawn_carla_servers"])
if APIAgent:
    __all__.extend(["APIAgent", "get_entry_point", "create_api_agent", "validate_action"])

__version__ = "1.0.0"