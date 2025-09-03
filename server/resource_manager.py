#!/usr/bin/env python3
"""
Resource Manager for Bench2Drive Microservices
Handles GPU assignment and port allocation for multiple CARLA instances
"""

import os
import yaml
import socket
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ServiceResources:
    """Resources allocated to a service"""
    service_id: int
    gpu_id: int
    api_port: int
    carla_port: int
    streaming_port: int
    tm_port: int  # Traffic Manager port
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'service_id': self.service_id,
            'gpu_id': self.gpu_id,
            'api_port': self.api_port,
            'carla_port': self.carla_port,
            'streaming_port': self.streaming_port,
            'tm_port': self.tm_port
        }


class ResourceManager:
    """Manages GPU and port allocation for services"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize resource manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.allocated_resources: Dict[int, ServiceResources] = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'gpu': {
                'strategy': 'same',
                'default_gpu': 0,
                'available_gpus': [0, 1]
            },
            'ports': {
                'api_base': 8080,
                'carla_base': 2000,
                'streaming_base': 3000,
                'api_offset': 1,
                'carla_offset': 4,
                'streaming_offset': 10
            },
            'carla': {
                'tm_default_port_offset': 1000
            }
        }
    
    def allocate_resources(self, service_id: int) -> ServiceResources:
        """
        Allocate resources for a service
        
        Args:
            service_id: Service identifier (0-based)
            
        Returns:
            ServiceResources object with allocated resources
        """
        # Check if already allocated
        if service_id in self.allocated_resources:
            return self.allocated_resources[service_id]
        
        # Allocate GPU
        gpu_id = self._allocate_gpu(service_id)
        
        # Allocate ports
        api_port = self._allocate_api_port(service_id)
        carla_port = self._allocate_carla_port(service_id)
        streaming_port = self._allocate_streaming_port(service_id)
        tm_port = carla_port + self.config['carla']['tm_default_port_offset']
        
        # Create resource object
        resources = ServiceResources(
            service_id=service_id,
            gpu_id=gpu_id,
            api_port=api_port,
            carla_port=carla_port,
            streaming_port=streaming_port,
            tm_port=tm_port
        )
        
        # Store allocation
        self.allocated_resources[service_id] = resources
        
        logger.info(f"Allocated resources for service {service_id}:")
        logger.info(f"  GPU: {gpu_id}")
        logger.info(f"  API Port: {api_port}")
        logger.info(f"  CARLA Port: {carla_port}")
        logger.info(f"  Streaming Port: {streaming_port}")
        logger.info(f"  TM Port: {tm_port}")
        
        return resources
    
    def _allocate_gpu(self, service_id: int) -> int:
        """Allocate GPU based on strategy"""
        strategy = self.config['gpu']['strategy']
        
        if strategy == 'same':
            # All services on same GPU
            return self.config['gpu']['default_gpu']
        
        elif strategy == 'distributed':
            # Round-robin across available GPUs
            available_gpus = self.config['gpu']['available_gpus']
            return available_gpus[service_id % len(available_gpus)]
        
        else:
            raise ValueError(f"Unknown GPU strategy: {strategy}")
    
    def _allocate_api_port(self, service_id: int) -> int:
        """Allocate API port for service"""
        base = self.config['ports']['api_base']
        offset = self.config['ports']['api_offset']
        return base + (service_id * offset)
    
    def _allocate_carla_port(self, service_id: int) -> int:
        """Allocate CARLA RPC port for service"""
        base = self.config['ports']['carla_base']
        offset = self.config['ports']['carla_offset']
        return base + (service_id * offset)
    
    def _allocate_streaming_port(self, service_id: int) -> int:
        """Allocate streaming port for service"""
        base = self.config['ports']['streaming_base']
        offset = self.config['ports']['streaming_offset']
        return base + (service_id * offset)
    
    def deallocate_resources(self, service_id: int):
        """Release resources for a service"""
        if service_id in self.allocated_resources:
            del self.allocated_resources[service_id]
            logger.info(f"Deallocated resources for service {service_id}")
    
    def is_port_free(self, port: int) -> bool:
        """Check if a port is free"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('', port))
            sock.close()
            return True
        except OSError:
            return False
    
    def check_port_availability(self, service_id: int) -> Tuple[bool, List[str]]:
        """
        Check if all ports for a service are available
        
        Returns:
            Tuple of (all_available, list_of_unavailable_ports)
        """
        resources = self.allocate_resources(service_id)
        unavailable = []
        
        ports_to_check = [
            ('API', resources.api_port),
            ('CARLA', resources.carla_port),
            ('CARLA+1', resources.carla_port + 1),
            ('CARLA+2', resources.carla_port + 2),
            ('Streaming', resources.streaming_port),
            ('TM', resources.tm_port)
        ]
        
        for name, port in ports_to_check:
            if not self.is_port_free(port):
                unavailable.append(f"{name}:{port}")
        
        return len(unavailable) == 0, unavailable
    
    def get_carla_command_args(self, service_id: int) -> List[str]:
        """
        Get CARLA command line arguments for a service
        
        Returns:
            List of command line arguments
        """
        resources = self.allocate_resources(service_id)
        
        args = [
            '-RenderOffScreen',
            '-nosound',
            '-quality-level=Epic',
            f'-carla-rpc-port={resources.carla_port}',
            f'-carla-streaming-port={resources.streaming_port}',
            f'-graphicsadapter={resources.gpu_id}'
        ]
        
        return args
    
    def get_summary(self) -> str:
        """Get summary of allocated resources"""
        if not self.allocated_resources:
            return "No resources allocated"
        
        lines = ["Resource Allocation Summary:"]
        lines.append("-" * 50)
        
        for sid, resources in sorted(self.allocated_resources.items()):
            lines.append(f"Service {sid}:")
            lines.append(f"  GPU: {resources.gpu_id}")
            lines.append(f"  Ports: API={resources.api_port}, "
                        f"CARLA={resources.carla_port}, "
                        f"Stream={resources.streaming_port}, "
                        f"TM={resources.tm_port}")
        
        return "\n".join(lines)


# Singleton instance
_resource_manager = None


def get_resource_manager(config_path: Optional[str] = None) -> ResourceManager:
    """Get or create resource manager singleton"""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager(config_path)
    return _resource_manager


if __name__ == "__main__":
    # Test the resource manager
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    manager = ResourceManager(config_path)
    
    # Test allocation for 4 services
    print("Testing resource allocation for 4 services:")
    print("=" * 50)
    
    for i in range(4):
        resources = manager.allocate_resources(i)
        available, unavailable = manager.check_port_availability(i)
        
        print(f"\nService {i}:")
        print(f"  Resources: {resources.to_dict()}")
        print(f"  Ports available: {available}")
        if not available:
            print(f"  Unavailable ports: {unavailable}")
        
        # Get CARLA command
        cmd_args = manager.get_carla_command_args(i)
        print(f"  CARLA args: {' '.join(cmd_args)}")
    
    print("\n" + "=" * 50)
    print(manager.get_summary())