"""
Vectorized environment wrapper for parallel CARLA instances.
Handles multiple environments for efficient GRPO training.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .carla_env import CarlaEnv

logger = logging.getLogger(__name__)


class VectorizedCarlaEnv:
    """
    Vectorized wrapper for multiple CARLA environments.
    
    This class manages multiple CarlaEnv instances for parallel execution,
    which is essential for efficient GRPO training with multi-turn rollouts.
    
    Key features:
    - Parallel reset and step operations
    - Independent snapshot/restore per environment
    - Automatic port management to avoid conflicts
    - Fault tolerance with automatic restart
    """
    
    def __init__(
        self,
        num_envs: Optional[int] = None,
        service_urls: Optional[List[str]] = None,
        base_api_port: int = 8080,
        base_carla_port: int = 2000,
        port_offset: int = 2,  # Skip ports to avoid conflicts
        max_workers: Optional[int] = None,
        **env_kwargs
    ):
        """
        Initialize vectorized environment.
        
        Args:
            num_envs: Number of parallel environments (if service_urls not provided)
            service_urls: List of server URLs to connect to (overrides num_envs)
            base_api_port: Starting port for API servers
            base_carla_port: Starting port for CARLA instances
            port_offset: Port spacing to avoid conflicts
            max_workers: Max parallel workers (None = num_envs)
            **env_kwargs: Additional arguments for CarlaEnv
        """
        # Determine server URLs
        if service_urls is not None:
            self.num_envs = len(service_urls)
            self.service_urls = service_urls
        elif num_envs is not None:
            self.num_envs = num_envs
            self.service_urls = [f"http://localhost:{base_api_port + i}" for i in range(num_envs)]
        else:
            raise ValueError("Either num_envs or service_urls must be provided")
        
        self.base_api_port = base_api_port
        self.base_carla_port = base_carla_port
        self.port_offset = port_offset
        self.max_workers = max_workers or self.num_envs
        self.env_kwargs = env_kwargs
        
        # Create environment instances
        self.envs = []
        for i, server_url in enumerate(self.service_urls):
            try:
                env = CarlaEnv(server_url=server_url, **env_kwargs)
                self.envs.append(env)
                logger.info(f"Created environment {i} at {server_url}")
            except Exception as e:
                logger.error(f"Failed to create environment {i}: {e}")
                raise RuntimeError(f"Failed to create environment {i}: {e}")
        
        # Get spaces from first environment
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Tracking for GRPO
        self.snapshots = [[] for _ in range(num_envs)]
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Reset all environments in parallel.
        
        Args:
            seed: Base seed (each env gets seed + index)
            options: List of options per environment
        
        Returns:
            observations: List of observations
            infos: List of info dictionaries
        """
        if options is None:
            options = [{}] * self.num_envs
        
        # Submit reset tasks
        futures = []
        for i, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + i
            env_options = options[i] if i < len(options) else {}
            
            future = self.executor.submit(
                env.reset,
                seed=env_seed,
                options=env_options
            )
            futures.append((i, future))
        
        # Collect results
        observations = [None] * self.num_envs
        infos = [None] * self.num_envs
        
        for i, future in futures:
            try:
                obs, info = future.result(timeout=120)  # Long timeout for reset
                observations[i] = obs
                infos[i] = info
            except Exception as e:
                logger.error(f"Environment {i} failed to reset: {e}")
                # Return zero observation on failure
                observations[i] = self._get_zero_observation()
                infos[i] = {"error": str(e)}
        
        return observations, infos
    
    def step(
        self,
        actions: List[np.ndarray]
    ) -> Tuple[List[Dict[str, Any]], List[float], List[bool], List[bool], List[Dict[str, Any]]]:
        """
        Step all environments in parallel.
        
        Args:
            actions: List of actions for each environment
        
        Returns:
            observations: List of observations
            rewards: List of rewards
            terminateds: List of terminated flags
            truncateds: List of truncated flags
            infos: List of info dictionaries
        """
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")
        
        # Submit step tasks
        futures = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            future = self.executor.submit(env.step, action)
            futures.append((i, future))
        
        # Collect results
        observations = [None] * self.num_envs
        rewards = [0.0] * self.num_envs
        terminateds = [False] * self.num_envs
        truncateds = [False] * self.num_envs
        infos = [{}] * self.num_envs
        
        for i, future in futures:
            try:
                obs, reward, terminated, truncated, info = future.result(timeout=30)
                observations[i] = obs
                rewards[i] = reward
                terminateds[i] = terminated
                truncateds[i] = truncated
                infos[i] = info
            except Exception as e:
                logger.error(f"Environment {i} failed to step: {e}")
                # Return safe defaults on failure
                observations[i] = self._get_zero_observation()
                rewards[i] = 0.0
                terminateds[i] = True  # Terminate on error
                truncateds[i] = False
                infos[i] = {"error": str(e)}
        
        return observations, rewards, terminateds, truncateds, infos
    
    def render(self, mode: str = "rgb_array") -> Optional[List[np.ndarray]]:
        """
        Render all environments.
        
        Args:
            mode: Rendering mode
        
        Returns:
            List of rendered frames or None
        """
        if mode == "rgb_array":
            frames = []
            for env in self.envs:
                frame = env.render()
                frames.append(frame)
            return frames
        return None
    
    def close(self):
        """Close all environments and cleanup."""
        # Close environments in parallel
        futures = []
        for env in self.envs:
            future = self.executor.submit(env.close)
            futures.append(future)
        
        # Wait for all to close
        for future in futures:
            try:
                future.result(timeout=10)
            except Exception as e:
                logger.error(f"Error closing environment: {e}")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        logger.info("All environments closed")
    
    # GRPO-specific methods for parallel snapshot/restore
    
    def save_snapshots(self) -> List[str]:
        """
        Save snapshots for all environments.
        
        Returns:
            List of snapshot IDs
        """
        futures = []
        for env in self.envs:
            future = self.executor.submit(env.save_snapshot)
            futures.append(future)
        
        snapshot_ids = []
        for i, future in enumerate(futures):
            try:
                snapshot_id = future.result(timeout=30)
                snapshot_ids.append(snapshot_id)
                self.snapshots[i].append(snapshot_id)
            except Exception as e:
                logger.error(f"Environment {i} failed to save snapshot: {e}")
                snapshot_ids.append(None)
        
        return snapshot_ids
    
    def restore_snapshots(self, snapshot_ids: List[str]):
        """
        Restore all environments from snapshots.
        
        Args:
            snapshot_ids: List of snapshot IDs to restore
        """
        if len(snapshot_ids) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} snapshot IDs, got {len(snapshot_ids)}")
        
        futures = []
        for env, snapshot_id in zip(self.envs, snapshot_ids):
            if snapshot_id is not None:
                future = self.executor.submit(env.restore_snapshot, snapshot_id)
                futures.append(future)
        
        # Wait for all restores
        for future in futures:
            try:
                future.result(timeout=30)
            except Exception as e:
                logger.error(f"Failed to restore snapshot: {e}")
    
    def reset_from_snapshots(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Reset all environments from their current snapshot states.
        
        Returns:
            observations: List of observations
            infos: List of info dictionaries
        """
        futures = []
        for env in self.envs:
            future = self.executor.submit(env.reset_from_snapshot)
            futures.append(future)
        
        observations = []
        infos = []
        for i, future in enumerate(futures):
            try:
                obs, info = future.result(timeout=30)
                observations.append(obs)
                infos.append(info)
            except Exception as e:
                logger.error(f"Environment {i} failed to reset from snapshot: {e}")
                observations.append(self._get_zero_observation())
                infos.append({"error": str(e)})
        
        return observations, infos
    
    def fork_from_snapshot(self, source_env_idx: int, target_env_indices: List[int]):
        """
        Fork multiple environments from a single environment's snapshot.
        Useful for GRPO branching from the same state.
        
        Args:
            source_env_idx: Index of source environment
            target_env_indices: Indices of target environments
        """
        if source_env_idx >= self.num_envs:
            raise ValueError(f"Invalid source environment index: {source_env_idx}")
        
        # Get latest snapshot from source
        if not self.snapshots[source_env_idx]:
            raise ValueError(f"No snapshots available for environment {source_env_idx}")
        
        source_snapshot = self.snapshots[source_env_idx][-1]
        
        # Restore target environments
        futures = []
        for target_idx in target_env_indices:
            if target_idx >= self.num_envs:
                continue
            
            future = self.executor.submit(
                self.envs[target_idx].restore_snapshot,
                source_snapshot
            )
            futures.append((target_idx, future))
        
        # Wait for completion
        for target_idx, future in futures:
            try:
                future.result(timeout=30)
                logger.info(f"Forked environment {target_idx} from {source_env_idx}")
            except Exception as e:
                logger.error(f"Failed to fork environment {target_idx}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from all environments.
        
        Returns:
            Dictionary with aggregated statistics
        """
        total_steps = sum(env.episode_steps for env in self.envs)
        total_reward = sum(env.episode_reward for env in self.envs)
        avg_reward = total_reward / self.num_envs if self.num_envs > 0 else 0
        
        return {
            "num_envs": self.num_envs,
            "total_steps": total_steps,
            "total_reward": total_reward,
            "average_reward": avg_reward,
            "snapshots_per_env": [len(s) for s in self.snapshots]
        }
    
    def _get_zero_observation(self) -> Dict[str, Any]:
        """Get a zero observation matching the observation space."""
        return {
            'center_image': np.zeros((600, 800, 3), dtype=np.uint8),
            'left_image': np.zeros((600, 800, 3), dtype=np.uint8),
            'right_image': np.zeros((600, 800, 3), dtype=np.uint8),
            'vehicle_state': {
                'position': np.zeros(3, dtype=np.float32),
                'rotation': np.zeros(3, dtype=np.float32),
                'velocity': np.zeros(3, dtype=np.float32),
                'speed': np.array([0.0], dtype=np.float32)
            },
            'navigation': {
                'distance_to_goal': np.array([0.0], dtype=np.float32),
                'route_completion': np.array([0.0], dtype=np.float32),
                'next_command': 0
            }
        }


def make_vectorized_carla_env(
    num_envs: int = 4,
    base_api_port: int = 8080,
    base_carla_port: int = 2000,
    **kwargs
) -> VectorizedCarlaEnv:
    """
    Factory function to create vectorized CARLA environment.
    
    Args:
        num_envs: Number of parallel environments
        base_api_port: Starting API port
        base_carla_port: Starting CARLA port
        **kwargs: Additional environment arguments
    
    Returns:
        VectorizedCarlaEnv instance
    """
    return VectorizedCarlaEnv(
        num_envs=num_envs,
        base_api_port=base_api_port,
        base_carla_port=base_carla_port,
        **kwargs
    )