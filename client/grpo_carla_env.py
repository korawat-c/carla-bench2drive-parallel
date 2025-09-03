"""
GRPO-optimized CARLA environment with dynamic branching support.

This environment operates in two modes:
1. Single mode: Uses one CARLA instance for normal exploration
2. Branching mode: Dynamically activates multiple instances for GRPO rollouts

Key design:
- num_services defines maximum branching capacity (not always-on instances)
- Snapshot/restore only happens at branching points
- After branching, continue with best trajectory (no reload needed)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, Future
import time
from enum import Enum
from dataclasses import dataclass

from .carla_env import CarlaEnv

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Environment execution modes"""
    SINGLE = "single"      # Single CARLA instance
    BRANCHING = "branching" # Multiple parallel instances


class EnvStatus(Enum):
    """Environment operation status"""
    READY = "ready"                    # Ready for next action
    BUSY = "busy"                      # Processing current request
    BRANCHING_SETUP = "branching_setup" # Setting up branching instances
    BRANCHING_READY = "branching_ready" # Branching setup complete
    ERROR = "error"                    # Error occurred
    TERMINATED = "terminated"          # Episode terminated
    WAITING_RETRY = "waiting_retry"    # Waiting for retry


@dataclass
class StatusInfo:
    """Status information for environment operations"""
    status: EnvStatus
    message: str
    ready: bool = True  # Whether env is ready for next action
    retry_after: Optional[float] = None  # Seconds to wait before retry
    progress: Optional[float] = None  # Progress percentage for long operations
    details: Optional[Dict[str, Any]] = None  # Additional details


class GRPOCarlaEnv:
    """
    GRPO-optimized environment with dynamic branching support.
    
    This is NOT a standard gym.Env - it's a specialized wrapper that can switch between
    single and parallel execution modes for GRPO training.
    
    Key Design:
    - Does NOT inherit from gym.Env (violates single env assumption)
    - Explicitly handles mode switching with clear interfaces
    - step() behavior changes based on current mode (check with is_branching property)
    
    Example workflow:
        # Initialize with max branching capacity
        env = GRPOCarlaEnv(num_services=4)
        
        # Phase 1: Normal exploration (single instance)
        obs, _ = env.reset()
        for step in range(50):
            action = policy(obs)
            # Single mode ONLY - will error if in branching mode
            obs, reward, terminated, truncated, info = env.single_step(action)
        
        # Phase 2: Save state and enable branching
        snapshot_id = env.save_snapshot()
        env.enable_branching(snapshot_id, num_branches=4)
        # Now env.is_branching == True, single_step() will raise error
        
        # Phase 3: Parallel exploration (4 instances, critical scenarios)
        for step in range(10):  # Short exploration for critical scenarios
            # Branching mode ONLY - will error if in single mode
            actions = [policy(obs_i, noise=i*0.2) for i, obs_i in enumerate(observations)]
            observations, rewards, terminateds, truncateds, infos = env.branch_step(actions)
        
        # Phase 4: External evaluation and selection
        # Use your own reward function (e.g., distance to waypoint, lane center)
        scores = evaluate_trajectories(observations, infos)  # Your scoring function
        best_idx = np.argmax(scores)
        
        # Phase 5: Select best and continue (no reload needed)
        env.select_branch(best_idx)
        # Now env.is_branching == False, branch_step() will raise error
        
        # Continue with best trajectory in single mode
        action = policy(observations[best_idx])
        obs, reward, terminated, truncated, info = env.single_step(action)
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }
    
    def __init__(
        self,
        num_services: int = 4,  # Maximum branching capacity
        service_urls: Optional[List[str]] = None,
        base_api_port: int = 8080,
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        timeout: float = 360.0,
        **env_kwargs
    ):
        """
        Initialize GRPO environment with dynamic branching support.
        
        Args:
            num_services: Maximum number of parallel branches (branching capacity)
            service_urls: Optional list of service URLs (overrides num_services)
            base_api_port: Starting API port for services
            render_mode: Rendering mode
            max_steps: Maximum steps per episode
            timeout: Request timeout
            **env_kwargs: Additional CarlaEnv arguments
        """
        # Note: We don't call super().__init__() because we're not inheriting from gym.Env
        
        # Configuration
        self.max_branches = num_services
        self.base_api_port = base_api_port
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.timeout = timeout
        self.env_kwargs = env_kwargs
        
        # Service URLs
        if service_urls:
            self.service_urls = service_urls
            self.max_branches = len(service_urls)
        else:
            self.service_urls = [
                f"http://localhost:{base_api_port + i}" 
                for i in range(num_services)
            ]
        
        # Current execution mode
        self.mode = ExecutionMode.SINGLE
        self.active_branches = 1
        
        # Environment instances (lazy initialization)
        self.envs = [None] * self.max_branches
        self.primary_env_idx = 0  # Index of primary environment in single mode
        
        # Create primary environment
        self._create_env(0)
        
        # Define action/observation spaces from primary env
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=self.max_branches)
        
        # Episode tracking
        self.episode_steps = 0
        self.episode_rewards = [0.0] * self.max_branches  # Track per branch
        self.cumulative_rewards = [0.0] * self.max_branches  # For GRPO selection
        
        # Branching state
        self.current_snapshot = None
        self.branch_start_step = None
        
        # Async operation tracking
        self.branching_setup_future = None
        self.branching_setup_progress = 0.0
        
    @property
    def is_branching(self) -> bool:
        """Check if currently in branching mode."""
        return self.mode == ExecutionMode.BRANCHING
    
    @property
    def current_mode(self) -> str:
        """Get current execution mode as string."""
        return self.mode.value
    
    def _create_env(self, idx: int):
        """Create environment instance at given index."""
        if self.envs[idx] is None:
            try:
                self.envs[idx] = CarlaEnv(
                    server_url=self.service_urls[idx],
                    render_mode=None,  # Handle rendering at wrapper level
                    max_steps=self.max_steps,
                    timeout=self.timeout,
                    **self.env_kwargs
                )
                logger.info(f"Created environment {idx} at {self.service_urls[idx]}")
            except Exception as e:
                logger.error(f"Failed to create environment {idx}: {e}")
                raise
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset environment to initial state.
        
        Always returns single observation (resets to single mode).
        To branch after reset, use save_snapshot() and enable_branching().
        """
        # Note: No super().reset() since we don't inherit from gym.Env
        
        # Reset tracking
        self.episode_steps = 0
        self.episode_rewards = [0.0] * self.max_branches
        self.cumulative_rewards = [0.0] * self.max_branches
        self.current_snapshot = None
        self.branch_start_step = None
        
        # Reset to single mode
        self.mode = ExecutionMode.SINGLE
        self.active_branches = 1
        
        # Reset primary environment
        obs, info = self.envs[self.primary_env_idx].reset(seed=seed, options=options)
        
        return obs, info
    
    def single_step(
        self, 
        action: np.ndarray
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute single action in single mode.
        
        This method ONLY works in single mode (env.is_branching == False).
        Will raise error if called in branching mode.
        
        Args:
            action: Single action array [throttle, brake, steer]
            
        Returns:
            observation: Single observation dict
            reward: Reward value
            terminated: Whether episode ended
            truncated: Whether episode was cut short
            info: Additional information with 'status' field
            
        Raises:
            RuntimeError: If called in branching mode
        """
        if self.mode == ExecutionMode.BRANCHING:
            # Return error status instead of raising
            error_info = {
                'status': StatusInfo(
                    status=EnvStatus.ERROR,
                    message=f"single_step() called in branching mode! Use branch_step() instead.",
                    ready=False,
                    details={'current_mode': self.current_mode, 'is_branching': self.is_branching}
                )
            }
            raise RuntimeError(
                f"single_step() can only be called in single mode! "
                f"Current mode: {self.current_mode}, is_branching: {self.is_branching}. "
                f"Use branch_step() for branching mode."
            )
        
        self.episode_steps += 1
        obs, reward, terminated, truncated, info = self._step_single(action)
        
        # Add status to info
        if 'status' not in info:
            if terminated:
                info['status'] = StatusInfo(
                    status=EnvStatus.TERMINATED,
                    message="Episode terminated",
                    ready=False
                )
            else:
                info['status'] = StatusInfo(
                    status=EnvStatus.READY,
                    message="Step completed successfully",
                    ready=True
                )
        
        return obs, reward, terminated, truncated, info
    
    def branch_step(
        self,
        actions: List[np.ndarray]
    ) -> Tuple[List[Dict], List[float], List[bool], List[bool], List[Dict]]:
        """
        Execute parallel actions in branching mode.
        
        This method ONLY works in branching mode (env.is_branching == True).
        Will raise error if called in single mode.
        
        Args:
            actions: List of action arrays, one per active branch
            
        Returns:
            observations: List of observation dicts
            rewards: List of rewards
            terminateds: List of terminated flags
            truncateds: List of truncated flags
            infos: List of info dicts with 'status' field
            
        Raises:
            RuntimeError: If called in single mode
            ValueError: If wrong number of actions provided
        """
        if self.mode == ExecutionMode.SINGLE:
            raise RuntimeError(
                f"branch_step() can only be called in branching mode! "
                f"Current mode: {self.current_mode}, is_branching: {self.is_branching}. "
                f"Use single_step() for single mode or call enable_branching() first."
            )
        
        if len(actions) != self.active_branches:
            raise ValueError(
                f"branch_step() expects {self.active_branches} actions, got {len(actions)}"
            )
        
        self.episode_steps += 1
        observations, rewards, terminateds, truncateds, infos = self._step_branching(actions)
        
        # Add status to each info
        for i in range(len(infos)):
            if 'status' not in infos[i]:
                if terminateds[i]:
                    infos[i]['status'] = StatusInfo(
                        status=EnvStatus.TERMINATED,
                        message=f"Branch {i} terminated",
                        ready=False,
                        details={'branch_id': i}
                    )
                else:
                    infos[i]['status'] = StatusInfo(
                        status=EnvStatus.READY,
                        message=f"Branch {i} step completed",
                        ready=True,
                        details={'branch_id': i}
                    )
        
        return observations, rewards, terminateds, truncateds, infos
    
    def step(
        self,
        action: Union[np.ndarray, List[np.ndarray]]
    ) -> Tuple[Union[Dict, List[Dict]], Union[float, List[float]], 
               Union[bool, List[bool]], Union[bool, List[bool]], 
               Union[Dict, List[Dict]]]:
        """
        Convenience method that delegates to single_step or branch_step based on mode.
        
        NOT RECOMMENDED: Use single_step() or branch_step() directly for clarity.
        This method exists for backward compatibility only.
        """
        if self.mode == ExecutionMode.SINGLE:
            if isinstance(action, list):
                raise ValueError(f"Single mode expects single action, got list")
            return self.single_step(action)
        else:
            if not isinstance(action, list):
                raise ValueError(f"Branching mode expects list of actions")
            return self.branch_step(action)
    
    def _step_single(self, action: np.ndarray):
        """Step in single mode (one instance)."""
        # Execute on primary environment
        obs, reward, terminated, truncated, info = self.envs[self.primary_env_idx].step(action)
        
        # Update tracking
        self.episode_rewards[self.primary_env_idx] += reward
        self.cumulative_rewards[self.primary_env_idx] += reward
        
        # Check max steps
        if self.episode_steps >= self.max_steps:
            truncated = True
        
        info['episode_steps'] = self.episode_steps
        info['mode'] = 'single'
        
        return obs, reward, terminated, truncated, info
    
    def _step_branching(self, actions: List[np.ndarray]):
        """Step in branching mode (parallel instances)."""
        if not isinstance(actions, list):
            raise ValueError(f"Branching mode requires list of actions, got {type(actions)}")
        
        if len(actions) != self.active_branches:
            raise ValueError(f"Expected {self.active_branches} actions, got {len(actions)}")
        
        # Submit parallel step tasks
        futures = []
        for i in range(self.active_branches):
            if self.envs[i] is not None:
                future = self.executor.submit(self.envs[i].step, actions[i])
                futures.append((i, future))
        
        # Collect results
        observations = [None] * self.active_branches
        rewards = [0.0] * self.active_branches
        terminateds = [False] * self.active_branches
        truncateds = [False] * self.active_branches
        infos = [{}] * self.active_branches
        
        for i, future in futures:
            try:
                obs, reward, terminated, truncated, info = future.result(timeout=self.timeout)
                observations[i] = obs
                rewards[i] = reward
                terminateds[i] = terminated
                truncateds[i] = truncated
                infos[i] = info
                
                # Update tracking
                self.episode_rewards[i] += reward
                self.cumulative_rewards[i] += reward
                
            except Exception as e:
                logger.error(f"Branch {i} failed to step: {e}")
                terminateds[i] = True
                infos[i] = {"error": str(e)}
        
        # Check max steps
        if self.episode_steps >= self.max_steps:
            truncateds = [True] * self.active_branches
        
        # Add episode info
        for i in range(self.active_branches):
            infos[i]['episode_steps'] = self.episode_steps
            infos[i]['branch_id'] = i
            infos[i]['cumulative_reward'] = self.cumulative_rewards[i]
            infos[i]['mode'] = 'branching'
        
        return observations, rewards, terminateds, truncateds, infos
    
    def save_snapshot(self) -> str:
        """
        Save current state for branching.
        
        ONLY works in single mode. Must be called before enable_branching().
        
        Returns:
            snapshot_id: Unique identifier for the snapshot
            
        Raises:
            RuntimeError: If called in branching mode
        """
        if self.mode != ExecutionMode.SINGLE:
            raise RuntimeError(
                f"save_snapshot() can only be called in single mode! "
                f"Current mode: {self.current_mode}, is_branching: {self.is_branching}. "
                f"Call select_branch() to return to single mode first."
            )
        
        # Save on primary environment
        snapshot_id = self.envs[self.primary_env_idx].save_snapshot()
        self.current_snapshot = snapshot_id
        self.branch_start_step = self.episode_steps
        
        logger.info(f"Saved snapshot '{snapshot_id}' at step {self.episode_steps}")
        return snapshot_id
    
    def enable_branching(
        self, 
        snapshot_id: Optional[str] = None,
        num_branches: int = None,
        async_setup: bool = False
    ) -> StatusInfo:
        """
        Switch from single to branching mode.
        Loads snapshot into multiple instances for parallel exploration.
        
        ONLY works in single mode. After calling this:
        - single_step() will raise error
        - branch_step() becomes available
        - env.is_branching == True
        
        Args:
            snapshot_id: Snapshot to branch from (uses current if None)
            num_branches: Number of branches to activate (max: num_services)
            async_setup: If True, return immediately and setup in background
            
        Returns:
            StatusInfo indicating setup status
            
        Raises:
            RuntimeError: If already in branching mode
            ValueError: If no snapshot available
        """
        if self.mode == ExecutionMode.BRANCHING:
            raise RuntimeError(
                f"Already in branching mode! "
                f"Current mode: {self.current_mode}, active_branches: {self.active_branches}. "
                f"Call select_branch() to return to single mode before branching again."
            )
        
        # Use current snapshot if not specified
        if snapshot_id is None:
            snapshot_id = self.current_snapshot
        if snapshot_id is None:
            raise ValueError("No snapshot available for branching")
        
        # Determine number of branches
        if num_branches is None:
            num_branches = self.max_branches
        num_branches = min(num_branches, self.max_branches)
        
        logger.info(f"Enabling branching with {num_branches} instances from snapshot '{snapshot_id}'")
        
        if async_setup:
            # Start async setup
            self.branching_setup_progress = 0.0
            self.branching_setup_future = self.executor.submit(
                self._setup_branching_async, snapshot_id, num_branches
            )
            
            return StatusInfo(
                status=EnvStatus.BRANCHING_SETUP,
                message=f"Setting up {num_branches} branching instances in background",
                ready=False,
                retry_after=2.0,
                progress=0.0,
                details={
                    'num_branches': num_branches,
                    'snapshot_id': snapshot_id,
                    'async': True
                }
            )
        else:
            # Synchronous setup
            try:
                # Create additional environments if needed
                for i in range(num_branches):
                    if self.envs[i] is None:
                        self._create_env(i)
                        self.branching_setup_progress = (i + 1) / num_branches * 0.5
                
                # Load snapshot into all branches (parallel)
                futures = []
                for i in range(num_branches):
                    if i != self.primary_env_idx:  # Primary already has the state
                        future = self.executor.submit(
                            self.envs[i].restore_snapshot, 
                            snapshot_id
                        )
                        futures.append((i, future))
                
                # Wait for all restores
                completed = 0
                for i, future in futures:
                    try:
                        future.result(timeout=self.timeout)
                        logger.info(f"Branch {i} restored from snapshot")
                        completed += 1
                        self.branching_setup_progress = 0.5 + (completed / len(futures) * 0.5)
                    except Exception as e:
                        logger.error(f"Failed to restore branch {i}: {e}")
                        return StatusInfo(
                            status=EnvStatus.ERROR,
                            message=f"Failed to setup branch {i}: {str(e)}",
                            ready=False,
                            details={'failed_branch': i, 'error': str(e)}
                        )
                
                # Switch mode
                self.mode = ExecutionMode.BRANCHING
                self.active_branches = num_branches
                
                # Reset cumulative rewards for fair comparison
                for i in range(num_branches):
                    self.cumulative_rewards[i] = self.episode_rewards[self.primary_env_idx]
                
                logger.info(f"Branching enabled with {num_branches} active instances")
                
                return StatusInfo(
                    status=EnvStatus.BRANCHING_READY,
                    message=f"Successfully enabled {num_branches} branches",
                    ready=True,
                    progress=1.0,
                    details={'num_branches': num_branches, 'snapshot_id': snapshot_id}
                )
                
            except Exception as e:
                logger.error(f"Failed to enable branching: {e}")
                return StatusInfo(
                    status=EnvStatus.ERROR,
                    message=f"Failed to enable branching: {str(e)}",
                    ready=False,
                    details={'error': str(e)}
                )
    
    def select_branch(self, branch_idx: int):
        """
        Select best branch and return to single mode.
        No snapshot/restore needed - just continue with selected branch.
        
        ONLY works in branching mode. After calling this:
        - branch_step() will raise error
        - single_step() becomes available
        - env.is_branching == False
        
        Args:
            branch_idx: Index of branch to continue with (0 to active_branches-1)
            
        Raises:
            RuntimeError: If not in branching mode
            ValueError: If invalid branch index
        """
        if self.mode != ExecutionMode.BRANCHING:
            raise RuntimeError(
                f"select_branch() can only be called in branching mode! "
                f"Current mode: {self.current_mode}, is_branching: {self.is_branching}. "
                f"Call enable_branching() first to enter branching mode."
            )
        
        if branch_idx >= self.active_branches:
            raise ValueError(f"Invalid branch index: {branch_idx}")
        
        logger.info(f"Selecting branch {branch_idx} with cumulative reward: "
                   f"{self.cumulative_rewards[branch_idx]:.2f}")
        
        # Switch primary environment to selected branch
        self.primary_env_idx = branch_idx
        
        # Keep only the selected branch's reward history
        selected_reward = self.episode_rewards[branch_idx]
        self.episode_rewards = [0.0] * self.max_branches
        self.episode_rewards[branch_idx] = selected_reward
        
        # Switch back to single mode
        self.mode = ExecutionMode.SINGLE
        self.active_branches = 1
        
        logger.info(f"Returned to single mode with branch {branch_idx}")
    
    def _setup_branching_async(self, snapshot_id: str, num_branches: int):
        """Async helper to setup branching in background."""
        try:
            # Create additional environments if needed
            for i in range(num_branches):
                if self.envs[i] is None:
                    self._create_env(i)
                    self.branching_setup_progress = (i + 1) / num_branches * 0.5
            
            # Load snapshot into all branches (parallel)
            futures = []
            for i in range(num_branches):
                if i != self.primary_env_idx:  # Primary already has the state
                    future = self.executor.submit(
                        self.envs[i].restore_snapshot, 
                        snapshot_id
                    )
                    futures.append((i, future))
            
            # Wait for all restores
            completed = 0
            for i, future in futures:
                future.result(timeout=self.timeout)
                completed += 1
                self.branching_setup_progress = 0.5 + (completed / len(futures) * 0.5)
            
            # Switch mode
            self.mode = ExecutionMode.BRANCHING
            self.active_branches = num_branches
            
            # Reset cumulative rewards
            for i in range(num_branches):
                self.cumulative_rewards[i] = self.episode_rewards[self.primary_env_idx]
            
            self.branching_setup_progress = 1.0
            logger.info(f"Async branching setup complete with {num_branches} instances")
            
        except Exception as e:
            logger.error(f"Async branching setup failed: {e}")
            self.branching_setup_progress = -1.0  # Indicates error
    
    def check_branching_status(self) -> StatusInfo:
        """
        Check status of async branching setup.
        
        Returns:
            StatusInfo with current setup status
        """
        if self.branching_setup_future is None:
            if self.mode == ExecutionMode.BRANCHING:
                return StatusInfo(
                    status=EnvStatus.BRANCHING_READY,
                    message="Branching mode active",
                    ready=True,
                    details={'active_branches': self.active_branches}
                )
            else:
                return StatusInfo(
                    status=EnvStatus.READY,
                    message="Single mode active",
                    ready=True,
                    details={'mode': self.current_mode}
                )
        
        # Check async setup status
        if self.branching_setup_future.done():
            if self.branching_setup_progress < 0:
                # Error occurred
                self.branching_setup_future = None
                return StatusInfo(
                    status=EnvStatus.ERROR,
                    message="Branching setup failed",
                    ready=False,
                    details={'progress': self.branching_setup_progress}
                )
            elif self.branching_setup_progress >= 1.0:
                # Success
                self.branching_setup_future = None
                return StatusInfo(
                    status=EnvStatus.BRANCHING_READY,
                    message="Branching setup complete",
                    ready=True,
                    progress=1.0,
                    details={'active_branches': self.active_branches}
                )
        
        # Still in progress
        return StatusInfo(
            status=EnvStatus.BRANCHING_SETUP,
            message=f"Setting up branches... {int(self.branching_setup_progress * 100)}%",
            ready=False,
            retry_after=1.0,
            progress=self.branching_setup_progress
        )
    
    def disable_branching(self):
        """
        Disable branching and return to single mode.
        Keeps the primary environment active.
        """
        if self.mode == ExecutionMode.SINGLE:
            return
        
        self.mode = ExecutionMode.SINGLE
        self.active_branches = 1
        logger.info("Branching disabled, returned to single mode")
    
    def get_branch_statistics(self) -> Dict[str, Any]:
        """Get statistics for all branches (useful for GRPO)."""
        if self.mode != ExecutionMode.BRANCHING:
            return {
                "mode": "single",
                "active_branches": 1,
                "primary_reward": self.episode_rewards[self.primary_env_idx]
            }
        
        return {
            "mode": "branching",
            "active_branches": self.active_branches,
            "branch_rewards": self.episode_rewards[:self.active_branches],
            "cumulative_rewards": self.cumulative_rewards[:self.active_branches],
            "best_branch": int(np.argmax(self.cumulative_rewards[:self.active_branches])),
            "steps_since_branch": self.episode_steps - (self.branch_start_step or 0)
        }
    
    def render(self) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
        """Render current frame(s)."""
        if self.render_mode == "rgb_array":
            if self.mode == ExecutionMode.SINGLE:
                return self.envs[self.primary_env_idx].render()
            else:
                # Return frames from all active branches
                frames = []
                for i in range(self.active_branches):
                    if self.envs[i]:
                        frames.append(self.envs[i].render())
                return frames
        return None
    
    def close(self):
        """Close all environments and cleanup."""
        # Close all created environments
        futures = []
        for env in self.envs:
            if env is not None:
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
        logger.info("GRPO environment closed")