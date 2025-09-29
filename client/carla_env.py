"""
Gymnasium-compliant CARLA environment with strict API adherence.
Supports GRPO snapshot/restore for multi-turn rollouts.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import requests
import json
import base64
from PIL import Image
import io
import logging
import time

logger = logging.getLogger(__name__)


class CarlaEnv(gym.Env):
    """
    Gymnasium-compliant environment for CARLA with Bench2Drive.
    
    This environment strictly follows the Gymnasium API specification:
    https://gymnasium.farama.org/api/env/
    
    Key features:
    - Real CARLA instance communication via REST API
    - Snapshot/restore support for GRPO branching
    - Proper observation and action spaces
    - Full compliance with gym.Env interface
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }
    
    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        frame_skip: int = 1,
        reward_type: str = "dense",
        timeout: float = 360.0,  # 6 minutes for Town12 loading
    ):
        """
        Initialize CARLA environment.
        
        Args:
            server_url: URL of the CARLA REST API server
            render_mode: Rendering mode ("human", "rgb_array", or None)
            max_steps: Maximum steps per episode
            frame_skip: Number of simulation steps per action
            reward_type: Reward calculation type ("dense" or "sparse")
            timeout: Request timeout in seconds
        """
        super().__init__()
        
        self.server_url = server_url.rstrip('/')
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.reward_type = reward_type
        self.timeout = timeout
        
        # Episode tracking
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.last_observation = None
        
        # Snapshot management for GRPO
        self.snapshots = {}
        self.current_snapshot = None
        
        # Define action space: [throttle, brake, steer]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space (matching our server's camera resolution)
        self.observation_space = spaces.Dict({
            'center_image': spaces.Box(0, 255, shape=(512, 1024, 3), dtype=np.uint8),
            'left_image': spaces.Box(0, 255, shape=(512, 1024, 3), dtype=np.uint8),
            'right_image': spaces.Box(0, 255, shape=(512, 1024, 3), dtype=np.uint8),
            'vehicle_state': spaces.Dict({
                'position': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                'rotation': spaces.Box(-360, 360, shape=(3,), dtype=np.float32),
                'velocity': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                'speed': spaces.Box(0, np.inf, shape=(1,), dtype=np.float32)
            }),
            'scenario_info': spaces.Dict({
                'route_id': spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                'step_count': spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
                'max_steps': spaces.Box(0, np.inf, shape=(1,), dtype=np.float32)
            }),
            'navigation': spaces.Dict({
                'command': spaces.Discrete(6),  # Follow lane, turn left, turn right, etc.
                'distance_to_goal': spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
                'route_completion': spaces.Box(0, 1, shape=(1,), dtype=np.float32)
            })
        })
        
        # Verify server connection
        self._verify_server()
    
    def _verify_server(self):
        """Verify connection to CARLA server."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            response.raise_for_status()
            logger.info(f"Connected to CARLA server at {self.server_url}")
        except Exception as e:
            logger.error(f"Failed to connect to CARLA server: {e}")
            raise RuntimeError(f"Cannot connect to CARLA server at {self.server_url}")
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        This method follows the Gymnasium API specification exactly.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (route_id, weather, etc.)
        
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        # Call parent reset with seed
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.episode_steps = 0
        self.episode_reward = 0.0
        
        # Set seed if provided
        if seed is not None:
            self._set_seed(seed)
        
        # Prepare reset request
        reset_data = {}
        if options:
            reset_data.update(options)
        
        try:
            # Send reset request to server
            response = requests.post(
                f"{self.server_url}/reset",
                json=reset_data,
                timeout=self.timeout * 2  # Longer timeout for reset
            )
            response.raise_for_status()
            data = response.json()
            
            # Process observation
            observation = self._process_observation(data['observation'])
            self.last_observation = observation
            
            # Create info dictionary
            info = data.get('info', {})
            info['episode_steps'] = self.episode_steps
            info['episode_reward'] = self.episode_reward
            
            return observation, info
            
        except Exception as e:
            logger.error(f"Failed to reset environment: {e}")
            raise RuntimeError(f"Failed to reset CARLA environment: {e}")
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute action in environment.
        
        This method follows the Gymnasium API specification exactly.
        
        Args:
            action: Action to execute [throttle, brake, steer]
        
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode ended (success/failure)
            truncated: Whether episode was cut short (max steps)
            info: Additional information dictionary
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Convert action to dictionary format
        action_dict = {
            "throttle": float(action[0]),
            "brake": float(action[1]),
            "steer": float(action[2])
        }
        
        try:
            # Execute action with frame skip
            total_reward = 0.0
            terminated = False
            truncated = False
            info = {}
            
            for _ in range(self.frame_skip):
                # Send step request
                response = requests.post(
                    f"{self.server_url}/step",
                    json={"action": action_dict, "n_steps": 1},
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                # Accumulate reward
                total_reward += data['reward']
                
                # Check termination
                terminated = terminated or data['terminated']
                truncated = truncated or data['truncated']
                
                # Update info
                info.update(data['info'])
                
                if terminated or truncated:
                    break
            
            # Process final observation
            observation = self._process_observation(data['observation'])
            self.last_observation = observation
            
            # Update episode tracking
            self.episode_steps += 1
            self.episode_reward += total_reward
            
            # Check for truncation due to max steps
            if self.episode_steps >= self.max_steps:
                truncated = True
            
            # Add episode info
            info['episode_steps'] = self.episode_steps
            info['episode_reward'] = self.episode_reward
            
            return observation, total_reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Failed to step environment: {e}")
            raise RuntimeError(f"Failed to step CARLA environment: {e}")
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            Rendered frame as RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "rgb_array" and self.last_observation is not None:
            # Return center camera image
            return self.last_observation['center_image']
        elif self.render_mode == "human" and self.last_observation is not None:
            # Display using matplotlib (optional dependency)
            try:
                import matplotlib.pyplot as plt
                plt.imshow(self.last_observation['center_image'])
                plt.axis('off')
                plt.show(block=False)
                plt.pause(0.001)
            except ImportError:
                logger.warning("matplotlib not installed, cannot render in human mode")
        return None
    
    def close(self):
        """
        Close the environment and clean up resources.
        """
        try:
            # Send close request to server
            response = requests.post(
                f"{self.server_url}/close",
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info("Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
    
    # GRPO-specific methods for snapshot/restore
    
    def save_snapshot(self) -> str:
        """
        Save current environment state for GRPO branching.
        
        Returns:
            snapshot_id: Unique identifier for the snapshot
        """
        try:
            # Send empty JSON body as server expects SnapshotRequest
            response = requests.post(
                f"{self.server_url}/snapshot",
                json={},  # Empty dict will use default snapshot_id on server
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            snapshot_id = data['snapshot_id']
            self.snapshots[snapshot_id] = {
                'episode_steps': self.episode_steps,
                'episode_reward': self.episode_reward,
                'last_observation': self.last_observation
            }
            
            logger.info(f"Saved snapshot: {snapshot_id}")
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            raise RuntimeError(f"Failed to save snapshot: {e}")
    
    def restore_snapshot(self, snapshot_id: str):
        """
        Restore environment to a previously saved state.
        
        Args:
            snapshot_id: Identifier of the snapshot to restore
        """
        if snapshot_id not in self.snapshots:
            raise ValueError(f"Unknown snapshot: {snapshot_id}")
        
        try:
            # Restore server state
            response = requests.post(
                f"{self.server_url}/restore",
                json={"snapshot_id": snapshot_id},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Restore local state
            local_state = self.snapshots[snapshot_id]
            self.episode_steps = local_state['episode_steps']
            self.episode_reward = local_state['episode_reward']
            self.last_observation = local_state['last_observation']
            
            logger.info(f"Restored snapshot: {snapshot_id}")
            
        except Exception as e:
            logger.error(f"Failed to restore snapshot: {e}")
            raise RuntimeError(f"Failed to restore snapshot: {e}")
    
    def reset_from_snapshot(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset to the current snapshot state (for GRPO rollouts).
        
        Returns:
            observation: Observation from snapshot state
            info: Information dictionary
        """
        if self.last_observation is None:
            return self.reset()
        
        info = {
            'episode_steps': self.episode_steps,
            'episode_reward': self.episode_reward,
            'from_snapshot': True
        }
        
        return self.last_observation, info
    
    def list_snapshots(self) -> List[str]:
        """
        Get list of available snapshots.
        
        Returns:
            List of snapshot IDs
        """
        return list(self.snapshots.keys())
    
    def delete_snapshot(self, snapshot_id: str):
        """
        Delete a snapshot to free memory.
        
        Args:
            snapshot_id: Identifier of the snapshot to delete
        """
        if snapshot_id in self.snapshots:
            del self.snapshots[snapshot_id]
            
            # Also delete on server
            try:
                response = requests.delete(
                    f"{self.server_url}/snapshot/{snapshot_id}",
                    timeout=self.timeout
                )
                response.raise_for_status()
                logger.info(f"Deleted snapshot: {snapshot_id}")
            except Exception as e:
                logger.error(f"Failed to delete snapshot on server: {e}")
    
    # Helper methods
    
    def _set_seed(self, seed: int):
        """Set random seed on server."""
        try:
            response = requests.post(
                f"{self.server_url}/seed",
                json={"seed": seed},
                timeout=self.timeout
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to set seed: {e}")
    
    def _process_observation(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw observation from server into gymnasium format."""
        obs = {
            'center_image': np.zeros((512, 1024, 3), dtype=np.uint8),
            'left_image': np.zeros((512, 1024, 3), dtype=np.uint8),
            'right_image': np.zeros((512, 1024, 3), dtype=np.uint8),
            'vehicle_state': {
                'position': np.zeros(3, dtype=np.float32),
                'rotation': np.zeros(3, dtype=np.float32),
                'velocity': np.zeros(3, dtype=np.float32),
                'speed': np.array([0.0], dtype=np.float32)
            },
            'scenario_info': {
                'route_id': np.array([0.0], dtype=np.float32),
                'step_count': np.array([0.0], dtype=np.float32),
                'max_steps': np.array([1000.0], dtype=np.float32)
            },
            'navigation': {
                'command': 0,  # Follow lane by default
                'distance_to_goal': np.array([100.0], dtype=np.float32),
                'route_completion': np.array([0.0], dtype=np.float32)
            }
        }
        
        # Decode images if present
        if 'images' in raw_obs:
            for cam_name, img_data in raw_obs['images'].items():
                if img_data:
                    img_array = self._decode_image(img_data)
                    if 'center' in cam_name.lower():
                        obs['center_image'] = img_array
                    elif 'left' in cam_name.lower():
                        obs['left_image'] = img_array
                    elif 'right' in cam_name.lower():
                        obs['right_image'] = img_array
        
        # Process vehicle state
        if 'vehicle_state' in raw_obs:
            state = raw_obs['vehicle_state']
            if 'position' in state:
                obs['vehicle_state']['position'] = np.array([
                    state['position'].get('x', 0),
                    state['position'].get('y', 0),
                    state['position'].get('z', 0)
                ], dtype=np.float32)
            if 'rotation' in state:
                obs['vehicle_state']['rotation'] = np.array([
                    state['rotation'].get('pitch', 0),
                    state['rotation'].get('yaw', 0),
                    state['rotation'].get('roll', 0)
                ], dtype=np.float32)
            if 'velocity' in state:
                obs['vehicle_state']['velocity'] = np.array([
                    state['velocity'].get('x', 0),
                    state['velocity'].get('y', 0),
                    state['velocity'].get('z', 0)
                ], dtype=np.float32)
            if 'speed' in state:
                obs['vehicle_state']['speed'] = np.array([state['speed']], dtype=np.float32)
        
        # Process scenario info
        if 'scenario_info' in raw_obs:
            info = raw_obs['scenario_info']
            if 'route_id' in info:
                obs['scenario_info']['route_id'] = np.array([float(info['route_id'])], dtype=np.float32)
            if 'step_count' in info:
                obs['scenario_info']['step_count'] = np.array([float(info['step_count'])], dtype=np.float32)
            if 'max_steps' in info:
                obs['scenario_info']['max_steps'] = np.array([float(info['max_steps'])], dtype=np.float32)
        
        # Process navigation info
        if 'navigation' in raw_obs:
            nav = raw_obs['navigation']
            if 'command' in nav:
                obs['navigation']['command'] = int(nav['command'])
            if 'distance_to_goal' in nav:
                obs['navigation']['distance_to_goal'] = np.array([float(nav['distance_to_goal'])], dtype=np.float32)
            if 'route_completion' in nav:
                obs['navigation']['route_completion'] = np.array([float(nav['route_completion'])], dtype=np.float32)
        
        return obs
    
    def _decode_image(self, base64_str: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        try:
            img_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_data))
            return np.array(img)
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return np.zeros((512, 1024, 3), dtype=np.uint8)