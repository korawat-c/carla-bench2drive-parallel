#!/usr/bin/env python3
"""
API Agent - A minimal agent that bridges between the server and REST API.

This agent doesn't make decisions itself, it just receives actions from the REST API
and provides sensor data back to the server for observations.

Based on the old-bench2drive-carla implementation but integrated into gymnasium_carla.
"""

import carla
import numpy as np
from queue import Queue, Empty
import json
from pathlib import Path
import sys
import os
import threading

# Add Bench2Drive paths if not already added
bench2drive_paths = [
    '/mnt3/Documents/AD_Framework/Bench2Drive/leaderboard',
    '/mnt3/Documents/AD_Framework/Bench2Drive/leaderboard/leaderboard',
]

for path in bench2drive_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

def get_entry_point():
    """Return the name of the class to be instantiated"""
    return 'APIAgent'

class APIAgent(AutonomousAgent):
    """
    API Agent that receives actions from the REST API.
    This agent doesn't make decisions - it's controlled via the API.
    
    Key features:
    - Receives actions from REST API calls
    - Provides sensor data to the server for observations  
    - Uses the same sensor configuration as baseline dummy_agent3
    - Thread-safe action handling
    - Compatible with Bench2Drive leaderboard framework
    """
    
    def __init__(self, carla_host='127.0.0.1', carla_port=2000, debug=False):
        # Initialize the base AutonomousAgent with the required parameters
        super().__init__(carla_host, carla_port, debug)
        self.track = Track.SENSORS
        self._last_action = None
        self._action_queue = Queue()
        self._last_sensor_data = {}
        self._vehicle = None
        self._last_input_data = None
        self.step = 0
        self._sensor_data_lock = threading.Lock()
        
        # For state serialization (snapshot/restore)
        self._internal_state = {}
        
        print("[APIAgent] Initialized - waiting for actions from API")
    
    def setup(self, path_to_conf_file):
        """
        Setup the agent.
        """
        self.track = Track.SENSORS
        print(f"[APIAgent] Setup complete with config: {path_to_conf_file}")
    
    def sensors(self):
        """
        Define the sensor suite required by the agent.
        
        Using the same sensor configuration as the baseline dummy_agent3.py:
        - 3 RGB cameras (Center, Left, Right) with proper positioning
        - GNSS sensor for position
        - IMU sensor for orientation
        - Speedometer for vehicle speed
        
        This matches the sensor setup from the notebook baseline_v3.
        """
        print("[APIAgent] sensors() called - defining sensor suite")
        sensors = [
            # Center camera - main front-facing camera
            {
                'type': 'sensor.camera.rgb',
                'id': 'Center',
                'x': -1.5, 'y': 0.0, 'z': 2.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 1024, 'height': 512, 'fov': 110
            },
            # Left camera - angled left for wider view
            {
                'type': 'sensor.camera.rgb', 
                'id': 'Left',
                'x': -1.5, 'y': -0.5, 'z': 2.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
                'width': 1024, 'height': 512, 'fov': 110
            },
            # Right camera - angled right for wider view
            {
                'type': 'sensor.camera.rgb',
                'id': 'Right', 
                'x': -1.5, 'y': 0.5, 'z': 2.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
                'width': 1024, 'height': 512, 'fov': 110
            },
            # GNSS sensor for GPS coordinates
            {
                'type': 'sensor.other.gnss',
                'id': 'GPS',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            },
            # IMU sensor for acceleration and rotation
            {
                'type': 'sensor.other.imu',
                'id': 'IMU',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            },
            # Speedometer for vehicle speed
            {
                'type': 'sensor.speedometer',
                'id': 'SPEED',
                'reading_frequency': 20
            }
        ]
        
        return sensors
    
    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the global plan for the agent.
        """
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._global_plan_gps = global_plan_gps
        self._global_plan_world = global_plan_world_coord
        print(f"[APIAgent] Global plan set with {len(global_plan_gps)} waypoints")
    
    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        
        This method is called by the simulator at each time step.
        For API Agent: Store sensor data and return last received action from API.
        
        Args:
            input_data: Dictionary containing sensor data from CARLA
            timestamp: Current simulation timestamp
            
        Returns:
            carla.VehicleControl: Control commands for the vehicle
        """
        # Store the input_data safely - this contains all sensor readings
        with self._sensor_data_lock:
            self._last_input_data = input_data
            self._last_sensor_data = input_data  # This is what the server will use for observations
            
            # Debug info for first few steps
            if self.step < 3:
                print(f"[APIAgent] Step {self.step} - run_step called")
                print(f"[APIAgent] input_data type: {type(input_data)}")
                print(f"[APIAgent] input_data keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'NOT A DICT'}")
                
                # Check camera data structure
                for cam in ['Center', 'Left', 'Right']:
                    if cam in input_data:
                        cam_data = input_data[cam]
                        print(f"[APIAgent] {cam} camera type: {type(cam_data)}")
                        if isinstance(cam_data, tuple) and len(cam_data) >= 2:
                            print(f"[APIAgent] {cam} tuple length: {len(cam_data)}")
                            if hasattr(cam_data[1], 'shape'):
                                print(f"[APIAgent] {cam} data shape: {cam_data[1].shape}")
                
                # Check speedometer data
                if 'SPEED' in input_data:
                    speed_data = input_data['SPEED']
                    print(f"[APIAgent] SPEED data type: {type(speed_data)}")
                    if isinstance(speed_data, tuple) and len(speed_data) >= 2:
                        print(f"[APIAgent] SPEED data: {speed_data[1]}")
            
            # Log occasionally to show the agent is working
            if self.step % 50 == 0 and len(input_data) > 0:
                print(f"[APIAgent] Step {self.step}: Got sensor data for {list(input_data.keys())}")
        
        self.step += 1
        
        # Get vehicle reference if not already stored
        if self._vehicle is None and hasattr(self, '_world'):
            actors = self._world.get_actors()
            for actor in actors:
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    self._vehicle = actor
                    break
        
        # Create default control (safe stop)
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.steer = 0.0
        control.brake = 1.0  # Brake by default for safety
        control.hand_brake = False
        control.manual_gear_shift = False
        
        # Use action from API if available
        if self._last_action is not None:
            try:
                control.throttle = float(self._last_action.get('throttle', 0.0))
                control.steer = float(self._last_action.get('steer', 0.0))
                control.brake = float(self._last_action.get('brake', 0.0))
                control.hand_brake = bool(self._last_action.get('hand_brake', False))
                control.manual_gear_shift = bool(self._last_action.get('manual_gear_shift', False))
                
                # Log actions occasionally
                if self.step % 50 == 0:
                    print(f"[APIAgent] Step {self.step}: throttle={control.throttle:.2f}, "
                          f"steer={control.steer:.2f}, brake={control.brake:.2f}")
            except (ValueError, TypeError) as e:
                print(f"[APIAgent] Error parsing action: {e}, using safe defaults")
        else:
            # No action received yet - use safe defaults
            if self.step % 50 == 0:
                print(f"[APIAgent] Step {self.step}: No action from API yet, using safe stop")
        
        return control
    
    def set_action(self, action):
        """
        Set the next action to be executed.
        
        This method is called by the API server when it receives an action
        from the Gymnasium environment.
        
        Args:
            action: Dictionary with keys 'throttle', 'brake', 'steer', etc.
        """
        self._last_action = action
        if self.step % 20 == 0:  # Log occasionally
            print(f"[APIAgent] Received action: {action}")
    
    def get_last_sensor_data(self):
        """
        Get the last sensor data received.
        
        This method is used by the API server to extract observations
        for the Gymnasium environment.
        
        Returns:
            Dictionary containing sensor data from the last run_step call
        """
        with self._sensor_data_lock:
            return self._last_sensor_data
    
    def destroy(self):
        """
        Cleanup the agent.
        """
        print("[APIAgent] Destroying agent")
        super().destroy()
    
    @staticmethod
    def get_ros_version():
        """
        Returns the ROS version.
        
        Returns:
            int: 0 for no ROS, 1 for ROS1, 2 for ROS2
        """
        return 0  # No ROS required
    
    # Additional utility methods for debugging and monitoring
    
    def get_sensor_info(self):
        """Get information about available sensors"""
        if not hasattr(self, '_last_sensor_data') or not self._last_sensor_data:
            return {}
        
        info = {}
        for sensor_id, sensor_data in self._last_sensor_data.items():
            if isinstance(sensor_data, tuple) and len(sensor_data) >= 2:
                timestamp, data = sensor_data[0], sensor_data[1]
                if hasattr(data, 'shape'):
                    info[sensor_id] = {
                        'timestamp': timestamp,
                        'shape': data.shape,
                        'dtype': str(data.dtype) if hasattr(data, 'dtype') else 'unknown'
                    }
                else:
                    info[sensor_id] = {
                        'timestamp': timestamp,
                        'type': type(data).__name__,
                        'value': str(data)[:100] + '...' if len(str(data)) > 100 else str(data)
                    }
        return info
    
    def capture_state(self):
        """
        Capture agent's internal state for snapshot.
        
        Returns:
            Dictionary containing all agent state that needs to be preserved
        """
        state = {
            'step': self.step,
            'last_action': self._last_action.copy() if self._last_action else None,
            'internal_state': self._internal_state.copy(),
            'global_plan_size': len(self._global_plan_gps) if hasattr(self, '_global_plan_gps') else 0
        }
        
        # Don't include sensor data (too large and regenerated each step)
        # Don't include vehicle reference (will be re-acquired)
        
        return state
    
    def restore_state(self, state):
        """
        Restore agent's internal state from snapshot.
        
        Args:
            state: Dictionary containing agent state to restore
        """
        if state:
            self.step = state.get('step', 0)
            self._last_action = state.get('last_action')
            self._internal_state = state.get('internal_state', {}).copy()
            
            # Note: global plan is managed by leaderboard, no need to restore
            
            print(f"[APIAgent] State restored to step {self.step}")
    
    def get_vehicle_state(self):
        """Get current vehicle state if available"""
        if self._vehicle is None:
            return None
        
        try:
            transform = self._vehicle.get_transform()
            velocity = self._vehicle.get_velocity()
            
            return {
                'location': {
                    'x': transform.location.x,
                    'y': transform.location.y,
                    'z': transform.location.z
                },
                'rotation': {
                    'pitch': transform.rotation.pitch,
                    'yaw': transform.rotation.yaw,
                    'roll': transform.rotation.roll
                },
                'velocity': {
                    'x': velocity.x,
                    'y': velocity.y,
                    'z': velocity.z
                },
                'speed': np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            }
        except Exception as e:
            print(f"[APIAgent] Error getting vehicle state: {e}")
            return None
    
    def is_ready(self):
        """Check if agent has received sensor data and is ready to operate"""
        return (hasattr(self, '_last_sensor_data') and 
                self._last_sensor_data is not None and 
                len(self._last_sensor_data) > 0)
    
    def get_action_history(self, n=10):
        """Get history of last n actions (for debugging)"""
        # This could be extended to keep a history of actions
        # For now, just return the current action
        return [self._last_action] if self._last_action else []

# Additional utility functions for the agent

def create_api_agent(carla_host='127.0.0.1', carla_port=2000, debug=False):
    """
    Factory function to create an APIAgent instance.
    
    Args:
        carla_host: CARLA server host
        carla_port: CARLA server port  
        debug: Enable debug mode
        
    Returns:
        APIAgent: Configured agent instance
    """
    return APIAgent(carla_host, carla_port, debug)

def validate_action(action):
    """
    Validate an action dictionary for the APIAgent.
    
    Args:
        action: Dictionary containing action values
        
    Returns:
        bool: True if action is valid
        str: Error message if invalid, None if valid
    """
    if not isinstance(action, dict):
        return False, "Action must be a dictionary"
    
    required_keys = ['throttle', 'brake', 'steer']
    for key in required_keys:
        if key not in action:
            return False, f"Missing required key: {key}"
        
        try:
            val = float(action[key])
        except (ValueError, TypeError):
            return False, f"Key '{key}' must be a number"
        
        # Validate ranges
        if key in ['throttle', 'brake']:
            if not 0.0 <= val <= 1.0:
                return False, f"Key '{key}' must be between 0.0 and 1.0"
        elif key == 'steer':
            if not -1.0 <= val <= 1.0:
                return False, f"Key '{key}' must be between -1.0 and 1.0"
    
    return True, None

if __name__ == "__main__":
    # Simple test script for the APIAgent
    print("Testing APIAgent...")
    
    # Create agent instance
    agent = create_api_agent(debug=True)
    
    # Test sensor configuration
    sensors = agent.sensors()
    print(f"Agent sensors: {len(sensors)}")
    for sensor in sensors:
        print(f"  - {sensor['id']}: {sensor['type']}")
    
    # Test action validation
    valid_action = {'throttle': 0.5, 'brake': 0.0, 'steer': 0.1}
    is_valid, error = validate_action(valid_action)
    print(f"Valid action test: {is_valid} (error: {error})")
    
    invalid_action = {'throttle': 1.5, 'brake': 0.0}  # Missing steer, invalid throttle
    is_valid, error = validate_action(invalid_action)
    print(f"Invalid action test: {is_valid} (error: {error})")
    
    print("APIAgent test complete!")