#!/usr/bin/env python3
"""
World Snapshot System for GRPO Support

Provides API-level snapshot/restore capability for CARLA + ScenarioRunner/Leaderboard.
Enables GRPO multi-turn rollouts with branching without OS-level dependencies.

Key Features:
- Complete world state capture
- Fast restoration for branching
- Memory-efficient storage
- GPU-safe implementation
- Deterministic for short horizons
"""

import time
import uuid
import json
import copy
import logging
import traceback
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import numpy as np

try:
    import carla
except ImportError:
    carla = None

logger = logging.getLogger(__name__)


@dataclass
class VehicleState:
    """State of a single vehicle"""
    id: int
    type_id: str
    location: Dict[str, float]  # x, y, z
    rotation: Dict[str, float]  # pitch, yaw, roll
    velocity: Dict[str, float]  # x, y, z
    angular_velocity: Dict[str, float]  # x, y, z
    control: Dict[str, float]  # throttle, steer, brake, hand_brake, reverse, manual_gear_shift, gear
    attributes: Dict[str, str] = field(default_factory=dict)
    is_hero: bool = False
    

@dataclass
class PedestrianState:
    """State of a single pedestrian"""
    id: int
    type_id: str
    location: Dict[str, float]
    rotation: Dict[str, float]
    velocity: Dict[str, float]
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrafficLightState:
    """State of a traffic light"""
    id: int
    state: str  # Red, Yellow, Green, Off, Unknown
    location: Dict[str, float]
    rotation: Dict[str, float]
    
    
@dataclass 
class WeatherState:
    """Weather parameters"""
    cloudiness: float
    precipitation: float
    precipitation_deposits: float
    wind_intensity: float
    sun_azimuth_angle: float
    sun_altitude_angle: float
    fog_density: float
    fog_distance: float
    wetness: float
    fog_falloff: float
    scattering_intensity: float
    mie_scattering_scale: float
    rayleigh_scattering_scale: float


@dataclass
class ScenarioState:
    """Scenario/Route state"""
    route_id: str
    route_completion: float
    current_waypoint_index: int
    total_waypoints: int
    active_triggers: List[str] = field(default_factory=list)
    completed_triggers: List[str] = field(default_factory=list)
    route_waypoints: List[Dict] = field(default_factory=list)  # Store waypoint transforms
    

@dataclass
class ScenarioManagerState:
    """Complete ScenarioManager internal state for proper restoration"""
    # Core identification
    route_index: Optional[int] = None
    repetition_number: Optional[int] = None
    
    # Timing state
    tick_count: int = 0
    timestamp_last_run: float = 0.0
    scenario_duration_system: float = 0.0
    scenario_duration_game: float = 0.0
    start_system_time: float = 0.0
    start_game_time: float = 0.0
    end_system_time: float = 0.0
    end_game_time: float = 0.0
    
    # Execution state
    running: bool = False
    debug_mode: int = 0
    timeout: float = 120.0
    
    # Vehicle and actor IDs
    ego_vehicle_ids: List[int] = field(default_factory=list)
    other_actor_ids: List[int] = field(default_factory=list)
    
    # Scenario tree state (behavior tree)
    scenario_tree_status: Optional[str] = None  # RUNNING, SUCCESS, FAILURE
    scenario_tree_blackboard: Dict[str, Any] = field(default_factory=dict)
    
    # Agent wrapper state
    agent_wallclock_t0: Optional[float] = None
    agent_sensors_list: List[int] = field(default_factory=list)  # Sensor IDs
    

@dataclass
class AgentState:
    """Agent internal state"""
    step_count: int
    last_action: Optional[Dict[str, float]]
    sensor_data: Dict[str, Any] = field(default_factory=dict)
    internal_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationMetrics:
    """Simulation metrics and statistics"""
    step_count: int
    game_time: float
    system_time: float
    cumulative_reward: float
    route_score: float = 0.0
    infraction_count: int = 0
    collision_count: int = 0


@dataclass
class LeaderboardEvaluatorState:
    """Complete LeaderboardEvaluator state for proper restore"""
    # Core components
    sensors_config: List[Dict[str, Any]] = field(default_factory=list)  # Sensor definitions
    sensors_initialized: bool = False
    sensor_icons: List[str] = field(default_factory=list)
    
    # Timing
    start_time: float = 0.0
    end_time: Optional[float] = None
    client_timeout: float = 300.0
    
    # Agent module info
    agent_module_name: Optional[str] = None
    agent_class_name: Optional[str] = None
    agent_config: Optional[str] = None
    
    # Route info
    route_index: int = 0
    repetition_index: int = 0
    
    # Statistics
    statistics_data: Optional[Dict[str, Any]] = None
    


class WorldSnapshot:
    """
    Complete world state snapshot for GRPO branching.
    
    Captures and restores:
    - All vehicle states (ego + NPCs)
    - Pedestrian states
    - Traffic light states
    - Weather and environment
    - Scenario progress
    - Agent state
    - Simulation metrics
    """
    
    def __init__(self, snapshot_id: Optional[str] = None):
        self.snapshot_id = snapshot_id or str(uuid.uuid4())[:8]
        self.timestamp = time.time()
        
        # State components
        self.vehicles: Dict[int, VehicleState] = {}
        self.pedestrians: Dict[int, PedestrianState] = {}
        self.traffic_lights: Dict[int, TrafficLightState] = {}
        self.weather: Optional[WeatherState] = None
        self.scenario: Optional[ScenarioState] = None
        self.scenario_manager: Optional[ScenarioManagerState] = None  # NEW: Complete manager state
        self.agent: Optional[AgentState] = None
        self.metrics: Optional[SimulationMetrics] = None
        self.leaderboard_evaluator: Optional[LeaderboardEvaluatorState] = None  # NEW: Full evaluator state
        
        # Additional metadata
        self.map_name: Optional[str] = None
        self.spectator_transform: Optional[Dict] = None
        
    @classmethod
    def capture(cls, sim_state, world=None) -> 'WorldSnapshot':
        """
        Capture complete world state from current simulation.
        
        Args:
            sim_state: Current SimulationState object
            world: CARLA world object (if not available in sim_state)
        """
        snapshot = cls()
        
        try:
            # Get world reference
            if world is None:
                if hasattr(sim_state, 'leaderboard_evaluator') and sim_state.leaderboard_evaluator:
                    if hasattr(sim_state.leaderboard_evaluator, 'world'):
                        world = sim_state.leaderboard_evaluator.world
                
            if world is None:
                logger.error("No world object available for snapshot")
                return snapshot
                
            # Capture map name
            snapshot.map_name = world.get_map().name
            
            # Capture spectator
            spectator = world.get_spectator()
            if spectator:
                transform = spectator.get_transform()
                snapshot.spectator_transform = {
                    'location': {'x': transform.location.x, 'y': transform.location.y, 'z': transform.location.z},
                    'rotation': {'pitch': transform.rotation.pitch, 'yaw': transform.rotation.yaw, 'roll': transform.rotation.roll}
                }
            
            # Capture all vehicles
            snapshot._capture_vehicles(world, sim_state)
            
            # Capture pedestrians
            snapshot._capture_pedestrians(world)
            
            # Capture traffic lights
            snapshot._capture_traffic_lights(world)
            
            # Capture weather
            snapshot._capture_weather(world)
            
            # Capture scenario state
            snapshot._capture_scenario(sim_state)
            
            # Capture scenario manager state (NEW)
            snapshot._capture_scenario_manager(sim_state)
            
            # Capture agent state
            snapshot._capture_agent(sim_state)
            
            # Capture LeaderboardEvaluator state
            snapshot._capture_leaderboard_evaluator(sim_state)
            
            # Capture metrics
            snapshot._capture_metrics(sim_state)
            
            logger.info(f"Captured snapshot {snapshot.snapshot_id} with {len(snapshot.vehicles)} vehicles")
            
        except Exception as e:
            logger.error(f"Error capturing snapshot: {e}")
            traceback.print_exc()
            
        return snapshot
    
    def _capture_vehicles(self, world, sim_state):
        """Capture all vehicle states"""
        actors = world.get_actors().filter('vehicle.*')
        
        # Find hero vehicle ID
        hero_id = None
        if hasattr(sim_state, 'agent_instance') and sim_state.agent_instance:
            if hasattr(sim_state.agent_instance, '_vehicle') and sim_state.agent_instance._vehicle:
                hero_id = sim_state.agent_instance._vehicle.id
        
        for vehicle in actors:
            try:
                transform = vehicle.get_transform()
                velocity = vehicle.get_velocity()
                angular_velocity = vehicle.get_angular_velocity()
                control = vehicle.get_control()
                
                state = VehicleState(
                    id=vehicle.id,
                    type_id=vehicle.type_id,
                    location={'x': transform.location.x, 'y': transform.location.y, 'z': transform.location.z},
                    rotation={'pitch': transform.rotation.pitch, 'yaw': transform.rotation.yaw, 'roll': transform.rotation.roll},
                    velocity={'x': velocity.x, 'y': velocity.y, 'z': velocity.z},
                    angular_velocity={'x': angular_velocity.x, 'y': angular_velocity.y, 'z': angular_velocity.z},
                    control={
                        'throttle': control.throttle,
                        'steer': control.steer,
                        'brake': control.brake,
                        'hand_brake': control.hand_brake,
                        'reverse': control.reverse,
                        'manual_gear_shift': control.manual_gear_shift,
                        'gear': control.gear
                    },
                    attributes=vehicle.attributes,
                    is_hero=(vehicle.id == hero_id)
                )
                
                self.vehicles[vehicle.id] = state
                
            except Exception as e:
                logger.warning(f"Failed to capture vehicle {vehicle.id}: {e}")
    
    def _capture_pedestrians(self, world):
        """Capture all pedestrian states"""
        actors = world.get_actors().filter('walker.pedestrian.*')
        
        for pedestrian in actors:
            try:
                transform = pedestrian.get_transform()
                velocity = pedestrian.get_velocity()
                
                state = PedestrianState(
                    id=pedestrian.id,
                    type_id=pedestrian.type_id,
                    location={'x': transform.location.x, 'y': transform.location.y, 'z': transform.location.z},
                    rotation={'pitch': transform.rotation.pitch, 'yaw': transform.rotation.yaw, 'roll': transform.rotation.roll},
                    velocity={'x': velocity.x, 'y': velocity.y, 'z': velocity.z},
                    attributes=pedestrian.attributes
                )
                
                self.pedestrians[pedestrian.id] = state
                
            except Exception as e:
                logger.warning(f"Failed to capture pedestrian {pedestrian.id}: {e}")
    
    def _capture_traffic_lights(self, world):
        """Capture traffic light states"""
        actors = world.get_actors().filter('traffic.traffic_light')
        
        for light in actors:
            try:
                transform = light.get_transform()
                state_enum = light.get_state()
                
                # Convert enum to string
                state_str = str(state_enum).split('.')[-1] if state_enum else "Unknown"
                
                state = TrafficLightState(
                    id=light.id,
                    state=state_str,
                    location={'x': transform.location.x, 'y': transform.location.y, 'z': transform.location.z},
                    rotation={'pitch': transform.rotation.pitch, 'yaw': transform.rotation.yaw, 'roll': transform.rotation.roll}
                )
                
                self.traffic_lights[light.id] = state
                
            except Exception as e:
                logger.warning(f"Failed to capture traffic light {light.id}: {e}")
    
    def _capture_weather(self, world):
        """Capture weather parameters"""
        try:
            weather = world.get_weather()
            
            self.weather = WeatherState(
                cloudiness=weather.cloudiness,
                precipitation=weather.precipitation,
                precipitation_deposits=weather.precipitation_deposits,
                wind_intensity=weather.wind_intensity,
                sun_azimuth_angle=weather.sun_azimuth_angle,
                sun_altitude_angle=weather.sun_altitude_angle,
                fog_density=weather.fog_density,
                fog_distance=weather.fog_distance,
                wetness=weather.wetness,
                fog_falloff=weather.fog_falloff,
                scattering_intensity=weather.scattering_intensity,
                mie_scattering_scale=weather.mie_scattering_scale,
                rayleigh_scattering_scale=weather.rayleigh_scattering_scale
            )
        except Exception as e:
            logger.warning(f"Failed to capture weather: {e}")
    
    def _capture_scenario(self, sim_state):
        """Capture scenario/route state"""
        try:
            route_completion = 0.0
            current_waypoint = 0
            total_waypoints = 0
            waypoints = []
            
            # Extract from leaderboard evaluator
            if hasattr(sim_state, 'leaderboard_evaluator') and sim_state.leaderboard_evaluator:
                evaluator = sim_state.leaderboard_evaluator
                
                # Get route progress from manager
                if hasattr(evaluator, 'manager') and evaluator.manager:
                    manager = evaluator.manager
                    
                    # Try to get route completion
                    if hasattr(manager, '_route_scenario'):
                        scenario = manager._route_scenario
                        if hasattr(scenario, 'route'):
                            route = scenario.route
                            if route:
                                total_waypoints = len(route)
                                
                                # Get current waypoint index
                                if hasattr(scenario, 'route_index'):
                                    current_waypoint = scenario.route_index
                                    route_completion = current_waypoint / total_waypoints if total_waypoints > 0 else 0.0
                                
                                # Store waypoint transforms (first 10 upcoming)
                                for i in range(min(10, total_waypoints - current_waypoint)):
                                    wp = route[current_waypoint + i][0]
                                    waypoints.append({
                                        'location': {'x': wp.location.x, 'y': wp.location.y, 'z': wp.location.z},
                                        'rotation': {'pitch': wp.rotation.pitch, 'yaw': wp.rotation.yaw, 'roll': wp.rotation.roll}
                                    })
            
            self.scenario = ScenarioState(
                route_id=str(sim_state.current_route) if sim_state.current_route else "unknown",
                route_completion=route_completion,
                current_waypoint_index=current_waypoint,
                total_waypoints=total_waypoints,
                route_waypoints=waypoints
            )
            
        except Exception as e:
            logger.warning(f"Failed to capture scenario state: {e}")
    
    def _capture_scenario_manager(self, sim_state):
        """Capture complete ScenarioManager state"""
        try:
            manager_state = ScenarioManagerState()
            
            # Get the scenario manager from leaderboard evaluator
            if hasattr(sim_state, 'leaderboard_evaluator') and sim_state.leaderboard_evaluator:
                evaluator = sim_state.leaderboard_evaluator
                if hasattr(evaluator, 'manager') and evaluator.manager:
                    manager = evaluator.manager
                    
                    # Capture all manager attributes
                    manager_state.route_index = getattr(manager, 'route_index', None)
                    manager_state.repetition_number = getattr(manager, 'repetition_number', None)
                    
                    # Timing state
                    manager_state.tick_count = getattr(manager, 'tick_count', 0)
                    manager_state.timestamp_last_run = getattr(manager, '_timestamp_last_run', 0.0)
                    manager_state.scenario_duration_system = getattr(manager, 'scenario_duration_system', 0.0)
                    manager_state.scenario_duration_game = getattr(manager, 'scenario_duration_game', 0.0)
                    manager_state.start_system_time = getattr(manager, 'start_system_time', 0.0)
                    manager_state.start_game_time = getattr(manager, 'start_game_time', 0.0)
                    manager_state.end_system_time = getattr(manager, 'end_system_time', 0.0)
                    manager_state.end_game_time = getattr(manager, 'end_game_time', 0.0)
                    
                    # Execution state
                    manager_state.running = getattr(manager, '_running', False)
                    manager_state.debug_mode = getattr(manager, '_debug_mode', 0)
                    manager_state.timeout = getattr(manager, '_timeout', 120.0)
                    
                    # Vehicle and actor IDs
                    if hasattr(manager, 'ego_vehicles'):
                        manager_state.ego_vehicle_ids = [v.id for v in manager.ego_vehicles]
                    if hasattr(manager, 'other_actors'):
                        manager_state.other_actor_ids = [a.id for a in manager.other_actors]
                    
                    # Scenario tree state
                    if hasattr(manager, 'scenario_tree'):
                        tree = manager.scenario_tree
                        if tree:
                            manager_state.scenario_tree_status = str(tree.status) if hasattr(tree, 'status') else None
                            
                            # Try to capture blackboard data
                            try:
                                import py_trees
                                bb = py_trees.blackboard.Blackboard()
                                # Capture specific blackboard values we care about
                                if bb.exists('AV_control'):
                                    control = bb.get('AV_control')
                                    if control:
                                        manager_state.scenario_tree_blackboard['AV_control'] = {
                                            'throttle': control.throttle,
                                            'steer': control.steer,
                                            'brake': control.brake
                                        }
                            except:
                                pass
                    
                    # Agent wrapper state
                    if hasattr(manager, '_agent_wrapper'):
                        wrapper = manager._agent_wrapper
                        if wrapper:
                            if hasattr(wrapper, '_agent'):
                                agent = wrapper._agent
                                if hasattr(agent, 'wallclock_t0'):
                                    manager_state.agent_wallclock_t0 = agent.wallclock_t0
                            if hasattr(wrapper, '_sensors_list'):
                                manager_state.agent_sensors_list = [s.id if hasattr(s, 'id') else 0 for s in wrapper._sensors_list]
                    
                    self.scenario_manager = manager_state
                    logger.info(f"Captured scenario manager state: tick={manager_state.tick_count}, running={manager_state.running}")
                    
                    # Debug output
                    print(f"DEBUG: Captured ScenarioManager state:")
                    print(f"  - tick_count: {manager_state.tick_count}")
                    print(f"  - _running: {manager_state.running}")
                    print(f"  - ego_vehicle_ids: {manager_state.ego_vehicle_ids}")
                    print(f"  - route_index: {manager_state.route_index}")
                    print(f"  - Has scenario_tree: {manager.scenario_tree is not None if hasattr(manager, 'scenario_tree') else False}")
                    print(f"  - Has _agent_wrapper: {hasattr(manager, '_agent_wrapper')}")
                    
        except Exception as e:
            logger.warning(f"Failed to capture scenario manager state: {e}")
    
    def _capture_agent(self, sim_state):
        """Capture agent state"""
        try:
            self.agent = AgentState(
                step_count=sim_state.step_count,
                last_action=copy.deepcopy(sim_state.last_action) if sim_state.last_action else None,
                sensor_data={},  # Could capture last sensor readings if needed
                internal_state={}  # Agent-specific internal state
            )
            
            # Capture agent-specific state if available
            if hasattr(sim_state, 'agent_instance') and sim_state.agent_instance:
                agent = sim_state.agent_instance
                
                # Capture last sensor data if available
                if hasattr(agent, '_last_sensor_data'):
                    # Don't copy image data, just metadata
                    for key, value in agent._last_sensor_data.items():
                        if key in ['GPS', 'IMU', 'SPEED']:
                            self.agent.sensor_data[key] = copy.deepcopy(value)
                
        except Exception as e:
            logger.warning(f"Failed to capture agent state: {e}")
    
    def _capture_leaderboard_evaluator(self, sim_state):
        """Capture complete LeaderboardEvaluator state"""
        try:
            if hasattr(sim_state, 'leaderboard_evaluator') and sim_state.leaderboard_evaluator:
                evaluator = sim_state.leaderboard_evaluator
                
                state = LeaderboardEvaluatorState()
                
                # Capture sensor configuration
                if hasattr(evaluator, 'sensors'):
                    state.sensors_config = evaluator.sensors.copy() if evaluator.sensors else []
                state.sensors_initialized = getattr(evaluator, 'sensors_initialized', False)
                if hasattr(evaluator, 'sensor_icons'):
                    state.sensor_icons = evaluator.sensor_icons.copy() if evaluator.sensor_icons else []
                
                # Capture timing
                state.start_time = getattr(evaluator, '_start_time', 0.0)
                state.end_time = getattr(evaluator, '_end_time', None)
                state.client_timeout = getattr(evaluator, 'client_timeout', 300.0)
                
                # Capture agent module info
                if hasattr(evaluator, 'agent_instance'):
                    agent = evaluator.agent_instance
                    state.agent_class_name = agent.__class__.__name__
                    state.agent_module_name = agent.__class__.__module__
                
                # Capture route info
                if hasattr(evaluator, 'manager') and evaluator.manager:
                    state.route_index = getattr(evaluator.manager, 'route_index', 0)
                    state.repetition_index = getattr(evaluator.manager, 'repetition_number', 0)
                
                # Capture statistics
                if hasattr(evaluator, 'statistics_manager'):
                    # We can't pickle the entire statistics manager, but we can save key data
                    stats_mgr = evaluator.statistics_manager
                    state.statistics_data = {
                        'route_id': getattr(stats_mgr, 'route_id', None),
                        'index': getattr(stats_mgr, 'index', 0),
                        'total': getattr(stats_mgr, 'total', 0)
                    }
                
                self.leaderboard_evaluator = state
                logger.info("Captured LeaderboardEvaluator state")
                
        except Exception as e:
            logger.warning(f"Failed to capture LeaderboardEvaluator state: {e}")
            import traceback
            traceback.print_exc()
    
    def _capture_metrics(self, sim_state):
        """Capture simulation metrics"""
        try:
            self.metrics = SimulationMetrics(
                step_count=sim_state.step_count,
                game_time=0.0,  # Would need GameTime access
                system_time=time.time(),
                cumulative_reward=sim_state.cumulative_reward
            )
            
            # Try to get additional metrics from evaluator
            if hasattr(sim_state, 'leaderboard_evaluator') and sim_state.leaderboard_evaluator:
                evaluator = sim_state.leaderboard_evaluator
                if hasattr(evaluator, 'manager') and evaluator.manager:
                    manager = evaluator.manager
                    
                    # Get game time if available
                    if hasattr(manager, 'start_game_time'):
                        self.metrics.game_time = manager.start_game_time
                        
        except Exception as e:
            logger.warning(f"Failed to capture metrics: {e}")
    
    def restore(self, sim_state, world=None):
        """
        Restore world state from snapshot.
        
        Args:
            sim_state: SimulationState to restore to
            world: CARLA world object
        """
        try:
            # Get world reference
            if world is None:
                if hasattr(sim_state, 'leaderboard_evaluator') and sim_state.leaderboard_evaluator:
                    if hasattr(sim_state.leaderboard_evaluator, 'world'):
                        world = sim_state.leaderboard_evaluator.world
            
            if world is None:
                logger.error("No world object available for restore")
                return False
            
            logger.info(f"Restoring snapshot {self.snapshot_id}")
            
            # Restore in specific order for consistency
            
            # 1. Restore weather first (affects physics)
            if self.weather:
                self._restore_weather(world)
            
            # 2. Restore vehicles
            self._restore_vehicles(world, sim_state)
            
            # 3. Restore pedestrians
            self._restore_pedestrians(world)
            
            # 4. Restore traffic lights
            self._restore_traffic_lights(world)
            
            # 5. Restore scenario state
            self._restore_scenario(sim_state)
            
            # 6. Restore scenario manager state (NEW - CRITICAL)
            self._restore_scenario_manager(sim_state)
            
            # 7. Restore agent state
            self._restore_agent(sim_state)
            
            # 7. Restore metrics
            self._restore_metrics(sim_state)
            
            # 8. Restore spectator
            if self.spectator_transform:
                self._restore_spectator(world)
            
            # CRITICAL: Ensure Traffic Manager and vehicles are properly synced
            # Get and configure Traffic Manager
            if hasattr(sim_state, 'leaderboard_evaluator'):
                evaluator = sim_state.leaderboard_evaluator
                if hasattr(evaluator, 'traffic_manager'):
                    tm = evaluator.traffic_manager
                    if tm:
                        # Ensure TM is in sync mode
                        tm.set_synchronous_mode(True)
                        # Force TM to update all vehicle controls
                        tm.set_hybrid_physics_mode(True)
                        logger.info(f"Traffic Manager configured: sync=True, hybrid=True, port={tm.get_port()}")
            
            # Single final tick to apply all changes at once
            world.tick()
            
            # CRITICAL: Force all vehicles with autopilot to update
            # Sometimes the first tick doesn't activate autopilot properly
            vehicles = world.get_actors().filter('vehicle.*')
            for v in vehicles:
                try:
                    if v.id != ego_vehicle_id:
                        # Give them a tiny nudge to ensure physics is active
                        v.add_impulse(carla.Vector3D(x=0.001, y=0.001, z=0))
                except:
                    pass
            
            logger.info(f"Restored snapshot {self.snapshot_id} successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring snapshot: {e}")
            traceback.print_exc()
            return False
    
    def _restore_vehicles(self, world, sim_state):
        """Restore vehicle states"""
        current_actors = world.get_actors().filter('vehicle.*')
        
        # Track ego vehicle for special handling
        ego_vehicle = None
        if hasattr(sim_state, 'leaderboard_evaluator') and sim_state.leaderboard_evaluator:
            if hasattr(sim_state.leaderboard_evaluator, 'manager') and sim_state.leaderboard_evaluator.manager:
                manager = sim_state.leaderboard_evaluator.manager
                if hasattr(manager, 'ego_vehicles') and len(manager.ego_vehicles) > 0:
                    ego_vehicle = manager.ego_vehicles[0]
        
        # CRITICAL: Match vehicles by position, not ID!
        # After destroy/spawn cycles, IDs change but positions should be similar
        logger.info(f"Current vehicles: {len(current_actors)}, Snapshot vehicles: {len(self.vehicles)}")
        
        # Build list of current vehicles (excluding ego)
        current_vehicles = []
        for actor in current_actors:
            if ego_vehicle and actor.id == ego_vehicle.id:
                continue
            loc = actor.get_location()
            current_vehicles.append({
                'actor': actor,
                'x': loc.x,
                'y': loc.y,
                'z': loc.z
            })
        
        # Build list of snapshot vehicles (excluding ego)
        snapshot_vehicles = []
        for vehicle_id, state in self.vehicles.items():
            if state.is_hero:
                continue
            snapshot_vehicles.append({
                'state': state,
                'id': vehicle_id,
                'x': state.location['x'],
                'y': state.location['y'],
                'z': state.location['z']
            })
        
        # Match vehicles by closest position
        matched_pairs = []
        used_current = set()
        for snap_v in snapshot_vehicles:
            best_match = None
            best_dist = float('inf')
            
            for i, curr_v in enumerate(current_vehicles):
                if i in used_current:
                    continue
                    
                # Calculate distance
                dist = ((curr_v['x'] - snap_v['x'])**2 + 
                       (curr_v['y'] - snap_v['y'])**2 + 
                       (curr_v['z'] - snap_v['z'])**2) ** 0.5
                
                if dist < best_dist and dist < 100:  # Within 100m is likely the same vehicle
                    best_dist = dist
                    best_match = i
            
            if best_match is not None:
                matched_pairs.append((snapshot_vehicles.index(snap_v), best_match))
                used_current.add(best_match)
                logger.debug(f"Matched snapshot vehicle at ({snap_v['x']:.1f}, {snap_v['y']:.1f}) with current vehicle at ({current_vehicles[best_match]['x']:.1f}, {current_vehicles[best_match]['y']:.1f}), distance: {best_dist:.2f}m")
        
        # Now restore matched vehicles
        for snap_idx, curr_idx in matched_pairs:
            state = snapshot_vehicles[snap_idx]['state']
            vehicle = current_vehicles[curr_idx]['actor']
            vehicle_id = vehicle.id  # Use the actual current vehicle ID
            
            try:
                # Log the restoration details
                logger.info(f"Restoring vehicle {vehicle_id} to snapshot position")
                logger.info(f"  Target position: x={state.location['x']:.2f}, y={state.location['y']:.2f}, z={state.location['z']:.2f}")
                
                # Get current position before restore
                current_transform = vehicle.get_transform()
                current_loc = current_transform.location
                logger.info(f"  Current position: x={current_loc.x:.2f}, y={current_loc.y:.2f}, z={current_loc.z:.2f}")
                
                # Set transform
                transform = carla.Transform(
                    carla.Location(x=state.location['x'], y=state.location['y'], z=state.location['z']),
                    carla.Rotation(pitch=state.rotation['pitch'], yaw=state.rotation['yaw'], roll=state.rotation['roll'])
                )
                vehicle.set_transform(transform)
                
                # CRITICAL FIX: Properly restore vehicle physics state
                # Reset physics without ticking to avoid position drift
                vehicle.set_simulate_physics(False)
                vehicle.set_simulate_physics(True)
                
                # Set the transform again after physics reset
                vehicle.set_transform(transform)
                
                # Set velocity using physics engine
                velocity = carla.Vector3D(x=state.velocity['x'], y=state.velocity['y'], z=state.velocity['z'])
                angular_velocity = carla.Vector3D(x=0, y=0, z=0)  # Reset angular velocity
                
                # Apply both linear and angular velocity
                vehicle.set_target_velocity(velocity)
                vehicle.set_target_angular_velocity(angular_velocity)
                        
                        # For ego vehicle, ensure proper control state
                        if vehicle_id == ego_vehicle_id:
                            # CRITICAL: Ensure vehicle can be controlled
                            vehicle.set_autopilot(False)
                            
                            # Enable vehicle control explicitly
                            physics_control = vehicle.get_physics_control()
                            physics_control.use_gear_autobox = True
                            vehicle.apply_physics_control(physics_control)
                            
                            # CRITICAL FIX: Wake up the vehicle physics
                            # Apply a small impulse to ensure physics are active
                            impulse = carla.Vector3D(x=0.1, y=0, z=0)
                            vehicle.add_impulse(impulse)
                            
                            # Reset control to neutral first
                            neutral_control = carla.VehicleControl(
                                throttle=0.0,
                                steer=0.0,
                                brake=0.0,
                                hand_brake=False,
                                reverse=False,
                                manual_gear_shift=False,
                                gear=0
                            )
                            vehicle.apply_control(neutral_control)
                            
                            # Enable the vehicle to receive controls
                            vehicle.set_simulate_physics(True)
                            
                            # Verify the position was actually restored (without ticking)
                            actual_transform = vehicle.get_transform()
                            actual_loc = actual_transform.location
                            distance = ((actual_loc.x - state.location['x'])**2 + 
                                      (actual_loc.y - state.location['y'])**2 + 
                                      (actual_loc.z - state.location['z'])**2) ** 0.5
                            logger.info(f"Ego vehicle {vehicle_id} restored:")
                            logger.info(f"  Target position: x={state.location['x']:.2f}, y={state.location['y']:.2f}, z={state.location['z']:.2f}")
                            logger.info(f"  Actual position: x={actual_loc.x:.2f}, y={actual_loc.y:.2f}, z={actual_loc.z:.2f}")
                            logger.info(f"  Distance from target: {distance:.2f}m")
                            if distance > 1.0:
                                logger.warning(f"Large position error after restore: {distance:.2f}m")
                        else:
                            # CRITICAL FIX: Re-enable autopilot for non-ego vehicles
                            # They need Traffic Manager control to continue moving
                            
                            # First ensure physics is active
                            vehicle.set_simulate_physics(True)
                            
                            # CRITICAL: Force vehicle to wake up
                            # Disable and re-enable physics to ensure it's not stuck
                            vehicle.set_simulate_physics(False)
                            vehicle.set_simulate_physics(True)
                            
                            # Apply the stored velocity to get it moving
                            if state.velocity:
                                velocity = carla.Vector3D(
                                    x=state.velocity['x'], 
                                    y=state.velocity['y'], 
                                    z=state.velocity['z']
                                )
                                vehicle.set_target_velocity(velocity)
                            
                            # Apply a stronger wake impulse based on velocity direction
                            if state.velocity['x'] != 0 or state.velocity['y'] != 0:
                                # Use velocity direction for impulse
                                wake_impulse = carla.Vector3D(
                                    x=state.velocity['x'] * 0.1, 
                                    y=state.velocity['y'] * 0.1, 
                                    z=0
                                )
                            else:
                                # Default small impulse
                                wake_impulse = carla.Vector3D(x=0.1, y=0.1, z=0)
                            vehicle.add_impulse(wake_impulse)
                            
                            # Get Traffic Manager instance from the simulation
                            tm = None
                            
                            # First try to get from sim_state if available
                            if hasattr(sim_state, 'leaderboard_evaluator'):
                                evaluator = sim_state.leaderboard_evaluator
                                if hasattr(evaluator, 'traffic_manager'):
                                    tm = evaluator.traffic_manager
                                    logger.debug(f"Got TM from evaluator, port: {tm.get_port() if tm else 'None'}")
                            
                            # If not found, try to get from client
                            if not tm:
                                try:
                                    client = world.get_client() if hasattr(world, 'get_client') else None
                                    if client:
                                        # Try common TM ports
                                        for tm_port in [3000, 8000, 3002, 8002]:
                                            try:
                                                tm = client.get_trafficmanager(tm_port)
                                                if tm:
                                                    logger.debug(f"Got TM from client on port {tm_port}")
                                                    break
                                            except:
                                                continue
                                except:
                                    pass
                            
                            # Note: Autopilot will be re-enabled in the restore endpoint
                            # after CarlaDataProvider is reconnected
                            logger.debug(f"Vehicle {vehicle_id} will be re-registered with TM in restore endpoint")
                            
                            # For now, just apply the last known control to keep vehicle state
                            control = carla.VehicleControl(
                                throttle=state.control['throttle'],
                                steer=state.control['steer'],
                                brake=state.control['brake'],
                                hand_brake=state.control['hand_brake'],
                                reverse=state.control['reverse'],
                                manual_gear_shift=state.control['manual_gear_shift'],
                                gear=state.control['gear']
                            )
                            vehicle.apply_control(control)
                else:
                    # DON'T spawn new vehicles - this causes duplicates!
                    # The existing vehicles should already be in the world
                    logger.warning(f"Vehicle {vehicle_id} not found in current world - skipping spawn to avoid duplicates")
                            
            except Exception as e:
                logger.warning(f"Failed to restore vehicle {vehicle_id}: {e}")
    
    def _restore_pedestrians(self, world):
        """Restore pedestrian states"""
        current_actors = world.get_actors().filter('walker.pedestrian.*')
        current_ids = {actor.id for actor in current_actors}
        snapshot_ids = set(self.pedestrians.keys())
        
        # Destroy pedestrians not in snapshot
        for actor in current_actors:
            if actor.id not in snapshot_ids:
                try:
                    actor.destroy()
                except:
                    pass
        
        # Update or spawn pedestrians
        for ped_id, state in self.pedestrians.items():
            try:
                if ped_id in current_ids:
                    # Update existing pedestrian
                    pedestrian = world.get_actor(ped_id)
                    if pedestrian:
                        transform = carla.Transform(
                            carla.Location(x=state.location['x'], y=state.location['y'], z=state.location['z']),
                            carla.Rotation(pitch=state.rotation['pitch'], yaw=state.rotation['yaw'], roll=state.rotation['roll'])
                        )
                        pedestrian.set_transform(transform)
                else:
                    # Spawn new pedestrian
                    blueprint_library = world.get_blueprint_library()
                    blueprint = blueprint_library.find(state.type_id)
                    
                    if blueprint:
                        transform = carla.Transform(
                            carla.Location(x=state.location['x'], y=state.location['y'], z=state.location['z']),
                            carla.Rotation(pitch=state.rotation['pitch'], yaw=state.rotation['yaw'], roll=state.rotation['roll'])
                        )
                        
                        pedestrian = world.try_spawn_actor(blueprint, transform)
                        
            except Exception as e:
                logger.warning(f"Failed to restore pedestrian {ped_id}: {e}")
    
    def _restore_traffic_lights(self, world):
        """Restore traffic light states"""
        actors = world.get_actors().filter('traffic.traffic_light')
        
        for light in actors:
            if light.id in self.traffic_lights:
                state = self.traffic_lights[light.id]
                try:
                    # Convert string back to enum
                    if state.state == "Red":
                        light.set_state(carla.TrafficLightState.Red)
                    elif state.state == "Yellow":
                        light.set_state(carla.TrafficLightState.Yellow)
                    elif state.state == "Green":
                        light.set_state(carla.TrafficLightState.Green)
                    elif state.state == "Off":
                        light.set_state(carla.TrafficLightState.Off)
                        
                except Exception as e:
                    logger.warning(f"Failed to restore traffic light {light.id}: {e}")
    
    def _restore_weather(self, world):
        """Restore weather parameters"""
        try:
            weather = carla.WeatherParameters(
                cloudiness=self.weather.cloudiness,
                precipitation=self.weather.precipitation,
                precipitation_deposits=self.weather.precipitation_deposits,
                wind_intensity=self.weather.wind_intensity,
                sun_azimuth_angle=self.weather.sun_azimuth_angle,
                sun_altitude_angle=self.weather.sun_altitude_angle,
                fog_density=self.weather.fog_density,
                fog_distance=self.weather.fog_distance,
                wetness=self.weather.wetness,
                fog_falloff=self.weather.fog_falloff,
                scattering_intensity=self.weather.scattering_intensity,
                mie_scattering_scale=self.weather.mie_scattering_scale,
                rayleigh_scattering_scale=self.weather.rayleigh_scattering_scale
            )
            world.set_weather(weather)
            
        except Exception as e:
            logger.warning(f"Failed to restore weather: {e}")
    
    def _restore_scenario(self, sim_state):
        """Restore scenario state"""
        try:
            if self.scenario and hasattr(sim_state, 'leaderboard_evaluator'):
                evaluator = sim_state.leaderboard_evaluator
                
                # Restore route waypoint index and scenario state
                if hasattr(evaluator, 'manager') and evaluator.manager:
                    manager = evaluator.manager
                    
                    if hasattr(manager, '_route_scenario'):
                        scenario = manager._route_scenario
                        
                        # Restore route index
                        if hasattr(scenario, 'route_index'):
                            scenario.route_index = self.scenario.current_waypoint_index
                            logger.info(f"Restored route index to {self.scenario.current_waypoint_index}")
                        
                        # Ensure other_actors list is populated
                        if hasattr(scenario, 'other_actors') and not scenario.other_actors:
                            scenario.other_actors = []
                        
                        # Reset any termination flags that might have been set
                        if hasattr(scenario, 'timeout'):
                            scenario.timeout = False
                    
                    # CRITICAL: Ensure the scenario manager is running
                    # The manager must be in RUNNING state to continue execution
                    if hasattr(manager, '_running'):
                        manager._running = True
                        logger.info("Set manager to running state in scenario restore")
                            
        except Exception as e:
            logger.warning(f"Failed to restore scenario state: {e}")
    
    def _restore_scenario_manager(self, sim_state):
        """Restore complete ScenarioManager state"""
        try:
            if self.scenario_manager and hasattr(sim_state, 'leaderboard_evaluator'):
                evaluator = sim_state.leaderboard_evaluator
                
                if hasattr(evaluator, 'manager') and evaluator.manager:
                    manager = evaluator.manager
                    state = self.scenario_manager
                    
                    # Restore all manager attributes
                    manager.route_index = state.route_index
                    manager.repetition_number = state.repetition_number
                    
                    # Restore timing state
                    manager.tick_count = state.tick_count
                    manager._timestamp_last_run = state.timestamp_last_run
                    manager.scenario_duration_system = state.scenario_duration_system
                    manager.scenario_duration_game = state.scenario_duration_game
                    manager.start_system_time = state.start_system_time
                    manager.start_game_time = state.start_game_time
                    manager.end_system_time = state.end_system_time
                    manager.end_game_time = state.end_game_time
                    
                    # Restore execution state - CRITICAL: Force running to True 
                    # to prevent scenario from resetting after restore
                    manager._running = True  # Always True to continue scenario
                    manager._debug_mode = state.debug_mode
                    manager._timeout = state.timeout
                    
                    # CRITICAL FIX: Ensure the scenario manager can process new inputs
                    # The manager might be blocked, so we need to reset its state
                    if hasattr(manager, '_watchdog'):
                        manager._watchdog = None
                    
                    # CRITICAL: Stop scenario building thread to prevent new scenarios
                    if hasattr(manager, '_scenario_thread'):
                        try:
                            if manager._scenario_thread and manager._scenario_thread.is_alive():
                                logger.warning("Scenario thread is still running, will not interfere")
                                # Don't kill it, but ensure it won't build new scenarios
                                if hasattr(manager, '_new_scenarios'):
                                    manager._new_scenarios = []
                        except:
                            pass
                    
                    # Restore vehicle references
                    if state.ego_vehicle_ids and hasattr(manager, 'ego_vehicles'):
                        # Update ego vehicle references with fresh actors
                        world = evaluator.world if hasattr(evaluator, 'world') else None
                        if world:
                            new_ego_vehicles = []
                            for ego_id in state.ego_vehicle_ids:
                                ego_actor = world.get_actor(ego_id)
                                if ego_actor:
                                    new_ego_vehicles.append(ego_actor)
                                    logger.info(f"Restored ego vehicle {ego_id}")
                            if new_ego_vehicles:
                                manager.ego_vehicles = new_ego_vehicles
                                
                                # Update agent's vehicle reference
                                if hasattr(manager, '_agent_wrapper') and manager._agent_wrapper:
                                    if hasattr(manager._agent_wrapper, '_agent'):
                                        agent = manager._agent_wrapper._agent
                                        if hasattr(agent, '_vehicle'):
                                            agent._vehicle = new_ego_vehicles[0]
                                            logger.info("Updated agent's vehicle reference")
                    
                    # Restore other actors
                    if state.other_actor_ids and hasattr(manager, 'other_actors'):
                        world = evaluator.world if hasattr(evaluator, 'world') else None
                        if world:
                            new_other_actors = []
                            for actor_id in state.other_actor_ids:
                                actor = world.get_actor(actor_id)
                                if actor:
                                    new_other_actors.append(actor)
                            manager.other_actors = new_other_actors
                    
                    # CRITICAL: Prevent scenario tree from resetting
                    if hasattr(manager, 'scenario_tree') and manager.scenario_tree:
                        try:
                            import py_trees
                            # Force the tree to RUNNING state to prevent reset
                            if hasattr(manager.scenario_tree, 'status'):
                                manager.scenario_tree.status = py_trees.common.Status.RUNNING
                                logger.info("Set scenario tree status to RUNNING to prevent reset")
                            
                            # Also reset any completion flags in the tree
                            if hasattr(manager.scenario_tree, 'root'):
                                root = manager.scenario_tree.root
                                if hasattr(root, 'status'):
                                    root.status = py_trees.common.Status.RUNNING
                                    
                                # Traverse and reset all child nodes to RUNNING
                                def reset_tree_status(node):
                                    if hasattr(node, 'status'):
                                        node.status = py_trees.common.Status.RUNNING
                                    if hasattr(node, 'children'):
                                        for child in node.children:
                                            reset_tree_status(child)
                                
                                reset_tree_status(root)
                                logger.info("Reset all scenario tree nodes to RUNNING")
                                
                        except Exception as e:
                            logger.debug(f"Could not set scenario tree status: {e}")
                    
                    # CRITICAL: Prevent scenario completion detection
                    # Reset any flags that might trigger scenario end
                    if hasattr(manager, 'scenario_class') and manager.scenario_class:
                        scenario = manager.scenario_class
                        if hasattr(scenario, 'timeout'):
                            # Reset timeout to prevent timeout-based completion
                            scenario.timeout = 999999
                        if hasattr(scenario, 'terminate_on_failure'):
                            scenario.terminate_on_failure = False
                    
                    # Restore scenario tree blackboard
                    if state.scenario_tree_blackboard:
                        try:
                            import py_trees
                            bb = py_trees.blackboard.Blackboard()
                            
                            # Restore AV_control if present
                            if 'AV_control' in state.scenario_tree_blackboard:
                                import carla
                                control_data = state.scenario_tree_blackboard['AV_control']
                                control = carla.VehicleControl(
                                    throttle=control_data.get('throttle', 0.0),
                                    steer=control_data.get('steer', 0.0),
                                    brake=control_data.get('brake', 0.0)
                                )
                                bb.set('AV_control', control, overwrite=True)
                        except:
                            pass
                    
                    # Restore agent wrapper wallclock
                    if state.agent_wallclock_t0 and hasattr(manager, '_agent_wrapper'):
                        wrapper = manager._agent_wrapper
                        if wrapper and hasattr(wrapper, '_agent'):
                            agent = wrapper._agent
                            if hasattr(agent, 'wallclock_t0'):
                                agent.wallclock_t0 = state.agent_wallclock_t0
                    
                    # CRITICAL: Ensure watchdogs are properly managed
                    # The watchdogs may need to be restarted or reset
                    if hasattr(manager, '_watchdog') and manager._watchdog:
                        manager._watchdog.update()
                    if hasattr(manager, '_agent_watchdog') and manager._agent_watchdog:
                        manager._agent_watchdog.update()
                    
                    logger.info(f"Restored scenario manager state: tick={state.tick_count}, running={state.running}, ego_vehicles={len(state.ego_vehicle_ids)}")
                    
                    # Debug output
                    print(f"DEBUG: Restored ScenarioManager state:")
                    print(f"  - tick_count: {manager.tick_count}")
                    print(f"  - _running: {manager._running}")
                    print(f"  - ego_vehicles: {manager.ego_vehicles if hasattr(manager, 'ego_vehicles') else 'None'}")
                    print(f"  - Has scenario_tree: {hasattr(manager, 'scenario_tree')}")
                    if hasattr(manager, 'ego_vehicles') and manager.ego_vehicles:
                        ego = manager.ego_vehicles[0]
                        print(f"  - Ego vehicle ID: {ego.id}")
                        print(f"  - Ego vehicle type: {type(ego)}")
                        print(f"  - Ego is valid actor: {ego.is_alive if hasattr(ego, 'is_alive') else 'Unknown'}")
                    
        except Exception as e:
            logger.warning(f"Failed to restore scenario manager state: {e}")
    
    def _restore_agent(self, sim_state):
        """Restore agent state"""
        try:
            if self.agent:
                sim_state.step_count = self.agent.step_count
                sim_state.last_action = copy.deepcopy(self.agent.last_action) if self.agent.last_action else None
                
                # Restore agent internal state if needed
                if hasattr(sim_state, 'agent_instance') and sim_state.agent_instance:
                    agent = sim_state.agent_instance
                    
                    # Restore step counter
                    if hasattr(agent, 'step'):
                        agent.step = self.agent.step_count
                    
                    # CRITICAL: Reset agent's control state
                    # This ensures the agent can receive new controls after restore
                    if hasattr(agent, '_last_control'):
                        agent._last_control = None
                    if hasattr(agent, 'control'):
                        agent.control = carla.VehicleControl()  # Reset to neutral
                        
        except Exception as e:
            logger.warning(f"Failed to restore agent state: {e}")
    
    def _restore_metrics(self, sim_state):
        """Restore simulation metrics"""
        try:
            if self.metrics:
                sim_state.step_count = self.metrics.step_count
                sim_state.cumulative_reward = self.metrics.cumulative_reward
                
        except Exception as e:
            logger.warning(f"Failed to restore metrics: {e}")
    
    def _restore_spectator(self, world):
        """Restore spectator position"""
        try:
            spectator = world.get_spectator()
            if spectator and self.spectator_transform:
                transform = carla.Transform(
                    carla.Location(
                        x=self.spectator_transform['location']['x'],
                        y=self.spectator_transform['location']['y'],
                        z=self.spectator_transform['location']['z']
                    ),
                    carla.Rotation(
                        pitch=self.spectator_transform['rotation']['pitch'],
                        yaw=self.spectator_transform['rotation']['yaw'],
                        roll=self.spectator_transform['rotation']['roll']
                    )
                )
                spectator.set_transform(transform)
                
        except Exception as e:
            logger.warning(f"Failed to restore spectator: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for serialization"""
        return {
            'snapshot_id': self.snapshot_id,
            'timestamp': self.timestamp,
            'map_name': self.map_name,
            'vehicles': {k: asdict(v) for k, v in self.vehicles.items()},
            'pedestrians': {k: asdict(v) for k, v in self.pedestrians.items()},
            'traffic_lights': {k: asdict(v) for k, v in self.traffic_lights.items()},
            'weather': asdict(self.weather) if self.weather else None,
            'scenario': asdict(self.scenario) if self.scenario else None,
            'scenario_manager': asdict(self.scenario_manager) if self.scenario_manager else None,
            'agent': asdict(self.agent) if self.agent else None,
            'metrics': asdict(self.metrics) if self.metrics else None,
            'spectator_transform': self.spectator_transform
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorldSnapshot':
        """Create snapshot from dictionary"""
        snapshot = cls(snapshot_id=data['snapshot_id'])
        snapshot.timestamp = data['timestamp']
        snapshot.map_name = data.get('map_name')
        snapshot.spectator_transform = data.get('spectator_transform')
        
        # Restore vehicles
        for k, v in data.get('vehicles', {}).items():
            snapshot.vehicles[int(k)] = VehicleState(**v)
        
        # Restore pedestrians
        for k, v in data.get('pedestrians', {}).items():
            snapshot.pedestrians[int(k)] = PedestrianState(**v)
        
        # Restore traffic lights
        for k, v in data.get('traffic_lights', {}).items():
            snapshot.traffic_lights[int(k)] = TrafficLightState(**v)
        
        # Restore weather
        if data.get('weather'):
            snapshot.weather = WeatherState(**data['weather'])
        
        # Restore scenario
        if data.get('scenario'):
            snapshot.scenario = ScenarioState(**data['scenario'])
        
        # Restore scenario manager
        if data.get('scenario_manager'):
            snapshot.scenario_manager = ScenarioManagerState(**data['scenario_manager'])
        
        # Restore agent
        if data.get('agent'):
            snapshot.agent = AgentState(**data['agent'])
        
        # Restore metrics
        if data.get('metrics'):
            snapshot.metrics = SimulationMetrics(**data['metrics'])
        
        return snapshot