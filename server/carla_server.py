#!/usr/bin/env python3
"""
Complete CARLA Server Implementation for Bench2Drive Gymnasium Environment

This server integrates:
- CARLA simulator connection and management
- Bench2Drive building blocks from baseline_v3
- REST API endpoints for Gymnasium interface
- Snapshot/restore support for GRPO
- Multi-instance support for parallel training

Uses real CARLA instances - NO MOCKING
"""

import os
import sys
import base64
import traceback
import time
import signal
import subprocess
import atexit
import threading
import socket
import importlib
import argparse
import json
import uuid
import logging
import pickle
import yaml
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import io

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import resource manager for modular GPU and port allocation
try:
    from resource_manager import get_resource_manager, ServiceResources
except ImportError:
    # Fallback if resource_manager not available
    get_resource_manager = None
    ServiceResources = None

### Never start this file. called it via the microservice_manager.py

# Shared snapshots directory for all services
SNAPSHOTS_DIR = Path("/mnt3/Documents/AD_Framework/bench2drive-gymnasium/bench2drive_microservices/snapshots")
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Setup comprehensive logging
def setup_logging(api_port, log_to_file=True, log_to_console=True):
    """Setup detailed logging to both file and console"""
    log_dir = Path("/mnt3/Documents/AD_Framework/bench2drive-gymnasium/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []  # Clear existing handlers
    
    if log_to_file:
        # File handler - overwrites on each run
        file_handler = logging.FileHandler(
            log_dir / f"carla_server_{api_port}.log",
            mode='w'  # Overwrite mode
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
    
    if log_to_console:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(console_handler)
    
    # Setup specific loggers
    for name in ['carla_server', 'uvicorn', 'fastapi']:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
    
    return logging.getLogger('carla_server')

# Initialize logger
logger = setup_logging(8080, log_to_file=False, log_to_console=True)

# Load configuration from config.yaml
def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}

# Setup Bench2Drive environment and paths
def setup_bench2drive_environment():
    """Setup environment variables and paths for Bench2Drive"""
    
    # Load configuration
    config = load_config()
    
    # Get paths from config or use defaults
    paths_config = config.get('paths', {})
    bench2drive_root = paths_config.get('bench2drive_root', '/mnt3/Documents/AD_Framework/Bench2Drive')
    leaderboard_path = paths_config.get('leaderboard_path', f'{bench2drive_root}/leaderboard')
    leaderboard_module_path = paths_config.get('leaderboard_module_path', f'{bench2drive_root}/leaderboard/leaderboard')
    scenario_runner_path = paths_config.get('scenario_runner_path', f'{bench2drive_root}/scenario_runner')
    carla_root = paths_config.get('carla_root', '/mnt3/Documents/AD_Framework/carla0915')
    
    # Set environment variables from config
    env_config = config.get('environment', {})
    os.environ['IS_BENCH2DRIVE'] = env_config.get('IS_BENCH2DRIVE', 'true')
    os.environ['REPETITION'] = env_config.get('REPETITION', '1')
    os.environ['TMP_VISU'] = env_config.get('TMP_VISU', '0')  # Enable camera sensors
    
    # Add Bench2Drive paths to sys.path
    paths_to_add = [
        leaderboard_path,
        leaderboard_module_path,
        scenario_runner_path,
    ]
    
    for path in paths_to_add:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            logger.debug(f"Added to sys.path: {path}")
    
    # Set WORK_DIR if not set
    if 'WORK_DIR' not in os.environ:
        os.environ['WORK_DIR'] = bench2drive_root
        
    # Set CARLA_ROOT if not set
    if 'CARLA_ROOT' not in os.environ:
        os.environ['CARLA_ROOT'] = carla_root
        
    logger.info("Bench2Drive environment setup complete")

# Setup the environment first
setup_bench2drive_environment()

# Now import CARLA and Bench2Drive components
try:
    import carla
    logger.debug("Imported carla")
    
    # Import Bench2Drive leaderboard components
    from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
    from leaderboard.utils.statistics_manager import StatisticsManager, FAILURE_MESSAGES
    from leaderboard.utils.route_indexer import RouteIndexer
    from leaderboard.scenarios.scenario_manager import ScenarioManager
    from leaderboard.scenarios.route_scenario import RouteScenario
    from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
    from leaderboard.autoagents.agent_wrapper import AgentError, validate_sensor_configuration, TickRuntimeError
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    from srunner.scenariomanager.timer import GameTime
    from srunner.scenariomanager.watchdog import Watchdog
    
    logger.info("All CARLA and Bench2Drive imports successful")
    
except Exception as e:
    logger.error(f"CARLA/Bench2Drive import failed: {e}")
    logger.error(traceback.format_exc())
    raise

# Import API agent using config paths
import sys
from pathlib import Path
import yaml

# Load config to get proper paths
config_paths = [
    Path(__file__).parent.parent / "configs" / "config.yaml",
    Path(__file__).parent.parent / "config.yaml",
    Path(__file__).parent / "config.yaml",
    Path("/mnt3/Documents/AD_Framework/bench2drive-gymnasium/bench2drive_microservices/configs/config.yaml")
]

config = None
for config_path in config_paths:
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        break

if config and 'paths' in config:
    working_dir = Path(config['paths']['working_dir'])
    client_dir = working_dir / config['paths'].get('client_dir', 'client')
    sys.path.insert(0, str(working_dir))
    sys.path.insert(0, str(client_dir))
else:
    # Fallback to hardcoded path if no config
    working_dir = Path("/mnt3/Documents/AD_Framework/bench2drive-gymnasium/bench2drive_microservices")
    sys.path.insert(0, str(working_dir))
    sys.path.insert(0, str(working_dir / "client"))

from api_agent import APIAgent, get_entry_point

# Building blocks from baseline_v3 (extracted from notebook)

def find_free_port(starting_port):
    """Find a free port starting from the given port"""
    port = starting_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            port += 1

def is_port_free(port):
    """Check if a port is free"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            return True
    except OSError:
        return False

import fcntl  # Add this import at the top of the file

def _setup_simulation(self, args):
    """Setup simulation with CARLA server"""
    logger.info(f"Attempting to setup CARLA on port {args.port}")
    
    # First, try to connect to existing CARLA instance
    try:
        import carla
        test_client = carla.Client('localhost', args.port)
        test_client.set_timeout(2.0)
        test_world = test_client.get_world()
        logger.info(f"Successfully connected to existing CARLA on port {args.port}")
        self.client = test_client
        self.world = test_world
        self.traffic_manager = test_client.get_trafficmanager(args.traffic_manager_port)
        return
    except Exception as e:
        logger.info(f"No existing CARLA on port {args.port}, will start new instance: {e}")
    
    # Find free port to avoid conflicts
    original_port = args.port
    max_attempts = 20
    for attempt in range(max_attempts):
        if is_port_free(args.port):
            logger.info(f"Port {args.port} is free, using it")
            break
        else:
            logger.info(f"Port {args.port} is in use, trying next...")
            args.port += 2  # Skip by 2 to avoid related ports
    else:
        raise RuntimeError(f"Could not find free port after {max_attempts} attempts starting from {original_port}")
    
    # Start CARLA server
    self.carla_path = os.environ["CARLA_ROOT"]
    
    # Use resource manager for modular GPU and port allocation
    service_num = 0
    if hasattr(sim_state, 'server_id') and 'service-' in sim_state.server_id:
        # Extract service number from server_id (e.g., "service-1" -> 1)
        service_num = int(sim_state.server_id.split('-')[-1])
    
    # Get resource allocation from manager (if available)
    if get_resource_manager:
        resource_mgr = get_resource_manager("config.yaml")
        resources = resource_mgr.allocate_resources(service_num)
        
        # Build CARLA command with allocated resources
        carla_args = resource_mgr.get_carla_command_args(service_num)
        cmd = f"{os.path.join(self.carla_path, 'CarlaUE4.sh')} {' '.join(carla_args)}"
        
        logger.info(f"[Resource Manager] Starting CARLA with GPU {resources.gpu_id}, "
                   f"RPC port {resources.carla_port}, "
                   f"streaming port {resources.streaming_port}")
    else:
        # Fallback to hardcoded configuration
        gpu_id = 0  # Default to GPU 0 for all services
        streaming_port = 3000 + (service_num * 10)
        
        cmd = f"{os.path.join(self.carla_path, 'CarlaUE4.sh')} -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port={args.port} -carla-streaming-port={streaming_port} -graphicsadapter={gpu_id}"
        
        logger.info(f"[Fallback] Starting CARLA with GPU {gpu_id}, RPC port {args.port}, streaming port {streaming_port}")
    self.server = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    atexit.register(os.killpg, self.server.pid, signal.SIGKILL)

    # Wait for server to be ready
    server_ready = False
    for i in range(30):
        try:
            with socket.create_connection(('localhost', args.port), timeout=1):
                server_ready = True
                break
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(5)
            logger.debug(f"Waiting for CARLA server... attempt {i+1}/30")
    
    if not server_ready:
        raise RuntimeError("CARLA server failed to start")
    
    # Initialize CARLA client and world
    attempts = 0
    num_max_restarts = 20
    logger.info("Loading world...")
    while attempts < num_max_restarts:
        try:
            client = carla.Client(args.host, args.port)
            client_timeout = args.timeout if args.timeout else 60.0
            client.set_timeout(client_timeout)

            settings = carla.WorldSettings(
                synchronous_mode=True,
                fixed_delta_seconds=1.0 / self.frame_rate,
                deterministic_ragdolls=True,
                spectator_as_ego=False
            )
            client.get_world().apply_settings(settings)
            logger.info(f"World loaded successfully, attempts={attempts}")
            break
        except Exception as e:
            logger.warning(f"World load failed, attempts={attempts}: {e}")
            attempts += 1
            time.sleep(5)
    
    # Initialize traffic manager
    attempts = 0
    num_max_restarts = 40
    while attempts < num_max_restarts:
        try:
            args.traffic_manager_port = find_free_port(args.traffic_manager_port)
            traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_hybrid_physics_mode(True)
            logger.info(f"Traffic manager initialized successfully, attempts={attempts}")
            break
        except Exception as e:
            logger.warning(f"Traffic manager init failed, attempts={attempts}: {e}")
            attempts += 1
            time.sleep(5)

    return client, client_timeout, traffic_manager

def get_weather_id(weather_conditions):
    """Get weather ID from weather conditions"""
    from xml.etree import ElementTree as ET
    WORK_DIR = os.environ.get("WORK_DIR", "")
    if WORK_DIR != "":
        WORK_DIR += "/"
    tree = ET.parse(WORK_DIR + 'leaderboard/data/weather.xml')
    root = tree.getroot()
    
    def conditions_match(weather, conditions):
        for (key, value) in weather:
            if key == 'route_percentage':
                continue
            if str(getattr(conditions, key)) != value:
                return False
        return True
    
    for case in root.findall('case'):
        weather = case[0].items()
        if conditions_match(weather, weather_conditions):
            return case.items()[0][1]
    return None

sensors_to_icons = {
    'sensor.camera.rgb': 'carla_camera',
    'sensor.lidar.ray_cast': 'carla_lidar',
    'sensor.other.radar': 'carla_radar',
    'sensor.other.gnss': 'carla_gnss',
    'sensor.other.imu': 'carla_imu',
    'sensor.opendrive_map': 'carla_opendrive_map',
    'sensor.speedometer': 'carla_speedometer'
}

def _load_and_wait_for_world(leaderboard_evaluator_self, args, town):
    """Load the specified town in CARLA"""
    try:
        # Make sure we have a world object
        if not hasattr(leaderboard_evaluator_self, 'world') or leaderboard_evaluator_self.world is None:
            if hasattr(leaderboard_evaluator_self, 'client') and leaderboard_evaluator_self.client:
                leaderboard_evaluator_self.world = leaderboard_evaluator_self.client.get_world()
                logger.info("Initialized world from client")
            else:
                raise RuntimeError("No client available to get world from")
        
        # Check if we need to reload the world
        current_map = leaderboard_evaluator_self.world.get_map()
        current_map_name = current_map.name.split('/')[-1]  # Get just the town name
        
        if current_map_name != town:
            logger.info(f"Current map: {current_map_name}, loading new town: {town}")
            leaderboard_evaluator_self.world = leaderboard_evaluator_self.client.load_world(town)
            leaderboard_evaluator_self.world.tick()
            logger.info(f"Town {town} loaded successfully")
        else:
            logger.info(f"Town {town} already loaded, skipping reload")
        
        # Apply world settings (always, regardless of reload)
        settings = leaderboard_evaluator_self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / leaderboard_evaluator_self.frame_rate
        leaderboard_evaluator_self.world.apply_settings(settings)
            
        # Set traffic manager
        leaderboard_evaluator_self.traffic_manager.set_synchronous_mode(True)
        leaderboard_evaluator_self.traffic_manager.set_hybrid_physics_mode(True)
        
        # Initialize CarlaDataProvider
        CarlaDataProvider.set_client(leaderboard_evaluator_self.client)
        CarlaDataProvider.set_world(leaderboard_evaluator_self.world)
        CarlaDataProvider.set_traffic_manager_port(leaderboard_evaluator_self.traffic_manager.get_port())
        
    except Exception as e:
        logger.error(f"Error loading world: {e}")
        raise

# Building block functions from baseline_v3

def my_load_scenario(leaderboard_evaluator_self, args, config):
    """Load and run the scenario given by config"""
    logger.info("my_load_scenario: Starting function")
    crash_message = ""
    entry_status = "Started"
    save_name = ""

    logger.info(f"========= Preparing {config.name} (repetition {config.repetition_index}) =========")

    # Prepare the statistics of the route
    route_name = f"{config.name}_rep{config.repetition_index}"
    scenario_name = config.scenario_configs[0].name if config.scenario_configs else "default"
    town_name = str(config.town)
    weather_id = get_weather_id(config.weather[0][1]) if config.weather else None
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%m_%d_%H_%M_%S")
    save_name = f"{route_name}_{town_name}_{scenario_name}_{weather_id}_{currentTime}"
    leaderboard_evaluator_self.statistics_manager.create_route_data(route_name, scenario_name, weather_id, save_name, town_name, config.index)

    logger.info("Loading the world")

    # Load the world and the scenario
    try:
        logger.info(f"my_load_scenario: Calling _load_and_wait_for_world for town {config.town}")
        _load_and_wait_for_world(leaderboard_evaluator_self, args, config.town)
        logger.info("my_load_scenario: Creating RouteScenario")
        leaderboard_evaluator_self.route_scenario = RouteScenario(world=leaderboard_evaluator_self.world, config=config, debug_mode=args.debug)
        logger.info("my_load_scenario: Setting scenario in statistics_manager")
        leaderboard_evaluator_self.statistics_manager.set_scenario(leaderboard_evaluator_self.route_scenario)

    except Exception as e:
        # The scenario is wrong -> set the execution to crashed and stop
        logger.error(f"The scenario could not be loaded: {e}")
        logger.error(traceback.format_exc())

        entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
        _register_statistics(leaderboard_evaluator_self, config.index, entry_status, crash_message)
        _cleanup(leaderboard_evaluator_self)
        return True, save_name, entry_status, crash_message

    return False, save_name, entry_status, crash_message

# Dummy watchdog for APIAgent that does nothing
class DummyWatchdog:
    """A no-op watchdog for API-controlled agents"""
    def __init__(self, timeout):
        self.timeout = timeout
        self._status = True
    
    def start(self):
        pass
    
    def stop(self):
        pass
    
    def pause(self):
        pass
    
    def resume(self):
        pass
    
    def update(self):
        pass
    
    def get_status(self):
        return self._status

def my_load_agent(leaderboard_evaluator_self, args, config, save_name, entry_status, crash_message):
    """Set up the user's agent and configure sensors"""
    logger.info("Setting up the agent")

    # Set up the user's agent, and the timer to avoid freezing the simulation
    try:
        # Import the agent module if not already done
        if not hasattr(leaderboard_evaluator_self, 'module_agent') or leaderboard_evaluator_self.module_agent is None:
            module_name = os.path.basename(args.agent).split('.')[0]
            sys.path.insert(0, os.path.dirname(args.agent))
            leaderboard_evaluator_self.module_agent = importlib.import_module(module_name)
            logger.info(f"Loaded agent module: {module_name}")
        
        # Get agent class info first
        agent_class_name = getattr(leaderboard_evaluator_self.module_agent, 'get_entry_point')()
        agent_class_obj = getattr(leaderboard_evaluator_self.module_agent, agent_class_name)
        
        # For APIAgent, use a dummy watchdog during setup
        if agent_class_name == 'APIAgent':
            logger.info("APIAgent detected - using dummy watchdog for setup")
            leaderboard_evaluator_self._agent_watchdog = DummyWatchdog(args.timeout)
        else:
            leaderboard_evaluator_self._agent_watchdog = Watchdog(args.timeout)
        leaderboard_evaluator_self._agent_watchdog.start()

        # Start the ROS1 bridge server only for ROS1 based agents.
        if getattr(agent_class_obj, 'get_ros_version')() == 1 and leaderboard_evaluator_self._ros1_server is None:
            from leaderboard.autoagents.ros1_agent import ROS1Server
            leaderboard_evaluator_self._ros1_server = ROS1Server()
            leaderboard_evaluator_self._ros1_server.start()

        leaderboard_evaluator_self.agent_instance = agent_class_obj(args.host, args.port, args.debug)
        leaderboard_evaluator_self.agent_instance.set_global_plan(leaderboard_evaluator_self.route_scenario.gps_route, leaderboard_evaluator_self.route_scenario.route)
        args.agent_config = args.agent_config + '+' + save_name
        leaderboard_evaluator_self.agent_instance.setup(args.agent_config)

        # Check and store the sensors
        if not leaderboard_evaluator_self.sensors:
            leaderboard_evaluator_self.sensors = leaderboard_evaluator_self.agent_instance.sensors()
            track = leaderboard_evaluator_self.agent_instance.track

            validate_sensor_configuration(leaderboard_evaluator_self.sensors, track, args.track)

            leaderboard_evaluator_self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in leaderboard_evaluator_self.sensors]
            leaderboard_evaluator_self.statistics_manager.save_sensors(leaderboard_evaluator_self.sensor_icons)
            leaderboard_evaluator_self.statistics_manager.write_statistics()

            leaderboard_evaluator_self.sensors_initialized = True

        leaderboard_evaluator_self._agent_watchdog.stop()
        leaderboard_evaluator_self._agent_watchdog = None

    except SensorConfigurationInvalid as e:
        # The sensors are invalid -> set the execution to rejected and stop
        logger.error("The sensor's configuration used is invalid:")
        logger.error(str(e))

        entry_status, crash_message = FAILURE_MESSAGES["Sensors"]
        _register_statistics(leaderboard_evaluator_self, config.index, entry_status, crash_message)
        _cleanup(leaderboard_evaluator_self)
        return True, entry_status, crash_message

    except Exception as e:
        # The agent setup has failed -> start the next route
        logger.error("Could not set up the required agent:")
        logger.error(traceback.format_exc())
        logger.error(str(e))

        entry_status, crash_message = FAILURE_MESSAGES["Agent_init"]
        _register_statistics(leaderboard_evaluator_self, config.index, entry_status, crash_message)
        _cleanup(leaderboard_evaluator_self)
        return True, entry_status, crash_message

    return False, entry_status, crash_message

def my_load_route_scenario(leaderboard_evaluator_self, args, config, entry_status, crash_message):
    """Load the route scenario and prepare for execution"""
    logger.info("Running the route")

    # Run the scenario
    try:
        # Initialize ScenarioManager if not exists
        if not hasattr(leaderboard_evaluator_self, 'manager') or leaderboard_evaluator_self.manager is None:
            leaderboard_evaluator_self.manager = ScenarioManager(
                args.timeout,
                leaderboard_evaluator_self.statistics_manager,
                debug_mode=args.debug > 0
            )
            logger.info("Initialized ScenarioManager")
        
        # Load scenario and run it
        if args.record:
            leaderboard_evaluator_self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
        leaderboard_evaluator_self.manager.load_scenario(leaderboard_evaluator_self.route_scenario, leaderboard_evaluator_self.agent_instance, config.index, config.repetition_index)
    except Exception:
        logger.error("Failed to load the scenario, the statistics might be empty:")
        logger.error("Loading the route, the agent has crashed:")
        entry_status, crash_message = FAILURE_MESSAGES["Agent_runtime"]

    return False, entry_status, crash_message

def my_run_scenario_setup(leaderboard_evaluator_self):
    """Trigger the start of the scenario and wait for it to finish/fail"""
    leaderboard_evaluator_self.manager.tick_count = 0
    leaderboard_evaluator_self.manager.start_system_time = time.time()
    leaderboard_evaluator_self.manager.start_game_time = GameTime.get_time()

    # Detects if the simulation is down
    leaderboard_evaluator_self.manager._watchdog = Watchdog(leaderboard_evaluator_self.manager._timeout)
    leaderboard_evaluator_self.manager._watchdog.start()

    # Stop the agent from freezing the simulation
    # For APIAgent, we use a dummy watchdog since it's controlled externally via REST API
    agent_name = leaderboard_evaluator_self.agent_instance.__class__.__name__
    if agent_name == 'APIAgent':
        logger.info("APIAgent detected - using dummy agent watchdog (controlled via REST API)")
        leaderboard_evaluator_self.manager._agent_watchdog = DummyWatchdog(leaderboard_evaluator_self.manager._timeout)
    else:
        leaderboard_evaluator_self.manager._agent_watchdog = Watchdog(leaderboard_evaluator_self.manager._timeout)
    leaderboard_evaluator_self.manager._agent_watchdog.start()

    leaderboard_evaluator_self.manager._running = True

    # Thread for build_scenarios
    leaderboard_evaluator_self.manager._scenario_thread = threading.Thread(target=leaderboard_evaluator_self.manager.build_scenarios_loop, args=(leaderboard_evaluator_self.manager._debug_mode > 0, ))
    leaderboard_evaluator_self.manager._scenario_thread.start()

def my_run_scenario_step(leaderboard_evaluator_self, entry_status, crash_message, n_steps=1):
    """Run n_steps of the scenario simulation"""
    # Initialize return values in case they're not set
    if not entry_status:
        entry_status = "Started"
    if not crash_message:
        crash_message = ""
    
    try:
        # Note: Check for just_restored flag if available
        # This would typically be passed through the evaluator or manager
        # For now, we'll check if the manager has a flag set
        if hasattr(leaderboard_evaluator_self, 'just_restored') and leaderboard_evaluator_self.just_restored:
            logger.warning("Skipping ALL ticks after restore (no scenario tick, no world tick)")
            leaderboard_evaluator_self.just_restored = False  # Clear flag
            # Don't tick at all! Just return
            return False, entry_status, crash_message
        
        # Debug: Check manager state
        if hasattr(leaderboard_evaluator_self, 'manager'):
            manager = leaderboard_evaluator_self.manager
            if not hasattr(manager, '_running'):
                logger.warning("Manager has no _running attribute!")
            elif not manager._running:
                logger.warning(f"Manager._running is False! Forcing it to True for scenario continuation")
                manager._running = True
        
        if leaderboard_evaluator_self.manager._running:
            logger.debug("Running scenario step")
            for _ in range(n_steps):
                leaderboard_evaluator_self.manager._tick_scenario()
        else:
            logger.error("Manager is not running - scenario step skipped!")

    except AgentError:
        # The agent has failed -> stop the route
        logger.error("Stopping the route, the agent has crashed:")
        logger.error(traceback.format_exc())

        entry_status, crash_message = FAILURE_MESSAGES["Agent_runtime"]
        return True, entry_status, crash_message

    except KeyboardInterrupt:
        return True, entry_status, crash_message
    
    except TickRuntimeError:
        entry_status, crash_message = "Started", "TickRuntime"
        return True, entry_status, crash_message
    
    except Exception:
        logger.error("Error during the simulation:")
        logger.error(traceback.format_exc())

        entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
        return True, entry_status, crash_message
    
    return False, entry_status, crash_message

def _register_statistics(leaderboard_evaluator_self, route_index, entry_status, crash_message):
    """Register statistics for the current route"""
    try:
        if hasattr(leaderboard_evaluator_self, 'statistics_manager'):
            # Make sure the route is created before computing statistics
            if hasattr(leaderboard_evaluator_self.statistics_manager, '_results'):
                if hasattr(leaderboard_evaluator_self.statistics_manager._results, 'checkpoint'):
                    if hasattr(leaderboard_evaluator_self.statistics_manager._results.checkpoint, 'records'):
                        if route_index < len(leaderboard_evaluator_self.statistics_manager._results.checkpoint.records):
                            leaderboard_evaluator_self.statistics_manager.compute_route_statistics(
                                route_index,
                                entry_status,
                                crash_message
                            )
                        else:
                            logger.warning(f"Route index {route_index} not in records, skipping statistics computation")
            
            # Always try to write statistics
            leaderboard_evaluator_self.statistics_manager.write_statistics()
    except Exception as e:
        logger.error(f"Error registering statistics: {e}")
        traceback.print_exc()

def _cleanup(leaderboard_evaluator_self):
    """Cleanup after scenario"""
    try:
        # Cleanup CarlaDataProvider
        CarlaDataProvider.cleanup()
        
        # Cleanup scenario if exists
        if hasattr(leaderboard_evaluator_self, 'route_scenario'):
            leaderboard_evaluator_self.route_scenario = None
            
        # Tick world to update
        if hasattr(leaderboard_evaluator_self, 'world'):
            leaderboard_evaluator_self.world.tick()
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def my_stop_scenario(leaderboard_evaluator_self, args, config, entry_status, crash_message):
    """Stop the scenario and register statistics"""
    # Stop the scenario
    try:
        logger.info("Stopping the route")
        if hasattr(leaderboard_evaluator_self, 'manager') and leaderboard_evaluator_self.manager:
            leaderboard_evaluator_self.manager.stop_scenario()
        _register_statistics(leaderboard_evaluator_self, config.index, entry_status, crash_message)

        if args.record:
            leaderboard_evaluator_self.client.stop_recorder()

        _cleanup(leaderboard_evaluator_self)

    except Exception:
        logger.error("Failed to stop the scenario, the statistics might be empty:")
        logger.error(traceback.format_exc())

        _, crash_message = FAILURE_MESSAGES["Simulation"]

    return False, entry_status, crash_message

# Pydantic models for request/response
class SeedRequest(BaseModel):
    seed: int

class ResetRequest(BaseModel):
    route_id: Optional[int] = 0
    weather: Optional[str] = "ClearNoon"
    traffic_density: Optional[float] = 0.5

class StepRequest(BaseModel):
    action: Dict[str, float]  # throttle, brake, steer
    n_steps: Optional[int] = 1

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

class SpecResponse(BaseModel):
    action_space: Dict[str, Any]
    observation_space: Dict[str, Any]

class SnapshotRequest(BaseModel):
    snapshot_id: Optional[str] = None

class RestoreRequest(BaseModel):
    snapshot_id: str

# Import snapshot system
try:
    from world_snapshot import WorldSnapshot
except ImportError:
    WorldSnapshot = None
    logger.warning("WorldSnapshot not available - snapshot/restore features disabled")

# Global state
class SimulationState:
    def __init__(self):
        self.leaderboard_evaluator = None
        self.route_indexer = None
        self.agent_instance = None
        self.args = None
        self.config = None
        self.current_config = None  # For quick reset check
        self.manager_scenario = None  # For quick reset check
        self.statistics_manager = None
        self.current_route = None
        self.step_count = 0
        self.max_steps = 1000
        self.seed = 0
        self.last_observation = None
        self.last_action = None
        self.cumulative_reward = 0.0
        self.entry_status = "Started"
        self.crash_message = ""
        self.snapshots = {}  # Store world snapshots for GRPO
        self.server_id = str(uuid.uuid4())[:8]  # Unique server ID
        
        # Setup snapshot directory
        self.snapshot_dir = Path("./snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Snapshot directory created/verified at: {self.snapshot_dir.absolute()}")
        self.carla_port = None
        self.api_port = None
        
    def reset(self):
        """Reset internal state"""
        if self.leaderboard_evaluator:
            try:
                my_stop_scenario(
                    self.leaderboard_evaluator,
                    self.args,
                    self.config,
                    self.entry_status,
                    self.crash_message
                )
            except:
                pass
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.last_observation = None
        self.last_action = None
        self.entry_status = "Started"
        self.crash_message = ""

def encode_image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded JPEG string"""
    if image_array is None:
        return ""
    
    # Convert to PIL Image
    if len(image_array.shape) == 3:
        if image_array.shape[2] == 4:  # RGBA
            image_array = image_array[:, :, :3]  # Remove alpha channel
    
    img = Image.fromarray(image_array.astype('uint8'))
    
    # Compress as JPEG
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    
    # Encode to base64
    return base64.b64encode(buffer.read()).decode('utf-8')

def get_observation_from_state(evaluator) -> Dict[str, Any]:
    """Extract observation from current simulation state"""
    obs = {
        "images": {},
        "vehicle_state": {},
        "scenario_info": {},
        "step": sim_state.step_count,
        "debug_info": {}  # Add debug info for tracking other vehicles
    }
    
    try:
        # Get sensor data from the API Agent
        if hasattr(evaluator, 'agent_instance') and hasattr(evaluator.agent_instance, 'get_last_sensor_data'):
            sensor_data = evaluator.agent_instance.get_last_sensor_data()
            
            if sensor_data and isinstance(sensor_data, dict):
                # Extract camera images using the same format as dummy_agent3
                for camera_id in ['Center', 'Left', 'Right']:
                    if camera_id in sensor_data:
                        try:
                            # Get RGB image from tuple (timestamp, image_array)
                            camera_tuple = sensor_data[camera_id]
                            if isinstance(camera_tuple, tuple) and len(camera_tuple) >= 2:
                                rgb_image = camera_tuple[1][:, :, :3]  # Get image array and remove alpha
                                obs["images"][camera_id.lower()] = encode_image_to_base64(rgb_image)
                                if sim_state.step_count < 3:
                                    logger.debug(f"Captured {camera_id} camera image: {rgb_image.shape}")
                        except Exception as e:
                            if sim_state.step_count < 3:
                                logger.error(f"Error processing {camera_id} camera: {e}")
                
                # Extract speed from speedometer
                if 'SPEED' in sensor_data:
                    try:
                        speed_tuple = sensor_data['SPEED']
                        if isinstance(speed_tuple, tuple) and len(speed_tuple) >= 2:
                            speed_data = speed_tuple[1]
                            if isinstance(speed_data, dict):
                                obs["vehicle_state"]["speed"] = float(speed_data.get('speed', 0))
                            elif isinstance(speed_data, (int, float)):
                                obs["vehicle_state"]["speed"] = float(speed_data)
                    except Exception as e:
                        logger.error(f"Error processing speed: {e}")
        
        # Get vehicle state
        ego_vehicle = CarlaDataProvider.get_hero_actor()
        if ego_vehicle:
            transform = ego_vehicle.get_transform()
            velocity = ego_vehicle.get_velocity()
            
            obs["vehicle_state"] = {
                "position": {
                    "x": transform.location.x,
                    "y": transform.location.y,
                    "z": transform.location.z
                },
                "rotation": {
                    "pitch": transform.rotation.pitch,
                    "yaw": transform.rotation.yaw,
                    "roll": transform.rotation.roll
                },
                "velocity": {
                    "x": velocity.x,
                    "y": velocity.y,
                    "z": velocity.z
                },
                "speed": np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            }
            
            # Track other vehicles for debugging
            try:
                if hasattr(evaluator, 'world'):
                    world = evaluator.world
                    vehicles = world.get_actors().filter('vehicle.*')
                    other_vehicles = []
                    for v in vehicles:
                        if v.id != ego_vehicle.id:
                            v_transform = v.get_transform()
                            vehicle_info = {
                                'id': v.id,
                                'x': v_transform.location.x,
                                'y': v_transform.location.y,
                                'z': v_transform.location.z,
                                'type_id': v.type_id
                            }
                            
                            # Try to get vehicle color from attributes
                            try:
                                if hasattr(v, 'attributes'):
                                    color_attr = v.attributes.get('color', None)
                                    if color_attr:
                                        # Parse color string like "(R,G,B)"
                                        color_vals = color_attr.strip('()').split(',')
                                        if len(color_vals) == 3:
                                            vehicle_info['color'] = {
                                                'r': int(color_vals[0]),
                                                'g': int(color_vals[1]),
                                                'b': int(color_vals[2])
                                            }
                            except:
                                pass  # Color extraction failed, skip it
                                
                            other_vehicles.append(vehicle_info)
                    # Sort by distance to ego
                    ego_x = transform.location.x
                    ego_y = transform.location.y
                    other_vehicles.sort(key=lambda v: (v['x']-ego_x)**2 + (v['y']-ego_y)**2)
                    # Keep only closest 5 vehicles
                    obs["debug_info"]["other_vehicles"] = other_vehicles[:5]
                    obs["debug_info"]["total_vehicles"] = len(other_vehicles)
            except Exception as e:
                logger.debug(f"Could not track other vehicles: {e}")
        
        # Get scenario info
        obs["scenario_info"] = {
            "route_id": sim_state.current_route,
            "step_count": sim_state.step_count,
            "max_steps": sim_state.max_steps
        }
        
    except Exception as e:
        logger.error(f"Error getting observation: {e}")
        traceback.print_exc()
    
    return obs

def calculate_reward(evaluator, action: Dict[str, float], info: Dict[str, Any]) -> float:
    """Calculate reward based on current state and action"""
    reward = 0.0
    
    try:
        # Base reward for survival
        reward += 0.1
        
        # Speed reward (encourage forward progress)
        if "vehicle_state" in sim_state.last_observation:
            speed = sim_state.last_observation["vehicle_state"].get("speed", 0)
            target_speed = 30.0  # m/s (~108 km/h)
            speed_reward = min(speed / target_speed, 1.0) * 0.5
            reward += speed_reward
        
        # Penalty for collisions
        if info.get("collision", False):
            reward -= 10.0
        
        # Penalty for lane violations
        if info.get("lane_violation", False):
            reward -= 1.0
        
        # Penalty for running red lights
        if info.get("red_light_violation", False):
            reward -= 5.0
        
        # Success bonus
        if info.get("route_completed", False):
            reward += 100.0
            
    except Exception as e:
        logger.error(f"Error calculating reward: {e}")
    
    return reward

def check_termination_conditions(evaluator) -> Tuple[bool, bool, Dict[str, Any]]:
    """Check if episode should terminate"""
    terminated = False
    truncated = False
    info = {}
    
    try:
        # Check collision
        if hasattr(evaluator, 'collision_sensor'):
            if evaluator.collision_sensor.has_collided:
                terminated = True
                info["collision"] = True
        
        # Check max steps
        if sim_state.step_count >= sim_state.max_steps:
            truncated = True
            info["max_steps_reached"] = True
        
        # Check route completion
        if hasattr(evaluator, 'route_completed'):
            if evaluator.route_completed:
                terminated = True
                info["route_completed"] = True
                
    except Exception as e:
        logger.error(f"Error checking termination: {e}")
        
    return terminated, truncated, info

# Initialize FastAPI app
app = FastAPI(title="Bench2Drive CARLA Server", version="1.0.0")

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize simulation state
sim_state = SimulationState()

@app.on_event("startup")
async def startup_event():
    """Initialize server on startup"""
    logger.info("=" * 60)
    logger.info(f"Bench2Drive CARLA Server [{sim_state.server_id}] starting...")
    logger.info(f"API Port: {sim_state.api_port}")
    logger.info(f"CARLA Port: {sim_state.carla_port}")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    logger.info("🛑 Shutting down server...")
    
    # Close any running scenario
    if sim_state.leaderboard_evaluator:
        try:
            # Stop scenario if running
            if hasattr(sim_state.leaderboard_evaluator, 'manager') and sim_state.leaderboard_evaluator.manager:
                sim_state.leaderboard_evaluator.manager.stop_scenario()
            
            # Cleanup data provider
            CarlaDataProvider.cleanup()
            
            # Write final statistics
            if hasattr(sim_state.leaderboard_evaluator, 'statistics_manager'):
                sim_state.leaderboard_evaluator.statistics_manager.write_statistics()
            
            logger.info("✅ Cleaned up CARLA resources")
        except Exception as e:
            logger.warning(f"⚠️ Error during cleanup: {e}")
    
    logger.info("👋 Server shutdown complete")

# REST API Endpoints

@app.get("/")
async def root():
    """Root endpoint - can be used for health checks"""
    return {
        "status": "Bench2Drive CARLA Server Running",
        "version": "1.0.0",
        "server_id": sim_state.server_id,
        "api_port": sim_state.api_port,
        "carla_port": sim_state.carla_port
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "server_id": sim_state.server_id,
        "step_count": sim_state.step_count,
        "has_evaluator": sim_state.leaderboard_evaluator is not None
    }

# @app.get("/world")
# async def get_world_state():
#     """Get current world state information"""
#     if sim_state.leaderboard_evaluator is None:
#         raise HTTPException(status_code=400, detail="Environment not initialized")
    
#     # Get world reference
#     world = None
#     if hasattr(sim_state.leaderboard_evaluator, 'world'):
#         world = sim_state.leaderboard_evaluator.world
    
#     if world is None:
#         raise HTTPException(status_code=500, detail="No world object available")
    
#     try:
#         # Get all actors
#         all_actors = world.get_actors()
#         vehicles = all_actors.filter('vehicle.*')
#         pedestrians = all_actors.filter('walker.pedestrian.*')
#         traffic_lights = all_actors.filter('traffic.traffic_light')
        
#         # Find ego vehicle
#         ego_info = None
#         ego_vehicle = CarlaDataProvider.get_hero_actor() if CarlaDataProvider else None
        
#         if not ego_vehicle:
#             # Try to find by role_name
#             for vehicle in vehicles:
#                 if 'hero' in vehicle.attributes.get('role_name', '').lower():
#                     ego_vehicle = vehicle
#                     break
        
#         if ego_vehicle:
#             transform = ego_vehicle.get_transform()
#             velocity = ego_vehicle.get_velocity()
#             ego_info = {
#                 "id": ego_vehicle.id,
#                 "type": ego_vehicle.type_id,
#                 "position": {
#                     "x": transform.location.x,
#                     "y": transform.location.y,
#                     "z": transform.location.z
#                 },
#                 "rotation": {
#                     "pitch": transform.rotation.pitch,
#                     "yaw": transform.rotation.yaw,
#                     "roll": transform.rotation.roll
#                 },
#                 "velocity": {
#                     "x": velocity.x,
#                     "y": velocity.y,
#                     "z": velocity.z
#                 },
#                 "speed": (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
#             }
        
#         # Get world settings
#         settings = world.get_settings()
        
#         # Get map info
#         carla_map = world.get_map()
        
#         return {
#             "world_info": {
#                 "map_name": carla_map.name if carla_map else "unknown",
#                 "synchronous_mode": settings.synchronous_mode,
#                 "fixed_delta_seconds": settings.fixed_delta_seconds,
#                 "max_substeps": settings.max_substeps,
#                 "max_substep_delta_time": settings.max_substep_delta_time
#             },
#             "actor_counts": {
#                 "total": len(all_actors),
#                 "vehicles": len(vehicles),
#                 "pedestrians": len(pedestrians),
#                 "traffic_lights": len(traffic_lights)
#             },
#             "ego_vehicle": ego_info,
#             "timestamp": time.time()
#         }
#     except Exception as e:
#         logger.error(f"Error getting world state: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to get world state: {str(e)}")

@app.get("/world/vehicles")
async def get_world_vehicles():
    """Get detailed information about all vehicles in the world"""
    if sim_state.leaderboard_evaluator is None:
        raise HTTPException(status_code=400, detail="Environment not initialized")
    
    world = None
    if hasattr(sim_state.leaderboard_evaluator, 'world'):
        world = sim_state.leaderboard_evaluator.world
    
    if world is None:
        raise HTTPException(status_code=500, detail="No world object available")
    
    try:
        vehicles = world.get_actors().filter('vehicle.*')
        vehicle_list = []
        
        for vehicle in vehicles:
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            
            vehicle_info = {
                "id": vehicle.id,
                "type": vehicle.type_id,
                "attributes": dict(vehicle.attributes),
                "position": {
                    "x": transform.location.x,
                    "y": transform.location.y,
                    "z": transform.location.z
                },
                "rotation": {
                    "pitch": transform.rotation.pitch,
                    "yaw": transform.rotation.yaw,
                    "roll": transform.rotation.roll
                },
                "velocity": {
                    "x": velocity.x,
                    "y": velocity.y,
                    "z": velocity.z
                },
                "speed": (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5,
                "is_hero": 'hero' in vehicle.attributes.get('role_name', '').lower()
            }
            vehicle_list.append(vehicle_info)
        
        # Sort by ID for consistency
        vehicle_list.sort(key=lambda v: v['id'])
        
        return {
            "vehicles": vehicle_list,
            "count": len(vehicle_list)
        }
    except Exception as e:
        logger.error(f"Error getting vehicles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get vehicles: {str(e)}")

@app.get("/info")
async def server_info():
    """Get server information"""
    return {
        "server_id": sim_state.server_id,
        "api_port": sim_state.api_port,
        "carla_port": sim_state.carla_port,
        "initialized": sim_state.leaderboard_evaluator is not None,
        "current_route": sim_state.current_route,
        "step_count": sim_state.step_count,
        "max_steps": sim_state.max_steps
    }

@app.get("/status")
async def get_status():
    """Get current server status"""
    return {
        "server_id": sim_state.server_id,
        "api_port": sim_state.api_port,
        "carla_port": sim_state.carla_port,
        "initialized": sim_state.leaderboard_evaluator is not None,
        "current_route": sim_state.current_route,
        "step_count": sim_state.step_count,
        "cumulative_reward": sim_state.cumulative_reward,
        "num_snapshots": len(sim_state.snapshots)
    }

@app.get("/spec", response_model=SpecResponse)
async def get_spec():
    """Return action and observation space specifications"""
    return SpecResponse(
        action_space={
            "type": "Box",
            "low": {"throttle": 0.0, "brake": 0.0, "steer": -1.0},
            "high": {"throttle": 1.0, "brake": 1.0, "steer": 1.0},
            "shape": [3],
            "dtype": "float32"
        },
        observation_space={
            "type": "Dict",
            "spaces": {
                "images": {
                    "type": "Dict",
                    "spaces": {
                        "center": {"type": "Image", "shape": [512, 1024, 3]},
                        "left": {"type": "Image", "shape": [512, 1024, 3]},
                        "right": {"type": "Image", "shape": [512, 1024, 3]}
                    }
                },
                "vehicle_state": {
                    "type": "Dict",
                    "spaces": {
                        "position": {"type": "Box", "shape": [3]},
                        "rotation": {"type": "Box", "shape": [3]},
                        "velocity": {"type": "Box", "shape": [3]},
                        "speed": {"type": "Box", "shape": [1]}
                    }
                },
                "scenario_info": {"type": "Dict"}
            }
        }
    )

@app.post("/seed")
async def set_seed(request: SeedRequest):
    """Set random seed for reproducibility"""
    sim_state.seed = request.seed
    np.random.seed(request.seed)
    return {"seed": request.seed, "status": "success"}

@app.post("/reset")
async def reset_environment(request: ResetRequest):
    """Reset environment and return initial observation"""
    try:
        logger.info(f"Reset endpoint called with route_id={request.route_id}")
        # Check if we can do a quick reset (same scenario)
        quick_reset = False
        if (sim_state.current_config is not None and 
            sim_state.current_route == request.route_id and
            sim_state.manager_scenario is not None):
            quick_reset = True
            logger.info(f"Quick reset: reusing existing scenario for route {request.route_id}")
        
        # Reset state
        sim_state.reset()
        sim_state.current_route = request.route_id
        
        # Setup simulation if needed
        if sim_state.leaderboard_evaluator is None:
            logger.info("Setting up new LeaderboardEvaluator")
            # Create arguments namespace
            args = argparse.Namespace()
            
            # Basic configuration
            args.host = '127.0.0.1'
            args.port = sim_state.carla_port if sim_state.carla_port else 2000
            args.timeout = 300  # 5 minutes
            args.frame_rate = 20.0
            args.gpu_rank = 0
            args.traffic_manager_port = args.port + 1000
            
            # Agent configuration
            agent_path = os.path.join(os.path.dirname(__file__), 'api_agent.py')
            args.agent = agent_path
            args.agent_config = ''
            args.debug = 1
            
            # Routes configuration
            args.routes = "/mnt3/Documents/AD_Framework/Bench2Drive/leaderboard/data/bench2drive220.xml"
            args.repetitions = 1
            args.track = 'SENSORS'  # Allow camera sensors
            args.record = ''
            args.routes_subset = 0
            args.resume = 0
            args.traffic_manager_seed = args.port - 3333 + 1
            
            # Statistics configuration
            stats_dir = f'/mnt3/Documents/AD_Framework/bench2drive-gymnasium/logs/{sim_state.server_id}'
            os.makedirs(stats_dir, exist_ok=True)
            args.checkpoint = f'{stats_dir}/stat_manager_results.json'
            args.debug_checkpoint = f'{stats_dir}/live_results.txt'
            
            # Set environment variables
            os.environ['SAVE_PATH'] = stats_dir
            
            sim_state.args = args
            
            # Initialize statistics manager
            statistics_manager = StatisticsManager(args.checkpoint, args.debug_checkpoint)
            
            # Replace _setup_simulation method with our custom one
            LeaderboardEvaluator._setup_simulation = _setup_simulation
            
            # Create the actual evaluator
            logger.info("Creating LeaderboardEvaluator instance")
            leaderboard_evaluator = LeaderboardEvaluator(args, statistics_manager)
            logger.info("LeaderboardEvaluator created successfully")
            
            # Ensure world is available
            if not hasattr(leaderboard_evaluator, 'world') or leaderboard_evaluator.world is None:
                if hasattr(leaderboard_evaluator, 'client') and leaderboard_evaluator.client:
                    leaderboard_evaluator.world = leaderboard_evaluator.client.get_world()
                    logger.info("World initialized from client")
            
            # Initialize route indexer 
            route_indexer = RouteIndexer(args.routes, args.repetitions, args.routes_subset)
            
            # Handle resume if needed
            if args.resume:
                resume = route_indexer.validate_and_resume(args.checkpoint)
            else:
                resume = False
                
            if resume:
                leaderboard_evaluator.statistics_manager.add_file_records(args.checkpoint)
            else:
                leaderboard_evaluator.statistics_manager.clear_records()
            
            leaderboard_evaluator.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
            leaderboard_evaluator.statistics_manager.write_statistics()
            
            # Store evaluator and route indexer
            sim_state.leaderboard_evaluator = leaderboard_evaluator
            sim_state.route_indexer = route_indexer
        
        # Get route configuration
        if request.route_id is not None:
            # Try to find specific route
            want_route = f"RouteScenario_{request.route_id}"
            route_config = None
            for i in range(sim_state.route_indexer.total):
                config_temp = sim_state.route_indexer.get_next_config()
                if config_temp is None:
                    break
                if config_temp.name == want_route or str(request.route_id) in config_temp.name:
                    route_config = config_temp
                    break
            
            # If not found, get the first available route
            if route_config is None:
                sim_state.route_indexer.index = 0  # Reset to beginning
                route_config = sim_state.route_indexer.get_next_config()
        else:
            # Get next available route
            route_config = sim_state.route_indexer.get_next_config()
        
        if route_config is None:
            raise HTTPException(status_code=400, detail="No routes available")
        
        sim_state.config = route_config
        sim_state.current_config = route_config  # Store for quick reset check
        
        # Load scenario with proper config
        logger.info("Calling my_load_scenario...")
        try:
            result = my_load_scenario(
                sim_state.leaderboard_evaluator, 
                sim_state.args, 
                route_config
            )
            logger.info(f"my_load_scenario returned: {result}")
            
            if result is None:
                logger.error("my_load_scenario returned None - this should not happen")
                raise HTTPException(status_code=500, detail="Failed to load scenario - my_load_scenario returned None")
            
            crash_flag, save_name, entry_status, crash_message = result
        except Exception as e:
            logger.error(f"Error calling my_load_scenario: {e}")
            traceback.print_exc()
            raise
        
        if crash_flag:
            raise HTTPException(status_code=500, detail=f"Failed to load scenario: {crash_message}")
        
        # Save name for the model
        model_name = 'api_agent'
        save_name = save_name + '_' + model_name
        
        # Load the agent (API Agent)
        crash_flag, entry_status, crash_message = my_load_agent(
            sim_state.leaderboard_evaluator,
            sim_state.args,
            route_config,
            save_name,
            entry_status,
            crash_message
        )
        
        if crash_flag:
            raise HTTPException(status_code=500, detail=f"Failed to load agent: {crash_message}")
        
        # Store the agent instance for API interaction
        sim_state.agent_instance = sim_state.leaderboard_evaluator.agent_instance
        
        # Load and setup the route scenario
        crash_flag, entry_status, crash_message = my_load_route_scenario(
            sim_state.leaderboard_evaluator,
            sim_state.args,
            route_config,
            entry_status,
            crash_message
        )
        
        if crash_flag:
            raise HTTPException(status_code=500, detail=f"Failed to load route scenario: {crash_message}")
        
        # Setup scenario execution
        my_run_scenario_setup(sim_state.leaderboard_evaluator)
        
        # Store manager for quick reset check
        if hasattr(sim_state.leaderboard_evaluator, 'manager'):
            sim_state.manager_scenario = sim_state.leaderboard_evaluator.manager
        
        # Store initial state
        sim_state.entry_status = entry_status
        sim_state.crash_message = crash_message
        
        # IMPORTANT: We need to tick once to get initial sensor data!
        logger.info("Ticking once to get initial sensor data...")
        try:
            # Do one tick to populate sensor data
            crash_flag, entry_status, crash_message = my_run_scenario_step(
                sim_state.leaderboard_evaluator,
                entry_status,
                crash_message,
                n_steps=1
            )
            if crash_flag:
                logger.warning(f"Initial tick had issue: {crash_message}")
        except Exception as e:
            logger.warning(f"Warning during initial tick: {e}")
        
        # Now get initial observation from agent's sensor data
        obs = get_observation_from_state(sim_state.leaderboard_evaluator)
        sim_state.last_observation = obs
        
        # Get vehicle state for info
        info = {
            "route_id": sim_state.current_route,
            "seed": sim_state.seed
        }
        
        # Add vehicle state to info if available in observation
        if isinstance(obs, dict) and "vehicle_state" in obs:
            info["vehicle_state"] = obs["vehicle_state"]
        
        return {
            "observation": obs,
            "info": info
        }
        
    except Exception as e:
        logger.error(f"Error in reset: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=StepResponse)
async def step_environment(request: StepRequest):
    """Execute action and return next observation"""
    try:
        if sim_state.leaderboard_evaluator is None:
            raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
        
        if sim_state.agent_instance is None:
            raise HTTPException(status_code=400, detail="Agent not initialized. Call /reset first.")
        
        # Store action
        sim_state.last_action = request.action
        
        # Pass action to the API Agent
        if hasattr(sim_state.agent_instance, 'set_action'):
            sim_state.agent_instance.set_action(request.action)
        
        # Run simulation steps
        crash_flag, sim_state.entry_status, sim_state.crash_message = my_run_scenario_step(
            sim_state.leaderboard_evaluator,
            sim_state.entry_status,
            sim_state.crash_message,
            n_steps=request.n_steps
        )
        
        sim_state.step_count += request.n_steps
        
        # Get new observation
        obs = get_observation_from_state(sim_state.leaderboard_evaluator)
        sim_state.last_observation = obs
        
        # Check termination
        terminated, truncated, info = check_termination_conditions(sim_state.leaderboard_evaluator)
        
        # Calculate reward
        reward = calculate_reward(sim_state.leaderboard_evaluator, request.action, info)
        sim_state.cumulative_reward += reward
        
        # Add additional info
        info.update({
            "step_count": sim_state.step_count,
            "cumulative_reward": sim_state.cumulative_reward,
            "crash_flag": crash_flag,
            "crash_message": sim_state.crash_message if crash_flag else None
        })
        
        # Add vehicle state to info if available in observation
        if isinstance(obs, dict) and "vehicle_state" in obs:
            info["vehicle_state"] = obs["vehicle_state"]
        
        return StepResponse(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info
        )
        
    except Exception as e:
        logger.error(f"Error in step: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Snapshot support for GRPO

@app.post("/snapshot")
async def save_snapshot(request: SnapshotRequest):
    """
    Save current world state for GRPO branching.
    Uses notebook-based implementation that saves vehicle states and watchdog info.
    Based on step_by_step_leaderboard_snapshot_restore3.ipynb
    """
    try:
        snapshot_id = request.snapshot_id or f"snap_{sim_state.server_id}_{len(sim_state.snapshots)}"
        
        if sim_state.leaderboard_evaluator is None:
            raise HTTPException(status_code=400, detail="No active simulation")
        
        world = sim_state.leaderboard_evaluator.world
        manager = sim_state.leaderboard_evaluator.manager
        
        # Record all vehicles state (from notebook)
        actors = world.get_actors()
        vehicles = actors.filter("vehicle.*")
        
        vehicles_state = []
        for v in vehicles:
            tf = v.get_transform()
            loc, rot = tf.location, tf.rotation
            vel = v.get_velocity()
            ang = v.get_angular_velocity()
            acc = v.get_acceleration()
            ctrl = v.get_control()
            
            record = {
                "id": v.id,
                "type_id": v.type_id,
                "x": loc.x,
                "y": loc.y,
                "z": loc.z,
                "pitch": rot.pitch,
                "yaw": rot.yaw,
                "roll": rot.roll,
                "vx": vel.x,
                "vy": vel.y,
                "vz": vel.z,
                "wx": ang.x,
                "wy": ang.y,
                "wz": ang.z,
                "ax": acc.x,
                "ay": acc.y,
                "az": acc.z,
                "throttle": ctrl.throttle,
                "steer": ctrl.steer,
                "brake": ctrl.brake,
                "hand_brake": ctrl.hand_brake,
                "reverse": ctrl.reverse,
                "gear": ctrl.gear,
            }
            vehicles_state.append(record)
        
        # Save watchdog state (from notebook)
        watchdog_state = None
        if hasattr(manager, '_watchdog') and manager._watchdog:
            watchdog = manager._watchdog
            watchdog_state = {
                "timeout": watchdog._timeout,
                "interval": watchdog._interval,
                "failed": watchdog._failed,
                "stopped": watchdog._watchdog_stopped,
            }
        
        # Get current observation
        obs = get_observation_from_state(sim_state.leaderboard_evaluator)
        
        # Create snapshot object
        snapshot = WorldSnapshot(snapshot_id=snapshot_id)
        
        # Set the captured state
        snapshot.vehicles_state = vehicles_state
        snapshot.watchdog_state = watchdog_state
        snapshot.step_count = getattr(sim_state, 'step_count', 0)
        snapshot.timestamp = datetime.now().isoformat()
        snapshot.observation = obs
        
        # Store snapshot to shared filesystem
        snapshot_file = SNAPSHOTS_DIR / f"{snapshot_id}.json"
        snapshot_data = {
            "snapshot_id": snapshot_id,
            "vehicles_state": vehicles_state,
            "watchdog_state": watchdog_state,
            "step_count": snapshot.step_count,
            "timestamp": snapshot.timestamp,
            "observation": obs if isinstance(obs, dict) else {"data": obs},
            "server_id": sim_state.server_id
        }
        
        # Save to JSON file
        with open(snapshot_file, "w") as f:
            json.dump(snapshot_data, f, indent=2, default=str)
        
        # Also keep in memory for backward compatibility
        sim_state.snapshots[snapshot_id] = snapshot
        
        logger.info(f"Snapshot saved to {snapshot_file}: {snapshot_id} with {len(vehicles_state)} vehicles")
        
        return {
            "status": "success",
            "message": f"Snapshot saved: {snapshot_id}",
            "snapshot_id": snapshot_id,
            "stats": {
                "vehicles": len(vehicles_state),
                "has_watchdog": watchdog_state is not None,
                "step_count": snapshot.step_count
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to save snapshot: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/leaderboard_state")
async def get_leaderboard_state():
    """Get current leaderboard evaluator state information (serializable)"""
    try:
        if sim_state.leaderboard_evaluator is None:
            raise HTTPException(status_code=400, detail="Environment not initialized")
        
        evaluator = sim_state.leaderboard_evaluator
        
        # Build serializable state information
        state_info = {
            "status": "success",
            "has_world": hasattr(evaluator, 'world') and evaluator.world is not None,
            "has_client": hasattr(evaluator, 'client') and evaluator.client is not None,
            "has_manager": hasattr(evaluator, 'manager') and evaluator.manager is not None,
            "has_agent": hasattr(evaluator, 'agent_instance') and evaluator.agent_instance is not None,
        }
        
        # Add manager details if available
        if hasattr(evaluator, 'manager') and evaluator.manager:
            manager = evaluator.manager
            state_info["manager"] = {
                "running": manager._running if hasattr(manager, '_running') else None,
                "tick_count": getattr(manager, 'tick_count', 0),
                "ego_vehicles_count": len(manager.ego_vehicles) if hasattr(manager, 'ego_vehicles') else 0,
                "other_actors_count": len(manager.other_actors) if hasattr(manager, 'other_actors') else 0,
            }
        
        # Add route info if available
        if hasattr(evaluator, 'route_index'):
            state_info["route_index"] = evaluator.route_index
        
        # Add traffic manager info
        if hasattr(evaluator, 'traffic_manager') and evaluator.traffic_manager:
            tm = evaluator.traffic_manager
            state_info["traffic_manager"] = {
                "port": tm.get_port() if hasattr(tm, 'get_port') else None,
                "synchronous_mode": True  # We always use sync mode
            }
        
        # Add agent info
        if hasattr(evaluator, 'agent_instance') and evaluator.agent_instance:
            agent = evaluator.agent_instance
            state_info["agent"] = {
                "class_name": agent.__class__.__name__,
                "has_vehicle": hasattr(agent, '_vehicle') and agent._vehicle is not None,
                "vehicle_id": agent._vehicle.id if hasattr(agent, '_vehicle') and agent._vehicle else None
            }
        
        # Add statistics if available
        if hasattr(evaluator, 'statistics_manager') and evaluator.statistics_manager:
            stats = evaluator.statistics_manager
            if hasattr(stats, 'compute_route_statistics'):
                try:
                    route_stats = stats.compute_route_statistics()
                    state_info["statistics"] = {
                        "score": route_stats.get('score_composed', 0),
                        "route_completed": route_stats.get('route_completed', 0),
                        "infraction_count": route_stats.get('number_of_infractions', 0)
                    }
                except:
                    pass
        
        return state_info

    except Exception as e:
        logger.error(f"Error getting leaderboard state: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/restore")
async def restore_snapshot(request: RestoreRequest):
    """
    Restore world state from snapshot for GRPO branching.
    Uses notebook-based implementation with pause_restore_resume function.
    Based on step_by_step_leaderboard_snapshot_restore3.ipynb
    """
    try:
        snapshot_id = request.snapshot_id
        
        if sim_state.leaderboard_evaluator is None:
            raise HTTPException(status_code=400, detail="No active simulation")
        
        # Try to load from filesystem first
        snapshot_file = SNAPSHOTS_DIR / f"{snapshot_id}.json"
        
        if snapshot_file.exists():
            # Load from filesystem
            with open(snapshot_file, "r") as f:
                snapshot_data = json.load(f)
            
            # Create snapshot object from loaded data
            snapshot = WorldSnapshot(snapshot_id=snapshot_id)
            snapshot.vehicles_state = snapshot_data.get("vehicles_state", [])
            snapshot.watchdog_state = snapshot_data.get("watchdog_state", None)
            snapshot.step_count = snapshot_data.get("step_count", 0)
            snapshot.timestamp = snapshot_data.get("timestamp", "")
            snapshot.observation = snapshot_data.get("observation", {})
            
            logger.info(f"Loaded snapshot from filesystem: {snapshot_file}")
        elif snapshot_id in sim_state.snapshots:
            # Fallback to memory if not in filesystem
            snapshot = sim_state.snapshots[snapshot_id]
            logger.info(f"Loaded snapshot from memory: {snapshot_id}")
        else:
            raise HTTPException(status_code=404, detail=f"Snapshot {snapshot_id} not found in filesystem or memory")
        
        client = sim_state.leaderboard_evaluator.client
        world = sim_state.leaderboard_evaluator.world
        manager = sim_state.leaderboard_evaluator.manager
        
        logger.info(f"=== RESTORE START: {snapshot_id} ===")
        
        # Helper functions from notebook
        def stop_manager(manager):
            """Stop manager threads"""
            manager._running = False
            
            t = getattr(manager, "_scenario_thread", None)
            if t and hasattr(t, "is_alive") and t.is_alive() and threading.current_thread() is not t:
                t.join(timeout=2.0)
            manager._scenario_thread = None
            
            tt = getattr(manager, "_tick_thread", None)
            if tt and hasattr(tt, "is_alive") and tt.is_alive() and threading.current_thread() is not tt:
                tt.join(timeout=2.0)
            manager._tick_thread = None
        
        def reinit_inroute_and_blocked(tree):
            """Re-initialise only InRouteTest and ActorBlockedTest"""
            import py_trees
            target = {"InRouteTest", "ActorBlockedTest"}
            for n in tree.iterate():
                if n.__class__.__name__ in target:
                    try:
                        n.terminate(py_trees.common.Status.INVALID)
                        n.initialise()
                    except Exception:
                        pass
        
        def start_manager_builder_only(manager):
            """Start ONLY the builder loop; NO auto-ticking"""
            manager._running = True
            t = getattr(manager, "_scenario_thread", None)
            if t is None or not (hasattr(t, "is_alive") and t.is_alive()):
                manager._scenario_thread = threading.Thread(
                    target=manager.build_scenarios_loop,
                    args=(manager._debug_mode > 0,),
                    daemon=True
                )
                manager._scenario_thread.start()
        
        def _existing_vehicle_ids(world):
            return {v.id for v in world.get_actors().filter("vehicle.*")}
        
        # 1) Stop manager activity
        stop_manager(manager)
        logger.info("Manager stopped")
        
        # 2) Ensure synchronous mode
        prev_settings = world.get_settings()
        changed = False
        if not prev_settings.synchronous_mode:
            new_settings = carla.WorldSettings()
            new_settings.no_rendering_mode = prev_settings.no_rendering_mode
            new_settings.synchronous_mode = True
            new_settings.fixed_delta_seconds = prev_settings.fixed_delta_seconds or 0.05
            world.apply_settings(new_settings)
            changed = True
            logger.info("Set synchronous mode")
        
        # 3) Build restore batch
        present_ids = _existing_vehicle_ids(world)
        batch = []
        
        for rec in snapshot.vehicles_state:
            vid = int(rec["id"])
            if vid not in present_ids:
                continue
            
            tf = carla.Transform(
                carla.Location(float(rec["x"]), float(rec["y"]), float(rec["z"])),
                carla.Rotation(
                    yaw=float(rec.get("yaw", 0.0)),
                    pitch=float(rec.get("pitch", 0.0)),
                    roll=float(rec.get("roll", 0.0))
                )
            )
            
            # Disable physics, set transform, re-enable physics
            batch += [
                carla.command.SetSimulatePhysics(vid, False),
                carla.command.ApplyTransform(vid, tf),
                carla.command.SetSimulatePhysics(vid, True),
            ]
        
        # Apply transforms
        client.apply_batch_sync(batch, True)
        world.tick()
        logger.info(f"Applied transforms for {len(batch)//3} vehicles")
        
        # 4) Apply velocities using enable_constant_velocity
        for rec in snapshot.vehicles_state:
            vid = int(rec["id"])
            if vid not in present_ids:
                continue
            
            actor = world.get_actor(vid)
            if actor:
                vx = float(rec.get("vx", 0.0))
                vy = float(rec.get("vy", 0.0))
                vz = float(rec.get("vz", 0.0))
                wx = float(rec.get("wx", 0.0))
                wy = float(rec.get("wy", 0.0))
                wz = float(rec.get("wz", 0.0))
                
                # Check yaw angle to determine if velocity direction needs flipping
                yaw = float(rec.get("yaw", 0.0))
                if abs(yaw) > 135:
                    # Vehicle is facing backward, flip velocity direction
                    vx = -vx
                    vy = -vy
                    logger.debug(f"Vehicle {vid} yaw={yaw:.1f}, flipping velocity direction")
                
                # Force exact velocity for one frame
                actor.enable_constant_velocity(carla.Vector3D(vx, vy, vz))
                actor.set_target_angular_velocity(carla.Vector3D(wx, wy, wz))
                
                # Debug print for ego
                if vid == 3695 or actor.attributes.get('role_name') == 'hero':
                    logger.info(f"Ego velocity restore: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yaw={yaw:.1f}")
        
        # Tick to apply velocities
        world.tick()
        
        # 5) Disable constant velocity and apply controls
        for rec in snapshot.vehicles_state:
            vid = int(rec["id"])
            if vid not in present_ids:
                continue
            
            actor = world.get_actor(vid)
            if actor:
                # Disable constant velocity
                actor.disable_constant_velocity()
                
                # Apply saved controls
                ctrl = carla.VehicleControl(
                    throttle=float(rec.get("throttle", 0.0)),
                    steer=float(rec.get("steer", 0.0)),
                    brake=float(rec.get("brake", 0.0)),
                    hand_brake=bool(rec.get("hand_brake", False)),
                    reverse=bool(rec.get("reverse", False)),
                    gear=int(rec.get("gear", 0))
                )
                actor.apply_control(ctrl)
        
        # Final tick
        world.tick()
        logger.info("Applied velocities and controls")
        
        # 6) Reinit criteria to prevent immediate failure
        if hasattr(manager, 'scenario_tree'):
            reinit_inroute_and_blocked(manager.scenario_tree)
            logger.info("Reinitialized criteria")
        
        # 7) Restore settings if changed (keep_sync default is True in notebook)
        keep_sync = getattr(request, 'keep_sync', True)
        if changed and not keep_sync:
            world.apply_settings(prev_settings)
            logger.info("Restored world settings")
        
        # 8) Restart manager builder thread
        start_manager_builder_only(manager)
        logger.info("Restarted manager builder thread")
        
        # 9) Restore step count
        if hasattr(snapshot, 'step_count'):
            sim_state.step_count = snapshot.step_count
            logger.info(f"Restored step count: {sim_state.step_count}")
        
        # 10) Get observation from restored state
        obs = get_observation_from_state(sim_state.leaderboard_evaluator)
        sim_state.last_observation = obs
        
        logger.info(f"=== RESTORE SUCCESS: {snapshot_id} ===")
        
        return {
            "status": "success",
            "message": f"Restore completed: {snapshot_id}",
            "stats": {
                "vehicles_restored": len([r for r in snapshot.vehicles_state if int(r["id"]) in present_ids]),
                "step_count": getattr(snapshot, 'step_count', 0)
            },
            "observation": obs
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Restore failed: {e}")

# Note: Removed old restore2-5 and notebook versions (duplicate code)

# Removed duplicate restore functions (restore2-5, snapshot_notebook, restore_notebook)

@app.get("/snapshots")
async def list_snapshots():
    """List all available snapshots from filesystem and memory"""
    try:
        # Get snapshots from filesystem
        filesystem_snapshots = []
        for snapshot_file in SNAPSHOTS_DIR.glob("*.json"):
            filesystem_snapshots.append(snapshot_file.stem)
        
        # Get snapshots from memory
        memory_snapshots = list(sim_state.snapshots.keys())
        
        # Combine and deduplicate
        all_snapshots = list(set(filesystem_snapshots + memory_snapshots))
        
        return {
            "status": "success",
            "snapshots": all_snapshots,
            "filesystem": filesystem_snapshots,
            "memory": memory_snapshots,
            "total": len(all_snapshots)
        }
    except Exception as e:
        logger.error(f"Error listing snapshots: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/close")
async def close_environment():
    """Close environment and cleanup resources"""
    try:
        if sim_state.leaderboard_evaluator:
            my_stop_scenario(
                sim_state.leaderboard_evaluator,
                sim_state.args,
                sim_state.config,
                sim_state.entry_status,
                sim_state.crash_message
            )
            sim_state.leaderboard_evaluator = None
        
        # Clear snapshots
        sim_state.snapshots.clear()
        
        return {"status": "success", "message": "Environment closed"}
        
    except Exception as e:
        logger.error(f"Error closing environment: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _post_restore_init(evaluator):
    """
    Re-init runtime pieces that are not serialized by WorldSnapshot.
    Must be idempotent. Called after snapshot restore.
    """
    import threading
    from srunner.scenariomanager.watchdog import Watchdog
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    
    logger.info("Starting post-restore initialization")
    
    # 1) World & TM settings
    world = evaluator.world
    client = evaluator.client if hasattr(evaluator, 'client') else None
    tm = evaluator.traffic_manager if hasattr(evaluator, 'traffic_manager') else None
    
    # World settings
    s = world.get_settings()
    s.synchronous_mode = True
    if hasattr(evaluator, "frame_rate") and evaluator.frame_rate:
        s.fixed_delta_seconds = 1.0 / float(evaluator.frame_rate)
    elif hasattr(LeaderboardEvaluator, 'frame_rate'):
        s.fixed_delta_seconds = 1.0 / LeaderboardEvaluator.frame_rate
    world.apply_settings(s)
    logger.info("World settings applied: synchronous mode enabled")
    
    # Traffic manager in sync
    if tm:
        tm.set_synchronous_mode(True)
        if hasattr(tm, "set_hybrid_physics_mode"):
            tm.set_hybrid_physics_mode(True)
        logger.info(f"Traffic Manager sync mode enabled on port {tm.get_port()}")
    
    # 2) CarlaDataProvider refresh
    try:
        CarlaDataProvider.set_client(client)
        CarlaDataProvider.set_world(world)
        if tm:
            CarlaDataProvider.set_traffic_manager_port(tm.get_port())
        logger.info("CarlaDataProvider refreshed with current world/client")
    except Exception as e:
        logger.debug(f"CarlaDataProvider refresh skipped/failed: {e}", exc_info=True)
    
    # 3) Watchdogs
    if hasattr(evaluator, 'manager') and evaluator.manager:
        manager = evaluator.manager
        
        # Simulation watchdog
        if getattr(manager, "_watchdog", None) is None:
            manager._watchdog = Watchdog(manager._timeout)
            logger.info("Created new simulation watchdog")
        try:
            manager._watchdog.start()
            logger.info("Started simulation watchdog")
        except Exception as e:
            logger.debug(f"Watchdog already started or failed: {e}")
        
        # Agent watchdog (Dummy for APIAgent)
        agent_cls = evaluator.agent_instance.__class__.__name__ if hasattr(evaluator, 'agent_instance') and evaluator.agent_instance else ""
        if getattr(manager, "_agent_watchdog", None) is None:
            if agent_cls == "APIAgent":
                # Use a dummy watchdog for APIAgent
                class DummyWatchdog:
                    def __init__(self, timeout):
                        self._timeout = timeout
                    def start(self):
                        pass
                    def stop(self):
                        pass
                    def pause(self):
                        pass
                    def resume(self):
                        pass
                    def update(self):
                        pass
                
                manager._agent_watchdog = DummyWatchdog(manager._timeout)
                logger.info("Created dummy agent watchdog for APIAgent")
            else:
                manager._agent_watchdog = Watchdog(manager._timeout)
                logger.info("Created agent watchdog")
        
        try:
            manager._agent_watchdog.start()
            logger.info("Started agent watchdog")
        except Exception as e:
            logger.debug(f"Agent watchdog already started or failed: {e}")
        
        # 4) Scenario thread (build_scenarios_loop) if needed
        try:
            th = getattr(manager, "_scenario_thread", None)
            alive = th.is_alive() if th else False
        except Exception:
            alive = False
        
        if not alive:
            manager._scenario_thread = threading.Thread(
                target=manager.build_scenarios_loop,
                args=(manager._debug_mode > 0,),
                daemon=True,
            )
            manager._scenario_thread.start()
            logger.info("Started scenario build thread")
        
        # 5) Ensure manager running flag
        manager._running = True
        logger.info("Manager set to running state")
    
    logger.info("Post-restore initialization complete")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Bench2Drive CARLA API Server")
    parser.add_argument("--port", type=int, default=8080,
                        help="API server port (default: 8080)")
    parser.add_argument("--carla-port", type=int, default=2000,
                        help="CARLA server port (default: 2000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="API server host (default: 0.0.0.0)")
    parser.add_argument("--carla-host", type=str, default="localhost",
                        help="CARLA server host (default: localhost)")
    parser.add_argument("--server-id", type=str, default=None,
                        help="Server ID for identification")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    cmd_args = parse_args()
    
    # Set server state from arguments
    sim_state.api_port = cmd_args.port
    sim_state.carla_port = cmd_args.carla_port
    if cmd_args.server_id:
        sim_state.server_id = cmd_args.server_id
    
    # Run server
    logger.info(f"Starting Bench2Drive CARLA server {sim_state.server_id} on port {cmd_args.port}...")
    uvicorn.run(app, host=cmd_args.host, port=cmd_args.port, log_level="info")