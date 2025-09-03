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

# Setup Bench2Drive environment and paths
def setup_bench2drive_environment():
    """Setup environment variables and paths for Bench2Drive"""
    
    # Set required environment variables
    os.environ['IS_BENCH2DRIVE'] = 'true'
    os.environ['REPETITION'] = '1'
    os.environ['TMP_VISU'] = '0'  # Enable camera sensors
    
    # Add Bench2Drive paths to sys.path
    bench2drive_root = '/mnt3/Documents/AD_Framework/Bench2Drive'
    leaderboard_path = f'{bench2drive_root}/leaderboard'
    leaderboard_module_path = f'{bench2drive_root}/leaderboard/leaderboard'
    scenario_runner_path = f'{bench2drive_root}/scenario_runner'
    
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
        os.environ['CARLA_ROOT'] = '/mnt3/Documents/AD_Framework/carla0915'
        
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

# Import API agent
try:
    from api_agent import APIAgent, get_entry_point
except ImportError:
    # If running as script, try absolute import
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
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
    """Save current world state for GRPO branching"""
    try:
        if sim_state.leaderboard_evaluator is None:
            raise HTTPException(status_code=400, detail="Environment not initialized")
        
        if WorldSnapshot is None:
            raise HTTPException(status_code=501, detail="Snapshot feature not available")
        
        # Verify scenario manager exists (CRITICAL for proper restore)
        if not hasattr(sim_state.leaderboard_evaluator, 'manager') or sim_state.leaderboard_evaluator.manager is None:
            raise HTTPException(status_code=400, detail="Scenario manager not initialized - cannot snapshot without manager")
        
        # Generate snapshot ID if not provided
        snapshot_id = request.snapshot_id or f"snap_{sim_state.server_id}_{len(sim_state.snapshots)}"
        
        # Capture complete world state
        world = None
        if hasattr(sim_state.leaderboard_evaluator, 'world'):
            world = sim_state.leaderboard_evaluator.world
            print(f"DEBUG: Got world from leaderboard_evaluator: {world}")
        else:
            print("ERROR: leaderboard_evaluator has no world attribute!")

        # Debug: Print available attributes
        print("DEBUG: leaderboard_evaluator attributes:")
        print(f"  - Has manager: {hasattr(sim_state.leaderboard_evaluator, 'manager')}")
        print(f"  - Has world: {hasattr(sim_state.leaderboard_evaluator, 'world')}")
        print(f"  - World object: {world}")
        print(f"  - Has agent_instance: {hasattr(sim_state.leaderboard_evaluator, 'agent_instance')}")
        print(f"  - Has route_scenario: {hasattr(sim_state.leaderboard_evaluator, 'route_scenario')}")
        
        # CRITICAL: Get the current observation BEFORE creating snapshot
        # This ensures the snapshot includes the current sensor data/images
        try:
            current_observation = get_observation_from_state(sim_state.leaderboard_evaluator)
            logger.info(f"Captured observation for snapshot with {len(current_observation.get('images', {}))} images")
        except Exception as e:
            logger.warning(f"Could not capture observation for snapshot: {e}")
            current_observation = sim_state.last_observation if sim_state.last_observation else {}
        
        # Mark this as a specific phase for debugging
        phase_marker = f"Step {sim_state.step_count} - {snapshot_id}"
        snapshot = WorldSnapshot.capture(sim_state, world, phase_marker=phase_marker)
        snapshot.snapshot_id = snapshot_id
        
        # CRITICAL: Store the observation in the snapshot
        snapshot.observation = current_observation
        logger.info(f"Stored observation in snapshot with images: {list(current_observation.get('images', {}).keys())}")
        
        # CRITICAL: Save RouteScenario state to prevent NPC respawning on restore
        try:
            if hasattr(sim_state.leaderboard_evaluator, 'manager') and sim_state.leaderboard_evaluator.manager:
                manager = sim_state.leaderboard_evaluator.manager
                route_scenario = manager._route_scenario if hasattr(manager, '_route_scenario') else manager.scenario
                
                if route_scenario and hasattr(route_scenario, 'list_scenarios'):
                    # Log detailed scenario information
                    logger.info("=== SNAPSHOT: RouteScenario State ===")
                    logger.info(f"Active scenarios ({len(route_scenario.list_scenarios)}):")
                    for idx, scenario in enumerate(route_scenario.list_scenarios):
                        logger.info(f"  [{idx}] {scenario.config.name}:")
                        logger.info(f"      Type: {scenario.config.type}")
                        logger.info(f"      Other actors in scenario: {len(scenario.other_actors)}")
                        if hasattr(scenario, 'other_actors') and scenario.other_actors:
                            for actor in scenario.other_actors[:3]:  # Show first 3
                                logger.info(f"        - Actor {actor.id}: {actor.type_id}")
                        if hasattr(scenario, 'behavior_tree'):
                            logger.info(f"      Has behavior tree: {scenario.behavior_tree is not None}")
                    
                    logger.info(f"Missing scenario configs ({len(route_scenario.missing_scenario_configurations)}):")
                    for config in route_scenario.missing_scenario_configurations[:5]:  # Show first 5
                        logger.info(f"  - {config.name} (type: {config.type})")
                    
                    logger.info(f"Route other_actors: {len(route_scenario.other_actors)} actors")
                    for actor in route_scenario.other_actors[:5]:  # Show first 5
                        logger.info(f"  - Actor {actor.id}: {actor.type_id}")
                    
                    # Save COMPLETE scenario state including instances
                    active_scenarios_data = []
                    for scenario in route_scenario.list_scenarios:
                        try:
                            # Extract only serializable config data
                            config_data = {
                                'name': scenario.config.name if hasattr(scenario.config, 'name') else 'unknown',
                                'type': scenario.config.type if hasattr(scenario.config, 'type') else 'unknown',
                                # Don't save the transform or other non-serializable objects
                            }
                            
                            scenario_data = {
                                'name': scenario.config.name,
                                'type': scenario.config.type,
                                'config_data': config_data,  # Save only serializable config data
                                'other_actor_ids': [],
                                'other_actor_types': [],
                                'other_actor_positions': [],
                                'has_behavior_tree': hasattr(scenario, 'behavior_tree') and scenario.behavior_tree is not None
                            }
                            
                            # Safely extract actor information
                            if hasattr(scenario, 'other_actors') and scenario.other_actors:
                                for actor in scenario.other_actors:
                                    try:
                                        scenario_data['other_actor_ids'].append(actor.id)
                                        scenario_data['other_actor_types'].append(actor.type_id)
                                        loc = actor.get_location()
                                        scenario_data['other_actor_positions'].append({
                                            'id': actor.id,
                                            'x': loc.x,
                                            'y': loc.y,
                                            'z': loc.z
                                        })
                                    except Exception as e:
                                        logger.warning(f"Could not save actor data: {e}")
                            
                            active_scenarios_data.append(scenario_data)
                            logger.info(f"  Saving scenario {scenario.config.name} with {len(scenario_data['other_actor_ids'])} actors")
                        except Exception as e:
                            logger.error(f"Error saving scenario data: {e}")
                            continue
                    
                    # Extract only serializable data from missing configs
                    missing_configs_data = []
                    if hasattr(route_scenario, 'missing_scenario_configurations'):
                        for config in route_scenario.missing_scenario_configurations:
                            try:
                                missing_configs_data.append({
                                    'name': config.name if hasattr(config, 'name') else 'unknown',
                                    'type': config.type if hasattr(config, 'type') else 'unknown'
                                })
                            except:
                                pass
                    
                    # Save only serializable scenario state data
                    route_scenario_state = {
                        'missing_configs_data': missing_configs_data,
                        'active_scenarios_data': active_scenarios_data,
                        # Don't save actual scenario instances or behavior/criteria nodes as they're not serializable
                        'num_other_actors': len(route_scenario.other_actors) if hasattr(route_scenario, 'other_actors') else 0,
                        'other_actor_ids': [a.id for a in route_scenario.other_actors] if hasattr(route_scenario, 'other_actors') else []
                    }
                    snapshot.route_scenario_state = route_scenario_state
                    logger.info(f"Saved RouteScenario state: {len(active_scenarios_data)} active scenarios, {route_scenario_state['num_other_actors']} other actors")
        except Exception as e:
            logger.warning(f"Could not save RouteScenario state: {e}")
            # Continue without scenario state - basic snapshot will still work
        
        # Store a deep copy of the snapshot to ensure complete isolation
        import copy
        sim_state.snapshots[snapshot_id] = copy.deepcopy(snapshot)
        
        # Save snapshot to disk for debugging and persistence
        snapshot_file = sim_state.snapshot_dir / f"{snapshot_id}.pkl"
        try:
            with open(snapshot_file, 'wb') as f:
                pickle.dump(snapshot, f)
            logger.info(f"Snapshot saved to disk: {snapshot_file}")
        except Exception as e:
            logger.error(f"Failed to save snapshot to disk: {e}")
        
        # Debug: Print what we captured
        print(f"Snapshot captured: {snapshot_id}")
        print(f"  - Vehicles: {len(snapshot.vehicles)}")
        print(f"  - Has scenario_manager state: {snapshot.scenario_manager is not None}")
        print(f"  - Saved to: {snapshot_file}")
        
        # Log ego vehicle position in the stored snapshot
        for vehicle_id, vehicle_state in sim_state.snapshots[snapshot_id].vehicles.items():
            if vehicle_state.is_hero:
                print(f"  - Stored ego position in snapshot: x={vehicle_state.location['x']:.2f}, y={vehicle_state.location['y']:.2f}")
                logger.info(f"SNAPSHOT SAVED with ego at: x={vehicle_state.location['x']:.2f}, y={vehicle_state.location['y']:.2f}")
        if snapshot.scenario_manager:
            print(f"    - Manager tick: {snapshot.scenario_manager.tick_count}")
            print(f"    - Manager running: {snapshot.scenario_manager.running}")
            print(f"    - Ego vehicles: {len(snapshot.scenario_manager.ego_vehicle_ids)}")
        
        # Build comprehensive stats including scenario manager state
        stats = {
            "vehicles": len(snapshot.vehicles),
            "pedestrians": len(snapshot.pedestrians),
            "traffic_lights": len(snapshot.traffic_lights),
            "has_scenario_manager": snapshot.scenario_manager is not None
        }
        
        if snapshot.scenario_manager:
            stats.update({
                "manager_tick": snapshot.scenario_manager.tick_count,
                "manager_running": snapshot.scenario_manager.running,
                "ego_vehicles": len(snapshot.scenario_manager.ego_vehicle_ids),
                "other_actors": len(snapshot.scenario_manager.other_actor_ids)
            })
        
        logger.info(f"Created snapshot {snapshot_id}: {stats}")
        
        return {
            "snapshot_id": snapshot_id,
            "status": "success",
            "message": f"Snapshot saved: {snapshot_id}",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error saving snapshot: {e}")
        traceback.print_exc()
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
    """Restore world state from snapshot for GRPO branching"""
    logger.info(f"Restore endpoint called with snapshot_id: {request.snapshot_id}")
    try:
        if sim_state.leaderboard_evaluator is None:
            raise HTTPException(status_code=400, detail="Environment not initialized")
        
        if WorldSnapshot is None:
            raise HTTPException(status_code=501, detail="Snapshot feature not available")
        
        # Try to get snapshot from memory first, then disk
        if request.snapshot_id in sim_state.snapshots:
            snapshot = sim_state.snapshots[request.snapshot_id]
            logger.info(f"Loading snapshot from memory: {request.snapshot_id}")
        else:
            # Try loading from disk
            snapshot_file = sim_state.snapshot_dir / f"{request.snapshot_id}.pkl"
            if snapshot_file.exists():
                try:
                    with open(snapshot_file, 'rb') as f:
                        snapshot = pickle.load(f)
                    logger.info(f"Loaded snapshot from disk: {snapshot_file}")
                    # Cache in memory for faster access
                    sim_state.snapshots[request.snapshot_id] = snapshot
                except Exception as e:
                    logger.error(f"Failed to load snapshot from disk: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to load snapshot: {e}")
            else:
                raise HTTPException(status_code=404, detail=f"Snapshot not found: {request.snapshot_id}")
        
        # Get snapshot (now guaranteed to exist)
        snapshot = sim_state.snapshots[request.snapshot_id]
        logger.info(f"Retrieved snapshot {request.snapshot_id}, type: {type(snapshot)}")
        
        # Log what position we're restoring from the snapshot
        if hasattr(snapshot, 'vehicles') and snapshot.vehicles:
            for _, vehicle_state in snapshot.vehicles.items():
                if vehicle_state.is_hero:
                    print(f"Restoring ego from snapshot position: x={vehicle_state.location['x']:.2f}, y={vehicle_state.location['y']:.2f}")
                    logger.info(f"RESTORING from snapshot with ego at: x={vehicle_state.location['x']:.2f}, y={vehicle_state.location['y']:.2f}")
                    break
        
        # Get world reference
        # NOTE: We cannot save the world object itself in the snapshot because:
        # 1. It's not serializable (contains network connections, GPU resources, etc.)
        # 2. It's a live handle to the CARLA simulator (like a database connection)
        # 3. There's only ONE world object per CARLA instance
        # Instead, we save the world STATE (vehicles, lights, etc.) and apply it to the current world
        world = None
        if hasattr(sim_state.leaderboard_evaluator, 'world'):
            world = sim_state.leaderboard_evaluator.world
            logger.info(f"Got world object handle: {world is not None}")
        else:
            logger.warning("No world attribute in leaderboard_evaluator")
            
        if world is None:
            logger.error("No world object available for restore!")
            raise HTTPException(status_code=500, detail="No world object available - cannot apply saved state")
            
        # CRITICAL: Ensure world is in sync mode during restore
        # This prevents the world from advancing without explicit ticks
        if world:
            settings = world.get_settings()
            if not settings.synchronous_mode:
                logger.warning("World was NOT in sync mode! Forcing sync mode for restore")
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / 20.0
            world.apply_settings(settings)
            logger.info(f"World in sync mode for restore")
        
        # CRITICAL: Pause scenario manager to prevent background ticking
        original_running = False
        if hasattr(sim_state.leaderboard_evaluator, 'manager') and sim_state.leaderboard_evaluator.manager:
            manager = sim_state.leaderboard_evaluator.manager
            # Store original running state
            original_running = getattr(manager, '_running', False)
            # Pause the manager
            manager._running = False
            logger.info(f"Paused scenario manager for restore (was running: {original_running})")
        
        # Restore world state
        if isinstance(snapshot, WorldSnapshot):
            try:
                success = snapshot.restore(sim_state, world)
                if not success:
                    # Restore running state on failure
                    if hasattr(sim_state.leaderboard_evaluator, 'manager'):
                        sim_state.leaderboard_evaluator.manager._running = original_running
                    raise HTTPException(status_code=500, detail="Failed to restore snapshot")
            except Exception as e:
                import traceback
                error_details = f"Restore error: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_details)
                # Restore running state on failure
                if hasattr(sim_state.leaderboard_evaluator, 'manager'):
                    sim_state.leaderboard_evaluator.manager._running = original_running
                raise HTTPException(status_code=500, detail=error_details)
        else:
            # Old-style snapshot (backward compatibility)
            sim_state.step_count = snapshot["step_count"]
            sim_state.cumulative_reward = snapshot["cumulative_reward"]
            sim_state.last_observation = snapshot["last_observation"]
            sim_state.last_action = snapshot["last_action"]
            sim_state.entry_status = snapshot["entry_status"]
            sim_state.crash_message = snapshot["crash_message"]
            sim_state.current_route = snapshot["current_route"]
        
        logger.info(f"Restored snapshot {request.snapshot_id}")
        
        # Restore scenario manager running state
        if hasattr(sim_state.leaderboard_evaluator, 'manager') and sim_state.leaderboard_evaluator.manager:
            sim_state.leaderboard_evaluator.manager._running = original_running
            logger.info(f"Restored scenario manager running state: {original_running}")
        
        # CRITICAL: Re-establish all connections after restore (like after world load)
        # This is based on _load_and_wait_for_world sequence
        if hasattr(sim_state.leaderboard_evaluator, 'world'):
            world = sim_state.leaderboard_evaluator.world
            client = sim_state.leaderboard_evaluator.client if hasattr(sim_state.leaderboard_evaluator, 'client') else None
            traffic_manager = sim_state.leaderboard_evaluator.traffic_manager if hasattr(sim_state.leaderboard_evaluator, 'traffic_manager') else None
            
            if client and world and traffic_manager:
                # Re-establish CarlaDataProvider connections
                CarlaDataProvider.set_client(client)
                CarlaDataProvider.set_world(world)
                CarlaDataProvider.set_traffic_manager_port(traffic_manager.get_port())
                logger.info(f"Re-established CarlaDataProvider connections: TM port {traffic_manager.get_port()}")
                
                # Ensure Traffic Manager is in sync mode
                traffic_manager.set_synchronous_mode(True)
                traffic_manager.set_hybrid_physics_mode(True)
                logger.info("Set Traffic Manager to sync mode")
                
                # CRITICAL FIX: DO NOT re-enable autopilot after restore!
                # This was causing Traffic Manager to respawn/reposition vehicles
                # The vehicles should maintain their restored state and control
                logger.info("SKIPPING Traffic Manager re-registration to preserve restored vehicle states")
                
                # Only ensure TM knows about the vehicles without changing their control
                vehicles = world.get_actors().filter('vehicle.*')
                ego_id = None
                if hasattr(sim_state.leaderboard_evaluator, 'manager') and sim_state.leaderboard_evaluator.manager:
                    if hasattr(sim_state.leaderboard_evaluator.manager, 'ego_vehicles') and len(sim_state.leaderboard_evaluator.manager.ego_vehicles) > 0:
                        ego_id = sim_state.leaderboard_evaluator.manager.ego_vehicles[0].id
                
                # Count vehicles but don't change their control state
                non_ego_count = sum(1 for v in vehicles if not (ego_id and v.id == ego_id))
                logger.info(f"Found {non_ego_count} non-ego vehicles, preserving their restored state")
                
                # CRITICAL: DO NOT TICK! This causes position drift
                # The tick was causing vehicles to move forward from restored position
                # world.tick()  # REMOVED - causes position drift!
                logger.info("Skipping world tick to prevent position drift")
        
        # Set a flag to skip scenario ticking on next step
        # This prevents the scenario from resetting when it detects inconsistencies
        sim_state.just_restored = True
        # Also set on leaderboard_evaluator where it's actually checked
        if sim_state.leaderboard_evaluator:
            sim_state.leaderboard_evaluator.just_restored = True
        logger.info("Set just_restored flag on both sim_state and leaderboard_evaluator to skip next scenario tick")
        
        
        # CRITICAL: Set step count to match snapshot exactly
        # This ensures we're at the right frame number
        if hasattr(snapshot, 'step_count'):
            sim_state.step_count = snapshot.step_count
            logger.info(f"Restored step count to {snapshot.step_count}")
        
        # CRITICAL: After restore, ensure scenario manager attributes are properly connected
        # Based on ScenarioManager.__init__, we need to ensure all attributes are valid
        try:
            ego_vehicle = None
            if hasattr(sim_state.leaderboard_evaluator, 'manager') and sim_state.leaderboard_evaluator.manager:
                manager = sim_state.leaderboard_evaluator.manager
                
                # CRITICAL: Ensure manager is in running state (from __init__)
                manager._running = True
                
                # The ego vehicle is stored in manager.ego_vehicles[0]
                if hasattr(manager, 'ego_vehicles') and len(manager.ego_vehicles) > 0:
                    ego_vehicle = manager.ego_vehicles[0]
                    logger.info(f"Got ego vehicle from manager: {ego_vehicle.id if ego_vehicle else 'None'}")
                    
                    # CRITICAL: After restore, the ego vehicle actor might be a stale reference
                    # We need to get the fresh actor from the world
                    if ego_vehicle and hasattr(sim_state.leaderboard_evaluator, 'world'):
                        world = sim_state.leaderboard_evaluator.world
                        fresh_ego = world.get_actor(ego_vehicle.id)
                        if fresh_ego:
                            ego_vehicle = fresh_ego
                            # Update the manager's reference
                            manager.ego_vehicles[0] = ego_vehicle
                            logger.info(f"Refreshed ego vehicle reference: {ego_vehicle.id}")
            
            # Re-establish agent's connection to the ego vehicle
            if sim_state.agent_instance and ego_vehicle:
                agent = sim_state.agent_instance
                
                # Reset vehicle reference in agent
                agent._vehicle = ego_vehicle
                logger.info("Re-established agent's vehicle reference")
                
                # Clear control state
                if hasattr(agent, '_last_control'):
                    agent._last_control = None
                if hasattr(agent, 'control'):
                    agent.control = carla.VehicleControl()
                if hasattr(agent, '_vehicle_control'):
                    agent._vehicle_control = carla.VehicleControl()
                
                # CRITICAL: Reset the agent's action state
                if hasattr(agent, '_action'):
                    agent._action = None
                    logger.info("Cleared agent's pending action")
                
                # Clear any pending actions
                if hasattr(agent, 'set_action'):
                    agent.set_action({'throttle': 0.0, 'brake': 0.0, 'steer': 0.0})
                    logger.info("Reset agent action to neutral")
            
            # Ensure scenario manager is in proper running state
            if hasattr(sim_state.leaderboard_evaluator, 'manager') and sim_state.leaderboard_evaluator.manager:
                manager = sim_state.leaderboard_evaluator.manager
                
                # Make sure the manager is in running state
                if hasattr(manager, '_running'):
                    manager._running = True
                    logger.info("Set manager to running state")
                
                # Update route scenario ego reference if needed
                if hasattr(manager, '_route_scenario') and ego_vehicle:
                    route_scenario = manager._route_scenario
                    if hasattr(route_scenario, 'ego_vehicles') and len(route_scenario.ego_vehicles) > 0:
                        route_scenario.ego_vehicles[0] = ego_vehicle
                        logger.info("Updated route_scenario's ego_vehicles")
                
                # Ensure the ego vehicle control is properly set
                if ego_vehicle:
                    # Make sure autopilot is off
                    ego_vehicle.set_autopilot(False)
                    # CRITICAL: Enable physics simulation
                    ego_vehicle.set_simulate_physics(True)
                    
                    # Wake up the physics engine with a tiny impulse
                    wake_impulse = carla.Vector3D(x=0.01, y=0, z=0)
                    ego_vehicle.add_impulse(wake_impulse)
                    
                    # Apply a neutral control to clear any stuck state
                    ego_vehicle.apply_control(carla.VehicleControl())
                    logger.info("Reset ego vehicle control state and enabled physics")
                
                # CRITICAL FIX: Re-initialize sensors after restore
                # The sensors need to be recreated at the restored vehicle position
                if ego_vehicle and hasattr(manager, '_agent_wrapper') and manager._agent_wrapper:
                    logger.info("Re-initializing sensors after restore")
                    agent_wrapper = manager._agent_wrapper
                    
                    # Clean up old sensors that are at the wrong position
                    if hasattr(agent_wrapper, '_sensors_list'):
                        logger.info(f"Cleaning up {len(agent_wrapper._sensors_list)} old sensors")
                        try:
                            # First cleanup existing sensors
                            agent_wrapper.cleanup()
                            
                            # IMPORTANT: Clear the sensor registry to avoid duplication errors
                            if hasattr(agent_wrapper, '_agent') and agent_wrapper._agent:
                                agent = agent_wrapper._agent
                                # Clear any sensor tracking in the agent
                                if hasattr(agent, 'sensor_interface'):
                                    sensor_interface = agent.sensor_interface
                                    if hasattr(sensor_interface, '_sensors_objects'):
                                        sensor_interface._sensors_objects.clear()
                                        logger.info("Cleared sensor interface registry")
                                        
                        except Exception as e:
                            logger.warning(f"Error cleaning up sensors: {e}")
                    
                    # Re-setup sensors on the restored vehicle
                    logger.info("Setting up fresh sensors on restored vehicle")
                    try:
                        agent_wrapper.setup_sensors(ego_vehicle)
                        logger.info("Sensors re-initialized at restored position")
                        
                        # Apply vehicle control state
                        ego_vehicle.set_autopilot(False)
                        ego_vehicle.set_simulate_physics(True)
                        ego_vehicle.apply_control(carla.VehicleControl())
                        
                    except Exception as e:
                        logger.error(f"Failed to re-setup sensors: {e}")
                        traceback.print_exc()
        
            else:
                logger.warning("Manager not found after restore")
                
        except Exception as e:
            logger.warning(f"Warning during post-restore sync: {e}")
            traceback.print_exc()
        
        # Call comprehensive post-restore initialization
        if sim_state.leaderboard_evaluator:
            _post_restore_init(sim_state.leaderboard_evaluator)
            
            # Optional: One safety tick for sensors to deliver first frame
            # Uncomment if you see empty images after restore
            # if world:
            #     world.tick()
            #     logger.info("Applied one safety tick for sensor initialization")
        
        # CRITICAL: Use the observation saved in the snapshot, NOT a new one!
        # This ensures we get the exact same images/sensor data from when the snapshot was taken
        if hasattr(snapshot, 'observation') and snapshot.observation:
            obs = snapshot.observation
            logger.info(f"Using saved observation from snapshot with {len(obs.get('images', {}))} images")
            sim_state.last_observation = obs
        else:
            # Fallback: Get new observation (this is the problematic path)
            logger.warning("No saved observation in snapshot - generating new one (WILL BE DIFFERENT!)")
            try:
                obs = get_observation_from_state(sim_state.leaderboard_evaluator)
                sim_state.last_observation = obs
            except Exception as e:
                logger.warning(f"Could not get observation after restore: {e}")
                obs = sim_state.last_observation if sim_state.last_observation else {}
        
        # LOG CRITICAL DEBUG INFO
        if obs and 'vehicle_state' in obs and 'position' in obs['vehicle_state']:
            actual_pos = obs['vehicle_state']['position']
            # Handle both dict and list formats for position
            if isinstance(actual_pos, dict):
                actual_x = actual_pos.get('x', 0)
                actual_y = actual_pos.get('y', 0)
            else:
                actual_x = actual_pos[0] if len(actual_pos) > 0 else 0
                actual_y = actual_pos[1] if len(actual_pos) > 1 else 0
            
            logger.error(f"FINAL POSITION AFTER RESTORE: x={actual_x:.2f}, y={actual_y:.2f}")
            # Check against what we tried to restore
            if hasattr(snapshot, 'vehicles'):
                for _, vehicle_state in snapshot.vehicles.items():
                    if vehicle_state.is_hero:
                        expected_x = vehicle_state.location['x']
                        drift = actual_x - expected_x
                        logger.error(f"POSITION DRIFT: {drift:.2f}m (expected x={expected_x:.2f}, got x={actual_x:.2f}")
                        break
        
        return {
            "status": "success",
            "message": f"Restored from snapshot: {request.snapshot_id}",
            "step_count": sim_state.step_count,
            "observation": obs
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring snapshot: {e}")
        import traceback
        traceback.print_exc()
        # Ensure we return meaningful error message
        error_msg = f"Restore failed: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/restore2")
async def restore_snapshot_v2(request: RestoreRequest):
    """
    Version 2 of restore using only building block functions.
    This version rebuilds the scenario from scratch using the saved state.
    """
    logger.info(f"Restore2 endpoint called with snapshot_id: {request.snapshot_id}")
    
    try:
        # Check if evaluator exists
        if sim_state.leaderboard_evaluator is None:
            raise HTTPException(status_code=400, detail="Environment not initialized")
        
        leaderboard_evaluator_self = sim_state.leaderboard_evaluator
        
        # 1. Load snapshot
        # snapshot = None
        # if request.snapshot_id in sim_state.saved_snapshots:
        #     snapshot = sim_state.saved_snapshots[request.snapshot_id]
        #     logger.info(f"Loaded snapshot from memory: {request.snapshot_id}")
        # else:
            # Try loading from disk
        snapshot_file = sim_state.snapshot_dir / f"{request.snapshot_id}.pkl"
        if snapshot_file.exists():
            with open(snapshot_file, 'rb') as f:
                snapshot = pickle.load(f)
            logger.info(f"Loaded snapshot from disk: {snapshot_file}")
        else:
            raise HTTPException(status_code=404, detail=f"Snapshot not found: {request.snapshot_id}")
        
        # 2. Get world reference
        world = leaderboard_evaluator_self.world
        if not world:
            raise HTTPException(status_code=500, detail="No world object available")
        
        logger.info("Starting restore2 with building blocks...")
        
        # 3. Pause the scenario manager to prevent interference
        was_running = False
        if hasattr(leaderboard_evaluator_self, 'manager') and leaderboard_evaluator_self.manager:
            manager = leaderboard_evaluator_self.manager
            # Save the running state
            was_running = getattr(manager, '_running', False)
            manager._running = False
            logger.info(f"Paused scenario manager (was_running={was_running})")
        
        # 4. Restore vehicles using WorldSnapshot's restore method
        logger.info("Restoring world state from snapshot...")
        success = snapshot.restore(sim_state, world)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to restore world state")
        
        # 5. Re-setup the agent on the restored ego vehicle
        ego_vehicle = None
        vehicles = world.get_actors().filter('vehicle.*')
        for vehicle in vehicles:
            if 'hero' in vehicle.attributes.get('role_name', '').lower():
                ego_vehicle = vehicle
                break
        
        if not ego_vehicle:
            logger.error("Could not find ego vehicle after restore")
            raise HTTPException(status_code=500, detail="Ego vehicle not found after restore")
        
        logger.info(f"Found ego vehicle at restored position: ID={ego_vehicle.id}")
        
        # 6. Update agent's vehicle reference if using APIAgent
        if hasattr(leaderboard_evaluator_self, 'agent_instance'):
            agent = leaderboard_evaluator_self.agent_instance
            
            # For APIAgent, update the vehicle reference
            if hasattr(agent, '_vehicle'):
                agent._vehicle = ego_vehicle
                logger.info("Updated agent's vehicle reference")
            
            # Update sensors wrapper if it exists
            if hasattr(agent, '_sensors_wrapper'):
                # Clean up old sensors
                if hasattr(agent._sensors_wrapper, 'cleanup'):
                    agent._sensors_wrapper.cleanup()
                
                # Re-setup sensors on new vehicle
                agent._sensors_wrapper.setup_sensors(ego_vehicle)
                logger.info("Re-setup sensors on restored vehicle")
        
        # 7. Update scenario manager references
        if hasattr(leaderboard_evaluator_self, 'manager') and leaderboard_evaluator_self.manager:
            manager = leaderboard_evaluator_self.manager
            
            # Update ego vehicle in route scenario
            if hasattr(manager, '_route_scenario') and manager._route_scenario:
                route_scenario = manager._route_scenario
                if hasattr(route_scenario, 'ego_vehicles'):
                    route_scenario.ego_vehicles = [ego_vehicle]
                    logger.info("Updated route scenario ego vehicles")
            
            # Update ego vehicle in scenario behaviors
            if hasattr(manager, '_scenario_behaviors'):
                for behavior in manager._scenario_behaviors:
                    if hasattr(behavior, 'ego_vehicles'):
                        behavior.ego_vehicles = [ego_vehicle]
            
            # Resume scenario manager if it was running
            if was_running:
                manager._running = True
                logger.info("Resumed scenario manager")
        
        # 8. Ensure the ego vehicle is ready
        ego_vehicle.set_autopilot(False)
        ego_vehicle.set_simulate_physics(True)
        
        # 9. Restore step count
        if hasattr(snapshot, 'step_count'):
            sim_state.step_count = snapshot.step_count
            logger.info(f"Restored step count to {snapshot.step_count}")
        
        # 10. Get observation - prefer saved one from snapshot
        if hasattr(snapshot, 'observation') and snapshot.observation:
            obs = snapshot.observation
            logger.info(f"Using saved observation from snapshot")
        else:
            logger.warning("No saved observation - generating new one")
            obs = get_observation_from_state(leaderboard_evaluator_self)
        
        sim_state.last_observation = obs
        
        # 11. Mark that we just restored (for the next tick)
        # Set flag on the evaluator so my_run_scenario_step can check it
        leaderboard_evaluator_self.just_restored = True
        
        logger.info(f"Restore2 complete - ego at position from snapshot")
        
        return {
            "status": "success",
            "message": f"Restored using building blocks: {request.snapshot_id}",
            "step_count": sim_state.step_count,
            "observation": obs,
            "method": "restore2_building_blocks"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in restore2: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Restore2 failed: {str(e)}")

@app.get("/snapshots")
async def list_snapshots():
    """List all available snapshots from disk and memory"""
    snapshots_list = []
    
    # Get snapshots from memory
    for snapshot_id in sim_state.snapshots:
        snapshots_list.append({
            "id": snapshot_id,
            "source": "memory",
            "available": True
        })
    
    # Get snapshots from disk
    if sim_state.snapshot_dir.exists():
        for snapshot_file in sim_state.snapshot_dir.glob("*.pkl"):
            snapshot_id = snapshot_file.stem
            # Add if not already in memory list
            if snapshot_id not in sim_state.snapshots:
                snapshots_list.append({
                    "id": snapshot_id,
                    "source": "disk",
                    "file": str(snapshot_file),
                    "size_kb": snapshot_file.stat().st_size / 1024,
                    "available": True
                })
            else:
                # Update existing entry with disk info
                for item in snapshots_list:
                    if item["id"] == snapshot_id:
                        item["file"] = str(snapshot_file)
                        item["size_kb"] = snapshot_file.stat().st_size / 1024
                        item["source"] = "memory+disk"
    
    return {
        "snapshots": snapshots_list,
        "count": len(snapshots_list),
        "snapshot_directory": str(sim_state.snapshot_dir.absolute())
    }

@app.get("/snapshot/{snapshot_id}")
async def get_snapshot_details(snapshot_id: str):
    """Get detailed information about a specific snapshot"""
    # Check memory first
    if snapshot_id in sim_state.snapshots:
        snapshot = sim_state.snapshots[snapshot_id]
    else:
        # Try loading from disk
        snapshot_file = sim_state.snapshot_dir / f"{snapshot_id}.pkl"
        if snapshot_file.exists():
            try:
                with open(snapshot_file, 'rb') as f:
                    snapshot = pickle.load(f)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load snapshot from disk: {e}")
        else:
            raise HTTPException(status_code=404, detail=f"Snapshot not found: {snapshot_id}")
    
    # Build detailed info
    info = {
        "snapshot_id": snapshot_id,
        "vehicles": {},
        "world_settings": {},
        "scenario_manager": None,
        "step_count": getattr(snapshot, 'step_count', None)
    }
    
    # Extract vehicle info
    if hasattr(snapshot, 'vehicles'):
        for vid, vstate in snapshot.vehicles.items():
            info["vehicles"][str(vid)] = {
                "is_hero": vstate.is_hero,
                "type_id": vstate.type_id,
                "location": vstate.location,
                "rotation": vstate.rotation,
                "velocity": vstate.velocity
            }
    
    # Extract world settings
    if hasattr(snapshot, 'world_settings'):
        info["world_settings"] = snapshot.world_settings
    
    # Extract scenario manager info
    if hasattr(snapshot, 'scenario_manager') and snapshot.scenario_manager:
        info["scenario_manager"] = {
            "tick_count": snapshot.scenario_manager.tick_count,
            "running": snapshot.scenario_manager.running,
            "ego_vehicle_ids": snapshot.scenario_manager.ego_vehicle_ids
        }
    
    return info

@app.post("/restore3")
async def restore_snapshot_v3(request: RestoreRequest):
    """
    Version 3 of restore that properly handles other_actors positions.
    This version ensures NPCs maintain their exact positions from the snapshot.
    """
    logger.info(f"Restore3 endpoint called with snapshot_id: {request.snapshot_id}")
    try:
        if sim_state.leaderboard_evaluator is None:
            raise HTTPException(status_code=400, detail="Environment not initialized")
        
        if WorldSnapshot is None:
            raise HTTPException(status_code=501, detail="Snapshot feature not available")
        
        # Get snapshot
        if request.snapshot_id in sim_state.snapshots:
            snapshot = sim_state.snapshots[request.snapshot_id]
            logger.info(f"Loading snapshot from memory: {request.snapshot_id}")
        else:
            snapshot_file = sim_state.snapshot_dir / f"{request.snapshot_id}.pkl"
            if snapshot_file.exists():
                try:
                    with open(snapshot_file, 'rb') as f:
                        snapshot = pickle.load(f)
                    logger.info(f"Loaded snapshot from disk: {snapshot_file}")
                    sim_state.snapshots[request.snapshot_id] = snapshot
                except Exception as e:
                    logger.error(f"Failed to load snapshot from disk: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to load snapshot: {e}")
            else:
                raise HTTPException(status_code=404, detail=f"Snapshot not found: {request.snapshot_id}")
        
        # Get world reference
        world = None
        if hasattr(sim_state.leaderboard_evaluator, 'world'):
            world = sim_state.leaderboard_evaluator.world
        
        if world is None:
            logger.error("No world object available for restore")
            raise HTTPException(status_code=500, detail="No world object available")
            
        # Ensure sync mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 20.0
        world.apply_settings(settings)
        
        # CRITICAL: Pause scenario manager
        original_running = False
        manager = None
        if hasattr(sim_state.leaderboard_evaluator, 'manager') and sim_state.leaderboard_evaluator.manager:
            manager = sim_state.leaderboard_evaluator.manager
            original_running = getattr(manager, '_running', False)
            manager._running = False
            logger.info(f"Paused scenario manager (was running: {original_running})")
        
        # CRITICAL NEW STEP: Build the other_actors list BEFORE restore
        # This list maps snapshot vehicle IDs to actual CARLA actor objects
        other_actors_map = {}
        
        if hasattr(snapshot, 'vehicles') and snapshot.vehicles:
            current_vehicles = world.get_actors().filter('vehicle.*')
            logger.info(f"Found {len(current_vehicles)} current vehicles in world")
            
            # First identify ego vehicle
            ego_vehicle_id = None
            if manager and hasattr(manager, 'ego_vehicles') and len(manager.ego_vehicles) > 0:
                ego_vehicle_id = manager.ego_vehicles[0].id
                logger.info(f"Ego vehicle ID: {ego_vehicle_id}")
            
            # Build mapping of snapshot vehicles to current actors
            for snap_vid, snap_vstate in snapshot.vehicles.items():
                if not snap_vstate.is_hero:  # Only care about NPCs
                    # Find closest matching vehicle by position
                    best_match = None
                    best_dist = float('inf')
                    
                    for vehicle in current_vehicles:
                        if ego_vehicle_id and vehicle.id == ego_vehicle_id:
                            continue  # Skip ego
                        
                        loc = vehicle.get_location()
                        # Calculate distance to snapshot position
                        dist = ((loc.x - snap_vstate.location['x'])**2 + 
                               (loc.y - snap_vstate.location['y'])**2 + 
                               (loc.z - snap_vstate.location['z'])**2) ** 0.5
                        
                        if dist < best_dist and dist < 100:  # Within 100m
                            best_dist = dist
                            best_match = vehicle
                    
                    if best_match:
                        other_actors_map[snap_vid] = best_match
                        logger.info(f"Mapped snapshot vehicle {snap_vid} to actor {best_match.id} (dist: {best_dist:.2f}m)")
        
        # CRITICAL: DO NOT use snapshot.restore() - it respawns NPCs!
        # Instead, manually restore vehicle positions without creating new vehicles
        if isinstance(snapshot, WorldSnapshot):
            try:
                logger.info("Using manual restore to preserve NPCs...")
                
                # 1. Get current vehicles in the world
                current_vehicles = world.get_actors().filter('vehicle.*')
                logger.info(f"Current vehicles before restore: {len(current_vehicles)}")
                
                # 2. Identify ego vehicle
                ego_vehicle = None
                if manager and hasattr(manager, 'ego_vehicles') and len(manager.ego_vehicles) > 0:
                    ego_vehicle = manager.ego_vehicles[0]
                    logger.info(f"Ego vehicle ID: {ego_vehicle.id}")
                
                # 3. Restore ego vehicle position from snapshot
                if ego_vehicle and hasattr(snapshot, 'vehicles') and snapshot.vehicles:
                    for _, vstate in snapshot.vehicles.items():
                        if vstate.is_hero:
                            transform = carla.Transform(
                                carla.Location(x=vstate.location['x'], y=vstate.location['y'], z=vstate.location['z']),
                                carla.Rotation(pitch=vstate.rotation['pitch'], yaw=vstate.rotation['yaw'], roll=vstate.rotation['roll'])
                            )
                            
                            # Set ego position
                            ego_vehicle.set_simulate_physics(False)
                            ego_vehicle.set_transform(transform)
                            ego_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                            ego_vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                            ego_vehicle.set_simulate_physics(True)
                            
                            logger.info(f"Restored ego to: x={vstate.location['x']:.2f}, y={vstate.location['y']:.2f}")
                            break
                
                # 4. Restore NPC positions - match by closest position at snapshot time
                npc_vehicles = []
                for vehicle in current_vehicles:
                    if ego_vehicle and vehicle.id == ego_vehicle.id:
                        continue
                    npc_vehicles.append(vehicle)
                
                logger.info(f"Found {len(npc_vehicles)} NPCs to restore")
                
                # Match NPCs to snapshot positions and restore them
                if hasattr(snapshot, 'vehicles') and snapshot.vehicles:
                    for snap_id, vstate in snapshot.vehicles.items():
                        if vstate.is_hero:
                            continue  # Skip ego
                        
                        # Find closest NPC to this snapshot position
                        best_match = None
                        best_dist = float('inf')
                        
                        for npc in npc_vehicles:
                            loc = npc.get_location()
                            # Use a larger threshold since vehicles may have moved
                            dist = ((loc.x - vstate.location['x'])**2 + 
                                   (loc.y - vstate.location['y'])**2) ** 0.5
                            
                            if dist < best_dist:
                                best_dist = dist
                                best_match = npc
                        
                        if best_match and best_dist < 200:  # Within 200m
                            # Restore this NPC's position
                            transform = carla.Transform(
                                carla.Location(x=vstate.location['x'], y=vstate.location['y'], z=vstate.location['z']),
                                carla.Rotation(pitch=vstate.rotation['pitch'], yaw=vstate.rotation['yaw'], roll=vstate.rotation['roll'])
                            )
                            
                            best_match.set_simulate_physics(False)
                            best_match.set_transform(transform)
                            best_match.set_target_velocity(carla.Vector3D(
                                vstate.velocity['x'], vstate.velocity['y'], vstate.velocity['z']
                            ))
                            best_match.set_simulate_physics(True)
                            
                            logger.info(f"Restored NPC {best_match.id} to x={vstate.location['x']:.2f}, y={vstate.location['y']:.2f} (dist: {best_dist:.2f}m)")
                
                # 5. Restore simulation state
                if hasattr(snapshot, 'step_count'):
                    sim_state.step_count = snapshot.step_count
                    logger.info(f"Restored step count to {snapshot.step_count}")
                
                if hasattr(snapshot, 'metrics') and snapshot.metrics:
                    sim_state.cumulative_reward = snapshot.metrics.cumulative_reward
                
                # 6. Restore scenario manager state WITHOUT respawning
                if manager and hasattr(snapshot, 'scenario_manager') and snapshot.scenario_manager:
                    # Just restore the tick count, don't recreate scenarios
                    if hasattr(manager, '_tick'):
                        manager._tick = snapshot.scenario_manager.tick_count
                        logger.info(f"Restored scenario manager tick to {snapshot.scenario_manager.tick_count}")
                
            except Exception as e:
                logger.error(f"Manual restore failed: {e}")
                if manager:
                    manager._running = original_running
                raise HTTPException(status_code=500, detail=str(e))
        else:
            # Old-style snapshot
            sim_state.step_count = snapshot["step_count"]
            sim_state.cumulative_reward = snapshot["cumulative_reward"]
            sim_state.last_observation = snapshot["last_observation"]
        
        # CRITICAL: Update the other_actors list in RouteScenario
        # This ensures the scenario manager knows about the correct vehicle positions
        if manager and hasattr(manager, '_route_scenario'):
            route_scenario = manager._route_scenario
            
            # Clear old other_actors list
            if hasattr(route_scenario, 'other_actors'):
                route_scenario.other_actors = []
                logger.info("Cleared old other_actors list")
            
            # Rebuild other_actors list with correct actors
            current_vehicles = world.get_actors().filter('vehicle.*')
            for vehicle in current_vehicles:
                if ego_vehicle_id and vehicle.id == ego_vehicle_id:
                    continue  # Skip ego
                    
                # Add to other_actors list
                route_scenario.other_actors.append(vehicle)
                logger.info(f"Added vehicle {vehicle.id} to other_actors")
            
            logger.info(f"Rebuilt other_actors list with {len(route_scenario.other_actors)} NPCs")
            
            # Also update the scenario manager's reference
            if hasattr(manager, 'other_actors'):
                manager.other_actors = route_scenario.other_actors
                logger.info("Updated scenario manager's other_actors reference")
        
        # Set flags to prevent scenario reset
        sim_state.just_restored = True
        if sim_state.leaderboard_evaluator:
            sim_state.leaderboard_evaluator.just_restored = True
        logger.info("Set just_restored flags to prevent scenario reset")
        
        # Restore scenario manager state
        if manager:
            manager._running = original_running
            logger.info(f"Restored scenario manager running state: {original_running}")
        
        # Re-establish connections
        if hasattr(sim_state.leaderboard_evaluator, 'world'):
            world = sim_state.leaderboard_evaluator.world
            client = sim_state.leaderboard_evaluator.client if hasattr(sim_state.leaderboard_evaluator, 'client') else None
            traffic_manager = sim_state.leaderboard_evaluator.traffic_manager if hasattr(sim_state.leaderboard_evaluator, 'traffic_manager') else None
            
            if client and world and traffic_manager:
                CarlaDataProvider.set_client(client)
                CarlaDataProvider.set_world(world)
                CarlaDataProvider.set_traffic_manager_port(traffic_manager.get_port())
                traffic_manager.set_synchronous_mode(True)
                traffic_manager.set_hybrid_physics_mode(True)
                logger.info("Re-established connections and traffic manager settings")
        
        # CRITICAL: Must generate NEW observation after moving vehicles!
        # The saved observation is from the wrong position
        logger.info("Generating fresh observation after restore...")
        
        # Need to tick once to update sensors at new position
        if world:
            world.tick()
            logger.info("Ticked world to update sensor positions")
        
        # Now get observation from the RESTORED position
        obs = get_observation_from_state(sim_state.leaderboard_evaluator)
        sim_state.last_observation = obs
        
        # Verify position in observation
        if obs and 'vehicle_state' in obs and 'position' in obs['vehicle_state']:
            pos = obs['vehicle_state']['position']
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                logger.info(f"Observation taken at position: x={pos[0]:.2f}, y={pos[1]:.2f}")
            elif isinstance(pos, dict) and 'x' in pos and 'y' in pos:
                logger.info(f"Observation taken at position: x={pos['x']:.2f}, y={pos['y']:.2f}")
        
        logger.info(f"Restore3 complete - NPCs properly positioned")
        
        return {
            "status": "success",
            "message": f"Restored with proper NPC handling: {request.snapshot_id}",
            "step_count": sim_state.step_count,
            "observation": obs,
            "method": "restore3_with_other_actors"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in restore3: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Restore3 failed: {str(e)}")

@app.post("/restore4")
async def restore_snapshot_v4(request: RestoreRequest):
    """
    Version 4 of restore that stops and restarts the scenario to prevent NPC respawning.
    This ensures NPCs maintain their positions without being recreated.
    """
    logger.info(f"Restore4 endpoint called with snapshot_id: {request.snapshot_id}")
    try:
        if sim_state.leaderboard_evaluator is None:
            raise HTTPException(status_code=400, detail="Environment not initialized")
        
        if WorldSnapshot is None:
            raise HTTPException(status_code=501, detail="Snapshot feature not available")
        
        # Get snapshot
        if request.snapshot_id in sim_state.snapshots:
            snapshot = sim_state.snapshots[request.snapshot_id]
            logger.info(f"Loading snapshot from memory: {request.snapshot_id}")
        else:
            snapshot_file = sim_state.snapshot_dir / f"{request.snapshot_id}.pkl"
            if snapshot_file.exists():
                try:
                    with open(snapshot_file, 'rb') as f:
                        snapshot = pickle.load(f)
                    logger.info(f"Loaded snapshot from disk: {snapshot_file}")
                    sim_state.snapshots[request.snapshot_id] = snapshot
                except Exception as e:
                    logger.error(f"Failed to load snapshot from disk: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to load snapshot: {e}")
            else:
                raise HTTPException(status_code=404, detail=f"Snapshot not found: {request.snapshot_id}")
        
        # Get world reference
        world = None
        if hasattr(sim_state.leaderboard_evaluator, 'world'):
            world = sim_state.leaderboard_evaluator.world
        
        if world is None:
            logger.error("No world object available for restore")
            raise HTTPException(status_code=500, detail="No world object available")
            
        # Get manager reference
        manager = None
        if hasattr(sim_state.leaderboard_evaluator, 'manager') and sim_state.leaderboard_evaluator.manager:
            manager = sim_state.leaderboard_evaluator.manager
            logger.info("Got scenario manager reference")
        
        # CRITICAL STEP 1: Stop the scenario to prevent NPC spawning
        if manager:
            # Stop the running flag first
            manager._running = False
            logger.info("Set manager._running to False")
            
            # Wait for threads to stop
            if hasattr(manager, '_scenario_thread') and manager._scenario_thread:
                logger.info("Waiting for scenario thread to stop...")
                manager._scenario_thread.join(timeout=2.0)
                logger.info("Scenario thread stopped")
        
        # STEP 2: Manually restore vehicle positions
        logger.info("Manually restoring vehicle positions...")
        
        # Get current vehicles
        current_vehicles = world.get_actors().filter('vehicle.*')
        logger.info(f"Found {len(current_vehicles)} vehicles in world")
        
        # Identify ego vehicle
        ego_vehicle = None
        if manager and hasattr(manager, 'ego_vehicles') and len(manager.ego_vehicles) > 0:
            ego_vehicle = manager.ego_vehicles[0]
            logger.info(f"Ego vehicle ID: {ego_vehicle.id}")
        
        # Restore ego position
        if ego_vehicle and hasattr(snapshot, 'vehicles') and snapshot.vehicles:
            for _, vstate in snapshot.vehicles.items():
                if vstate.is_hero:
                    transform = carla.Transform(
                        carla.Location(x=vstate.location['x'], y=vstate.location['y'], z=vstate.location['z']),
                        carla.Rotation(pitch=vstate.rotation['pitch'], yaw=vstate.rotation['yaw'], roll=vstate.rotation['roll'])
                    )
                    
                    ego_vehicle.set_simulate_physics(False)
                    ego_vehicle.set_transform(transform)
                    ego_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                    ego_vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                    ego_vehicle.set_simulate_physics(True)
                    
                    logger.info(f"Restored ego to: x={vstate.location['x']:.2f}, y={vstate.location['y']:.2f}")
                    break
        
        # Collect NPC vehicles
        npc_vehicles = []
        for vehicle in current_vehicles:
            if ego_vehicle and vehicle.id == ego_vehicle.id:
                continue
            npc_vehicles.append(vehicle)
        
        logger.info(f"Found {len(npc_vehicles)} NPCs")
        
        # Restore NPC positions
        if hasattr(snapshot, 'vehicles') and snapshot.vehicles:
            restored_count = 0
            for snap_id, vstate in snapshot.vehicles.items():
                if vstate.is_hero:
                    continue
                
                # Find best matching NPC
                best_match = None
                best_dist = float('inf')
                
                for npc in npc_vehicles:
                    loc = npc.get_location()
                    dist = ((loc.x - vstate.location['x'])**2 + 
                           (loc.y - vstate.location['y'])**2) ** 0.5
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_match = npc
                
                if best_match and best_dist < 200:
                    transform = carla.Transform(
                        carla.Location(x=vstate.location['x'], y=vstate.location['y'], z=vstate.location['z']),
                        carla.Rotation(pitch=vstate.rotation['pitch'], yaw=vstate.rotation['yaw'], roll=vstate.rotation['roll'])
                    )
                    
                    best_match.set_simulate_physics(False)
                    best_match.set_transform(transform)
                    best_match.set_target_velocity(carla.Vector3D(
                        vstate.velocity['x'], vstate.velocity['y'], vstate.velocity['z']
                    ))
                    best_match.set_simulate_physics(True)
                    
                    restored_count += 1
                    logger.info(f"Restored NPC {best_match.id} (dist was {best_dist:.2f}m)")
            
            logger.info(f"Restored {restored_count} NPCs to snapshot positions")
        
        # STEP 3: Update other_actors list
        if manager:
            # Update scenario's other_actors list
            if hasattr(manager, '_route_scenario') and manager._route_scenario:
                route_scenario = manager._route_scenario
                
                # Clear and rebuild other_actors
                route_scenario.other_actors = []
                for npc in npc_vehicles:
                    route_scenario.other_actors.append(npc)
                
                logger.info(f"Updated route_scenario.other_actors with {len(route_scenario.other_actors)} NPCs")
                
                # Also update manager's reference
                manager.other_actors = route_scenario.other_actors
                logger.info("Updated manager.other_actors reference")
            
            # Update the scenario object reference
            if hasattr(manager, 'scenario') and manager.scenario:
                manager.scenario.other_actors = npc_vehicles
                logger.info(f"Updated manager.scenario.other_actors with {len(npc_vehicles)} NPCs")
        
        # STEP 4: Restore simulation state
        if hasattr(snapshot, 'step_count'):
            sim_state.step_count = snapshot.step_count
            logger.info(f"Restored step count to {snapshot.step_count}")
        
        if hasattr(snapshot, 'metrics') and snapshot.metrics:
            sim_state.cumulative_reward = snapshot.metrics.cumulative_reward
        
        # STEP 5: Restart the scenario WITHOUT respawning
        if manager:
            # Set flags to prevent scenario from resetting
            sim_state.just_restored = True
            if sim_state.leaderboard_evaluator:
                sim_state.leaderboard_evaluator.just_restored = True
            
            # Restart the running flag but NOT the thread
            # This prevents build_scenarios_loop from spawning new vehicles
            manager._running = True
            logger.info("Set manager._running back to True (but not restarting thread)")
            
            # Update tick count if available
            if hasattr(snapshot, 'scenario_manager') and snapshot.scenario_manager:
                if hasattr(manager, '_tick'):
                    manager._tick = snapshot.scenario_manager.tick_count
                    logger.info(f"Restored manager tick to {snapshot.scenario_manager.tick_count}")
                if hasattr(manager, 'tick_count'):
                    manager.tick_count = snapshot.scenario_manager.tick_count
        
        # STEP 6: Re-establish connections
        if hasattr(sim_state.leaderboard_evaluator, 'world'):
            world = sim_state.leaderboard_evaluator.world
            client = sim_state.leaderboard_evaluator.client if hasattr(sim_state.leaderboard_evaluator, 'client') else None
            traffic_manager = sim_state.leaderboard_evaluator.traffic_manager if hasattr(sim_state.leaderboard_evaluator, 'traffic_manager') else None
            
            if client and world and traffic_manager:
                CarlaDataProvider.set_client(client)
                CarlaDataProvider.set_world(world)
                CarlaDataProvider.set_traffic_manager_port(traffic_manager.get_port())
                traffic_manager.set_synchronous_mode(True)
                traffic_manager.set_hybrid_physics_mode(True)
                logger.info("Re-established connections")
        
        # STEP 7: Get fresh observation from restored position
        logger.info("Generating observation from restored position...")
        
        # Tick once to update sensors
        if world:
            world.tick()
            logger.info("Ticked world to update sensors")
        
        obs = get_observation_from_state(sim_state.leaderboard_evaluator)
        sim_state.last_observation = obs
        
        # Verify position
        if obs and 'vehicle_state' in obs and 'position' in obs['vehicle_state']:
            pos = obs['vehicle_state']['position']
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                logger.info(f"Observation at: x={pos[0]:.2f}, y={pos[1]:.2f}")
            elif isinstance(pos, dict):
                logger.info(f"Observation at: x={pos.get('x', 0):.2f}, y={pos.get('y', 0):.2f}")
        
        logger.info(f"Restore4 complete - scenario stopped/restarted, NPCs preserved")
        
        return {
            "status": "success",
            "message": f"Restored with scenario restart: {request.snapshot_id}",
            "step_count": sim_state.step_count,
            "observation": obs,
            "method": "restore4_stop_restart_scenario"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in restore4: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Restore4 failed: {str(e)}")

@app.post("/restore5")
async def restore_snapshot_v5(request: RestoreRequest):
    """
    Version 5: Simple position restore without triggering scenario rebuilds.
    - Only restores vehicle positions
    - Completely blocks scenario rebuilding
    - Keeps existing scenario state intact
    """
    logger.info(f"Restore5 endpoint called with snapshot_id: {request.snapshot_id}")
    try:
        if sim_state.leaderboard_evaluator is None:
            raise HTTPException(status_code=400, detail="Environment not initialized")
        
        # Get snapshot
        if request.snapshot_id in sim_state.snapshots:
            snapshot = sim_state.snapshots[request.snapshot_id]
        else:
            snapshot_file = sim_state.snapshot_dir / f"{request.snapshot_id}.pkl"
            if snapshot_file.exists():
                with open(snapshot_file, 'rb') as f:
                    snapshot = pickle.load(f)
                sim_state.snapshots[request.snapshot_id] = snapshot
            else:
                raise HTTPException(status_code=404, detail=f"Snapshot not found")
        
        world = sim_state.leaderboard_evaluator.world
        if world is None:
            raise HTTPException(status_code=500, detail="No world object available")
        
        manager = sim_state.leaderboard_evaluator.manager
        if manager is None:
            raise HTTPException(status_code=500, detail="No scenario manager")
        
        # CRITICAL: Stop ALL scenario activities
        original_running = manager._running
        manager._running = False
        logger.info("Stopped scenario manager completely")
        
        # Wait for any ongoing operations
        import time
        time.sleep(0.5)
        
        # Get RouteScenario and DISABLE its update mechanism
        route_scenario = manager._route_scenario if hasattr(manager, '_route_scenario') else manager.scenario
        if route_scenario:
            # Block the build_scenarios method completely
            def blocked_build_scenarios(*args, **kwargs):
                logger.debug("build_scenarios BLOCKED during restore")
                return
            
            if hasattr(route_scenario, 'build_scenarios'):
                route_scenario._original_build_scenarios = route_scenario.build_scenarios
                route_scenario.build_scenarios = blocked_build_scenarios
                logger.info("Blocked build_scenarios to prevent respawning")
            
            # Restore ego position first
            ego_vehicle = manager.ego_vehicles[0]
            for _, vstate in snapshot.vehicles.items():
                if vstate.is_hero:
                    transform = carla.Transform(
                        carla.Location(x=vstate.location['x'], y=vstate.location['y'], z=vstate.location['z']),
                        carla.Rotation(pitch=vstate.rotation['pitch'], yaw=vstate.rotation['yaw'], roll=vstate.rotation['roll'])
                    )
                    ego_vehicle.set_simulate_physics(False)
                    ego_vehicle.set_transform(transform)
                    ego_vehicle.set_simulate_physics(True)
                    logger.info(f"Restored ego to x={vstate.location['x']:.2f}, y={vstate.location['y']:.2f}")
                    break
            
            # Restore NPCs to exact positions
            current_vehicles = world.get_actors().filter('vehicle.*')
            npc_vehicles = [v for v in current_vehicles if v.id != ego_vehicle.id]
            
            for snap_id, vstate in snapshot.vehicles.items():
                if vstate.is_hero:
                    continue
                    
                # Find matching NPC and restore position
                for npc in npc_vehicles:
                    loc = npc.get_location()
                    dist = ((loc.x - vstate.location['x'])**2 + (loc.y - vstate.location['y'])**2) ** 0.5
                    if dist < 200:
                        transform = carla.Transform(
                            carla.Location(x=vstate.location['x'], y=vstate.location['y'], z=vstate.location['z']),
                            carla.Rotation(pitch=vstate.rotation['pitch'], yaw=vstate.rotation['yaw'], roll=vstate.rotation['roll'])
                        )
                        npc.set_simulate_physics(False)
                        npc.set_transform(transform)
                        npc.set_simulate_physics(True)
                        break
            
            # Restore other_actors list
            route_scenario.other_actors = npc_vehicles
            manager.other_actors = npc_vehicles
            
            # Remove configs for scenarios we've already restored
            # This prevents them from being re-triggered by build_scenarios
            if 'active_scenarios_data' in scenario_state:
                # Make sure missing_scenario_configurations exists
                if not hasattr(route_scenario, 'missing_scenario_configurations'):
                    route_scenario.missing_scenario_configurations = []
                
                for scenario_data in scenario_state['active_scenarios_data']:
                    scenario_name = scenario_data['name']
                    if hasattr(route_scenario, 'missing_scenario_configurations') and route_scenario.missing_scenario_configurations:
                        route_scenario.missing_scenario_configurations = [
                            config for config in route_scenario.missing_scenario_configurations
                            if hasattr(config, 'name') and config.name != scenario_name
                        ]
                        logger.info(f"Removed {scenario_name} from missing_configs to prevent re-triggering")
        
        # STEP 6: Restore simulation state
        if hasattr(snapshot, 'step_count'):
            sim_state.step_count = snapshot.step_count
        
        # STEP 7: Set flags to prevent immediate rebuilding
        sim_state.just_restored = True
        if hasattr(manager, '_tick'):
            manager._tick = snapshot.scenario_manager.tick_count if hasattr(snapshot, 'scenario_manager') else 0
        
        # STEP 8: Resume scenario manager with protection
        manager._running = original_running
        
        # Add temporary protection for build_scenarios
        if hasattr(route_scenario, 'build_scenarios'):
            original_build = route_scenario.build_scenarios
            skip_count = [3]
            
            def protected_build(ego, debug=False):
                if skip_count[0] > 0:
                    skip_count[0] -= 1
                    logger.info(f"Skipping build_scenarios call ({skip_count[0]} remaining)")
                    return
                return original_build(ego, debug)
            
            route_scenario.build_scenarios = protected_build
            
            def restore_original():
                time.sleep(3)
                route_scenario.build_scenarios = original_build
                manager._restore_in_progress = False
                logger.info("Restored original build_scenarios")
            
            import threading
            threading.Thread(target=restore_original, daemon=True).start()
        
        # STEP 9: Tick and get observation
        world.tick()
        obs = get_observation_from_state(sim_state.leaderboard_evaluator)
        sim_state.last_observation = obs
        
        if obs and 'vehicle_state' in obs and 'position' in obs['vehicle_state']:
            pos = obs['vehicle_state']['position']
            if isinstance(pos, (list, tuple)):
                logger.info(f"Observation at: x={pos[0]:.2f}, y={pos[1]:.2f}")
        
        logger.info("Restore5 complete - scenario state preserved")
        
        return {
            "status": "success",
            "message": f"Restored with scenario preservation: {request.snapshot_id}",
            "step_count": sim_state.step_count,
            "observation": obs,
            "method": "restore5_preserve_scenarios"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in restore5: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Restore5 failed: {str(e)}")

@app.delete("/snapshot/{snapshot_id}")
async def delete_snapshot(snapshot_id: str):
    """Delete a specific snapshot from memory and disk"""
    deleted = False
    
    # Delete from memory
    if snapshot_id in sim_state.snapshots:
        del sim_state.snapshots[snapshot_id]
        deleted = True
    
    # Delete from disk
    snapshot_file = sim_state.snapshot_dir / f"{snapshot_id}.pkl"
    if snapshot_file.exists():
        snapshot_file.unlink()
        deleted = True
    
    if deleted:
        return {"status": "success", "message": f"Deleted snapshot: {snapshot_id}"}
    else:
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {snapshot_id}")

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