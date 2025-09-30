#!/usr/bin/env python3
"""
Visual test for GRPO environment with image capture.

Test workflow:
1. Phase 1: Single mode exploration for 90 steps
2. Save snapshot and enable branching (4 branches)
3. Phase 2: Branching mode for 20 steps with 4 different agent behaviors:
   - Agent 1: Straight (throttle=1.0, steer=0.0)
   - Agent 2: Left turn (throttle=1.0, steer=-1.0)
   - Agent 3: Right turn (throttle=1.0, steer=1.0)
   - Agent 4: Random steering (throttle=1.0, steer=random[-1,1])
"""

import numpy as np
import logging
import argparse
from pathlib import Path
import json
import time
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from typing import Dict, Optional
import random
import subprocess
import requests
import carla
import threading
from datetime import datetime

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from client.grpo_carla_env import GRPOCarlaEnv, EnvStatus

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class GRPOVisualTester:
    """Visual tester for GRPO environment with image capture."""
    
    def __init__(self, base_port: int = 8080, output_dir: str = "grpo_test_images"):
        """
        Initialize GRPO visual tester.

        Args:
            base_port: Base API port for services
            output_dir: Directory to save images
        """
        self.base_port = base_port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different phases
        self.phase_dirs = {
            "phase1_single": self.output_dir / "1_single_exploration_90steps",
            "phase2_branch": self.output_dir / "2_branching_50steps",
            "branch0_straight": self.output_dir / "2_branching_50steps" / "agent0_straight",
            "branch1_left": self.output_dir / "2_branching_50steps" / "agent1_left",
            "phase3_continue": self.output_dir / "3_continue_agent1_50steps"
        }

        for phase_dir in self.phase_dirs.values():
            phase_dir.mkdir(parents=True, exist_ok=True)

        # Create GRPO environment with 2 branches max and better error handling
        logger.info("Creating GRPOCarlaEnv with robust connection handling...")
        self.env = None
        max_retries = 3
        retry_delay = 10

        for attempt in range(max_retries):
            try:
                self.env = GRPOCarlaEnv(
                    num_services=2,
                    base_api_port=base_port,
                    render_mode="rgb_array",
                    max_steps=200,
                    timeout=180.0
                )
                logger.info("GRPOCarlaEnv created successfully")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to create GRPOCarlaEnv: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Failed to create GRPOCarlaEnv after all retries")
                    raise
        
        # Position tracking
        self.position_history = {
            "phase1": [],
            "branch0": [],
            "branch1": [],
            "phase3": []
        }
        
        # Random steering history for agent 1
        self.random_steers = []

        # Snapshot functionality from notebook
        self.vehicles_state = None
        self.watchdog_state = None
        self.snapshot_dir = Path("snapshots")
        self.snapshot_dir.mkdir(exist_ok=True)
        
    def record_vehicles(self, world_obj, save_path=None, save_json=False):
        """Record vehicle states from notebook implementation"""
        actors = world_obj.get_actors()
        vehicles = actors.filter("vehicle.*")

        records = []
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
                "vx": vel.x, "vy": vel.y, "vz": vel.z,
                "wx": ang.x, "wy": ang.y, "wz": ang.z,
                "ax": acc.x, "ay": acc.y, "az": acc.z,
                "throttle": ctrl.throttle,
                "steer": ctrl.steer,
                "brake": ctrl.brake,
                "hand_brake": ctrl.hand_brake,
                "reverse": ctrl.reverse,
                "gear": ctrl.gear,
            }
            records.append(record)

        if save_path:
            if save_json:
                with open(save_path, "w") as f:
                    json.dump(records, f, indent=4)
                print(f"Saved {len(records)} vehicle records to {save_path} (JSON)")
            else:
                import csv
                with open(save_path, mode="w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)
                print(f"Saved {len(records)} vehicle records to {save_path} (CSV)")

        return records

    def watchdog_save_state(self, watchdog_object):
        """Save watchdog state from notebook implementation"""
        record_dict = {
            "timeout": watchdog_object._timeout,
            "interval": watchdog_object._interval,
            "failed": watchdog_object._failed,
            "stopped": watchdog_object._watchdog_stopped,
        }
        return record_dict

    def _existing_vehicle_ids(self, world):
        """Get existing vehicle IDs from notebook implementation"""
        return {v.id for v in world.get_actors().filter("vehicle.*")}

    def stop_manager(self, manager):
        """Stop manager from notebook implementation"""
        manager._running = False

        t = getattr(manager, "_scenario_thread", None)
        if t and hasattr(t, "is_alive") and t.is_alive() and threading.current_thread() is not t:
            t.join(timeout=2.0)
        manager._scenario_thread = None

        tt = getattr(manager, "_tick_thread", None)
        if tt and hasattr(tt, "is_alive") and tt.is_alive() and threading.current_thread() is not tt:
            tt.join(timeout=2.0)
        manager._tick_thread = None

    def reinit_inroute_and_blocked(self, tree):
        """Re-initialise only InRouteTest and ActorBlockedTest from notebook implementation"""
        try:
            import py_trees.common
            target = {"InRouteTest", "ActorBlockedTest"}
            for n in tree.iterate():
                if n.__class__.__name__ in target:
                    try:
                        n.terminate(py_trees.common.Status.INVALID)
                        n.initialise()
                    except Exception:
                        pass
        except ImportError:
            logger.warning("py_trees not available, skipping tree reinitialization")

    def start_manager_builder_only(self, manager):
        """Start ONLY the builder loop from notebook implementation"""
        manager._running = True
        t = getattr(manager, "_scenario_thread", None)
        if t is None or not (hasattr(t, "is_alive") and t.is_alive()):
            manager._scenario_thread = threading.Thread(
                target=manager.build_scenarios_loop,
                args=(manager._debug_mode > 0,),
                daemon=True
            )
            manager._scenario_thread.start()

    def pause_restore_resume(self, client, world, manager, vehicles_state, mode="snapshot_strict",
                            apply_controls=True, keep_sync=True):
        """Pause, restore, and resume simulation from notebook implementation"""
        # 1) Pause any manager activity
        self.stop_manager(manager)

        # 2) Ensure sync (paused) and remember previous settings
        prev = world.get_settings()
        changed = False
        if not prev.synchronous_mode:
            new = carla.WorldSettings()
            new.no_rendering_mode = prev.no_rendering_mode
            new.synchronous_mode = True
            new.fixed_delta_seconds = prev.fixed_delta_seconds or 0.05
            world.apply_settings(new)
            changed = True

        # 3) Build atomic batch
        present = self._existing_vehicle_ids(world)

        # --- MAIN RESTORE BATCH ---
        batch = []

        for rec in vehicles_state:
            vid = int(rec["id"])
            if vid not in present:
                continue

            tf = carla.Transform(
                carla.Location(float(rec["x"]), float(rec["y"]), float(rec["z"])),
                carla.Rotation(
                    yaw=float(rec.get("yaw", 0.0)),
                    pitch=float(rec.get("pitch", 0.0)),
                    roll=float(rec.get("roll", 0.0)),
                )
            )

            # Disable physics, set transform, re-enable physics
            batch += [
                carla.command.SetSimulatePhysics(vid, False),
                carla.command.ApplyTransform(vid, tf),
                carla.command.SetSimulatePhysics(vid, True),
            ]

        # Apply transforms first
        client.apply_batch_sync(batch, True)
        world.tick()

        # Apply velocities using set_target_velocity directly (more reliable)
        if mode == "snapshot_strict":
            for rec in vehicles_state:
                vid = int(rec["id"])
                if vid not in present:
                    continue

                actor = world.get_actor(vid)
                if actor:
                    # Get saved velocities
                    vx = abs(rec.get("vx", 0.0))
                    vy = abs(rec.get("vy", 0.0))
                    vz = abs(rec.get("vz", 0.0))
                    wx = rec.get("wx", 0.0)
                    wy = rec.get("wy", 0.0)
                    wz = rec.get("wz", 0.0)

                    # Debug print
                    if vid == 3695:  # ego vehicle ID from notebook
                        v_cur = actor.get_velocity()
                        print(f"Ego before impulse: cur=({v_cur.x:.2f}, {v_cur.y:.2f}, {v_cur.z:.2f}), target=({vx:.2f}, {vy:.2f}, {vz:.2f})")

                    # Force the exact velocity using enable_constant_velocity for one frame
                    actor.enable_constant_velocity(carla.Vector3D(vx, vy, vz))
                    actor.set_target_angular_velocity(carla.Vector3D(wx, wy, wz))

        # Let constant velocity apply for one tick
        world.tick()

        # Disable constant velocity and apply controls
        for rec in vehicles_state:
            vid = int(rec["id"])
            if vid not in present:
                continue

            actor = world.get_actor(vid)
            if actor:
                # Disable constant velocity
                actor.disable_constant_velocity()

                # Apply controls
                if apply_controls:
                    ctrl = carla.VehicleControl(
                        throttle=float(rec.get("throttle", 0.0)),
                        steer=float(rec.get("steer", 0.0)),
                        brake=float(rec.get("brake", 0.0)),
                        hand_brake=bool(rec.get("hand_brake", False)),
                        reverse=bool(rec.get("reverse", False)),
                        gear=int(rec.get("gear", 0)),
                    )
                    actor.apply_control(ctrl)

        # Final tick
        world.tick()

        # Finally, reinit those two criteria only (prevents immediate FAILURE)
        self.reinit_inroute_and_blocked(manager.scenario_tree)

        if changed and not keep_sync:
            world.apply_settings(prev)

        self.start_manager_builder_only(manager)

    def save_snapshot_to_disk(self, snapshot_id, vehicles_state, watchdog_state, observation):
        """Save snapshot to disk following notebook approach"""
        snapshot_file = self.snapshot_dir / f"{snapshot_id}.json"
        try:
            snapshot_data = {
                "snapshot_id": snapshot_id,
                "timestamp": datetime.now().isoformat(),
                "vehicles_state": vehicles_state,
                "watchdog_state": watchdog_state,
                "observation": observation
            }
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot_data, f, indent=2)
            logger.info(f"Snapshot saved to disk: {snapshot_file}")
            return snapshot_file
        except Exception as e:
            logger.error(f"Failed to save snapshot to disk: {e}")
            return None

    def restore_snapshot_from_disk(self, snapshot_id):
        """Restore snapshot from disk following notebook approach"""
        snapshot_file = self.snapshot_dir / f"{snapshot_id}.json"
        try:
            with open(snapshot_file, 'r') as f:
                snapshot_data = json.load(f)
            logger.info(f"Snapshot loaded from disk: {snapshot_file}")
            return snapshot_data
        except Exception as e:
            logger.error(f"Failed to load snapshot from disk: {e}")
            return None

    def extract_position(self, observation: Dict) -> Optional[Dict[str, float]]:
        """Extract vehicle position from observation."""
        try:
            if 'vehicle_state' in observation:
                state = observation['vehicle_state']
                if 'position' in state:
                    pos = state['position']
                    if isinstance(pos, dict):
                        return {'x': pos.get('x', 0), 'y': pos.get('y', 0), 'z': pos.get('z', 0)}
                    elif isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 3:
                        return {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])}
            return None
        except Exception as e:
            logger.warning(f"Could not extract position: {e}")
            return None
    
    def extract_and_save_image(self, observation: Dict, step: int, phase: str, 
                               branch_id: Optional[int] = None,
                               action: Optional[np.ndarray] = None) -> Optional[str]:
        """Extract image from observation and save to disk."""
        try:
            # Try to get image data
            image_data = None
            
            # Check for center_image directly
            if 'center_image' in observation:
                image_data = observation['center_image']
            # Check in images dict
            elif 'images' in observation and isinstance(observation['images'], dict):
                for key in ['center', 'Center', 'rgb']:
                    if key in observation['images']:
                        image_data = observation['images'][key]
                        break
            
            if image_data is None:
                logger.warning(f"No image found in observation")
                return None
            
            # Handle different image formats
            if isinstance(image_data, str):
                # Base64 encoded
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
            elif isinstance(image_data, np.ndarray):
                # Direct numpy array
                if image_data.dtype != np.uint8:
                    image_data = (image_data * 255).astype(np.uint8)
                image = Image.fromarray(image_data)
            elif isinstance(image_data, dict) and 'data' in image_data:
                # Wrapped image data
                image_bytes = base64.b64decode(image_data['data'])
                image = Image.open(BytesIO(image_bytes))
            else:
                logger.warning(f"Unknown image data type: {type(image_data)}")
                return None
            
            # Add overlay with information
            draw = ImageDraw.Draw(image)
            
            # Try to use a larger font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Extract position
            position = self.extract_position(observation)
            
            # Create text overlay
            text_lines = [f"Step {step:03d} - {phase}"]
            
            if branch_id is not None:
                agent_types = ["Straight", "Left Turn", "Right Turn", "Random Steer"]
                text_lines.append(f"Agent {branch_id}: {agent_types[branch_id]}")
            
            if position:
                text_lines.append(f"Position: X={position['x']:.1f} Y={position['y']:.1f} Z={position['z']:.1f}")
            
            if action is not None:
                text_lines.append(f"Action: T={action[0]:.2f} S={action[2]:.2f} B={action[1]:.2f}")
            
            # Draw text with background
            y_offset = 10
            for line in text_lines:
                bbox = draw.textbbox((10, y_offset), line, font=font)
                # Semi-transparent black background
                draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill=(0, 0, 0, 180))
                # Yellow text
                draw.text((10, y_offset), line, fill='yellow', font=font)
                y_offset += (bbox[3] - bbox[1]) + 5
            
            # Determine save path
            if branch_id is not None:
                # Branching phase - save to branch-specific folder
                branch_dirs = ["branch0_straight", "branch1_left", "branch2_right", "branch3_random"]
                save_dir = self.phase_dirs[branch_dirs[branch_id]]
                filename = f"step_{step:03d}_t{action[0]:.2f}_s{action[2]:.2f}.png"
            elif "Phase 3" in phase:
                # Phase 3 continuation
                save_dir = self.phase_dirs["phase3_continue"]
                action_str = f"_t{action[0]:.2f}_s{action[2]:.2f}" if action is not None else ""
                filename = f"step_{step:03d}{action_str}.png"
            else:
                # Phase 1 (single phase)
                save_dir = self.phase_dirs["phase1_single"]
                action_str = f"_t{action[0]:.2f}_s{action[2]:.2f}" if action is not None else ""
                filename = f"step_{step:03d}{action_str}.png"
            
            filepath = save_dir / filename
            image.save(filepath)
            logger.info(f"Saved image: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_phase1_single_exploration(self):
        """Phase 1: Single mode exploration for 90 steps."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: Single Mode Exploration (90 steps)")
        logger.info("="*60)

        # Check if environment is already initialized (from pre-initialization)
        if hasattr(self.env.envs[0], 'last_observation') and self.env.envs[0].last_observation is not None:
            logger.info("Using pre-initialized environment state")
            obs = self.env.envs[0].last_observation
            info = {}
            logger.info(f"Using pre-initialized state. Mode: {self.env.current_mode}, is_branching: {self.env.is_branching}")
        else:
            logger.info("Resetting environment for Phase 1...")
            obs, info = self.env.reset(options={"route_id": 0})
            logger.info(f"Reset complete. Mode: {self.env.current_mode}, is_branching: {self.env.is_branching}")

        # Save initial image
        self.extract_and_save_image(obs, 0, "Phase 1 - Initial")
        initial_pos = self.extract_position(obs)
        if initial_pos:
            logger.info(f"Initial position: X={initial_pos['x']:.1f}, Y={initial_pos['y']:.1f}")

        # Run 90 steps with forward action
        for step in range(1, 91):
            # Simple forward action
            action = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # [throttle, brake, steer]
            
            # Single step (not branch_step since we're in single mode)
            try:
                obs, reward, terminated, truncated, info = self.env.single_step(action)
                
                # Check status
                if 'status' in info:
                    status = info['status']
                    if not status.ready and status.status != EnvStatus.TERMINATED:
                        logger.warning(f"Step {step}: Not ready - {status.message}")
                
                # Extract and save position
                position = self.extract_position(obs)
                if position:
                    self.position_history["phase1"].append(position)
                
                # Save image every 10 steps
                if step % 10 == 0:
                    self.extract_and_save_image(obs, step, "Phase 1 - Single", action=action)
                    if position:
                        logger.info(f"Step {step}: X={position['x']:.1f}, Y={position['y']:.1f}, "
                                  f"reward={reward:.2f}")
                
                if terminated or truncated:
                    logger.info(f"Episode ended at step {step}")
                    break
                    
            except Exception as e:
                logger.error(f"Error at step {step}: {e}")
                break
        
        # Save final image
        self.extract_and_save_image(obs, 90, "Phase 1 - Final", action=action)
        logger.info(f"Phase 1 complete: {len(self.position_history['phase1'])} positions recorded")
        
        return obs  # Return last observation for continuity
    
    def run_phase2_branching(self, last_obs: Dict):
        """Phase 2: Branching mode with 4 agents for 20 steps."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Branching Mode (4 agents, 20 steps)")
        logger.info("="*60)
        
        # Save snapshot before branching using enhanced functionality
        logger.info("Saving snapshot for branching...")
        snapshot_id = self.env.save_snapshot()

        # Get the underlying CARLA world and manager for enhanced snapshot
        if hasattr(self.env.envs[0], 'client') and hasattr(self.env.envs[0], 'world'):
            try:
                # Record vehicle states using notebook approach
                self.vehicles_state = self.record_vehicles(self.env.envs[0].world)
                self.watchdog_state = self.watchdog_save_state(self.env.envs[0].manager._watchdog)

                # Save snapshot to disk with detailed data
                snapshot_file = self.save_snapshot_to_disk(
                    snapshot_id,
                    self.vehicles_state,
                    self.watchdog_state,
                    last_obs
                )
                logger.info(f"Enhanced snapshot saved: {snapshot_file}")
            except Exception as e:
                logger.warning(f"Could not enhance snapshot: {e}")

        logger.info(f"Snapshot saved: {snapshot_id}")
        
        # Enable branching with 2 branches
        logger.info("Enabling branching mode with 2 agents...")
        status = self.env.enable_branching(snapshot_id, num_branches=2, async_setup=False)

        if status.status != EnvStatus.BRANCHING_READY:
            logger.error(f"Failed to enable branching: {status.message}")
            return

        logger.info(f"Branching enabled. Mode: {self.env.current_mode}, is_branching: {self.env.is_branching}")
        logger.info("Agent behaviors:")
        logger.info("  Agent 0: Straight (throttle=1.0, steer=0.0)")
        logger.info("  Agent 1: Random steering (throttle=1.0, steer=random[-1,1])")
        
        # Run 50 steps with different behaviors
        for step in range(91, 141):  # Steps 91-140
            # Generate actions for each agent
            actions = []

            # Agent 0: Straight
            actions.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))

            # Agent 1: Random steering
            random_steer = random.uniform(-1.0, 1.0)
            self.random_steers.append(random_steer)
            actions.append(np.array([1.0, 0.0, random_steer], dtype=np.float32))
            
            # Branch step (not single_step since we're in branching mode)
            try:
                observations, rewards, terminateds, truncateds, infos = self.env.branch_step(actions)
                
                # Process each branch
                for i in range(2):
                    # Check status
                    if 'status' in infos[i]:
                        status = infos[i]['status']
                        if not status.ready and status.status != EnvStatus.TERMINATED:
                            logger.warning(f"Step {step}, Agent {i}: Not ready - {status.message}")
                    
                    # Extract position
                    position = self.extract_position(observations[i])
                    if position:
                        self.position_history[f"branch{i}"].append(position)
                    
                    # Save images every 3 frames for more detailed analysis
                    if (step - 91) % 3 == 0 or step == 91 or step == 110:
                        self.extract_and_save_image(
                            observations[i],
                            step,
                            f"Phase 2 - Branch {i}",
                            branch_id=i,
                            action=actions[i]
                        )
                    
                    if terminateds[i] or truncateds[i]:
                        logger.info(f"Agent {i} terminated at step {step}")
                
                # Log progress every 3 frames for more detailed analysis
                if (step - 91) % 3 == 0:
                    logger.info(f"Step {step}:")
                    for i in range(2):
                        pos = self.extract_position(observations[i])
                        if pos:
                            agent_types = ["Straight", f"Random({self.random_steers[-1]:.2f})"]
                            logger.info(f"  Agent {i} ({agent_types[i]}): "
                                      f"X={pos['x']:.1f}, Y={pos['y']:.1f}, reward={rewards[i]:.2f}")
                
            except Exception as e:
                logger.error(f"Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        logger.info(f"Phase 2 complete")
        
        # Return observations for Phase 3
        self.last_branch_observations = observations
        return observations
    
    def run_phase3_continue_agent1(self, branch_observations):
        """Phase 3: Select Agent 1 (random) and continue for 50 steps."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: Continue with Agent 1 (Random Steering, 50 steps)")
        logger.info("="*60)
        
        # Select Agent 1 (index 1) - the random steering agent
        logger.info("Selecting Agent 1 (random steering) to continue...")
        self.env.select_branch(1)
        
        logger.info(f"Returned to single mode. Mode: {self.env.current_mode}, is_branching: {self.env.is_branching}")
        
        # Get the last position from Agent 3
        if branch_observations and len(branch_observations) > 3:
            last_pos = self.extract_position(branch_observations[3])
            if last_pos:
                logger.info(f"Continuing from Agent 3 position: X={last_pos['x']:.1f}, Y={last_pos['y']:.1f}")
        
        # Continue for 50 steps with random steering
        logger.info("Continuing with random steering strategy for 50 steps...")
        
        for step in range(141, 191):  # Steps 141-190
            # Generate random steering like Agent 1 did
            random_steer = random.uniform(-1.0, 1.0)
            self.random_steers.append(random_steer)
            action = np.array([1.0, 0.0, random_steer], dtype=np.float32)  # [throttle, brake, steer]
            
            try:
                # Use single_step since we're back in single mode
                obs, reward, terminated, truncated, info = self.env.single_step(action)
                
                # Check status
                if 'status' in info:
                    status = info['status']
                    if not status.ready and status.status != EnvStatus.TERMINATED:
                        logger.warning(f"Step {step}: Not ready - {status.message}")
                
                # Extract and save position
                position = self.extract_position(obs)
                if position:
                    self.position_history["phase3"].append(position)
                
                # Save image every 10 steps
                if (step - 111) % 10 == 0 or step == 111 or step == 160:
                    self.extract_and_save_image(
                        obs, 
                        step, 
                        f"Phase 3 - Continue Agent 3",
                        action=action
                    )
                    
                    if position:
                        logger.info(f"Step {step}: X={position['x']:.1f}, Y={position['y']:.1f}, "
                                  f"steer={random_steer:.2f}, reward={reward:.2f}")
                
                if terminated or truncated:
                    logger.info(f"Episode ended at step {step}")
                    break
                    
            except Exception as e:
                logger.error(f"Error at step {step}: {e}")
                break
        
        # Save final image
        self.extract_and_save_image(obs, 160, "Phase 3 - Final", action=action)
        logger.info(f"Phase 3 complete: {len(self.position_history['phase3'])} positions recorded")
        
        # Calculate total distance traveled in Phase 3
        if len(self.position_history['phase3']) > 0:
            first = self.position_history['phase3'][0]
            last = self.position_history['phase3'][-1]
            distance = np.sqrt((last['x'] - first['x'])**2 + (last['y'] - first['y'])**2)
            logger.info(f"Phase 3 distance traveled: {distance:.1f} units with random steering")
    
    def save_position_data(self):
        """Save all position data to JSON file."""
        position_file = self.output_dir / "position_history.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, positions in self.position_history.items():
            json_data[key] = positions
        
        # Add random steering history
        json_data["random_steers"] = self.random_steers
        
        with open(position_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Position data saved to: {position_file}")
        
        # Print summary
        logger.info("\nPosition Summary:")
        for key, positions in self.position_history.items():
            if positions:
                logger.info(f"  {key}: {len(positions)} positions recorded")
                if len(positions) > 0:
                    first = positions[0]
                    last = positions[-1]
                    distance = np.sqrt((last['x'] - first['x'])**2 + (last['y'] - first['y'])**2)
                    logger.info(f"    Distance traveled: {distance:.1f} units")
    
    def run_test(self):
        """Run complete GRPO visual test."""
        logger.info("="*60)
        logger.info("GRPO VISUAL TEST")
        logger.info("="*60)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Environment mode: {self.env.current_mode}")
        logger.info(f"Max branches: {self.env.max_branches}")
        
        try:
            # Pre-initialize all services for fast branching
            logger.info("Pre-initializing all services for fast branching...")
            try:
                init_status = self.env.initialize_all_services(route_id=0)
                if init_status.ready:
                    logger.info("✓ All services pre-initialized successfully")
                else:
                    logger.warning(f"⚠ Service pre-initialization issues: {init_status.message}")
                    logger.info("Continuing with test despite pre-initialization issues...")
            except Exception as e:
                logger.warning(f"Service pre-initialization failed: {e}")
                logger.info("Continuing with test without pre-initialization...")

            # Phase 1: Single exploration (90 steps)
            last_obs = self.run_phase1_single_exploration()

            # Phase 2: Branching with 2 agents (50 steps)
            branch_observations = self.run_phase2_branching(last_obs)

            # Phase 3: Continue with Agent 1 (50 steps)
            self.run_phase3_continue_agent1(branch_observations)
            
            # Save all position data
            self.save_position_data()
            
            logger.info("\n" + "="*60)
            logger.info("TEST COMPLETE - ALL 3 PHASES")
            logger.info("="*60)
            logger.info(f"Images saved to: {self.output_dir}")
            logger.info(f"Total steps: 160 (Phase1: 90, Phase2: 20, Phase3: 50)")
            logger.info(f"Total random steers generated: {len(self.random_steers)}")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info("Closing environment...")
            self.env.close()


def start_servers(num_services: int = 2):
    """Start CARLA servers if needed."""
    logger.info(f"Starting {num_services} CARLA services...")
    
    # Kill existing processes comprehensively
    logger.info("Cleaning up existing processes...")
    processes_to_kill = [
        "carla_server.py",
        "microservice_manager.py",
        "CarlaUE4",
        "server_manager.py"
    ]
    
    for process in processes_to_kill:
        result = subprocess.run(["pkill", "-f", process], capture_output=True)
        if result.returncode == 0:
            logger.info(f"  Killed {process} processes")
    
    # Comprehensive port cleanup - clean all potential ports first
    logger.info("Cleaning up ports comprehensively...")

    # Clean API ports (8080-8083)
    for port in range(8080, 8084):
        try:
            result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        subprocess.run(["kill", "-9", pid], capture_output=True)
                        logger.info(f"  Killed process {pid} on port {port}")
        except Exception as e:
            logger.debug(f"  Error cleaning port {port}: {e}")

    # Clean CARLA ports (2000-2012)
    for port in range(2000, 2013):
        try:
            result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        subprocess.run(["kill", "-9", pid], capture_output=True)
                        logger.info(f"  Killed process {pid} on port {port}")
        except Exception as e:
            logger.debug(f"  Error cleaning port {port}: {e}")

    # Clean traffic manager ports (3000-3012)
    for port in range(3000, 3013):
        try:
            result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        subprocess.run(["kill", "-9", pid], capture_output=True)
                        logger.info(f"  Killed process {pid} on port {port}")
        except Exception as e:
            logger.debug(f"  Error cleaning port {port}: {e}")

    # Wait a moment for ports to be fully released
    time.sleep(2)
    
    time.sleep(3)  # Wait for processes to fully terminate
    
    # Start microservice manager
    cmd = [
        "python",
        str(Path(__file__).parent.parent / "server" / "microservice_manager.py"),
        "--num-services", str(num_services),
        "--startup-delay", "0"
    ]

    proc = subprocess.Popen(cmd)
    logger.info("Microservice manager started, waiting for servers...")

    # Wait for servers to be ready with health checks
    max_wait = 60  # Increased wait time
    wait_interval = 5
    total_wait = 0

    while total_wait < max_wait:
        all_ready = True
        for i in range(num_services):
            try:
                response = requests.get(f"http://localhost:{8080 + i}/health", timeout=5)
                if response.status_code != 200:
                    all_ready = False
                    break
            except:
                all_ready = False
                break

        if all_ready:
            logger.info(f"All {num_services} servers are ready after {total_wait}s")
            break

        logger.info(f"Waiting for servers... ({total_wait}/{max_wait}s)")
        time.sleep(wait_interval)
        total_wait += wait_interval

    if not all_ready:
        logger.warning(f"Not all servers are ready after {max_wait}s, but continuing...")

    return proc


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Visual test for GRPO environment')
    parser.add_argument('--base-port', type=int, default=8080,
                       help='Base API port (default: 8080)')
    parser.add_argument('--output-dir', type=str, default='grpo_test_images',
                       help='Output directory for images (default: grpo_test_images)')
    parser.add_argument('--start-servers', action='store_true',
                       help='Start CARLA servers automatically')
    
    args = parser.parse_args()
    
    server_proc = None
    if args.start_servers:
        server_proc = start_servers(2)
    
    try:
        # Run test
        tester = GRPOVisualTester(
            base_port=args.base_port,
            output_dir=args.output_dir
        )
        tester.run_test()
        
    finally:
        if server_proc:
            logger.info("Stopping servers...")
            server_proc.terminate()
            server_proc.wait()


if __name__ == "__main__":
    main()