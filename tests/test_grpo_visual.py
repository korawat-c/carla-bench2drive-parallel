#!/usr/bin/env python3
"""
Visual test for GRPO environment with image capture.

Test workflow:
1. Phase 1: Single mode exploration for 90 steps
2. Save snapshot and enable branching (2 branches)
3. Phase 2: Branching mode for 50 steps with 2 different agent behaviors:
   - Agent 0: Straight (throttle=1.0, steer=0.0)
   - Agent 1: Left turn (throttle=0.8, steer=-0.3)
4. Phase 3: Select Agent 1 and continue for 50 steps
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
        
        # Create GRPO environment with 2 branches max
        self.env = GRPOCarlaEnv(
            num_services=2,
            base_api_port=base_port,
            render_mode="rgb_array",
            max_steps=200,
            timeout=180.0
        )
        
        # Position tracking
        self.position_history = {
            "phase1": [],
            "branch0": [],
            "branch1": [],
            "branch2": [],
            "branch3": [],
            "phase3": []
        }
        
        # Random steering history for agent 4
        self.random_steers = []
        
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
        
        # Reset environment
        logger.info("Resetting environment...")
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
            action = np.array([1.0, 0.0, 0.0], dtype=np.float32) # [throttle, brake, steer]
            
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
                
                # Save image EVERY step as requested
                self.extract_and_save_image(obs, step, "Phase 1 - Single", action=action)
                if step % 10 == 0 and position:
                    logger.info(f"Step {step}: X={position['x']:.1f}, Y={position['y']:.1f}, "
                              f"reward={reward:.2f}")
                
                if terminated or truncated:
                    logger.info(f"Episode ended at step {step}")
                    break
                    
            except Exception as e:
                logger.error(f"Error at step {step}: {e}")
                # Still save image even on error
                self.extract_and_save_image(obs, step, f"Phase 1 - Error at step {step}", action=action)
                break
        logger.info(f"Phase 1 complete: {len(self.position_history['phase1'])} positions recorded")
        
        return obs  # Return last observation for continuity
    
    def run_phase2_branching(self, last_obs: Dict):
        """Phase 2: Branching mode with 2 agents for 50 steps."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Branching Mode (2 agents, 50 steps)")
        logger.info("="*60)
        
        # Save snapshot before branching
        logger.info("Saving snapshot for branching...")
        snapshot_id = self.env.save_snapshot()
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
        logger.info("  Agent 1: Left turn (throttle=0.8, steer=-0.3)")
        
        # Run 50 steps with different behaviors
        for step in range(91, 141):  # Steps 91-140
            # Generate actions for each agent
            actions = []
            
            # Agent 0: Straight (throttle, brake, steer)
            actions.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
            
            # Agent 1: Left turn (throttle, brake, steer)
            actions.append(np.array([1.0, 0.0, -1.0], dtype=np.float32))
            
            # Branch step (not single_step since we're in branching mode)
            try:
                observations, rewards, terminateds, truncateds, infos = self.env.branch_step(actions)
                
                # Process each branch
                for i in range(2):  # Only 2 agents
                    # Check status
                    if 'status' in infos[i]:
                        status = infos[i]['status']
                        if not status.ready and status.status != EnvStatus.TERMINATED:
                            logger.warning(f"Step {step}, Agent {i}: Not ready - {status.message}")
                    
                    # Extract position
                    position = self.extract_position(observations[i])
                    if position:
                        self.position_history[f"branch{i}"].append(position)
                    
                    # Save images EVERY step as requested
                    self.extract_and_save_image(
                        observations[i], 
                        step, 
                        f"Phase 2 - Branch {i}",
                        branch_id=i,
                        action=actions[i]
                    )
                    
                    if terminateds[i] or truncateds[i]:
                        logger.info(f"Agent {i} terminated at step {step}")
                
                # Log progress every step
                logger.info(f"Step {step}:")
                for i in range(2):  # Only 2 agents
                    pos = self.extract_position(observations[i])
                    if pos:
                        agent_types = ["Straight", "Gentle Left"]
                        logger.info(f"  Agent {i} ({agent_types[i]}): "
                                  f"X={pos['x']:.1f}, Y={pos['y']:.1f}, reward={rewards[i]:.2f}")
                    # Log that image was saved  
                    logger.info(f"  Saved image for Agent {i} at step {step}")
                
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
        """Phase 3: Select Agent 1 (left turn) and continue for 50 steps."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: Continue with Agent 1 (Left Turn, 50 steps)")
        logger.info("="*60)
        
        # Select Agent 1 (index 1) - the left turn agent
        logger.info("Selecting Agent 1 (left turn) to continue...")
        self.env.select_branch(1)
        
        logger.info(f"Returned to single mode. Mode: {self.env.current_mode}, is_branching: {self.env.is_branching}")
        
        # Get the last position from Agent 1
        if branch_observations and len(branch_observations) > 1:
            last_pos = self.extract_position(branch_observations[1])
            if last_pos:
                logger.info(f"Continuing from Agent 1 position: X={last_pos['x']:.1f}, Y={last_pos['y']:.1f}")
        
        # Continue for 50 steps with forward movement
        logger.info("Continuing with forward movement for 50 steps...")
        
        for step in range(141, 191):  # Steps 141-190
            # Simple forward action to maintain speed
            action = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # [throttle, brake, steer]
            
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
                
                # Save image EVERY step as requested
                self.extract_and_save_image(
                    obs, 
                    step, 
                    f"Phase 3 - Continue Agent 1",
                    action=action
                )
                
                # Log progress every 10 steps
                if step % 10 == 0 or step == 141:
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
            logger.info(f"Total steps: 190 (Phase1: 90, Phase2: 50, Phase3: 50)")
            
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
    
    # Also kill processes on specific ports
    for port in range(8080, 8080 + num_services):
        subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True)
        subprocess.run(f"lsof -ti:{port} | xargs kill -9 2>/dev/null", shell=True, capture_output=True)
    
    for port in range(2000, 2000 + num_services * 2, 2):
        subprocess.run(f"lsof -ti:{port} | xargs kill -9 2>/dev/null", shell=True, capture_output=True)
    
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
    time.sleep(15)  # Wait for servers to start
    
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