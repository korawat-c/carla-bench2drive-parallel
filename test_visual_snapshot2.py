#!/usr/bin/env python3
"""
Visual test for snapshot/restore functionality.

This test:
1. Connects to a single CARLA instance
2. Drives straight for 30 steps (saves images)
3. Creates a snapshot
4. Continues straight for 20 steps (saves images)
5. Restores the snapshot
6. Drives turning right for 20 steps (saves images)
7. Compares the trajectories visually
"""

import os
import time
import json
import base64
import logging
import argparse
import requests
import numpy as np
import subprocess
from pathlib import Path
from PIL import Image
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def cleanup_ports(ports: List[int] = [8080, 2000, 2002, 2004]):
    """Kill processes using specified ports to ensure clean startup"""
    logger.info(f"Cleaning up ports: {ports}")
    
    for port in ports:
        try:
            # Check if port is in use
            result = subprocess.run(
                f"lsof -i :{port} | grep LISTEN",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                # Extract PIDs and kill them
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split()
                    if len(parts) > 1:
                        pid = parts[1]
                        logger.info(f"Killing process {pid} using port {port}")
                        try:
                            subprocess.run(f"kill -9 {pid}", shell=True, check=True)
                        except:
                            pass
        except Exception as e:
            logger.debug(f"Error checking port {port}: {e}")
    
    # Also kill any lingering microservice managers and carla servers
    try:
        subprocess.run("pkill -f microservice_manager", shell=True)
        subprocess.run("pkill -f carla_server.py", shell=True)
    except:
        pass
    
    logger.info("Port cleanup complete")
    time.sleep(1)  # Brief pause to ensure ports are released


class VisualCarlaClient:
    """Client for CARLA server with image capture capabilities"""
    
    def __init__(self, base_url: str, output_dir: str = "snapshot_test_images"):
        self.base_url = base_url
        self.session = requests.Session()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different phases
        self.phase_dirs = {
            "initial": self.output_dir / "1_initial_50steps",
            "continued": self.output_dir / "2_continued_50steps",
            "restored": self.output_dir / "3_restored_straight_50steps_should_match_phase2"
        }
        
        for phase_dir in self.phase_dirs.values():
            phase_dir.mkdir(parents=True, exist_ok=True)
        
        self.step_counter = 0
        self.current_phase = "initial"
        
        # Position tracking
        self.position_history = {
            "initial": [],
            "continued": [],
            "restored": []
        }
        
    def health_check(self) -> bool:
        """Check if server is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def reset(self, route_id: int = 0) -> Dict:
        """Reset environment"""
        logger.info(f"Resetting environment with route_id={route_id}")
        logger.info("Note: First reset may take 60-120 seconds to load CARLA world and scenarios...")
        response = self.session.post(
            f"{self.base_url}/reset",
            json={"route_id": route_id},
            timeout=180  # Increased timeout for CARLA world loading
        )
        response.raise_for_status()
        logger.info("Reset successful!")
        return response.json()
    
    def step(self, action: Dict[str, float]) -> Dict:
        """Execute step and return result"""
        response = self.session.post(
            f"{self.base_url}/step",
            json={"action": action},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def save_snapshot(self, snapshot_id: str = None, use_notebook: bool = True) -> str:
        """Save snapshot"""
        endpoint = "/snapshot_notebook" if use_notebook else "/snapshot"
        logger.info(f"Saving snapshot: {snapshot_id} (using {'notebook' if use_notebook else 'original'} endpoint)")
        response = self.session.post(
            f"{self.base_url}{endpoint}",
            json={"snapshot_id": snapshot_id} if snapshot_id else {},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"Snapshot saved successfully: {result}")
        return snapshot_id or result.get("snapshot_id", "unknown")
    
    def restore_snapshot(self, snapshot_id: str, use_notebook: bool = True) -> Dict:
        """Restore snapshot"""
        if use_notebook:
            endpoint = "/restore_notebook"
            version = "notebook"
        else:
            endpoint = "/restore5"
            version = "v5"
        logger.info(f"Restoring snapshot: {snapshot_id} (using {version})")
        response = self.session.post(
            f"{self.base_url}{endpoint}",
            json={"snapshot_id": snapshot_id},
            timeout=30
        )
        if response.status_code != 200:
            try:
                error_detail = response.json().get('detail', 'Unknown error')
                logger.error(f"Restore failed with status {response.status_code}: {error_detail}")
            except:
                logger.error(f"Restore failed with status {response.status_code}: {response.text}")
        response.raise_for_status()
        result = response.json()
        logger.info(f"Snapshot restored successfully using {version}: {result}")
        return result
    
    def extract_position(self, observation: Dict) -> Optional[Tuple[float, float, float]]:
        """Extract vehicle position from observation"""
        try:
            # Try different possible locations for position data
            if 'vehicle_state' in observation:
                state = observation['vehicle_state']
                if 'position' in state:
                    pos = state['position']
                    if isinstance(pos, dict):
                        return (pos.get('x', 0), pos.get('y', 0), pos.get('z', 0))
                    elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
                        return (pos[0], pos[1], pos[2])
                    
            # Check for position in other locations
            if 'position' in observation:
                pos = observation['position']
                if isinstance(pos, dict):
                    return (pos.get('x', 0), pos.get('y', 0), pos.get('z', 0))
                elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    return (pos[0], pos[1], pos[2])
            
            # Check for x,y,z directly in observation
            if 'x' in observation and 'y' in observation:
                return (observation['x'], observation['y'], observation.get('z', 0))
                
            return None
        except Exception as e:
            logger.warning(f"Could not extract position: {e}")
            return None
    
    def extract_and_save_image(self, observation: Dict, step_num: int, phase: str, 
                               action: Dict[str, float] = None) -> Optional[str]:
        """Extract image from observation and save to disk with coordinate overlay"""
        try:
            # Try to get center camera image from images dict
            image_data = None
            image_key = None
            
            # Check if we have an images dictionary
            if 'images' in observation and isinstance(observation['images'], dict):
                # Try to get center camera from images dict
                for key in ['center', 'Center', 'left', 'Left']:
                    if key in observation['images']:
                        image_data = observation['images'][key]
                        image_key = key
                        break
            
            # If not found in images dict, check top level
            if image_data is None:
                for key in ['center_image', 'Center', 'center', 'rgb', 'camera']:
                    if key in observation:
                        image_data = observation[key]
                        image_key = key
                        break
            
            if image_data is None:
                logger.warning(f"No image found in observation. Top-level keys: {observation.keys()}")
                if 'images' in observation:
                    logger.warning(f"Images dict keys: {observation['images'].keys() if isinstance(observation['images'], dict) else 'not a dict'}")
                return None
            
            # Handle base64 encoded image
            if isinstance(image_data, str):
                # Decode base64 string
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
            elif isinstance(image_data, dict) and 'data' in image_data:
                # Handle wrapped image data
                image_bytes = base64.b64decode(image_data['data'])
                image = Image.open(BytesIO(image_bytes))
            elif isinstance(image_data, np.ndarray):
                # Direct numpy array
                image = Image.fromarray(image_data.astype('uint8'))
            else:
                logger.warning(f"Unknown image data type: {type(image_data)}")
                return None
            
            # Add coordinate overlay with larger font
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            # Try to use a larger font
            font_size = 20
            font = None
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            # Extract position for overlay
            position = self.extract_position(observation)
            
            # Track other vehicles if available
            other_vehicles_text = []
            if 'debug_info' in observation and 'other_vehicles' in observation['debug_info']:
                vehicles = observation['debug_info']['other_vehicles']
                other_vehicles_text.append(f"Other Vehicles: {len(vehicles)}")
                # Show details of first 3 vehicles with IDs, model, and color
                for i, v in enumerate(vehicles[:3]):
                    vehicle_id = v.get('id', f'unknown_{i}')
                    
                    # Extract vehicle model from type_id (e.g., "vehicle.audi.a2" -> "Audi A2")
                    type_id = v.get('type_id', 'unknown')
                    if 'vehicle.' in type_id:
                        parts = type_id.replace('vehicle.', '').split('.')
                        if len(parts) >= 2:
                            make = parts[0].title()
                            model = parts[1].upper() if len(parts[1]) <= 3 else parts[1].title()
                            vehicle_model = f"{make} {model}"
                        else:
                            vehicle_model = type_id.replace('vehicle.', '').title()
                    else:
                        vehicle_model = type_id
                    
                    # Extract vehicle color if available
                    color = v.get('color', 'unknown')
                    if isinstance(color, dict):
                        # Color might be RGB values
                        if 'r' in color and 'g' in color and 'b' in color:
                            # Try to identify common colors from RGB
                            r, g, b = color.get('r', 0), color.get('g', 0), color.get('b', 0)
                            if r > 200 and g < 100 and b < 100:
                                color_name = "Red"
                            elif r < 100 and g > 200 and b < 100:
                                color_name = "Green"
                            elif r < 100 and g < 100 and b > 200:
                                color_name = "Blue"
                            elif r > 200 and g > 200 and b < 100:
                                color_name = "Yellow"
                            elif r > 200 and g > 100 and b < 50:
                                color_name = "Orange"
                            elif r < 50 and g < 50 and b < 50:
                                color_name = "Black"
                            elif r > 200 and g > 200 and b > 200:
                                color_name = "White"
                            elif r < 150 and g > 200 and b > 200:
                                color_name = "Cyan"
                            elif r > 100 and g < 100 and b > 100:
                                color_name = "Purple"
                            else:
                                color_name = f"RGB({r},{g},{b})"
                        else:
                            color_name = str(color)
                    elif isinstance(color, str):
                        color_name = color.title()
                    else:
                        color_name = "N/A"
                    
                    # Format the vehicle info line
                    other_vehicles_text.append(f"  [{vehicle_id}] {vehicle_model} ({color_name})")
                    other_vehicles_text.append(f"       Pos: x={v['x']:.1f} y={v['y']:.1f}")
            
            if position:
                # Create text overlay with larger, more detailed info
                text_lines = [
                    f"Step {step_num:03d} - Phase: {phase.upper()}",
                    f"Ego Vehicle:",
                    f"  X={position[0]:.2f} Y={position[1]:.2f} Z={position[2]:.2f}"
                ]
                
                if action:
                    text_lines.append(f"Action: T={action['throttle']:.1f} S={action['steer']:.2f} B={action['brake']:.1f}")
                
                # Add other vehicles info
                text_lines.extend(other_vehicles_text)
                
                # Draw text with background for visibility
                y_offset = 10
                for line in text_lines:
                    if font:
                        bbox = draw.textbbox((10, y_offset), line, font=font)
                    else:
                        bbox = draw.textbbox((10, y_offset), line)
                    
                    # Draw semi-transparent background
                    draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill='black')
                    
                    # Draw text in yellow for better visibility
                    if font:
                        draw.text((10, y_offset), line, fill='yellow', font=font)
                    else:
                        draw.text((10, y_offset), line, fill='yellow')
                    
                    y_offset += (bbox[3] - bbox[1]) + 5
            
            # Create filename with action info
            action_str = ""
            if action:
                action_str = f"_t{action['throttle']:.2f}_s{action['steer']:.2f}_b{action['brake']:.2f}"
            
            filename = f"step_{step_num:03d}{action_str}.png"
            filepath = self.phase_dirs[phase] / filename
            
            # Save image
            image.save(filepath)
            logger.info(f"Saved image: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return None
    
    def save_position_data(self):
        """Save all position data to JSON file"""
        position_file = self.output_dir / "position_history.json"
        with open(position_file, 'w') as f:
            json.dump(self.position_history, f, indent=2)
        logger.info(f"Position data saved to: {position_file}")
        return position_file
    
    def close(self):
        """Close environment and session"""
        try:
            self.session.post(f"{self.base_url}/close")
        except:
            pass
        self.session.close()


def create_summary_image(output_dir: Path):
    """Create a summary image showing all three phases side by side"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Get sample images from each phase
        phase_images = []
        phase_names = ["1_initial_50steps", "2_continued_50steps", "3_restored_straight_50steps_should_match_phase2"]
        
        for phase_name in phase_names:
            phase_dir = output_dir / phase_name
            # Get middle image from phase
            images = sorted(list(phase_dir.glob("step_*.png")))
            if images:
                mid_idx = len(images) // 2
                img = Image.open(images[mid_idx])
                # Resize for summary
                img.thumbnail((400, 300))
                phase_images.append(img)
        
        if len(phase_images) == 3:
            # Create summary canvas
            width = sum(img.width for img in phase_images) + 40
            height = max(img.height for img in phase_images) + 100
            summary = Image.new('RGB', (width, height), 'white')
            
            # Paste images
            x_offset = 10
            for i, img in enumerate(phase_images):
                y_offset = 50
                summary.paste(img, (x_offset, y_offset))
                
                # Add label
                draw = ImageDraw.Draw(summary)
                label = ["Initial (straight)", "Continued (straight)", "Restored (right turn)"][i]
                draw.text((x_offset + 10, 10), label, fill='black')
                
                x_offset += img.width + 10
            
            # Save summary
            summary_path = output_dir / "summary_comparison.png"
            summary.save(summary_path)
            logger.info(f"Created summary image: {summary_path}")
            
    except Exception as e:
        logger.warning(f"Could not create summary image: {e}")


def run_visual_test(client: VisualCarlaClient):
    """Run the visual snapshot test"""
    
    logger.info("="*60)
    logger.info("VISUAL SNAPSHOT/RESTORE TEST")
    logger.info("="*60)
    
    try:
        # Reset environment
        logger.info("\n=== Phase 0: Reset Environment ===")
        reset_result = client.reset(route_id=0)
        logger.info(f"Environment reset complete")
        
        # Save initial observation and position
        if 'observation' in reset_result:
            client.extract_and_save_image(reset_result['observation'], 0, 'initial')
            
            # Extract and save initial position
            position = client.extract_position(reset_result['observation'])
            if position:
                logger.info(f"Initial position: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}")
                client.position_history['initial'].append({
                    'step': 0,
                    'x': position[0],
                    'y': position[1],
                    'z': position[2],
                    'action': {'throttle': 0, 'brake': 0, 'steer': 0}
                })
        
        # Phase 1: Drive straight for 50 steps
        logger.info("\n=== Phase 1: Initial - Drive Straight for 50 Steps ===")
        client.current_phase = "initial"
        
        for step in range(50):
            # Straight driving action with full throttle
            action = {
                "throttle": 1.0,  # Full throttle for visible movement
                "brake": 0.0,
                "steer": 0.0  # Straight
            }
            
            logger.info(f"Step {step+1}/50: Driving straight (throttle=1.0, steer=0.0)")
            step_result = client.step(action)
            
            # Save image and position
            if 'observation' in step_result:
                client.extract_and_save_image(
                    step_result['observation'], 
                    step + 1, 
                    'initial',
                    action
                )
                
                # Extract and save position
                position = client.extract_position(step_result['observation'])
                if position:
                    logger.info(f"  Position: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}")
                    client.position_history['initial'].append({
                        'step': step + 1,
                        'x': position[0],
                        'y': position[1],
                        'z': position[2],
                        'action': action
                    })
            
            # Log progress
            if (step + 1) % 10 == 0:
                logger.info(f"  Progress: {step+1}/50 steps complete")
            
            # Check if terminated
            if step_result.get('terminated', False):
                logger.warning(f"Episode terminated at step {step+1}")
                break
        
        # Create snapshot
        logger.info("\n=== Creating Snapshot ===")
        # Log current position BEFORE saving
        # step_result contains the last observation from step 50
        last_obs = step_result['observation'] if 'observation' in step_result else None
        if last_obs:
            last_pos = client.extract_position(last_obs)
            logger.info(f"BEFORE SAVE - Current ego position at step 50: x={last_pos[0]:.2f}, y={last_pos[1]:.2f}, z={last_pos[2]:.2f}")
        
        snapshot_id = client.save_snapshot("visual_test_snapshot", use_notebook=True)
        logger.info(f"Snapshot created: {snapshot_id}")
        logger.info("This is our branching point for GRPO")
        
        # Log the saved position for verification
        logger.info(f"Snapshot saved at position x={last_pos[0]:.2f}, y={last_pos[1]:.2f}")
        
        # Phase 2: Continue straight for 50 more steps
        logger.info("\n=== Phase 2: Continue - Drive Straight for 50 More Steps ===")
        client.current_phase = "continued"
        
        for step in range(50):
            # Continue straight with full throttle
            action = {
                "throttle": 1.0,  # Full throttle
                "brake": 0.0,
                "steer": 0.0  # Still straight
            }
            
            logger.info(f"Step {step+51}/100: Continuing straight (throttle=1.0, steer=0.0)")
            step_result = client.step(action)
            
            # Save image and position
            if 'observation' in step_result:
                client.extract_and_save_image(
                    step_result['observation'], 
                    step + 51, 
                    'continued',
                    action
                )
                
                # Extract and save position
                position = client.extract_position(step_result['observation'])
                if position:
                    logger.info(f"  Position: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}")
                    client.position_history['continued'].append({
                        'step': step + 51,
                        'x': position[0],
                        'y': position[1],
                        'z': position[2],
                        'action': action
                    })
            
            # Check if terminated
            if step_result.get('terminated', False):
                logger.warning(f"Episode terminated at step {step+51}")
                break
        
        logger.info("Continued trajectory complete (straight driving)")
        
        # Phase 3: Restore and continue straight (SAME as Phase 2 for verification)
        logger.info("\n=== Phase 3: Restore Snapshot and Continue Straight (Same as Phase 2) ===")
        logger.info("Using same steering (0.0) as Phase 2 to verify identical trajectory")
        
        # Log position of Phase 2 end BEFORE restore
        # step_result contains the last observation from Phase 2
        last_phase2_obs = step_result['observation'] if 'observation' in step_result else None
        if last_phase2_obs:
            last_phase2_pos = client.extract_position(last_phase2_obs)
            logger.info(f"BEFORE RESTORE - Phase 2 end position: x={last_phase2_pos[0]:.2f}, y={last_phase2_pos[1]:.2f}, z={last_phase2_pos[2]:.2f}")
        
        # Use notebook restore endpoint with proper velocity restoration
        restore_result = client.restore_snapshot(snapshot_id, use_notebook=True)
        logger.info(f"Restored to step: {restore_result.get('step_count', 'unknown')}")
        
        # Check restore observation
        if 'observation' in restore_result:
            restore_obs = restore_result['observation']
            restore_pos = client.extract_position(restore_obs)
            if restore_pos:
                logger.info(f"AFTER RESTORE - Ego position: x={restore_pos[0]:.2f}, y={restore_pos[1]:.2f}, z={restore_pos[2]:.2f}")
                logger.info(f"Expected position from Phase 1 snapshot: x=580.42, y=3910.73")
                
                if abs(restore_pos[0] - 580.42) > 1.0:
                    logger.error(f"ERROR: Restored to WRONG position! Expected x=580.42, got x={restore_pos[0]:.2f}")
        
        # Verify restore position matches snapshot position
        if 'observation' in restore_result:
            restored_pos = client.extract_position(restore_result['observation'])
            if restored_pos:
                logger.info(f"Restored position: x={restored_pos[0]:.2f}, y={restored_pos[1]:.2f}, z={restored_pos[2]:.2f}")
                
                # Compare with position at step 50 (end of phase 1)
                if client.position_history['initial']:
                    snapshot_pos = client.position_history['initial'][-1]
                    dx = abs(restored_pos[0] - snapshot_pos['x'])
                    dy = abs(restored_pos[1] - snapshot_pos['y'])
                    distance = (dx**2 + dy**2)**0.5
                    logger.info(f"Distance from snapshot point: {distance:.2f}m")
                    if distance > 1.0:
                        logger.warning(f"WARNING: Restored position is {distance:.2f}m away from snapshot point!")
                
                # Save the restored state image
                client.extract_and_save_image(
                    restore_result['observation'], 
                    50,  # This is step 50, the snapshot point
                    'restored'
                )
        
        logger.info("Now driving straight - IDENTICAL to Phase 2 for verification")
        client.current_phase = "restored"
        
        # CRITICAL FIX: Step 050 was already saved above, now we continue from step 51
        # The loop should start at step 51 (index 0 corresponds to step 51)
        for step in range(50):
            # SAME action as Phase 2: Continue straight
            action = {
                "throttle": 1.0,
                "brake": 0.0,
                "steer": 0.0  # STRAIGHT - Same as Phase 2
            }
            
            # This is step 51 and onwards (step+51)
            logger.info(f"Step {step+51}/100 (restored): Driving straight (throttle=1.0, steer=0.0)")
            step_result = client.step(action)
            
            # Save image and position
            if 'observation' in step_result:
                client.extract_and_save_image(
                    step_result['observation'], 
                    step + 51, 
                    'restored',
                    action
                )
                
                # Extract and save position
                position = client.extract_position(step_result['observation'])
                if position:
                    logger.info(f"  Position: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}")
                    client.position_history['restored'].append({
                        'step': step + 51,
                        'x': position[0],
                        'y': position[1],
                        'z': position[2],
                        'action': action
                    })
            
            # Check if terminated
            if step_result.get('terminated', False):
                logger.warning(f"Episode terminated at step {step+51}")
                break
        
        logger.info("Restored trajectory complete (straight driving - should match Phase 2)")
        
        # Save position data to JSON
        logger.info("\n=== Saving Position Data ===")
        position_file = client.save_position_data()
        
        # Create summary
        logger.info("\n=== Creating Summary ===")
        create_summary_image(client.output_dir)
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("TEST COMPLETE - Results saved to: " + str(client.output_dir))
        logger.info("="*60)
        logger.info("\nPhase directories:")
        for phase_name, phase_dir in client.phase_dirs.items():
            num_images = len(list(phase_dir.glob("step_*.png")))
            logger.info(f"  {phase_name}: {num_images} images in {phase_dir}")
        
        logger.info(f"\nPosition data saved to: {position_file}")
        logger.info("\nPosition summary:")
        for phase_name, positions in client.position_history.items():
            if positions:
                first_pos = positions[0]
                last_pos = positions[-1]
                dx = last_pos['x'] - first_pos['x']
                dy = last_pos['y'] - first_pos['y']
                distance = (dx**2 + dy**2)**0.5
                logger.info(f"  {phase_name}: {len(positions)} positions tracked")
                logger.info(f"    Start: ({first_pos['x']:.2f}, {first_pos['y']:.2f})")
                logger.info(f"    End: ({last_pos['x']:.2f}, {last_pos['y']:.2f})")
                logger.info(f"    Distance traveled: {distance:.2f}m")
        
        logger.info("\nKey observations:")
        logger.info("  1. Initial 50 steps: Vehicle drives straight")
        logger.info("  2. Continued 50 steps: Vehicle continues straight (original timeline)")
        logger.info("  3. Restored + 50 steps: Vehicle drives straight (SHOULD MATCH Phase 2!)")
        logger.info("\nVERIFICATION TEST: Phase 3 should be IDENTICAL to Phase 2:")
        logger.info("  - Same actions (throttle=1.0, steer=0.0)")
        logger.info("  - Same positions at each step")
        logger.info("  - Same images (proving perfect restore)")
        logger.info("\nIf positions/images match, the restore is working perfectly!")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def start_single_server():
    """Start a single CARLA server using microservice_manager"""
    logger.info("Starting single CARLA server...")
    
    import subprocess
    import time
    
    # Kill any existing servers more aggressively
    logger.info("Cleaning up existing servers...")
    # First try normal kill
    subprocess.run(["pkill", "-f", "microservice_manager.py"], capture_output=True)
    subprocess.run(["pkill", "-f", "carla_server"], capture_output=True)
    subprocess.run(["pkill", "-f", "carla_server.py"], capture_output=True)
    subprocess.run(["pkill", "-f", "CarlaUE4"], capture_output=True)
    subprocess.run(["pkill", "-f", "CarlaUE4.sh"], capture_output=True)
    subprocess.run(["pkill", "-f", "microservice"], capture_output=True)
    subprocess.run(["pkill", "-f", "CarlaUE4-Linux-Shipping"], capture_output=True)
    time.sleep(2)
    
    # Then force kill anything remaining
    subprocess.run(["pkill", "-9", "-f", "CarlaUE4-Linux-Shipping"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "microservice"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "microservice_manager"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "carla_server"], capture_output=True)
    time.sleep(5)  # Give time for cleanup
    
    # Start single server
    cmd = [
        "python", 
        "microservice_manager.py",
        "--num-services", "1",
        "--startup-delay", "0"
    ]
    
    logger.info(f"Starting server with command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Wait for server to be ready
    logger.info("Waiting for server to be ready...")
    for i in range(60):  # Increased to 120 seconds
        try:
            response = requests.get("http://localhost:8080/health", timeout=2)
            if response.status_code == 200:
                logger.info("Server is ready!")
                return process
        except:
            pass
        time.sleep(2)
        
        # Check if process died
        if process.poll() is not None:
            logger.error("Server process died unexpectedly")
            stdout, _ = process.communicate()
            logger.error(f"Server output:\n{stdout}")
            return None
    
    logger.error("Server failed to start within 120 seconds")
    return None


def main():
    parser = argparse.ArgumentParser(description="Visual test for snapshot/restore")
    parser.add_argument("--port", type=int, default=8080,
                       help="CARLA server API port")
    parser.add_argument("--start-server", action="store_true",
                       help="Start CARLA server automatically")
    parser.add_argument("--output-dir", type=str, default="snapshot_test_images",
                       help="Directory to save test images")
    args = parser.parse_args()
    
    server_process = None
    
    try:
        # Clean up ports before starting
        if args.start_server:
            cleanup_ports([args.port, 2000])  # Clean up API port and CARLA port
            
        # Start server if requested
        if args.start_server:
            server_process = start_single_server()
            if server_process is None:
                logger.error("Failed to start server")
                return
            time.sleep(5)  # Extra wait for stability
        
        # Create client
        client = VisualCarlaClient(
            f"http://localhost:{args.port}",
            output_dir=args.output_dir
        )
        
        # Check server health
        if not client.health_check():
            logger.error(f"Server on port {args.port} is not responding")
            logger.info("Please start the server with:")
            logger.info("  python microservice_manager.py --num-services 1")
            return
        
        logger.info(f"Connected to server on port {args.port}")
        
        # Run visual test
        success = run_visual_test(client)
        
        if success:
            logger.info("\n✅ Visual snapshot test PASSED!")
            logger.info(f"Check images in: {client.output_dir}")
        else:
            logger.error("\n❌ Visual snapshot test FAILED!")
        
        # Cleanup
        client.close()
        
    finally:
        # Stop server if we started it
        if server_process:
            logger.info("Stopping server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except:
                server_process.kill()
            logger.info("Server stopped")


if __name__ == "__main__":
    main()