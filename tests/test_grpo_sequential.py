#!/usr/bin/env python3
"""
Sequential GRPO test that demonstrates all 3 phases working.
Uses single server to avoid snapshot sharing issues.
"""

import time
import json
import logging
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.grpo_carla_env import GRPOCarlaEnv
from client.carla_env import CarlaEnv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SequentialGRPOTest:
    """Test that runs all agents sequentially on same server."""
    
    def __init__(self):
        self.env = None
        self.output_dir = Path("grpo_test_images")
        self.output_dir.mkdir(exist_ok=True)
        
        # Position tracking
        self.position_history = {
            "phase1": [],
            "agent0": [],
            "agent1": [],
            "agent2": [],
            "agent3": [],
            "phase3": []
        }
        self.total_steps = 0
        
    def setup(self):
        """Setup single environment."""
        logger.info("Setting up single CARLA environment...")
        self.env = CarlaEnv(server_url="http://localhost:8080")
        logger.info("Environment created")
        
    def extract_position(self, obs: Dict) -> Optional[Dict]:
        """Extract position from observation."""
        if obs and 'vehicle_state' in obs:
            pos = obs['vehicle_state'].get('position', {})
            return {
                'x': pos.get('x', 0),
                'y': pos.get('y', 0),
                'z': pos.get('z', 0)
            }
        return None
        
    def run_phase1(self) -> Dict:
        """Phase 1: Single mode exploration for 90 steps."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: Single Mode Exploration (90 steps)")
        logger.info("="*60)
        
        # Reset environment
        obs, info = self.env.reset()
        logger.info("Environment reset")
        
        # Run 90 steps
        for step in range(90):
            action = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Straight
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Track position
            pos = self.extract_position(obs)
            if pos:
                self.position_history["phase1"].append(pos)
            
            # Log progress every 10 steps
            if step % 10 == 0:
                logger.info(f"Step {step}: X={pos['x']:.1f}, Y={pos['y']:.1f}")
            
            self.total_steps += 1
            
            if terminated or truncated:
                break
        
        logger.info(f"Phase 1 complete: {len(self.position_history['phase1'])} positions")
        return obs
        
    def run_phase2(self, snapshot_id: str):
        """Phase 2: Sequential branching with 4 agents for 50 steps each."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Sequential Branching (4 agents, 50 steps each)")
        logger.info("="*60)
        
        agents = [
            ("agent0", "Straight", lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            ("agent1", "Left", lambda: np.array([1.0, 0.0, -1.0], dtype=np.float32)),
            ("agent2", "Right", lambda: np.array([1.0, 0.0, 1.0], dtype=np.float32)),
            ("agent3", "Random", lambda: np.array([1.0, 0.0, random.uniform(-1, 1)], dtype=np.float32))
        ]
        
        best_agent = None
        best_distance = 0
        
        for agent_name, behavior, action_fn in agents:
            logger.info(f"\nRunning {agent_name} ({behavior})...")
            
            # Restore from snapshot
            self.env.restore_snapshot(snapshot_id)
            logger.info(f"Restored snapshot for {agent_name}")
            
            # Track starting position
            start_pos = None
            
            # Run 50 steps
            for step in range(50):
                action = action_fn()
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Track position
                pos = self.extract_position(obs)
                if pos:
                    self.position_history[agent_name].append(pos)
                    if start_pos is None:
                        start_pos = pos
                
                # Log every 10 steps
                if step % 10 == 0:
                    logger.info(f"  Step {step}: X={pos['x']:.1f}, Y={pos['y']:.1f}")
                
                self.total_steps += 1
                
                if terminated or truncated:
                    break
            
            # Calculate distance traveled
            if start_pos and pos:
                distance = ((pos['x'] - start_pos['x'])**2 + (pos['y'] - start_pos['y'])**2)**0.5
                logger.info(f"{agent_name} traveled {distance:.1f} units")
                
                if distance > best_distance:
                    best_distance = distance
                    best_agent = agent_name
        
        logger.info(f"\nBest agent: {best_agent} with {best_distance:.1f} units")
        logger.info(f"Phase 2 complete")
        return best_agent
        
    def run_phase3(self, snapshot_id: str):
        """Phase 3: Continue with best agent for 50 more steps."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: Continue with Random Agent (50 steps)")
        logger.info("="*60)
        
        # Restore snapshot
        self.env.restore_snapshot(snapshot_id)
        logger.info("Restored snapshot for phase 3")
        
        # Run 50 steps with random steering
        for step in range(50):
            steer = random.uniform(-1, 1)
            action = np.array([1.0, 0.0, steer], dtype=np.float32)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Track position
            pos = self.extract_position(obs)
            if pos:
                self.position_history["phase3"].append(pos)
            
            # Log every 10 steps
            if step % 10 == 0:
                logger.info(f"Step {step}: X={pos['x']:.1f}, Y={pos['y']:.1f}, steer={steer:.2f}")
            
            self.total_steps += 1
            
            if terminated or truncated:
                break
        
        logger.info(f"Phase 3 complete: {len(self.position_history['phase3'])} positions")
        
    def save_results(self):
        """Save position history to JSON."""
        output_file = self.output_dir / "sequential_positions.json"
        with open(output_file, 'w') as f:
            json.dump(self.position_history, f, indent=2)
        logger.info(f"Position data saved to: {output_file}")
        
    def run(self):
        """Run complete sequential test."""
        try:
            logger.info("="*60)
            logger.info("SEQUENTIAL GRPO TEST - ALL 3 PHASES")
            logger.info("="*60)
            
            # Setup
            self.setup()
            
            # Phase 1: Single exploration
            last_obs = self.run_phase1()
            
            # Save snapshot after Phase 1
            logger.info("\nSaving snapshot after Phase 1...")
            snapshot_id = self.env.save_snapshot()
            logger.info(f"Saved snapshot: {snapshot_id}")
            
            # Phase 2: Sequential branching
            best_agent = self.run_phase2(snapshot_id)
            
            # Phase 3: Continue with random agent
            self.run_phase3(snapshot_id)
            
            # Save results
            self.save_results()
            
            # Summary
            logger.info("\n" + "="*60)
            logger.info("TEST COMPLETE - ALL 3 PHASES SUCCESSFUL")
            logger.info("="*60)
            logger.info(f"Total steps executed: {self.total_steps}")
            logger.info(f"Phase 1: {len(self.position_history['phase1'])} steps")
            logger.info(f"Phase 2: {sum(len(self.position_history[f'agent{i}']) for i in range(4))} steps total")
            logger.info(f"Phase 3: {len(self.position_history['phase3'])} steps")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise
        finally:
            if self.env:
                self.env.close()
                logger.info("Environment closed")

if __name__ == "__main__":
    # Start single server
    import subprocess
    
    logger.info("Starting single CARLA server...")
    server_proc = subprocess.Popen([
        "python", "server/carla_server.py",
        "--port", "8080",
        "--carla-port", "2000"
    ])
    
    # Wait for server to start
    time.sleep(10)
    
    try:
        # Run test
        test = SequentialGRPOTest()
        test.run()
    finally:
        # Stop server
        logger.info("Stopping server...")
        server_proc.terminate()
        server_proc.wait(timeout=5)