#!/usr/bin/env python3
"""
Test GRPO branching functionality with dynamic single/parallel execution.

This test demonstrates:
1. Single mode exploration
2. Saving snapshot at branching point
3. Switching to parallel mode for GRPO rollouts
4. Collecting multiple trajectories with different exploration
5. Selecting best branch and continuing in single mode
"""

import numpy as np
import logging
import argparse
from pathlib import Path
import json
from typing import List, Dict, Any

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from client.grpo_carla_env import GRPOCarlaEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class GRPOTester:
    """Test GRPO branching with visual output."""
    
    def __init__(self, num_services: int = 4, base_port: int = 8080):
        """Initialize GRPO tester."""
        self.num_services = num_services
        self.base_port = base_port
        
        # Create GRPO environment
        self.env = GRPOCarlaEnv(
            num_services=num_services,
            base_api_port=base_port,
            render_mode="rgb_array",
            max_steps=200
        )
        
        # Trajectory storage
        self.trajectories = []
        self.branch_rewards = []
        
    def run_test(self):
        """Run complete GRPO test workflow."""
        logger.info("="*60)
        logger.info("GRPO DYNAMIC BRANCHING TEST")
        logger.info("="*60)
        
        try:
            # Phase 1: Single mode exploration
            self.phase1_single_exploration()
            
            # Phase 2: Create branching point
            self.phase2_create_snapshot()
            
            # Phase 3: Parallel branching exploration
            self.phase3_parallel_exploration()
            
            # Phase 4: Select best branch and continue
            self.phase4_select_best()
            
            # Phase 5: Continue single mode with best trajectory
            self.phase5_continue_best()
            
            # Report results
            self.report_results()
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise
        finally:
            self.env.close()
            logger.info("Test environment closed")
    
    def phase1_single_exploration(self):
        """Phase 1: Explore in single mode."""
        logger.info("\n=== Phase 1: Single Mode Exploration (Steps 0-49) ===")
        
        # Reset environment
        obs, info = self.env.reset(options={"route_id": 0})
        logger.info("Environment reset in single mode")
        
        # Explore for 50 steps
        trajectory = []
        for step in range(50):
            # Simple forward action
            action = np.array([0.5, 0.0, 0.0])  # throttle, brake, steer
            
            # Use single_step in single mode
            obs, reward, terminated, truncated, info = self.env.single_step(action)
            trajectory.append({
                "step": step,
                "action": action.tolist(),
                "reward": reward,
                "mode": info.get("mode", "unknown")
            })
            
            if step % 10 == 0:
                logger.info(f"  Step {step}: reward={reward:.2f}, mode={info.get('mode')}")
            
            if terminated or truncated:
                logger.info(f"  Episode ended at step {step}")
                break
        
        self.trajectories.append(("single_exploration", trajectory))
        total_reward = sum(t["reward"] for t in trajectory)
        logger.info(f"  Phase 1 complete: Total reward = {total_reward:.2f}")
    
    def phase2_create_snapshot(self):
        """Phase 2: Save snapshot for branching."""
        logger.info("\n=== Phase 2: Creating Branching Point ===")
        
        # Save current state
        snapshot_id = self.env.save_snapshot()
        logger.info(f"  Snapshot saved: {snapshot_id}")
        
        # Get current statistics
        stats = self.env.get_branch_statistics()
        logger.info(f"  Current mode: {stats['mode']}")
        logger.info(f"  Primary reward: {stats.get('primary_reward', 0):.2f}")
        
        self.snapshot_id = snapshot_id
    
    def phase3_parallel_exploration(self):
        """Phase 3: Parallel exploration with branching."""
        logger.info("\n=== Phase 3: Parallel Branching (Critical Scenarios) ===")
        
        # Enable branching with all available services
        self.env.enable_branching(
            snapshot_id=self.snapshot_id,
            num_branches=self.num_services
        )
        logger.info(f"  Branching enabled with {self.num_services} instances")
        logger.info(f"  Mode: {self.env.current_mode}, is_branching: {self.env.is_branching}")
        
        # Store last observations for next actions
        self.last_observations = None
        
        # Parallel exploration with different strategies per branch
        branch_trajectories = [[] for _ in range(self.num_services)]
        
        # Only explore critical scenarios (10-15 steps)
        for step in range(50, 65):  # Reduced from 50-100 to 50-65
            # Check mode before preparing actions
            if not self.env.is_branching:
                logger.error(f"Expected branching mode but got {self.env.current_mode}")
                break
            
            # Generate different actions for each branch (critical scenarios)
            actions = []
            for branch_id in range(self.num_services):
                # Different critical scenarios per branch
                if branch_id == 0:
                    # Branch 0: Sharp left turn
                    action = np.array([0.3, 0.0, -0.8])
                elif branch_id == 1:
                    # Branch 1: Sharp right turn
                    action = np.array([0.3, 0.0, 0.8])
                elif branch_id == 2:
                    # Branch 2: Hard brake
                    action = np.array([0.0, 1.0, 0.0])
                else:
                    # Branch 3: Accelerate straight
                    action = np.array([1.0, 0.0, 0.0])
                
                actions.append(action)
            
            # Use branch_step in branching mode (list input required)
            observations, rewards, terminateds, truncateds, infos = self.env.branch_step(actions)
            self.last_observations = observations
            
            # Record trajectories
            for i in range(self.num_services):
                branch_trajectories[i].append({
                    "step": step,
                    "action": actions[i].tolist(),
                    "reward": rewards[i],
                    "cumulative": infos[i].get("cumulative_reward", 0),
                    "branch_id": i,
                    "position": self._extract_position(observations[i])
                })
            
            # Log progress every 5 steps
            if step % 5 == 0:
                logger.info(f"  Step {step}:")
                for i in range(self.num_services):
                    scenario = ["sharp left", "sharp right", "hard brake", "accelerate"][i]
                    logger.info(f"    Branch {i} ({scenario}): reward={rewards[i]:.2f}, "
                              f"cumulative={infos[i].get('cumulative_reward', 0):.2f}")
            
            # Check if any branch terminated
            if any(terminateds) or any(truncateds):
                terminated_branches = [i for i, t in enumerate(terminateds) if t]
                if terminated_branches:
                    logger.info(f"  Branches {terminated_branches} terminated at step {step}")
        
        # Store branch trajectories
        for i, traj in enumerate(branch_trajectories):
            self.trajectories.append((f"branch_{i}", traj))
            total_reward = sum(t["reward"] for t in traj)
            self.branch_rewards.append(total_reward)
            scenario = ["sharp left", "sharp right", "hard brake", "accelerate"][i]
            logger.info(f"  Branch {i} ({scenario}) total reward: {total_reward:.2f}")
    
    def _extract_position(self, observation):
        """Extract position from observation for external evaluation."""
        if isinstance(observation, dict) and 'vehicle_state' in observation:
            pos = observation['vehicle_state'].get('position', {})
            return {'x': pos.get('x', 0), 'y': pos.get('y', 0), 'z': pos.get('z', 0)}
        return {'x': 0, 'y': 0, 'z': 0}
    
    def phase4_select_best(self):
        """Phase 4: Select best branch using external evaluation."""
        logger.info("\n=== Phase 4: External Evaluation and Selection ===")
        
        # Get branch statistics
        stats = self.env.get_branch_statistics()
        logger.info(f"  Active branches: {stats['active_branches']}")
        logger.info(f"  Internal cumulative rewards: {stats['cumulative_rewards']}")
        
        # External evaluation based on custom metrics
        external_scores = []
        for i in range(self.num_services):
            # Get last position for this branch
            branch_traj = [t for name, traj in self.trajectories if name == f"branch_{i}" for t in traj]
            if branch_traj:
                last_pos = branch_traj[-1].get('position', {})
                
                # Custom scoring: prefer straight progress (higher x) and staying centered (y close to initial)
                # This is just an example - you would use your own metrics
                initial_y = 3910.66  # From initial position in logs
                distance_score = last_pos.get('x', 0)  # Progress forward
                lane_center_score = -abs(last_pos.get('y', initial_y) - initial_y)  # Penalty for deviation
                
                # Combine scores (you can weight differently)
                external_score = distance_score + lane_center_score * 0.5
                external_scores.append(external_score)
                
                scenario = ["sharp left", "sharp right", "hard brake", "accelerate"][i]
                logger.info(f"  Branch {i} ({scenario}): "
                          f"x={last_pos.get('x', 0):.1f}, "
                          f"y={last_pos.get('y', 0):.1f}, "
                          f"score={external_score:.2f}")
            else:
                external_scores.append(float('-inf'))
        
        # Select best based on external evaluation
        best_branch = int(np.argmax(external_scores))
        best_score = external_scores[best_branch]
        scenario = ["sharp left", "sharp right", "hard brake", "accelerate"][best_branch]
        logger.info(f"  Best branch: {best_branch} ({scenario}) with score {best_score:.2f}")
        
        # Switch back to single mode with best branch
        self.env.select_branch(best_branch)
        logger.info(f"  Switched to single mode with branch {best_branch}")
        logger.info(f"  Mode: {self.env.current_mode}, is_branching: {self.env.is_branching}")
        
        self.best_branch = best_branch
        self.external_scores = external_scores
    
    def phase5_continue_best(self):
        """Phase 5: Continue with best trajectory in single mode."""
        logger.info("\n=== Phase 5: Continue Best Trajectory (Single Mode) ===")
        
        # Verify we're back in single mode
        if self.env.is_branching:
            logger.error(f"Expected single mode but still in {self.env.current_mode}")
            return
        
        logger.info(f"  Continuing with branch {self.best_branch} strategy")
        logger.info(f"  Mode: {self.env.current_mode}, is_branching: {self.env.is_branching}")
        
        # Continue for 20 more steps (enough to demonstrate continuation)
        trajectory = []
        for step in range(65, 85):  # Reduced steps for demo
            # Check we're still in single mode
            if self.env.is_branching:
                logger.error("Unexpectedly switched to branching mode")
                break
            
            # Continue with best branch's strategy
            if self.best_branch == 0:
                # Was sharp left - now gentle left
                action = np.array([0.5, 0.0, -0.2])
            elif self.best_branch == 1:
                # Was sharp right - now gentle right
                action = np.array([0.5, 0.0, 0.2])
            elif self.best_branch == 2:
                # Was hard brake - now normal speed
                action = np.array([0.5, 0.0, 0.0])
            else:
                # Was accelerate - continue fast
                action = np.array([0.8, 0.0, 0.0])
            
            # Use single_step (not branch_step) since we're back in single mode
            obs, reward, terminated, truncated, info = self.env.single_step(action)
            trajectory.append({
                "step": step,
                "action": action.tolist(),
                "reward": reward,
                "mode": info.get("mode", "unknown"),
                "position": self._extract_position(obs)
            })
            
            if step % 5 == 0:
                pos = trajectory[-1].get("position", {})
                logger.info(f"  Step {step}: reward={reward:.2f}, mode={info.get('mode')}, "
                          f"x={pos.get('x', 0):.1f}, y={pos.get('y', 0):.1f}")
            
            if terminated or truncated:
                logger.info(f"  Episode ended at step {step}")
                break
        
        self.trajectories.append(("best_continuation", trajectory))
        total_reward = sum(t["reward"] for t in trajectory)
        logger.info(f"  Phase 5 complete: Continuation reward = {total_reward:.2f}")
    
    def report_results(self):
        """Generate test report."""
        logger.info("\n" + "="*60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*60)
        
        # Summary statistics
        logger.info("\nTrajectory Summary:")
        for name, trajectory in self.trajectories:
            total_reward = sum(t["reward"] for t in trajectory)
            logger.info(f"  {name}: {len(trajectory)} steps, total reward = {total_reward:.2f}")
        
        # Branch comparison
        if self.branch_rewards:
            logger.info("\nBranch Comparison:")
            scenarios = ["sharp left", "sharp right", "hard brake", "accelerate"]
            for i, reward in enumerate(self.branch_rewards):
                scenario = scenarios[i] if i < len(scenarios) else f"branch_{i}"
                external = self.external_scores[i] if hasattr(self, 'external_scores') else 0
                marker = " <- SELECTED" if i == self.best_branch else ""
                logger.info(f"  Branch {i} ({scenario}): "
                          f"reward={reward:.2f}, external_score={external:.2f}{marker}")
        
        # Save results to file
        results = {
            "num_services": self.num_services,
            "trajectories": {name: traj for name, traj in self.trajectories},
            "branch_rewards": self.branch_rewards,
            "best_branch": getattr(self, 'best_branch', None)
        }
        
        results_file = Path("grpo_test_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {results_file}")
        
        logger.info("\n" + "="*60)
        logger.info("GRPO BRANCHING TEST COMPLETE")
        logger.info("="*60)


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Test GRPO branching functionality')
    parser.add_argument('--num-services', type=int, default=4,
                       help='Number of services for branching (default: 4)')
    parser.add_argument('--base-port', type=int, default=8080,
                       help='Base API port (default: 8080)')
    parser.add_argument('--start-server', action='store_true',
                       help='Start CARLA server automatically')
    
    args = parser.parse_args()
    
    # Start server if requested
    if args.start_server:
        logger.info("Starting CARLA servers...")
        import subprocess
        import time
        
        # Start microservice manager
        cmd = [
            "python", 
            str(Path(__file__).parent.parent / "server" / "microservice_manager.py"),
            "--num-services", str(args.num_services),
            "--startup-delay", "0"
        ]
        
        server_proc = subprocess.Popen(cmd)
        logger.info(f"Started microservice manager with {args.num_services} services")
        time.sleep(10)  # Wait for servers to start
    
    try:
        # Run test
        tester = GRPOTester(
            num_services=args.num_services,
            base_port=args.base_port
        )
        tester.run_test()
        
    finally:
        if args.start_server and 'server_proc' in locals():
            logger.info("Stopping servers...")
            server_proc.terminate()
            server_proc.wait()


if __name__ == "__main__":
    main()