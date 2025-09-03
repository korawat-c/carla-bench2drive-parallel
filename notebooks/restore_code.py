# Function to restore a snapshot (adapted from carla_server.py)
def restore_snapshot(snapshot_id):
    """Restore world state from snapshot"""
    try:
        print(f"\n=== Starting Restore of Snapshot: {snapshot_id} ===")
        
        # Check if snapshot exists
        if snapshot_id not in sim_state.snapshots:
            # Try to load from disk
            snapshot_file = sim_state.snapshot_dir / f"{snapshot_id}.pkl"
            if snapshot_file.exists():
                print(f"Loading snapshot from disk: {snapshot_file}")
                with open(snapshot_file, 'rb') as f:
                    snapshot = pickle.load(f)
                    sim_state.snapshots[snapshot_id] = snapshot
            else:
                raise ValueError(f"Snapshot not found: {snapshot_id}")
        
        snapshot = sim_state.snapshots[snapshot_id]
        print(f"Found snapshot: {snapshot_id}")
        print(f"  - Contains {len(snapshot.vehicles)} vehicles")
        if hasattr(snapshot, 'phase_marker'):
            print(f"  - Phase marker: {snapshot.phase_marker}")
        
        # Get the world
        world = sim_state.leaderboard_evaluator.world
        if world is None:
            raise ValueError("World is not available")
        
        # Log ego position before restore
        ego_vehicle = None
        if hasattr(sim_state.leaderboard_evaluator, 'agent_instance'):
            agent = sim_state.leaderboard_evaluator.agent_instance
            if hasattr(agent, '_vehicle') and agent._vehicle:
                ego_vehicle = agent._vehicle
                transform = ego_vehicle.get_transform()
                print(f"Current ego position BEFORE restore: x={transform.location.x:.2f}, y={transform.location.y:.2f}, z={transform.location.z:.2f}")
        
        # Restore the world state
        print("\nRestoring world state...")
        snapshot.restore(world, sim_state.leaderboard_evaluator)
        
        # Verify ego position after restore
        if ego_vehicle:
            transform = ego_vehicle.get_transform()
            print(f"Ego position AFTER restore: x={transform.location.x:.2f}, y={transform.location.y:.2f}, z={transform.location.z:.2f}")
        
        # Check what was in the snapshot
        for vehicle_id, vehicle_state in snapshot.vehicles.items():
            if vehicle_state.is_hero:
                print(f"Snapshot had ego at: x={vehicle_state.location['x']:.2f}, y={vehicle_state.location['y']:.2f}, z={vehicle_state.location['z']:.2f}")
                break
        
        # Restore observation if available
        if hasattr(snapshot, 'observation') and snapshot.observation:
            sim_state.last_observation = snapshot.observation
            print(f"Restored observation with {len(snapshot.observation.get('images', {}))} images")
        
        # Update step count if stored
        if hasattr(snapshot, 'step_count'):
            sim_state.step_count = snapshot.step_count
            print(f"Restored step count: {sim_state.step_count}")
        
        print(f"\n✅ Successfully restored snapshot: {snapshot_id}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error restoring snapshot: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test restore functionality
print("\n" + "="*60)
print("Testing Snapshot Restore")
print("="*60)

# First, let's save the current position for comparison
if hasattr(leaderboard_evaluator.agent_instance, '_vehicle') and leaderboard_evaluator.agent_instance._vehicle:
    current_transform = leaderboard_evaluator.agent_instance._vehicle.get_transform()
    print(f"Current position: x={current_transform.location.x:.2f}, y={current_transform.location.y:.2f}")

# List available snapshots
print(f"\nAvailable snapshots: {list(sim_state.snapshots.keys())}")

# Try to restore the snapshot we just saved
if snapshot_id:
    success = restore_snapshot(snapshot_id)
    if success:
        print("\nSnapshot restored successfully!")
        # Check if position changed
        if hasattr(leaderboard_evaluator.agent_instance, '_vehicle') and leaderboard_evaluator.agent_instance._vehicle:
            restored_transform = leaderboard_evaluator.agent_instance._vehicle.get_transform()
            print(f"Position after restore: x={restored_transform.location.x:.2f}, y={restored_transform.location.y:.2f}")
else:
    print("No snapshot available to restore")