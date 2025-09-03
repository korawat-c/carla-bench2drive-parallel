@app.post("/restore5")
async def restore_snapshot_v5(request: RestoreRequest):
    """
    Version 5: Simplified restore - positions only, no scenario rebuilding.
    """
    logger.info(f"Restore5: Restoring snapshot {request.snapshot_id}")
    try:
        # Get snapshot
        if request.snapshot_id in sim_state.snapshots:
            snapshot = sim_state.snapshots[request.snapshot_id]
        else:
            snapshot_file = sim_state.snapshot_dir / f"{request.snapshot_id}.pkl"
            if snapshot_file.exists():
                with open(snapshot_file, 'rb') as f:
                    snapshot = pickle.load(f)
            else:
                raise HTTPException(status_code=404, detail=f"Snapshot not found")
        
        world = sim_state.leaderboard_evaluator.world
        manager = sim_state.leaderboard_evaluator.manager
        
        # 1. Completely stop scenario manager
        manager._running = False
        time.sleep(0.5)
        
        # 2. Restore ego vehicle position
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
                ego_vehicle.set_target_velocity(carla.Vector3D(
                    x=vstate.velocity['x'],
                    y=vstate.velocity['y'],
                    z=vstate.velocity['z']
                ))
                logger.info(f"Restored ego to x={vstate.location['x']:.2f}, y={vstate.location['y']:.2f}")
                break
        
        # 3. Restore existing NPC positions (don't spawn new ones!)
        current_vehicles = world.get_actors().filter('vehicle.*')
        npcs = [v for v in current_vehicles if v.id != ego_vehicle.id]
        
        for _, vstate in snapshot.vehicles.items():
            if vstate.is_hero:
                continue
            
            # Find closest NPC to restore
            best_npc = None
            best_dist = float('inf')
            for npc in npcs:
                loc = npc.get_location()
                dist = ((loc.x - vstate.location['x'])**2 + (loc.y - vstate.location['y'])**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_npc = npc
            
            if best_npc and best_dist < 500:  # Only restore if reasonably close
                transform = carla.Transform(
                    carla.Location(x=vstate.location['x'], y=vstate.location['y'], z=vstate.location['z']),
                    carla.Rotation(pitch=vstate.rotation['pitch'], yaw=vstate.rotation['yaw'], roll=vstate.rotation['roll'])
                )
                best_npc.set_simulate_physics(False)
                best_npc.set_transform(transform)
                best_npc.set_simulate_physics(True)
                best_npc.set_target_velocity(carla.Vector3D(
                    x=vstate.velocity['x'],
                    y=vstate.velocity['y'],
                    z=vstate.velocity['z']
                ))
        
        # 4. Restore step count
        if hasattr(snapshot, 'step_count'):
            sim_state.step_count = snapshot.step_count
        
        # 5. Restore observation if available
        restored_observation = None
        if hasattr(snapshot, 'observation') and snapshot.observation:
            restored_observation = snapshot.observation
            logger.info("Restored saved observation")
        
        # 6. Resume scenario manager but prevent rebuilding
        manager._running = True
        
        # Success response
        return {
            "status": "success",
            "message": f"Restored snapshot: {request.snapshot_id}",
            "step_count": sim_state.step_count,
            "observation": restored_observation
        }
        
    except Exception as e:
        logger.error(f"Restore5 failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))