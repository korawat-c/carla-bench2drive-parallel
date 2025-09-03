# This is the complete replacement for restore5 endpoint
# Copy this entire function to replace the existing restore5

@app.post("/restore5")
async def restore_snapshot_v5(request: RestoreRequest):
    """
    Version 5: Minimal restore that prevents scenario respawning.
    - Restores positions only
    - Does NOT trigger any scenario rebuilds
    - Keeps existing actors intact
    """
    logger.info(f"=== RESTORE5 START: {request.snapshot_id} ===")
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
        manager = sim_state.leaderboard_evaluator.manager
        
        # 1. CRITICAL: Pause scenario manager to prevent any updates
        original_running = manager._running
        manager._running = False
        logger.info("Paused scenario manager")
        
        # Wait for any ongoing operations to complete
        import time
        time.sleep(0.3)
        
        # 2. Get route scenario and PREVENT it from rebuilding scenarios
        route_scenario = manager._route_scenario if hasattr(manager, '_route_scenario') else manager.scenario
        if route_scenario and hasattr(route_scenario, 'build_scenarios'):
            # Replace build_scenarios with a no-op to prevent spawning
            original_build_scenarios = route_scenario.build_scenarios
            def no_op_build(*args, **kwargs):
                logger.debug("build_scenarios blocked after restore")
                return
            route_scenario.build_scenarios = no_op_build
            logger.info("Disabled build_scenarios to prevent respawning")
        
        # 3. Restore ego vehicle position and velocity
        ego_vehicle = manager.ego_vehicles[0]
        ego_restored = False
        for _, vstate in snapshot.vehicles.items():
            if vstate.is_hero:
                transform = carla.Transform(
                    carla.Location(x=vstate.location['x'], y=vstate.location['y'], z=vstate.location['z']),
                    carla.Rotation(pitch=vstate.rotation['pitch'], yaw=vstate.rotation['yaw'], roll=vstate.rotation['roll'])
                )
                
                # Temporarily disable physics for precise positioning
                ego_vehicle.set_simulate_physics(False)
                ego_vehicle.set_transform(transform)
                ego_vehicle.set_simulate_physics(True)
                
                # Restore velocity
                ego_vehicle.set_target_velocity(carla.Vector3D(
                    x=vstate.velocity['x'],
                    y=vstate.velocity['y'],
                    z=vstate.velocity['z']
                ))
                
                logger.info(f"Restored ego: pos=({vstate.location['x']:.1f}, {vstate.location['y']:.1f}), vel={vstate.velocity['x']:.1f}")
                ego_restored = True
                break
        
        if not ego_restored:
            logger.warning("Could not restore ego vehicle position")
        
        # 4. Restore NPC positions (existing NPCs only - don't spawn new ones!)
        current_vehicles = world.get_actors().filter('vehicle.*')
        npcs = [v for v in current_vehicles if v.id != ego_vehicle.id]
        logger.info(f"Found {len(npcs)} existing NPCs to restore")
        
        # Match NPCs to snapshot positions
        restored_count = 0
        for _, vstate in snapshot.vehicles.items():
            if vstate.is_hero:
                continue
            
            # Find the closest NPC to this snapshot position
            best_match = None
            best_distance = float('inf')
            
            for npc in npcs:
                if not npc.is_alive:
                    continue
                    
                loc = npc.get_location()
                dist = ((loc.x - vstate.location['x'])**2 + 
                       (loc.y - vstate.location['y'])**2) ** 0.5
                
                if dist < best_distance and dist < 300:  # Within 300m
                    best_distance = dist
                    best_match = npc
            
            if best_match:
                transform = carla.Transform(
                    carla.Location(x=vstate.location['x'], y=vstate.location['y'], z=vstate.location['z']),
                    carla.Rotation(pitch=vstate.rotation['pitch'], yaw=vstate.rotation['yaw'], roll=vstate.rotation['roll'])
                )
                
                best_match.set_simulate_physics(False)
                best_match.set_transform(transform)
                best_match.set_simulate_physics(True)
                best_match.set_target_velocity(carla.Vector3D(
                    x=vstate.velocity['x'],
                    y=vstate.velocity['y'],
                    z=vstate.velocity['z']
                ))
                restored_count += 1
        
        logger.info(f"Restored {restored_count} NPC positions")
        
        # 5. Restore simulation state
        if hasattr(snapshot, 'step_count'):
            sim_state.step_count = snapshot.step_count
            logger.info(f"Restored step count: {sim_state.step_count}")
        
        # 6. Restore saved observation if available
        restored_observation = None
        if hasattr(snapshot, 'observation') and snapshot.observation:
            restored_observation = snapshot.observation
            logger.info("Using saved observation from snapshot")
        
        # 7. Resume scenario manager (but scenarios won't rebuild due to our no-op)
        manager._running = original_running
        logger.info("Resumed scenario manager")
        
        # 8. After a few ticks, restore the original build_scenarios
        # (This allows normal operation to continue after restore)
        def delayed_restore():
            time.sleep(2)
            if route_scenario and 'original_build_scenarios' in locals():
                route_scenario.build_scenarios = original_build_scenarios
                logger.info("Re-enabled build_scenarios after restore")
        
        import threading
        threading.Thread(target=delayed_restore, daemon=True).start()
        
        logger.info(f"=== RESTORE5 SUCCESS ===")
        
        return {
            "status": "success",
            "message": f"Restored with scenario preservation: {request.snapshot_id}",
            "step_count": sim_state.step_count,
            "observation": restored_observation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Restore5 failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))