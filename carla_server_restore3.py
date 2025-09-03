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
        
        # Restore world state using the snapshot
        if isinstance(snapshot, WorldSnapshot):
            try:
                success = snapshot.restore(sim_state, world)
                if not success:
                    if manager:
                        manager._running = original_running
                    raise HTTPException(status_code=500, detail="Failed to restore snapshot")
            except Exception as e:
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
        
        # Get observation
        if hasattr(snapshot, 'observation') and snapshot.observation:
            obs = snapshot.observation
            logger.info("Using saved observation from snapshot")
            sim_state.last_observation = obs
        else:
            logger.warning("No saved observation - generating new one")
            obs = get_observation_from_state(sim_state.leaderboard_evaluator)
            sim_state.last_observation = obs
        
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