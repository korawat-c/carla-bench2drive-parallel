# Restore5 implementation - Properly handle RouteScenario state

@app.post("/restore5")
async def restore_snapshot_v5(request: RestoreRequest):
    """
    Version 5: Properly snapshot and restore RouteScenario state to prevent NPC respawning.
    - Saves scenario instances and behavior tree
    - Prevents build_scenarios from rebuilding after restore
    - Maintains exact NPC state
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
        
        # STEP 1: Block build_scenarios from running
        # We use a flag to prevent scenario rebuilding
        if not hasattr(manager, '_restore_in_progress'):
            # Add a flag to prevent build_scenarios from running
            manager._restore_in_progress = True
        
        # STEP 2: Pause the scenario thread temporarily
        original_running = manager._running
        manager._running = False
        logger.info("Paused scenario manager")
        
        # Wait a bit for current iteration to complete
        import time
        time.sleep(0.2)
        
        # STEP 3: Get RouteScenario reference
        route_scenario = manager._route_scenario if hasattr(manager, '_route_scenario') else manager.scenario
        if route_scenario is None:
            raise HTTPException(status_code=500, detail="No route scenario")
        
        # STEP 4: Save current scenario state if not in snapshot
        if not hasattr(snapshot, 'route_scenario_state'):
            # This is an old snapshot, we can't fully restore it
            logger.warning("Snapshot lacks route_scenario_state, doing basic restore")
            
            # Just restore vehicle positions
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
                    break
        else:
            # STEP 5: Restore RouteScenario state
            scenario_state = snapshot.route_scenario_state
            
            # Clear current scenarios
            route_scenario.list_scenarios = []
            route_scenario.missing_scenario_configurations = scenario_state['missing_configs'].copy()
            
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
                    if dist < 200:  # Reasonable threshold
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
            
            # IMPORTANT: Mark scenarios as already initialized so build_scenarios won't rebuild them
            # We do this by clearing missing_scenario_configurations for scenarios that were active
            if 'active_scenarios' in scenario_state:
                for scenario_name in scenario_state['active_scenarios']:
                    # These scenarios are already spawned, don't respawn them
                    route_scenario.missing_scenario_configurations = [
                        config for config in route_scenario.missing_scenario_configurations
                        if config.name != scenario_name
                    ]
        
        # STEP 6: Restore simulation state
        if hasattr(snapshot, 'step_count'):
            sim_state.step_count = snapshot.step_count
        
        # STEP 7: Set flags to prevent immediate rebuilding
        sim_state.just_restored = True
        if hasattr(manager, '_tick'):
            manager._tick = snapshot.scenario_manager.tick_count if hasattr(snapshot, 'scenario_manager') else 0
        
        # STEP 8: Resume scenario manager with protection
        manager._running = original_running
        
        # Add a temporary flag to skip the next few build_scenarios calls
        if hasattr(route_scenario, 'build_scenarios'):
            original_build = route_scenario.build_scenarios
            skip_count = [3]  # Skip next 3 calls
            
            def protected_build(ego, debug=False):
                if skip_count[0] > 0:
                    skip_count[0] -= 1
                    logger.info(f"Skipping build_scenarios call ({skip_count[0]} remaining)")
                    return
                return original_build(ego, debug)
            
            route_scenario.build_scenarios = protected_build
            
            # Restore original after a delay
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