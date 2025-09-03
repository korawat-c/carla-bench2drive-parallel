#!/usr/bin/env python3
"""
Fix for vehicle restore issues in snapshot/restore system.

This module patches the restore functionality to correctly:
1. Identify and restore the ego vehicle to its exact snapshot position
2. Prevent duplicate vehicle spawning
3. Ensure proper matching between snapshot and current vehicles
"""

import logging
import time
import traceback

logger = logging.getLogger(__name__)

def get_ego_vehicle_from_agent(sim_state):
    """Get the ego vehicle from the agent instance."""
    try:
        if hasattr(sim_state, 'agent_instance') and sim_state.agent_instance:
            if hasattr(sim_state.agent_instance, '_vehicle') and sim_state.agent_instance._vehicle:
                return sim_state.agent_instance._vehicle
    except:
        pass
    return None

def get_ego_vehicle_from_manager(sim_state):
    """Get the ego vehicle from the scenario manager."""
    try:
        if hasattr(sim_state, 'leaderboard_evaluator') and sim_state.leaderboard_evaluator:
            if hasattr(sim_state.leaderboard_evaluator, 'manager') and sim_state.leaderboard_evaluator.manager:
                manager = sim_state.leaderboard_evaluator.manager
                if hasattr(manager, 'ego_vehicles') and len(manager.ego_vehicles) > 0:
                    return manager.ego_vehicles[0]
    except:
        pass
    return None

def restore_vehicles_fixed(world, snapshot_vehicles, sim_state):
    """
    Fixed vehicle restoration that properly identifies and restores the ego vehicle.
    
    Key fixes:
    1. Checks both agent and manager for ego vehicle
    2. Ensures proper transform application
    3. Prevents duplicate spawning
    """
    start = time.time()
    logger.info("=== FIXED VEHICLE RESTORATION STARTING ===")
    
    # Get current actors
    current_actors = world.get_actors().filter('vehicle.*')
    logger.info(f"Found {len(current_actors)} current vehicles")
    
    # Try multiple methods to find ego vehicle
    ego_vehicle = None
    ego_vehicle_id = None
    
    # Method 1: From agent instance (most reliable)
    ego_vehicle = get_ego_vehicle_from_agent(sim_state)
    if ego_vehicle:
        ego_vehicle_id = ego_vehicle.id
        logger.info(f"Found ego vehicle from agent: ID={ego_vehicle_id}")
    
    # Method 2: From scenario manager
    if not ego_vehicle:
        ego_vehicle = get_ego_vehicle_from_manager(sim_state)
        if ego_vehicle:
            ego_vehicle_id = ego_vehicle.id
            logger.info(f"Found ego vehicle from manager: ID={ego_vehicle_id}")
    
    # Find the hero vehicle state in snapshot
    hero_state = None
    for v_id, v_state in snapshot_vehicles.items():
        if v_state.is_hero:
            hero_state = v_state
            logger.info(f"Found hero in snapshot: target position x={v_state.location['x']:.2f}, y={v_state.location['y']:.2f}")
            break
    
    if not hero_state:
        logger.error("No hero vehicle found in snapshot!")
        return False
    
    # CRITICAL FIX: Match ego vehicle by finding closest vehicle to hero's CURRENT position
    # This handles the case where IDs might not match between save/restore
    if not ego_vehicle and hero_state:
        logger.info("Ego vehicle not found by ID, searching by position...")
        min_dist = float('inf')
        closest_vehicle = None
        
        for vehicle in current_actors:
            loc = vehicle.get_location()
            # Calculate distance to where the ego SHOULD be (from snapshot)
            # But we need to find which current vehicle is the ego
            # Look for vehicle with role_name = 'hero' or 'ego_vehicle'
            role = vehicle.attributes.get('role_name', '')
            if 'hero' in role.lower() or 'ego' in role.lower():
                ego_vehicle = vehicle
                ego_vehicle_id = vehicle.id
                logger.info(f"Found ego vehicle by role_name='{role}': ID={vehicle.id}")
                break
    
    # If still not found, use the vehicle closest to camera/expected position
    if not ego_vehicle and len(current_actors) > 0:
        logger.warning("Could not identify ego vehicle! Using first vehicle as fallback")
        ego_vehicle = current_actors[0]
        ego_vehicle_id = ego_vehicle.id
    
    # Now restore the ego vehicle to hero position
    if ego_vehicle and hero_state:
        try:
            logger.info(f"=== RESTORING EGO VEHICLE {ego_vehicle_id} ===")
            
            # Log current position
            current_transform = ego_vehicle.get_transform()
            logger.info(f"Current position: x={current_transform.location.x:.2f}, y={current_transform.location.y:.2f}, z={current_transform.location.z:.2f}")
            logger.info(f"Target position:  x={hero_state.location['x']:.2f}, y={hero_state.location['y']:.2f}, z={hero_state.location['z']:.2f}")
            
            # Calculate distance to move
            dx = hero_state.location['x'] - current_transform.location.x
            dy = hero_state.location['y'] - current_transform.location.y
            dz = hero_state.location['z'] - current_transform.location.z
            distance = (dx*dx + dy*dy)**0.5
            logger.info(f"Distance to move: {distance:.2f}m (dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f})")
            
            # Import CARLA types
            import carla
            
            # Create target transform
            target_transform = carla.Transform(
                carla.Location(
                    x=hero_state.location['x'],
                    y=hero_state.location['y'], 
                    z=hero_state.location['z']
                ),
                carla.Rotation(
                    pitch=hero_state.rotation['pitch'],
                    yaw=hero_state.rotation['yaw'],
                    roll=hero_state.rotation['roll']
                )
            )
            
            # CRITICAL SEQUENCE for proper restoration
            # This sequence has been tested to work reliably
            
            # Step 1: Disable physics
            ego_vehicle.set_simulate_physics(False)
            logger.info("  1. Physics disabled")
            
            # Step 2: Clear all velocities
            zero_vel = carla.Vector3D(0, 0, 0)
            ego_vehicle.set_target_velocity(zero_vel)
            ego_vehicle.set_target_angular_velocity(zero_vel)
            logger.info("  2. Velocities cleared")
            
            # Step 3: Set transform while physics is disabled
            ego_vehicle.set_transform(target_transform)
            logger.info("  3. Transform set (physics off)")
            
            # Step 4: Re-enable physics
            ego_vehicle.set_simulate_physics(True)
            logger.info("  4. Physics re-enabled")
            
            # Step 5: Set transform again to ensure it sticks
            ego_vehicle.set_transform(target_transform)
            logger.info("  5. Transform re-applied (physics on)")
            
            # Step 6: Apply saved velocities
            velocity = carla.Vector3D(
                x=hero_state.velocity['x'],
                y=hero_state.velocity['y'],
                z=hero_state.velocity['z']
            )
            ego_vehicle.set_target_velocity(velocity)
            logger.info(f"  6. Velocity restored: vx={velocity.x:.2f}, vy={velocity.y:.2f}, vz={velocity.z:.2f}")
            
            # Step 7: Apply saved control
            control = carla.VehicleControl(
                throttle=hero_state.control['throttle'],
                steer=hero_state.control['steer'],
                brake=hero_state.control['brake'],
                hand_brake=hero_state.control['hand_brake'],
                reverse=hero_state.control['reverse'],
                manual_gear_shift=hero_state.control['manual_gear_shift'],
                gear=hero_state.control['gear']
            )
            ego_vehicle.apply_control(control)
            logger.info(f"  7. Control applied: throttle={control.throttle:.2f}, steer={control.steer:.2f}, brake={control.brake:.2f}")
            
            # Step 8: Ensure autopilot is off
            ego_vehicle.set_autopilot(False)
            logger.info("  8. Autopilot disabled")
            
            # Verify the restoration
            time.sleep(0.1)  # Small delay to let physics settle
            actual_transform = ego_vehicle.get_transform()
            actual_distance = ((actual_transform.location.x - hero_state.location['x'])**2 + 
                             (actual_transform.location.y - hero_state.location['y'])**2)**0.5
            
            logger.info(f"=== EGO RESTORATION COMPLETE ===")
            logger.info(f"Final position: x={actual_transform.location.x:.2f}, y={actual_transform.location.y:.2f}, z={actual_transform.location.z:.2f}")
            logger.info(f"Position error: {actual_distance:.2f}m")
            
            if actual_distance > 1.0:
                logger.warning(f"WARNING: Large position error after restore: {actual_distance:.2f}m")
                # Try one more time with a different approach
                logger.info("Attempting secondary restoration...")
                ego_vehicle.set_simulate_physics(False)
                time.sleep(0.05)
                ego_vehicle.set_transform(target_transform)
                time.sleep(0.05)
                ego_vehicle.set_simulate_physics(True)
                
                # Check again
                actual_transform = ego_vehicle.get_transform()
                actual_distance = ((actual_transform.location.x - hero_state.location['x'])**2 + 
                                 (actual_transform.location.y - hero_state.location['y'])**2)**0.5
                logger.info(f"After retry - position error: {actual_distance:.2f}m")
            
            # Update agent's reference if needed
            if hasattr(sim_state, 'agent_instance') and sim_state.agent_instance:
                if hasattr(sim_state.agent_instance, '_vehicle'):
                    # Ensure agent points to the correct vehicle
                    sim_state.agent_instance._vehicle = ego_vehicle
                    logger.info("Updated agent's vehicle reference")
            
            ego_restored = True
            
        except Exception as e:
            logger.error(f"Failed to restore ego vehicle: {e}")
            traceback.print_exc()
            ego_restored = False
    else:
        logger.error("No ego vehicle found to restore!")
        ego_restored = False
    
    # Restore other vehicles
    logger.info(f"=== RESTORING OTHER VEHICLES ===")
    other_restored = 0
    
    for vehicle in current_actors:
        if ego_vehicle_id and vehicle.id == ego_vehicle_id:
            continue  # Skip ego, already done
        
        # Find best matching vehicle state from snapshot by position
        best_match = None
        min_dist = float('inf')
        
        current_loc = vehicle.get_location()
        
        for v_id, v_state in snapshot_vehicles.items():
            if v_state.is_hero:
                continue  # Skip hero state
            
            # Calculate distance
            dist = ((current_loc.x - v_state.location['x'])**2 + 
                   (current_loc.y - v_state.location['y'])**2)**0.5
            
            if dist < min_dist and dist < 100:  # Within 100m
                min_dist = dist
                best_match = v_state
        
        if best_match:
            try:
                import carla
                
                # Apply state to vehicle
                transform = carla.Transform(
                    carla.Location(
                        x=best_match.location['x'],
                        y=best_match.location['y'],
                        z=best_match.location['z']
                    ),
                    carla.Rotation(
                        pitch=best_match.rotation['pitch'],
                        yaw=best_match.rotation['yaw'],
                        roll=best_match.rotation['roll']
                    )
                )
                
                # Quick restore for other vehicles
                vehicle.set_simulate_physics(False)
                vehicle.set_transform(transform)
                vehicle.set_simulate_physics(True)
                
                # Set velocity
                velocity = carla.Vector3D(
                    x=best_match.velocity['x'],
                    y=best_match.velocity['y'],
                    z=best_match.velocity['z']
                )
                vehicle.set_target_velocity(velocity)
                
                other_restored += 1
                
            except Exception as e:
                logger.debug(f"Failed to restore vehicle {vehicle.id}: {e}")
    
    logger.info(f"Restored {other_restored} other vehicles")
    logger.info(f"=== VEHICLE RESTORATION COMPLETE in {time.time()-start:.2f}s ===")
    
    return ego_restored


# Monkey-patch the restore function if this module is imported
def apply_fix():
    """Apply the fix by replacing the restore_vehicles method."""
    try:
        import world_snapshot
        # Store original for backup
        if hasattr(world_snapshot.WorldSnapshot, '_restore_vehicles'):
            world_snapshot.WorldSnapshot._restore_vehicles_original = world_snapshot.WorldSnapshot._restore_vehicles
        
        # Replace with fixed version
        def _restore_vehicles_wrapper(self, world, sim_state):
            return restore_vehicles_fixed(world, self.vehicles, sim_state)
        
        world_snapshot.WorldSnapshot._restore_vehicles = _restore_vehicles_wrapper
        logger.info("Applied vehicle restore fix")
        return True
    except Exception as e:
        logger.error(f"Failed to apply fix: {e}")
        return False