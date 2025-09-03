"""
Simplified vehicle restore for snapshot - matches vehicles by position
"""
import logging

logger = logging.getLogger(__name__)

def restore_vehicles_simplified(world, snapshot_vehicles, ego_vehicle_id=None):
    """
    Restore vehicles by matching current vehicles to snapshot positions.
    Don't destroy or spawn - just update existing vehicles.
    """
    current_actors = world.get_actors().filter('vehicle.*')
    
    # Build list of non-ego vehicles currently in world
    current_vehicles = []
    for actor in current_actors:
        if ego_vehicle_id and actor.id == ego_vehicle_id:
            continue
        loc = actor.get_location()
        current_vehicles.append({
            'actor': actor,
            'x': loc.x,
            'y': loc.y, 
            'z': loc.z
        })
    
    # Build list of non-ego vehicles from snapshot  
    snap_vehicles = []
    for vehicle_id, state in snapshot_vehicles.items():
        if state.get('is_hero', False):
            continue
        snap_vehicles.append({
            'state': state,
            'x': state['location']['x'],
            'y': state['location']['y'],
            'z': state['location']['z']
        })
    
    logger.info(f"Matching {len(snap_vehicles)} snapshot vehicles to {len(current_vehicles)} current vehicles")
    
    # Match by closest position
    matched = []
    used = set()
    
    for snap in snap_vehicles:
        best_idx = None
        best_dist = float('inf')
        
        for i, curr in enumerate(current_vehicles):
            if i in used:
                continue
            
            dist = ((curr['x'] - snap['x'])**2 + 
                   (curr['y'] - snap['y'])**2 + 
                   (curr['z'] - snap['z'])**2) ** 0.5
            
            if dist < best_dist and dist < 100:  # Within 100m
                best_dist = dist
                best_idx = i
        
        if best_idx is not None:
            matched.append((snap, current_vehicles[best_idx]))
            used.add(best_idx)
            logger.debug(f"Matched vehicle at distance {best_dist:.2f}m")
    
    # Restore matched vehicles
    import carla
    for snap, curr in matched:
        vehicle = curr['actor']
        state = snap['state']
        
        try:
            # Set transform
            transform = carla.Transform(
                carla.Location(
                    x=state['location']['x'],
                    y=state['location']['y'], 
                    z=state['location']['z']
                ),
                carla.Rotation(
                    pitch=state['rotation']['pitch'],
                    yaw=state['rotation']['yaw'],
                    roll=state['rotation']['roll']
                )
            )
            vehicle.set_transform(transform)
            
            # Set velocity
            velocity = carla.Vector3D(
                x=state['velocity']['x'],
                y=state['velocity']['y'],
                z=state['velocity']['z']
            )
            vehicle.set_target_velocity(velocity)
            
            # Apply control
            control = carla.VehicleControl(
                throttle=state['control']['throttle'],
                steer=state['control']['steer'],
                brake=state['control']['brake']
            )
            vehicle.apply_control(control)
            
            logger.debug(f"Restored vehicle {vehicle.id} to position ({state['location']['x']:.1f}, {state['location']['y']:.1f})")
            
        except Exception as e:
            logger.warning(f"Failed to restore vehicle: {e}")
    
    logger.info(f"Restored {len(matched)} vehicles")
    return len(matched)