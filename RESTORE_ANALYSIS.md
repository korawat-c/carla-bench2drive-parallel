# Snapshot/Restore Analysis for GRPO Support

## Problem Summary
After restoring a snapshot, the ego vehicle gets stuck and doesn't respond to control commands, even though:
- The vehicle position is correctly restored
- The agent receives and processes control commands
- The scenario manager appears to be running

## Architecture Overview

### Control Flow (Normal Operation)
```
1. ScenarioManager._tick_scenario() 
   ↓
2. ego_action = self._agent_wrapper()
   ↓
3. AgentWrapper.__call__() → self._agent()
   ↓
4. AutonomousAgent.__call__() → self.run_step(input_data, timestamp)
   ↓
5. APIAgent.run_step() → returns control from self._last_action
   ↓
6. ScenarioManager: self.ego_vehicles[0].apply_control(ego_action)
```

### Key Components
- **LeaderboardEvaluator**: Top-level orchestrator
- **ScenarioManager**: Manages scenario execution and applies controls
- **RouteScenario**: Contains route and ego vehicle configuration
- **APIAgent**: Receives controls from REST API
- **CARLA World**: The simulation environment

## Current Restore Implementation

### 1. Save Snapshot
```python
# Captures:
- All vehicle positions, velocities, controls
- Traffic light states
- Weather conditions
- Scenario state (route index, etc.)
- Agent state (step count, last action)
```

### 2. Restore Process
```python
# Step 1: Restore world state
snapshot.restore(sim_state, world)
  - Restores vehicle positions and velocities
  - Restores traffic lights and weather
  
# Step 2: Get ego vehicle from manager
ego_vehicle = manager.ego_vehicles[0]

# Step 3: Refresh ego vehicle actor reference
fresh_ego = world.get_actor(ego_vehicle.id)
manager.ego_vehicles[0] = fresh_ego

# Step 4: Update agent's vehicle reference
agent._vehicle = fresh_ego

# Step 5: Set manager to running
manager._running = True

# Step 6: Sync with one tick
my_run_scenario_step(...)
```

## Issues Identified

### 1. Stale Actor References
After restoring world state, the CARLA actor IDs might change or the actor objects become stale. The ego vehicle reference in `manager.ego_vehicles[0]` no longer points to a valid, controllable actor.

### 2. Scenario Manager State
The scenario manager has internal state beyond just `_running`:
- `self.scenario_tree` - The behavior tree that might need re-initialization
- `self._agent_wrapper` - May need to re-establish sensor connections
- `self._watchdog` - Timing watchdogs that monitor execution

### 3. Agent-Vehicle Connection
The agent's connection to the vehicle through sensors and control application is broken. The `agent._vehicle` reference alone isn't enough - the entire sensor-control pipeline needs restoration.

### 4. Bench2Drive Building Blocks
The Bench2Drive framework has its own state management through:
- Route scenarios with waypoint tracking
- Traffic scenarios that spawn/despawn actors
- Complex behavior trees that manage the simulation

## Potential Solutions

### Solution 1: Full Scenario Re-initialization
Instead of just updating references, fully re-initialize the scenario:
```python
# After restore, re-run scenario setup
my_load_route_scenario(evaluator, args, route_config, ...)
my_run_scenario_setup(evaluator)
```

### Solution 2: Deep State Restoration
Capture and restore more internal state:
- Scenario manager's full internal state
- Behavior tree state
- All actor references in the scenario tree

### Solution 3: Control Override
Bypass the scenario manager temporarily and apply controls directly:
```python
# After restore, apply control directly to ego vehicle
ego_vehicle = world.get_actor(ego_id)
ego_vehicle.apply_control(control)
```

## Test Results

### Phase 1 (Initial - 50 steps straight)
- Start: x=592.31, y=3910.66
- End: x=580.42, y=3910.73
- **Status: ✅ Working** - Vehicle moves correctly

### Phase 2 (Continued - 50 more steps straight)
- Start: x=579.80, y=3910.74
- End: x=536.94, y=3911.00
- **Status: ✅ Working** - Vehicle continues moving

### Phase 3 (Restored - 50 steps turning right)
- Start: x=579.14, y=3910.73
- End: x=579.14, y=3910.73 (NO MOVEMENT!)
- **Status: ❌ BROKEN** - Vehicle stuck after restore

## Next Steps

1. **Investigate Scenario Re-initialization**: Try re-running the scenario setup after restore
2. **Check Actor ID Persistence**: Verify if actor IDs change after world state restore
3. **Test Direct Control Application**: Bypass scenario manager to test if vehicle can be controlled
4. **Examine Bench2Drive Source**: Look deeper into how scenarios are initialized and managed

## Critical Finding

The core issue is that **Bench2Drive's scenario management system is not designed for mid-execution state restoration**. The framework expects a continuous execution from start to finish. Implementing true GRPO support requires either:
1. Modifying Bench2Drive internals to support state restoration
2. Creating a wrapper that can re-initialize scenarios while preserving state
3. Finding an alternative approach to branching that doesn't require full restoration

## UPDATE: Complete Scenario Manager Restoration

We've now implemented complete scenario manager state capture and restoration:

### New ScenarioManagerState Class
Captures ALL internal state of the ScenarioManager:
- Route index and repetition number
- All timing variables (tick_count, timestamps, durations)
- Execution state (_running, _debug_mode, _timeout)
- Vehicle and actor IDs
- Scenario tree status and blackboard
- Agent wrapper state and sensors

### Enhanced Capture Process
1. Verify manager exists via `evaluator.manager`
2. Capture all manager attributes using getattr
3. Save ego_vehicle_ids and other_actor_ids
4. Capture scenario tree status and blackboard data
5. Save agent wrapper state including wallclock_t0

### Enhanced Restore Process
1. Restore all manager attributes directly
2. Refresh ego vehicle references with fresh actors from world
3. Update agent's vehicle reference
4. Restore scenario tree blackboard (including AV_control)
5. Ensure watchdogs are properly managed
6. Set manager._running = True to ensure continuation

### Key Insight from User
"why we restore only self.ego_vehicles[0]. Can we just restore the whole scenario manager at that time step?"

This was correct - we need to restore the ENTIRE scenario manager state, not just update vehicle references. The manager contains critical execution state that must be preserved for the scenario to continue running after restore.