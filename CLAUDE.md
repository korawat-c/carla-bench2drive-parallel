# CLAUDE.md - Bench2Drive Gymnasium Microservices

## Overview

A **production-ready microservices architecture** for running CARLA + Bench2Drive environments with true parallel execution and GRPO (Generative Reinforcement Learning via Pairwise Ranking Optimization) support. This implementation provides a distributed, scalable solution for RL training with real CARLA instances - **NO MOCKING**.

## Core Architecture

### Design Philosophy
- **Real CARLA Only**: Every environment connects to actual running CARLA instances
- **Microservices Pattern**: Each CARLA instance runs as an independent service with REST API
- **Gymnasium Compliance**: Strict adherence to Gymnasium API specification
- **GRPO-Ready**: Full snapshot/restore support for multi-turn rollouts with branching
- **Production-Grade**: Robust error handling, resource management, and monitoring

### System Architecture
```
[RL Training Framework (VeRL/RLlib/etc.)]
           |
    [Gymnasium API Layer]
           |
    [CarlaEnv Instances]
           |
    [REST API Gateway]
           |
   [Microservice Layer]
    ├── Service 0: CARLA:2000, API:8080, GPU:0
    ├── Service 1: CARLA:2002, API:8081, GPU:0  
    ├── Service 2: CARLA:2004, API:8082, GPU:1
    └── Service N: CARLA:200N, API:808N, GPU:M
```

## Core Components

### 1. **carla_server.py** (Main Server Implementation)
The heart of the system - a complete CARLA server with REST API:
- Integrates CARLA 0.9.15 with Bench2Drive building blocks
- Provides REST endpoints for Gymnasium interface
- Handles snapshot/restore for GRPO branching
- Manages ego vehicle, sensors, and scenario execution
- **Key Features**:
  - Full leaderboard integration
  - Real-time sensor data streaming
  - Deterministic replay support
  - GPU-safe multi-instance operation

### 2. **microservice_manager.py** (Service Orchestrator)
Manages multiple CARLA microservices for parallel training:
- Spawns and monitors multiple carla_server instances
- Handles port allocation and GPU assignment
- Provides health monitoring and automatic recovery
- **Key Features**:
  - Dynamic scaling (add/remove instances)
  - Load balancing across GPUs
  - Automatic port conflict resolution
  - Graceful shutdown and cleanup

### 3. **carla_env.py** (Gymnasium Environment)
Standard Gymnasium-compliant environment interface:
- Inherits from `gymnasium.Env`
- Implements `reset()`, `step()`, `render()`, `close()`
- Communicates with CARLA servers via REST API
- **Observation Space**:
  ```python
  Dict({
      'rgb': Box(0, 255, (3, 224, 400), uint8),  # Camera images
      'vehicle_state': Dict({...}),               # Position, velocity, etc.
      'navigation': Dict({...}),                   # Route info, commands
      'measurements': Dict({...})                  # Speed, collision, etc.
  })
  ```
- **Action Space**:
  ```python
  Box([-1, -1, 0], [1, 1, 1], (3,), float32)  # [steer, throttle, brake]
  ```

### 4. **world_snapshot.py** (GRPO Support System)
Sophisticated snapshot/restore mechanism for GRPO:
- Captures complete world state (vehicles, pedestrians, traffic)
- Preserves scenario manager and agent states
- Fast restoration for branching (< 1 second)
- **Snapshot Components**:
  - All actor positions and velocities
  - Traffic light states and timers
  - Weather and time of day
  - Scenario progress and triggers
  - Agent internal state

### 5. **resource_manager.py** (Resource Allocation)
Intelligent resource management for multi-instance operation:
- GPU memory allocation and monitoring
- Port assignment (CARLA, Traffic Manager, API)
- Process group isolation
- Memory leak detection and mitigation

### 6. **vectorized_env.py** (Parallel Environment Wrapper)
Efficient parallel environment execution:
- Compatible with `gymnasium.vector.AsyncVectorEnv`
- Batch action/observation processing
- Synchronized stepping across instances
- Automatic failure recovery

### 7. **api_agent.py** (Bench2Drive Agent Interface)
Bridge between Gymnasium actions and Bench2Drive agent:
- Receives actions via REST API
- Integrates with Bench2Drive's leaderboard system
- Handles sensor setup and data collection
- Provides compatibility with existing Bench2Drive agents

## Installation & Setup

### Prerequisites
```bash
# CARLA 0.9.15 
cd /path/to/carla-0.9.15
./CarlaUE4.sh -help  # Verify installation

# Python dependencies
pip install fastapi uvicorn gymnasium numpy pillow pyyaml psutil requests
```

### Quick Start
```bash
# 1. Start a single CARLA microservice
python microservice_manager.py --num-services 1

# 2. Run a test episode
python test_visual_snapshot2.py

# 3. Start multiple services for parallel training
python microservice_manager.py --num-services 4 --gpus 0,1
```

## Usage Examples

### Basic Single Environment
```python
from bench2drive_microservices import CarlaEnv

# Create environment
env = CarlaEnv(server_url="http://localhost:8080")

# Standard Gymnasium loop
obs, info = env.reset()
for _ in range(1000):
    action = [0.0, 0.5, 0.0]  # steer, throttle, brake
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
```

### Parallel Training with VeRL
```python
from bench2drive_microservices import VectorizedCarlaEnv

# Create 4 parallel environments
vec_env = VectorizedCarlaEnv(
    server_urls=[f"http://localhost:{8080+i}" for i in range(4)]
)

# Batch processing
observations = vec_env.reset()
for _ in range(1000):
    actions = policy(observations)  # Your policy network
    observations, rewards, dones, infos = vec_env.step(actions)
```

### GRPO with Branching
```python
# Save initial state
obs, _ = env.reset()
snapshot_id = env.save_snapshot()

# Collect multiple rollouts from same state
rollouts = []
for branch in range(4):
    env.restore_snapshot(snapshot_id)
    trajectory = collect_trajectory(env, policy, exploration_noise=branch*0.1)
    rollouts.append(trajectory)

# Select best trajectories for policy update
best_rollouts = select_top_k(rollouts, k=2)
policy.update_grpo(best_rollouts)
```

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check server health |
| `/reset` | POST | Reset environment |
| `/step` | POST | Execute action |
| `/observation` | GET | Get current observation |
| `/render` | GET | Get RGB image |
| `/snapshot/create` | POST | Create world snapshot |
| `/snapshot/restore` | POST | Restore from snapshot |
| `/snapshot/list` | GET | List available snapshots |
| `/info` | GET | Get environment info |

### Environment Configuration
```python
config = {
    'server_url': 'http://localhost:8080',
    'timeout': 60.0,           # API request timeout
    'max_steps': 1000,         # Episode length
    'frame_skip': 1,           # Action repeat
    'reward_type': 'dense',    # 'dense' or 'sparse'
    'render_mode': 'rgb_array' # 'human' or 'rgb_array'
}
```

## Testing

### Visual Snapshot Test
The main test file `test_visual_snapshot2.py` provides comprehensive testing:
```bash
# Test single instance with visual output
python test_visual_snapshot2.py --save-images

# Test parallel instances
python test_visual_snapshot2.py --num-services 4

# Test GRPO branching
python test_visual_snapshot2.py --test-branching
```

### Component Tests
- `test_component_verification.py` - Verify all components save/restore correctly
- `simple_snapshot_test.py` - Basic snapshot functionality
- `quick_snapshot_test.py` - Performance benchmarking

## File Organization

### Core Files (Production)
```
bench2drive_microservices/
├── carla_server.py           # Main CARLA server with REST API
├── microservice_manager.py    # Multi-service orchestrator
├── carla_env.py              # Gymnasium environment
├── world_snapshot.py         # GRPO snapshot system
├── resource_manager.py       # Resource allocation
├── vectorized_env.py         # Parallel environments
├── api_agent.py              # Bench2Drive agent interface
└── __init__.py               # Package exports
```

### Test Files (Development)
```
bench2drive_microservices/
├── test_visual_snapshot2.py  # Main visual test
├── test_component_verification.py
├── simple_snapshot_test.py
└── quick_snapshot_test.py
```

### Auxiliary Files
```
bench2drive_microservices/
├── notebooks/                # Jupyter notebooks for debugging
│   ├── debug_snapshot_restore.ipynb
│   └── step_by_step_*.ipynb
├── logs/                     # Service logs
└── server_manager.py         # (deprecated - use microservice_manager)
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Kill processes on specific ports
   lsof -ti:8080 | xargs kill -9
   lsof -ti:2000 | xargs kill -9
   ```

2. **GPU Memory Issues**
   ```bash
   # Monitor GPU usage
   watch -n 1 nvidia-smi
   
   # Set memory growth for TensorFlow
   export TF_FORCE_GPU_ALLOW_GROWTH=true
   ```

3. **CARLA Connection Timeout**
   - Increase timeout in config
   - Check CARLA logs: `tail -f ~/carla/CarlaUE4/Saved/Logs/CarlaUE4.log`
   - Ensure CARLA version matches (0.9.15)

4. **Snapshot/Restore Failures**
   - Verify sufficient disk space for snapshots
   - Check snapshot compatibility across CARLA versions
   - Clear old snapshots: `rm -rf /tmp/carla_snapshots/*`

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

env = CarlaEnv(server_url="http://localhost:8080", debug=True)
```

## Performance Optimization

### GPU Optimization
- Distribute services across GPUs: `--gpus 0,0,1,1`
- Use offscreen rendering: `-RenderOffScreen`
- Limit sensor resolution for training

### Network Optimization
- Use local connections when possible
- Enable response compression
- Batch observations for multiple agents

### Memory Management
- Periodic server restarts for long training
- Snapshot cleanup after episodes
- Monitor memory leaks with `psutil`

## Future Improvements

### Planned Features
- [ ] Kubernetes deployment for cloud scaling
- [ ] gRPC support for lower latency
- [ ] Built-in curriculum learning
- [ ] Advanced scenario generation
- [ ] Multi-agent support
- [ ] Docker containerization

### Known Limitations
- Snapshot size grows with world complexity
- Limited to local network communication
- Requires manual GPU assignment
- No built-in checkpointing for training

## Contributing

We follow strict code quality standards:
1. Type hints for all functions
2. Comprehensive docstrings
3. Unit tests for new features
4. No mocking - test with real CARLA

## License

This project is part of the Bench2Drive framework and follows its licensing terms.

## Acknowledgments

- CARLA Simulator Team
- Bench2Drive Contributors
- Gymnasium/OpenAI Gym Community
- VeRL Framework Developers

---

**Remember**: This implementation uses **REAL CARLA INSTANCES ONLY** - no mocking, no shortcuts. What you test is what you deploy in production.