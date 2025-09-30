#!/usr/bin/env python3
"""Test creating GRPO environment with available services."""

import sys
import requests
from pathlib import Path

# Add client path to Python path
client_path = str(Path.cwd() / "client")
if client_path not in sys.path:
    sys.path.insert(0, client_path)

def test_service_health():
    """Test which services are responding."""
    available_services = []
    for i in range(2):
        try:
            response = requests.get(f"http://localhost:{8080 + i}/health", timeout=5)
            if response.status_code == 200:
                available_services.append(f"http://localhost:{8080 + i}")
                print(f"✅ Service {i} is ready at {8080 + i}")
            else:
                print(f"⚠️  Service {i} returned status {response.status_code}")
        except Exception as e:
            print(f"❌ Service {i} is not responding: {e}")

    print(f"Available services: {len(available_services)}/2")
    return available_services

def test_grpo_creation():
    """Test creating GRPO environment."""
    print("\n=== Testing GRPO Environment Creation ===")

    # Check service availability
    available_services = test_service_health()

    if len(available_services) == 0:
        print("❌ No services available. Cannot create GRPO environment.")
        return False

    try:
        from grpo_carla_env import GRPOCarlaEnv

        # Create GRPO environment with available services
        print(f"Creating GRPO environment with services: {available_services}")

        env = GRPOCarlaEnv(
            service_urls=available_services,
            render_mode="rgb_array",
            max_steps=100,
            timeout=60.0
        )

        print(f"✅ GRPO Environment created successfully!")
        print(f"   Max branches: {env.max_branches}")
        print(f"   Service URLs: {env.service_urls}")
        print(f"   Current mode: {env.current_mode}")

        # Test basic functionality
        print("\n=== Testing Basic Functionality ===")

        # Create a test action
        def create_action(throttle=0.0, brake=0.0, steer=0.0):
            import numpy as np
            action = np.array([throttle, brake, steer], dtype=np.float32)
            return np.clip(action, env.action_space.low, env.action_space.high)

        action = create_action(throttle=0.5, brake=0.0, steer=0.0)
        print(f"✅ Test action created: {action}")

        # Reset environment
        print("Resetting environment...")
        obs, info = env.reset(options={"route_id": 0})
        print(f"✅ Environment reset successful")
        print(f"   Observation keys: {list(obs.keys())}")

        # Take a step
        print("Taking environment step...")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Environment step successful")
        print(f"   Reward: {reward}")
        print(f"   Terminated: {terminated}")
        print(f"   Truncated: {truncated}")

        # Clean up
        env.close()
        print("✅ Environment closed successfully")

        return True

    except Exception as e:
        print(f"❌ Failed to create GRPO environment: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_grpo_creation()
    if success:
        print("\n🎉 All tests passed! The notebook fix should work.")
    else:
        print("\n❌ Tests failed. Need to debug further.")