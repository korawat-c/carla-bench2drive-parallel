#!/usr/bin/env python3
"""
Debug script to test environment initialization step by step
"""
import sys
import time
import requests
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add client path to Python path
client_path = str(Path(__file__).parent / "client")
if client_path not in sys.path:
    sys.path.insert(0, client_path)
    logger.info(f"Added to Python path: {client_path}")

def test_server_connection():
    """Test basic server connection"""
    print("🔍 Testing server connection...")

    try:
        response = requests.get("http://localhost:8080/health", timeout=30)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Server is healthy: {health_data.get('status', 'unknown')}")
            print(f"   Server ID: {health_data.get('server_id', 'unknown')}")
            print(f"   Step count: {health_data.get('step_count', 0)}")
            return True
        else:
            print(f"⚠️ Server responded with status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        return False

def test_reset_endpoint():
    """Test reset endpoint with long timeout"""
    print("\n🔄 Testing reset endpoint...")

    try:
        reset_data = {"route_id": 0, "weather": "ClearNoon"}
        print(f"   Sending reset request: {reset_data}")

        start_time = time.time()
        response = requests.post(
            "http://localhost:8080/reset",
            json=reset_data,
            timeout=180  # 3 minutes timeout
        )
        end_time = time.time()

        print(f"   Reset took {end_time - start_time:.1f} seconds")

        if response.status_code == 200:
            data = response.json()
            print("✅ Reset successful")
            print(f"   Response keys: {list(data.keys())}")
            return True
        else:
            print(f"⚠️ Reset returned status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return False

def test_imports():
    """Test client imports"""
    print("\n📦 Testing client imports...")

    try:
        from carla_env import CarlaEnv
        print("✅ carla_env imported successfully")

        from grpo_carla_env import GRPOCarlaEnv
        print("✅ grpo_carla_env imported successfully")

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_carla_env_creation():
    """Test CarlaEnv creation"""
    print("\n🏗️ Testing CarlaEnv creation...")

    try:
        from carla_env import CarlaEnv

        env = CarlaEnv(
            server_url="http://localhost:8080",
            render_mode="rgb_array",
            max_steps=1000,
            timeout=300
        )

        print("✅ CarlaEnv created successfully")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        return True

    except Exception as e:
        print(f"❌ CarlaEnv creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_grpo_carla_env_creation():
    """Test GRPOCarlaEnv creation"""
    print("\n🏗️ Testing GRPOCarlaEnv creation...")

    try:
        from grpo_carla_env import GRPOCarlaEnv

        env = GRPOCarlaEnv(
            service_urls=["http://localhost:8080"],
            num_services=1,
            render_mode="rgb_array",
            max_steps=1000,
            timeout=300
        )

        print("✅ GRPOCarlaEnv created successfully")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Service URLs: {env.service_urls}")
        print(f"   Max branches: {env.max_branches}")
        return True

    except Exception as e:
        print(f"❌ GRPOCarlaEnv creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests"""
    print("🔧 Environment Initialization Debug Test")
    print("=" * 50)

    tests = [
        ("Server Connection", test_server_connection),
        ("Reset Endpoint", test_reset_endpoint),
        ("Client Imports", test_imports),
        ("CarlaEnv Creation", test_carla_env_creation),
        ("GRPOCarlaEnv Creation", test_grpo_carla_env_creation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))

    print("\n📊 Test Results Summary")
    print("=" * 50)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25s}: {status}")

    failed_tests = [name for name, result in results if not result]
    if failed_tests:
        print(f"\n❌ Failed tests: {', '.join(failed_tests)}")
        return 1
    else:
        print(f"\n✅ All tests passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())