#!/usr/bin/env python3
"""Test just the snapshot/restore logic without scenarios"""

from world_snapshot import WorldSnapshot
import logging

logging.basicConfig(level=logging.INFO)

# Create a snapshot
print("\n=== Testing WorldSnapshot Class Directly ===\n")

# Create empty snapshot
snapshot = WorldSnapshot("test_snap")
print(f"Created snapshot: {snapshot.snapshot_id}")

# Check if we can save/restore without world
print("\nTrying restore on empty snapshot...")
result = snapshot.restore(None, None)
print(f"Restore result: {result}")

# The real issue: snapshots need vehicles
print("\nSnapshot has vehicles: ", len(snapshot.vehicles))
print("This is why restore fails - no vehicles captured!")

print("\nTo fix this, we need:")
print("1. World object must be passed correctly to capture()")
print("2. Vehicles must exist in the world")
print("3. Restore must handle the vehicles properly")
