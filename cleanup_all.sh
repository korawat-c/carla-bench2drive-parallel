#!/bin/bash
# COMPREHENSIVE CLEANUP SCRIPT - KILL EVERYTHING CARLA/SERVER RELATED

echo "=== COMPREHENSIVE CLEANUP STARTING ==="

# Kill all Python processes related to our services
echo "Killing Python processes..."
pkill -9 -f "carla_server.py"
pkill -9 -f "microservice_manager.py"
pkill -9 -f "server_manager.py"
pkill -9 -f "test_grpo"
pkill -9 -f "test_snapshot"
pkill -9 -f "uvicorn"
pkill -9 -f "fastapi"

# Kill CARLA processes
echo "Killing CARLA processes..."
pkill -9 -f "CarlaUE4"
pkill -9 -f "CARLA"

# Kill processes on ALL ports we use
echo "Killing processes on ports..."
for port in 8080 8081 8082 8083 8084 8085; do
    echo "  Cleaning port $port"
    lsof -ti:$port 2>/dev/null | xargs -r kill -9 2>/dev/null
    fuser -k $port/tcp 2>/dev/null
done

# CARLA ports
for port in 2000 2002 2004 2006 2008; do
    echo "  Cleaning port $port"
    lsof -ti:$port 2>/dev/null | xargs -r kill -9 2>/dev/null
    fuser -k $port/tcp 2>/dev/null
done

# Traffic Manager ports
for port in 3000 3002 3004 3006 3008; do
    echo "  Cleaning port $port"
    lsof -ti:$port 2>/dev/null | xargs -r kill -9 2>/dev/null
    fuser -k $port/tcp 2>/dev/null
done

# Kill any remaining Python processes that might be holding ports
echo "Final cleanup..."
ps aux | grep -E "python.*8080|python.*2000|python.*3000" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null

# Clean up any zombie processes
echo "Cleaning zombie processes..."
ps aux | grep defunct | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null

# Remove any lock files
echo "Removing lock files..."
rm -f /tmp/*.lock 2>/dev/null
rm -f /tmp/carla_* 2>/dev/null

echo "=== CLEANUP COMPLETE ==="

# Verify no processes on our ports
echo ""
echo "Verification:"
for port in 8080 8081 8082 2000 2002 3000; do
    if lsof -ti:$port 2>/dev/null; then
        echo "WARNING: Port $port still has process!"
    else
        echo "✓ Port $port is free"
    fi
done

echo ""
echo "All cleanup operations completed!"