#!/usr/bin/env bash
# scripts/cleanup_containers.sh
# Safely stops and removes any orphaned sibling containers created by the Purple Agent.

set -e

echo "🔍 Searching for orphaned Purple Agent containers..."

# Find all containers (running or exited) matching our naming convention
ORPHANS=$(docker ps -a -q --filter="name=purple-exec-")

if [ -z "$ORPHANS" ]; then
    echo "✅ No orphaned containers found. Your environment is clean."
    exit 0
fi

echo "⚠️ Found orphaned containers. Stopping and removing..."

# Stop and remove them
docker stop $ORPHANS > /dev/null 2>&1 || true
docker rm $ORPHANS > /dev/null 2>&1 || true

echo "🧹 Cleanup complete. Reclaimed host resources."