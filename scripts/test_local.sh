#!/usr/bin/env bash
# scripts/test_local.sh
# Sends a mock A2A payload to the local server to verify the Phase 2 pipeline.

# Ensure the server is running on the expected port
PORT=${PORT:-9010}
URL="http://localhost:$PORT/"

echo "🚀 Sending mock SWE-Bench task to Purple Agent at $URL..."

# This JSON structure perfectly mimics the AgentBeats envelope we set up in server.py
curl -X POST $URL \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "test-req-12345",
    "method": "agent.task",
    "params": {
      "message": {
        "contextId": "test-session-999",
        "parts": [
          {
            "kind": "data",
            "data": {
              "problem_statement": "The `calculate_total` function in `src/billing.py` crashes with a TypeError when the `items` list is empty. It should return 0 instead.",
              "repo": "alpine", 
              "container_image": "python:3.9-slim",
              "instance_id": "mock-instance-001"
            }
          }
        ]
      }
    }
  }'

echo -e "\n\n🏁 Request finished."