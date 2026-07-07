#!/usr/bin/env bash
# =============================================================================
# api_demo.sh — end-to-end walkthrough of the SRE Incident Copilot HTTP API.
#
# Drives one incident from a seeded scenario: list scenarios -> start ->
# inspect -> approve the high-risk remediation -> audit -> LLM-as-judge eval.
#
# Prereqs:
#   * API running:      uvicorn app.api.server:app --reload
#   * Gateway running:   docker compose up -d litellm   (real model calls)
#   * jq installed:      brew install jq
#
# Usage:
#   ./scripts/api_demo.sh                        # default scenario
#   BASE_URL=http://localhost:8000 SCENARIO=cdn-latency-spike ./scripts/api_demo.sh
# =============================================================================
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
SCENARIO="${SCENARIO:-checkout-5xx-spike}"

hr() { printf '\n\033[1;36m== %s ==\033[0m\n' "$1"; }

hr "1. Health check"
curl -sS "${BASE_URL}/healthz" | jq .

hr "2. List available scenarios"
curl -sS "${BASE_URL}/scenarios" | jq -r '.[] | "- \(.id)  [\(.severity)]  \(.title)"'

hr "3. Start incident from scenario: ${SCENARIO}"
START_RESP="$(curl -sS -X POST "${BASE_URL}/incidents" \
  -H 'Content-Type: application/json' \
  -d "{\"scenario_id\": \"${SCENARIO}\"}")"
echo "${START_RESP}" | jq '{session_id, status, completed_agents, pending_approval}'

SESSION_ID="$(echo "${START_RESP}" | jq -r '.session_id')"
STATUS="$(echo "${START_RESP}" | jq -r '.status')"
echo "session_id=${SESSION_ID}  status=${STATUS}"

hr "4. Findings so far"
curl -sS "${BASE_URL}/incidents/${SESSION_ID}" | jq '.findings'

if [ "${STATUS}" = "awaiting_approval" ]; then
  hr "5. Human-in-the-loop: approve the high-risk remediation"
  curl -sS -X POST "${BASE_URL}/incidents/${SESSION_ID}/approve" \
    -H 'Content-Type: application/json' \
    -d '{"approved": true}' | jq '{status, completed_agents, actions_performed}'
else
  hr "5. No approval required (no high-risk step gated)"
fi

hr "6. Audit trail (harm-tier decisions)"
curl -sS "${BASE_URL}/incidents/${SESSION_ID}/audit" \
  | jq -r '.[] | "- \(.tool_name) [\(.harm_tier)] -> \(.decision) (executed=\(.executed))"'

hr "7. LLM-as-judge evaluation vs. ground truth"
curl -sS -X POST "${BASE_URL}/incidents/${SESSION_ID}/eval" \
  | jq '{passed, mean_score, dimensions: [.dimensions[] | {dimension, score, passed}]}'

hr "Done"
