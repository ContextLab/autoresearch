#!/usr/bin/env bash
# Usage: ./scripts/ssh.sh [command...]
# If no command given, opens interactive SSH session.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CREDS_FILE="${PROJECT_DIR}/credentials.yaml"

if [[ ! -f "$CREDS_FILE" ]]; then
    echo "Error: credentials.yaml not found at $CREDS_FILE" >&2
    exit 1
fi

# Parse YAML — extract fields from the "default:" server block
HOST="$(grep 'host:' "$CREDS_FILE" | head -1 | sed 's/.*host:[[:space:]]*//')"
PORT="$(grep 'port:' "$CREDS_FILE" | head -1 | sed 's/.*port:[[:space:]]*//')"
USERNAME="$(grep 'username:' "$CREDS_FILE" | head -1 | sed 's/.*username:[[:space:]]*//')"
PASSWORD="$(grep 'password:' "$CREDS_FILE" | head -1 | sed 's/.*password:[[:space:]]*//')"

if [[ -z "$HOST" || -z "$USERNAME" || -z "$PASSWORD" ]]; then
    echo "Error: Could not parse host, username, or password from $CREDS_FILE" >&2
    exit 1
fi

PORT="${PORT:-22}"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -p "$PORT")

if [[ $# -gt 0 ]]; then
    # Run command remotely
    sshpass -p "$PASSWORD" ssh "${SSH_OPTS[@]}" "${USERNAME}@${HOST}" "$@"
else
    # Interactive session
    sshpass -p "$PASSWORD" ssh "${SSH_OPTS[@]}" "${USERNAME}@${HOST}"
fi
