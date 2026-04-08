#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CREDS_FILE="${PROJECT_DIR}/credentials.yaml"

# Parse credentials
HOST=$(grep -A5 'default:' "$CREDS_FILE" | grep 'host:' | head -1 | awk '{print $2}')
PORT=$(grep -A5 'default:' "$CREDS_FILE" | grep 'port:' | head -1 | awk '{print $2}')
USERNAME=$(grep -A5 'default:' "$CREDS_FILE" | grep 'username:' | head -1 | awk '{print $2}')
PASSWORD=$(grep -A5 'default:' "$CREDS_FILE" | grep 'password:' | head -1 | awk '{print $2}')
PORT="${PORT:-22}"

LOCAL_RESULTS="${PROJECT_DIR}/results"
mkdir -p "$LOCAL_RESULTS"

SSH_CMD="sshpass -p $PASSWORD ssh -o StrictHostKeyChecking=accept-new -p $PORT"

echo "Syncing from ${HOST}..."
rsync -avz --progress -e "$SSH_CMD" "${USERNAME}@${HOST}:~/autoresearch/checkpoints/" "$LOCAL_RESULTS/checkpoints/"
echo "Synced to: $LOCAL_RESULTS/"
