#!/usr/bin/env bash
set -euo pipefail
CHECKPOINT_DIR="$HOME/autoresearch/checkpoints"

echo "=== Autoresearch Status ==="

# Agent status
if screen -list 2>/dev/null | grep -q "autoresearch-agent"; then
    echo "Agent: RUNNING"
else
    echo "Agent: STOPPED"
fi
echo ""

# Read each GPU checkpoint
for dir in "$CHECKPOINT_DIR"/gpu*/; do
    [ -d "$dir" ] || continue
    STATE_FILE="$dir/state.json"
    [ -f "$STATE_FILE" ] || continue
    cat "$STATE_FILE"
    echo "---SEPARATOR---"
done
