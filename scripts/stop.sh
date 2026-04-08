#!/usr/bin/env bash
set -euo pipefail
TARGET="${1:-all}"

stop_screen() {
    local name="$1"
    if screen -list | grep -q "$name"; then
        screen -S "$name" -X quit
        echo "Stopped: $name"
    else
        echo "Not running: $name"
    fi
}

case "$TARGET" in
    all)
        stop_screen "autoresearch-agent"
        for session in $(screen -list | grep "autoresearch-gpu" | awk '{print $1}' | cut -d. -f2-); do
            stop_screen "$session"
        done
        ;;
    agent) stop_screen "autoresearch-agent" ;;
    experiments)
        for session in $(screen -list | grep "autoresearch-gpu" | awk '{print $1}' | cut -d. -f2-); do
            stop_screen "$session"
        done
        ;;
    gpu*) stop_screen "autoresearch-gpu${TARGET#gpu}" ;;
    *) echo "Usage: stop.sh [all|agent|experiments|gpu<N>]" >&2; exit 1 ;;
esac
