#!/usr/bin/env bash
# ==============================================================================
# GraphRAG CLI Question Answering Script
# Usage: ./ask.sh "Votre question juridique..."
# Or run without arguments for interactive mode.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "[!] Virtual environment 'venv' not found. Please run ./1_setup_env.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

if [ -z "$1" ]; then
    echo "============================================================"
    echo " Starting Interactive Mode (Type 'exit' to quit)"
    echo "============================================================"
    exec python main.py -i
else
    echo "============================================================"
    echo " Query: $1"
    echo "============================================================"
    exec python main.py -q "$@"
fi
