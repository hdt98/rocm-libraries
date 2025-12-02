#!/bin/bash
set -e

# Get the directory where this script is located
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${EXPERIMENT_DIR}/.venv"

echo "=========================================="
echo "Setting up Experiment Environment"
echo "=========================================="

# Create virtual environment if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment in ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Install invoke
echo "Installing dependencies..."
pip install invoke

echo "=========================================="
echo "Running Experiments"
echo "=========================================="

# Change to the experiment directory so invoke finds tasks.py easily
cd "${EXPERIMENT_DIR}"

# List tasks to confirm
invoke --list

echo ""
echo ">>> Starting Experiment A <<<"
invoke experiment-a

echo ""
echo ">>> Starting Experiment B <<<"
invoke experiment-b

echo ""
echo ">>> Starting Experiment C <<<"
invoke experiment-c

echo ""
echo ">>> Starting Experiment D <<<"
invoke experiment-d

echo "=========================================="
echo "All Experiments Completed Successfully"
echo "=========================================="

