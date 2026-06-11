#!/bin/bash
# Scripts for setting up the Trainer environment

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "Setting up Trainer..."

# Create a virtual environment specifically for the trainer if desired
if [ ! -d "trainer/venv" ]; then
    echo "Creating virtual environment for trainer..."
    python3 -m venv trainer/venv
fi

# Activate venv and install requirements
source trainer/venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r scripts/requirements_trainer.txt

echo "Trainer setup complete! You can start it using scripts/start_trainer.sh"
