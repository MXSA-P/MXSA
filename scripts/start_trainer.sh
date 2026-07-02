#!/bin/bash
# _max_cyan_ — project_mxsa — start trainer interface

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  🎓 simba — start trainer interface${NC}"
echo -e "${CYAN}  _max_cyan_ — project_mxsa${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

# Dependency Validation
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed.${NC}"
    exit 1
fi

# activate virtual environment
if [ -d "trainer/venv" ]; then
    set +u
    source trainer/venv/bin/activate
    set -u
    # Ensure missing UI dependencies are installed
    pip install flask-cors Flask-HTTPAuth >/dev/null 2>&1 || true
else
    echo -e "${RED}Error: virtual environment 'trainer/venv' not found. Please install dependencies first.${NC}"
    exit 1
fi

echo -e "${GREEN}🎓 starting simba trainer...${NC}"
echo -e "${CYAN}   open http://localhost:5000 in your browser${NC}"
echo ""

# open browser (if available)
if command -v xdg-open &> /dev/null; then
    (
        for i in {1..10}; do
            if curl -s http://localhost:5000 > /dev/null; then
                xdg-open http://localhost:5000 2>/dev/null
                break
            fi
            sleep 0.5
        done
    ) &
else
    echo -e "${YELLOW}Note: xdg-open not found, please open browser manually.${NC}"
fi

if ! python3 -m trainer.app "$@"; then
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 130 ] && [ $EXIT_CODE -ne 0 ]; then
        echo -e "${RED}Error: trainer.app failed to start (exit code $EXIT_CODE).${NC}"
        exit $EXIT_CODE
    fi
fi
