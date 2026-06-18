#!/bin/bash
# _max_cyan_ — project_mxsa — start simba robot

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  🦁 simba — start robot${NC}"
echo -e "${CYAN}  _max_cyan_ — project_mxsa${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo -e "${YELLOW}Cleaning up any dangling zombie processes...${NC}"
pkill -f "python.*simba.main" 2>/dev/null || true
fuser -k 8080/tcp 2>/dev/null || true
sleep 1

# Dependency Validation
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed.${NC}"
    exit 1
fi

if ! command -v sudo &> /dev/null; then
    echo -e "${RED}Error: sudo is not installed.${NC}"
    exit 1
fi

# start pigpiod if not running
if systemctl is-active --quiet pigpiod; then
    echo -e "${GREEN}  ✓ pigpiod is running${NC}"
else
    echo -e "${RED}pigpiod is not running. please start it with 'sudo systemctl start pigpiod'${NC}"
    exit 1
fi

# check and start ollama if available
if systemctl list-unit-files | grep -q "ollama.service"; then
    echo -e "${YELLOW}  checking ollama daemon (systemd)...${NC}"
    if ! systemctl is-active --quiet ollama; then
        sudo systemctl start ollama
        echo -e "${GREEN}  ✓ ollama service started${NC}"
    else
        echo -e "${GREEN}  ✓ ollama service is running${NC}"
    fi
elif command -v ollama &> /dev/null; then
    if ! pgrep -x "ollama" > /dev/null; then
        echo -e "${YELLOW}  starting ollama daemon manually (ollama serve)...${NC}"
        ollama serve > /dev/null 2>&1 &
        sleep 5
        echo -e "${GREEN}  ✓ ollama manual daemon started${NC}"
    else
        echo -e "${GREEN}  ✓ ollama manual daemon is already running${NC}"
    fi
fi

# Ensure model is pulled
if command -v ollama &> /dev/null; then
    echo -e "${YELLOW}  verifying qwen2.5:0.5b model is pulled...${NC}"
    ollama pull qwen2.5:0.5b > /dev/null 2>&1 || echo -e "${RED}  failed to pull qwen2.5:0.5b, ai responses might fail${NC}"
fi

# activate virtual environment
if [ -d "venv" ]; then
    set +u
    source venv/bin/activate
    set -u
else
    echo -e "${RED}Error: virtual environment 'venv' not found. Please run the install script first.${NC}"
    exit 1
fi

# health check
python3 -c 'from simba.core.brain import SimbaBrain; print("brain import OK")' 2>/dev/null || echo 'WARNING: brain import failed'

# trap for clean shutdown
cleanup() {
    echo ""
    echo -e "${YELLOW}shutting down simba...${NC}"
    kill $PID 2>/dev/null || true
    wait $PID 2>/dev/null || true
    echo -e "${GREEN}goodbye! 🦁${NC}"
}
trap cleanup EXIT

echo -e "${GREEN}🦁 starting simba...${NC}"
python3 -m simba.main "$@" &
PID=$!
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8080"
wait $PID
