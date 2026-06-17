#!/bin/bash
# _max_cyan_ — project_mxsa — model downloader

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  🦁 simba — download models${NC}"
echo -e "${CYAN}  _max_cyan_ — project_mxsa${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

MODELS_DIR="${DIR}/models"
mkdir -p "${MODELS_DIR}"

# 1. vosk small english model
if [ ! -d "${MODELS_DIR}/vosk-model-small-en-us-0.15" ]; then
    echo -e "${YELLOW}  downloading vosk speech model (~40mb)...${NC}"
    cd "${MODELS_DIR}"
    wget -q "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" -O vosk.zip || { echo -e "${RED}wget failed${NC}"; exit 1; }
    unzip -q vosk.zip || { echo -e "${RED}unzip failed${NC}"; exit 1; }
    rm vosk.zip || true
    echo -e "${GREEN}  ✓ vosk model downloaded${NC}"
else
    echo -e "${GREEN}  ✓ vosk model already exists${NC}"
fi

# 2. mobilenetv2 tflite model
if [ ! -f "${MODELS_DIR}/mobilenetv2_feature_extractor.tflite" ]; then
    echo -e "${YELLOW}  downloading mobilenetv2 tflite model (~14mb)...${NC}"
    cd "${MODELS_DIR}"
    wget -q "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz" -O mn.tgz || { echo -e "${RED}wget failed${NC}"; exit 1; }
    tar xzf mn.tgz || { echo -e "${RED}tar extract failed${NC}"; exit 1; }
    mv mobilenet_v2_1.0_224.tflite "${MODELS_DIR}/mobilenetv2_feature_extractor.tflite" 2>/dev/null || true
    rm -f mn.tgz *.tflite 2>/dev/null || true
    echo -e "${GREEN}  ✓ mobilenetv2 model downloaded${NC}"
else
    echo -e "${GREEN}  ✓ mobilenetv2 model already exists${NC}"
fi

echo -e "${GREEN}All models are ready!${NC}"
