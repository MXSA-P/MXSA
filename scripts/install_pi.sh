#!/bin/bash
# _max_cyan_ — project_mxsa — raspberry pi setup script
# run as: sudo bash scripts/install_pi.sh

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ---- disk space check ----
FREE_KB=$(df / --output=avail | tail -1 | tr -d ' ')
REQUIRED_KB=2097152  # 2 GB in KB
if [ "$FREE_KB" -lt "$REQUIRED_KB" ]; then
    FREE_MB=$((FREE_KB / 1024))
    echo -e "${RED}⚠ Warning: Only ${FREE_MB} MB free on /. At least 2 GB recommended.${NC}"
    echo -e "${YELLOW}  Installation may fail due to insufficient disk space.${NC}"
    sleep 3
else
    FREE_MB=$((FREE_KB / 1024))
    echo -e "${GREEN}Disk space OK (${FREE_MB} MB free)${NC}"
fi

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  🦁 simba — raspberry pi installation${NC}"
echo -e "${CYAN}  _max_cyan_ — project_mxsa${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

# Dependency validation
for cmd in apt-get systemctl python3 grep awk; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}Error: required command '$cmd' is not installed.${NC}"
        exit 1
    fi
done

# ---- python version check ----
if command -v python3 &> /dev/null; then
    PY_VER_FULL=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
        echo -e "${RED}Error: Python >= 3.9 is required (found ${PY_VER_FULL}).${NC}"
        exit 1
    fi
    echo -e "${GREEN}Python version OK (${PY_VER_FULL})${NC}"
else
    echo -e "${RED}Error: python3 is not installed.${NC}"
    exit 1
fi

if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: Please run as root (sudo bash scripts/install_pi.sh)${NC}"
    exit 1
fi

echo -e "${YELLOW}Validating GPU Memory for 5MP CSI Camera...${NC}"
GPU_MEM=$(vcgencmd get_mem gpu | grep -o -E '[0-9]+' || echo "0")
if [ "$GPU_MEM" -lt 128 ]; then
    echo -e "${RED}Warning: GPU memory is less than 128MB ($GPU_MEM MB).${NC}"
    echo -e "${YELLOW}A 5MP CSI camera requires at least 128MB of GPU memory to process frames.${NC}"
    echo -e "${YELLOW}Please run 'sudo raspi-config' -> Performance Options -> GPU Memory -> set to 128 or higher.${NC}"
    sleep 3
else
    echo -e "${GREEN}GPU Memory OK ($GPU_MEM MB)${NC}"
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${PROJECT_DIR}/models"
DATA_DIR="${PROJECT_DIR}/data"

# FIX: Bookworm moved boot config from /boot/config.txt to /boot/firmware/config.txt
BOOT_CFG="/boot/config.txt"
[ -f "/boot/firmware/config.txt" ] && BOOT_CFG="/boot/firmware/config.txt"
echo -e "${GREEN}  boot config: ${BOOT_CFG}${NC}"

# ---- enable hardware interfaces ----
echo -e "${YELLOW}[1/7] enabling hardware interfaces...${NC}"

# enable i2c
if ! grep -q "dtparam=i2c_arm=on" "${BOOT_CFG}" 2>/dev/null; then
    echo "dtparam=i2c_arm=on" >> "${BOOT_CFG}" || echo -e "${RED}Failed to enable i2c${NC}"
    echo -e "${GREEN}  ✓ i2c enabled${NC}"
fi

# enable i2s
if ! grep -q "dtparam=i2s=on" "${BOOT_CFG}" 2>/dev/null; then
    echo "dtparam=i2s=on" >> "${BOOT_CFG}" || echo -e "${RED}Failed to enable i2s${NC}"
    echo -e "${GREEN}  ✓ i2s enabled${NC}"
fi

# enable inmp441 i2s microphone
if ! grep -q "dtoverlay=googlevoicehat-soundcard" "${BOOT_CFG}" 2>/dev/null; then
    echo "dtoverlay=googlevoicehat-soundcard" >> "${BOOT_CFG}" || echo -e "${RED}Failed to enable I2S mic overlay${NC}"
    echo -e "${GREEN}  ✓ inmp441 overlay enabled${NC}"
fi

# enable camera
if ! grep -q "start_x=1" "${BOOT_CFG}" 2>/dev/null; then
    echo "start_x=1" >> "${BOOT_CFG}" || echo -e "${RED}Failed to enable camera${NC}"
    echo "gpu_mem=128" >> "${BOOT_CFG}"
    echo -e "${GREEN}  ✓ camera enabled${NC}"
fi

# ---- install system dependencies ----
echo ""
echo -e "${YELLOW}[2/7] installing system dependencies...${NC}"
apt-get update -qq || { echo -e "${RED}apt-get update failed${NC}"; exit 1; }

# FIX: libatlas-base-dev dropped in Bookworm (libopenblas-dev covers it)
#      pigpio dropped in Bookworm — built from source below
apt-get install -y -qq \
    python3-dev python3-pip python3-venv \
    libportaudio2 libportaudiocpp0 portaudio19-dev \
    libopenblas-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libcamera-dev libcap-dev python3-picamera2 \
    i2c-tools \
    wget unzip curl git cmake build-essential || { echo -e "${RED}apt-get install failed${NC}"; exit 1; }

echo -e "${GREEN}  ✓ system packages installed${NC}"

# FIX: pigpio not in Bookworm repos — build from source
if ! command -v pigpiod &> /dev/null; then
    echo -e "${YELLOW}  building pigpio from source (~2 min)...${NC}"
    cd /tmp
    wget -q https://github.com/joan2937/pigpio/archive/master.zip -O pigpio.zip \
        || { echo -e "${RED}pigpio download failed${NC}"; exit 1; }
    unzip -q pigpio.zip
    cd pigpio-master
    make -j$(nproc) > /dev/null || { echo -e "${RED}make failed${NC}"; exit 1; }
    make install > /dev/null || { echo -e "${RED}make install failed${NC}"; exit 1; }
    cd "${PROJECT_DIR}"
    rm -rf /tmp/pigpio.zip /tmp/pigpio-master
    echo -e "${GREEN}  ✓ pigpio installed from source${NC}"
else
    echo -e "${GREEN}  ✓ pigpio already installed${NC}"
fi

# source build doesn't ship a systemd unit — create it if missing
if [ ! -f /etc/systemd/system/pigpiod.service ] && [ ! -f /lib/systemd/system/pigpiod.service ]; then
    echo -e "${YELLOW}  creating pigpiod systemd unit...${NC}"
    cat > /etc/systemd/system/pigpiod.service << 'UNIT'
[Unit]
Description=Pigpio Daemon
After=network.target

[Service]
Type=forking
PIDFile=/var/run/pigpio.pid
ExecStart=/usr/local/bin/pigpiod -t 0
ExecStop=/bin/kill -SIGINT $MAINPID
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT
    systemctl daemon-reload
    echo -e "${GREEN}  ✓ pigpiod.service created${NC}"
fi

# ---- start pigpio daemon ----
echo ""
echo -e "${YELLOW}[3/7] starting pigpio daemon...${NC}"
systemctl enable pigpiod || true
systemctl start pigpiod || { echo -e "${RED}failed to start pigpiod${NC}"; exit 1; }
echo -e "${GREEN}  ✓ pigpiod enabled and started${NC}"

# ---- create virtual environment ----
echo ""
echo -e "${YELLOW}[4/7] setting up python environment...${NC}"
cd "${PROJECT_DIR}"

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}  python: ${PY_VER}${NC}"
# numpy 2.x and scikit-learn 1.5+ ship pre-built ARM64 wheels on PyPI for 3.13
# apt pre-install skipped: freezes on large packages and paths won't match
# a non-system Python venv anyway

if [ ! -d "venv" ]; then
    python3 -m venv --system-site-packages venv \
        || { echo -e "${RED}failed to create venv${NC}"; exit 1; }
    echo -e "${GREEN}  ✓ virtual environment created${NC}"
fi

source venv/bin/activate
pip install --upgrade pip wheel setuptools -q \
    || { echo -e "${RED}pip upgrade failed${NC}"; exit 1; }

echo -e "${YELLOW}  installing python packages...${NC}"
# --prefer-binary: always use pre-built wheels, never compile from source
# removed -q so failures are visible
if ! pip install --prefer-binary -r scripts/requirements_pi.txt; then
    echo -e "${YELLOW}  ⚠ requirements.txt failed, installing individually...${NC}"
    pip install --prefer-binary pigpio smbus2 \
        || echo -e "${RED}  failed pigpio/smbus2${NC}"
    pip install --prefer-binary numpy scikit-learn joblib \
        || echo -e "${RED}  failed numpy/ml${NC}"
    pip install --prefer-binary ai-edge-litert \
        || echo -e "${YELLOW}  ⚠ ai-edge-litert unavailable (no Python ${PY_VER} wheel) — tflite disabled${NC}"
    pip install --prefer-binary flask flask-socketio flask-cors Flask-HTTPAuth \
        || echo -e "${RED}  failed flask stack${NC}"
    pip install --prefer-binary vosk sounddevice librosa soundfile \
        || echo -e "${YELLOW}  ⚠ vosk may lack a Python ${PY_VER} wheel — speech features may be disabled${NC}"
    pip install --prefer-binary opencv-python-headless Pillow \
        || echo -e "${RED}  failed vision stack${NC}"
    pip install --prefer-binary pyyaml psutil requests \
        || echo -e "${RED}  failed utils stack${NC}"
fi

# Fix root ownership of venv created via sudo
REAL_USER=${SUDO_USER:-$USER}
echo -e "${YELLOW}  fixing venv permissions for $REAL_USER...${NC}"
chown -R $REAL_USER:$REAL_USER venv/

echo -e "${GREEN}  ✓ python packages installed${NC}"

# ---- download ai models ----
echo ""
echo -e "${YELLOW}[5/7] downloading ai models...${NC}"
mkdir -p "${MODELS_DIR}"

# vosk small english model
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

# mobilenetv2 tflite model
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

# install ollama and pull qwen2.5:0.5b
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}  installing ollama daemon...${NC}"
    curl -fsSL https://ollama.com/install.sh | sh || { echo -e "${RED}ollama install failed${NC}"; exit 1; }
    echo -e "${GREEN}  ✓ ollama installed${NC}"
else
    echo -e "${GREEN}  ✓ ollama already installed${NC}"
fi

echo -e "${YELLOW}  waiting for ollama daemon to initialize...${NC}"
sleep 5
systemctl start ollama 2>/dev/null || true
echo -e "${YELLOW}  pulling qwen2.5:0.5b model via ollama (~350mb)...${NC}"
ollama pull qwen2.5:0.5b || echo -e "${RED}ollama pull failed${NC}"
echo -e "${GREEN}  ✓ qwen2.5:0.5b model ready${NC}"

cd "${PROJECT_DIR}"

# ---- create data directories ----
echo ""
echo -e "${YELLOW}[6/7] creating data directories...${NC}"
mkdir -p "${DATA_DIR}/logs"
mkdir -p "${DATA_DIR}/objects"
mkdir -p "${DATA_DIR}/voice"
echo -e "${GREEN}  ✓ data directories created${NC}"

# ---- create systemd service ----
echo ""
echo -e "${YELLOW}[7/7] creating systemd service...${NC}"

REAL_USER=$(logname || echo $SUDO_USER)
if [ -z "$REAL_USER" ]; then
    REAL_USER="pi"
fi

cat > /etc/systemd/system/simba.service << EOF
[Unit]
Description=simba bionic arm robot
After=network.target pigpiod.service
Wants=pigpiod.service

[Service]
Type=simple
User=${REAL_USER}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PROJECT_DIR}/venv/bin/python -m simba.main
Restart=on-failure
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload || true
systemctl enable simba || true
echo -e "${GREEN}  ✓ systemd service created (simba.service)${NC}"

# ---- done ----
echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${GREEN}  ✅ installation complete!${NC}"
echo ""
echo -e "${CYAN}  start simba:  sudo systemctl start simba${NC}"
echo -e "${CYAN}  stop simba:   sudo systemctl stop simba${NC}"
echo -e "${CYAN}  view logs:    journalctl -u simba -f${NC}"
echo -e "${CYAN}  dashboard:    http://$(hostname -I | awk '{print $1}'):8080${NC}"
echo ""
PI_IP=$(hostname -I | awk '{print $1}')
echo -e "${GREEN}  🌐 Pi IP address: ${PI_IP}${NC}"
echo ""
echo -e "${YELLOW}  ⚠ reboot recommended to apply hardware changes:${NC}"
echo -e "${YELLOW}     sudo reboot${NC}"
echo -e "${CYAN}============================================${NC}"
