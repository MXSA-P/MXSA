<div align="center">
  <h1>🦁 Project MXSA — Simba Robot</h1>
  <pre>
███████╗██╗███╗   ███╗██████╗  █████╗       ███╗   ███╗██╗  ██╗███████╗ █████╗
██╔════╝██║████╗ ████║██╔══██╗██╔══██╗      ████╗ ████║╚██╗██╔╝██╔════╝██╔══██╗
███████╗██║██╔████╔██║██████╔╝███████║█████╗██╔████╔██║ ╚███╔╝ ███████╗███████║
╚════██║██║██║╚██╔╝██║██╔══██╗██╔══██║╚════╝██║╚██╔╝██║ ██╔██╗ ╚════██║██╔══██║
███████║██║██║ ╚═╝ ██║██████╔╝██║  ██║      ██║ ╚═╝ ██║██╔╝ ██╗███████║██║  ██║
╚══════╝╚═╝╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝      ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
  </pre>

  <p><b>Autonomous Edge AI Bionic Arm Robot</b></p>
  <p><i>Built from scratch — no ROS, no cloud, no training wheels.</i></p>

  <p>
    <img src="https://img.shields.io/badge/platform-Raspberry_Pi_4-c51a4a?style=for-the-badge&logo=raspberrypi&logoColor=white" alt="Platform">
    <img src="https://img.shields.io/badge/python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/AI-Edge_LLM-ff6f00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="AI">
    <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License">
  </p>

  <br>
  <img src="docs/MXSA_BP.png" alt="MXSA Blueprint" width="800">
</div>

---

## What is Simba?

Simba is a fully autonomous bionic arm robot that runs **entirely offline** on a Raspberry Pi 4. It combines edge AI inference, computer vision, voice recognition, emotional awareness, and a 4-DOF articulated arm with a 3-finger gripper — all without any cloud dependencies.

At its core, Simba uses **Qwen2.5-0.5B** (via Ollama) for real-time conversational intelligence, **MobileNetV2 + SVM** for custom object detection, **Vosk** for offline speech recognition, and **pigpio** for hardware-timed servo control. Everything is configurable, trainable, and deployable from a sleek web interface.

---

## ✨ Key Features

### 🧠 Autonomous Brain
- Real-time LLM inference (Qwen2.5-0.5B) for conversations, decisions, and emotional responses
- Finite state machine with 8 behavioral states (idle, roaming, tracking, fetching, playing, charging, resting, error)
- Persistent memory system that remembers objects, people, and past interactions
- Dynamic emotion engine with mood decay, excitement thresholds, and reactive animations

### 👁️ Edge Computer Vision
- Hybrid detection pipeline: MobileNetV2 feature extraction → LinearSVC classifier
- Native support for the **5MP CSI Camera (OV5647)** at full `2592×1944` resolution
- Optional YOLOv8n acceleration for enhanced detection
- Environmental scanning with 5-angle servo sweep (0°, 45°, 90°, 135°, 180°)

### 🎙️ Offline Voice Recognition
- Vosk-based speech recognition — no internet required
- 50+ voice commands across motion, gestures, personality, and system control
- Speaker verification for owner recognition
- INMP441 I2S MEMS microphone with automatic channel/sample-rate fallback

### 🦾 Precision Motion Control
- **Arm:** 4-DOF articulated arm (rotation, elbow×2, wrist) with inverse kinematics
- **Hand:** 3-finger triangle gripper with adaptive grip based on object width
- **Chassis:** 2WD differential drive via L298N motor driver with active braking
- **IMU:** MPU6050 for orientation tracking and tilt detection
- Per-servo calibration: inversion, angle limits, trim offsets, speed multipliers

### 🎓 Custom Training Studio
- Dark glassmorphism web UI for training custom vision and voice profiles
- Capture training data directly from your webcam
- One-click model training with real-time progress tracking
- Deploy trained models to the Pi over SSH

### 🕹️ Hardware Command Center
- Web-based remote control dashboard with live camera feed
- WASD chassis controls with active motor braking
- Individual servo sliders for all 7 servos (4 arm + 3 fingers)
- Real-time telemetry: emotions, IMU data, motor speeds, grip state

---

## 📐 Architecture

```
┌───────────────────────────────────────────────────┐
│                    SIMBA BRAIN                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ State    │  │ Emotion  │  │ Memory           │ │
│  │ Machine  │◄─┤ Engine   │  │ (JSON persistent)│ │
│  └────┬─────┘  └────┬─────┘  └──────────────────┘ │
│       │             │                             │
│  ┌────▼─────────────▼───────────────────────────┐ │
│  │              Brain Controller                │ │
│  │  (Qwen2.5-0.5B via Ollama + decision logic)  │ │
│  └──────┬───────────┬───────────┬───────────────┘ │   
│         │           │           │                 │   
│  ┌──────▼───┐ ┌─────▼────┐ ┌───▼──────┐           │   
│  │ Vision   │ │ Voice    │ │ Motion   │           │   
│  │ Pipeline │ │ Listener │ │ Control  │           │   
│  │          │ │          │ │          │           │   
│  │ Camera   │ │ Vosk STT │ │  Arm(4)  │           │   
│  │ Detector │ │ Speaker  │ │  Hand(3) │           │   
│  │ Scanner  │ │ Verify   │ │  Chassis │           │   
│  │ YOLO     │ │ Commands │ │  IMU     │           │   
│  └──────────┘ └──────────┘ └──────────┘           │   
│                                                   │   
│  ┌────────────────────────────────────────────┐   │   
│  │            Web Dashboard (Flask+SocketIO)  │   │   
│  │  Dashboard │ Hardware Control │ Diagnostics│   │   
│  └────────────────────────────────────────────┘   │   
└───────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│              TRAINER (Windows PC)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Vision   │  │ Voice    │  │ Model            │   │
│  │ Trainer  │  │ Trainer  │  │ Exporter (SSH)   │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
│  ┌────────────────────────────────────────────────┐ │
│  │         Trainer Web UI (Flask)                 │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## ⚙️ Hardware Requirements

| Component | Part | Notes |
| :--- | :--- | :--- |
| **SBC** | Raspberry Pi 4 Model B | 4GB+ RAM recommended |
| **Camera** | 5MP CSI Module V1 (OV5647) | Set GPU memory ≥128MB |
| **Motor Driver** | L298N (6-pin) | ENA/ENB for PWM speed control |
| **Servos** | 7× SG90 (or equivalent) | 4 arm + 3 fingers |
| **IMU** | MPU6050 | I2C on GPIO 2/3 |
| **Microphone** | INMP441 I2S MEMS | Wired to GPIO 18/19/20 |
| **Motors** | 2× TT DC Motors | With wheels and ball caster |
| **Power** | 2×18650 + 5V Powerbank | Batteries for motors, powerbank for Pi |

> 📌 **Wiring:** See the full [Hardware Connections Guide](hardware/connections.md) for GPIO pinout, logic tables, and power connections.

---

## 🚀 Installation

### Raspberry Pi (Robot)

```bash
# 1. Clone
git clone https://github.com/MXSA-P/MXSA.git
cd MXSA

# 2. Install everything (apt packages, venv, I2C/I2S config, GPU memory check)
sudo bash scripts/install_pi.sh

# 3. Download AI models (Vosk, MobileNetV2)
bash scripts/download_models.sh

# 4. Start Simba
./scripts/start_simba.sh
```

### Windows PC (Trainer)

```batch
:: 1. Clone (if not already done)
git clone https://github.com/MXSA-P/MXSA.git
cd MXSA

:: 2. Install dependencies
install_trainer.bat

:: 3. Start the trainer
start_trainer.bat
```

---

## 🖥️ Web Interfaces

Both web interfaces are protected with HTTP Basic Authentication.

### 🔑 Credentials

| Interface | URL | User | Pass | Env Overrides |
| :--- | :--- | :---: | :---: | :--- |
| **Simba Dashboard** | `http://<PI_IP>:8080/` | `mxsa` | `mx` | `SIMBA_WEB_USER` / `SIMBA_WEB_PASS` |
| **Trainer UI** | `http://localhost:5000/` | `mxsa` | `mx` | `TRAINER_USER` / `TRAINER_PASS` |

> ⚠️ **Security:** Always override default passwords via environment variables on shared networks.

### Simba Dashboard (`http://<PI_IP>:8080/`)
- **Dashboard:** Live telemetry — emotions, detected objects, system health, conversation log
- **Hardware Control:** Remote control with WASD, servo sliders, camera feed, active braking
- **Diagnostics:** System diagnostics, sensor checks, log viewer

### Trainer UI (`http://localhost:5000/`)
- **Vision Training:** Capture images via webcam, train custom object classifiers, manage training data
- **Voice Training:** Record voice samples, train speaker verification models
- **Deployment:** One-click model export to the Pi over SSH

---

## 🔧 Calibration

Simba supports per-servo calibration and per-motor speed balancing via `config/simba_config.yaml`:

```yaml
calibration:
  servos:
    arm_rotation:
      inverted: false     # flip 0↔180 for backwards-mounted servos
      min_angle: 0        # mechanical minimum
      max_angle: 180      # mechanical maximum
      trim: 0             # fine offset in degrees (-20 to +20)
    finger_3:
      inverted: false
      min_angle: 0
      max_angle: 50       # this servo only goes 50°
      trim: 0

  motors:
    motor_a_multiplier: 1.0   # left tire speed multiplier (0.0–2.0)
    motor_b_multiplier: 1.0   # right tire speed multiplier (0.0–2.0)
```

**Common calibration tasks:**
- **Servo mounted backwards?** → Set `inverted: true`
- **Servo has limited range?** → Adjust `min_angle` / `max_angle`
- **Servo doesn't center properly?** → Add a `trim` offset
- **One tire faster than the other?** → Lower that motor's multiplier

---

## 🗣️ Voice Commands

Simba understands **50+ voice commands** across 5 categories. Full list: [Voice Commands Reference](docs/voice_commands.md)

**Quick examples:**
```
"go forward"           → drives forward
"fetch the ball"       → locates, grabs, and returns the ball
"scan"                 → sweeps the environment for objects
"who are you"          → Simba introduces himself
"good boy"             → increases happiness
"go charge"            → retraces path to charging station
```

---

## 📁 Project Structure

```
MXSA/
├── config/
│   └── simba_config.yaml          # All robot configuration + calibration
├── data/
│   └── path_log.json              # Movement breadcrumb trail
├── docs/
│   ├── MXSA_BP.png                # Blueprint diagram
│   ├── simba_manual.html          # Full manual
│   └── voice_commands.md          # Voice command reference
├── hardware/
│   └── connections.md             # GPIO wiring guide
├── models/                        # AI model files (downloaded)
│   ├── vosk-model-small-en-us-0.15/
│   ├── mobilenetv2_feature_extractor.tflite
│   ├── object_classifier.joblib
│   └── object_labels.json
├── scripts/
│   ├── install_pi.sh              # Pi installer (apt + venv + I2S)
│   ├── install_trainer.bat        # Windows trainer installer
│   ├── start_simba.sh             # Start the robot
│   ├── start_trainer.bat          # Start the trainer
│   ├── download_models.sh         # Download AI models
│   ├── requirements_pi.txt        # Pi Python dependencies
│   └── requirements_trainer.txt   # Trainer Python dependencies
├── simba/                         # Robot runtime (runs on Pi)
│   ├── core/
│   │   ├── brain.py               # Main brain controller (70KB)
│   │   ├── emotions.py            # Emotion engine with mood decay
│   │   ├── memory.py              # Persistent JSON memory
│   │   ├── path_recorder.py       # Movement breadcrumb recorder
│   │   └── state_machine.py       # 8-state behavioral FSM
│   ├── motion/
│   │   ├── arm.py                 # 4-DOF arm with inverse kinematics
│   │   ├── chassis.py             # 2WD differential drive
│   │   ├── hand.py                # 3-finger triangle gripper
│   │   └── imu.py                 # MPU6050 orientation tracker
│   ├── vision/
│   │   ├── camera.py              # CSI camera interface
│   │   ├── detector.py            # MobileNetV2+SVM object detector
│   │   ├── hybrid_detector.py     # Multi-backend detector
│   │   ├── scanner.py             # Environmental sweep scanner
│   │   └── yolo_detector.py       # Optional YOLOv8n backend
│   ├── voice/
│   │   ├── command_parser.py      # NLU command extraction
│   │   ├── listener.py            # Vosk STT + I2S mic
│   │   └── speaker_verify.py      # Speaker identification
│   ├── web/
│   │   ├── server.py              # Flask+SocketIO dashboard
│   │   ├── static/                # CSS + JS assets
│   │   └── templates/             # HTML templates
│   ├── utils/
│   │   └── logger.py              # Structured JSON logging
│   └── main.py                    # Entry point
├── trainer/                       # Training studio (runs on Windows PC)
│   ├── app.py                     # Flask trainer web app
│   ├── export_model.py            # SSH model deployer
│   ├── train_behavior.py          # Behavior model trainer
│   ├── train_vision.py            # Vision model trainer
│   ├── static/                    # CSS + JS assets
│   └── templates/                 # HTML templates
├── tests/                         # Unit tests
├── install_trainer.bat            # Quick-start installer
├── start_trainer.bat              # Quick-start launcher
├── LICENSE                        # MIT License
└── README.md                      # You are here
```

---

## 🛡️ Reliability & Safety

| Feature | Details |
| :--- | :--- |
| **Memory Management** | Forced `gc.collect()` after ML operations to prevent Pi OOM |
| **Log Rotation** | `simba_system.jsonl` hard-capped at 5MB with 3 backups |
| **Hardware Fallbacks** | Disconnected sensors fall back to mock drivers automatically |
| **Servo Stagger** | Servos initialize one-at-a-time with 150ms delays to prevent power brownouts |
| **Per-Servo Calibration** | Inversion, angle limits, and trim offsets per servo |
| **Auto-Return** | Movement breadcrumb trail with path reversal for charging |
| **Auth Protected** | All web interfaces require HTTP Basic Authentication |

---

## 🔍 Troubleshooting

| Problem | Solution |
| :--- | :--- |
| `list index out of range` on camera init | Enable camera in `raspi-config` → Interface Options. Check ribbon cable. |
| `Error querying device -1` for audio | No default mic set. Listener auto-falls back to device 0. |
| Servos jitter on startup | Check power supply. Simba staggers servo init to reduce current spikes. |
| One tire faster than the other | Adjust `calibration.motors.motor_X_multiplier` in config. |
| Login loops on web interface | Ensure `Flask-HTTPAuth` is installed. Check browser isn't blocking Basic Auth. |
| Can't connect to dashboard SocketIO | WebSocket connections don't need separate auth — page-level auth suffices. |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/awesome-feature`)
3. Commit your changes (`git commit -m 'Add awesome feature'`)
4. Push to the branch (`git push origin feature/awesome-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub><b>_max_cyan_</b> — project_mxsa</sub>
</div>
