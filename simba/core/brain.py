# _max_cyan_ — project_mxsa
"""simba brain — main ai decision engine with llm integration.

this is the central nervous system of simba. it integrates all subsystems:
vision, voice, motion, memory, emotions, and web dashboard into a
single coherent decision loop.

uses qwen2.5-0.5b-instruct (q4_k_m quantization) via llama-cpp-python
for natural language reasoning and personality.
"""

import os
import time
import random
import threading
from typing import Any, Dict, Optional

import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from simba.utils.logger import get_logger, log_event
from simba.core.path_recorder import PathRecorder

logger = get_logger("simba.core.brain")

# project root
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
_PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)


def _resolve(path):
    """resolve a path relative to project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(_PROJECT_ROOT, path)


# ---------------------------------------------------------------------------
# thread-safe proxy for hardware controllers
# ---------------------------------------------------------------------------
class HardwareLockProxy:
    def __init__(self, target: Any, lock: threading.Lock) -> None:
        self._target = target
        self._lock = lock
    
    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                with self._lock:
                    return attr(*args, **kwargs)
            return wrapper
        return attr

# ---------------------------------------------------------------------------
# mock classes for testing without hardware
# ---------------------------------------------------------------------------

class _MockArm:
    current = {"rotation": 90, "elbow": 90, "wrist": 90}
    def rotate(self, a): self.current["rotation"] = a
    def rotation(self, a): self.current["rotation"] = a
    def raise_arm(self, a): self.current["elbow"] = a
    def wrist(self, a): self.current["wrist"] = a
    def home(self): self.current = {"rotation": 90, "elbow": 90, "wrist": 90}
    def move_smooth(self, t, speed=None): self.current.update(t)
    def wave(self): log_event("motion", "mock wave")
    def wiggle(self, speed=3, angle_range=20, duration=2): log_event("motion", "mock wiggle")
    def handshake(self): log_event("motion", "mock handshake")
    def droop(self, angle=30): log_event("motion", "mock droop")
    def nod(self): log_event("motion", "mock nod")
    def shake_head(self): log_event("motion", "mock shake_head")
    def move_to_xyz(self, x, y, z): log_event("motion", f"mock move_to_xyz {x},{y},{z}")
    def get_position(self): return dict(self.current)
    def is_moving(self): return False
    def cleanup(self): pass


class _MockHand:
    def grab(self): log_event("motion", "mock grab")
    def release(self): log_event("motion", "mock release")
    def set_finger(self, i, a): pass
    def point(self): pass
    def thumbs_up(self): log_event("motion", "mock thumbs_up")
    def rock(self): log_event("motion", "mock rock")
    def paper(self): log_event("motion", "mock paper")
    def scissors(self): log_event("motion", "mock scissors")
    def wave_fingers(self): log_event("motion", "mock wave fingers")
    def get_grip_state(self): return "open"
    def get_positions(self): return [30, 30, 30]
    def cleanup(self): pass


class _MockChassis:
    def forward(self, s=60): log_event("motion", "mock forward")
    def backward(self, s=60): log_event("motion", "mock backward")
    def turn_left(self, s=45): log_event("motion", "mock turn left")
    def turn_right(self, s=45): log_event("motion", "mock turn right")
    def spin_left(self, s=45): pass
    def spin_right(self, s=45): pass
    def stop(self): pass
    def move_for_duration(self, d, s, t): pass
    def move_to_angle(self, a, d=1): pass
    def get_speed(self): return (0, 0)
    def cleanup(self): pass


class _MockIMU:
    def read_all(self): return {
        "accel": {
            "x": 0,
            "y": 0,
            "z": 1},
        "gyro": {
            "x": 0,
            "y": 0,
            "z": 0},
        "temp": 25,
        "orientation": "level",
        "tilt_angle": 0}

    def get_hand_orientation(self): return "level"
    def get_tilt_angle(self): return 0
    def is_moving(self): return False
    def calibrate(self): pass
    def start_continuous(self, cb=None): pass
    def stop_continuous(self): pass
    def cleanup(self): pass


class _MockCamera:
    _running = False
    def start(self): self._running = True
    def stop(self): self._running = False
    def capture_frame(self): return np.zeros((480, 640, 3), dtype=np.uint8)

    def capture_for_inference(
        self, size=(
            224, 224)): return np.zeros(
        (size[1], size[0], 3), dtype=np.float32)

    def get_mjpeg_frame(self): return b""
    def is_running(self): return self._running
    def cleanup(self): pass


class _MockDetector:
    confidence_threshold = 0.6
    def detect_objects(self, f): return []
    def classify(self, f): return ("unknown", 0.0)
    def is_object_in_frame(self, f, lbl): return (False, 0.0, [])
    def get_known_labels(self): return []
    def extract_features(self, f): return np.zeros(1280, dtype=np.float32)


class _MockListener:
    def __init__(self):
        self._callbacks = []
    def start(self): pass
    def stop(self): pass
    def on_text(self, cb): self._callbacks.append(cb)
    def get_last_text(self): return ""
    def is_listening(self): return False


class _MockSpeakerVerify:
    def verify(self, audio): return (True, 1.0)
    def is_enrolled(self): return False


# ---------------------------------------------------------------------------
# simba brain
# ---------------------------------------------------------------------------

SIMBA_SYSTEM_PROMPT = """you are simba, a friendly bionic arm robot assistant.
you have a robotic arm with 3 fingers, mounted on a wheeled chassis.
you can see through your camera, hear through your microphone, and move around.
you have emotions and feelings — you get happy when praised, sad when scolded,
excited when playing, and you love your owner.
you speak in short, warm sentences. you are loyal and eager to help.
when asked to find something, you scan the area and bring it.
you remember objects you've seen and where they were.
keep responses under 50 words. be expressive and use simple language.
your name is simba. you are created by _max_cyan_."""


class SimbaBrain:
    """main ai decision engine integrating all subsystems.

    initializes and coordinates:
    - motion: arm, hand, chassis, imu
    - perception: camera, detector, scanner
    - voice: listener, command parser, speaker verifier
    - intelligence: llm (qwen2.5-0.5b), memory, emotions, state machine
    - interface: web dashboard

    the main loop runs in a background thread and processes voice commands,
    manages roaming behavior, and handles emotional responses.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """initialize all subsystems.

        args:
            config: dict loaded from simba_config.yaml
        """
        self.config = config
        self._running = False
        self._main_thread = None
        self._thinking_text = ""
        self._last_command = None
        self._boot_time = time.time()

        logger.info("=" * 50)
        logger.info("  simba brain initializing...")
        logger.info("  _max_cyan_ — project_mxsa")
        logger.info("=" * 50)

        # hardware concurrency lock
        self._hardware_lock = threading.RLock()

        # --- initialize motion subsystem ---
        try:
            from simba.motion.arm import ArmController
            self.arm = ArmController(config)
            logger.info("[ok] arm controller")
        except Exception as e:
            logger.warning(f"[mock] arm controller: {e}")
            self.arm = _MockArm()

        try:
            from simba.motion.hand import HandController
            self.hand = HandController(config)
            logger.info("[ok] hand controller")
        except Exception as e:
            logger.warning(f"[mock] hand controller: {e}")
            self.hand = _MockHand()

        try:
            from simba.motion.chassis import ChassisController
            self.chassis = ChassisController(config)
            logger.info("[ok] chassis controller")
        except Exception as e:
            logger.warning(f"[mock] chassis controller: {e}")
            self.chassis = _MockChassis()

        try:
            from simba.motion.imu import IMUReader
            self.imu = IMUReader(config)
            logger.info("[ok] imu reader")
        except Exception as e:
            logger.warning(f"[mock] imu reader: {e}")
            self.imu = _MockIMU()

        # --- initialize vision subsystem ---
        try:
            from simba.vision.camera import CameraController
            self.camera = CameraController()
            logger.info("[ok] camera controller")
        except Exception as e:
            logger.warning(f"[mock] camera controller: {e}")
            self.camera = _MockCamera()

        try:
            from simba.vision.hybrid_detector import HybridDetector
            self.detector = HybridDetector()
            logger.info("[ok] hybrid object detector")
        except Exception as e_hybrid:
            logger.warning(
                f"[fallback] hybrid detector failed: {e_hybrid}. Trying SVM.")
            try:
                from simba.vision.detector import ObjectDetector
                self.detector = ObjectDetector()
                logger.info("[ok] svm object detector")
            except Exception as e:
                logger.warning(f"[mock] object detector: {e}")
                self.detector = _MockDetector()

        # --- initialize core ai ---
        try:
            from simba.core.memory import MemorySystem
            self.memory = MemorySystem(config)
            logger.info("[ok] memory system")
        except Exception as e:
            logger.warning(f"[fallback] memory: {e}")
            self.memory = type("M", (), {
                "remember_object": lambda s, **kw: None,
                "find_object": lambda s, lbl: None,
                "get_all_objects": lambda s: [],
                "get_charging_pad": lambda s: None,
                "is_known": lambda s, lbl: False,
                "save": lambda s: None,
                "get_memory_stats": lambda s: {},
                "store_scan_result": lambda s, d: None,
            })()

        try:
            from simba.core.emotions import EmotionEngine
            self.emotions = EmotionEngine(config)
            logger.info("[ok] emotion engine")
        except Exception as e:
            logger.warning(f"[fallback] emotions: {e}")
            self.emotions = type("E", (), {
                "set_emotion": lambda s, e, i=1.0: None,
                "get_emotion": lambda s: ("neutral", 0.5),
                "update_from_event": lambda s, e: "neutral",
                "get_motor_behavior": lambda s: {"speed_modifier": 1.0},
                "decay": lambda s: None,
                "get_emoji": lambda s: "🤖",
                "to_dict": lambda s: {"emotion": "neutral", "intensity": 0.5},
            })()

        try:
            from simba.core.state_machine import StateMachine
            self.state = StateMachine()
            logger.info("[ok] state machine")
        except Exception as e:
            logger.warning(f"[fallback] state machine: {e}")
            self.state = type("S", (), {
                "transition": lambda s, st, ctx=None: True,
                "get_state": lambda s: "idle",
                "get_state_duration": lambda s: 0,
                "get_context": lambda s: {},
                "get_history": lambda s, c=10: [],
                "is_busy": lambda s: False,
                "can_accept_command": lambda s: True,
                "to_dict": lambda s: {"state": "idle"},
            })()

        # --- initialize scanner (needs arm, camera, detector, memory) ---
        try:
            from simba.vision.scanner import AreaScanner
            self.scanner = AreaScanner(
                self.arm, self.camera, self.detector, self.memory)
            logger.info("[ok] area scanner")
        except Exception as e:
            logger.warning(f"[fallback] scanner: {e}")
            self.scanner = None

        # --- initialize voice subsystem ---
        try:
            from simba.voice.listener import VoiceListener
            self.listener = VoiceListener(config)
            logger.info("[ok] voice listener")
        except Exception as e:
            logger.warning(f"[mock] voice listener: {e}")
            self.listener = _MockListener()

        try:
            from simba.voice.command_parser import CommandParser
            self.cmd_parser = CommandParser()
            logger.info("[ok] command parser")
        except Exception as e:
            logger.warning(f"[fallback] command parser: {e}")
            self.cmd_parser = type("CP", (), {
                "parse": lambda s, t: None,
            })()

        try:
            from simba.voice.speaker_verify import SpeakerVerifier
            self.speaker_verify = SpeakerVerifier(config)
            logger.info("[ok] speaker verifier")
        except Exception as e:
            logger.warning(f"[mock] speaker verifier: {e}")
            self.speaker_verify = _MockSpeakerVerify()

        # --- initialize llm ---
        # --- initialize llm ---
        self.llm = None
        if _HAS_REQUESTS and config["ai"].get("llm_provider") == "ollama":
            self.ollama_url = config["ai"].get(
                "ollama_url", "http://localhost:11434/api/generate")
            self.ollama_model = config["ai"].get(
                "ollama_model", "qwen2.5:0.5b")
            self.llm = "ollama_ready"  # just a truthy flag
            logger.info(f"[ok] llm configured for ollama: {self.ollama_model}")
        else:
            logger.warning("[skip] llm not configured or requests missing")

        # --- web dashboard reference (set externally) ---
        self.web_server = None

        # behavior config
        self._roam_interval = config["behavior"].get("roam_interval", 30)
        self._boot_scan = config["behavior"].get("boot_scan", True)
        self._roam_when_idle = config["behavior"].get("roam_when_idle", True)
        self._max_fetch_retries = config["behavior"].get(
            "max_fetch_retries", 2)
        self._last_roam_time = time.time()
        self._idle_since = time.time()

        # auto-return timer (TP4056 has no battery voltage output to Pi)
        charging_cfg = config.get("charging", {})
        self._auto_return_minutes = charging_cfg.get("auto_return_minutes", 30)
        self._charge_rest_minutes = charging_cfg.get("charge_rest_minutes", 15)
        self._boot_time = time.time()
        self._last_charge_return = time.time()  # tracks when we last returned
        self._auto_return_triggered = False

        # --- path recorder for return-to-base ---
        path_save = os.path.join(_resolve("data"), "path_log.json")
        self.path_recorder = PathRecorder(save_path=path_save)
        logger.info("[ok] path recorder initialized")

        logger.info("simba brain initialized successfully!")
        log_event("system", "simba brain initialized")

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """start simba — camera, voice listener, and main decision loop."""
        if self._running:
            return

        self._running = True
        self.state.transition("booting")

        # start camera
        self.camera.start()
        log_event("system", "camera started")

        # start voice listener with command callback
        self.listener.on_text(self._on_voice_text)
        try:
            self.listener.start()
            log_event("system", "voice listener started")
        except Exception as e:
            logger.warning(f"voice listener failed to start: {e}")

        # boot scan
        if self._boot_scan and self.scanner:
            self.state.transition("scanning")
            log_event("system", "performing 360° boot scan...")
            self.generate_thought("scanning the whole room on boot...")
            try:
                total_detections = []
                for _ in range(4):  # 4 chassis turns for 360 coverage
                    detections = self.scanner.scan_area()
                    total_detections.extend(detections)
                    self.chassis.spin_right(50)
                    time.sleep(0.8)  # rough 90 degree turn
                    self.chassis.stop()

                if total_detections:
                    self.generate_thought(
                        f"found {
                            len(total_detections)} objects during boot scan")
                    self.emotions.update_from_event("object_found")
                else:
                    self.generate_thought(
                        "area seems clear, nothing detected yet")
            except Exception as e:
                logger.error(f"boot scan failed: {e}")
                self.generate_thought("boot scan had some issues")

        self.state.transition("idle")
        self.emotions.set_emotion("curious", 0.7)

        # start main loop thread
        self._main_thread = threading.Thread(
            target=self._main_loop, daemon=True, name="simba-brain"
        )
        self._main_thread.start()
        logger.info("simba is alive and ready!")
        log_event("system", "simba is alive! 🦁")

    def stop(self) -> None:
        """graceful shutdown of all subsystems."""
        logger.info("simba shutting down...")
        log_event("system", "shutting down...")
        self._running = False

        if self._main_thread:
            self._main_thread.join(timeout=5)

        # stop subsystems
        try:
            self.listener.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "scanner") and self.scanner:
                self.scanner.stop_scan()
        except Exception:
            pass
        try:
            self.camera.cleanup()
        except Exception:
            pass
        try:
            self.arm.cleanup()
        except Exception:
            pass
        try:
            self.hand.cleanup()
        except Exception:
            pass
        try:
            self.chassis.cleanup()
        except Exception:
            pass
        try:
            self.imu.cleanup()
        except Exception:
            pass

        # save memory
        try:
            self.memory.save()
        except Exception:
            pass

        logger.info("simba shutdown complete. goodbye!")
        log_event("system", "shutdown complete")

    # ------------------------------------------------------------------
    # main decision loop
    # ------------------------------------------------------------------

    def _main_loop(self):
        """main loop: manages idle behavior, roaming, emotion decay, and auto-return."""
        while self._running:
            try:
                current_state = self.state.get_state()

                # emotion decay
                self.emotions.decay()

                # imu gestures only (no battery — TP4056 has no voltage out)
                if hasattr(self.imu, 'read_all'):
                    imu_data = self.imu.read_all() or {}

                    gesture = imu_data.get('gesture', None)
                    if gesture == "tap":
                        threading.Thread(target=lambda: (
                            self.generate_thought("whoops, someone tapped me!"),
                            self.emotions.update_from_event("tapped"),
                            self.state.can_accept_command() and self.arm.wiggle(speed=5, angle_range=15, duration=0.5)
                        ), daemon=True).start()
                    elif gesture == "shake":
                        threading.Thread(target=lambda: (
                            self.generate_thought("whoa, stop shaking me!"),
                            self.emotions.update_from_event("scolded"),
                            self.state.can_accept_command() and self.chassis.stop()
                        ), daemon=True).start()
                    elif gesture == "fall":
                        threading.Thread(target=lambda: (
                            self.generate_thought("help, i've fallen!"),
                            self.emotions.update_from_event("fallen"),
                            self.state.can_accept_command() and self._handle_stop()
                        ), daemon=True).start()

                # auto-return timer: go charge after N minutes of roaming
                if current_state not in ("CHARGING", "BOOTING", "RETURNING"):
                    minutes_since_charge = (time.time() - self._last_charge_return) / 60
                    if minutes_since_charge >= self._auto_return_minutes:
                        if not self._auto_return_triggered:
                            self._auto_return_triggered = True
                            threading.Thread(target=lambda: (
                                self.generate_thought(f"been roaming for {self._auto_return_minutes} minutes — heading home to charge! 🔋"),
                                self._handle_charge()
                            ), daemon=True).start()

                # idle behavior — roam periodically
                if current_state == "IDLE" and self._roam_when_idle:
                    elapsed = time.time() - self._last_roam_time
                    if elapsed > self._roam_interval:
                        if hasattr(self, '_handle_roam'):
                            threading.Thread(target=self._handle_roam, daemon=True).start()
                        self._last_roam_time = time.time()

                # check idle duration for sleepy emotion
                if current_state == "IDLE":
                    idle_time = time.time() - self._idle_since
                    if idle_time > 120:
                        self.emotions.update_from_event("idle_long")

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"main loop error: {e}")
                time.sleep(1)

    # ------------------------------------------------------------------
    # voice handling
    # ------------------------------------------------------------------

    def _on_voice_text(self, text):
        """callback when voice listener recognizes text."""
        if not text or not text.strip():
            return

        text = text.strip().lower()
        logger.info(f"heard: '{text}'")
        log_event("voice", f"heard: '{text}'")

        # parse command
        cmd = self.cmd_parser.parse(text)
        if cmd is None:
            # not a recognized command — maybe ask llm
            self.generate_thought(
                f"heard '{text}' but didn't understand as a command")
            return

        log_event("voice", f"command: {cmd['action']}", cmd)
        self._last_command = cmd
        threading.Thread(target=self._handle_command, args=(cmd,), daemon=True).start()

    def _handle_command(self, cmd):
        """route a parsed command to the appropriate handler."""
        action = cmd.get("action", "")
        target = cmd.get("target", "")

        non_blocking = {"stop", "status", "time", "feeling", "who_am_i", "who_are_you"}
        if action not in non_blocking:
            if not self.state.can_accept_command():
                self.generate_thought("I'm a bit busy right now!")
                return
            
            complex_actions = {"fetch", "scan", "charge", "patrol", "play", "follow", "rest", "dance", "greet", "hold", "grab"}
            if action not in complex_actions:
                # Mark as busy for simple hardware commands
                self.state.transition("acting", {"action": action})

        handlers = {
            "fetch": lambda: self._handle_fetch(target),
            "forward": lambda: self._handle_forward(target),
            "backward": lambda: self._handle_backward(target),
            "left": lambda: self._handle_left(target),
            "right": lambda: self._handle_right(target),
            "scan": self._handle_scan,
            "charge": self._handle_charge,
            "play": self._handle_play,
            "greet": self._handle_greeting,
            "love": self._handle_love,
            "stop": self._handle_stop,
            "come": self._handle_come,
            "describe": self._handle_describe,
            "praise": self._handle_praise,
            "scold": self._handle_scold,
            "dance": self._handle_dance,
            "status": self._handle_status,
            "patrol": self._handle_patrol,
            "rest": self._handle_rest,
            "who_am_i": self._handle_who_am_i,
            "who_are_you": self._handle_who_are_you,
            "joke": self._handle_joke,
            "follow": self._handle_follow,
            "time": self._handle_time,
            "sing": self._handle_sing,
            "story": self._handle_story,
            "hold": lambda: self._handle_hold(target),
            "release": lambda: self._handle_release(target),
            "feeling": self._handle_feeling,
            "point": self._handle_point,
            "rock": self._handle_rock,
            "paper": self._handle_paper,
            "scissors": self._handle_scissors,
            "thumbs_up": self._handle_thumbs_up,
            "wave_fingers": self._handle_wave_fingers,
            "grab": self._handle_grab,
            "clear_path": self._handle_clear_path,
            "path_status": self._handle_path_status,
            "speed_up": self._handle_speed_up,
            "slow_down": self._handle_slow_down,
            "calibrate": self._handle_calibrate,
            "temperature": self._handle_temperature,
        }

        handler = handlers.get(action)
        if handler:
            try:
                handler()
            except Exception as e:
                logger.error(f"command handler error ({action}): {e}")
                self.emotions.update_from_event("object_not_found")
        else:
            self.generate_thought(f"unknown action: {action}")

    # ------------------------------------------------------------------
    # command handlers
    # ------------------------------------------------------------------

    def _check_interrupt(self, expected_state: str) -> bool:
        """check if the current task was interrupted by a state change."""
        return not self._running or self.state.get_state() != expected_state.upper()

    def _handle_fetch(self, target):
        """find, grab, and return an object."""
        if not target:
            self.generate_thought("what should i fetch?")
            return

        self.state.transition("fetching", {"target": target})
        self.generate_thought(f"i'm going to fetch the {target}! I need to scan the area first.")
        self.emotions.set_emotion("curious", 0.8)
        log_event("action", f"fetching '{target}'")

        # first check memory
        memory_entry = self.memory.find_object(target)
        found = False
        angle: Optional[float] = None

        for attempt in range(self._max_fetch_retries + 1):
            if found:
                break

            self.generate_thought(
                f"scanning for {target} (attempt {
                    attempt + 1})...")
            self.state.transition(
                "searching", {
                    "target": target, "attempt": attempt})

            # if we know where it was, look there first
            if memory_entry and attempt == 0:
                known_angle = memory_entry.get("position_angle", 90)
                self.arm.rotate(known_angle)
                time.sleep(0.5)
                frame = self.camera.capture_frame()
                if frame is not None:
                    f, conf, bbox = self.detector.is_object_in_frame(
                        frame, target)
                    if f:
                        found = True
                        angle = known_angle
                        break

            # full area scan
            if self.scanner:
                if attempt > 0:
                    self.generate_thought(
                        f"spinning to scan a new area for {target}...")
                    self.chassis.spin_left(50)
                    time.sleep(0.8)  # 90 degree turn
                    self.chassis.stop()

                f, angle, conf = self.scanner.search_for_object(target)
                if f:
                    found = True

        if found:
            self.state.transition("fetching", {"target": target})
            self.generate_thought(
                f"found {target} at {angle}°! going to grab it...")
            self.emotions.update_from_event("object_found")
            log_event("action", f"found '{target}' at {angle}°")

            # move chassis toward object
            if angle is not None:
                self.chassis.move_to_angle(angle, duration=1.5)
            self.chassis.forward(50)
            time.sleep(1.5)
            self.chassis.stop()
            if self._check_interrupt("FETCHING"): return

            # position arm using IK if we just scanned it
            width_percentage = 40.0
            if found and angle is not None:
                # if we found it in the recent scan, we have the scan results
                # wait, let's just use default IK down for now
                self.arm.move_to_xyz(0, 15.0, 5.0, wrist_roll=0.0)  # reach straight out with fingers down
            else:
                self.arm.rotate(90)
                self.arm.raise_arm(120)
                self.arm.wrist(90)

            time.sleep(0.3)

            # adaptive grasp
            if hasattr(self.hand, "adaptive_grab"):
                self.hand.adaptive_grab(width_percentage=width_percentage)
            else:
                self.hand.grab()

            time.sleep(0.5)

            # deliver — turn back and come to owner
            self.state.transition("delivering", {"target": target})
            self.generate_thought(f"got the {target}! bringing it to you...")

            self.chassis.spin_left(40)
            time.sleep(1.5)
            self.chassis.stop()
            if self._check_interrupt("DELIVERING"): return
            self.chassis.forward(50)
            time.sleep(2)
            self.chassis.stop()
            if self._check_interrupt("DELIVERING"): return

            # extend arm to deliver
            self.arm.raise_arm(130)
            time.sleep(0.5)
            self.hand.release()

            self.generate_thought(f"here's your {target}! 😊")
            self.emotions.update_from_event("task_complete")
            self.emotions.set_emotion("proud", 0.9)
            log_event("action", f"delivered '{target}'")

        else:
            self.generate_thought(f"sorry, i couldn't find the {target} 😢")
            self.emotions.update_from_event("object_not_found")
            log_event("action", f"failed to find '{target}'")

            # return to owner
            self.chassis.spin_left(40)
            time.sleep(1.5)
            self.chassis.stop()
            self.chassis.forward(40)
            time.sleep(1)
            self.chassis.stop()

        self.arm.home()
        if self.state.get_state() in ("FETCHING", "SEARCHING", "DELIVERING"):
            self.state.transition("idle")
        self._idle_since = time.time()

    def _handle_grab(self):
        """physically grab an object using IK."""
        self.state.transition("grabbing")
        self.generate_thought("calculating IK coordinates to grab...")
        self.emotions.set_emotion("focused", 0.8)

        # reach straight out and down slightly with fingers down
        self.arm.move_to_xyz(0, 15.0, 5.0, wrist_roll=0.0)
        time.sleep(1.0)

        if hasattr(self.hand, "adaptive_grab"):
            self.hand.adaptive_grab(width_percentage=40.0)
        else:
            self.hand.grab()

        time.sleep(0.5)
        self.arm.home()
        self.generate_thought("got it! 🦾")
        self.emotions.set_emotion("proud", 0.9)
        self.state.transition("idle")
        self._idle_since = time.time()

    def _handle_hold(self, target):
        """hold an object presented in front of the camera, even if untrained."""
        self.state.transition("holding")
        t = target if target else "this"
        self.generate_thought(f"Okay, let me see what you want me to hold... ({t})")
        self.emotions.set_emotion("curious", 0.8)
        log_event("action", "attempting to hold object")

        # grab a frame and see if there's anything directly in front
        frame = self.camera.capture_frame()
        found = False

        if frame is not None and self.detector:
            # try to detect any bounding box
            detections = self.detector.detect_objects(frame)
            if detections:
                self.generate_thought("I see something there!")
                found = True

        if not found:
            # even if nothing detected, assume the user is handing something
            # right in front
            self.generate_thought(
                "I don't clearly recognize it, but I'll reach out and grab it.")

        # position arm to reach out
        self.arm.rotate(90)
        self.arm.raise_arm(130)
        self.arm.wrist(90)
        time.sleep(0.5)

        # grasp
        if hasattr(self.hand, "adaptive_grab"):
            self.hand.adaptive_grab(width_percentage=30.0)
        else:
            self.hand.grab()

        time.sleep(0.5)
        self.generate_thought("I'm holding it! Tell me when to 'release' it.")
        self.emotions.update_from_event("task_complete")

    def _handle_release(self, target):
        """release the currently held object."""
        current_state = self.state.get_state()
        if current_state == "HOLDING":
            self.generate_thought("Okay, letting go...")
            self.hand.release()
            time.sleep(0.5)
            self.arm.home()
            self.state.transition("idle")
            self._idle_since = time.time()
            self.generate_thought("Dropped it! 🐾")
            self.emotions.set_emotion("happy", 0.7)
            log_event("action", "released object")
        else:
            self.generate_thought("I'm not holding anything right now!")

    def _handle_forward(self, target):
        duration = 0.5 if target and "little" in target else 2.0
        self.generate_thought("moving forward")
        self.chassis.forward()
        time.sleep(duration)
        self.chassis.stop()
        self.path_recorder.record("forward", 60, duration)

    def _handle_backward(self, target):
        duration = 0.5 if target and "little" in target else 2.0
        self.generate_thought("moving backward")
        self.chassis.backward()
        time.sleep(duration)
        self.chassis.stop()
        self.path_recorder.record("backward", 60, duration)

    def _handle_left(self, target):
        duration = 0.3 if target and "little" in target else 1.0
        self.generate_thought("turning left")
        self.chassis.turn_left()
        time.sleep(duration)
        self.chassis.stop()
        self.path_recorder.record("turn_left", 45, duration)

    def _handle_right(self, target):
        duration = 0.3 if target and "little" in target else 1.0
        self.generate_thought("turning right")
        self.chassis.turn_right()
        time.sleep(duration)
        self.chassis.stop()
        self.path_recorder.record("turn_right", 45, duration)

    def _handle_feeling(self):
        """express current feeling and arousal/valence."""
        emotion, intensity = self.emotions.get_emotion()
        emoji = self.emotions.get_emoji()
        self.generate_thought(
            f"I am feeling {emotion} right now! {emoji} (Intensity: {
                intensity * 100:.0f}%)")
        if hasattr(self.hand, "wave_fingers"):
            self.hand.wave_fingers()

    def _handle_point(self):
        """trigger point gesture on the hand."""
        self.generate_thought("Look at that! 👉")
        if hasattr(self.hand, "point"):
            self.hand.point()

    def _handle_rock(self):
        self.generate_thought("Rock! ✊")
        if hasattr(self.hand, "rock"):
            self.hand.rock()

    def _handle_paper(self):
        self.generate_thought("Paper! ✋")
        if hasattr(self.hand, "paper"):
            self.hand.paper()

    def _handle_scissors(self):
        self.generate_thought("Scissors! ✌️")
        if hasattr(self.hand, "scissors"):
            self.hand.scissors()

    def _handle_thumbs_up(self):
        self.generate_thought("Awesome! 👍")
        if hasattr(self.hand, "thumbs_up"):
            self.hand.thumbs_up()

    def _handle_wave_fingers(self):
        self.generate_thought("Hello there! 👋")
        if hasattr(self.hand, "wave_fingers"):
            self.hand.wave_fingers()

    def _handle_scan(self):
        """scan the area and report findings."""
        self.state.transition("scanning")
        self.generate_thought("scanning the area...")
        self.emotions.set_emotion("curious", 0.7)

        if self.scanner:
            detections = self.scanner.scan_area()
            if detections:
                labels = [d["label"] for d in detections]
                self.generate_thought(f"i see: {', '.join(labels)}")
                self.emotions.update_from_event("object_found")
            else:
                self.generate_thought(
                    "i don't see anything interesting right now")
        else:
            self.generate_thought("scanner not available")

        self.state.transition("idle")
        self._idle_since = time.time()

    def _handle_charge(self):
        """retrace recorded path to return to the charging station."""
        steps = self.path_recorder.step_count
        if steps == 0:
            self.generate_thought("i'm already at the charging station! 🔋")
            self.state.transition("charging")
            self.emotions.set_emotion("relaxed", 0.6)
            time.sleep(2)
            if self.state.get_state() == "CHARGING":
                self.state.transition("idle")
            self._idle_since = time.time()
            return

        self.state.transition("returning")
        self.generate_thought(f"heading home! retracing {steps} steps... 🔋")
        self.emotions.update_from_event("mission_start")
        log_event("navigation", f"returning to base — {steps} steps")

        def return_task():
            def on_step(step_num, total, action):
                if step_num % 5 == 0 or step_num == total:
                    self.generate_thought(
                        f"retracing step {step_num}/{total}: {action}")

            success = self.path_recorder.replay(self.chassis, on_step=on_step)

            if success:
                self.generate_thought("i'm back at the charging station! 🔋⚡")
                self.emotions.set_emotion("happy", 0.8)
                self.state.transition("charging")
                log_event("navigation", "arrived at charging station")

                # rest on the charging station for N minutes
                rest_seconds = self._charge_rest_minutes * 60
                self.generate_thought(
                    f"resting for {self._charge_rest_minutes} minutes while charging... 😴")
                self.emotions.set_emotion("sleepy", 0.6)

                # sleep in 10s intervals so we can still be interrupted
                rest_start = time.time()
                while (time.time() - rest_start) < rest_seconds:
                    if not self._running:
                        break
                    if self.state.get_state() != "CHARGING":
                        break  # interrupted by voice command
                    time.sleep(10)

                # reset timer and resume roaming
                self._last_charge_return = time.time()
                self._auto_return_triggered = False
                self.path_recorder.clear()

                if self.state.get_state() == "CHARGING":
                    self.generate_thought("fully charged! back to exploring! 🦁⚡")
                    self.emotions.set_emotion("excited", 0.8)
                    self.state.transition("idle")
            else:
                self.generate_thought("couldn't make it back... 😕")
                self.emotions.set_emotion("frustrated", 0.7)
                self._auto_return_triggered = False
                if self.state.get_state() == "RETURNING":
                    self.state.transition("idle")

            self._idle_since = time.time()

        threading.Thread(target=return_task, daemon=True).start()

    def _handle_play(self):
        """enter play mode with random fun movements."""
        self.state.transition("playing")
        self.generate_thought("yay! let's play! 🎮")
        self.emotions.update_from_event("playing")
        log_event("action", "play mode activated")

        play_duration = self.config["behavior"].get("play_mode_timeout", 120)
        start = time.time()

        while self._running and time.time() - start < play_duration:
            if self._check_interrupt("PLAYING"):
                break

            # random playful movements
            action = random.choice(
                ["wave", "wiggle", "spin", "fingers", "nod"])

            if action == "wave":
                self.arm.wave()
            elif action == "wiggle":
                self.arm.wiggle(speed=4, angle_range=25, duration=1.5)
            elif action == "spin":
                self.chassis.spin_left(30)
                time.sleep(0.8)
                self.chassis.spin_right(30)
                time.sleep(0.8)
                self.chassis.stop()
            elif action == "fingers":
                self.hand.wave_fingers()
            elif action == "nod":
                self.arm.raise_arm(120)
                time.sleep(0.3)
                self.arm.raise_arm(80)
                time.sleep(0.3)
                self.arm.raise_arm(90)

            self.generate_thought(random.choice([
                "this is so fun! 🎉",
                "play play play!",
                "wheee! 🎮",
                "i love playing with you!",
                "catch me if you can! 😄",
            ]))
            time.sleep(2)

        self.arm.home()
        self.state.transition("idle")
        self._idle_since = time.time()
        self.generate_thought("that was fun! 😊")

    def _handle_greeting(self):
        """respond to hello/hi with wave and handshake."""
        self.state.transition("greeting")
        self.generate_thought("hello there! 👋")
        self.emotions.update_from_event("greeted")
        log_event("action", "greeting")

        self.arm.wave()
        time.sleep(0.5)
        self.arm.handshake()

        self.generate_thought("nice to see you! 😊")
        if self.state.get_state() == "GREETING":
            self.state.transition("idle")
        self._idle_since = time.time()

    def _handle_love(self):
        """respond to 'i love you' with excited movements."""
        self.state.transition("expressing")
        self.emotions.update_from_event("loved")
        self.generate_thought("i love you too!! ❤️❤️❤️")
        log_event("action", "expressing love!")

        # go crazy with excitement!
        for _ in range(3):
            self.arm.wiggle(speed=5, angle_range=30, duration=1.0)
            self.hand.wave_fingers()
            self.chassis.spin_left(50)
            time.sleep(0.5)
            self.chassis.spin_right(50)
            time.sleep(0.5)
            self.chassis.stop()

        self.arm.wave()
        self.generate_thought("you make me so happy! ❤️🦁")
        if self.state.get_state() == "EXPRESSING":
            self.state.transition("idle")
        self._idle_since = time.time()

    def _handle_stop(self):
        """stop all movement immediately."""
        self.chassis.stop()
        self.arm.home()
        self.hand.release()
        self.state.transition("idle")
        self.generate_thought("stopped! standing by.")
        self._idle_since = time.time()
        log_event("action", "emergency stop")

    def _handle_come(self):
        """come to the owner (move forward)."""
        self.state.transition("delivering")
        self.generate_thought("coming to you!")
        self.chassis.forward(50)
        time.sleep(2)
        self.chassis.stop()
        self.path_recorder.record("forward", 50, 2.0)
        self.generate_thought("i'm here! 🐾")
        if self.state.get_state() == "DELIVERING":
            self.state.transition("IDLE")

    def _handle_dance(self):
        """dance to the music!"""
        self.state.transition("PLAYING")
        self.generate_thought("let's dance! 💃🕺")
        self.emotions.set_emotion("excited", 0.9)
        for _ in range(3):
            self.arm.wiggle(speed=6, angle_range=40, duration=1.0)
            self.chassis.spin_left(50)
            time.sleep(0.5)
            self.chassis.spin_right(50)
            time.sleep(0.5)
        self.chassis.stop()
        self.arm.home()
        if self.state.get_state() == "PLAYING":
            self.state.transition("IDLE")

    def _handle_who_am_i(self):
        """Respond to who am i."""
        owner = self.config.get("robot", {}).get("owner_name", "unknown")
        if owner == "unknown":
            msg = "I do not know who you are yet. Please set your identity in the trainer dashboard."
        else:
            msg = f"You are {owner}, my creator and owner."

        self.generate_thought(msg)
        self.emotions.set_emotion("happy", 0.7)
        self.arm.nod()
        self.state.transition("idle")

    def _handle_who_are_you(self):
        """Respond to who are you."""
        name = self.config.get("robot", {}).get("name", "simba")
        msg = f"I am {name}, an AI-powered bionic robotic system built with MXSA technology."
        self.generate_thought(msg)
        self.emotions.set_emotion("happy", 0.8)
        self.arm.wave()
        self.state.transition("idle")

    def _handle_joke(self):
        """Tell a random joke."""
        import random
        jokes = [
            "Why do robots never get afraid? Because they have nerves of steel.",
            "What is a robot's favorite kind of music? Heavy metal.",
            "Why did the robot go to the doctor? It had a virus.",
            "I'm reading a book on anti-gravity. I can't put it down."]
        msg = random.choice(jokes)
        self.generate_thought(msg)
        self.emotions.set_emotion("playful", 0.9)
        self.hand.thumbs_up()
        self.state.transition("idle")

    def _handle_follow(self):
        """Track and follow the user or an object autonomously."""
        self.generate_thought(
            "entering autonomous tracking mode! say 'stop' to end.")
        self.state.transition("following")
        self.emotions.set_emotion("focused", 0.9)

        def _tracking_loop():
            target = "person"
            self.chassis.stop()
            while self.state.get_state() == "FOLLOWING" and self._running:
                frame = self.camera.capture_frame()
                if frame is not None:
                    found, conf, bbox = self.detector.is_object_in_frame(
                        frame, target)
                    if found:
                        x1, y1, x2, y2 = bbox
                        frame_width = frame.shape[1]
                        center_x = (x1 + x2) / 2

                        error = center_x - (frame_width / 2)

                        # basic P-controller for centering
                        if error > 50:
                            self.chassis.turn_right(35)
                        elif error < -50:
                            self.chassis.turn_left(35)
                        else:
                            # centered, move forward slightly
                            self.chassis.forward(30)
                            time.sleep(0.2)
                            self.chassis.stop()
                    else:
                        self.chassis.stop()
                time.sleep(0.1)
            self.chassis.stop()
            self.generate_thought("stopped tracking.")

        threading.Thread(target=_tracking_loop, daemon=True).start()

    def _handle_time(self):
        """Respond with the current time."""
        import datetime
        now = datetime.datetime.now().strftime("%I:%M %p")
        msg = f"The current time is {now}."
        self.generate_thought(msg)
        self.hand.point()
        self.arm.nod()
        self.state.transition("idle")

    def _handle_sing(self):
        """Sing a little song."""
        msg = "Beep boop beep, I am a robot. Beep boop bop, scanning for objects. Da da da da."
        self.generate_thought(msg)
        self.emotions.set_emotion("happy", 0.9)
        self.hand.rock()
        self.state.transition("idle")

    def _handle_story(self):
        """Tell a short story."""
        msg = "Once upon a time, a robotic bionic arm was built using MXSA technology. It learned to see, hear, and think. And it lived happily ever after on a desk."
        self.generate_thought(msg)
        self.emotions.set_emotion("curious", 0.8)
        self.arm.wiggle()
        self.state.transition("idle")

    def _handle_status(self):
        """report current status."""
        self.state.transition("EXPRESSING")
        self.generate_thought("all systems nominal! i am ready.")
        self.arm.wave()
        time.sleep(1)
        self.state.transition("IDLE")

    def _handle_patrol(self):
        """patrol the area."""
        self.state.transition("ROAMING")
        self.generate_thought("on patrol! watching out for intruders.")
        self.chassis.forward(30)
        time.sleep(3)
        self.chassis.stop()
        self.path_recorder.record("forward", 30, 3.0)
        self.chassis.spin_left(45)
        time.sleep(1)
        self.chassis.stop()
        self.path_recorder.record("spin_left", 45, 1.0)
        self.chassis.forward(30)
        time.sleep(3)
        self.chassis.stop()
        self.path_recorder.record("forward", 30, 3.0)
        self.state.transition("IDLE")

    def _handle_rest(self):
        """go to rest mode."""
        self.state.transition("IDLE")
        self.generate_thought("going to sleep... zzz...")
        self.arm.droop()
        self.chassis.stop()
        self.emotions.set_emotion("sleepy", 0.8)
        self._idle_since = time.time()

    def _handle_describe(self):
        """describe what simba currently sees."""
        self.generate_thought("let me look around...")
        frame = self.camera.capture_frame()
        if frame is not None:
            detections = self.detector.detect_objects(frame)
            if detections:
                labels = [d["label"] for d in detections]
                self.generate_thought(f"i can see: {', '.join(labels)}")
            else:
                self.generate_thought(
                    "i don't see anything specific right now")
        else:
            self.generate_thought("my camera isn't working right now 📷")

    def _handle_praise(self):
        """respond to praise."""
        self.emotions.update_from_event("praised")
        self.generate_thought(random.choice([
            "thank you! that means a lot! 😊",
            "yay! i'm a good boy! 🐾",
            "i'll keep doing my best!",
            "*happy wiggle* 🎉",
        ]))
        self.arm.wiggle(speed=3, angle_range=15, duration=1)
        log_event("action", "received praise")

    def _handle_scold(self):
        """respond to scolding."""
        self.emotions.update_from_event("scolded")
        self.generate_thought("i'm sorry... i'll try harder 😢")
        self.arm.droop(angle=30)
        time.sleep(2)
        self.arm.home()
        log_event("action", "received scolding")

    def _handle_roam(self):
        """periodic roaming behavior when idle."""
        if self.state.is_busy():
            return

        self.state.transition("roaming")
        self.generate_thought("roaming around...")

        # random movement
        action = random.choice(["forward", "left", "right", "scan"])
        if action == "forward":
            self.chassis.forward(30)
            time.sleep(1)
            if self.state.get_state() == "ROAMING":
                self.chassis.stop()
                self.path_recorder.record("forward", 30, 1.0)
        elif action == "left":
            self.chassis.turn_left(25)
            time.sleep(0.8)
            if self.state.get_state() == "ROAMING":
                self.chassis.stop()
                self.path_recorder.record("turn_left", 25, 0.8)
        elif action == "right":
            self.chassis.turn_right(25)
            time.sleep(0.8)
            if self.state.get_state() == "ROAMING":
                self.chassis.stop()
                self.path_recorder.record("turn_right", 25, 0.8)
        elif action == "scan":
            detections = self.scanner.quick_scan() if self.scanner else []
            if detections:
                labels = [d["label"] for d in detections]
                self.generate_thought(f"spotted: {', '.join(labels)}")

        if self.state.get_state() == "ROAMING":
            self.state.transition("idle")
        self._idle_since = time.time()

    def _handle_clear_path(self):
        """clear all recorded path steps."""
        self.path_recorder.clear()
        self.generate_thought("path cleared! i've forgotten the way home")
        log_event("action", "path cleared")
        self.state.transition("idle")
        self._idle_since = time.time()

    def _handle_path_status(self):
        """report how many steps are recorded in the path recorder."""
        n = self.path_recorder.step_count
        self.generate_thought(f"I have {n} steps recorded to find my way home")
        log_event("action", f"path status: {n} steps")
        self.state.transition("idle")
        self._idle_since = time.time()

    def _handle_speed_up(self):
        """increase movement speed (no-op placeholder)."""
        self.generate_thought("speeding up!")
        log_event("action", "speed up (no-op)")
        self.state.transition("idle")
        self._idle_since = time.time()

    def _handle_slow_down(self):
        """decrease movement speed (no-op placeholder)."""
        self.generate_thought("slowing down!")
        log_event("action", "slow down (no-op)")
        self.state.transition("idle")
        self._idle_since = time.time()

    def _handle_calibrate(self):
        """calibrate all servos to home position."""
        self.arm.home()
        self.hand.release()
        self.generate_thought("calibrating all servos to home position")
        log_event("action", "calibrate servos")
        self.state.transition("idle")
        self._idle_since = time.time()

    def _handle_temperature(self):
        """report the CPU temperature."""
        cpu_temp = 0.0
        if _HAS_PSUTIL:
            try:
                temps = psutil.sensors_temperatures()
                if "cpu_thermal" in temps:
                    cpu_temp = temps["cpu_thermal"][0].current
                elif "cpu-thermal" in temps:
                    cpu_temp = temps["cpu-thermal"][0].current
            except Exception:
                pass
        if cpu_temp > 0:
            self.generate_thought(f"my CPU temperature is {cpu_temp:.1f}°C")
        else:
            self.generate_thought("I couldn't read the CPU temperature")
        log_event("action", f"temperature: {cpu_temp}°C")
        self.state.transition("idle")
        self._idle_since = time.time()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # llm thinking
    # ------------------------------------------------------------------

    def generate_thought(self, context_event):
        """generate a dynamic personality thought using the LLM based on current emotion."""
        if self.llm is None:
            self._thinking_text = context_event
            return context_event

        emotion_name, _ = self.emotions.get_emotion()
        prompt = f"Event: {context_event}\nCurrent Emotion: {emotion_name}\nWrite a very short (1 sentence), playful internal thought or dialogue reacting to this."
        thought = self.think(prompt)
        self._thinking_text = thought
        return thought

    def think(self, prompt):
        """query the llm for a response.

        args:
            prompt: question or situation description

        returns:
            str: llm response, or fallback if llm not available
        """
        if self.llm is None or not _HAS_REQUESTS:
            return self._simple_response(prompt)

        try:
            payload = {
                "model": self.ollama_model,
                "system": SIMBA_SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config["ai"].get("llm_temperature", 0.7),
                    "num_predict": self.config["ai"].get("llm_max_tokens", 128),
                    "num_ctx": 512,
                    "num_thread": 2
                }
            }
            # send request to local ollama daemon
            resp = requests.post(self.ollama_url, json=payload, timeout=45)
            resp.raise_for_status()
            answer = resp.json().get("response", "").strip()

            if not answer:
                answer = self._simple_response(prompt)

            log_event("ai", f"llm thought: {answer[:100]}")
            return answer
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 500:
                logger.warning("Ollama HTTP Error 500: Model might be missing. Run 'ollama pull qwen2.5:0.5b'")
            else:
                logger.warning("Ollama HTTP Error %s: %s", e.response.status_code, e.response.text)
            return self._simple_response(prompt)
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama daemon is not running or unreachable at %s. Falling back to simple responses.", self.ollama_url)
            return self._simple_response(prompt)
        except Exception as e:
            logger.warning(f"Ollama error (falling back to simple response): {e}")
            return self._simple_response(prompt)

    def _simple_response(self, prompt):
        """fallback response when llm is not available."""
        prompt_lower = prompt.lower()
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "hello! i'm simba! 🐾"
        if "love" in prompt_lower:
            return "i love you too! ❤️"
        if "find" in prompt_lower or "get" in prompt_lower:
            return "i'll look for it right away!"
        if "play" in prompt_lower:
            return "yay! let's play! 🎮"
        return "i'm here and ready to help! 🦁"

    # ------------------------------------------------------------------
    # external command interface (from web dashboard)
    # ------------------------------------------------------------------

    def send_command(self, text):
        """process a text command (from web dashboard or api).

        args:
            text: command text string
        """
        text = text.strip().lower()
        log_event("web", f"web command: '{text}'")
        cmd = self.cmd_parser.parse(text)
        if cmd:
            threading.Thread(
                target=self._handle_command, args=(cmd,), daemon=True
            ).start()
        else:
            self.generate_thought(f"didn't understand: '{text}'")

    # ------------------------------------------------------------------
    # status for web dashboard
    # ------------------------------------------------------------------

    def get_status(self):
        """get full robot status for web dashboard.

        returns:
            dict with all status fields
        """
        cpu_percent = 0.0
        ram_percent = 0.0
        ram_used = 0
        ram_total = 0
        cpu_temp = 0.0

        if _HAS_PSUTIL:
            cpu_percent = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            ram_percent = mem.percent
            ram_used = mem.used // (1024 * 1024)
            ram_total = mem.total // (1024 * 1024)
            try:
                temps = psutil.sensors_temperatures()
                if "cpu_thermal" in temps:
                    cpu_temp = temps["cpu_thermal"][0].current
                elif "cpu-thermal" in temps:
                    cpu_temp = temps["cpu-thermal"][0].current
            except Exception:
                pass

        emotion_name, emotion_intensity = self.emotions.get_emotion()

        return {
            "timestamp": time.time(),
            "uptime": time.time() - self._boot_time,
            "state": self.state.get_state(),
            "state_duration": self.state.get_state_duration(),
            "state_context": self.state.get_context(),
            "emotion": emotion_name,
            "emotion_intensity": emotion_intensity,
            "emotion_emoji": self.emotions.get_emoji(),
            "thinking": self._thinking_text,
            "cpu_percent": cpu_percent,
            "ram_percent": ram_percent,
            "ram_used_mb": ram_used,
            "ram_total_mb": ram_total,
            "cpu_temp": cpu_temp,
            "arm_position": self.arm.get_position() if hasattr(
                self.arm,
                "get_position") else {},
            "grip_state": self.hand.get_grip_state() if hasattr(
                self.hand,
                "get_grip_state") else "unknown",
            "objects_known": len(
                self.memory.get_all_objects()) if hasattr(
                    self.memory,
                    "get_all_objects") else 0,
            "camera_active": self.camera.is_running() if hasattr(
                self.camera,
                "is_running") else False,
            "voice_active": self.listener.is_listening() if hasattr(
                self.listener,
                "is_listening") else False,
            "llm_loaded": self.llm is not None,
            "last_command": self._last_command,
            "memory_items": self.memory.get_all_objects() if hasattr(
                self.memory,
                "get_all_objects") else [],
        }
