# _max_cyan_ — project_mxsa
"""flask web server with socketio for the simba dashboard.

serves the real-time monitoring dashboard, rest api endpoints,
and mjpeg camera stream. pushes status updates every second
via websocket to all connected clients.
"""

import os
import time
import threading
import secrets
from datetime import datetime
from typing import Any, Optional, Dict, Tuple

import yaml

try:
    import psutil
except ImportError:
    psutil = None

from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
try:
    from flask_cors import CORS
except ImportError:
    CORS = None
from flask_httpauth import HTTPBasicAuth

auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username: str, password: str) -> Optional[str]:
    if secrets.compare_digest(username, "admin") and secrets.compare_digest(password, "mxsa123"):
        return username
    return None

from simba.utils.logger import get_logger, log_event, get_log_history

logger = get_logger("simba.web.server")

# ---------------------------------------------------------------------------
# load config
# ---------------------------------------------------------------------------
_config_path = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))),
    "config",
    "simba_config.yaml")

with open(_config_path, "r") as _f:
    _config = yaml.safe_load(_f)

_web_config = _config.get("web", {})

# ---------------------------------------------------------------------------
# flask + socketio setup
# ---------------------------------------------------------------------------
_template_dir = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    "templates")
_static_dir = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    "static")

app = Flask(
    __name__,
    template_folder=_template_dir,
    static_folder=_static_dir,
)
if CORS is not None:
    CORS(app, resources={r"/*": {"origins": _web_config.get("cors_origins", "*")}})
else:
    # Fallback to manual headers if flask-cors is missing
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        return response
app.url_map.strict_slashes = False
app.config["SECRET_KEY"] = "simba_mxsa_dashboard_2026"

socketio = SocketIO(app, cors_allowed_origins=_web_config.get("cors_origins", "*"), async_mode="threading")

@socketio.on_error_default
def default_error_handler(e):
    logger.error(f"WebSocket error: {e}")

# ---------------------------------------------------------------------------
# brain reference — set by main entry point via get_brain_ref()
# ---------------------------------------------------------------------------
_brain = None
_boot_time = datetime.now()


def get_brain_ref(brain: Any) -> None:
    """connect the brain instance so the server can query robot state.

    args:
        brain: the simba brain instance (simba.core.brain.Brain).
    """
    global _brain
    _brain = brain
    logger.info("brain reference connected to web server")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_system_stats() -> dict:
    """gather cpu, ram, and temperature stats via psutil.

    returns:
        dict with cpu_percent, ram_percent, ram_used_mb, ram_total_mb,
        temperature keys.
    """
    stats = {
        "cpu_percent": 0.0,
        "ram_percent": 0.0,
        "ram_used_mb": 0,
        "ram_total_mb": 0,
        "temperature": 0.0,
    }

    if psutil is None:
        return stats

    try:
        stats["cpu_percent"] = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        stats["ram_percent"] = mem.percent
        stats["ram_used_mb"] = round(mem.used / (1024 * 1024), 1)
        stats["ram_total_mb"] = round(mem.total / (1024 * 1024), 1)
    except Exception as exc:
        logger.warning(f"psutil stats error: {exc}")

    # raspberry pi thermal zone
    try:
        temp_path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(temp_path):
            with open(temp_path, "r") as tf:
                stats["temperature"] = round(
                    int(tf.read().strip()) / 1000.0, 1)
    except Exception:
        pass

    return stats


def _get_uptime_seconds() -> float:
    """return seconds since the server booted."""
    return (datetime.now() - _boot_time).total_seconds()


def _get_robot_state() -> dict:
    """query the brain for current robot state.

    returns:
        dict with state, emotion, thinking, detected_objects, memory keys.
        provides sensible defaults when brain is not connected.
    """
    state = {
        "state": "idle",
        "state_description": "waiting for input",
        "emotion": "curious",
        "thinking": "",
        "detected_objects": [],
        "memory": [],
        "last_command": "",
    }

    if _brain is None:
        return state

    # pull from brain — each attr is optional so we guard gracefully
    try:
        state["state"] = getattr(_brain, "current_state", "idle")
    except Exception:
        pass

    try:
        state["state_description"] = getattr(
            _brain, "state_description", "waiting for input"
        )
    except Exception:
        pass

    try:
        state["emotion"] = getattr(_brain, "current_emotion", "curious")
    except Exception:
        pass

    try:
        state["thinking"] = getattr(_brain, "thinking_text", "")
    except Exception:
        pass

    try:
        if _brain is not None and hasattr(_brain, "detected_objects"):
            detected = _brain.detected_objects
        else:
            detected = []
        state["detected_objects"] = list(detected) if detected else []
    except Exception:
        pass

    try:
        memory = getattr(_brain, "object_memory", [])
        state["memory"] = list(memory) if memory else []
    except Exception:
        pass

    try:
        state["last_command"] = getattr(_brain, "last_command", "")
    except Exception:
        pass

    return state


def _build_status_payload() -> dict:
    """build the complete status payload for websocket push.

    returns:
        dict combining system stats, robot state, logs, and uptime.
    """
    system = _get_system_stats()
    robot = _get_robot_state()
    logs = get_log_history(30)
    uptime = _get_uptime_seconds()

    # if brain has get_status(), use that for more complete data
    if _brain is not None and hasattr(_brain, "get_status"):
        try:
            brain_status = _brain.get_status()
            robot.update({
                "state": brain_status.get("state", robot["state"]),
                "emotion": brain_status.get("emotion", robot["emotion"]),
                "emotion_intensity": brain_status.get("emotion_intensity", 0.5),
                "emotion_emoji": brain_status.get("emotion_emoji", "🤖"),
                "thinking": brain_status.get("thinking", robot["thinking"]),
                "grip_state": brain_status.get("grip_state", "unknown"),
                "state_context": brain_status.get("state_context", {}),
            })
            system.update({
                "cpu_percent": brain_status.get("cpu_percent", system["cpu_percent"]),
                "ram_percent": brain_status.get("ram_percent", system["ram_percent"]),
                "ram_used_mb": brain_status.get("ram_used_mb", system["ram_used_mb"]),
                "ram_total_mb": brain_status.get("ram_total_mb", system["ram_total_mb"]),
                "temperature": brain_status.get("cpu_temp", system["temperature"]),
            })
        except Exception as exc:
            logger.debug(f"brain get_status error: {exc}")

    # servo angles — arm + fingers
    servo_angles = {
        "rotation": 90, "elbow": 90, "wrist": 90,
        "finger_1": 30, "finger_2": 30, "finger_3": 30,
    }
    if _brain is not None:
        try:
            arm_pos = _brain.arm.get_position()
            servo_angles["rotation"] = arm_pos.get("rotation", 90)
            servo_angles["elbow"] = arm_pos.get("elbow", 90)
            servo_angles["wrist"] = arm_pos.get("wrist", 90)
        except Exception as exc:
            logger.debug(f"arm position read error: {exc}")

        try:
            finger_pos = _brain.hand.get_positions()
            if finger_pos and len(finger_pos) >= 3:
                servo_angles["finger_1"] = finger_pos[0]
                servo_angles["finger_2"] = finger_pos[1]
                servo_angles["finger_3"] = finger_pos[2]
        except Exception as exc:
            logger.debug(f"hand positions read error: {exc}")

    robot["servo_angles"] = servo_angles

    # motor state — chassis speeds + direction
    motor_state = {
        "left_speed": 0, "right_speed": 0, "direction": "stopped",
    }
    if _brain is not None:
        try:
            speed = _brain.chassis.get_speed()
            if speed and len(speed) >= 2:
                left, right = speed[0], speed[1]
                motor_state["left_speed"] = left
                motor_state["right_speed"] = right
                # derive direction
                if left == 0 and right == 0:
                    motor_state["direction"] = "stopped"
                elif left > 0 and right > 0:
                    if abs(left - right) < 10:
                        motor_state["direction"] = "forward"
                    elif left > right:
                        motor_state["direction"] = "turning_right"
                    else:
                        motor_state["direction"] = "turning_left"
                elif left < 0 and right < 0:
                    motor_state["direction"] = "backward"
                elif left < 0 and right > 0:
                    motor_state["direction"] = "turning_left"
                elif left > 0 and right < 0:
                    motor_state["direction"] = "turning_right"
        except Exception as exc:
            logger.debug(f"chassis speed read error: {exc}")

    robot["motor_state"] = motor_state

    # imu data — orientation + tilt
    imu_data = {"orientation": "level", "tilt_angle": 0.0}
    if _brain is not None:
        try:
            imu_reading = _brain.imu.read_all()
            if imu_reading:
                imu_data["orientation"] = imu_reading.get(
                    "orientation", "level")
                imu_data["tilt_angle"] = imu_reading.get("tilt_angle", 0.0)
        except Exception as exc:
            logger.debug(f"imu read error: {exc}")

    robot["imu_data"] = imu_data

    return {
        "system": system,
        "robot": robot,
        "logs": logs,
        "uptime": uptime,
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# camera stream helpers
# ---------------------------------------------------------------------------

# minimal 1x1 white JPEG placeholder (valid)
_PLACEHOLDER_JPEG = bytes([
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
    0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
    0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
    0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
    0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
    0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
    0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
    0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
    0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0A, 0x0B, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F,
    0x00, 0x54, 0xDB, 0x9E, 0xA7, 0xA3, 0xFF, 0xD9,
])


def _generate_camera_frames():
    """generator that yields mjpeg frames from the brain's camera.

    yields:
        bytes: multipart/x-mixed-replace jpeg frame data.
    """
    try:
        while True:
            frame = None

            if _brain is not None:
                try:
                    cam = getattr(_brain, "camera", None)
                    if cam and hasattr(cam, "get_mjpeg_frame"):
                        frame = cam.get_mjpeg_frame()
                except Exception:
                    pass

            if frame is None:
                frame = _PLACEHOLDER_JPEG

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            )
            socketio.sleep(1.0 / 15)  # ~15 fps cap
    except GeneratorExit:
        logger.debug("mjpeg generator closed by client")


# ---------------------------------------------------------------------------
# background status push thread
# ---------------------------------------------------------------------------

_push_thread = None
_push_thread_running = False
_push_lock = threading.Lock()


def _status_push_loop() -> None:
    """background loop that emits status_update every interval."""

    interval = _web_config.get("update_interval", 1.0)

    # Throttle to max 10Hz (0.1s) to prevent network flooding
    if interval < 0.1:
        interval = 0.1

    logger.info(f"status push thread started — interval {interval}s")

    # initial cpu measurement so psutil doesn't return 0 on first call
    if psutil:
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    last_emit_hash = None

    while _push_thread_running:
        try:
            payload = _build_status_payload()

            # --- God-Mode Vision YOLO Tracking ---
            if _brain is not None and hasattr(
                    _brain, "camera") and hasattr(
                    _brain, "detector"):
                try:
                    with _brain._hardware_lock:
                        frame = _brain.camera.get_frame()
                    if frame is not None:
                        detections = _brain.detector.detect_objects(frame)
                        payload["robot"]["detected_objects"] = detections
                except Exception as e_vis:
                    logger.debug(f"yolo vision loop error: {e_vis}")

            # Cache check to avoid network flooding
            # Ignore uptime and timestamp which change every tick
            import json
            state_to_hash = {
                "system": payload.get("system"),
                "robot": payload.get("robot"),
                "logs_len": len(payload.get("logs", []))
            }
            try:
                current_hash = hash(json.dumps(state_to_hash, sort_keys=True))
            except Exception:
                current_hash = None  # Fallback if not serializable

            if current_hash is None or current_hash != last_emit_hash:
                socketio.emit("status_update", payload)
                last_emit_hash = current_hash
        except Exception as exc:
            logger.warning(f"status push error: {exc}")

        socketio.sleep(interval)

    logger.info("status push thread stopped")


# ---------------------------------------------------------------------------
# hardware control endpoints
# ---------------------------------------------------------------------------

@app.route("/hardware")
@auth.login_required
def hardware_page() -> str:
    """serve the direct hardware control panel."""
    return render_template("hardware.html",
                           camera_url="/video_feed"
                           )


@app.route("/diagnostics")
@auth.login_required
def diagnostics_page() -> str:
    """serve the hardware diagnostics panel."""
    return render_template("diagnostics.html",
                           camera_url="/video_feed"
                           )


@app.route("/api/diagnostics/run", methods=["POST"])
@auth.login_required
def api_diagnostics_run():
    """run automated hardware diagnostics tests."""
    if _brain is None:
        return jsonify({"error": "brain not connected"}), 503
        
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "invalid payload format"}), 400
    test_name = data.get("test")
    
    if test_name == "arm_sweep":
        try:
            _brain.arm.move_smooth({"rotation": 90, "elbow": 90, "elbow_2": 90, "wrist": 90}, duration=0.5)
            time.sleep(0.5)
            _brain.arm.move_smooth({"rotation": 45, "elbow": 110, "elbow_2": 80, "wrist": 45}, duration=0.5)
            time.sleep(0.5)
            _brain.arm.move_smooth({"rotation": 135, "elbow": 70, "elbow_2": 100, "wrist": 135}, duration=0.5)
            time.sleep(0.5)
            _brain.arm.move_smooth({"rotation": 90, "elbow": 90, "elbow_2": 90, "wrist": 90}, duration=0.5)
            return jsonify({"status": "arm sweep completed"})
        except Exception as e:
            return jsonify({"error": f"arm sweep failed: {e}"}), 500
            
    elif test_name == "chassis_spin":
        try:
            _brain.chassis.turn_left(speed=50)
            time.sleep(1)
            _brain.chassis.turn_right(speed=50)
            time.sleep(1)
            _brain.chassis.stop()
            return jsonify({"status": "chassis spin completed"})
        except Exception as e:
            return jsonify({"error": f"chassis spin failed: {e}"}), 500
            
    elif test_name == "camera_feed":
        try:
            if hasattr(_brain, 'camera') and _brain.camera.is_active():
                return jsonify({"status": "camera feed is active and capturing frames"})
            else:
                return jsonify({"error": "camera feed is offline"}), 500
        except Exception as e:
            return jsonify({"error": f"camera test failed: {e}"}), 500
            
    elif test_name == "voice_module":
        try:
            if hasattr(_brain, 'listener') and _brain.listener.is_listening():
                energy = _brain.listener.get_energy()
                return jsonify({"status": f"voice module is active (energy: {energy:.3f})"})
            else:
                return jsonify({"error": "voice module is offline"}), 500
        except Exception as e:
            return jsonify({"error": f"voice test failed: {e}"}), 500
            
    else:
        return jsonify({"error": "invalid test name"}), 400


@app.route("/api/hardware/motor", methods=["POST"])
@auth.login_required
def api_hardware_motor():
    """manually actuate the chassis motors."""
    if _brain is None:
        return jsonify({"error": "brain not connected"}), 503

    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "invalid payload format"}), 400
    action = data.get("action")
    speed = data.get("speed", 50)

    try:
        speed = int(speed)
    except (ValueError, TypeError):
        return jsonify({"error": "invalid speed"}), 400

    try:
        if action == "forward":
            _brain.chassis.forward(speed)
        elif action == "backward":
            _brain.chassis.backward(speed)
        elif action == "left":
            _brain.chassis.turn_left(speed)
        elif action == "right":
            _brain.chassis.turn_right(speed)
        elif action == "brake":
            _brain.chassis.brake()
        elif action == "stop":
            _brain.chassis.stop()
        else:
            return jsonify({"error": "invalid motor action"}), 400
    except Exception as e:
        return jsonify({"error": f"chassis error: {e}"}), 500

    return jsonify({"status": f"chassis {action}"})


@app.route("/api/hardware/arm", methods=["POST"])
@auth.login_required
def api_hardware_arm():
    """manually actuate the arm servos."""
    if _brain is None:
        return jsonify({"error": "brain not connected"}), 503

    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "invalid payload format"}), 400
    joint = data.get("joint")
    angle = data.get("angle", 90)

    try:
        angle = int(angle)
    except (ValueError, TypeError):
        return jsonify({"error": "invalid angle"}), 400

    try:
        if joint == "rotation":
            _brain.arm.rotate(angle)
        elif joint == "elbow":
            _brain.arm.raise_arm(angle)
        elif joint == "elbow_2":
            _brain.arm.move_smooth({"elbow_2": angle})
        elif joint == "wrist":
            _brain.arm.wrist(angle)
        else:
            return jsonify({"error": "invalid arm joint"}), 400
    except Exception as e:
        return jsonify({"error": f"arm error: {e}"}), 500

    return jsonify({"status": f"arm {joint} set to {angle}"})


@app.route("/api/hardware/hand", methods=["POST"])
@auth.login_required
def api_hardware_hand():
    """manually actuate the hand servos."""
    if _brain is None:
        return jsonify({"error": "brain not connected"}), 503

    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "invalid payload format"}), 400
    grip = data.get("grip")
    action = data.get("action")

    try:
        if grip is not None:
            if not isinstance(grip, str):
                return jsonify({"error": "invalid grip type"}), 400
            _brain.hand.set_grip(grip)
            return jsonify({"status": f"hand grip set to {grip}"})

        if action == "rock":
            _brain.hand.set_fingers("rock")
        elif action == "paper":
            _brain.hand.set_fingers("paper")
        elif action == "scissors":
            _brain.hand.set_fingers("scissors")
        elif action == "open":
            _brain.hand.set_fingers("open")
        elif action == "close":
            _brain.hand.set_fingers("close")
        elif action == "point":
            _brain.hand.set_fingers("point")
        elif action == "thumbs_up":
            _brain.hand.set_fingers("thumbs_up")
        else:
            return jsonify({"error": "invalid hand action"}), 400
    except Exception as e:
        return jsonify({"error": f"hand error: {e}"}), 500

    return jsonify({"status": f"hand action {action} triggered"})


# ---------------------------------------------------------------------------
# socketio events
# ---------------------------------------------------------------------------

@socketio.on("connect")
def handle_connect():
    """send full status snapshot when a client connects."""
    auth_ok = False
    if request.authorization:
        if verify_password(request.authorization.username, request.authorization.password):
            auth_ok = True
    if not auth_ok:
        return False
        
    logger.info("dashboard client connected")
    log_event("web", "dashboard client connected")
    payload = _build_status_payload()
    emit("status_update", payload)


@socketio.on("disconnect")
def handle_disconnect():
    """log when a client disconnects."""
    logger.info("dashboard client disconnected")


@socketio.on("send_command")
def handle_command_ws(data):
    """receive a command via websocket.

    args:
        data: dict with 'command' key.
    """
    command = data.get(
        "command",
        "").strip().lower() if isinstance(
        data,
        dict) else ""
    if not command:
        return

    logger.info(f"websocket command received: {command}")
    log_event("command", f"user sent: {command}")

    if _brain is not None:
        try:
            if hasattr(_brain, "send_command"):
                _brain.send_command(command)
            elif hasattr(_brain, "process_command"):
                _brain.process_command(command)
        except Exception as exc:
            logger.error(f"command processing error: {exc}")


@socketio.on("request_servo_stream")
def handle_servo_stream():
    """stream live servo angles."""
    if _brain is not None and hasattr(_brain, "arm"):
        try:
            arm_pos = _brain.arm.get_position()
            emit("servo_stream", arm_pos)
        except Exception as exc:
            logger.error(f"servo stream error: {exc}")


@socketio.on("request_motor_stream")
def handle_motor_stream():
    """stream live motor bars."""
    if _brain is not None and hasattr(_brain, "chassis"):
        try:
            speed = _brain.chassis.get_speed()
            emit("motor_stream", {"speed": speed})
        except Exception as exc:
            logger.error(f"motor stream error: {exc}")


@socketio.on("request_imu_stream")
def handle_imu_stream():
    """stream live imu data."""
    if _brain is not None and hasattr(_brain, "imu"):
        try:
            imu_data = _brain.imu.read_all()
            emit("imu_stream", imu_data)
        except Exception as exc:
            logger.error(f"imu stream error: {exc}")


# ---------------------------------------------------------------------------
# rest api routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    """serve the main dashboard page."""
    return render_template("dashboard.html")


@app.route("/api/status")
@auth.login_required
def api_status():
    """return full robot status as json."""
    return jsonify(_build_status_payload())


@app.route("/api/memory")
@auth.login_required
def api_memory():
    """return all remembered objects from the brain's memory."""
    if _brain is not None:
        try:
            if hasattr(
                    _brain,
                    "memory") and hasattr(
                    _brain.memory,
                    "get_all_objects"):
                objects = _brain.memory.get_all_objects()
                return jsonify({"memory": objects})
        except Exception as exc:
            logger.error(f"memory api error: {exc}")

    return jsonify({"memory": []})


@app.route("/api/logs")
@auth.login_required
def api_logs():
    """return recent log history."""
    count = request.args.get("count", 50, type=int)
    return jsonify({"logs": get_log_history(count)})


@app.route("/api/command", methods=["POST"])
@auth.login_required
def api_command():
    """receive a text command via post and forward to the brain."""
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "invalid payload format"}), 400
    command = data.get("command")
    
    if not isinstance(command, str):
        return jsonify({"error": "command must be a string"}), 400
        
    command = command.strip()
    if not command:
        return jsonify({"error": "no command provided"}), 400

    logger.info(f"rest command received: {command}")
    log_event("command", f"user sent: {command}")

    result = {"status": "ok", "command": command, "response": ""}

    if _brain is not None:
        try:
            if hasattr(_brain, "send_command"):
                _brain.send_command(command)
                result["response"] = "command accepted"
            else:
                result["response"] = "brain has no command handler"
        except Exception as exc:
            logger.error(f"command processing error: {exc}")
            result["status"] = "error"
            result["response"] = str(exc)
    else:
        result["response"] = "brain not connected"

    return jsonify(result)


@app.route("/video_feed")
@auth.login_required
def video_feed():
    """serve mjpeg camera stream."""
    return Response(
        _generate_camera_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------------------------------------------
# factory function for main.py
# ---------------------------------------------------------------------------

def create_app(brain=None):
    """create and configure the flask app with brain reference.

    args:
        brain: the simba brain instance.

    returns:
        tuple: (app, socketio) flask app and socketio instance.
    """
    global _brain, _push_thread, _push_thread_running, _boot_time

    if brain is not None:
        _brain = brain
        logger.info("brain reference connected via create_app")

    _boot_time = datetime.now()

    # start the background status push thread
    with _push_lock:
        if not _push_thread_running:
            _push_thread_running = True
            _push_thread = socketio.start_background_task(target=_status_push_loop)

    return app, socketio


# ---------------------------------------------------------------------------
# server start (standalone)
# ---------------------------------------------------------------------------

def start(host: str = None, port: int = None):
    """start the flask-socketio web server.

    args:
        host: bind address, defaults to config value or '0.0.0.0'.
        port: bind port, defaults to config value or 8080.
    """
    global _push_thread, _push_thread_running, _boot_time

    _boot_time = datetime.now()

    host = host or _web_config.get("host", "0.0.0.0")
    port = port or _web_config.get("port", 8080)
    debug = _web_config.get("debug", False)

    # start the background status push thread
    with _push_lock:
        if not _push_thread_running:
            _push_thread_running = True
            _push_thread = socketio.start_background_task(target=_status_push_loop)

    logger.info(f"simba dashboard starting on http://{host}:{port}")
    log_event("web", f"dashboard started on http://{host}:{port}")

    try:
        socketio.run(
            app,
            host=host,
            port=port,
            debug=debug,
            use_reloader=False,
            allow_unsafe_werkzeug=True,
        )
    except KeyboardInterrupt:
        logger.info("dashboard shutting down")
    finally:
        _push_thread_running = False


def stop():
    """stop the background push thread and signal server shutdown."""
    global _push_thread_running
    _push_thread_running = False
    logger.info("dashboard stop requested")
    try:
        socketio.stop()
    except Exception as exc:
        logger.debug(f"socketio stop error: {exc}")


# ---------------------------------------------------------------------------
# standalone run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    start()
