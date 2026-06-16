# _max_cyan_ — project_mxsa
"""flask training interface for simba — runs on linux pc.

provides a web ui for:
  - managing object classes and training images
  - recording voice samples for speaker verification
  - triggering model training with progress tracking
  - exporting trained models as a deployable zip
"""

import base64
import io
import os
import shutil
import threading
import uuid
import shlex
import secrets
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any, Optional, Dict

try:
    import paramiko
    _HAS_PARAMIKO = True
except ImportError:
    paramiko = None
    _HAS_PARAMIKO = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

import yaml
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth

auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username: str, password: str) -> Optional[str]:
    if secrets.compare_digest(username, "mxsa") and secrets.compare_digest(password, "0"):
        return username
    return None

# project root
_project_root = Path(__file__).resolve().parent.parent

# load config
_config_path = _project_root / "config" / "simba_config.yaml"
with open(_config_path, "r") as _f:
    _config = yaml.safe_load(_f)

# data directories
_data_dir = _project_root / "data"
_objects_dir = _data_dir / "objects"
_voice_dir = _data_dir / "voice"
_models_dir = _project_root / "models"

# ensure directories exist
_objects_dir.mkdir(parents=True, exist_ok=True)
_voice_dir.mkdir(parents=True, exist_ok=True)
_models_dir.mkdir(parents=True, exist_ok=True)

# valid image extensions
_valid_image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# flask app
app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5000", "http://127.0.0.1:5000"]}})
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50mb max upload

# --- training state ---
_training_status = {
    "active": False,
    "task": None,
    "phase": "idle",
    "progress": 0,
    "total": 0,
    "message": "",
    "result": None,
    "error": None,
}
_status_lock = threading.Lock()
_model_lock = threading.Lock()
_config_lock = threading.Lock()


def _update_status(
    active: bool = None,
    task: str = None,
    phase: str = None,
    progress: int = None,
    total: int = None,
    message: str = None,
    result: dict = None,
    error: str = None,
) -> None:
    """thread-safe training status update."""
    with _status_lock:
        if active is not None:
            _training_status["active"] = active
        if task is not None:
            _training_status["task"] = task
        if phase is not None:
            _training_status["phase"] = phase
        if progress is not None:
            _training_status["progress"] = progress
        if total is not None:
            _training_status["total"] = total
        if message is not None:
            _training_status["message"] = message
        if result is not None:
            _training_status["result"] = result
        if error is not None:
            _training_status["error"] = error


# --- routes ---

@app.route("/")
def index():
    """render the main training interface."""
    return render_template(
        "trainer.html",
        version=_config["robot"]["version"],
        robot_name=_config["robot"]["name"],
    )


@app.route('/test')
def test_page():
    return render_template('test.html')





# lazy-loaded detector
_detector = None


@app.route("/api/predict", methods=["POST"])
@auth.login_required
def predict():
    """run live prediction on a webcam frame."""
    global _detector
    data = request.get_json(silent=True) or {}
    if not data or "image" not in data:
        return jsonify({"error": "No image data"}), 400

    try:
        import numpy as np
        from simba.vision.detector import ObjectDetector

        if _detector is None:
            _detector = ObjectDetector()

        # decode base64
        header, encoded = data["image"].split(",", 1)
        image_bytes = base64.b64decode(encoded)
        if not _HAS_PIL:
            return jsonify({"error": "PIL is not installed"}), 500
        Image.MAX_IMAGE_PIXELS = 10000000 # Prevent decompression bombs
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = np.array(image)

        label, conf = _detector.classify(frame)
        return jsonify(
            {"label": label, "confidence": round(float(conf) * 100, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/add_object", methods=["POST"])
@auth.login_required
def add_object():
    """add a new object class with uploaded images.

    expects multipart form with:
      - name: object class name (normalized to lowercase)
      - images: one or more image files
    """
    name = request.form.get("name", "").strip().lower()
    if not name:
        return jsonify({"error": "object name is required"}), 400

    safe_name = "".join(
        c if c.isalnum() or c == "_" else "_" for c in name.strip().lower()
    )
    if not safe_name:
        return jsonify({"error": "invalid object name"}), 400

    class_dir = _objects_dir / safe_name
    class_dir.mkdir(parents=True, exist_ok=True)

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "no images uploaded"}), 400

    saved = 0
    for f in files:
        if f.filename:
            ext = Path(f.filename).suffix.lower()
            if ext in _valid_image_ext:
                filename = f"{uuid.uuid4().hex[:8]}{ext}"
                f.save(str(class_dir / filename))
                saved += 1

    if saved == 0:
        return jsonify({"error": "no valid images uploaded"}), 400

    total = len(list(class_dir.iterdir()))
    return jsonify({
        "name": safe_name,
        "saved": saved,
        "total": total,
        "message": f"added {saved} images to class '{safe_name}'",
    })


@app.route("/api/objects", methods=["GET"])
@auth.login_required
def list_objects():
    """list all object classes with image counts."""
    classes = []
    if _objects_dir.exists():
        for class_dir in sorted(_objects_dir.iterdir()):
            if class_dir.is_dir():
                images = [
                    f.name for f in class_dir.iterdir()
                    if f.suffix.lower() in _valid_image_ext
                ]
                classes.append({
                    "name": class_dir.name,
                    "count": len(images),
                    "images": images[:6],  # first 6 for thumbnails
                })
    return jsonify({"classes": classes, "total": len(classes)})


@app.route("/api/objects/<name>", methods=["DELETE"])
@auth.login_required
def delete_object(name: str):
    """remove an object class and all its images."""
    safe_name = "".join(c if c.isalnum() or c ==
                        "_" else "_" for c in name.strip().lower())
    if not safe_name:
        return jsonify({"error": "invalid object name"}), 400
    class_dir = _objects_dir / safe_name
    if not class_dir.exists():
        return jsonify({"error": f"class '{safe_name}' not found"}), 404

    try:
        shutil.rmtree(str(class_dir))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"message": f"deleted class '{safe_name}'"})


@app.route("/api/objects/<name>/images", methods=["GET"])
@auth.login_required
def list_object_images(name: str):
    """list all images inside a specific object class."""
    safe_name = "".join(c if c.isalnum() or c ==
                        "_" else "_" for c in name.strip().lower())
    if not safe_name:
        return jsonify({"error": "invalid object name"}), 400
    class_dir = _objects_dir / safe_name
    if not class_dir.exists():
        return jsonify({"error": f"class '{safe_name}' not found"}), 404

    images = [f.name for f in class_dir.iterdir() if f.suffix.lower()
              in _valid_image_ext]
    return jsonify({"images": images, "total": len(images)})


@app.route("/api/objects/<name>/<filename>", methods=["DELETE"])
@auth.login_required
def delete_object_image(name: str, filename: str):
    """remove a specific image from an object class."""
    safe_name = "".join(c if c.isalnum() or c ==
                        "_" else "_" for c in name.strip().lower())
    if not safe_name:
        return jsonify({"error": "invalid object name"}), 400
    class_dir = _objects_dir / safe_name
    if not class_dir.exists():
        return jsonify({"error": f"class '{safe_name}' not found"}), 404

    from werkzeug.utils import secure_filename
    filename = secure_filename(filename)
    if not filename:
        return jsonify({"error": "invalid filename"}), 400

    img_path = class_dir / filename
    if not img_path.exists() or not img_path.is_file():
        return jsonify({"error": f"image '{filename}' not found"}), 404

    try:
        img_path.unlink()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(
        {"message": f"deleted image '{filename}' from '{safe_name}'"})


@app.route("/api/object_thumbnail/<name>/<filename>")
@auth.login_required
def object_thumbnail(name: str, filename: str):
    """serve an object class image thumbnail."""
    safe_name = "".join(c if c.isalnum() or c ==
                        "_" else "_" for c in name.strip().lower())
    if not safe_name:
        return jsonify({"error": "invalid object name"}), 400
    class_dir = _objects_dir / safe_name
    if not class_dir.exists():
        return jsonify({"error": "class not found"}), 404
    from werkzeug.utils import secure_filename
    filename = secure_filename(filename)
    return send_from_directory(str(class_dir), filename)


@app.route("/api/train_vision", methods=["POST"])
@auth.login_required
def train_vision():
    """trigger vision model training in a background thread."""
    with _status_lock:
        if _training_status["active"]:
            return jsonify({"error": "training already in progress"}), 409
        _training_status["active"] = True

    def _train_thread():
        try:
            # verify we have enough classes
            classes = [
                d for d in _objects_dir.iterdir()
                if d.is_dir() and any(
                    f.suffix.lower() in _valid_image_ext for f in d.iterdir()
                )
            ]
            if len(classes) < 2:
                _update_status(active=False, error="need at least 2 object classes with images to train")
                return

            _update_status(
                active=True, task="vision", phase="initializing",
                progress=0, total=100, message="loading mobilenetv2...",
                error=None, result=None,
            )

            from trainer.train_vision import VisionTrainer

            trainer = VisionTrainer()
            _update_status(message="extracting features...")

            model, label_map, metrics = trainer.train(str(_objects_dir))
            _update_status(
                phase="saving", progress=90, message="saving model..."
            )

            with _model_lock:
                trainer.save_model(model, label_map)

            _update_status(
                active=False, phase="complete", progress=100,
                message="vision training complete",
                result=metrics,
            )
        except Exception as e:
            _update_status(
                active=False, phase="error", message=str(e), error=str(e),
            )

    thread = threading.Thread(target=_train_thread, daemon=True)
    thread.start()
    return jsonify({"message": "vision training started"})


@app.route("/api/record_voice", methods=["POST"])
@auth.login_required
def record_voice():
    """save an uploaded voice sample (.wav)."""
    if "audio" not in request.files:
        return jsonify({"error": "no audio file uploaded"}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "empty filename"}), 400

    _voice_dir.mkdir(parents=True, exist_ok=True)

    # generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"voice_{timestamp}_{uuid.uuid4().hex[:6]}.wav"
    filepath = _voice_dir / filename

    try:
        audio_file.save(str(filepath))
    except Exception as e:
        return jsonify({"error": f"Failed to save audio: {e}"}), 500

    # count total samples
    total = len(list(_voice_dir.glob("*.wav"))) + 1
    return jsonify({
        "filename": filename,
        "total": total,
        "message": f"saved voice sample: {filename}",
    })


@app.route("/api/voice_samples", methods=["GET"])
@auth.login_required
def list_voice_samples():
    """list all voice samples."""
    samples = []
    if _voice_dir.exists():
        for f in sorted(_voice_dir.glob("*.wav")):
            samples.append({
                "name": f.name,
                "size": f.stat().st_size,
            })
    return jsonify({"samples": samples, "total": len(samples)})


@app.route("/api/voice_samples/<name>", methods=["DELETE"])
@auth.login_required
def delete_voice_sample(name: str):
    """delete a voice sample."""
    from werkzeug.utils import secure_filename
    filename = secure_filename(name)
    if not filename:
        return jsonify({"error": "invalid filename"}), 400

    filepath = _voice_dir / filename
    if not filepath.exists() or not filepath.is_file():
        return jsonify({"error": "sample not found"}), 404

    try:
        filepath.unlink()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"message": f"deleted {filename}"})


@app.route("/api/train_voice", methods=["POST"])
@auth.login_required
def train_voice():
    """trigger speaker model training in a background thread."""
    with _status_lock:
        if _training_status["active"]:
            return jsonify({"error": "training already in progress"}), 409
        _training_status["active"] = True

    def _train_thread():
        try:
            samples = list(_voice_dir.glob("*.wav"))
            if len(samples) < 3:
                _update_status(active=False, error="need at least 3 voice samples to train")
                return

            _update_status(
                active=True, task="voice", phase="initializing",
                progress=0, total=100, message="starting voice training...",
                error=None, result=None,
            )

            from trainer.train_voice import VoiceTrainer

            trainer = VoiceTrainer()
            _update_status(
                phase="extracting", progress=30,
                message="extracting mfcc features...",
            )

            model = trainer.train_speaker_model(str(_voice_dir))
            _update_status(
                phase="saving", progress=90, message="saving speaker model...",
            )

            with _model_lock:
                trainer.save_model(model)

            _update_status(
                active=False, phase="complete", progress=100,
                message="speaker model training complete",
                result={"status": "trained", "n_samples": len(samples)},
            )
        except Exception as e:
            _update_status(
                active=False, phase="error", message=str(e), error=str(e),
            )

    thread = threading.Thread(target=_train_thread, daemon=True)
    thread.start()
    return jsonify({"message": "voice training started"})


@app.route("/api/test_voice", methods=["POST"])
@auth.login_required
def test_voice():
    """test an uploaded or provided voice text against the system."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    import sys
    sys.path.append(str(_project_root))
    from simba.voice.command_parser import CommandParser

    parser = CommandParser()
    cmd = parser.parse(text)

    return jsonify({
        "text": text,
        "command": cmd
    })


@app.route("/api/train_behavior", methods=["POST"])
@auth.login_required
def train_behavior():
    """trigger behavior model training."""
    with _status_lock:
        if _training_status["active"]:
            return jsonify({"error": "training already in progress"}), 409
        _training_status["active"] = True

    def _train_thread():
        try:
            _update_status(
                active=True, task="behavior", phase="initializing",
                progress=0, total=100, message="generating training data...",
                error=None, result=None,
            )

            from trainer.train_behavior import BehaviorTrainer

            trainer = BehaviorTrainer()
            _update_status(
                phase="training", progress=50,
                message="training behavior model...",
            )

            model, metrics = trainer.create_default_behavior_model()
            _update_status(
                phase="saving",
                progress=90,
                message="saving behavior model...",
            )

            with _model_lock:
                trainer.save_model(model)

            _update_status(
                active=False, phase="complete", progress=100,
                message="behavior model training complete",
                result=metrics,
            )
        except Exception as e:
            _update_status(
                active=False, phase="error", message=str(e), error=str(e),
            )

    thread = threading.Thread(target=_train_thread, daemon=True)
    thread.start()
    return jsonify({"message": "behavior training started"})


@app.route("/api/export", methods=["POST"])
@auth.login_required
def export_models():
    """export all trained models to a zip archive."""
    with _status_lock:
        if _training_status["active"]:
            return jsonify(
                {"error": "training in progress, wait to export"}), 409
        _training_status["active"] = True

    def _export_thread():
        try:
            _update_status(
                active=True, task="export", phase="packaging",
                progress=0, total=100, message="packaging models...",
                error=None, result=None,
            )

            from trainer.export_model import ModelExporter

            exporter = ModelExporter()
            with _model_lock:
                zip_path = exporter.export()

            _update_status(
                active=False, phase="complete", progress=100,
                message="export complete",
                result={
                    "zip_path": zip_path,
                    "filename": os.path.basename(zip_path),
                    "size_bytes": os.path.getsize(zip_path),
                },
            )
        except Exception as e:
            _update_status(
                active=False, phase="error", message=str(e), error=str(e),
            )

    thread = threading.Thread(target=_export_thread, daemon=True)
    thread.start()
    return jsonify({"message": "export started"})


@app.route("/api/training_status", methods=["GET"])
@auth.login_required
def training_status():
    """get current training progress."""
    with _status_lock:
        return jsonify(_training_status.copy())


@app.route("/download/<filename>")
@auth.login_required
def download_file(filename: str):
    """download an exported model zip."""
    from werkzeug.utils import secure_filename
    filename = secure_filename(filename)
    return send_from_directory(str(_models_dir), filename, as_attachment=True)


@app.route("/api/model_stats", methods=["GET"])
@auth.login_required
def model_stats():
    """get statistics about trained models."""
    stats = {}
    model_files = {
        "object_classifier": _models_dir / "object_classifier.joblib",
        "object_labels": _models_dir / "object_labels.json",
        "speaker_model": _models_dir / "speaker_model.joblib",
        "behavior_model": _models_dir / "behavior_model.joblib",
    }

    for name, path in model_files.items():
        if path.exists():
            stats[name] = {
                "exists": True,
                "size_bytes": path.stat().st_size,
                "modified": datetime.fromtimestamp(
                    path.stat().st_mtime
                ).isoformat(),
            }
        else:
            stats[name] = {"exists": False}

    # check for exported zip
    zip_path = _models_dir / "simba_model.zip"
    if zip_path.exists():
        stats["export_zip"] = {
            "exists": True,
            "size_bytes": zip_path.stat().st_size,
            "modified": datetime.fromtimestamp(
                zip_path.stat().st_mtime
            ).isoformat(),
            "filename": "simba_model.zip",
        }
    else:
        stats["export_zip"] = {"exists": False}

    return jsonify(stats)


@app.route("/api/default_objects", methods=["POST"])
@auth.login_required
def create_default_objects():
    import subprocess
    import sys

    script_path = _project_root / "download_massive_dataset.py"
    if not script_path.exists():
        return jsonify({"error": "download_massive_dataset.py not found"}), 404

    # Start the massive dataset download in the background so we don't freeze
    # the web UI
    subprocess.Popen([sys.executable, str(script_path)])

    return jsonify({
        "message": "Massive dataset download (500+ objects) started in the background! Please check the terminal running this server for live progress. Once complete, click 'Train Model'."
    })


@app.route("/api/deploy", methods=["POST"])
@auth.login_required
def deploy_models():
    if not _HAS_PARAMIKO:
        return jsonify(
            {"error": "paramiko is not installed. Run: pip install paramiko"}), 400

    data = request.get_json(silent=True) or {}
    host = data.get("host")
    username = data.get("username")
    password = data.get("password")
    remote_path = data.get("remote_path", "/home/pi/MXSA/models")

    if not all([host, username, password]):
        return jsonify(
            {"error": "Host, username, and password are required"}), 400

    try:
        def _deploy_thread():
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(
                    hostname=host,
                    username=username,
                    password=password,
                    timeout=10)

                sftp = ssh.open_sftp()
                safe_remote_path = shlex.quote(remote_path)
                try:
                    sftp.stat(remote_path)
                except IOError:
                    ssh.exec_command(f"mkdir -p {safe_remote_path}")

                for f in _models_dir.iterdir():
                    if f.is_file():
                        sftp.put(str(f), f"{remote_path}/{f.name}")

                sftp.close()
                ssh.close()
            except Exception as e:
                print(f"Deploy error: {e}")

        threading.Thread(target=_deploy_thread, daemon=True).start()
        return jsonify(
            {"message": f"Successfully started deployment to {username}@{host}:{remote_path}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/owner", methods=["POST", "GET"])
@auth.login_required
def manage_owner():
    config_path = _project_root / "config" / "simba_config.yaml"
    import yaml

    if request.method == "GET":
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            owner = cfg.get("robot", {}).get("owner_name", "unknown")
            return jsonify({"owner": owner})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        owner_name = data.get("owner_name", "").strip()
        if not owner_name:
            return jsonify({"error": "owner_name cannot be empty"}), 400

        try:
            with _config_lock:
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f)
    
                if "robot" not in cfg:
                    cfg["robot"] = {}
                cfg["robot"]["owner_name"] = owner_name
    
                with open(config_path, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False)

            return jsonify(
                {"message": f"owner identity updated to {owner_name}"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/api/camera_capture", methods=["POST"])
@auth.login_required
def camera_capture():
    data = request.get_json(silent=True) or {}
    b64_img = data.get("image")
    name = data.get("name", "").strip().lower()

    if not b64_img or not name:
        return jsonify({"error": "Image and name are required"}), 400

    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if not safe_name:
        return jsonify({"error": "invalid object name"}), 400
    class_dir = _objects_dir / safe_name
    class_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Strip data URL prefix if present
        if "," in b64_img:
            b64_img = b64_img.split(",", 1)[1]

        img_data = base64.b64decode(b64_img)
        filename = f"cam_{uuid.uuid4().hex[:8]}.jpg"
        filepath = class_dir / filename

        try:
            with open(filepath, "wb") as f:
                f.write(img_data)
        except Exception as e:
            return jsonify({"error": f"Failed to save image: {e}"}), 500

        total = len(list(class_dir.iterdir())) + 1
        return jsonify({
            "name": safe_name,
            "filename": filename,
            "saved": 1,
            "total": total,
            "message": f"Captured image for '{safe_name}'"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    """run the training flask server."""
    print("=" * 60)
    print(
        f"  simba trainer — {
            _config['robot']['name']} v{
            _config['robot']['version']}")
    print("  _max_cyan_ — project_mxsa")
    print("=" * 60)
    print(f"  data directory:   {_data_dir}")
    print(f"  models directory: {_models_dir}")
    print(f"  objects directory: {_objects_dir}")
    print(f"  voice directory:  {_voice_dir}")
    print("=" * 60)
    print("  open http://localhost:5000 in your browser")
    print("=" * 60)

    try:
        app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
    except Exception as e:
        print(f"\nCRITICAL: Failed to start trainer server: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
