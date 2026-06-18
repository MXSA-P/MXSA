# _max_cyan_ — project_mxsa
"""camera controller — picamera2 interface for capture and streaming.

manages the raspberry pi camera module via picamera2:
  - rgb frame capture for vision pipeline
  - resized frames for mobilenetv2 inference (224×224)
  - jpeg-encoded frames for web mjpeg streaming
  - thread-safe access to the latest frame

uses create_still_configuration for reliable capture.
"""

import gc
import io
import time
import threading
from typing import Optional, Tuple

import yaml
import os
import subprocess
import shutil

import numpy as np

from simba.utils.logger import get_logger, log_event

try:
    from picamera2 import Picamera2, Transform
except ImportError:
    Picamera2 = None
    Transform = None

try:
    from PIL import Image
except ImportError:
    Image = None

logger = get_logger("simba.vision.camera")

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
_config_path = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))),
    "config",
    "simba_config.yaml")


def _load_config() -> dict:
    """load and return the simba configuration dictionary."""
    try:
        with open(_config_path, "r") as fh:
            cfg = yaml.safe_load(fh)
            return cfg if cfg is not None else {}
    except Exception as exc:
        logger.error("failed to load config: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
_MIN_INFERENCE_INTERVAL: float = 0.1  # seconds — caps inference at 10 FPS

# ---------------------------------------------------------------------------
# fallback camera process (cv2 for generic usb webcams / trainer pc)
# ---------------------------------------------------------------------------

class CV2CameraProcess:
    """reads frames natively using robust OpenCV fallback."""
    def __init__(self, resolution, framerate):
        self.resolution = resolution
        self.framerate = framerate
        self.cap = None

    def start(self):
        import cv2
        for idx in [0, 2, 1, 3, 4]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                # Verify we can actually read a frame (avoids metadata nodes)
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    logger.info(f"cv2 locked onto /dev/video{idx}")
                    return
                cap.release()
        raise RuntimeError("cv2 failed to find any working video devices")

    def capture_array(self):
        if self.cap is None:
            raise RuntimeError("Camera not started")
        import cv2
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from cv2")
        # Resize in software rather than hardware to avoid v4l2 property crashes
        frame = cv2.resize(frame, self.resolution)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def stop(self):
        if self.cap:
            self.cap.release()
            self.cap = None

# ---------------------------------------------------------------------------
# fallback camera process (ramdisk / rpicam-still looping)
# ---------------------------------------------------------------------------
class RamdiskCameraProcess:
    """reads frames by repeatedly invoking rpicam-still as requested by user."""
    def __init__(self, resolution, framerate):
        self.resolution = resolution
        self.framerate = framerate
        self._running = False
        self._thread = None
        
        # Use /tmp as it is universally writable on Linux and mapped to RAM by default
        self.shm_dir = "/tmp"
        self.target_file = f"{self.shm_dir}/simba_capture.jpg"
        
        self._latest_frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        self._lock = threading.Lock()

    def start(self):
        os.makedirs(self.shm_dir, exist_ok=True)
        if os.path.exists(self.target_file):
            try:
                os.remove(self.target_file)
            except Exception:
                pass

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="rpicam-loop")
        self._thread.start()
        
        # Wait up to 10 seconds for the first frame to ensure it actually works on slower Pis
        start_wait = time.time()
        while time.time() - start_wait < 10.0:
            if os.path.exists(self.target_file):
                return
            time.sleep(0.1)
            
        self.stop()
        raise RuntimeError("rpicam-still failed to capture a frame within 10 seconds")

    def _capture_loop(self):
        while self._running:
            try:
                # 10ms timeout breaks the sensor warmup. We use 1000ms.
                result = subprocess.run([
                    "rpicam-still",
                    "-o", self.target_file,
                    "-n",  # use -n instead of --nopreview for universal support
                    "-t", "1000",
                    "--width", str(self.resolution[0]),
                    "--height", str(self.resolution[1])
                ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                
                if result.returncode != 0:
                    logger.debug(f"rpicam-still failed: {result.stderr}")
                
                if os.path.exists(self.target_file):
                    if Image is not None:
                        img = Image.open(self.target_file)
                        frame = np.array(img.convert("RGB"))
                    else:
                        import cv2
                        bgr = cv2.imread(self.target_file)
                        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        
                    with self._lock:
                        self._latest_frame = frame
                        
            except Exception as e:
                logger.debug(f"rpicam-still loop error: {e}")
                
            time.sleep(0.1)

    def capture_array(self):
        if not self._running:
            raise RuntimeError("Camera not started")
            
        with self._lock:
            return self._latest_frame.copy()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

# ---------------------------------------------------------------------------
# fallback camera process (rpicam-vid / libcamera-vid / ffmpeg stdout parser)
# ---------------------------------------------------------------------------

class FallbackCameraProcess:
    """reads mjpeg stream from rpicam-vid, libcamera-vid, or ffmpeg via subprocess."""
    def __init__(self, resolution, framerate):
        self.resolution = resolution
        self.framerate = framerate
        self.process = None
        self._buffer = b""

    def _find_executable(self):
        # rpicam-hello cannot stream to stdout and forces a monitor preview,
        # so we strictly rely on rpicam-vid or libcamera-vid.
        for cmd in ["rpicam-vid", "libcamera-vid"]:
            if shutil.which(cmd):
                return cmd, [
                    cmd, "--nopreview", "--timeout", "0", "--codec", "mjpeg", 
                    "--width", str(self.resolution[0]), 
                    "--height", str(self.resolution[1]), 
                    "--framerate", str(self.framerate), "-o", "-"
                ]
        # fallback to ffmpeg for generic usb webcams on linux
        if shutil.which("ffmpeg") and os.path.exists("/dev/video0"):
            return "ffmpeg", [
                "ffmpeg", "-f", "v4l2", "-framerate", str(self.framerate),
                "-video_size", f"{self.resolution[0]}x{self.resolution[1]}",
                "-i", "/dev/video0", "-f", "image2pipe", "-vcodec", "mjpeg", "-"
            ]
        return None, None

    def start(self):
        cmd_name, cmd_args = self._find_executable()
        if not cmd_args:
            raise RuntimeError("No suitable fallback camera executable found (rpicam-vid/libcamera-vid/ffmpeg)")
        
        logger.info(f"starting fallback camera using {cmd_name}")
        self.process = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._buffer = b""

    def capture_array(self):
        if self.process is None:
            raise RuntimeError("Camera not started")
        
        while True:
            # Use read1 to avoid blocking until exactly 64KB is accumulated
            chunk = self.process.stdout.read1(65536)
            if not chunk:
                # Process died or EOF
                time.sleep(0.1)
                raise RuntimeError("Camera stream ended")
            
            self._buffer += chunk
            
            # Find the start of a JPEG
            a = self._buffer.find(b'\xff\xd8')
            if a == -1:
                # Keep the last byte in case the marker is split
                if len(self._buffer) > 1024:
                    self._buffer = self._buffer[-2:]
                continue
                
            # Find the LAST end marker to cleanly bypass any embedded thumbnails.
            # If we grab multiple frames, Image.open() automatically decodes the first one.
            b = self._buffer.rfind(b'\xff\xd9')
            if b != -1 and b > a:
                jpg = self._buffer[a:b+2]
                
                # Aggressively clear the buffer to drop stale frames and stay realtime
                self._buffer = b""
                
                if Image is not None:
                    try:
                        img = Image.open(io.BytesIO(jpg))
                        return np.array(img.convert("RGB"))
                    except Exception:
                        pass
                else:
                    return np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        return np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            if self.process.stdout:
                self.process.stdout.close()
            self.process = None

# ---------------------------------------------------------------------------
# camera controller
# ---------------------------------------------------------------------------

class CameraController:
    """thread-safe camera interface using picamera2.

    attributes:
        camera:      picamera2 instance (or none in simulation).
        resolution:  (width, height) of the main capture.
        stream_res:  (width, height) for mjpeg streaming.
        jpeg_quality: jpeg compression quality (0–100).
        _running:    whether the camera is actively capturing.
    """

    def __init__(self) -> None:
        """initialise the camera controller (does not start capture)."""
        cfg = _load_config()
        cam_cfg = cfg.get("camera", {})

        res = cam_cfg.get("resolution", [640, 480])
        if not isinstance(res, list) or len(res) < 2:
            res = [640, 480]
        self.resolution: Tuple[int, int] = (int(res[0]), int(res[1]))

        stream_res = cam_cfg.get("stream_resolution", [320, 240])
        if not isinstance(stream_res, list) or len(stream_res) < 2:
            stream_res = [320, 240]
        self.stream_resolution: Tuple[int, int] = (
            int(stream_res[0]), int(stream_res[1]))

        self.jpeg_quality: int = int(cam_cfg.get("jpeg_quality", 75))
        self._framerate: int = int(cam_cfg.get("framerate", 15))
        self._format: str = str(cam_cfg.get("format", "rgb888"))

        # internal state
        self._running: bool = False
        self._frame: Optional[np.ndarray] = None
        self._last_inference_time: float = 0.0
        self._frame_lock = threading.Lock()
        self._capture_thread: Optional[threading.Thread] = None

        # picamera2 instance (initialized in start() to prevent sequencer lockups)
        self.camera = None

        log_event("vision", "camera controller initialised", {
            "resolution": list(self.resolution),
        })

    # ------------------------------------------------------------------
    # start / stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """configure and start the camera capture loop."""
        if self._running:
            logger.warning("camera already running")
            return

        success = False

        has_rpicam = shutil.which("rpicam-still") is not None

        strategies = []
        if has_rpicam:
            strategies = [
                ("ramdisk", RamdiskCameraProcess),
                ("cv2", CV2CameraProcess),
                ("fallback", FallbackCameraProcess)
            ]
        else:
            strategies = [
                ("cv2", CV2CameraProcess),
                ("ramdisk", RamdiskCameraProcess),
                ("fallback", FallbackCameraProcess)
            ]

        for name, ProcessClass in strategies:
            try:
                self.camera = ProcessClass(self.resolution, self._framerate)
                self.camera.start()
                self._running = True
                self._capture_thread = threading.Thread(
                    target=self._capture_loop, daemon=True, name="camera-capture")
                self._capture_thread.start()
                logger.info(f"camera started ({name} main mode)")
                log_event("vision", f"camera started via {name}")
                success = True
                break
            except Exception as e:
                logger.warning(f"{name} camera failed: {e}. Trying next fallback...")
                if self.camera:
                    try:
                        self.camera.stop()
                    except Exception:
                        pass
                self.camera = None

        # 3. TRY PICAMERA2 (TERTIARY)
        if not success and Picamera2 is not None:
            import time
            for attempt in range(3):
                try:
                    self.camera = Picamera2()
                    break
                except Exception as exc:
                    logger.warning("Picamera2 init attempt %d failed: %s", attempt + 1, exc)
                    self.camera = None
                    time.sleep(1.0)
            
            if self.camera is not None:
                for attempt in range(3):
                    try:
                        config = self.camera.create_video_configuration(
                            main={"size": self.resolution, "format": self._format},
                            transform=Transform(hflip=False, vflip=False) if Transform else None
                        )
                        self.camera.configure(config)
                        try:
                            self.camera.set_controls({"AfMode": 1, "AwbMode": 1})
                        except Exception:
                            pass
                        
                        self.camera.start()
                        self._running = True
                        self._capture_thread = threading.Thread(
                            target=self._capture_loop, daemon=True, name="camera-capture")
                        self._capture_thread.start()
                        
                        logger.info("camera started (picamera2 secondary mode)")
                        log_event("vision", "camera started via picamera2")
                        success = True
                        break
                    except Exception as exc:
                        logger.warning("Picamera2 start attempt %d failed: %s", attempt + 1, exc)
                        try:
                            self.camera.stop()
                        except Exception:
                            pass
                        time.sleep(0.5)

        # 3. FALLBACK TO SIMULATION
        if not success:
            logger.error("All physical cameras failed. Dropping to simulation.")
            self.camera = None
            self._running = True
            logger.info("camera started (simulation mode)")

    def stop(self) -> None:
        """stop the camera capture loop."""
        if not self._running:
            return

        self._running = False

        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception as exc:
                logger.warning("error stopping camera: %s", exc)

        if self._capture_thread is not None and self._capture_thread is not threading.current_thread():
            self._capture_thread.join(timeout=3.0)
        self._capture_thread = None

        with self._frame_lock:
            self._frame = None

        logger.info("camera stopped")
        log_event("vision", "camera stopped")

    # ------------------------------------------------------------------
    # background capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """continuously capture frames at the configured framerate."""
        interval = 1.0 / self._framerate
        while self._running:
            try:
                if self.camera is not None:
                    frame = self.camera.capture_array()
                    with self._frame_lock:
                        if self._frame is not None:
                            del self._frame
                        self._frame = frame
            except Exception as exc:
                logger.error("capture error: %s", exc)
            time.sleep(interval)
            
            # GC to prevent memory leaks from picamera buffer reallocation
            gc.collect()

    # ------------------------------------------------------------------
    # frame access
    # ------------------------------------------------------------------

    def capture_frame(self) -> Optional[np.ndarray]:
        """return the latest captured frame as an rgb numpy array.

        returns:
            numpy array of shape (h, w, 3) in rgb format, or none if
            no frame is available.
        """
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()

        if self._running and self.camera is None:
            # Simulation Mode: Return a dark blue frame with some noise to clearly indicate
            # it's a simulated feed, not a broken camera.
            frame = np.full(
                (self.resolution[1], self.resolution[0], 3),
                (50, 50, 150),  # Dark blue background
                dtype=np.uint8,
            )
            # Add some random static
            noise = np.random.randint(0, 50, (self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if self.camera is not None and self._running:
            for _ in range(10):
                time.sleep(0.01)
                with self._frame_lock:
                    if self._frame is not None:
                        return self._frame.copy()

        return None

    def capture_for_inference(
        self,
        size: Tuple[int, int] = (224, 224),
    ) -> Optional[np.ndarray]:
        """capture and resize a frame for mobilenetv2 inference.

        args:
            size: target (width, height) — default 224×224.

        returns:
            numpy array of shape (size[1], size[0], 3), dtype uint8,
            or none if capture fails.  rate-limited to 10 fps max.
        """
        # ---- rate limiter (max 10 FPS for inference captures) ----
        now = time.monotonic()
        elapsed = now - self._last_inference_time
        if elapsed < _MIN_INFERENCE_INTERVAL:
            time.sleep(_MIN_INFERENCE_INTERVAL - elapsed)
        self._last_inference_time = time.monotonic()

        frame = self.capture_frame()
        if frame is None:
            return None

        try:
            if Image is not None:
                try:
                    img = Image.fromarray(frame)
                    resample = getattr(Image, "Resampling", Image).BILINEAR
                    img = img.resize(size, resample)
                    result = np.array(img)
                    del frame
                    gc.collect()
                    return result
                except Exception as e:
                    logger.warning(f"Pillow resize failed: {e}")

            # fallback: simple nearest-neighbour resize with numpy
            h, w = frame.shape[:2]
            target_h, target_w = size[1], size[0]
            row_idx = (np.arange(target_h) * h // target_h).astype(int)
            col_idx = (np.arange(target_w) * w // target_w).astype(int)
            result = frame[np.ix_(row_idx, col_idx)]
            del frame
            gc.collect()
            return result
        finally:
            gc.collect()

    def get_mjpeg_frame(self) -> Optional[bytes]:
        """encode the latest frame as a jpeg byte string for web streaming.

        returns:
            jpeg-encoded bytes, or none if no frame is available.
        """
        frame = self.capture_frame()
        if frame is None:
            return None

        # Prefer cv2 for extreme performance if available
        try:
            import cv2
            resized = cv2.resize(
                frame,
                self.stream_resolution,
                interpolation=cv2.INTER_LINEAR,
            )
            # convert rgb -> bgr for opencv encoding
            bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            # frame is RGB, encode directly
            ret, buf = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if ret:
                return buf.tobytes()
        except Exception as e:
            logger.error(f"cv2 mjpeg encode error: {e}")
            try:
                from PIL import Image
                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                return buf.getvalue()
            except Exception as pe:
                logger.error(f"PIL mjpeg encode error: {pe}")
        
        logger.error("neither cv2 nor pillow available for jpeg encoding")
        return None

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    def set_resolution(self, width: int, height: int) -> None:
        """change the capture resolution.  restarts the camera.

        args:
            width:  new width in pixels.
            height: new height in pixels.
        """
        logger.info("set resolution to %dx%d", width, height)
        was_running = self._running

        if was_running:
            self.stop()

        self.resolution = (width, height)

        if was_running:
            self.start()

    def is_running(self) -> bool:
        """check whether the camera is actively capturing.

        returns:
            true if the camera is running.
        """
        return self._running

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """stop the camera and release resources."""
        logger.info("cleaning up camera controller")
        self.stop()

        if self.camera is not None:
            try:
                self.camera.close()
            except Exception as exc:
                logger.warning("camera close error: %s", exc)
            self.camera = None

        log_event("vision", "camera controller cleaned up")
