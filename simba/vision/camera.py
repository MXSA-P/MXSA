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

import numpy as np

from simba.utils.logger import get_logger, log_event

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

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

        # Initialize picamera2 right before starting to prevent stale locks
        if self.camera is None and Picamera2 is not None:
            try:
                self.camera = Picamera2()
            except Exception as exc:
                logger.error("failed to initialise picamera2: %s", exc)
                self.camera = None

        if self.camera is not None:
            import time
            success = False
            for attempt in range(3):
                try:
                    # Optimized configuration for 5MP CSI sensor (OV5647)
                    config = self.camera.create_video_configuration(
                        main={
                            "size": self.resolution,
                            "format": self._format,
                        },
                        transform={"hflip": False, "vflip": False}
                    )
                    # Tune ISP for sharper captures and faster autofocus
                    self.camera.configure(config)
                    try:
                        self.camera.set_controls({"AfMode": 1, "AwbMode": 1})
                    except Exception as ctrl_exc:
                        logger.warning("Failed to set AF/AWB controls (maybe fixed-focus sensor): %s", ctrl_exc)
                    
                    self.camera.start_and_step()  # Use start_and_step to prevent sequencer errors
                    success = True
                    break
                except Exception as exc:
                    logger.warning("Camera start attempt %d failed: %s", attempt + 1, exc)
                    if self.camera:
                        try:
                            self.camera.stop()
                        except:
                            pass
                    time.sleep(0.5)

            if not success:
                logger.error("Failed to start picamera2 after 3 attempts.")
                self.camera = None
                return

            self._running = True

            # start background capture thread
            self._capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="camera-capture", )
            self._capture_thread.start()

            logger.info("camera started")
            log_event("vision", "camera started")
        else:
            # simulation mode — mark as running so other methods work
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
            return np.zeros(
                (self.resolution[1], self.resolution[0], 3),
                dtype=np.uint8,
            )

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
            _, buf = cv2.imencode(
                ".jpg", bgr,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            return buf.tobytes()
        except Exception as e:
            pass

        if Image is not None:
            try:
                img = Image.fromarray(frame)
                resample = getattr(Image, "Resampling", Image).BILINEAR
                # resize to stream resolution
                img = img.resize(self.stream_resolution, resample)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=self.jpeg_quality)
                return buf.getvalue()
            except Exception as e:
                logger.error(f"Pillow jpeg encoding failed: {e}")

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
