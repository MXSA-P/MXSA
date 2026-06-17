# _max_cyan_ — project_mxsa
"""area scanner — arm rotation + camera + detector for spatial awareness.

combines the arm controller, camera, and object detector to perform
sweeping scans of the surrounding area.  detected objects are tagged
with the arm rotation angle at which they were found and optionally
stored in the memory module for future reference.

scan angles are loaded from config (ai.scan_sweep_angles).
"""

import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
import os

from simba.utils.logger import get_logger, log_event

logger = get_logger("simba.vision.scanner")

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
            return yaml.safe_load(fh)
    except Exception as exc:
        logger.error("failed to load config: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# area scanner
# ---------------------------------------------------------------------------

class AreaScanner:
    """spatial awareness scanner using arm rotation, camera, and detector.

    the scanner rotates the arm to a series of angles, captures a frame
    at each position, and runs object detection.  results include the
    rotation angle so the robot knows *where* objects are.

    attributes:
        arm:       arm controller instance (simba.motion.arm.ArmController).
        camera:    camera controller instance (simba.vision.camera.CameraController).
        detector:  object detector instance (simba.vision.detector.ObjectDetector).
        memory:    memory module instance (must support store_scan_result(dict)).
        angles:    list of rotation angles to sweep.
    """

    def __init__(
        self,
        arm_controller: Any,
        camera_controller: Any,
        detector: Any,
        memory: Any = None,
    ) -> None:
        """initialise the area scanner.

        args:
            arm_controller:    armcontroller for rotating the arm.
            camera_controller: cameracontroller for frame capture.
            detector:          objectdetector for frame analysis.
            memory:            optional memory module for persisting results.
        """
        cfg = _load_config()
        ai_cfg = cfg.get("ai", {})

        self.arm = arm_controller
        self.camera = camera_controller
        self.detector = detector
        self.memory = memory

        self.angles: List[int] = [
            int(a) for a in ai_cfg.get(
                "scan_sweep_angles", [
                    0, 45, 90, 135, 180])]

        # continuous scanning state
        self._scanning: bool = False
        self._scan_thread: Optional[threading.Thread] = None

        # cache for smarter target acquisition
        self._last_known_positions: Dict[str, int] = {}

        # settle time after arm movement before capture (seconds)
        self._settle_time: float = 0.4

        logger.info("area scanner initialised — angles %s", self.angles)
        log_event(
            "vision", "area scanner initialised", {
                "angles": self.angles})

    # ------------------------------------------------------------------
    # full sweep scan
    # ------------------------------------------------------------------

    def scan_area(self) -> List[Dict]:
        """perform a full sweep scan across all configured angles.

        at each angle the arm is rotated, a frame is captured, and
        object detection is run.  all detections are collected and
        optionally stored in memory.

        returns:
            list of detection dicts, each augmented with:
                'angle'     — arm rotation angle (degrees).
                'timestamp' — epoch float.
        """
        logger.info("starting area scan (%d positions)", len(self.angles))
        log_event(
            "vision", "area scan started", {
                "positions": len(
                    self.angles)})

        all_detections: List[Dict] = []
        camera_failed = True

        for angle in self.angles:
            detections = self._scan_at_angle(angle)
            if detections is not None:
                camera_failed = False
                all_detections.extend(detections)

        if camera_failed:
            return None

        # return arm to centre
        mid_angle = self.angles[len(self.angles) // 2] if self.angles else 90
        self.arm.move_smooth({"rotation": mid_angle}, speed=2.0)

        logger.info(
            "area scan complete — %d objects across %d angles",
            len(all_detections), len(self.angles),
        )
        log_event("vision", "area scan complete", {
            "total_detections": len(all_detections),
        })

        # persist to memory
        self._store_results(all_detections)

        return all_detections

    # ------------------------------------------------------------------
    # targeted object search
    # ------------------------------------------------------------------

    def search_for_object(
        self,
        target_label: str,
    ) -> Tuple[bool, int, float]:
        """scan all angles looking for a specific object.

        stops as soon as the object is found (returns immediately).

        args:
            target_label: label string to search for.

        returns:
            (found, angle, confidence) tuple.
            angle is 0 and confidence is 0.0 if not found.
        """
        target_label_lower = target_label.lower()
        search_angles = self.angles.copy()

        # smarter target acquisition: prioritize last known position
        if target_label_lower in self._last_known_positions:
            last_angle = self._last_known_positions[target_label_lower]
            if last_angle in search_angles:
                search_angles.remove(last_angle)
                search_angles.insert(0, last_angle)
                logger.debug(
                    "prioritising last known angle %d° for '%s'",
                    last_angle,
                    target_label)

        logger.info(
            "searching for '%s' across %d positions",
            target_label,
            len(search_angles))
        log_event("vision", "object search started", {"target": target_label})

        for angle in search_angles:
            # rotate to angle
            self.arm.move_smooth({"rotation": angle}, speed=2.0)
            time.sleep(self._settle_time)

            # capture frame
            frame = self.camera.capture_frame()
            if frame is None:
                logger.warning("no frame at angle %d — skipping", angle)
                continue

            # check for target
            found, confidence, bbox = self.detector.is_object_in_frame(
                frame, target_label,
            )

            if found:
                logger.info(
                    "found '%s' at angle %d° (confidence %.2f)",
                    target_label, angle, confidence,
                )
                log_event("vision", "object found", {
                    "target": target_label,
                    "angle": angle,
                    "confidence": confidence,
                })

                result = {
                    "label": target_label,
                    "confidence": confidence,
                    "bbox": bbox,
                    "angle": angle,
                    "timestamp": time.time(),
                }
                self._store_results([result])

                return (True, angle, confidence)

        logger.info("'%s' not found in scan", target_label)
        log_event("vision", "object not found", {"target": target_label})
        return (False, 0, 0.0)

    # ------------------------------------------------------------------
    # quick scan (current position only)
    # ------------------------------------------------------------------

    def quick_scan(self) -> List[Dict]:
        """run detection at the current arm position without rotating.

        returns:
            list of detection dicts (same format as scan_area).
        """
        logger.info("quick scan at current position")

        frame = self.camera.capture_frame()
        if frame is None:
            logger.warning("quick scan — no frame available")
            return []

        detections = self.detector.detect_objects(frame)
        if detections is None:
            detections = []

        # tag with current rotation angle
        current_angle = self.arm.current.get("rotation", 90)
        timestamp = time.time()

        for det in detections:
            det["angle"] = current_angle
            det["timestamp"] = timestamp

        logger.info("quick scan — %d detections", len(detections))
        log_event("vision", "quick scan", {"detections": len(detections)})

        self._store_results(detections)

        return detections

    # ------------------------------------------------------------------
    # continuous background scanning
    # ------------------------------------------------------------------

    def continuous_scan(
        self,
        callback: Callable[[List[Dict]], None],
        interval: float = 5.0,
    ) -> None:
        """start a background thread that repeatedly scans and reports.

        each cycle performs a full area scan and passes the results to
        *callback*.

        args:
            callback: function receiving a list of detection dicts.
            interval: seconds to wait between scan cycles.
        """
        if self._scanning:
            logger.warning("continuous scanning already active")
            return

        self._scanning = True

        def _scan_loop() -> None:
            logger.info("continuous scanning started")
            fail_count = 0
            while self._scanning:
                try:
                    # do not scan if arm is currently in use
                    if hasattr(self.arm, "_motion_lock") and self.arm._motion_lock.locked():
                        time.sleep(1.0)
                        continue

                    results = self.scan_area()
                    
                    if results is None:
                        fail_count += 1
                        if fail_count > 5:
                            logger.error("Too many scan failures (camera disconnected?). Stopping.")
                            self._scanning = False
                            break
                    else:
                        fail_count = 0
                        if results:
                            callback(results)
                except Exception as exc:
                    logger.error("continuous scan error: %s", exc)
                time.sleep(interval)
            logger.info("continuous scanning stopped")

        self._scan_thread = threading.Thread(
            target=_scan_loop, daemon=True, name="area-scanner",
        )
        self._scan_thread.start()
        log_event("vision", "continuous scanning started")

    def stop_scan(self) -> None:
        """stop the continuous scanning background thread."""
        if not self._scanning:
            return

        self._scanning = False
        if self._scan_thread is not None and self._scan_thread is not threading.current_thread():
            self._scan_thread.join(timeout=30.0)
        self._scan_thread = None

        logger.info("continuous scanning stopped")
        log_event("vision", "continuous scanning stopped")

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _scan_at_angle(self, angle: int) -> List[Dict]:
        """rotate to *angle*, capture a frame, and detect objects.

        args:
            angle: arm rotation angle in degrees.

        returns:
            list of detection dicts augmented with 'angle' and 'timestamp'.
        """
        self.arm.move_smooth({"rotation": angle}, speed=2.0)
        time.sleep(self._settle_time)

        frame = self.camera.capture_frame()
        if frame is None:
            logger.warning("no frame at angle %d°", angle)
            return None

        detections = self.detector.detect_objects(frame)
        if detections is None:
            detections = []
        timestamp = time.time()

        for det in detections:
            det["angle"] = angle
            det["timestamp"] = timestamp

        if detections:
            logger.debug(
                "angle %d° — %d detections: %s",
                angle, len(detections),
                [d["label"] for d in detections],
            )

        return detections

    def _store_results(self, detections: List[Dict]) -> None:
        """persist detection results to the memory module (if available).

        args:
            detections: list of detection dicts.
        """
        if not detections:
            return

        for det in detections:
            self._last_known_positions[det["label"].lower()] = det["angle"]

        import math

        if self.memory is None:
            return

        try:
            for det in detections:
                if hasattr(self.memory, "remember_object"):
                    # estimate distance based on bounding box width
                    # (a very naive focal length assumption for 2D mapping)
                    if "bbox" in det:
                        width = det["bbox"][2]  # [x, y, w, h]
                    else:
                        box = det.get("box", [0, 0, 1, 1])
                        width = box[3] - box[1]  # xmax - xmin
                    distance_cm = (1.0 / (width + 0.001)) * \
                        30.0  # dummy calibration

                    # angle 90 is forward (+y), 0 is right (+x), 180 is left
                    # (-x)
                    rad = math.radians(det["angle"])
                    x = distance_cm * math.cos(rad)
                    y = distance_cm * math.sin(rad)

                    self.memory.remember_object(
                        label=det["label"],
                        position_angle=det["angle"],
                        confidence=det["confidence"],
                        x=x,
                        y=y
                    )
                else:
                    logger.debug(
                        "memory module has no remember_object method — skipping")
                    break
        except Exception as exc:
            logger.warning("failed to store scan results: %s", exc)
