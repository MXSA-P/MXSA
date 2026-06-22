# _max_cyan_ — project_mxsa
import concurrent.futures
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simba.utils.logger import get_logger
from simba.vision.detector import ObjectDetector

try:
    from simba.vision.yolo_detector import YoloDetector
except ImportError:
    YoloDetector = None

logger = get_logger("simba.vision.hybrid")


class HybridDetector:
    """Combines YOLOv8n tracking with the custom MobileNet+SVM classifier.

    YOLO tracks standard 80 COCO objects (person, cup, phone, etc).
    SVM tracks custom trained objects (wrench, charging_pad, user's face, etc).
    """

    def __init__(self) -> None:
        """Initialize the hybrid detector with YOLO and SVM."""
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.svm = ObjectDetector()
        try:
            self.yolo = YoloDetector() if YoloDetector is not None else None
        except Exception as e:
            logger.warning(f"Failed to load YOLO, falling back to SVM: {e}")
            self.yolo = None
        if self.yolo:
            logger.info("[ok] Hybrid Detector (YOLO + SVM) initialized")
        else:
            logger.info("[fallback] Hybrid Detector (SVM Only) initialized")

    def __del__(self):
        """Clean up resources by shutting down the executor."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)

    def detect_objects(
        self, frame_np: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Detect objects using both models.

        Since SVM only returns one main dominant object without bounding boxes,
        we infer a pseudo-box for the SVM detection if it is highly confident.
        """
        if frame_np is None:
            return []

        detections: List[Dict[str, Any]] = []

        yolo_future = None
        if self.yolo:
            try:
                yolo_future = self.executor.submit(
                    self.yolo.detect_objects, frame_np
                )
            except Exception as e:
                logger.error(f"Failed to submit YOLO detect_objects task: {e}")

        try:
            svm_label, svm_conf = self.svm.classify(frame_np)
        except Exception as e:
            logger.error(f"SVM classification failed: {e}")
            svm_label, svm_conf = "unknown", 0.0

        if yolo_future:
            try:
                yolo_dets = yolo_future.result(timeout=5.0)
                for d in yolo_dets:
                    try:
                        conf = float(d.get("confidence", 0.0))
                        if math.isnan(conf):
                            conf = 0.0
                    except (ValueError, TypeError):
                        conf = 0.0
                    d["confidence"] = max(0.0, min(1.0, conf))
                detections.extend(yolo_dets)
            except concurrent.futures.TimeoutError:
                logger.error("YOLO detection timed out.")
            except concurrent.futures.CancelledError:
                logger.error("YOLO detection was cancelled.")
            except Exception as e:
                logger.error(f"YOLO detection failed: {e}")

        try:
            svm_conf_float = float(svm_conf)
            if math.isnan(svm_conf_float):
                svm_conf_float = 0.0
        except (ValueError, TypeError):
            svm_conf_float = 0.0
        svm_conf = max(0.0, min(1.0, svm_conf_float))

        # if SVM found something custom that YOLO didn't
        if svm_label != "unknown" and svm_conf > 0.5:
            # check if YOLO already found this exact label (unlikely for custom
            # objects, but possible)
            if not any(d.get("label") == svm_label for d in detections):
                h, w = frame_np.shape[:2]
                # SVM doesn't have boxes, we provide a full-frame pseudo-box
                detections.append(
                    {
                        "label": svm_label,
                        "confidence": svm_conf,
                        "bbox": [
                            int(w * 0.1),
                            int(h * 0.1),
                            int(w * 0.8),
                            int(h * 0.8),
                        ],
                    }
                )

        return detections

    def is_object_in_frame(
        self,
        frame_np: Optional[np.ndarray],
        target_label: str,
        threshold: float = 0.5,
    ) -> Tuple[bool, float, Optional[List[float]]]:
        """check if a specific object is in the frame using both models."""
        if frame_np is None:
            return False, 0.0, None

        yolo_future = None
        if self.yolo:
            try:
                yolo_future = self.executor.submit(
                    self.yolo.is_object_in_frame,
                    frame_np,
                    target_label,
                    threshold,
                )
            except Exception as e:
                logger.error(
                    f"Failed to submit YOLO is_object_in_frame task: {e}"
                )

        try:
            svm_label, svm_conf = self.svm.classify(frame_np)
            try:
                svm_conf_float = float(svm_conf)
                if math.isnan(svm_conf_float):
                    svm_conf_float = 0.0
            except (ValueError, TypeError):
                svm_conf_float = 0.0
            svm_conf_float = max(0.0, min(1.0, svm_conf_float))
            if svm_label == target_label and svm_conf_float >= threshold:
                if yolo_future:
                    yolo_future.cancel()
                h, w = frame_np.shape[:2]
                return (
                    True,
                    svm_conf_float,
                    [int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8)],
                )
        except Exception as e:
            logger.error(f"SVM is_object_in_frame failed: {e}")

        if yolo_future:
            try:
                found, conf, bbox = yolo_future.result(timeout=5.0)
                if found:
                    try:
                        conf_float = float(conf)
                        if math.isnan(conf_float):
                            conf_float = 0.0
                    except (ValueError, TypeError):
                        conf_float = 0.0
                    conf_float = max(0.0, min(1.0, conf_float))
                    return found, conf_float, bbox
            except concurrent.futures.TimeoutError:
                logger.error("YOLO is_object_in_frame timed out.")
            except concurrent.futures.CancelledError:
                logger.error("YOLO is_object_in_frame was cancelled.")
            except Exception as e:
                logger.error(f"YOLO is_object_in_frame failed: {e}")

        return False, 0.0, None

    def classify(self, frame_np: Optional[np.ndarray]) -> Tuple[str, float]:
        """Return the most confident dominant object from both models."""
        if frame_np is None:
            return "unknown", 0.0

        best_label = "unknown"
        best_conf = 0.0

        yolo_future = None
        if self.yolo:
            try:
                yolo_future = self.executor.submit(
                    self.yolo.classify, frame_np
                )
            except Exception as e:
                logger.error(f"Failed to submit YOLO classify task: {e}")

        try:
            svm_label, svm_conf = self.svm.classify(frame_np)
            try:
                svm_conf_float = float(svm_conf)
                if math.isnan(svm_conf_float):
                    svm_conf_float = 0.0
            except (ValueError, TypeError):
                svm_conf_float = 0.0
            svm_conf_float = max(0.0, min(1.0, svm_conf_float))
            if svm_conf_float > best_conf:
                best_label = svm_label
                best_conf = svm_conf_float
        except Exception as e:
            logger.error(f"SVM classify failed: {e}")

        if yolo_future:
            try:
                yolo_label, yolo_conf = yolo_future.result(timeout=5.0)
                try:
                    yolo_conf_float = float(yolo_conf)
                    if math.isnan(yolo_conf_float):
                        yolo_conf_float = 0.0
                except (ValueError, TypeError):
                    yolo_conf_float = 0.0
                yolo_conf_float = max(0.0, min(1.0, yolo_conf_float))
                if yolo_conf_float > best_conf:
                    best_label = yolo_label
                    best_conf = yolo_conf_float
            except concurrent.futures.TimeoutError:
                logger.error("YOLO classify timed out.")
            except concurrent.futures.CancelledError:
                logger.error("YOLO classify was cancelled.")
            except Exception as e:
                logger.error(f"YOLO classify failed: {e}")

        return best_label, best_conf
