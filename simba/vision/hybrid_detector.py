# _max_cyan_ — project_mxsa
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from simba.utils.logger import get_logger
from simba.vision.detector import ObjectDetector
try:
    from simba.vision.yolo_detector import YoloDetector
except ImportError:
    YoloDetector = None

logger = get_logger("simba.vision.hybrid")


class HybridDetector:
    """combines YOLOv8n tracking with the custom MobileNet+SVM classifier.

    YOLO tracks standard 80 COCO objects (person, cup, phone, etc).
    SVM tracks custom trained objects (wrench, charging_pad, user's face, etc).
    """

    def __init__(self) -> None:
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

    def detect_objects(self, frame_np: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        """detect objects using both models.

        since SVM only returns one main dominant object without bounding boxes,
        we infer a pseudo-box for the SVM detection if it is highly confident.
        """
        if frame_np is None:
            return []
        
        detections: List[Dict[str, Any]] = []

        # 1. YOLO detections
        if self.yolo:
            try:
                detections.extend(self.yolo.detect_objects(frame_np))
            except Exception as e:
                logger.error(f"YOLO detection failed: {e}")

        # 2. Custom SVM detection (fallback for custom objects)
        svm_label, svm_conf = self.svm.classify(frame_np)

        # if SVM found something custom that YOLO didn't
        if svm_label != "unknown" and svm_conf > 0.5:
            # check if YOLO already found this exact label (unlikely for custom
            # objects, but possible)
            if not any(d["label"] == svm_label for d in detections):
                h, w = frame_np.shape[:2]
                # SVM doesn't have boxes, we provide a full-frame pseudo-box
                detections.append({
                    "label": svm_label,
                    "confidence": svm_conf,
                    "box": [w * 0.1, h * 0.1, w * 0.9, h * 0.9]
                })

        return detections

    def is_object_in_frame(self, frame_np: Optional[np.ndarray], target_label: str, threshold: float = 0.5) -> Tuple[bool, float, Optional[List[float]]]:
        """check if a specific object is in the frame using both models."""
        if frame_np is None:
            return False, 0.0, None

        # check YOLO first
        if self.yolo:
            try:
                found, conf, bbox = self.yolo.is_object_in_frame(
                    frame_np, target_label, threshold)
                if found:
                    return found, conf, bbox
            except Exception as e:
                logger.error(f"YOLO is_object_in_frame failed: {e}")

        # check SVM
        svm_label, svm_conf = self.svm.classify(frame_np)
        if svm_label == target_label and svm_conf >= threshold:
            h, w = frame_np.shape[:2]
            return True, svm_conf, [w * 0.1, h * 0.1, w * 0.9, h * 0.9]

        return False, 0.0, None

    def classify(self, frame_np: Optional[np.ndarray]) -> Tuple[str, float]:
        """return the most confident dominant object from both models."""
        if frame_np is None:
            return "unknown", 0.0

        best_label = "unknown"
        best_conf = 0.0

        # check YOLO
        if self.yolo:
            try:
                yolo_label, yolo_conf = self.yolo.classify(frame_np)
                if yolo_conf > best_conf:
                    best_label = yolo_label
                    best_conf = yolo_conf
            except Exception as e:
                logger.error(f"YOLO classify failed: {e}")

        # check SVM
        svm_label, svm_conf = self.svm.classify(frame_np)
        if svm_conf > best_conf:
            best_label = svm_label
            best_conf = svm_conf

        return best_label, best_conf
