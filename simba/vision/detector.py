# _max_cyan_ — project_mxsa
"""object detector — mobilenetv2 features + svm classifier.

two-stage detection pipeline:
  1. feature extraction  — tflite mobilenetv2 produces a 1280-dim vector.
  2. classification      — scikit-learn svm maps features to object labels.

also provides a simple sliding-window detector that tiles the frame,
extracts features from each tile, and classifies them independently.

models are lazy-loaded: if model files are missing the module degrades
gracefully (returns 'unknown' labels with zero confidence).
"""

import gc
import json
import os
import threading
from typing import Dict, List, Tuple

import yaml
import numpy as np

from simba.utils.logger import get_logger, log_event

try:
    from ai_edge_litert import interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite  # type: ignore[import]
        except ImportError:
            tflite = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    from PIL import Image
except ImportError:
    Image = None

logger = get_logger("simba.vision.detector")

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_config_path = os.path.join(_project_root, "config", "simba_config.yaml")


def _load_config() -> dict:
    """load and return the simba configuration dictionary."""
    try:
        with open(_config_path, "r") as fh:
            return yaml.safe_load(fh)
    except Exception as exc:
        logger.error("failed to load config: %s", exc)
        return {}


def _resolve_path(relative: str) -> str:
    """resolve a path relative to the project root.

    args:
        relative: path string from config (e.g. 'models/foo.tflite').

    returns:
        absolute path string.
    """
    if os.path.isabs(relative):
        return relative
    return os.path.join(_project_root, relative)


# ---------------------------------------------------------------------------
# object detector
# ---------------------------------------------------------------------------

class ObjectDetector:
    """mobilenetv2 feature extractor + svm object classifier.

    attributes:
        interpreter:         tflite interpreter for mobilenetv2 (or none).
        classifier:          scikit-learn svm pipeline (or none).
        labels:              list of known class labels.
        confidence_threshold: minimum svm probability to accept a detection.
        input_size:          expected input image size (w, h) — typically (224, 224).
        feature_dim:         output feature vector dimensionality (1280).
    """

    def __init__(self) -> None:
        """initialise the detector (models are loaded lazily)."""
        cfg = _load_config()
        ai_cfg = cfg.get("ai", {})

        feature_model = ai_cfg.get("vision_model_path", "models/mobilenetv2_feature_extractor.tflite")
        self._feature_model_path: str = _resolve_path(feature_model)
        if not os.path.isfile(self._feature_model_path):
            fallback_path = _resolve_path("models/mobilenet_v2_1.0_224.tflite")
            if os.path.isfile(fallback_path):
                try:
                    os.rename(fallback_path, self._feature_model_path)
                    logger.info(f"Renamed {fallback_path} to {self._feature_model_path}")
                except Exception as e:
                    logger.warning(f"Failed to rename model, using fallback path: {e}")
                    self._feature_model_path = fallback_path

        self._classifier_path: str = _resolve_path(
            ai_cfg.get(
                "object_classifier_path",
                "models/object_classifier.joblib"))
        self._label_map_path: str = _resolve_path(
            ai_cfg.get("object_label_map_path", "models/object_labels.json")
        )
        self.confidence_threshold: float = float(
            ai_cfg.get("confidence_threshold", 0.6)
        )

        # model state
        self.interpreter = None
        self.classifier = None
        self.labels: List[str] = []
        self.input_size: Tuple[int, int] = (224, 224)
        self.feature_dim: int = 1280
        self._lock = threading.Lock()

        # tflite tensor indices (populated on load)
        self._input_index: int = 0
        self._output_index: int = 0

        # sliding window config
        self._window_scales: List[float] = [1.0, 0.5]
        self._window_stride: float = 0.5  # fraction of window size

        # object tracking state
        self._trackers: Dict[str, List[float]] = {}
        self._smoothing_factor: float = 0.6

        # attempt to load models at init
        self.load_feature_extractor()
        self.load_classifier()
        
        try:
            from simba.vision.yolo_detector import YoloDetector
            self.yolo = YoloDetector()
        except Exception as e:
            logger.warning(f"Failed to load YoloDetector: {e}")
            self.yolo = None

        log_event("vision", "object detector initialised", {
            "feature_model": os.path.basename(self._feature_model_path),
            "classifier": os.path.basename(self._classifier_path),
            "labels_count": len(self.labels),
        })

    # ------------------------------------------------------------------
    # model loading
    # ------------------------------------------------------------------

    def load_feature_extractor(self) -> bool:
        """load the mobilenetv2 tflite model for feature extraction.

        returns:
            true if the model was loaded successfully.
        """
        if tflite is None:
            logger.warning(
                "tflite runtime not available — feature extraction disabled")
            return False

        if not os.path.isfile(self._feature_model_path):
            logger.warning(
                "feature model not found: %s", self._feature_model_path,
            )
            return False

        try:
            self.interpreter = tflite.Interpreter(
                model_path=self._feature_model_path)
            self.interpreter.allocate_tensors()

            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

            self._input_index = input_details[0]["index"]
            self._output_index = output_details[0]["index"]

            # infer input size from model
            input_shape = input_details[0]["shape"]  # e.g. [1, 224, 224, 3]
            self.input_size = (int(input_shape[2]), int(input_shape[1]))

            # infer feature dimension from output
            output_shape = output_details[0]["shape"]
            self.feature_dim = int(np.prod(output_shape[1:]))

            logger.info(
                "feature extractor loaded — input %s, output dim %d",
                self.input_size, self.feature_dim,
            )
            return True

        except Exception as exc:
            logger.error("failed to load feature extractor: %s", exc)
            self.interpreter = None
            return False

    def load_classifier(self) -> bool:
        """load the scikit-learn svm classifier and label map.

        returns:
            true if the classifier was loaded successfully.
        """
        # load label map
        if os.path.isfile(self._label_map_path):
            try:
                with open(self._label_map_path, "r") as fh:
                    self.labels = json.load(fh)
                logger.info(
                    "loaded %d labels from %s", len(
                        self.labels), self._label_map_path)
            except Exception as exc:
                logger.warning("failed to load label map: %s", exc)
                self.labels = []

        # load svm
        if joblib is None:
            logger.warning("joblib not available — classifier disabled")
            return False

        if not os.path.isfile(self._classifier_path):
            logger.info("classifier not found (expected on fresh install, falling back to YOLO): %s", self._classifier_path)
            return False

        try:
            self.classifier = joblib.load(self._classifier_path)
            logger.info("classifier loaded from %s", self._classifier_path)

            # if no explicit label map, try to get labels from the classifier
            if not self.labels and hasattr(self.classifier, "classes_"):
                self.labels = list(self.classifier.classes_)

            return True

        except Exception as exc:
            logger.error("failed to load classifier: %s", exc)
            self.classifier = None
            return False

    # ------------------------------------------------------------------
    # feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """extract a 1280-dim feature vector from a frame using mobilenetv2.

        args:
            frame: rgb numpy array of any size (will be resized).

        returns:
            1-d numpy array of shape (feature_dim,).  returns zeros
            if the model is not loaded.
        """
        if self.interpreter is None or frame is None:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # resize to model input size
        resized = self._resize_frame(frame, self.input_size)
        if len(resized.shape) == 2:
            resized = np.stack((resized,) * 3, axis=-1)
        elif resized.shape[2] == 4:
            resized = resized[..., :3]

        # prepare input tensor: (1, h, w, 3) float32 normalised to [0, 1]
        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        # check expected dtype from model
        input_details = self.interpreter.get_input_details()
        expected_dtype = input_details[0]["dtype"]
        if expected_dtype == np.uint8:
            input_data = np.expand_dims(resized.astype(np.uint8), axis=0)
        elif expected_dtype == np.int8:
            input_data = np.expand_dims((resized.astype(np.int32) - 128).astype(np.int8), axis=0)

        try:
            with self._lock:
                self.interpreter.set_tensor(self._input_index, input_data)
                self.interpreter.invoke()
                output = self.interpreter.get_tensor(self._output_index)
            features = output.flatten().astype(np.float32)

            return features
        except Exception as exc:
            logger.error("feature extraction error: %s", exc)
            return np.zeros(self.feature_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # classification
    # ------------------------------------------------------------------

    def classify(self, frame: np.ndarray) -> Tuple[str, float]:
        """classify a frame: extract features then run svm prediction.

        args:
            frame: rgb numpy array.

        returns:
            (label, confidence) tuple.  returns ('unknown', 0.0) if
            models are not loaded.
        """
        if self.classifier is None:
            return ("unknown", 0.0)

        features = self.extract_features(frame)
        if np.all(features == 0):
            return ("unknown", 0.0)

        try:
            features_2d = features.reshape(1, -1)

            label = self.classifier.predict(features_2d)[0]

            # get probability if the classifier supports it
            confidence = 0.0
            if hasattr(self.classifier, "predict_proba"):
                probas = self.classifier.predict_proba(features_2d)[0]
                confidence = float(np.max(probas))
            elif hasattr(self.classifier, "decision_function"):
                decision = self.classifier.decision_function(features_2d)
                # map decision function to pseudo-probability via sigmoid
                confidence = float(
                    1.0 / (1.0 + np.exp(-np.max(np.abs(decision)))))

            return (str(label), confidence)

        except Exception as exc:
            logger.error("classification error: %s", exc)
            return ("unknown", 0.0)

    # ------------------------------------------------------------------
    # sliding window detection
    # ------------------------------------------------------------------

    def detect_objects(
        self,
        frame: np.ndarray,
    ) -> List[Dict]:
        """detect objects using a sliding window approach.

        tiles the frame at multiple scales, extracts features from each
        tile, and classifies them.  returns tiles that exceed the
        confidence threshold.

        args:
            frame: rgb numpy array of shape (h, w, 3).

        returns:
            list of dicts, each with keys:
                'label'      — predicted class string.
                'confidence' — float 0–1.
                'bbox'       — [x, y, w, h] in pixel coordinates.
        """
        if frame is None or not hasattr(frame, 'shape') or len(frame.shape) < 2:
            return []

        h, w = frame.shape[:2]
        detections: List[Dict] = []
        
        if hasattr(self, 'yolo') and self.yolo and self.yolo.model is not None:
            yolo_dets = self.yolo.detect_objects(frame)
            if yolo_dets is not None:
                detections.extend(yolo_dets)

        if self.classifier is not None and self.interpreter is not None:
            label, conf = self.classify(frame)
            if label != "unknown" and conf >= self.confidence_threshold:
                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": [0, 0, w, h],
                })

        # simple non-maximum suppression: keep highest confidence per label
        detections = self._nms(detections)

        # apply bounding box smoothing (ema tracking)
        smoothed_detections = []
        current_labels = set()
        for det in detections:
            label = det["label"]
            bbox = det["bbox"]
            current_labels.add(label)
            if label in self._trackers:
                prev_bbox = self._trackers[label]
                new_bbox = [
                    int(prev * (1 - self._smoothing_factor) + curr * self._smoothing_factor)
                    for prev, curr in zip(prev_bbox, bbox)
                ]
                self._trackers[label] = new_bbox
                det["bbox"] = new_bbox
            else:
                self._trackers[label] = [float(v) for v in bbox]
            smoothed_detections.append(det)

        # clear missing labels
        self._trackers = {
            k: v for k,
            v in self._trackers.items() if k in current_labels}

        if smoothed_detections:
            logger.debug("detected %d objects", len(smoothed_detections))
            log_event("vision", "objects detected", {
                "count": len(smoothed_detections),
                "labels": [d["label"] for d in smoothed_detections],
            })

        # free intermediate arrays from the full ML pipeline
        # gc.collect() removed for performance

        return smoothed_detections

    # ------------------------------------------------------------------
    # targeted search
    # ------------------------------------------------------------------

    def is_object_in_frame(
        self,
        frame: np.ndarray,
        target_label: str,
    ) -> Tuple[bool, float, List[int]]:
        """check whether *target_label* is present in the frame.

        args:
            frame:        rgb numpy array.
            target_label: label string to search for.

        returns:
            (found, confidence, bbox) — bbox is [x, y, w, h] or empty list.
        """
        detections = self.detect_objects(frame)
        if detections is None:
            detections = []
        for det in detections:
            if det["label"].lower() == target_label.lower():
                return (True, det["confidence"], det["bbox"])
        return (False, 0.0, [])

    # ------------------------------------------------------------------
    # label access
    # ------------------------------------------------------------------

    def get_known_labels(self) -> List[str]:
        """return the list of known class labels.

        returns:
            list of label strings, or empty list if no labels loaded.
        """
        return list(self.labels)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_frame(
        frame: np.ndarray,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """resize a frame to (width, height) using pillow or numpy fallback.

        args:
            frame: rgb numpy array.
            size:  (width, height) target size.

        returns:
            resized numpy array.
        """
        if Image is not None:
            try:
                img = Image.fromarray(frame)
                resample = getattr(Image, "Resampling", Image).BILINEAR
                img_resized = img.resize(size, resample)
                result = np.array(img_resized)
                img.close()
                img_resized.close()
                return result
            except Exception as e:
                logger.warning(f"Pillow resize failed: {e}")

        # numpy nearest-neighbour fallback
        h, w = frame.shape[:2]
        target_w, target_h = size
        row_idx = (np.arange(target_h) * h // target_h).astype(int)
        col_idx = (np.arange(target_w) * w // target_w).astype(int)
        return frame[np.ix_(row_idx, col_idx)]

    @staticmethod
    def _nms(
        detections: List[Dict],
        iou_threshold: float = 0.5,
    ) -> List[Dict]:
        """simple non-maximum suppression.

        for each label, keeps the detection with the highest confidence
        and removes overlapping boxes exceeding *iou_threshold*.

        args:
            detections:    list of detection dicts.
            iou_threshold: iou above which a detection is suppressed.

        returns:
            filtered list of detection dicts.
        """
        if not detections:
            return []

        # sort by confidence descending
        detections = sorted(
            detections,
            key=lambda d: d["confidence"],
            reverse=True)
        keep: List[Dict] = []

        for det in detections:
            overlap = False
            for kept in keep:
                if det["label"] != kept["label"]:
                    continue
                iou = ObjectDetector._compute_iou(det["bbox"], kept["bbox"])
                if iou > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(det)

        return keep

    @staticmethod
    def _compute_iou(box_a: List[int], box_b: List[int]) -> float:
        """compute intersection over union of two [x, y, w, h] boxes.

        args:
            box_a: [x, y, w, h].
            box_b: [x, y, w, h].

        returns:
            iou value (0.0–1.0).
        """
        xa1, ya1 = box_a[0], box_a[1]
        xa2, ya2 = xa1 + box_a[2], ya1 + box_a[3]

        xb1, yb1 = box_b[0], box_b[1]
        xb2, yb2 = xb1 + box_b[2], yb1 + box_b[3]

        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = box_a[2] * box_a[3]
        area_b = box_b[2] * box_b[3]
        union_area = area_a + area_b - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area
