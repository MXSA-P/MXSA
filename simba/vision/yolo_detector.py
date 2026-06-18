# _max_cyan_ — project_mxsa
from simba.utils.logger import get_logger
import threading

_yolo_lock = threading.Lock()

logger = get_logger("simba.vision.yolo")

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    YOLO = None
    torch = None


class YoloDetector:
    def __init__(self):
        self.model = None
        if YOLO is not None:
            with _yolo_lock:
                try:
                    self.model = YOLO('yolov8n.pt')
                    logger.info("[ok] YOLOv8n loaded")
                except Exception as e:
                    logger.error(f"YOLO load failed: {e}")

    def detect_objects(self, frame_np):
        if self.model is None or frame_np is None or not hasattr(frame_np, 'shape') or len(frame_np.shape) < 2:
            return []

        with _yolo_lock:
            try:
                with torch.no_grad():
                    results = self.model(frame_np, verbose=False)
                detections = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        conf = float(box.conf[0].cpu().item() if hasattr(box.conf[0], 'cpu') else box.conf[0])
                        if conf > 0.4:
                            cls_idx = int(box.cls[0].cpu().item() if hasattr(box.cls[0], 'cpu') else box.cls[0])
                            label = self.model.names[cls_idx]
                            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0].tolist()
                            detections.append({
                                "label": label,
                                "confidence": conf,
                                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                            })
                del results
                return detections
            except Exception as e:
                logger.error(f"YOLO inference error: {e}")
                return []

    def is_object_in_frame(self, frame_np, target_label, threshold=0.5):
        """check if a specific object is in the frame.

        returns:
            tuple: (found_bool, confidence, bbox)
        """
        detections = self.detect_objects(frame_np)
        for det in detections:
            if det["label"] == target_label and det["confidence"] >= threshold:
                return True, det["confidence"], det["bbox"]
        return False, 0.0, []

    def classify(self, frame_np):
        """classify the dominant object in the frame (for trainer compatibility)."""
        detections = self.detect_objects(frame_np)
        if not detections:
            return "unknown", 0.0

        # return the one with highest confidence
        best_det = max(detections, key=lambda x: x["confidence"])
        return best_det["label"], best_det["confidence"]
